import os
import argparse
import torch
import time
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
from peft import PeftModel
from huggingface_hub import HfFolder
from accelerate import Accelerator
from data_preprocessing import create_dataset_dict, get_unique_labels
from metrics import process_predictions_and_labels
from model_prepare import test_formatting_func, generate_prompt_max_length_check


def set_random_seed(seed):
    """
    Set the random seed for various libraries to ensure reproducibility.

    This function sets the random seed for Python's `random` module, NumPy, and PyTorch.
    It also configures environment variables to ensure deterministic behavior in CuBLAS and other libraries.
    """

    print(f"Setting random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.environ["WANDB_DISABLED"] = "true"


def batch(iterable, n=4):
    """
    Splits an iterable into smaller batches of a specified size.

    Args:
        iterable (iterable): The iterable to be split into batches.
        n (int, optional): The size of each batch. Defaults to 4.

    Yields:
        iterable: Subsequent batches of the original iterable, each of size `n` or smaller if there are fewer than `n` items remaining.

    Example:
        >>> list(batch(range(10), 3))
        [range(0, 3), range(3, 6), range(6, 9), range(9, 10)]
    """

    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def main(args):
    """
    Main function to run the inference pipeline.

    Args:
        args (Namespace): Parsed command line arguments.

    The function performs the following steps:
        1. Initializes the Accelerator for distributed training.
        2. Sets the random seed for reproducibility.
        3. Saves the Hugging Face token.
        4. Creates the dataset dictionary for testing.
        5. Retrieves unique labels with and without BIO (Begin, Inside, Outside) tagging.
        6. Prepares test prompts and token:label for evaluation.
        7. Loads the appropriate model based on the specified model type and configuration.
        8. Tokenize the training and validation datasets to determines the maximum sequence length for padding.
        9. Prepares the model for evaluation, including handling multiple GPUs if available.
        10. Generates predictions for the test prompts.
        11. Process the predictions and calculate the scores.
    """

    accelerator = Accelerator()

    SEED = 42
    set_random_seed(SEED)

    HfFolder().save_token(args.hf_token)

    dataset = create_dataset_dict(
        args.dataset_name,
        args.datasets_path + args.dataset_name + "/train.txt",
        args.datasets_path + args.dataset_name + "/dev.txt",
        args.datasets_path + args.dataset_name + "/test.txt",
    )

    unique_labels_bio, unique_labels_no_bio = get_unique_labels()

    test_prompts = []
    decoded_true = []
    test_tokens = []

    for i in range(len(dataset["test"])):
        text = " ".join(dataset["test"]["tokens"][i])
        labels = dataset["test"]["tokens_labels"][i]

        test_prompts.append(test_formatting_func(text, unique_labels_no_bio))
        decoded_true.append(labels)
        test_tokens.append(dataset["test"]["tokens"][i])

    test_prompts = list(batch(test_prompts, n=args.generation_batch_size))

    base_model_id = args.model_name

    trained_model_checkpoint_path = (
        args.model_dir + "checkpoint-" + args.trained_model_checkpoint_number
    )

    if args.isQLoRA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        model = PeftModel.from_pretrained(base_model, trained_model_checkpoint_path)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            trained_model_checkpoint_path, device_map="auto", trust_remote_code=True
        )

    eval_tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, padding_side="left", add_bos_token=True, trust_remote_code=True
    )

    eval_tokenizer.pad_token = eval_tokenizer.eos_token
    EOS_TOKEN = eval_tokenizer.eos_token

    tokenized_train_dataset = []
    for i in range(len(dataset["train"])):
        text = " ".join(dataset["train"]["tokens"][i])
        labels = dataset["train"]["tokens_labels"][i]

        tokenized_train_dataset.append(
            generate_prompt_max_length_check(
                text,
                labels,
                unique_labels_no_bio,
                eval_tokenizer,
                EOS_TOKEN,
            )
        )

    tokenized_val_dataset = []
    for i in range(len(dataset["validation"])):
        text = " ".join(dataset["validation"]["tokens"][i])
        labels = dataset["validation"]["tokens_labels"][i]

        tokenized_val_dataset.append(
            generate_prompt_max_length_check(
                text,
                labels,
                unique_labels_no_bio,
                eval_tokenizer,
                EOS_TOKEN,
            )
        )

    lengths = [len(x["input_ids"]) for x in tokenized_train_dataset]
    lengths += [len(x["input_ids"]) for x in tokenized_val_dataset]

    max_length = max(lengths)
    MAX_MODEL_LENGTH_VAL = 2048
    max_length = min(max_length, MAX_MODEL_LENGTH_VAL)
    print("The max_length is:", max_length)

    model.config.use_cache = True
    model.eval()

    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")

    if gpu_count > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    model = accelerator.prepare_model(model)

    decoded_preds = []
    start_time = time.time()
    for prompt_batch in tqdm(test_prompts):
        model_input = eval_tokenizer(
            prompt_batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **model_input,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=eval_tokenizer.pad_token_id,
                eos_token_id=eval_tokenizer.eos_token_id,
            )

            answers = [
                eval_tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            decoded_preds.extend(answers)

    strict_F1_score = process_predictions_and_labels(
        test_tokens,
        decoded_true,
        decoded_preds,
        unique_labels_no_bio,
        unique_labels_bio,
        args,
        dataset["test"]["ner_tags"],
    )
    end_time = time.time()

    testing_time = end_time - start_time
    print(f"Testing completed in {testing_time} seconds.")
    print("Inference completed.")


def str2bool(v):
    """
    Convert a string representation of truth to a boolean.

    Args:
        v (str or bool): The value to convert. If the value is already a boolean, it is returned as is.
                         Otherwise, the string is checked against common representations of true and false.
    """

    if isinstance(v, bool):
        return v

    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True

    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False

    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    """
    Command line arguments containing the following attributes:
        --model_dir (str): The directory where the model checkpoint is located.
        --dataset_name (str): The name of the dataset to be used.
        --datasets_path (str): The path to the datasets.
        --trained_model_checkpoint_number (str, optional): The checkpoint number of the model to be used.
        --model_name (str): The name of the model to use.
        --isQLoRA (bool): Whether to use the QLoRA or not.
        --isTest (bool, default=True): Whether to run in test mode.
        --generation_batch_size (int, default=4): The batch size for test generation.
        --max_new_tokens (int): The maximum number of new tokens to generate.
        --hf_token (str): The Hugging Face token for authentication.
    """

    parser = argparse.ArgumentParser(description="Script for model inference")

    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--datasets_path", type=str, required=True)
    parser.add_argument("--trained_model_checkpoint_number", type=str, required=False)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--isQLoRA", type=str2bool, required=True)
    parser.add_argument("--isTest", type=str2bool, default=True)
    parser.add_argument("--generation_batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--hf_token", type=str, required=True)

    args = parser.parse_args()
    main(args)
