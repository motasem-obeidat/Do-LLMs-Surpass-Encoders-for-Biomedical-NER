import argparse
import os
import torch
import random
import pandas as pd
import numpy as np
import time
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from model_prepare import prepare_model
from data_preprocessing import create_dataset_dict, get_unique_labels
from metrics import process_predictions_and_labels
from huggingface_hub import HfFolder
from tqdm import tqdm
from accelerate import Accelerator

best_strict_F1 = -1.0


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


class SaveModelCallback(TrainerCallback):
    """
    A custom callback for saving the model and tokenizer during training.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer used for the model.

    Methods:
        on_save(args, state, control, **kwargs):
            Saves the model and tokenizer at the specified checkpoint during training.
            Removes the default PyTorch model file to save space.

            Args:
                args (TrainingArguments): The training arguments.
                state (TrainerState): The current state of the trainer.
                control (TrainerControl): The control object for the trainer.
                **kwargs: Additional keyword arguments, including the model to be saved.

            Returns:
                TrainerControl: The control object, potentially modified.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        self.tokenizer.save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")

        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

        return control


def main(args):
    """
    Main function to train a model for Named Entity Recognition (NER) using the Hugging Face Transformers library.

    Args:
        args (Namespace): Parsed command line arguments.

    The function performs the following steps:
        1. Initializes the Accelerator for distributed training.
        2. Sets a random seed for reproducibility.
        3. Saves the Hugging Face token.
        4. Creates a dataset dictionary for training and validation.
        5. Retrieves unique labels with and without BIO (Begin, Inside, Outside) tagging.
        6. Prepares the model, tokenizer, and tokenized datasets.
        7. Defines functions for preprocessing logits and computing evaluation metrics.
        8. Configures training arguments and initializes the Trainer.
        9. Trains the model for the specified number of epochs.
        10. Evaluate the model on the validation dataset after each epoch.
        11. Saves the best model based on the evaluation metric.
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

    (
        model,
        tokenizer,
        tokenized_train_dataset,
        tokenized_val_dataset,
        val_tokens,
        decoded_true,
        val_prompts,
        max_length,
    ) = prepare_model(dataset, unique_labels_no_bio, args, accelerator)

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]

        return logits.argmax(dim=-1)

    def compute_metrics(eval_preds):
        global best_strict_F1

        decoded_preds = []
        for prompt_batch in tqdm(val_prompts):
            model_input = tokenizer(
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
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                answers = [
                    tokenizer.decode(output, skip_special_tokens=True)
                    for output in outputs
                ]
                decoded_preds.extend(answers)

        strict_F1_score = process_predictions_and_labels(
            val_tokens,
            decoded_true,
            decoded_preds,
            unique_labels_no_bio,
            unique_labels_bio,
            args,
            dataset["validation"]["ner_tags"],
        )

        if strict_F1_score > best_strict_F1:
            best_strict_F1 = strict_F1_score

        return {"strict_F1_score": strict_F1_score}

    total_train_steps = (
        (len(tokenized_train_dataset) // args.per_device_train_batch_size)
        // args.gradient_accumulation_steps
        * args.num_train_epochs
    )
    warmup_steps = int(0.1 * total_train_steps)
    print("warmup_steps", warmup_steps)

    argss = TrainingArguments(
        seed=42,
        output_dir=args.output_dir,
        warmup_steps=warmup_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        optim=args.optim,
        lr_scheduler_type="linear",
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        do_eval=True,
        load_best_model_at_end=True,
        metric_for_best_model="strict_F1_score",
        greater_is_better=True,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=argss,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            SaveModelCallback(tokenizer),
        ],
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training completed in {training_time} seconds.")
    print("Model training completed.")


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
        --output_dir (str): The directory where the model and tokenizer will be saved.
        --dataset_name (str): The name of the dataset to use.
        --datasets_path (str): The path to the datasets.
        --model_name (str): The name of the model to use.
        --isQLoRA (bool): Whether to use the QLoRA or not.
        --isTest (bool): Whether to run in test mode or not.
        --per_device_train_batch_size (int): The batch size per device during training.
        --gradient_accumulation_steps (int): The number of gradient accumulation steps.
        --num_train_epochs (int): The number of training epochs.
        --learning_rate (float): The learning rate.
        --optim (str): The optimizer to use.
        --generation_batch_size (int): The batch size for validation generation.
        --max_new_tokens (int): The maximum number of new tokens to generate.
        --hf_token (str): The Hugging Face token for authentication.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--datasets_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--isQLoRA", type=str2bool, required=True)
    parser.add_argument("--isTest", type=str2bool, default=False)
    parser.add_argument("--per_device_train_batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--optim", type=str, required=True)
    parser.add_argument("--generation_batch_size", type=int, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--hf_token", type=str, required=True)

    args = parser.parse_args()
    main(args)
