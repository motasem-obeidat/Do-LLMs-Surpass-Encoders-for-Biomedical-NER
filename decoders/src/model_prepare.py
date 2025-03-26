import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

OUTPUT_SEPARATOR = {
    "output_separator": "Response:",
}


def formatting_func(text, labels, entities):
    """
    Formats a prompt for a task that involves identifying entities within a text and applying the BIO labeling scheme.

    Args:
        text (str): The input text in which entities need to be identified.
        labels (str): The BIO labeled response for the given text.
        entities (list): A list of entity labels to be used for categorizing each entity.

    Returns:
        str: A formatted string that includes the instruction, input text, and the labeled response.
    """

    entity_tags = ", ".join(entities)

    full_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Your task is to identify entities within the text and apply the BIO labeling scheme. Use the following labels to categorize each entity: {entity_tags}.

### Input:
{text}

### Response:
{labels}
"""
    return full_prompt


def test_formatting_func(text, entities):
    """
    Generates a formatted prompt for identifying entities within a given text using the BIO labeling scheme.

    Args:
        text (str): The input text in which entities need to be identified.
        entities (list): A list of entity labels to be used in the BIO labeling scheme.

    Returns:
        str: A formatted prompt string that includes the instruction, input text, and placeholders for the response.
    """

    entity_tags = ", ".join(entities)

    full_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Your task is to identify entities within the text and apply the BIO labeling scheme. Use the following labels to categorize each entity: {entity_tags}.

### Input:
{text}

### Response:
"""
    return full_prompt


def generate_prompt_max_length_check(text, labels, entities, tokenizer, EOS_TOKEN):
    """
    Generates a tokenized prompt to get the maximum length (max_length).

    This function formats the input text, labels, and entities, appends an end-of-sequence token (EOS_TOKEN), and then tokenizes the resulting string using the provided tokenizer.

    Args:
        text (str): The input text to be formatted and tokenized.
        labels (list): A list of labels associated with the text.
        entities (list): A list of entities.
        tokenizer (Callable): A tokenizer function or object that converts the formatted string into tokens.
        EOS_TOKEN (str): The end-of-sequence token to be appended to the formatted string.

    Returns:
        list: A list of token ids representing the tokenized prompt.
    """

    return tokenizer(formatting_func(text, labels, entities) + EOS_TOKEN)


def generate_and_tokenize_prompt(
    text, labels, entities, tokenizer, EOS_TOKEN, max_length
):
    """
    Generates a prompt by formatting the input text with labels and entities, then tokenizes the prompt using the provided tokenizer.

    Args:
        text (str): The input text to be formatted and tokenized.
        labels (list): A list of labels to be included in the formatted text.
        entities (list): A list of entities.
        tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenizing the formatted text.
        EOS_TOKEN (str): The end-of-sequence token to be appended to the formatted text.
        max_length (int): The maximum length of the tokenized sequence.

    Returns:
        dict: A dictionary containing the tokenized input with keys 'input_ids' and 'labels'.
              'input_ids' are the tokenized input IDs and 'labels' are a copy of 'input_ids'.
    """

    result = tokenizer(
        formatting_func(text, labels, entities) + EOS_TOKEN,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()

    return result


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


def prepare_model(dataset, unique_labels_no_bio, args, accelerator):
    """
    Prepares the model, tokenizer, and datasets.

    Args:
        dataset (dict): A dictionary containing the training and validation datasets.
        unique_labels_no_bio (list): A list of unique labels excluding the "BIO" format.
        args (Namespace): A namespace object containing various arguments and configurations.
        accelerator (Accelerator): An accelerator object for handling distributed training.

    Returns:
        tuple: A tuple containing the following elements:
            - model (AutoModelForCausalLM): The prepared model for training.
            - tokenizer (AutoTokenizer): The tokenizer associated with the model.
            - tokenized_train_dataset (list): A list of tokenized training data.
            - tokenized_val_dataset (list): A list of tokenized validation data.
            - val_tokens (list): A list of original validation tokens.
            - decoded_true (list): A list of token:label for the validation data.
            - val_prompts (list): A list of batched validation prompts.
            - max_length (int): The maximum length for padding the sequences.

    Notes:
        - The function supports different model configurations based on the model name provided in args.
        - It handles both QLoRA and non-QLoRA configurations.
        - The function calculates the maximum sequence length for padding based on the training and validation datasets.
        - It enables gradient checkpointing for the model to save memory during training.
        - If multiple GPUs are available, the model is set to be parallelizable.
    """

    base_model_id = args.model_name
    lora_r = 128
    lora_alpha = 256

    if args.isQLoRA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, device_map="auto", trust_remote_code=True
        )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, padding_side="left", add_bos_token=True, trust_remote_code=True
    )

    EOS_TOKEN = tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_train_dataset_test = []
    for i in range(len(dataset["train"])):
        text = " ".join(dataset["train"]["tokens"][i])
        labels = dataset["train"]["tokens_labels"][i]

        tokenized_train_dataset_test.append(
            generate_prompt_max_length_check(
                text,
                labels,
                unique_labels_no_bio,
                tokenizer,
                EOS_TOKEN,
            )
        )

    tokenized_val_dataset_test = []
    for i in range(len(dataset["validation"])):
        text = " ".join(dataset["validation"]["tokens"][i])
        labels = dataset["validation"]["tokens_labels"][i]

        tokenized_val_dataset_test.append(
            generate_prompt_max_length_check(
                text,
                labels,
                unique_labels_no_bio,
                tokenizer,
                EOS_TOKEN,
            )
        )

    lengths = [len(x["input_ids"]) for x in tokenized_train_dataset_test]
    lengths += [len(x["input_ids"]) for x in tokenized_val_dataset_test]

    max_length = max(lengths)
    MAX_MODEL_LENGTH_VAL = 2048
    max_length = min(max_length, MAX_MODEL_LENGTH_VAL)
    print("The max_length is:", max_length)

    tokenized_train_dataset = []
    for i in range(len(dataset["train"])):
        text = " ".join(dataset["train"]["tokens"][i])
        labels = dataset["train"]["tokens_labels"][i]

        tokenized_train_dataset.append(
            generate_and_tokenize_prompt(
                text,
                labels,
                unique_labels_no_bio,
                tokenizer,
                EOS_TOKEN,
                max_length,
            )
        )

    tokenized_val_dataset = []
    val_prompts = []
    decoded_true = []
    val_tokens = []
    for i in range(len(dataset["validation"])):
        text = " ".join(dataset["validation"]["tokens"][i])
        labels = dataset["validation"]["tokens_labels"][i]

        tokenized_val_dataset.append(
            generate_and_tokenize_prompt(
                text,
                labels,
                unique_labels_no_bio,
                tokenizer,
                EOS_TOKEN,
                max_length,
            )
        )

        val_prompts.append(test_formatting_func(text, unique_labels_no_bio))
        decoded_true.append(labels)
        val_tokens.append(dataset["validation"]["tokens"][i])

    val_prompts = list(batch(val_prompts, n=args.generation_batch_size))

    untokenized_text_sample = tokenizer.decode(tokenized_train_dataset[1]["input_ids"])
    print("untokenized_text_sample", untokenized_text_sample)

    model.gradient_checkpointing_enable()

    if args.isQLoRA:
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            bias="none",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    else:
        trainable_params = 0
        all_param = 0

        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    print("Model:", model)

    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")

    if gpu_count > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    model = accelerator.prepare_model(model)
    model.config.use_cache = False

    return (
        model,
        tokenizer,
        tokenized_train_dataset,
        tokenized_val_dataset,
        val_tokens,
        decoded_true,
        val_prompts,
        max_length,
    )
