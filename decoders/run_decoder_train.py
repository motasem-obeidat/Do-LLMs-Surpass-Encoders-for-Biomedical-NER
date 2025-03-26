import os

"""
This script executes a command to run a Python script located at 'src/main.py' with specified arguments.

Arguments:
    --output_dir (str): The directory where the model will be saved.
    --datasets_path (str): The path to the datasets.
    --num_train_epochs (int): The number of training epochs.
    --learning_rate (float): The learning rate for the optimizer.
    --optim (str): The optimizer to use.
    --per_device_train_batch_size (int): The batch size per device during training.
    --gradient_accumulation_steps (int): The number of gradient accumulation steps.
    --generation_batch_size (int): The batch size for validation generation.
    --max_new_tokens (int): The maximum number of new tokens to generate.
    --model_name (str): The name of the model to use.
    --isQLoRA (bool): A flag indicating whether to use QLoRA (Quantized Low-Rank Adaptation).
    --dataset_name (str): The name of the dataset to use. Supported datasets are "JNLPBA", "BioRED", "BC5CDR", "ChemProt", and "Reddit_Impacts".
    --hf_token (str): The Hugging Face token for authentication.
"""

os.system(
    "python /src/main.py \
    --output_dir /results/ \
    --datasets_path /datasets/ \
    --num_train_epochs 20 \
    --learning_rate 4e-5 \
    --optim paged_adamw_8bit \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --generation_batch_size 16 \
    --max_new_tokens 3000 \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --isQLoRA True \
    --dataset_name JNLPBA \
    --hf_token <YOUR_HF_TOKEN>"
)
