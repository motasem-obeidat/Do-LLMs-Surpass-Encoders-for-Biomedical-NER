import os

"""
This script executes a command to run a Python script located at 'src/inference.py' with specified arguments.

Arguments:
    --model_dir (str): The directory where the model checkpoint is located.
    --datasets_path (str): The path to the directory containing the datasets.
    --generation_batch_size (int): The batch size to use during test generation.
    --max_new_tokens (int): The maximum number of new tokens to generate.
    --model_name (str): The name of the model to use for inference.
    --isQLoRA (bool): A flag indicating whether to use QLoRA (Quantized Low-Rank Adaptation).
    --dataset_name (str): The name of the dataset to use for inference. Supported datasets are "JNLPBA", "BioRED", "BC5CDR", "ChemProt", and "Reddit_Impacts".
    --trained_model_checkpoint_number (int): The checkpoint number of the trained model.
    --hf_token (str): The Hugging Face token for authentication.
"""

os.system(
    "python /src/inference.py \
    --model_dir /results/ \
    --datasets_path /datasets/ \
    --generation_batch_size 16 \
    --max_new_tokens 3000 \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --isQLoRA True \
    --dataset_name JNLPBA \
    --trained_model_checkpoint_number <CHECKPOINT_NUMBER> \
    --hf_token <YOUR_HF_TOKEN>"
)
