import os

"""
This script executes a shell command to run a Python script with specified arguments.

The command runs the script located at /src/inference.py with the following parameters:
    - --model_dir: Specifies the directory where the saved model is located.
    - --datasets_path: Specifies the path to the datasets.
    - --model_name: Specifies the name of the model to be used.
    - --dataset_name: Specifies the name of the dataset to be used. Supported datasets are "JNLPBA", "BioRED", "BC5CDR", "ChemProt", and "Reddit_Impacts".

The os.system function is used to execute the shell command.
"""

os.system(
    "python /src/inference.py \
    --model_dir /results/ \
    --datasets_path /datasets/ \
    --model_name google-bert/bert-large-uncased \
    --dataset_name JNLPBA"
)
