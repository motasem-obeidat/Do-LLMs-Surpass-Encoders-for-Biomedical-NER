import os

"""
This script executes a shell command to run a Python script with specified arguments.

The command runs the script located at /src/main.py with the following parameters:
    - --output_dir: Specifies the directory where the model will be saved.
    - --datasets_path: Specifies the path to the datasets.
    - --num_train_epochs: Specifies the number of training epochs.
    - --model_name: Specifies the name of the model to be used.
    - --learning_rate: Specifies the learning rate for the optimizer.
    - --batch_size: Specifies the batch size for training and validation.
    - --dataset_name: Specifies the name of the dataset to be used. Supported datasets are "JNLPBA", "BioRED", "BC5CDR", "ChemProt", and "Reddit_Impacts".

The os.system function is used to execute the shell command.
"""

os.system(
    "python /src/main.py \
    --output_dir /results/ \
    --datasets_path /datasets/ \
    --num_train_epochs 20 \
    --model_name google-bert/bert-large-uncased \
    --learning_rate 2e-5 \
    --batch_size 8 \
    --dataset_name JNLPBA"
)
