import pandas as pd
from datasets import Dataset, DatasetDict

unique_labels_bio = set()
unique_labels_no_bio = set()


def data_preprocess(input_txt_path, parts_length, tokenID, tagID):
    """
    Preprocesses the input text file for named entity recognition (NER) tasks.
    This function reads a text file where each line contains a token and its corresponding tag.
    Sentences are separated by empty lines. Lines starting with "-DOCSTART- O" are ignored.
    It processes the file to extract sentences and their labels, and also tracks unique BIO labels and entity types.

    Args:
        input_txt_path (str): The path to the input text file.
        parts_length (int): The expected number of parts in each line of the input file.
        tokenID (int): The index of the token in the split line.
        tagID (int): The index of the tag in the split line.

    Returns:
        tuple: A tuple containing:
            - sentences (list): A list of sentences, where each sentence is a list of tokens.
            - labels (list): A list of labels corresponding to the tokens in the sentences.
            - token_label_strings (list): A list of strings where each string represents a sentence with tokens and their tags (token:label).
    """

    with open(input_txt_path, "r") as file:
        lines = file.readlines()

    sentences = []
    labels = []
    token_label_strings = []
    sentence = []
    label = []

    for line in lines:
        stripped_line = line.strip()

        if stripped_line == "-DOCSTART- O":
            continue

        if stripped_line == "":
            if sentence and label:
                sentences.append(sentence)
                labels.append(label)

                token_label_string = " ".join(
                    [f"{token}:{tag}" for token, tag in zip(sentence, label)]
                )

                token_label_string = token_label_string.replace(" ", "\n")
                token_label_strings.append(token_label_string)

                sentence = []
                label = []
        else:
            parts = stripped_line.split()

            if len(parts) == parts_length:
                token = parts[tokenID]
                tag = parts[tagID]
                sentence.append(token)
                label.append(tag)

                if tag != "O":
                    bio_tag, entity_type = tag.split("-", 1)
                    unique_labels_bio.add(tag)
                    unique_labels_no_bio.add(entity_type)

    if sentence and label:
        sentences.append(sentence)
        labels.append(label)

        token_label_string = " ".join(
            [f"{token}:{tag}" for token, tag in zip(sentence, label)]
        )

        token_label_string = token_label_string.replace(" ", "\n")
        token_label_strings.append(token_label_string)

    return sentences, labels, token_label_strings


def load_custom_dataset(input_txt_path, dataset_name):
    """
    Load a custom dataset and preprocess it for Named Entity Recognition (NER) tasks.

    Parameters:
    -----------
    input_txt_path : str
        The file path to the input text file containing the dataset.
    dataset_name : str
        The name of the dataset to be loaded. Supported values are "JNLPBA", "BioRED", "ChemProt", "BC5CDR", and "Reddit_Impacts".

    Returns:
    --------
    Dataset
        A Hugging Face Dataset object containing the preprocessed data with the following columns:
        - 'tokens': List of tokens in each sentence.
        - 'ner_tags': List of NER tags corresponding to each token.
        - 'tokens_labels': List of tokens labels (token:label).
        - 'id': Unique identifier for each sentence.

    Notes:
    ------
        The function preprocesses the input text file based on the specified dataset name.
        Different datasets have different formats, and the function handles these variations by setting appropriate values for `parts_length`, `tokenID`, and `tagID`.
    """

    if dataset_name == "JNLPBA":
        parts_length = 4
        tokenID = 0
        tagID = 3

    elif (
        dataset_name == "BioRED"
        or dataset_name == "ChemProt"
        or dataset_name == "BC5CDR"
        or dataset_name == "Reddit_Impacts"
    ):
        parts_length = 2
        tokenID = 0
        tagID = 1

    sentences, labels, token_label = data_preprocess(
        input_txt_path, parts_length, tokenID, tagID
    )

    data = {
        "tokens": sentences,
        "ner_tags": labels,
        "tokens_labels": token_label,
    }

    df = pd.DataFrame(data)
    df["id"] = df.index.astype(str)

    return Dataset.from_pandas(df)


def create_dataset_dict(dataset_name, train_txt_path, val_txt_path, test_txt_path):
    """
    Creates a dictionary of datasets for training, validation, and testing.

    Args:
        dataset_name (str): The name of the dataset to be loaded.
        train_txt_path (str): The file path to the training dataset text file.
        val_txt_path (str): The file path to the validation dataset text file.
        test_txt_path (str): The file path to the testing dataset text file.

    Returns:
        DatasetDict: A dictionary containing the training, validation, and testing datasets.
                     The keys are "train", "validation", and "test" respectively.
    """

    train_dataset = load_custom_dataset(train_txt_path, dataset_name)
    val_dataset = load_custom_dataset(val_txt_path, dataset_name)
    test_dataset = load_custom_dataset(test_txt_path, dataset_name)

    return DatasetDict(
        {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
    )


# This function returns two lists of unique labels: one with BIO labels and one with a non-BIO labels.
def get_unique_labels():
    return list(unique_labels_bio), list(unique_labels_no_bio)
