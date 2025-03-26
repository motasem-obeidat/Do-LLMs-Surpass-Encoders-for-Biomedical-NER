import pandas as pd


def convert_txt_to_csv(input_txt_path, output_csv_path, parts_length, tokenID, tagID):
    """
    Converts a text file with token and label information into a CSV file.
    The input text file is expected to have lines with tokens and labels separated by spaces.
    Sentences are separated by empty lines. Lines starting with "-DOCSTART- O" are ignored.

    Args:
        input_txt_path (str): The path to the input text file.
        output_csv_path (str): The path to the output CSV file.
        parts_length (int): The expected number of parts in each line of the input file.
        tokenID (int): The index of the token in the parts of each line.
        tagID (int): The index of the label in the parts of each line.

    Returns:
        pd.DataFrame: A DataFrame containing the sentences and their corresponding labels.
    """

    with open(input_txt_path, "r") as file:
        lines = file.readlines()

    sentences = []
    current_sentence = []
    current_labels = []

    for line in lines:
        if line.strip() == "-DOCSTART- O":
            continue
        elif line.strip() == "":
            if current_sentence:
                sentences.append((current_sentence, current_labels))
                current_sentence = []
                current_labels = []
        else:
            parts = line.strip().split()
            if len(parts) == parts_length:
                current_sentence.append(parts[tokenID])
                current_labels.append(parts[tagID])

    if current_sentence:
        sentences.append((current_sentence, current_labels))

    data_tuples = [
        (" ".join(sentence), " ".join(labels)) for sentence, labels in sentences
    ]
    df = pd.DataFrame(data_tuples, columns=["text", "labels"])

    # df.to_csv(output_csv_path, index=False)
    return df


def prepare_datasets(
    dataset_name, train_file, dev_file, test_file, train_csv, dev_csv, test_csv
):
    """
    Prepares datasets for training, validation, and testing by converting text files to CSV format and generating label mappings.

    Args:
        dataset_name (str): The name of the dataset. Supported values are "JNLPBA", "BioRED", "BC5CDR", "ChemProt", and "Reddit_Impacts".
        train_file (str): Path to the training data text file.
        dev_file (str): Path to the validation data text file.
        test_file (str): Path to the test data text file.
        train_csv (str): Path to save the converted training data CSV file.
        dev_csv (str): Path to save the converted validation data CSV file.
        test_csv (str): Path to save the converted test data CSV file.

    Returns:
        tuple: A tuple containing:
            - df_train (pd.DataFrame): DataFrame containing the training data.
            - df_val (pd.DataFrame): DataFrame containing the validation data.
            - df_test (pd.DataFrame): DataFrame containing the test data.
            - unique_labels (list): Sorted list of unique labels found in the training data.
            - labels_to_ids (dict): Dictionary mapping labels to their corresponding IDs.
            - ids_to_labels (dict): Dictionary mapping IDs to their corresponding labels.

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
        or dataset_name == "BC5CDR"
        or dataset_name == "ChemProt"
        or dataset_name == "Reddit_Impacts"
    ):
        parts_length = 2
        tokenID = 0
        tagID = 1

    df_train = convert_txt_to_csv(train_file, train_csv, parts_length, tokenID, tagID)
    df_val = convert_txt_to_csv(dev_file, dev_csv, parts_length, tokenID, tagID)
    df_test = convert_txt_to_csv(test_file, test_csv, parts_length, tokenID, tagID)

    labels = [i.split() for i in df_train["labels"].values.tolist()]
    unique_labels = set()

    for lb in labels:
        [unique_labels.add(i) for i in lb if i not in unique_labels]

    unique_labels = sorted(unique_labels)

    labels_to_ids = {k: v for v, k in enumerate(unique_labels)}
    ids_to_labels = {v: k for v, k in enumerate(unique_labels)}

    return df_train, df_val, df_test, unique_labels, labels_to_ids, ids_to_labels
