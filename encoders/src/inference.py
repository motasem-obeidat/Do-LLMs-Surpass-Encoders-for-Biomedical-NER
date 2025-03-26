import argparse
import os
import random
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from data_preprocessing import prepare_datasets
from model_prepare import model_func, data_sequence, get_tokenizer, untokenize
from metrics import matrices_compute, entity_size


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

    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def evaluate(model, tokenizer, df_test, labels_to_ids, ids_to_labels):
    """
     Evaluates the performance of a given model on a test dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for encoding the text data.
        df_test (pd.DataFrame): The test dataset containing the text and labels.
        labels_to_ids (dict): A dictionary mapping label names to their corresponding IDs.
        ids_to_labels (dict): A dictionary mapping label IDs to their corresponding names.

    Returns:
        tuple: A tuple containing:
            - test_true_consolidated_tokens (list): The untokenized tokens from the test dataset.
            - test_true_consolidated_labels (list): The untokenized true labels for the test dataset.
            - test_preds_consolidated_labels (list): The untokenized predicted labels for the test dataset.
    """

    test_dataset = data_sequence(df_test, tokenizer, labels_to_ids)
    test_tokenized_texts = test_dataset.get_all_tokenized_texts_filtered(tokenizer)
    test_word_ids = test_dataset.get_word_ids()
    test_org_labels = test_dataset.get_labels()
    test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_test_acc = 0.0
    test_preds_labels = []
    test_true_labels = []

    model.eval()

    for test_data, test_label in test_dataloader:
        test_label = test_label.to(device)
        mask = test_data["attention_mask"].squeeze(1).to(device)

        input_id = test_data["input_ids"].squeeze(1).to(device)

        loss, logits = model(input_id, mask, test_label)

        for i in range(logits.shape[0]):
            logits_clean = logits[i][test_label[i] != -100]
            label_clean = test_label[i][test_label[i] != -100]

            predictions = logits_clean.argmax(dim=1)

            acc = (predictions == label_clean).float().mean()
            total_test_acc += acc

            test_preds_labels.append(predictions.tolist())
            test_true_labels.append(label_clean.tolist())

    test_true_labels = [
        [ids_to_labels[entity] for entity in sublist] for sublist in test_true_labels
    ]
    test_preds_labels = [
        [ids_to_labels[entity] for entity in sublist] for sublist in test_preds_labels
    ]

    test_true_consolidated_tokens, test_true_consolidated_labels = untokenize(
        test_tokenized_texts, test_true_labels, test_word_ids
    )
    test_preds_consolidated_tokens, test_preds_consolidated_labels = untokenize(
        test_tokenized_texts, test_preds_labels, test_word_ids
    )

    test_strict_F1, test_relaxed_F1, test_token_level = matrices_compute(
        test_true_consolidated_labels, test_preds_consolidated_labels, labels_to_ids
    )

    print(
        f"Test Accuracy: {total_test_acc / len(df_test): .4f} | "
        f"Test Strict Precision: {test_strict_F1[0]:.4f} | "
        f"Test Strict Recall: {test_strict_F1[1]:.4f} | "
        f"Test Strict F1: {test_strict_F1[2]:.4f} | "
        f"Test Relaxed Precision: {test_relaxed_F1[0]:.4f} | "
        f"Test Relaxed Recall: {test_relaxed_F1[1]:.4f} | "
        f"Test Relaxed F1: {test_relaxed_F1[2]:.4f}"
    )

    print("Test Token Level Scores", test_token_level, "\n")

    return (
        test_true_consolidated_tokens,
        test_true_consolidated_labels,
        test_preds_consolidated_labels,
    )


def model_eval(args, df_test, unique_labels, labels_to_ids, ids_to_labels):
    """
    Evaluate the performance of a trained model on a test dataset.

    Args:
        args (Namespace): A namespace object containing the following attributes:
            - model_name (str): The name of the pre-trained model to use.
            - model_dir (str): The directory where the saved model is located.
        df_test (DataFrame): The test dataset in the form of a pandas DataFrame.
        unique_labels (list): A list of unique labels in the dataset.
        labels_to_ids (dict): A dictionary mapping label names to their corresponding IDs.
        ids_to_labels (dict): A dictionary mapping label IDs to their corresponding names.
    """

    tokenizer = get_tokenizer(args.model_name)
    model = model_func(args.model_name, tokenizer, num_labels=len(unique_labels))
    tokenizer = model.get_tokenizer()

    model.load_state_dict(torch.load(args.model_dir + "model.pth"))

    start_time = time.time()
    test_tokenized_texts, test_true_labels, test_preds_labels = evaluate(
        model, tokenizer, df_test, labels_to_ids, ids_to_labels
    )

    true_entites_counts, pred_entites_counts = entity_size(
        test_true_labels, test_preds_labels
    )

    test1_strict_F1, test1_relaxed_F1, test1_token_level = matrices_compute(
        true_entites_counts[0], pred_entites_counts[0], labels_to_ids
    )
    print(
        f"Entity Size = 1: Test Strict Precision: {test1_strict_F1[0]:.4f} | "
        f"Entity Size = 1: Test Strict Recall: {test1_strict_F1[1]:.4f} | "
        f"Entity Size = 1: Test Strict F1: {test1_strict_F1[2]:.4f} | "
        f"Entity Size = 1: Test Relaxed Precision: {test1_relaxed_F1[0]:.4f} | "
        f"Entity Size = 1: Test Relaxed Recall: {test1_relaxed_F1[1]:.4f} | "
        f"Entity Size = 1: Test Relaxed F1: {test1_relaxed_F1[2]:.4f}"
    )
    print("Entity Size = 1: Test Token Level Scores", test1_token_level, "\n")

    test2_strict_F1, test2_relaxed_F1, test2_token_level = matrices_compute(
        true_entites_counts[1], pred_entites_counts[1], labels_to_ids
    )
    print(
        f"Entity Size = 2: Test Strict Precision: {test2_strict_F1[0]:.4f} | "
        f"Entity Size = 2: Test Strict Recall: {test2_strict_F1[1]:.4f} | "
        f"Entity Size = 2: Test Strict F1: {test2_strict_F1[2]:.4f} | "
        f"Entity Size = 2: Test Relaxed Precision: {test2_relaxed_F1[0]:.4f} | "
        f"Entity Size = 2: Test Relaxed Recall: {test2_relaxed_F1[1]:.4f} | "
        f"Entity Size = 2: Test Relaxed F1: {test2_relaxed_F1[2]:.4f}"
    )
    print("Entity Size = 2: Test Token Level Scores", test2_token_level, "\n")

    test3_strict_F1, test3_relaxed_F1, test3_token_level = matrices_compute(
        true_entites_counts[2], pred_entites_counts[2], labels_to_ids
    )
    print(
        f"Entity Size >= 3: Test Strict Precision: {test3_strict_F1[0]:.4f} | "
        f"Entity Size >= 3: Test Strict Recall: {test3_strict_F1[1]:.4f} | "
        f"Entity Size >= 3: Test Strict F1: {test3_strict_F1[2]:.4f} | "
        f"Entity Size >= 3: Test Relaxed Precision: {test3_relaxed_F1[0]:.4f} | "
        f"Entity Size >= 3: Test Relaxed Recall: {test3_relaxed_F1[1]:.4f} | "
        f"Entity Size >= 3: Test Relaxed F1: {test3_relaxed_F1[2]:.4f}"
    )
    print(
        "Entity Size >= 3: Test Token Level Scores",
        test3_token_level,
        "\n",
    )
    end_time = time.time()

    testing_time = end_time - start_time
    print(f"Testing completed in {testing_time} seconds.")


def main(args):
    """
    Main function that handles the model evaluation on a test dataset.

    Workflow:
        1. Sets a random seed for reproducibility.
        2. Prepares the datasets for testing.
        3. Evaluates the model using the test dataset.
    """

    SEED = 42
    set_random_seed(SEED)

    df_train, df_val, df_test, unique_labels, labels_to_ids, ids_to_labels = (
        prepare_datasets(
            args.dataset_name,
            args.datasets_path + args.dataset_name + "/train.txt",
            args.datasets_path + args.dataset_name + "/dev.txt",
            args.datasets_path + args.dataset_name + "/test.txt",
            args.datasets_path + args.dataset_name + "/train.csv",
            args.datasets_path + args.dataset_name + "/dev.csv",
            args.datasets_path + args.dataset_name + "/test.csv",
        )
    )

    print(
        f"len test: {len(df_test)}, "
        f"unique_labels: {unique_labels}, labels_to_ids: {labels_to_ids}, ids_to_labels: {ids_to_labels}"
    )

    model_eval(args, df_test, unique_labels, labels_to_ids, ids_to_labels)
    print("Inference completed.")


if __name__ == "__main__":
    """
    Command line arguments containing the following attributes:
        --model_dir (str): Directory  where the saved model is located.
        --dataset_name (str): Name of the dataset to be used.
        --datasets_path (str): Path to the directory containing the dataset files.
        --model_name (str): Name of the model to be used.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--datasets_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)

    args = parser.parse_args()
    main(args)
