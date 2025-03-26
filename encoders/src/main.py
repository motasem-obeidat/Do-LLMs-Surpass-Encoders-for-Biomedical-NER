import torch
import numpy as np
import time
import argparse
import os
import random
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from model_prepare import model_func, data_sequence, get_tokenizer, untokenize
from metrics import matrices_compute
from data_preprocessing import prepare_datasets


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


def shuffle_data(dataset, tokenized_data, word_ids, train_org_labels):
    """
    Shuffle the train dataset and associated lists in unison.

    Parameters:
        dataset (Dataset): The train dataset to be shuffled.
        tokenized_data (list): A list of tokenized texts corresponding to the dataset.
        word_ids (list): A list of word IDs corresponding to the dataset.
        train_org_labels (list): A list of original labels corresponding to the dataset.

    Returns:
        tuple: A tuple containing the shuffled dataset and the shuffled lists:
            - shuffled_dataset (Subset): The shuffled dataset.
            - shuffled_tok_txts (list): The shuffled list of tokenized texts.
            - shuffled_word_ids (list): The shuffled list of word IDs.
            - shuffled_train_org_labels (list): The shuffled list of original labels.
    """

    indices = np.random.permutation(len(dataset))
    shuffled_dataset = Subset(dataset, indices)

    shuffled_tok_txts = [tokenized_data[i] for i in indices]
    shuffled_word_ids = [word_ids[i] for i in indices]
    shuffled_train_org_labels = [train_org_labels[i] for i in indices]

    return (
        shuffled_dataset,
        shuffled_tok_txts,
        shuffled_word_ids,
        shuffled_train_org_labels,
    )


def training_loop(
    model,
    tokenizer,
    df_train,
    df_val,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    labels_to_ids,
    ids_to_labels,
    output_dir,
):
    """
    Trains and validates a given model using the provided training and validation datasets.

    Args:
        model (torch.nn.Module): The model to be trained.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for processing the text data.
        df_train (pd.DataFrame): The training dataset.
        df_val (pd.DataFrame): The validation dataset.
        BATCH_SIZE (int): The batch size for training and validation.
        LEARNING_RATE (float): The learning rate for the optimizer.
        EPOCHS (int): The number of epochs to train the model.
        labels_to_ids (dict): A dictionary mapping label names to label IDs.
        ids_to_labels (dict): A dictionary mapping label IDs to label names.
        output_dir (str): The directory where the best model will be saved.

    The function performs the following steps:
        1. Prepares the training and validation datasets.
        2. Initializes the optimizer and learning rate scheduler.
        3. Trains the model for the specified number of epochs.
        4. Evaluate the model on the validation dataset after each epoch.
        5. Saves the model with the best validation strict F1 score.
        6. Implements early stopping if the validation strict F1 score does not improve for 3 consecutive epochs.
    """

    train_dataset = data_sequence(df_train, tokenizer, labels_to_ids)
    train_tokenized_texts = train_dataset.get_all_tokenized_texts_filtered(tokenizer)
    train_word_ids = train_dataset.get_word_ids()
    train_org_labels = train_dataset.get_labels()

    val_dataset = data_sequence(df_val, tokenizer, labels_to_ids)
    val_tokenized_texts = val_dataset.get_all_tokenized_texts_filtered(tokenizer)
    val_word_ids = val_dataset.get_word_ids()
    val_org_labels = val_dataset.get_labels()
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)

    num_training_steps = len(train_dataset) // BATCH_SIZE * EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    if use_cuda:
        model = model.cuda()

    # -------------------------------- Training Phase -----------------------------------

    best_strict_F1 = 0
    no_improve_epochs_counter = 0
    best_model_state = None

    for epoch_num in range(EPOCHS):

        train_dataset, train_tokenized_texts, train_word_ids, train_org_labels = (
            shuffle_data(
                train_dataset, train_tokenized_texts, train_word_ids, train_org_labels
            )
        )
        train_dataloader = DataLoader(
            train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False
        )

        total_train_acc = 0
        total_train_loss = 0
        train_true_labels, train_preds_labels = [], []

        model.train()

        for train_data, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_data["attention_mask"].squeeze(1).to(device)
            input_id = train_data["input_ids"].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)

            for i in range(logits.shape[0]):
                logits_clean = logits[i][train_label[i] != -100]
                label_clean = train_label[i][train_label[i] != -100]

                predictions = logits_clean.argmax(dim=1)

                total_train_acc += (predictions == label_clean).float().mean()
                total_train_loss += loss.item()

                train_preds_labels.append(predictions.tolist())
                train_true_labels.append(label_clean.tolist())

            loss.backward()
            optimizer.step()
            scheduler.step()

        for param_group in optimizer.param_groups:
            print("\n\tCurrent Learning rate: ", param_group["lr"])

        train_true_labels = [
            [ids_to_labels[entity] for entity in sublist]
            for sublist in train_true_labels
        ]
        train_preds_labels = [
            [ids_to_labels[entity] for entity in sublist]
            for sublist in train_preds_labels
        ]

        train_true_consolidated_tokens, train_true_consolidated_labels = untokenize(
            train_tokenized_texts, train_true_labels, train_word_ids
        )
        train_preds_consolidated_tokens, train_preds_consolidated_labels = untokenize(
            train_tokenized_texts, train_preds_labels, train_word_ids
        )

        train_strict_F1, train_relaxed_F1, train_token_level = matrices_compute(
            train_true_consolidated_labels,
            train_preds_consolidated_labels,
            labels_to_ids,
        )

        # ----------------------------------- Validation Phase -----------------------------------

        model.eval()

        total_val_acc = 0
        total_val_loss = 0
        val_true_labels, val_preds_labels = [], []

        for val_data, val_label in val_dataloader:

            val_label = val_label.to(device)
            mask = val_data["attention_mask"].squeeze(1).to(device)
            input_id = val_data["input_ids"].squeeze(1).to(device)

            with torch.no_grad():
                loss, logits = model(input_id, mask, val_label)

            for i in range(logits.shape[0]):
                logits_clean = logits[i][val_label[i] != -100]
                label_clean = val_label[i][val_label[i] != -100]

                predictions = logits_clean.argmax(dim=1)

                total_val_acc += (predictions == label_clean).float().mean()
                total_val_loss += loss.item()

                val_preds_labels.append(predictions.tolist())
                val_true_labels.append(label_clean.tolist())

        val_true_labels = [
            [ids_to_labels[entity] for entity in sublist] for sublist in val_true_labels
        ]
        val_preds_labels = [
            [ids_to_labels[entity] for entity in sublist]
            for sublist in val_preds_labels
        ]

        val_true_consolidated_tokens, val_true_consolidated_labels = untokenize(
            val_tokenized_texts, val_true_labels, val_word_ids
        )
        val_preds_consolidated_tokens, val_preds_consolidated_labels = untokenize(
            val_tokenized_texts, val_preds_labels, val_word_ids
        )

        val_strict_F1, val_relaxed_F1, val_token_level = matrices_compute(
            val_true_consolidated_labels, val_preds_consolidated_labels, labels_to_ids
        )

        if val_strict_F1[2] > best_strict_F1:
            best_strict_F1 = val_strict_F1[2]
            best_model_state = model.state_dict()
            print(f"Model saved with best strict F1 score at Epoch {epoch_num+1}")
            no_improve_epochs_counter = 0
        else:
            no_improve_epochs_counter += 1

        if no_improve_epochs_counter >= 3:
            print("Stopping early due to no improvement")
            break

        print(
            f"Epoch: {epoch_num + 1}/{EPOCHS} | "
            f"Train Loss: {total_train_loss / len(df_train): .4f} | "
            f"Train Accuracy: {total_train_acc / len(df_train): .4f} | "
            f"Train Strict Precision: {train_strict_F1[0]:.4f} | "
            f"Train Strict Recall: {train_strict_F1[1]:.4f} | "
            f"Train Strict F1: {train_strict_F1[2]:.4f} | "
            f"Train Relaxed Precision: {train_relaxed_F1[0]:.4f} | "
            f"Train Relaxed Recall: {train_relaxed_F1[1]:.4f} | "
            f"Train Relaxed F1: {train_relaxed_F1[2]:.4f} | "
            f"Val Loss: {total_val_loss / len(df_val): .4f} | "
            f"Val Accuracy: {total_val_acc / len(df_val): .4f} | "
            f"Val Strict Precision: {val_strict_F1[0]:.4f} | "
            f"Val Strict Recall: {val_strict_F1[1]:.4f} | "
            f"Val Strict F1: {val_strict_F1[2]:.4f} | "
            f"Val Relaxed Precision: {val_relaxed_F1[0]:.4f} | "
            f"Val Relaxed Recall: {val_relaxed_F1[1]:.4f} | "
            f"Val Relaxed F1: {val_relaxed_F1[2]:.4f}"
        )

        print("Train Token Level Scores", train_token_level, "\n")
        print("Val Token Level Scores", val_token_level, "\n")

    if best_model_state:
        torch.save(best_model_state, output_dir + "model.pth")
        print("Model with the best strict F1 score saved.")


def model_train(args, df_train, df_val, unique_labels, labels_to_ids, ids_to_labels):
    """
    Initializes the tokenizer and trains a machine learning model using the provided training and validation data.

    Args:
        args (Namespace): A namespace object containing the following attributes:
            - model_name (str): The name of the pre-trained model to use.
            - batch_size (int): The batch size for training.
            - learning_rate (float): The learning rate for the optimizer.
            - num_train_epochs (int): The number of training epochs.
            - output_dir (str): The directory where the trained model will be saved.
        df_train (DataFrame): The training dataset.
        df_val (DataFrame): The validation dataset.
        unique_labels (list): A list of unique labels in the dataset.
        labels_to_ids (dict): A dictionary mapping labels to their corresponding IDs.
        ids_to_labels (dict): A dictionary mapping IDs to their corresponding labels.
    """

    tokenizer = get_tokenizer(args.model_name)
    model = model_func(args.model_name, tokenizer, num_labels=len(unique_labels))
    tokenizer = model.get_tokenizer()

    start_time = time.time()
    training_loop(
        model,
        tokenizer,
        df_train,
        df_val,
        args.batch_size,
        args.learning_rate,
        args.num_train_epochs,
        labels_to_ids,
        ids_to_labels,
        args.output_dir,
    )
    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training completed in {training_time} seconds.")


def main(args):
    """
    Main function that handles the data preparation and model training.

    Workflow:
        1. Sets a random seed for reproducibility.
        2. Prepares the datasets for training and validation.
        3. Trains the model using the training and validation datasets.
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
        f"len train: {len(df_train)}, len val: {len(df_val)}, "
        f"unique_labels: {unique_labels}, labels_to_ids: {labels_to_ids}, ids_to_labels: {ids_to_labels}"
    )

    model_train(args, df_train, df_val, unique_labels, labels_to_ids, ids_to_labels)
    print("Model training completed.")


if __name__ == "__main__":
    """
    Command line arguments containing the following attributes:
        --output_dir (str): Directory where the model will be saved.
        --dataset_name (str): Name of the dataset to be used.
        --datasets_path (str): Path to the directory containing the dataset files.
        --model_name (str): Name of the model to be used.
        --batch_size (int): Batch size for training and validation.
        --learning_rate (float): Learning rate for the optimizer.
        --num_train_epochs (int): Number of epochs for training.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--datasets_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--num_train_epochs", type=int, required=True)

    args = parser.parse_args()
    main(args)
