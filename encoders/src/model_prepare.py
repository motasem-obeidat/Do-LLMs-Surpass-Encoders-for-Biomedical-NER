import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import Counter


def get_tokenizer(model_name):
    """
    Initializes and returns a tokenizer for the specified model.
    If the model is BiomedBERT, it ensures that the double-quote character (`"`) is included in the tokenizer's vocabulary.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_name == "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract":
        if (
            '"' not in tokenizer.get_vocab()
        ):  # Ensure " is not already in the vocabulary
            tokenizer.add_tokens(['"'])  # Add " as a special token
            print(f'Added `"` to the tokenizer for model: {model_name}')

    return tokenizer


class model_func(torch.nn.Module):
    """
    model_func is a custom PyTorch neural network module designed for token classification tasks.
    It leverages a pre-trained model from the Hugging Face Transformers library and allows for customization of the tokenizer and number of labels.

    tokenizer (PreTrainedTokenizer): The tokenizer associated with the pre-trained model.

    __init__(model_name: str, tokenizer: PreTrainedTokenizer, num_labels: int):
            Initializes the model_func class with a specified pre-trained model, tokenizer, and number of labels for the classification task.
            If the model is BiomedBERT, the token embeddings are resized to match the tokenizer's vocabulary size.

    forward(input_id: torch.Tensor, mask: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            Performs a forward pass through the model. Computes the loss and logits for the token classification task.
                Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                    - loss (torch.Tensor): The computed loss if labels are provided.
                    - logits (torch.Tensor): The raw, unnormalized predictions from the model.

    get_tokenizer() -> PreTrainedTokenizer:
            Returns the tokenizer associated with the model.
    """

    def __init__(self, model_name, tokenizer, num_labels):
        super(model_func, self).__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

        if model_name == "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract":
            self.model.resize_token_embeddings(len(tokenizer))
            print(f"Model embeddings resized to {len(tokenizer)} tokens")

        self.tokenizer = tokenizer

    def forward(self, input_id, mask, label):
        output = self.model(
            input_ids=input_id, attention_mask=mask, labels=label, return_dict=False
        )
        return output[0], output[1]

    def get_tokenizer(self):
        return self.tokenizer


def align_label(texts, labels, tokenizer, labels_to_ids, label_all_tokens):
    """
    Aligns labels with tokenized inputs for sequence labeling tasks.

    Args:
        texts (list): List of input texts to be tokenized.
        labels (list): List of labels corresponding to each token in the input texts.
        tokenizer (PreTrainedTokenizer): Tokenizer to be used for tokenizing the input texts.
        labels_to_ids (dict): Dictionary mapping label names to their corresponding IDs.
        label_all_tokens (bool): If True, assigns a label to each token, otherwise assigns -100 to subword tokens.

    Returns:
        Tuple[list, list]:
            - List of label IDs aligned with the tokenized inputs.
            - List of word IDs corresponding to each token in the tokenized inputs.
    """

    tokenized_inputs = tokenizer(
        texts,
        padding="max_length",
        max_length=512,
        truncation=True,
        is_split_into_words=True,
    )

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)

        else:
            try:
                label_ids.append(
                    labels_to_ids[labels[word_idx]] if label_all_tokens else -100
                )
            except:
                label_ids.append(-100)

        previous_word_idx = word_idx

    return label_ids, word_ids


class data_sequence(torch.utils.data.Dataset):
    """
    A custom Dataset class for handling tokenized text data and corresponding labels.

    Args:
        df (pd.DataFrame): DataFrame containing the text and labels.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to convert text into tokens.
        labels_to_ids (dict): Dictionary mapping label names to label IDs.

    Attributes:
        labels_list (list): List of lists containing labels for each text.
        tokens_list (list): List of lists containing tokens for each text.
        texts (list): List of tokenized texts.
        labels (list): List of aligned label IDs.
        word_ids_list (list): List of word IDs corresponding to the tokenized texts.

    Methods:
        get_clean_labels(): Returns the labels with padding tokens (-100) removed.

        get_labels(): Returns the original labels list.

        get_all_tokenized_texts_filtered(tokenizer): Returns all tokenized texts with special tokens filtered out.

        get_word_ids(): Returns the word IDs with None values removed.

        __len__(): Returns the number of samples in the dataset.

        get_batch_data(idx): Returns the tokenized text data for a given index.

        get_batch_labels(idx): Returns the label IDs for a given index.

        __getitem__(idx): Returns the tokenized text data and label IDs for a given index.
    """

    def __init__(self, df, tokenizer, labels_to_ids):
        self.labels_list = [label.split() for label in df["labels"].values.tolist()]

        self.tokens_list = [text.split() for text in df["text"].values.tolist()]

        self.texts = [
            tokenizer(
                tokens,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
                is_split_into_words=True,
            )
            for tokens in self.tokens_list
        ]

        self.labels = []
        self.word_ids_list = []

        for tokens, labels in zip(self.tokens_list, self.labels_list):
            label_ids, word_ids = align_label(
                tokens, labels, tokenizer, labels_to_ids, True
            )

            self.labels.append(label_ids)
            self.word_ids_list.append(word_ids)

    def get_clean_labels(self):
        return [
            [label for label in sublist if label != -100] for sublist in self.labels
        ]

    def get_labels(self):
        return self.labels_list

    def get_all_tokenized_texts_filtered(self, tokenizer):
        all_tokenized_texts = []
        for tokenized in self.texts:
            filtered_ids = [
                token_id
                for token_id in tokenized["input_ids"][0]
                if token_id not in tokenizer.all_special_ids
            ]

            filtered_text = tokenizer.convert_ids_to_tokens(filtered_ids)
            all_tokenized_texts.append(filtered_text)

        return all_tokenized_texts

    def get_word_ids(self):
        return [[x for x in sublist if x is not None] for sublist in self.word_ids_list]

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_data, batch_labels


def untokenize(tokens, labels, word_ids):
    """
    Reconstructs original tokens from tokenized subwords and consolidates their labels.

    Args:
        tokens (list): A list of token lists, where each token list represents a sequence of subword tokens.
        labels (list): A list of label lists, where each label list corresponds to the labels of the subword tokens.
        word_ids (list): A list of word ID lists, where each word ID list indicates the word ID for each subword token.

    Returns:
        tuple: A tuple containing:
            - all_consolidated_tokens (list): A list of untokenized tokens lists, where each tokens list represents a sequence of reconstructed tokens.
            - all_consolidated_labels (list): A list of consolidated label lists, where each label list represents the most frequent label for each reconstructed token.

    Example:
        tokens = [["I", "##'m", "go", "##ing"], ["to", "the", "store"]]
        labels = [["O", "O", "B-LOC", "I-LOC"], ["O", "O", "O"]]
        word_ids = [[0, 0, 1, 1], [2, 3, 4]]

        # consolidated_tokens: [["I'm", "going"], ["to", "the", "store"]]
        # consolidated_labels: [["O", "B-LOC"], ["O", "O", "O"]]
    """

    all_consolidated_tokens = []
    all_consolidated_labels = []

    for token_list, label_list, word_id_list in zip(tokens, labels, word_ids):
        consolidated_tokens = []
        consolidated_labels = []
        current_token = ""
        current_label = []
        previous_word_id = None

        for token, label, word_id in zip(token_list, label_list, word_id_list):
            if word_id != previous_word_id:
                if current_token:
                    consolidated_tokens.append(current_token)
                    label_counts = Counter(current_label)
                    most_frequent_label, _ = label_counts.most_common(1)[0]
                    consolidated_labels.append(most_frequent_label)

                if token.startswith("##"):
                    current_token = token[2:]

                else:
                    current_token = token
                current_label = [label]

            else:
                if token.startswith("##"):
                    current_token += token[2:]

                else:
                    current_token += token
                current_label.append(label)

            previous_word_id = word_id

        if current_token:
            consolidated_tokens.append(current_token)
            label_counts = Counter(current_label)
            most_frequent_label, _ = label_counts.most_common(1)[0]
            consolidated_labels.append(most_frequent_label)

        all_consolidated_tokens.append(consolidated_tokens)
        all_consolidated_labels.append(consolidated_labels)

    return all_consolidated_tokens, all_consolidated_labels
