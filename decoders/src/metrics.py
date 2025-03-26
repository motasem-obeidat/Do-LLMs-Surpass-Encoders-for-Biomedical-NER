import re
import difflib
from sklearn_crfsuite.metrics import flat_classification_report
from model_prepare import OUTPUT_SEPARATOR
from ner_SemEval import Evaluator


# Replace Unicode characters with their \\uXXXX escape sequences.
def escape_unicode(text):
    return re.sub(
        r"[\u0080-\uFFFF]", lambda x: "\\u{:04x}".format(ord(x.group(0))), text
    )


def create_bio_labels(original_list, predictions_list):
    """
    Generate BIO (Beginning, Inside, Outside) labels for a list of original tokens based on predictions.

    Args:
        original_list (list): A list where each element is a list of tokens representing the original text.
        predictions_list (list): A list of prediction strings, where each string contains tokens and their corresponding tags (token:label).

    Returns:
        list: A list where each element is a list of BIO labels corresponding to the original tokens.

    The function processes each pair of original text and prediction string by:
        - Splitting the prediction string into tokens and tags.
        - Initializing BIO labels for the original tokens as "O" (Outside).
        - Using difflib.SequenceMatcher to find matching sequences between original tokens and predicted tokens.
        - Assigning the predicted tags to the corresponding positions in the BIO labels.
        - Counting the number of ignored tokens due to mismatches (replacements, deletions, insertions).
    """

    all_bio_labels = []
    mismatches_counter = 0

    for original, predictions in zip(original_list, predictions_list):
        predictions = predictions.split("\n")

        pred_tokens = [
            p.rsplit(":", 1)[0].strip()
            for p in predictions
            if len(p.rsplit(":", 1)) == 2
        ]

        pred_tags = [
            p.rsplit(":", 1)[1].strip()
            for p in predictions
            if len(p.rsplit(":", 1)) == 2
        ]

        bio_labels = ["O"] * len(original)

        matcher = difflib.SequenceMatcher(None, original, pred_tokens)
        matches = matcher.get_opcodes()

        for tag, i1, i2, j1, j2 in matches:
            if tag == "equal":
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    bio_labels[i] = pred_tags[j]

            elif tag == "replace" or tag == "delete" or tag == "insert":
                mismatches_counter += abs(i2 - i1)

        all_bio_labels.append(bio_labels)

    print("mismatches_counter:", mismatches_counter)

    return all_bio_labels


def extract_output(input_string):
    """
    Extracts the output portion from the given input string based on a predefined output separator.

    Args:
        input_string (str): The input string containing the output portion to be extracted.

    Returns:
        str: The extracted output portion of the input string.

    Notes:
        - If the input string ends with a colon (":"), a space and newline are appended to it.
        - The function searches for the output separator defined in the OUTPUT_SEPARATOR dictionary.
        - The output portion is extracted starting from the position immediately after the output separator.
    """

    if input_string.endswith(":"):
        input_string += " \n"

    answer_start_idx = input_string.find(OUTPUT_SEPARATOR["output_separator"])
    output_separator_length = len(OUTPUT_SEPARATOR["output_separator"])

    input_string = input_string[answer_start_idx + output_separator_length + 1 :]

    return input_string


# Calculates the F1 score given precision and recall values.
def calculate_F1(precision, recall):
    if (precision + recall) == 0:
        return 0

    return 2 * (precision * recall) / (precision + recall)


def matrices_compute(true_labels, pred_labels, unique_labels_no_bio, unique_labels_bio):
    """
    Compute various evaluation metrics for predicted labels against true labels.
    This function calculates strict, relaxed, and token-level F1 scores for the given true and predicted labels.
    It uses SemEval Evaluator to compute the metrics and returns the results in a structured format.

    Parameters:
    -----------
    true_labels : list
        A list of true labels for the dataset.
    pred_labels : list
        A list of predicted labels for the dataset.
    labels_to_ids : dict
        A dictionary mapping label names to their corresponding IDs.

    Returns:
    --------
    tuple
        A tuple containing three elements:
        - strict_F1_score_results: A list containing precision, recall, and F1 score for strict evaluation.
        - relaxed_F1_score_results: A list containing precision, recall, and F1 score for relaxed evaluation.
        - token_level_score_results: A string containing the token-level classification report.
    """

    strict_F1_score_results = []
    relaxed_F1_score_results = []
    token_level_score_results = []

    evaluator = Evaluator(true_labels, pred_labels, unique_labels_no_bio)
    results, results_agg = evaluator.evaluate()

    strict_metrics = results.get("strict", {})
    strict_precision = strict_metrics.get("precision", 0)
    strict_recall = strict_metrics.get("recall", 0)
    strict_F1_score = calculate_F1(strict_precision, strict_recall)
    strict_F1_score_results = [strict_precision, strict_recall, strict_F1_score]

    relaxed_metrics = results.get("partial", {})
    relaxed_precision = relaxed_metrics.get("precision", 0)
    relaxed_recall = relaxed_metrics.get("recall", 0)
    relaxed_F1_score = calculate_F1(relaxed_precision, relaxed_recall)
    relaxed_F1_score_results = [relaxed_precision, relaxed_recall, relaxed_F1_score]

    sorted_labels = sorted(unique_labels_bio, key=lambda name: (name[1:], name[0]))
    token_level_score_results = flat_classification_report(
        true_labels, pred_labels, labels=sorted_labels, digits=4
    )

    return (
        strict_F1_score_results,
        relaxed_F1_score_results,
        token_level_score_results,
    )


def entity_size(all_lbls, all_preds):
    """
    Categorizes entities based on their lengths from true and predicted labels.

    Args:
        all_lbls (list): A list of sentences, where each sentence is a list of true labels.
        all_preds (list): A list of sentences, where each sentence is a list of predicted labels.

    Returns:
        tuple: A tuple containing two lists:
            - true_entities_counts (list): A list containing three lists of true entities categorized by their lengths:
                - true_entity1: Entities of length 1.
                - true_entity2: Entities of length 2.
                - true_entity3_plus: Entities of length 3 or more.

            - pred_entities_counts (list): A list containing three lists of predicted entities categorized by their lengths:
                - pred_entity1: Entities of length 1.
                - pred_entity2: Entities of length 2.
                - pred_entity3_plus: Entities of length 3 or more.
    """

    true_entity1, true_entity2, true_entity3_plus = [], [], []
    pred_entity1, pred_entity2, pred_entity3_plus = [], [], []

    for sentence_true, sentence_pred in zip(all_lbls, all_preds):
        tmp_true, tmp_pred = [], []
        count = 0

        for true_label, pred_label in zip(sentence_true, sentence_pred):
            if true_label.startswith("B-"):
                if count > 0:
                    if count == 1:
                        true_entity1.append(tmp_true)
                        pred_entity1.append(tmp_pred)
                    elif count == 2:
                        true_entity2.append(tmp_true)
                        pred_entity2.append(tmp_pred)
                    else:
                        true_entity3_plus.append(tmp_true)
                        pred_entity3_plus.append(tmp_pred)

                tmp_true, tmp_pred = [true_label], [pred_label]
                count = 1

            elif true_label.startswith("I-"):
                tmp_true.append(true_label)
                tmp_pred.append(pred_label)
                count += 1

            elif true_label.startswith("O"):
                if count > 0:
                    if count == 1:
                        true_entity1.append(tmp_true)
                        pred_entity1.append(tmp_pred)
                    elif count == 2:
                        true_entity2.append(tmp_true)
                        pred_entity2.append(tmp_pred)
                    else:
                        true_entity3_plus.append(tmp_true)
                        pred_entity3_plus.append(tmp_pred)

                tmp_true, tmp_pred = [], []
                count = 0
                true_entity1.append([true_label])
                pred_entity1.append([pred_label])

        if tmp_true:
            if count == 1:
                true_entity1.append(tmp_true)
                pred_entity1.append(tmp_pred)

            elif count == 2:
                true_entity2.append(tmp_true)
                pred_entity2.append(tmp_pred)

            else:
                true_entity3_plus.append(tmp_true)
                pred_entity3_plus.append(tmp_pred)

    true_entities_counts = [true_entity1, true_entity2, true_entity3_plus]
    pred_entities_counts = [pred_entity1, pred_entity2, pred_entity3_plus]

    return true_entities_counts, pred_entities_counts


def process_predictions_and_labels(
    org_tokens,
    decoded_true,
    decoded_preds,
    unique_labels_no_bio,
    unique_labels_bio,
    args,
    targets_BIO,
):
    """
    Processes predictions and labels, computes evaluation metrics, and prints results.

    Args:
        org_tokens (list): Original tokens from the dataset.
        decoded_true (list): List of true decoded labels.
        decoded_preds (list): List of predicted decoded labels.
        unique_labels_no_bio (list): List of unique labels without BIO encoding.
        unique_labels_bio (list): List of unique labels with BIO encoding.
        args (Namespace): Arguments containing dataset name and test flag.
        targets_BIO (list): List of true labels in BIO format.

    Returns:
        strict_F1[2] (float): Strict F1 score.
    """

    decoded_preds_output = []
    for val in decoded_preds:
        if args.dataset_name == "ChemProt":
            decoded_preds_output.append(escape_unicode(extract_output(val)))

        else:
            decoded_preds_output.append(extract_output(val))

    predictions_BIO = create_bio_labels(org_tokens, decoded_preds_output)

    strict_F1, relaxed_F1, token_level = matrices_compute(
        targets_BIO, predictions_BIO, unique_labels_no_bio, unique_labels_bio
    )

    if args.isTest == False:

        print(
            f"Validation Strict Precision: {strict_F1[0]:.4f} | "
            f"Validation Strict Recall: {strict_F1[1]:.4f} | "
            f"Validation Strict F1: {strict_F1[2]:.4f} | "
            f"Validation Relaxed Precision: {relaxed_F1[0]:.4f} | "
            f"Validation Relaxed Recall: {relaxed_F1[1]:.4f} | "
            f"Validation Relaxed F1: {relaxed_F1[2]:.4f}"
        )
        print("Validation Token Level Scores", token_level, "\n")

    else:

        print(
            f"Test Strict Precision: {strict_F1[0]:.4f} | "
            f"Test Strict Recall: {strict_F1[1]:.4f} | "
            f"Test Strict F1: {strict_F1[2]:.4f} | "
            f"Test Relaxed Precision: {relaxed_F1[0]:.4f} | "
            f"Test Relaxed Recall: {relaxed_F1[1]:.4f} | "
            f"Test Relaxed F1: {relaxed_F1[2]:.4f}"
        )
        print("Test Token Level Scores", token_level, "\n")

        true_entites_counts, pred_entites_counts = entity_size(
            targets_BIO, predictions_BIO
        )

        test1_strict_F1, test1_relaxed_F1, test1_token_level = matrices_compute(
            true_entites_counts[0],
            pred_entites_counts[0],
            unique_labels_no_bio,
            unique_labels_bio,
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
            true_entites_counts[1],
            pred_entites_counts[1],
            unique_labels_no_bio,
            unique_labels_bio,
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
            true_entites_counts[2],
            pred_entites_counts[2],
            unique_labels_no_bio,
            unique_labels_bio,
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

    return strict_F1[2]
