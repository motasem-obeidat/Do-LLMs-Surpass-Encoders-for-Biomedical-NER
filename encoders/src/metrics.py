from ner_SemEval import Evaluator
from sklearn_crfsuite.metrics import flat_classification_report


# Calculates the F1 score given precision and recall values.
def calculate_F1(precision, recall):
    if (precision + recall) == 0:
        return 0

    return 2 * (precision * recall) / (precision + recall)


def matrices_compute(true_labels, pred_labels, labels_to_ids):
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

    entity_level_labels = set(
        label[2:] for label in labels_to_ids.keys() if label != "O"
    )

    evaluator = Evaluator(true_labels, pred_labels, entity_level_labels)
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

    token_level_labels = [label for label in labels_to_ids.keys() if label != "O"]
    sorted_labels = sorted(token_level_labels, key=lambda name: (name[1:], name[0]))
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
