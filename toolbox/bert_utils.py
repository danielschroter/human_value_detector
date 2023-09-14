
import numpy as np
from torchmetrics import F1Score, Recall, Precision, Accuracy
from sklearn.metrics import classification_report
import torch


def max_for_thres(y_pred, y_true, label_columns, average):
    """
    Calculates the maximum F1 score or custom F1 score threshold for a given set of predicted and true labels.

    Args:
        y_pred (list or array-like): A list or array-like object containing the predicted labels.
        y_true (list or array-like): A list or array-like object containing the true labels.
        label_columns (list): A list of column labels for multilabel classification.
        average (str): A string indicating the averaging strategy for calculating the F1 score. 
            It can be 'macro', 'micro', 'weighted', or 'custom'.

    Returns:
        float: The threshold value that corresponds to the maximum F1 score or custom F1 score.
    """
    values = {}
    for i in np.arange(0.0,1.0,0.01):
        if average=="custom":
            recall = Recall(num_labels=len(label_columns), threshold=i, average="macro", task="multilabel")
            precision = Precision(num_labels=len(label_columns), threshold=i, average="macro", task="multilabel")

            rec_t = recall(y_pred, y_true)
            pre_t = precision(y_pred, y_true)

            if (rec_t + pre_t) != 0:
                f1_custom = (2*rec_t*pre_t/(rec_t+pre_t))
                values[i] = f1_custom

        else:
            f1_score = F1Score(num_labels=len(label_columns), threshold=i, average=average, task="multilabel")
            score = f1_score(y_pred,y_true)
            values[i] = score

    return max(values, key=values.get)


def max_for_thres_accuracy(y_pred, y_true, label_columns, average):
    """
    Calculates the maximum accuracy score for different threshold values in a binary classification problem.

    Args:
        y_pred (list or array-like): Predicted probabilities or scores for each sample.
        y_true (list or array-like): True labels for each sample.
        label_columns (list): Column names or labels for the binary classes.
        average (str): Type of averaging to be performed for multi-class classification.

    Returns:
        float: The threshold value that yields the maximum accuracy score for the given predicted probabilities and true labels.
    """
    values = {}
    for i in np.arange(0.0, 1.0, 0.01):
        acc = Accuracy(threshold=i, task="binary")
        score = acc(y_pred, y_true)
        values[i] = score

    return max(values, key=values.get)



