
import numpy as np
from torchmetrics import F1Score, Recall, Precision, Accuracy
from sklearn.metrics import classification_report
import torch


def max_for_thres(y_pred, y_true, label_columns, average):

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

    values = {}
    for i in np.arange(0.0,1.0,0.01):
        acc = Accuracy(threshold=i, task="binary")
        score = acc(y_pred,y_true)
        values[i] = score

    return max(values, key=values.get)




# target = torch.tensor([[0, 1, 0], [1, 0, 1]])
# preds = torch.tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
#
# max_for_thres(preds, target, label_columns=["a","b","c"], average="macro")



