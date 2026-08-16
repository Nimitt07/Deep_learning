"""
Evaluation metrics implemented from scratch (no scikit-learn / built-in metric functions).
Covers classification metrics (confusion matrix, accuracy, precision, recall, F-measure)
and regression metrics (RMSE, %RMSE).
"""


def confusion_matrix(y_true, y_pred, labels):
    """Build a confusion matrix as a list of lists.

    cm[i][j] = number of samples whose true label is labels[i] and predicted label is labels[j].
    """
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    n = len(labels)
    cm = [[0 for _ in range(n)] for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        i = label_to_idx[t]
        j = label_to_idx[p]
        cm[i][j] += 1
    return cm


def accuracy(y_true, y_pred):
    correct = 0
    total = len(y_true)
    for t, p in zip(y_true, y_pred):
        if t == p:
            correct += 1
    return correct / total if total > 0 else 0.0


def precision_per_class(cm):
    """Precision for class j = TP_j / (sum over i of cm[i][j])  (predicted-positive column sum)."""
    n = len(cm)
    precisions = []
    for j in range(n):
        tp = cm[j][j]
        predicted_positive = 0
        for i in range(n):
            predicted_positive += cm[i][j]
        precisions.append(tp / predicted_positive if predicted_positive > 0 else 0.0)
    return precisions


def recall_per_class(cm):
    """Recall for class i = TP_i / (sum over j of cm[i][j])  (actual-positive row sum)."""
    n = len(cm)
    recalls = []
    for i in range(n):
        tp = cm[i][i]
        actual_positive = 0
        for j in range(n):
            actual_positive += cm[i][j]
        recalls.append(tp / actual_positive if actual_positive > 0 else 0.0)
    return recalls


def f_measure_per_class(precisions, recalls):
    f_scores = []
    for p, r in zip(precisions, recalls):
        if (p + r) > 0:
            f_scores.append(2 * p * r / (p + r))
        else:
            f_scores.append(0.0)
    return f_scores


def mean(values):
    return sum(values) / len(values) if len(values) > 0 else 0.0


def rmse(y_true, y_pred):
    n = len(y_true)
    sq_error_sum = 0.0
    for t, p in zip(y_true, y_pred):
        sq_error_sum += (t - p) ** 2
    return (sq_error_sum / n) ** 0.5 if n > 0 else 0.0


def percent_rmse(y_true, y_pred):
    """% RMSE = RMSE normalized by the mean of the true target values, expressed as a percentage."""
    r = rmse(y_true, y_pred)
    mean_true = mean(list(y_true))
    return (r / abs(mean_true)) * 100 if mean_true != 0 else float("inf")
