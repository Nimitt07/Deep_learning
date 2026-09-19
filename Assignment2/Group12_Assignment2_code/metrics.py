"""
Evaluation metrics implemented from scratch (no scikit-learn / built-in metric functions).

Classification : confusion matrix, accuracy, precision / recall / F-measure per class and
                 their means.
Regression     : RMSE and %RMSE.

Carried over from Assignment-1 (metrics.py) so that the Assignment-2 numbers are computed
with exactly the same definitions and are therefore directly comparable.
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


# ---------------------------------------------------------------------------
# Convenience: full classification report as text
# ---------------------------------------------------------------------------

def classification_report_text(y_true, y_pred, labels, title):
    """Confusion matrix + accuracy + per-class and mean precision/recall/F-measure as text."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels)
    acc = accuracy(y_true, y_pred)
    prec = precision_per_class(cm)
    rec = recall_per_class(cm)
    f1 = f_measure_per_class(prec, rec)

    lines = [title, "=" * max(60, len(title))]
    lines.append("Confusion Matrix (rows = true class, cols = predicted class)")
    lines.append("        " + "".join(f"Pred{c:<8}" for c in labels))
    for i, c in enumerate(labels):
        lines.append(f"True{c:<4}" + "".join(f"{cm[i][j]:<12}" for j in range(len(labels))))
    lines.append("")
    lines.append(f"Classification accuracy : {acc * 100:.2f}%")
    lines.append("")
    lines.append(f"{'Class':<8}{'Precision':<12}{'Recall':<12}{'F-measure':<12}")
    for i, c in enumerate(labels):
        lines.append(f"{c:<8}{prec[i]:<12.4f}{rec[i]:<12.4f}{f1[i]:<12.4f}")
    lines.append(f"{'Mean':<8}{mean(prec):<12.4f}{mean(rec):<12.4f}{mean(f1):<12.4f}")

    stats = {
        "confusion_matrix": cm,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f_measure": f1,
        "mean_precision": mean(prec),
        "mean_recall": mean(rec),
        "mean_f_measure": mean(f1),
    }
    return "\n".join(lines), stats
