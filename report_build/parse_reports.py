"""Parse the plain-text evaluation reports produced by the assignment code, so the
report generator always reflects the actual numbers on disk (no manual transcription)."""
import re


def parse_classification_report(path):
    with open(path) as f:
        text = f.read()

    classes = [int(c) for c in re.findall(r"Pred(\d+)", text.splitlines()[3])]
    n = len(classes)

    cm = []
    for line in text.splitlines():
        m = re.match(r"True\d+\s+(.*)", line)
        if m:
            cm.append([int(x) for x in m.group(1).split()[:n]])

    acc = float(re.search(r"Overall Accuracy:\s*([\d.]+)%", text).group(1))

    per_class = {}
    mean_row = {}
    for line in text.splitlines():
        m = re.match(r"(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            per_class[int(m.group(1))] = {
                "precision": float(m.group(2)),
                "recall": float(m.group(3)),
                "f1": float(m.group(4)),
            }
        m2 = re.match(r"Mean\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m2:
            mean_row = {
                "precision": float(m2.group(1)),
                "recall": float(m2.group(2)),
                "f1": float(m2.group(3)),
            }

    return {
        "classes": classes,
        "confusion_matrix": cm,
        "accuracy": acc,
        "per_class": per_class,
        "mean": mean_row,
    }


def parse_regression_report(path):
    with open(path) as f:
        text = f.read()

    def find(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    return {
        "rmse_train": find(r"RMSE \(train\):\s*([\d.]+)"),
        "rmse_test": find(r"RMSE \(test\):\s*([\d.]+)"),
        "pct_rmse_train": find(r"%RMSE \(train\):\s*([\d.]+)"),
        "pct_rmse_test": find(r"%RMSE \(test\):\s*([\d.]+)"),
    }
