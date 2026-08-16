"""
Shared logic for the two classification experiments (LS and NLS datasets):
  - one-against-one training of pairwise perceptrons for 3-class problems
  - one-against-one voting to obtain a combined multi-class prediction
  - decision region plotting (per pair, and combined)
  - training-error curve plotting
  - evaluation report generation using the from-scratch metrics in metrics.py

No ML/plotting shortcuts from sklearn are used; matplotlib is only for rendering plots.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perceptron import Perceptron
import metrics as M

CLASS_COLORS = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}


def _short_name(dataset_name):
    """Short code for plot titles so long dataset names don't get clipped by matplotlib."""
    return dataset_name.split(" ")[0]


def target_for(activation, is_c1):
    """Encode the binary target for a one-vs-one pair given the chosen activation."""
    if activation == "logistic":
        return 1.0 if is_c1 else 0.0
    return 1.0 if is_c1 else -1.0  # tanh


def train_ovo_classifiers(X_train, y_train, activation, learning_rate, epochs, seed=0):
    """Train one perceptron per class pair. Returns (classifiers dict, error_curves dict)."""
    classes = sorted(set(y_train.tolist()))
    classifiers = {}
    error_curves = {}
    pair_id = 0
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            c1, c2 = classes[i], classes[j]
            mask = (y_train == c1) | (y_train == c2)
            Xp = X_train[mask]
            y_orig = y_train[mask]
            yp = np.array([target_for(activation, label == c1) for label in y_orig])
            p = Perceptron(n_inputs=Xp.shape[1], activation=activation,
                            learning_rate=learning_rate, seed=seed + pair_id)
            errs = p.train(Xp, yp, epochs=epochs, seed=seed + pair_id)
            classifiers[(c1, c2)] = p
            error_curves[(c1, c2)] = errs
            pair_id += 1
    return classifiers, error_curves


def predict_pair(p, X, c1, c2, activation):
    """Predict class labels (c1 or c2) and a confidence score for a single pairwise classifier."""
    out = p.predict_raw(X)
    if activation == "logistic":
        pred = np.where(out >= 0.5, c1, c2)
        conf = np.abs(out - 0.5)
    else:  # tanh
        pred = np.where(out >= 0.0, c1, c2)
        conf = np.abs(out)
    return pred, conf


def predict_ovo(classifiers, X, activation):
    """Combine pairwise predictions via majority voting; ties broken by summed confidence."""
    classes = sorted(set(c for pair in classifiers for c in pair))
    n = X.shape[0]
    votes = {c: np.zeros(n) for c in classes}
    conf_sum = {c: np.zeros(n) for c in classes}
    for (c1, c2), p in classifiers.items():
        pred, conf = predict_pair(p, X, c1, c2, activation)
        for c in (c1, c2):
            mask = pred == c
            votes[c][mask] += 1
            conf_sum[c][mask] += conf[mask]
    vote_matrix = np.stack([votes[c] for c in classes], axis=1)
    conf_matrix = np.stack([conf_sum[c] for c in classes], axis=1)
    final = np.empty(n, dtype=int)
    for idx in range(n):
        max_v = vote_matrix[idx].max()
        candidates = np.where(vote_matrix[idx] == max_v)[0]
        if len(candidates) == 1:
            final[idx] = classes[candidates[0]]
        else:
            best = candidates[np.argmax(conf_matrix[idx, candidates])]
            final[idx] = classes[best]
    return final


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_error_curves(error_curves, activation, dataset_name, out_path):
    plt.figure(figsize=(7, 5))
    for (c1, c2), errs in error_curves.items():
        plt.plot(errs, label=f"Class {c1} vs Class {c2}")
    plt.xlabel("Epochs")
    plt.ylabel("Average Error")
    plt.title(f"{_short_name(dataset_name)}: Training Error vs Epochs\n({activation} activation)", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _mesh_grid(X, pad=1.0, n_points=300):
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_points), np.linspace(y_min, y_max, n_points))
    return xx, yy


def plot_decision_region_pair(p, X_train, y_train, c1, c2, activation, dataset_name, out_path):
    mask = (y_train == c1) | (y_train == c2)
    Xp, yp = X_train[mask], y_train[mask]
    xx, yy = _mesh_grid(Xp)
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred, _ = predict_pair(p, grid, c1, c2, activation)
    Z = pred.reshape(xx.shape)

    lo, hi = min(c1, c2), max(c1, c2)
    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, Z, levels=[lo - 0.5, (lo + hi) / 2, hi + 0.5],
                 colors=[CLASS_COLORS[lo], CLASS_COLORS[hi]],
                 alpha=0.25)
    for c in (c1, c2):
        pts = Xp[yp == c]
        plt.scatter(pts[:, 0], pts[:, 1], color=CLASS_COLORS[c], label=f"Class {c}", s=12, edgecolors="k", linewidths=0.3)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"{_short_name(dataset_name)}: Class {c1} vs Class {c2} ({activation})\n(training data)", fontsize=11.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_decision_region_combined(classifiers, X_train, y_train, activation, dataset_name, out_path):
    classes = sorted(set(y_train.tolist()))
    xx, yy = _mesh_grid(X_train)
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred = predict_ovo(classifiers, grid, activation)
    Z = pred.reshape(xx.shape)

    plt.figure(figsize=(6.5, 6.5))
    levels = [c - 0.5 for c in classes] + [classes[-1] + 0.5]
    plt.contourf(xx, yy, Z, levels=levels, colors=[CLASS_COLORS[c] for c in classes], alpha=0.25)
    for c in classes:
        pts = X_train[y_train == c]
        plt.scatter(pts[:, 0], pts[:, 1], color=CLASS_COLORS[c], label=f"Class {c}", s=12, edgecolors="k", linewidths=0.3)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"{_short_name(dataset_name)}: Combined Decision Regions ({activation})\n(one-against-one, training data)", fontsize=11.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------

def evaluate_and_report(y_true, y_pred, classes, dataset_name, activation, out_path):
    y_true_l = y_true.tolist()
    y_pred_l = y_pred.tolist()
    cm = M.confusion_matrix(y_true_l, y_pred_l, classes)
    acc = M.accuracy(y_true_l, y_pred_l)
    prec = M.precision_per_class(cm)
    rec = M.recall_per_class(cm)
    f1 = M.f_measure_per_class(prec, rec)
    mean_prec = M.mean(prec)
    mean_rec = M.mean(rec)
    mean_f1 = M.mean(f1)

    lines = []
    lines.append(f"{dataset_name} - Activation: {activation}")
    lines.append("=" * 60)
    lines.append("Confusion Matrix (rows = true, cols = predicted)")
    header = "        " + "".join(f"Pred{c:<8}" for c in classes)
    lines.append(header)
    for i, c in enumerate(classes):
        row = f"True{c:<4}" + "".join(f"{cm[i][j]:<12}" for j in range(len(classes)))
        lines.append(row)
    lines.append("")
    lines.append(f"Overall Accuracy: {acc * 100:.2f}%")
    lines.append("")
    lines.append(f"{'Class':<8}{'Precision':<12}{'Recall':<12}{'F-measure':<12}")
    for i, c in enumerate(classes):
        lines.append(f"{c:<8}{prec[i]:<12.4f}{rec[i]:<12.4f}{f1[i]:<12.4f}")
    lines.append(f"{'Mean':<8}{mean_prec:<12.4f}{mean_rec:<12.4f}{mean_f1:<12.4f}")

    report = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(report)
    return report
