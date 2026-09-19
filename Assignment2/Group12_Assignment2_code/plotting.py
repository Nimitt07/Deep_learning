"""
All plotting helpers shared by the four Assignment-2 experiment scripts.

matplotlib is used only to render figures -- every model, metric and decision boundary is
computed by our own code in fcnn.py / metrics.py.

Figures produced here cover the items asked for in the assignment:
  * average error vs epochs                       -> plot_error_curve
  * decision region superimposed on training data -> plot_decision_regions
  * average error vs epochs, ONE FIGURE PER ARCHITECTURE -> save_error_curve_per_architecture
  * outputs of every hidden/output node, ONE FIGURE PER NODE  -> save_all_node_output_plots
  * model output superimposed on target output    -> plot_fit_1d / plot_fit_2d
  * scatter of target output vs model output      -> plot_target_vs_model
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")          # head-less backend: write PNG files, never open a window
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)

CLASS_COLORS = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}


def _ensure_dir(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


# ---------------------------------------------------------------------------
# 1) Average error vs epochs
# ---------------------------------------------------------------------------

def plot_error_curve(history, title, out_path, total_epochs=None):
    """Average error E_av (y-axis) against epoch number (x-axis).

    The curve ENDS at the last epoch that was actually trained. When the stopping criterion
    fires early there is no training after that epoch and therefore no average error to plot,
    so nothing is drawn past the end of the run -- drawing a flat continuation would suggest
    the error had been measured while training carried on, which is not what happened. That
    the run stopped early is stated in the title instead.

    `total_epochs` is the epoch budget the run was given; it is only used for that note.
    """
    _ensure_dir(out_path)
    train = list(history["train_error"])
    val = list(history.get("val_error") or [])
    run = len(train)
    epochs = np.arange(1, run + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train, label="Training data", linewidth=1.6)
    if val:
        plt.plot(epochs, val, label="Validation data", linewidth=1.6)
        best = int(np.argmin(val)) + 1
        plt.plot(best, val[best - 1], "o", markersize=7, markerfacecolor="none",
                 markeredgecolor="black", markeredgewidth=1.4,
                 label=f"lowest validation error (epoch {best}, weights restored)")

    plt.xlim(1, run)
    plt.xlabel("Epochs")
    plt.ylabel("Average error  $E_{av}$")
    stopped_early = bool(total_epochs) and run < total_epochs
    subtitle = (f"\nstopped early after {run} of {total_epochs} epochs"
                if stopped_early else "")
    plt.title(title + subtitle, fontsize=11.5)
    plt.legend(fontsize=8.5)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def architecture_curve_file(label):
    """File name of one architecture's error curve: hidden_50.png, hidden_35-35.png, ..."""
    return f"hidden_{label}.png"


def save_error_curve_per_architecture(architectures, out_dir, dataset_title,
                                      learning_rate=None, total_epochs=None):
    """One error-vs-epochs figure PER architecture, written to <out_dir>/error_curves/.

    Every architecture gets its own graph rather than all of them being overlaid on a single
    axis with a legend: with several architectures the overlaid curves sit on top of each
    other and nothing can be read off them.

    `architectures` is a list of dicts with keys: label, arch, train_error_curve and
    (optionally) val_error_curve.
    """
    root = os.path.join(out_dir, "error_curves")
    written = []
    for a in architectures:
        out_path = os.path.join(root, architecture_curve_file(a["label"]))
        eta = f", eta = {learning_rate}" if learning_rate is not None else ""
        plot_error_curve(
            {"train_error": a["train_error_curve"],
             "val_error": a.get("val_error_curve") or []},
            f"{dataset_title}\nAverage error vs epochs - architecture {a['arch']} "
            f"(hidden nodes: {a['label']}{eta})",
            out_path,
            total_epochs=total_epochs)
        written.append(out_path)
    return written


# ---------------------------------------------------------------------------
# 2) Decision regions
# ---------------------------------------------------------------------------

def plot_decision_regions(predict_labels, X, y, classes, title, out_path,
                          n_points=350, pad=0.8):
    """Decision regions of the trained network, superimposed with the given data.

    `predict_labels` maps raw (unscaled) 2-D inputs to predicted class labels.
    """
    _ensure_dir(out_path)
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_points),
                         np.linspace(y_min, y_max, n_points))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.asarray(predict_labels(grid)).reshape(xx.shape)

    plt.figure(figsize=(6.8, 6.2))
    levels = [c - 0.5 for c in classes] + [classes[-1] + 0.5]
    plt.contourf(xx, yy, Z, levels=levels,
                 colors=[CLASS_COLORS[c] for c in classes], alpha=0.25)
    for c in classes:
        pts = X[y == c]
        plt.scatter(pts[:, 0], pts[:, 1], color=CLASS_COLORS[c], label=f"Class {c}",
                    s=12, edgecolors="k", linewidths=0.3)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(title, fontsize=11.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 4) / 5) Outputs of hidden nodes and output nodes
# ---------------------------------------------------------------------------

# Every hidden node and every output node gets its OWN figure, written to
#     <best_architecture>/node_outputs/<layer>/<split>/node_001.png, node_002.png, ...
# With 50-100 nodes per layer a shared grid figure would be far too crowded to read, so the
# nodes are kept in separate files and can be inspected one at a time.

NODE_FIG_DPI = 100


def plot_one_node_2d(X, values, title, out_path, cmap="viridis"):
    """One hidden / output node, 2-D input case.

    x-axis = x1, y-axis = x2 (the two input variables of each example),
    z-axis = the output of this node for that example.
    """
    _ensure_dir(out_path)
    fig = plt.figure(figsize=(4.6, 3.7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], values, c=values, cmap=cmap, s=4, alpha=0.8)
    ax.set_xlabel("$x_1$", fontsize=9, labelpad=2)
    ax.set_ylabel("$x_2$", fontsize=9, labelpad=2)
    ax.set_zlabel("node output", fontsize=9, labelpad=2)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=NODE_FIG_DPI)
    plt.close(fig)


def plot_one_node_1d(X, values, title, out_path):
    """One hidden / output node, 1-D input case: node output against the single input x."""
    _ensure_dir(out_path)
    order = np.argsort(X[:, 0])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(X[order, 0], values[order], linewidth=1.6, color="tab:blue")
    ax.set_xlabel("$x$", fontsize=9)
    ax.set_ylabel("node output", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=NODE_FIG_DPI)
    plt.close(fig)


def layer_dir_name(layer_index, n_hidden_layers):
    """Folder name for a layer: hidden_layer1, hidden_layer2, ..., output_layer."""
    return ("output_layer" if layer_index == n_hidden_layers
            else f"hidden_layer{layer_index + 1}")


def node_file_name(node_index):
    """File name of one node's figure (1-based, zero padded so they sort naturally)."""
    return f"node_{node_index + 1:03d}.png"


def save_all_node_output_plots(net, splits, out_dir, dataset_title, input_dim,
                               transform=None):
    """Write ONE figure per node, for every layer and every split (train / val / test).

    `splits` is a list of (split_name, X_raw) pairs; `transform` maps raw inputs to the
    scaled inputs the network was trained on. Returns {layer_dir: number of nodes} so the
    caller knows how many figures were produced per layer.
    """
    root = os.path.join(out_dir, "node_outputs")
    node_counts = {}
    for split_name, X_raw in splits:
        X_in = transform(X_raw) if transform is not None else X_raw
        layer_outputs = net.forward_all(X_in)
        n_hidden_layers = len(layer_outputs) - 1
        for li, values in enumerate(layer_outputs):
            layer_dir = layer_dir_name(li, n_hidden_layers)
            label = ("Output" if li == n_hidden_layers else f"Hidden layer {li + 1}")
            node_counts[layer_dir] = values.shape[1]
            for k in range(values.shape[1]):
                out_path = os.path.join(root, layer_dir, split_name, node_file_name(k))
                title = (f"{dataset_title}\n{label} node {k + 1} - {split_name} data")
                if input_dim == 1:
                    plot_one_node_1d(X_raw, values[:, k], title, out_path)
                else:
                    plot_one_node_2d(X_raw, values[:, k], title, out_path)
    return node_counts


# ---------------------------------------------------------------------------
# Regression-specific figures
# ---------------------------------------------------------------------------

def plot_fit_1d(X, y_true, y_pred, title, out_path):
    """Model output superimposed on the target output (x on the x-axis, y on the y-axis)."""
    _ensure_dir(out_path)
    order = np.argsort(X[:, 0])
    plt.figure(figsize=(7, 5))
    plt.plot(X[order, 0], y_true[order], "o", markersize=3, alpha=0.55,
             label="Target output")
    plt.plot(X[order, 0], y_pred[order], "-", color="red", linewidth=1.8,
             label="Model output")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title(title, fontsize=11.5)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_fit_2d(X, y_true, y_pred, title, out_path):
    """Model output superimposed on the target output, with x1 and x2 as the input axes."""
    _ensure_dir(out_path)
    fig = plt.figure(figsize=(8.5, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], y_true, s=5, alpha=0.35, color="tab:blue",
               label="Target output")
    ax.scatter(X[:, 0], X[:, 1], y_pred, s=5, alpha=0.35, color="tab:red",
               label="Model output")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$y$")
    ax.set_title(title, fontsize=11.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_target_vs_model(y_true, y_pred, title, out_path):
    """Scatter plot with the target output on the x-axis and the model output on the y-axis."""
    _ensure_dir(out_path)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=9, alpha=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "k--", linewidth=1, label="Ideal ($y_{model} = y_{target}$)")
    plt.xlabel("Target output")
    plt.ylabel("Model output")
    plt.title(title, fontsize=11.5)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Model-selection summary figures
# ---------------------------------------------------------------------------

def plot_model_selection(labels, train_values, val_values, ylabel, title, out_path):
    """Bar chart of a metric across the architectures that were compared."""
    _ensure_dir(out_path)
    idx = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(max(7, 1.3 * len(labels)), 5))
    plt.bar(idx - width / 2, train_values, width, label="Training")
    plt.bar(idx + width / 2, val_values, width, label="Validation")
    plt.xticks(idx, labels, rotation=30, ha="right", fontsize=8)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=11.5)
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
