"""
Regression Task - Dataset 2: Bivariate (2-dimensional input).

Trains a single perceptron with a linear activation function using gradient descent
(from scratch). Produces:
  1) average error vs epochs plot
  2) RMSE and %RMSE on training and test data
  3) 3D plot of model output superimposed on target output (x1, x2 axes; y-axis is target/model value)
  4) scatter plot of target output (x-axis) vs model output (y-axis)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from data_prep import load_bivariate_regression
from perceptron import Perceptron
import metrics as M

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "regression_bivariate")
LEARNING_RATE = 0.01
EPOCHS = 200


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    X_train, y_train, X_test, y_test = load_bivariate_regression()

    p = Perceptron(n_inputs=2, activation="linear", learning_rate=LEARNING_RATE, seed=20)
    errors = p.train(X_train, y_train, epochs=EPOCHS, seed=20)

    # 1) Error vs epochs
    plt.figure(figsize=(7, 5))
    plt.plot(errors)
    plt.xlabel("Epochs")
    plt.ylabel("Average Error")
    plt.title("Bivariate Regression: Training Error vs Epochs")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "error_vs_epochs.png"), dpi=150)
    plt.close()

    # 2) RMSE / %RMSE
    pred_train = p.predict_raw(X_train)
    pred_test = p.predict_raw(X_test)
    rmse_train = M.rmse(y_train.tolist(), pred_train.tolist())
    rmse_test = M.rmse(y_test.tolist(), pred_test.tolist())
    pct_rmse_train = M.percent_rmse(y_train.tolist(), pred_train.tolist())
    pct_rmse_test = M.percent_rmse(y_test.tolist(), pred_test.tolist())

    report_lines = [
        "Bivariate Regression - Linear Perceptron",
        "=" * 50,
        f"Learned weights: {p.w[1:].tolist()}, bias (w0): {p.w[0]:.6f}",
        "",
        f"RMSE (train): {rmse_train:.6f}",
        f"RMSE (test):  {rmse_test:.6f}",
        f"%RMSE (train): {pct_rmse_train:.4f}%",
        f"%RMSE (test):  {pct_rmse_test:.4f}%",
    ]
    report = "\n".join(report_lines)
    with open(os.path.join(OUT_DIR, "evaluation_report.txt"), "w") as f:
        f.write(report)
    print(report)

    # 3) 3D plot: model output superimposed on target output
    for split_name, X, y, pred in (
        ("train", X_train, y_train, pred_train),
        ("test", X_test, y_test, pred_test),
    ):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X[:, 0], X[:, 1], y, s=6, alpha=0.4, label="Target output", color="tab:blue")
        ax.scatter(X[:, 0], X[:, 1], pred, s=6, alpha=0.4, label="Model output", color="tab:red")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        ax.set_title(f"Bivariate Regression: Model vs Target Output ({split_name})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"model_vs_target_{split_name}.png"), dpi=150)
        plt.close()

    # 4) Scatter: target (x-axis) vs model output (y-axis)
    for split_name, y, pred in (("train", y_train, pred_train), ("test", y_test, pred_test)):
        plt.figure(figsize=(6, 6))
        plt.scatter(y, pred, s=8, alpha=0.5)
        lims = [min(y.min(), pred.min()), max(y.max(), pred.max())]
        plt.plot(lims, lims, "k--", linewidth=1, label="Ideal (y = x)")
        plt.xlabel("Target output")
        plt.ylabel("Model output")
        plt.title(f"Bivariate Regression: Target vs Model Output ({split_name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"scatter_target_vs_model_{split_name}.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    run()
