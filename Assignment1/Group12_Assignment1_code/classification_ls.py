"""
Classification Task - Dataset 1: Linearly Separable (LS), 3 classes, 2D.

For each activation function (logistic, tan-hyperbolic):
  - train one-against-one perceptrons (Class1 vs Class2, Class1 vs Class3, Class2 vs Class3)
    using gradient descent, from scratch
  - plot average error vs epochs
  - plot decision regions for each pair, and the combined 3-class decision region
    (superimposed on training data only)
  - compute confusion matrix, accuracy, precision/recall/F-measure per class and their means
    on the held-out test data
"""

import os
from data_prep import load_ls_classification
from classification_common import (
    train_ovo_classifiers, predict_ovo,
    plot_error_curves, plot_decision_region_pair, plot_decision_region_combined,
    evaluate_and_report,
)

DATASET_NAME = "LS Dataset (Linearly Separable)"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "classification_ls")
LEARNING_RATE = 0.05
EPOCHS = 300


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    X_train, y_train, X_test, y_test = load_ls_classification()
    classes = sorted(set(y_train.tolist()))

    for activation in ("logistic", "tanh"):
        act_dir = os.path.join(OUT_DIR, activation)
        os.makedirs(act_dir, exist_ok=True)

        classifiers, error_curves = train_ovo_classifiers(
            X_train, y_train, activation, LEARNING_RATE, EPOCHS, seed=1
        )

        plot_error_curves(error_curves, activation, DATASET_NAME,
                           os.path.join(act_dir, "error_vs_epochs.png"))

        for (c1, c2), p in classifiers.items():
            plot_decision_region_pair(p, X_train, y_train, c1, c2, activation, DATASET_NAME,
                                       os.path.join(act_dir, f"decision_region_{c1}v{c2}.png"))

        plot_decision_region_combined(classifiers, X_train, y_train, activation, DATASET_NAME,
                                       os.path.join(act_dir, "decision_region_combined.png"))

        y_pred = predict_ovo(classifiers, X_test, activation)
        report = evaluate_and_report(y_test, y_pred, classes, DATASET_NAME, activation,
                                      os.path.join(act_dir, "evaluation_report.txt"))
        print(report)
        print()


if __name__ == "__main__":
    run()
