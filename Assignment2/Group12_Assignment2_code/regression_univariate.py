"""
Regression Task - Dataset 1: Univariate (1-dimensional input).

Model: FCNN with ONE hidden layer (as required by the assignment for Dataset 1), a single
linear output node, trained by stochastic gradient descent backpropagation with the squared
error as the instantaneous loss. Several hidden-node counts are tried and the best one is
chosen on the validation set.

Run:  python regression_univariate.py
"""

import os

from data_prep import load_univariate_regression
from regression_experiment import run_regression_experiment

DATASET_TITLE = "Univariate regression dataset (1-D input)"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "regression_univariate")

# One hidden layer; each entry is the number of nodes in that hidden layer.
# The hidden-node counts are powers of two, up to 64.
ARCHITECTURES = [[8], [16], [32], [64]]

LEARNING_RATE = 0.01
EPOCHS = 300
PATIENCE = 30


def run():
    return run_regression_experiment(
        DATASET_TITLE, load_univariate_regression, ARCHITECTURES, OUT_DIR,
        learning_rate=LEARNING_RATE, epochs=EPOCHS, patience=PATIENCE,
        hidden_activation="logistic", seed=10,
        baseline_lr=0.01, baseline_epochs=300)


if __name__ == "__main__":
    run()
