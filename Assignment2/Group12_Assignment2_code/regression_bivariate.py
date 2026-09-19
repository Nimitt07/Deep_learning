"""
Regression Task - Dataset 2: Bivariate (2-dimensional input).

Model: FCNN with ONE as well as TWO hidden layers (both are required by the assignment for
Dataset 2), a single linear output node, trained by stochastic gradient descent
backpropagation with the squared error as the instantaneous loss. Several hidden-node counts
are tried in each case and the best architecture overall is chosen on the validation set.

Run:  python regression_bivariate.py
"""

import os

from data_prep import load_bivariate_regression
from regression_experiment import run_regression_experiment

DATASET_TITLE = "Bivariate regression dataset (2-D input)"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "regression_bivariate")

# The assignment asks for one AS WELL AS two hidden layers on this dataset. The hidden-node
# counts are powers of two up to 64, and where there are two hidden layers the first is wider
# than the second (the network narrows towards the output).
# One hidden layer ...                 ... and two hidden layers
ARCHITECTURES = [[16], [32], [64],     [32, 16], [64, 32]]

LEARNING_RATE = 0.01
# This dataset has 6121 training examples, so one epoch is 6121 weight updates and training is
# comparatively slow. 60 epochs was verified to be too few: the larger architectures were still
# improving, which made the RMSE-vs-complexity comparison an artefact of undertraining rather
# than a property of the models. 200 epochs lets every architecture settle.
EPOCHS = 200
PATIENCE = 25


def run():
    return run_regression_experiment(
        DATASET_TITLE, load_bivariate_regression, ARCHITECTURES, OUT_DIR,
        learning_rate=LEARNING_RATE, epochs=EPOCHS, patience=PATIENCE,
        hidden_activation="logistic", seed=20,
        baseline_lr=0.01, baseline_epochs=60)


if __name__ == "__main__":
    run()
