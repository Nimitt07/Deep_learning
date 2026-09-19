"""
Classification Task - Dataset 1: Linearly Separable (LS), 3 classes, 2-D input.

Model: FCNN with ONE hidden layer (as required by the assignment for Dataset 1), trained by
stochastic gradient descent backpropagation with the squared error as the instantaneous loss.
Several hidden-node counts are tried and the best one is chosen on the validation set.

Run:  python classification_ls.py
"""

import os

from data_prep import load_ls_classification
from classification_experiment import run_classification_experiment

DATASET_TITLE = "LS dataset (linearly separable, 3 classes)"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "classification_ls")

# One hidden layer; each entry is the number of nodes in that hidden layer.
# The hidden-node counts are powers of two, up to 64.
ARCHITECTURES = [[8], [16], [32], [64]]

LEARNING_RATE = 0.05
EPOCHS = 150
PATIENCE = 20


def run():
    return run_classification_experiment(
        DATASET_TITLE, load_ls_classification, ARCHITECTURES, OUT_DIR,
        learning_rate=LEARNING_RATE, epochs=EPOCHS, patience=PATIENCE,
        hidden_activation="logistic", seed=1,
        baseline_lr=0.05, baseline_epochs=300)


if __name__ == "__main__":
    run()
