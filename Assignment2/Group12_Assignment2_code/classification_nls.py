"""
Classification Task - Dataset 2: Nonlinearly Separable (NLS), 3 classes, 2-D input.

Model: FCNN with TWO hidden layers (as required by the assignment for Dataset 2), trained by
stochastic gradient descent backpropagation with the squared error as the instantaneous loss.
Several (h1, h2) hidden-node combinations are tried and the best one is chosen on the
validation set.

Run:  python classification_nls.py
"""

import os

from data_prep import load_nls_classification
from classification_experiment import run_classification_experiment

DATASET_TITLE = "NLS dataset (nonlinearly separable, 3 classes)"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "classification_nls")

# Two hidden layers; each entry is [nodes in hidden layer 1, nodes in hidden layer 2].
# The hidden-node counts are powers of two up to 64, and the first hidden layer is always
# wider than the second (the network narrows towards the output).
ARCHITECTURES = [[16, 8], [32, 16], [64, 16], [64, 32]]

LEARNING_RATE = 0.05
EPOCHS = 300
PATIENCE = 40


def run():
    return run_classification_experiment(
        DATASET_TITLE, load_nls_classification, ARCHITECTURES, OUT_DIR,
        learning_rate=LEARNING_RATE, epochs=EPOCHS, patience=PATIENCE,
        hidden_activation="logistic", seed=2,
        baseline_lr=0.05, baseline_epochs=300)


if __name__ == "__main__":
    run()
