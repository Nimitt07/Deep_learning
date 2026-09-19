"""
Assignment-1 baseline: the single-neuron (perceptron) model.

This module reproduces the Assignment-1 models so that requirement "comparison of
performance with that of the single neuron model (Assignment-1)" can be answered on
*exactly* the same 60/20/20 split and with exactly the same metric code that the FCNN
results use. Nothing here is new modelling work -- it is the Assignment-1 algorithm:

  * classification : one perceptron per pair of classes (one-against-one), combined by
                     majority voting with confidence used to break ties;
  * regression     : a single neuron with a linear activation.
"""

import numpy as np

from perceptron import Perceptron


# ---------------------------------------------------------------------------
# Classification: one-against-one perceptrons
# ---------------------------------------------------------------------------

def _target_for(activation, is_c1):
    if activation == "logistic":
        return 1.0 if is_c1 else 0.0
    return 1.0 if is_c1 else -1.0          # tanh


def train_ovo_perceptrons(X_train, y_train, activation="logistic",
                          learning_rate=0.05, epochs=300, seed=0):
    """Train one perceptron per class pair. Returns (classifiers, error_curves)."""
    classes = sorted(set(y_train.tolist()))
    classifiers, error_curves = {}, {}
    pair_id = 0
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            c1, c2 = classes[i], classes[j]
            mask = (y_train == c1) | (y_train == c2)
            Xp = X_train[mask]
            yp = np.array([_target_for(activation, lbl == c1) for lbl in y_train[mask]])
            p = Perceptron(n_inputs=Xp.shape[1], activation=activation,
                           learning_rate=learning_rate, seed=seed + pair_id)
            error_curves[(c1, c2)] = p.train(Xp, yp, epochs=epochs, seed=seed + pair_id)
            classifiers[(c1, c2)] = p
            pair_id += 1
    return classifiers, error_curves


def _predict_pair(p, X, c1, c2, activation):
    out = p.predict_raw(X)
    if activation == "logistic":
        return np.where(out >= 0.5, c1, c2), np.abs(out - 0.5)
    return np.where(out >= 0.0, c1, c2), np.abs(out)


def predict_ovo(classifiers, X, activation="logistic"):
    """Majority voting over the pairwise classifiers; ties broken by summed confidence."""
    classes = sorted({c for pair in classifiers for c in pair})
    n = X.shape[0]
    votes = {c: np.zeros(n) for c in classes}
    conf_sum = {c: np.zeros(n) for c in classes}
    for (c1, c2), p in classifiers.items():
        pred, conf = _predict_pair(p, X, c1, c2, activation)
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
            final[idx] = classes[candidates[np.argmax(conf_matrix[idx, candidates])]]
    return final


# ---------------------------------------------------------------------------
# Regression: single linear neuron
# ---------------------------------------------------------------------------

def train_linear_neuron(X_train, y_train, learning_rate=0.01, epochs=300, seed=0):
    p = Perceptron(n_inputs=X_train.shape[1], activation="linear",
                   learning_rate=learning_rate, seed=seed)
    errors = p.train(X_train, y_train, epochs=epochs, seed=seed)
    return p, errors
