"""
Single-neuron Perceptron trained with the gradient-descent parameter-learning procedure
(from scratch; numpy is used only for array bookkeeping, not as an ML library).

This follows the exact algorithm taught in class:

Given training data D = {(x_n, y_n)}_{n=1}^N, x_n in R^d, and y_n in {0,1} or {-1,+1}
(classification: logistic / tanh neuron) or y_n in R^1 (regression: linear neuron):

  1. Initialize w with random values.
     w = [w0, w1, ..., wd]^T -- the bias is absorbed into w as w0, using the augmented
     input x_hat_n = [1, x_n1, ..., x_nd]^T.
  2. Choose a training example x_n.
  3. Compute output of the neuron: s_n = f(a_n), where a_n = w^T x_hat_n.
  4. Compute instantaneous error: E_n(w) = 1/2 (y_n - s_n)^2.
  5. Compute change in weight: delta_w = eta * delta_o * x_hat_n,
     where delta_o = (y_n - s_n) * f'(a_n)   (0 < eta <= 1 is the learning rate).
  6. Update the weight: w = w + delta_w.
  7. Repeat steps 2 and 6 till all training examples are presented once (an epoch).
  8. Compute the average error: E_av = (1/N) * sum_n E_n(w).
  9. Repeat steps 2 to 8 till the convergence criterion is satisfied.

Activation functions f and their derivatives f'(a), expressed in terms of the neuron's
own output s = f(a) as in the slides:
  - logistic : f(a) = 1 / (1 + exp(-a)),  f'(a) = f(a) (1 - f(a))   -> targets in {0, 1}
  - tanh     : f(a) = tanh(a),             f'(a) = 1 - f(a)^2        -> targets in {-1, 1}
  - linear   : f(a) = a,                   f'(a) = 1                 -> real-valued targets
"""

import numpy as np


class Perceptron:
    def __init__(self, n_inputs, activation="logistic", learning_rate=0.01, seed=None):
        if activation not in ("logistic", "tanh", "linear"):
            raise ValueError(f"Unsupported activation: {activation}")
        rng = np.random.default_rng(seed)
        # Step 1: initialize w = [w0, w1, ..., wd]^T with random values (w0 is the bias).
        self.w = rng.uniform(-0.5, 0.5, size=n_inputs + 1)
        self.activation = activation
        self.eta = learning_rate

    @staticmethod
    def _augment(X):
        """Builds x_hat_n = [1, x_n1, ..., x_nd]^T for every row of X."""
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def _f(self, a):
        """Activation function f(a)."""
        if self.activation == "logistic":
            return 1.0 / (1.0 + np.exp(-a))
        if self.activation == "tanh":
            return np.tanh(a)
        return a  # linear

    def _f_prime(self, s):
        """f'(a), expressed in terms of the neuron output s = f(a)."""
        if self.activation == "logistic":
            return s * (1.0 - s)
        if self.activation == "tanh":
            return 1.0 - s ** 2
        return np.ones_like(s)  # linear

    def net_input(self, X):
        """a_n = w^T x_hat_n for every row of X."""
        return self._augment(X) @ self.w

    def predict_raw(self, X):
        """s_n = f(a_n) for every row of X."""
        return self._f(self.net_input(X))

    def train(self, X, y, epochs=500, tol=1e-6, seed=None):
        """
        Online gradient-descent training (steps 2-9 above).
        Returns E_av for every epoch, i.e. the average-error-vs-epochs curve.
        """
        rng = np.random.default_rng(seed)
        X_hat = self._augment(X)          # x_hat_n = [1, x_n1, ..., x_nd]^T, precomputed once
        N = X.shape[0]
        error_history = []

        for _epoch in range(epochs):                          # Step 9
            order = rng.permutation(N)
            epoch_error_sum = 0.0

            for n in order:                                    # Step 7: one epoch = N examples
                x_hat_n = X_hat[n]                              # Step 2: choose x_n
                y_n = y[n]

                a_n = self.w @ x_hat_n                          # Step 3: net input
                s_n = self._f(a_n)                              # Step 3: neuron output

                E_n = 0.5 * (y_n - s_n) ** 2                    # Step 4: instantaneous error

                delta_o = (y_n - s_n) * self._f_prime(s_n)      # local gradient delta^o
                delta_w = self.eta * delta_o * x_hat_n          # Step 5: change in weight

                self.w = self.w + delta_w                       # Step 6: update the weight

                epoch_error_sum += E_n

            E_av = epoch_error_sum / N                          # Step 8: average error
            error_history.append(E_av)
            if E_av < tol:                                      # convergence criterion
                break

        return error_history
