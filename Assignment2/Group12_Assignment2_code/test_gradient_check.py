"""
Correctness check for the backpropagation implemented in fcnn.py.

The analytic weight change computed by one SGD step,

    Delta W[l] = eta * delta^(l) [1; s^(l-1)]^T   ==>   dE_n/dW[l] = -delta^(l) [1; s^(l-1)]^T,

is compared against a central finite-difference estimate of the same derivative,

    dE_n/dW[l]_ij  ~  ( E_n(W_ij + h) - E_n(W_ij - h) ) / (2h).

If the two agree to ~1e-8 the forward pass, the local gradients and the weight update are all
consistent. This is run as a plain script (no test framework needed):

    python test_gradient_check.py
"""

import numpy as np

from fcnn import FCNN

H = 1e-5
# The standard norm-based relative error ||g_analytic - g_numeric|| / (||g_analytic|| +
# ||g_numeric||). Comparing entry by entry is misleading: entries whose true gradient is
# ~1e-12 have no significant digits left in a finite difference, so their *relative* error is
# large even when the implementation is exact.
TOLERANCE = 1e-9


def instantaneous_error(net, x, y):
    """E_n(W) = 1/2 ||y - s^(L)||^2 for a single example."""
    s = net.predict(x.reshape(1, -1))[0]
    return 0.5 * float(np.sum((y - s) ** 2))


def analytic_gradients(net, x, y):
    """dE_n/dW[l] recovered from one SGD step: the step is W += eta * (-dE/dW)."""
    eta = net.eta
    before = [w.copy() for w in net.W]
    net._update_on_one_example(x, y)
    grads = [-(net.W[l] - before[l]) / eta for l in range(net.n_layers)]
    net.W = before                      # undo the step
    return grads


def numerical_gradients(net, x, y):
    grads = []
    for l in range(net.n_layers):
        g = np.zeros_like(net.W[l])
        for i in range(net.W[l].shape[0]):
            for j in range(net.W[l].shape[1]):
                original = net.W[l][i, j]
                net.W[l][i, j] = original + H
                e_plus = instantaneous_error(net, x, y)
                net.W[l][i, j] = original - H
                e_minus = instantaneous_error(net, x, y)
                net.W[l][i, j] = original
                g[i, j] = (e_plus - e_minus) / (2 * H)
        grads.append(g)
    return grads


def check(layer_sizes, hidden_activation, output_activation, seed):
    rng = np.random.default_rng(seed)
    net = FCNN(layer_sizes, hidden_activation, output_activation,
               learning_rate=0.01, seed=seed)
    x = rng.normal(size=layer_sizes[0])
    y = rng.normal(size=layer_sizes[-1])

    a_grads = analytic_gradients(net, x, y)
    n_grads = numerical_gradients(net, x, y)

    flat_a = np.concatenate([g.ravel() for g in a_grads])
    flat_n = np.concatenate([g.ravel() for g in n_grads])
    denom = np.linalg.norm(flat_a) + np.linalg.norm(flat_n)
    rel = float(np.linalg.norm(flat_a - flat_n) / denom) if denom > 0 else 0.0

    arch = "-".join(str(n) for n in layer_sizes)
    status = "PASS" if rel < TOLERANCE else "FAIL"
    print(f"  {status}  {arch:<14} hidden={hidden_activation:<8} "
          f"output={output_activation:<8} relative error = {rel:.3e}")
    return rel < TOLERANCE


def main():
    print("Gradient check: analytic backpropagation vs central finite differences")
    print(f"(step h = {H}, tolerance on the relative error = {TOLERANCE})\n")
    cases = [
        ([2, 4, 3], "logistic", "logistic", 0),          # classification, one hidden layer
        ([2, 5, 4, 3], "logistic", "logistic", 1),       # classification, two hidden layers
        ([1, 6, 1], "logistic", "linear", 2),           # regression, one hidden layer
        ([2, 6, 4, 1], "logistic", "linear", 3),        # regression, two hidden layers
        ([3, 5, 2], "tanh", "linear", 4),              # tanh hidden units
        ([2, 3, 3, 3, 2], "logistic", "logistic", 5),    # three hidden layers
    ]
    ok = all(check(*c) for c in cases)
    print()
    print("All gradient checks passed." if ok else "SOME GRADIENT CHECKS FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
