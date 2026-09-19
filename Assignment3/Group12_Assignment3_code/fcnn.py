"""Fully connected neural network, back-propagation and seven optimizers written
from scratch (NumPy + Numba JIT, no deep-learning library).

Why not a library?  For a fair comparison every optimizer must see exactly the
same initial weights, the same sample order, the same loss definition and the
same stopping rule, and the update rules must be the textbook ones (NAG in its
look-ahead form, AdaGrad/RMSProp/Adam with zero-initialised accumulators and
eps INSIDE the square root, Adam with bias correction), exactly as written in
the course slides (Class 11-14, Optimization methods for BP).  Library optimizers deviate
in small ways (e.g. PyTorch re-parameterises NAG and exposes an initial
accumulator for Adagrad), so everything is implemented by hand here.

Parameter layout: all weights and biases live in ONE flat float32 vector
theta = [W1 (row-major), b1, W2, b2, ...].  Optimizer state vectors (s1, s2) use
the same layout, so every update is a single fused loop over the parameters.
The SAME per-parameter update functions are used for batch_size = 1 and for
batch_size = N, so the only difference between optimizers is the update rule.
"""
import sys
import ctypes
import numpy as np
from numba import njit

DTYPE = np.float32

# hyper-parameter vector layout (float32 array passed to the kernels)
LR, GAMMA, BETA, BETA1, BETA2, EPS = range(6)


def make_hparams(lr=1e-3, gamma=0.9, beta=0.99, beta1=0.9, beta2=0.999, eps=1e-8):
    return np.array([lr, gamma, beta, beta1, beta2, eps], dtype=DTYPE)


def enable_flush_denormal():
    """Round sub-normal floats (< 1.2e-38) to zero, like torch.set_flush_denormal.
    Momentum / Adam accumulators decay geometrically towards zero; without this
    the CPU falls into a very slow micro-code path for denormal numbers.  The
    effect on the results is nil (such values are far below float32 precision)."""
    if sys.platform == "win32":
        ctypes.cdll.ucrtbase._controlfp(0x01000000, 0x03000000)  # _DN_FLUSH, _MCW_DN


# ----------------------------------------------------------------------------
# Per-parameter update rules  (k = parameter index, g = its gradient)
#   c1 = 1/(1-beta1^t), c2 = 1/(1-beta2^t)  -- only used by Adam
# ----------------------------------------------------------------------------
@njit(inline="always", error_model="numpy")
def update_gd(theta, s1, s2, k, g, hp, c1, c2):
    """(Stochastic / batch) gradient descent:  w <- w - lr*g"""
    theta[k] -= hp[LR] * g


@njit(inline="always", error_model="numpy")
def update_momentum(theta, s1, s2, k, g, hp, c1, c2):
    """Generalised delta rule / NAG (slides 17, 25):
         dw(m) = alpha*dw(m-1) - lr*g ;  w(m+1) = w(m) + dw(m)
    stored as v = -dw, i.e.  v <- alpha*v + lr*g ;  w <- w - v
    (for NAG g is evaluated at the look-ahead point w + alpha*dw(m-1) = w - alpha*v)"""
    v = hp[GAMMA] * s1[k] + hp[LR] * g
    s1[k] = v
    theta[k] -= v


@njit(inline="always", error_model="numpy")
def update_adagrad(theta, s1, s2, k, g, hp, c1, c2):
    """AdaGrad (slide 37):  v(m) = v(m-1) + g^2 ;  w(m+1) = w(m) - lr/sqrt(v(m)+eps) * g"""
    G = s1[k] + g * g
    s1[k] = G
    theta[k] -= hp[LR] * g / np.sqrt(G + hp[EPS])


@njit(inline="always", error_model="numpy")
def update_rmsprop(theta, s1, s2, k, g, hp, c1, c2):
    """RMSProp (slide 45):  v(m) = beta*v(m-1) + (1-beta)*g^2 ;
    w(m+1) = w(m) - lr/sqrt(v(m)+eps) * g"""
    E = hp[BETA] * s1[k] + (1.0 - hp[BETA]) * g * g
    s1[k] = E
    theta[k] -= hp[LR] * g / np.sqrt(E + hp[EPS])


@njit(inline="always", error_model="numpy")
def update_adam(theta, s1, s2, k, g, hp, c1, c2):
    """Adam (slide 53):  u(m) = b1*u(m-1) + (1-b1)*g ;  v(m) = b2*v(m-1) + (1-b2)*g^2 ;
    u_hat = u/(1-b1^m) ;  v_hat = v/(1-b2^m) ;  w(m+1) = w(m) - lr/sqrt(v_hat+eps) * u_hat"""
    m = hp[BETA1] * s1[k] + (1.0 - hp[BETA1]) * g
    v = hp[BETA2] * s2[k] + (1.0 - hp[BETA2]) * g * g
    s1[k] = m
    s2[k] = v
    theta[k] -= hp[LR] * (m * c1) / np.sqrt(v * c2 + hp[EPS])


# name -> (update rule, look-ahead gradient?, batch size; None = whole training set)
OPTIMIZERS = {
    "SGD":      (update_gd,       False, 1),
    "BatchGD":  (update_gd,       False, None),
    "Momentum": (update_momentum, False, 1),
    "NAG":      (update_momentum, True,  1),
    "AdaGrad":  (update_adagrad,  False, None),
    "RMSProp":  (update_rmsprop,  False, None),
    "Adam":     (update_adam,     False, 1),
}


# ----------------------------------------------------------------------------
# Network layout and initialisation
# ----------------------------------------------------------------------------
class Layout:
    """Offsets of every weight matrix / bias vector inside the flat vector."""

    def __init__(self, layer_sizes):
        self.sizes = np.array(layer_sizes, dtype=np.int64)
        n_layers = len(layer_sizes) - 1
        self.woff = np.zeros(n_layers, dtype=np.int64)
        self.boff = np.zeros(n_layers, dtype=np.int64)
        off = 0
        for l in range(n_layers):
            self.woff[l] = off
            off += layer_sizes[l] * layer_sizes[l + 1]
            self.boff[l] = off
            off += layer_sizes[l + 1]
        self.n_params = off
        # offsets of each layer's activations (layer 0 = input) in a flat buffer
        self.aoff = np.concatenate([[0], np.cumsum(layer_sizes)]).astype(np.int64)

    def views(self, theta):
        """List of (W, b) NumPy views into a flat vector."""
        out = []
        for l in range(len(self.sizes) - 1):
            n_in, n_out = self.sizes[l], self.sizes[l + 1]
            W = theta[self.woff[l]:self.woff[l] + n_in * n_out].reshape(n_out, n_in)
            b = theta[self.boff[l]:self.boff[l] + n_out]
            out.append((W, b))
        return out


def init_params(layout, seed):
    """He (Kaiming) normal initialisation (suited to ReLU), zero biases."""
    rng = np.random.default_rng(seed)
    theta = np.zeros(layout.n_params, dtype=DTYPE)
    for W, b in layout.views(theta):
        W[:] = rng.standard_normal(W.shape) * np.sqrt(2.0 / W.shape[1])
    return theta


# ----------------------------------------------------------------------------
# Whole-data-set forward pass, loss, accuracy and gradient (NumPy)
# ----------------------------------------------------------------------------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(layout, theta, X):
    """Softmax class probabilities for a batch X of shape (N, 784)."""
    layers = layout.views(theta)
    h = X
    for l, (W, b) in enumerate(layers):
        h = h @ W.T + b
        if l < len(layers) - 1:
            h = np.maximum(h, 0)          # ReLU in hidden layers
    return softmax(h)


def evaluate(layout, theta, X, y, chunk=4096):
    """Average cross-entropy loss (float64) and accuracy over a whole data set."""
    loss_sum, correct = 0.0, 0
    for s in range(0, len(y), chunk):
        p = forward(layout, theta, X[s:s + chunk])
        yt = y[s:s + chunk]
        pt = p[np.arange(len(yt)), yt].astype(np.float64)
        loss_sum += -np.log(np.clip(pt, 1e-12, None)).sum()
        correct += int((p.argmax(1) == yt).sum())
    return loss_sum / len(y), correct / len(y)


def predict(layout, theta, X):
    return forward(layout, theta, X).argmax(1)


def full_batch_gradient(layout, theta, X, Y_onehot, grad):
    """Gradient of the mean cross-entropy over the whole batch, written into the
    flat vector `grad` (used by BatchGD / AdaGrad / RMSProp)."""
    layers = layout.views(theta)
    gviews = layout.views(grad)
    acts = [X]
    h = X
    for l, (W, b) in enumerate(layers):
        h = h @ W.T + b
        if l < len(layers) - 1:
            h = np.maximum(h, 0)
        acts.append(h)
    delta = (softmax(h) - Y_onehot) / X.shape[0]     # dL/dz of the output layer
    for l in range(len(layers) - 1, -1, -1):
        gW, gb = gviews[l]
        np.matmul(delta.T, acts[l], out=gW)
        gb[:] = delta.sum(axis=0)
        if l > 0:
            delta = (delta @ layers[l][0]) * (acts[l] > 0)


@njit(error_model="numpy")
def full_batch_step(update, theta, grad, s1, s2, hp, t):
    """Applies one optimizer step to every parameter from a full-batch gradient."""
    c1 = np.float32(1.0 / (1.0 - hp[BETA1] ** t))
    c2 = np.float32(1.0 / (1.0 - hp[BETA2] ** t))
    for k in range(theta.shape[0]):
        update(theta, s1, s2, k, grad[k], hp, c1, c2)


# ----------------------------------------------------------------------------
# batch_size = 1 : per-sample forward/backward and update (Numba)
# ----------------------------------------------------------------------------
@njit(error_model="numpy")
def forward_backward_single(w, sizes, woff, boff, aoff, x, label, acts, deltas):
    """Forward + backward pass for ONE sample with parameters w.  Fills the
    activations and error signals dL/dz of every layer (flat buffers)."""
    n_layers = sizes.shape[0] - 1
    for j in range(sizes[0]):
        acts[j] = x[j]
    for l in range(n_layers):
        n_in, n_out = sizes[l], sizes[l + 1]
        a_in, a_out = aoff[l], aoff[l + 1]
        for i in range(n_out):
            z = w[boff[l] + i]
            base = woff[l] + i * n_in
            for j in range(n_in):
                z += w[base + j] * acts[a_in + j]
            if l < n_layers - 1 and z < 0.0:
                z = 0.0                                   # ReLU
            acts[a_out + i] = z
    # softmax + cross-entropy  ->  delta = p - onehot(label)
    a_last, n_cls = aoff[n_layers], sizes[n_layers]
    zmax = acts[a_last]
    for i in range(n_cls):
        zmax = max(zmax, acts[a_last + i])
    s = 0.0
    for i in range(n_cls):
        s += np.exp(acts[a_last + i] - zmax)
    for i in range(n_cls):
        deltas[a_last + i] = np.exp(acts[a_last + i] - zmax) / s
    deltas[a_last + label] -= 1.0
    # back-propagate through the hidden layers (ReLU derivative = 1[a > 0])
    for l in range(n_layers - 1, 0, -1):
        n_in, n_out = sizes[l], sizes[l + 1]
        a_in, a_out = aoff[l], aoff[l + 1]
        for j in range(n_in):
            if acts[a_in + j] > 0.0:
                s = 0.0
                for i in range(n_out):
                    s += w[woff[l] + i * n_in + j] * deltas[a_out + i]
                deltas[a_in + j] = s
            else:
                deltas[a_in + j] = 0.0


@njit(error_model="numpy")
def stochastic_epoch(update, nesterov, theta, s1, s2, look, X, y, order,
                     sizes, woff, boff, aoff, acts, deltas, hp, t0):
    """One epoch with batch_size = 1 over the samples in `order`.
    Returns the global step counter t (Adam bias correction)."""
    n_layers = sizes.shape[0] - 1
    t = t0
    for idx in range(order.shape[0]):
        n = order[idx]
        t += 1
        c1 = np.float32(1.0 / (1.0 - hp[BETA1] ** t))
        c2 = np.float32(1.0 / (1.0 - hp[BETA2] ** t))
        if nesterov:                              # gradient at w - gamma*v
            for k in range(theta.shape[0]):
                look[k] = theta[k] - hp[GAMMA] * s1[k]
            forward_backward_single(look, sizes, woff, boff, aoff, X[n], y[n], acts, deltas)
        else:
            forward_backward_single(theta, sizes, woff, boff, aoff, X[n], y[n], acts, deltas)
        # dL/dW_ij = delta_i * a_j   and   dL/db_i = delta_i
        for l in range(n_layers):
            n_in, n_out = sizes[l], sizes[l + 1]
            a_in, a_out = aoff[l], aoff[l + 1]
            for i in range(n_out):
                d = deltas[a_out + i]
                base = woff[l] + i * n_in
                for j in range(n_in):
                    update(theta, s1, s2, base + j, d * acts[a_in + j], hp, c1, c2)
                update(theta, s1, s2, boff[l] + i, d, hp, c1, c2)
    return t
