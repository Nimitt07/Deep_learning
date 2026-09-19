"""
Fully Connected Neural Network (FCNN) trained with Backpropagation Learning in
Pattern Mode (Stochastic Gradient Descent), implemented from scratch.

Everything below follows the notation of the course slides
("Class07-10: Basics of Neural Networks / FCNN", slides 32-36) so that the code can be read
side by side with the lecture. No TensorFlow / PyTorch / scikit-learn / autograd is used;
numpy only does array arithmetic.

===========================================================================================
NOTATION (slides 32-34)
===========================================================================================
    Training data      D = {(x_n, y_n)}_{n=1}^N ,   x_n in R^d ,   y_n in R^K
    d                  number of input nodes            (index i = 1..d)
    J                  number of hidden nodes           (index j = 1..J)
    K                  number of output nodes           (index k = 1..K)

    W^(h)              weights between INPUT and HIDDEN layer,  entries w_ij^(h)
    W^(o)              weights between HIDDEN and OUTPUT layer, entries w_jk^(o)

    a_nj               net input of hidden node j     h_nj      = g(a_nj)
    a_nk^(o)           net input of output node k     yhat_nk   = f(a_nk^(o))

    g(.)               hidden-layer activation  (logistic or tan-hyperbolic)
    f(.)               output-layer activation  (logistic / tanh for classification,
                                                 linear for regression)
    eta                learning rate, 0 < eta <= 1
    delta_nk^(o)       local gradient of output node k
    delta_nj^(h)       local gradient of hidden node j
    E_n                instantaneous error,  E_av = (1/N) sum_n E_n

The bias is the weight of index 0, exactly as in the slides' sums that start at j = 0:
        a_k^(o) = sum_{j=0}^{J} w_jk^(o) h_j      with h_0 = 1 .
So every weight matrix is stored with the FROM-index as its row and the TO-index as its
column -- W_h[i][j] IS w_ij^(h) and W_o[j][k] IS w_jk^(o) -- and row 0 holds the bias.

===========================================================================================
ALGORITHM (slide 34: steps 1-7)
===========================================================================================
    1. Initialise W^(h) and W^(o) with random values.
    2. Randomly choose a training example x_n.
    3. Forward computation: compute the output of all output neurons, yhat_nk.
    4. Backward operation:
           instantaneous error   E_n = 1/2 sum_{k=1}^{K} (y_nk - yhat_nk)^2

           hidden -> output:     delta_nk^(o) = (y_nk - yhat_nk) * df(a_nk^(o))/da_nk^(o)
                                 dw_jk^(o)    = eta * delta_nk^(o) * h_nj
                                 w_jk^(o)     = w_jk^(o) + dw_jk^(o)

           input  -> hidden:     delta_nj^(h) = [ sum_{k=1}^{K} delta_nk^(o) w_jk^(o) ]
                                                * dg(a_nj)/da_nj
                                 dw_ij^(h)    = eta * delta_nj^(h) * x_i
                                 w_ij^(h)     = w_ij^(h) + dw_ij^(h)
    5. Repeat steps 2 to 4 till all the training examples are presented once  (= one EPOCH).
    6. Compute the average error   E_av = (1/N) sum_{n=1}^{N} E_n .
    7. Repeat steps 2 to 6 till the stopping criterion is satisfied.

With more than one hidden layer (Dataset 2 of each task), step 4's hidden-layer rule is
simply applied again, layer by layer, from the last hidden layer back to the first: the
"sum over k" is then taken over the nodes of the layer above.

===========================================================================================
ACTIVATION FUNCTIONS (slides 33-34)
===========================================================================================
    hidden nodes    g(a_j) = 1 / (1 + exp(-a_j))     dg/da = g(a_j) (1 - g(a_j))
                    g(a_j) = tanh(a_j)               dg/da = 1 - g(a_j)^2

    output nodes    classification:  f(a_k) = 1 / (1 + exp(-a_k))  df/da = f(1 - f)
                                     f(a_k) = tanh(a_k)            df/da = 1 - f^2
                    regression:      f(a_k) = a_k                  df/da = 1

===========================================================================================
STOPPING CRITERION and INITIALISATION (slide 35)
===========================================================================================
  * Stop when a fixed number of epochs is reached, OR when the absolute difference between
    the average errors of successive epochs falls below a threshold.
  * Weights are drawn at random with a spread set by 1/sqrt(n). The slides give the
    unit-normal form; the default here is the uniform +-2/sqrt(n) form, chosen by
    measurement -- see the initialisation section below and compare_initialisation.py.
  * Inputs are normalised (z-score) -- see Standardizer in data_prep.py.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Activation functions and their derivatives.
#
# Each derivative is written in terms of the node's own OUTPUT (as on the slides, where
# dg/da is expressed through g(a_j) and df/da through f(a_k)), because the output is what
# the forward pass has already computed.
# ---------------------------------------------------------------------------

def _logistic(a):
    # the clip only guards exp() against overflow for strongly saturated nodes; the ndarray
    # method is used instead of np.clip because this sits in the innermost SGD loop
    return 1.0 / (1.0 + np.exp(-a.clip(-500.0, 500.0)))


ACTIVATIONS = {
    # name       : (function,  derivative expressed through the output value)
    "logistic": (_logistic, lambda out: out * (1.0 - out)),
    "tanh":     (np.tanh,   lambda out: 1.0 - out * out),
    "linear":   (lambda a: a, lambda out: np.ones_like(out)),
}

# "sigmoid" is accepted as a synonym for the logistic function
ACTIVATIONS["sigmoid"] = ACTIVATIONS["logistic"]


def _augment(values):
    """Prepend the constant 1 that multiplies the bias weight (index 0 of the sums)."""
    return np.concatenate(([1.0], values))


# ---------------------------------------------------------------------------
# Step 1: weight initialisation
#
# Every scheme centres the weights on mean 0 and sets their spread from the layer size, so
# that the initial net inputs a_j = sum_i w_ij x_i stay inside the active (non-saturated)
# region of the logistic function. They differ only in the shape of the distribution and in
# how wide it is; `compare_initialisation.py` measures them side by side.
#
#   "normal"   w ~ N(0, 1) / sqrt(n_in)                 -- the rule given in the slides
#   "uniform"  w ~ U(-sigma, +sigma),   sigma = 1/sqrt(n_in)      (mean +- 1 sigma)
#   "uniform2" w ~ U(-2 sigma, +2 sigma)                          (mean +- 2 sigma)
#   "xavier"   w ~ U(-limit, +limit),   limit = sqrt(6/(n_in + n_out))
#
# "uniform2" is the DEFAULT. That is not a guess: compare_initialisation.py trains the four
# datasets with all four schemes, same architecture / learning rate / epochs, averaged over
# three seeds, and uniform2 came out best both on how fast E_av descends (mean rank 1.75 of 4)
# and on the final validation metric (mean rank 2.13 of 4). Widening the uniform range from
# +-1 sigma to +-2 sigma is what does the work: at +-1 sigma the initial net inputs are so
# small that every logistic unit starts in the near-linear part of its curve and the hidden
# units take many epochs to differentiate from one another.
#
# Run `python compare_initialisation.py` to reproduce the comparison; the other schemes stay
# available through the `init=` argument.
# ---------------------------------------------------------------------------

INITIALISERS = ("xavier", "normal", "uniform", "uniform2")


def _init_weights(rng, n_in, n_out, scheme):
    """Weights for one layer, shape (n_in + 1, n_out); row 0 is the bias."""
    shape = (n_in + 1, n_out)
    if scheme == "normal":
        return rng.standard_normal(shape) / np.sqrt(n_in)
    if scheme == "uniform":
        s = 1.0 / np.sqrt(n_in)
        return rng.uniform(-s, s, size=shape)
    if scheme == "uniform2":
        s = 2.0 / np.sqrt(n_in)
        return rng.uniform(-s, s, size=shape)
    if scheme == "xavier":
        limit = np.sqrt(6.0 / (n_in + n_out))
        return rng.uniform(-limit, limit, size=shape)
    raise ValueError(f"Unknown initialisation: {scheme!r}; expected one of {INITIALISERS}")


class FCNN:
    """A fully connected feedforward network trained by backpropagation in pattern mode.

    Parameters
    ----------
    layer_sizes : list[int]
        [d, J1, ..., JH, K] -- input nodes, hidden nodes of each hidden layer, output nodes.
        len(layer_sizes) - 2 is the number of hidden layers.
    hidden_activation : {"logistic"/"sigmoid", "tanh"}
        g(.), the activation of every hidden node.
    output_activation : {"logistic"/"sigmoid", "tanh", "linear"}
        f(.). "logistic" for classification (1-of-K coded targets), "linear" for regression.
    learning_rate : float
        eta, the proportionality constant of the weight update, 0 < eta <= 1.
    seed : int
        Seed for the random initialisation (step 1) and for the order in which the training
        examples are picked (step 2).
    init : {"xavier", "normal", "uniform", "uniform2"}
        Weight-initialisation scheme; see INITIALISERS above.

    Attributes
    ----------
    W : list of numpy arrays
        The weight matrices, ordered input-side first:
            W[0]  is W^(h)   -- input -> first hidden layer,   W[0][i][j] = w_ij^(h)
            ...
            W[-1] is W^(o)   -- last hidden layer -> output,   W[-1][j][k] = w_jk^(o)
        W[l] has shape (nodes_below + 1, nodes_above); row 0 is the bias weight.
    """

    def __init__(self, layer_sizes, hidden_activation="logistic",
                 output_activation="linear", learning_rate=0.01, seed=0,
                 init="uniform2"):
        if len(layer_sizes) < 3:
            raise ValueError("layer_sizes must contain at least one hidden layer")
        for name in (hidden_activation, output_activation):
            if name not in ACTIVATIONS:
                raise ValueError(f"Unsupported activation: {name}")

        self.layer_sizes = list(layer_sizes)
        self.n_layers = len(layer_sizes) - 1          # number of weight matrices
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.eta = learning_rate
        self.seed = seed

        # ---- Step 1: initialise W^(h) and W^(o) with random values --------------------
        self.init = init
        rng = np.random.default_rng(seed)
        self.W = [_init_weights(rng, layer_sizes[l], layer_sizes[l + 1], init)
                  for l in range(self.n_layers)]

        # g(.) for every hidden layer, then f(.) for the output layer
        self._act = [ACTIVATIONS[hidden_activation]] * (self.n_layers - 1)
        self._act.append(ACTIVATIONS[output_activation])

    # ------------------------------------------------------------------
    # Step 3 -- forward computation
    # ------------------------------------------------------------------

    def forward_all(self, X):
        """Forward pass over a whole dataset at once.

        Returns [h^(1), ..., h^(H), yhat], i.e. the output of every hidden layer followed by
        the output layer, each of shape (n_samples, nodes in that layer). Used for
        evaluation, for the decision regions, and for the hidden-node / output-node plots
        requested by the assignment.
        """
        X = np.asarray(X, dtype=float)
        outputs = []
        out = X
        for l in range(self.n_layers):
            activation = self._act[l][0]                       # g(.) or, last, f(.)
            ones = np.ones((out.shape[0], 1))                  # the constant-1 bias input
            net_input = np.hstack([ones, out]) @ self.W[l]     # a = sum_{j=0} w_jk * h_j
            out = activation(net_input)
            outputs.append(out)
        return outputs

    def predict(self, X):
        """yhat -- the output-layer values f(a^(o)), shape (n_samples, K)."""
        return self.forward_all(X)[-1]

    def hidden_outputs(self, X):
        """The h_j values of every hidden layer (the output layer is excluded)."""
        return self.forward_all(X)[:-1]

    # ------------------------------------------------------------------
    # Test phase (slide 36)
    # ------------------------------------------------------------------

    def predict_class(self, X, classes):
        """CLASSIFICATION test phase: class label for x = argmax_k yhat_k."""
        return np.asarray(classes)[np.argmax(self.predict(X), axis=1)]

    def predict_value(self, X):
        """REGRESSION test phase: yhat_k = a_k^(o) (the single linear output node)."""
        return self.predict(X).ravel()

    # ------------------------------------------------------------------
    # Step 6 -- average error
    # ------------------------------------------------------------------

    def average_error(self, X, Y):
        """E_av = (1/N) sum_n E_n , with E_n = 1/2 sum_k (y_nk - yhat_nk)^2."""
        yhat = self.predict(X)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        return float(np.mean(0.5 * np.sum((Y - yhat) ** 2, axis=1)))

    # ------------------------------------------------------------------
    # Steps 3-4 -- one pattern-mode update
    # ------------------------------------------------------------------

    def _update_on_one_example(self, x_n, y_n):
        """Present ONE training example and update every weight. Returns E_n.

        This is the heart of the assignment: steps 3 and 4 of slide 34, for a network with
        any number of hidden layers.
        """
        W, act, L = self.W, self._act, self.n_layers   # local aliases: this is the hot loop

        # ---- Step 3: forward computation -------------------------------------------
        # inputs[l] is the augmented input of layer l, i.e. [1; h^(l-1)] (with h^(0) = x_n);
        # outputs[l] is h^(l) for a hidden layer and yhat for the output layer.
        inputs, outputs = [], []
        out = x_n
        for l in range(L):
            activation = act[l][0]                     # g(.) for hidden, f(.) for output
            aug = _augment(out)                        # [1; h^(l-1)]
            out = activation(aug @ W[l])               # a = W^T [1;h] ; then g(a) or f(a)
            inputs.append(aug)
            outputs.append(out)

        yhat_n = outputs[-1]

        # ---- Step 4: instantaneous error -------------------------------------------
        error = y_n - yhat_n                                   # (y_nk - yhat_nk)
        E_n = 0.5 * float(np.dot(error, error))                # 1/2 sum_k (y_nk-yhat_nk)^2

        # ---- Step 4: local gradients (the "backward operation") --------------------
        delta = [None] * L

        # output layer:  delta_nk^(o) = (y_nk - yhat_nk) * df(a_nk^(o))/da_nk^(o)
        df = act[-1][1]
        delta[-1] = error * df(yhat_n)

        # hidden layers, from the top down:
        #   delta_nj^(h) = [ sum_k delta_nk^(o) w_jk^(o) ] * dg(a_nj)/da_nj
        # W[l + 1][1:] drops row 0, the bias row: the bias input is a constant, so no error
        # is propagated back through it.
        for l in range(L - 2, -1, -1):
            back_propagated = W[l + 1][1:] @ delta[l + 1]       # sum_k delta_k w_jk
            dg = act[l][1]
            delta[l] = back_propagated * dg(outputs[l])

        # ---- Step 4: weight change and update --------------------------------------
        #   dw_jk^(o) = eta * delta_nk^(o) * h_nj      w_jk^(o) = w_jk^(o) + dw_jk^(o)
        #   dw_ij^(h) = eta * delta_nj^(h) * x_i       w_ij^(h) = w_ij^(h) + dw_ij^(h)
        # np.outer(inputs[l], delta[l]) builds the whole matrix of products at once; its
        # [row, col] entry is (input of the node below) * (local gradient of the node above),
        # which is exactly the pairing in the two lines above.
        eta = self.eta
        for l in range(L):
            W[l] += eta * np.outer(inputs[l], delta[l])

        return E_n

    # ------------------------------------------------------------------
    # Steps 2, 5, 6, 7 -- the training loop
    # ------------------------------------------------------------------

    def train(self, X, Y, X_val=None, Y_val=None, epochs=500, threshold=1e-8, patience=20,
              verbose=False):
        """Train the network (steps 2 to 7 of slide 34).

        Stopping criterion (slide 35): training stops when a fixed number of epochs is
        reached, OR when the absolute difference between the average errors of successive
        epochs, |E_av(t) - E_av(t-1)|, falls below `threshold`. The slide's example
        threshold is 1e-3; because the weights here are updated after every single example
        the E_av curve is noisy, so the condition must hold for `patience` consecutive
        epochs before training is stopped, and a smaller threshold is used (a single flat
        epoch is noise, not convergence).

        In addition -- this part is not on the slide, it is the usual way of using the
        validation set the assignment asks for -- training also stops if the validation
        error has not improved for `patience` epochs, and the weights of the epoch with the
        lowest validation error are restored at the end (early stopping).

        Returns
        -------
        dict with
            train_error : E_av after every epoch -- the "average error vs epochs" curve
            val_error   : the same quantity on the validation data (if one is given)
            epochs_run  : number of epochs actually executed
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        N = X.shape[0]
        rng = np.random.default_rng(self.seed + 1000)

        train_curve, val_curve = [], []
        best_val_error, best_W, epochs_without_improvement = np.inf, None, 0
        previous_E_av, flat_epochs = np.inf, 0

        for epoch in range(epochs):                            # Step 7
            # Step 2: the examples are presented in a fresh random order every epoch
            order = rng.permutation(N)

            # Step 5: one epoch presents every training example exactly once
            sum_E_n = 0.0
            for n in order:
                sum_E_n += self._update_on_one_example(X[n], Y[n])

            E_av = sum_E_n / N                                 # Step 6
            train_curve.append(E_av)

            if X_val is not None:
                E_av_val = self.average_error(X_val, Y_val)
                val_curve.append(E_av_val)
                if E_av_val < best_val_error - 1e-12:
                    best_val_error = E_av_val
                    best_W = [w.copy() for w in self.W]
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                message = f"  epoch {epoch + 1:4d}/{epochs}  E_av = {E_av:.6f}"
                if X_val is not None:
                    message += f"   E_av(validation) = {val_curve[-1]:.6f}"
                print(message)

            # stopping criterion: |E_av(t) - E_av(t-1)| below the threshold, sustained
            flat_epochs = flat_epochs + 1 if abs(previous_E_av - E_av) < threshold else 0
            previous_E_av = E_av
            if flat_epochs >= patience:
                break
            # ... or the validation error has stopped improving
            if X_val is not None and epochs_without_improvement >= patience:
                break

        if best_W is not None:
            self.W = best_W

        return {
            "train_error": train_curve,
            "val_error": val_curve,
            "epochs_run": len(train_curve),
        }

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def architecture_string(self):
        """e.g. "2-8-4-3" -- d, then the hidden layer sizes, then K."""
        return "-".join(str(n) for n in self.layer_sizes)

    def n_parameters(self):
        """Total number of weights, biases included."""
        return int(sum(w.size for w in self.W))

    def save(self, path):
        """Store the trained weights and the network configuration in a .npz file."""
        np.savez(path,
                 layer_sizes=np.array(self.layer_sizes),
                 hidden_activation=self.hidden_activation,
                 output_activation=self.output_activation,
                 learning_rate=self.eta,
                 init=self.init,
                 **{f"W{l}": w for l, w in enumerate(self.W)})

    @classmethod
    def load(cls, path):
        """Rebuild a trained network saved by save()."""
        d = np.load(path, allow_pickle=False)
        net = cls([int(n) for n in d["layer_sizes"]],
                  str(d["hidden_activation"]), str(d["output_activation"]),
                  float(d["learning_rate"]),
                  init=str(d["init"]) if "init" in d else "uniform2")
        net.W = [d[f"W{l}"] for l in range(net.n_layers)]
        return net


# ---------------------------------------------------------------------------
# 1-of-K coding of the class labels (y_n in R^K, slide 32)
# ---------------------------------------------------------------------------

def one_hot(y, classes, activation="logistic"):
    """1-of-K code the integer class labels into the target vectors y_n in R^K.

    The target of the node of the correct class is 1; the others take the low value of the
    output activation -- 0 for the logistic function, -1 for tanh.
    """
    low = -1.0 if activation == "tanh" else 0.0
    index_of = {c: i for i, c in enumerate(classes)}
    Y = np.full((len(y), len(classes)), low, dtype=float)
    for n, label in enumerate(y):
        Y[n, index_of[label]] = 1.0
    return Y


def decode_one_hot(yhat, classes):
    """Test phase for classification (slide 36): class label for x = argmax_k yhat_k."""
    return np.asarray(classes)[np.argmax(yhat, axis=1)]
