"""
Compares the weight-initialisation schemes of fcnn.py on all four datasets.

Step 1 of the algorithm only says "initialise the weights with random values"; the slides add
that they should be drawn from the unit normal distribution and scaled by 1/sqrt(n). That
leaves the shape and the width of the distribution open, and the choice visibly changes how
fast SGD converges. This script measures it instead of guessing:

    python compare_initialisation.py

Every scheme is given the same architecture, the same learning rate, the same epoch budget
and the same seeds, so the only difference is the starting weights. Results are averaged over
several seeds. What is reported is the average error E_av at a few checkpoints (how quickly
the run descends) and the final metric. This decides the `init=` default of the experiments.

The comparison deliberately uses a SHORT epoch budget with early stopping disabled: the
question is which scheme gets down fastest, not which reaches the best final error given
unlimited epochs.
"""

import time

import metrics as M
from data_prep import (Standardizer, load_ls_classification, load_nls_classification,
                       load_univariate_regression, load_bivariate_regression)
from fcnn import FCNN, INITIALISERS, one_hot

EPOCHS = 50            # short on purpose: this measures speed of convergence
CHECKPOINTS = (10, 25, 50)
SEEDS = (7, 21, 42)    # averaged over several seeds: one seed is far too noisy to decide on

# one representative architecture per dataset, matching the experiment scripts
CASES = [
    ("LS classification",    "classification", load_ls_classification,    [16],     0.05),
    ("NLS classification",   "classification", load_nls_classification,   [64, 32], 0.05),
    ("Univariate regression", "regression",    load_univariate_regression, [8],     0.01),
    ("Bivariate regression",  "regression",    load_bivariate_regression,  [64, 32], 0.01),
]


def run_case(name, kind, loader, hidden, eta):
    X_train, y_train, X_val, y_val, _, _ = loader()
    sx = Standardizer(X_train)
    Xtr, Xva = sx.transform(X_train), sx.transform(X_val)

    if kind == "classification":
        classes = sorted(set(y_train.tolist()))
        Ytr, Yva = one_hot(y_train, classes), one_hot(y_val, classes)
        layer_sizes = [X_train.shape[1]] + hidden + [len(classes)]
        out_act, metric_name = "logistic", "val accuracy %"
    else:
        sy = Standardizer(y_train)
        Ytr, Yva = sy.transform(y_train).reshape(-1, 1), sy.transform(y_val).reshape(-1, 1)
        layer_sizes = [X_train.shape[1]] + hidden + [1]
        out_act, metric_name = "linear", "val RMSE"

    arch = "-".join(str(n) for n in layer_sizes)
    print(f"\n{name}   architecture {arch}   eta = {eta}   {EPOCHS} epochs, no early "
          f"stopping\n  averaged over seeds {SEEDS}  (spread = max - min of the final metric)")
    header = ("  " + "scheme".ljust(10)
              + "".join(f"E_av@{e:<7}" for e in CHECKPOINTS)
              + metric_name.ljust(11) + "spread".ljust(11) + "time")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for scheme in INITIALISERS:
        curves, scores = [], []
        t0 = time.time()
        for seed in SEEDS:
            net = FCNN(layer_sizes, "logistic", out_act, eta, seed=seed, init=scheme)
            # threshold=0 and a huge patience disable both stopping rules, so every scheme
            # is given exactly the same number of weight updates
            history = net.train(Xtr, Ytr, Xva, Yva, epochs=EPOCHS, threshold=0.0,
                                patience=10 ** 9)
            curves.append(history["train_error"])
            if kind == "classification":
                pred = net.predict_class(Xva, classes)
                scores.append(M.accuracy(y_val.tolist(), pred.tolist()) * 100)
            else:
                pred = sy.inverse(net.predict_value(Xva))
                scores.append(M.rmse(y_val.tolist(), pred.tolist()))
        elapsed = time.time() - t0

        better = max if kind == "classification" else min
        mean_curve = [sum(c[e - 1] for c in curves) / len(curves) for e in CHECKPOINTS]
        mean_score = sum(scores) / len(scores)
        spread = max(scores) - min(scores)

        rows.append((scheme, mean_curve, mean_score))
        print("  " + scheme.ljust(10)
              + "".join(f"{v:<12.5f}" for v in mean_curve)
              + f"{mean_score:<11.4f}{spread:<11.4f}{elapsed:.1f}s")

    best_final = better(rows, key=lambda r: r[2])[0]
    # "fastest" = lowest average error at the first checkpoint
    fastest = min(rows, key=lambda r: r[1][0])[0]
    print(f"  -> fastest early descent: {fastest};   best {metric_name}: {best_final}")
    return fastest, best_final


def main():
    print("Weight-initialisation comparison (same architecture, eta, epochs and seeds;")
    print("only the starting weights differ).")
    votes_fast, votes_best = {}, {}
    for case in CASES:
        fastest, best = run_case(*case)
        votes_fast[fastest] = votes_fast.get(fastest, 0) + 1
        votes_best[best] = votes_best.get(best, 0) + 1

    print("\n" + "=" * 78)
    print("Fastest early descent, counted over the four datasets: "
          + ", ".join(f"{k} x{v}" for k, v in sorted(votes_fast.items(),
                                                     key=lambda kv: -kv[1])))
    print("Best final validation metric:                          "
          + ", ".join(f"{k} x{v}" for k, v in sorted(votes_best.items(),
                                                     key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
