"""Trains every architecture with every optimizer and stores the results.

    python train.py                       # all architectures x all optimizers
    python train.py --archs A1 --opts SGD Adam --workers 4

Fairness guarantees
  * one initial weight vector per architecture, shared by all optimizers
    (the epoch-0 loss is therefore identical for every optimizer);
  * batch_size = 1 optimizers visit the samples in the same shuffled order
    (the permutation depends only on the epoch number);
  * same loss (mean cross-entropy over the full training set, evaluated after
    every epoch), same learning rate and same stopping rule for everyone:
        stop at epoch t when |E(t) - E(t-1)| < 1e-4
    (--max-epochs is only a safety cap; it is set far above what is needed).
"""
import os

# one BLAS thread per worker process -- the runs themselves are parallelised
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

import fcnn
from data_loader import load_data

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# hidden layers between the 784-d input and the 5-way softmax output
ARCHITECTURES = {
    # wide networks
    "A1": [128, 64, 32],                 # 3 hidden layers
    "A2": [256, 128, 64, 32],            # 4 hidden layers
    "A3": [128, 96, 64, 48, 32],         # 5 hidden layers
    # narrow networks (same depths, fewer nodes per layer)
    "A4": [64, 32, 16],                  # 3 hidden layers
    "A5": [64, 64, 32, 16],              # 4 hidden layers
    "A6": [64, 48, 32, 24, 16],          # 5 hidden layers
}
INIT_SEED = {"A1": 101, "A2": 202, "A3": 303, "A4": 404, "A5": 505, "A6": 606}
SHUFFLE_SEED = 12345
N_CLASSES = 5

LEARNING_RATE = 1e-3
TOLERANCE = 1e-4
MAX_EPOCHS = 10000


def layer_sizes(arch):
    return [784] + ARCHITECTURES[arch] + [N_CLASSES]


def run(arch, opt_name, max_epochs=MAX_EPOCHS, tol=TOLERANCE, out_dir=RESULTS):
    fcnn.enable_flush_denormal()
    d = load_data()
    X, y, Xv, yv = d["X_train"], d["y_train"], d["X_val"], d["y_val"]
    Y = np.eye(N_CLASSES, dtype=fcnn.DTYPE)[y]
    N = len(y)

    layout = fcnn.Layout(layer_sizes(arch))
    theta = fcnn.init_params(layout, INIT_SEED[arch])   # identical for all optimizers
    init_checksum = float(np.abs(theta).astype(np.float64).sum())
    update, nesterov, batch_size = fcnn.OPTIMIZERS[opt_name]
    hp = fcnn.make_hparams(lr=LEARNING_RATE)
    s1, s2 = np.zeros_like(theta), np.zeros_like(theta)
    look, grad = np.zeros_like(theta), np.zeros_like(theta)
    acts = np.zeros(layout.aoff[-1], dtype=fcnn.DTYPE)
    deltas = np.zeros_like(acts)

    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}

    def record():
        tl, ta = fcnn.evaluate(layout, theta, X, y)
        vl, va = fcnn.evaluate(layout, theta, Xv, yv)
        for k, v in zip(hist, (tl, ta, vl, va)):
            hist[k].append(v)

    record()                                            # epoch 0 = before any update
    log_path = os.path.join(out_dir, f"{arch}_{opt_name}.log")
    log = open(log_path, "w")
    log.write(f"{arch} {opt_name} params={layout.n_params} init |w| sum={init_checksum:.6f}\n"
              f"epoch 0 loss {hist['train_loss'][0]:.6f}\n")
    log.flush()

    t, converged, start = 0, False, time.time()
    for epoch in range(1, max_epochs + 1):
        if batch_size == 1:
            order = np.random.default_rng(SHUFFLE_SEED + epoch).permutation(N)
            t = fcnn.stochastic_epoch(update, nesterov, theta, s1, s2, look, X, y, order,
                                      layout.sizes, layout.woff, layout.boff, layout.aoff,
                                      acts, deltas, hp, t)
        else:
            t += 1
            fcnn.full_batch_gradient(layout, theta, X, Y, grad)
            fcnn.full_batch_step(update, theta, grad, s1, s2, hp, t)
        record()
        diff = abs(hist["train_loss"][-1] - hist["train_loss"][-2])
        if epoch <= 20 or epoch % 50 == 0 or diff < tol:
            log.write(f"epoch {epoch} loss {hist['train_loss'][-1]:.6f} diff {diff:.2e} "
                      f"train_acc {hist['train_acc'][-1]:.4f} val_acc {hist['val_acc'][-1]:.4f} "
                      f"elapsed {time.time() - start:.0f}s\n")
            log.flush()
        if not np.isfinite(hist["train_loss"][-1]):
            break
        if diff < tol:
            converged = True
            break

    elapsed = time.time() - start
    summary = {
        "arch": arch, "optimizer": opt_name, "layers": layer_sizes(arch),
        "n_params": int(layout.n_params), "batch_size": batch_size or N,
        "epochs": len(hist["train_loss"]) - 1, "converged": converged,
        "initial_loss": hist["train_loss"][0], "final_loss": hist["train_loss"][-1],
        "train_acc": hist["train_acc"][-1], "val_acc": hist["val_acc"][-1],
        "init_checksum": init_checksum, "seconds": elapsed,
    }
    np.savez(os.path.join(out_dir, f"{arch}_{opt_name}.npz"), theta=theta,
             **{k: np.array(v) for k, v in hist.items()})
    with open(os.path.join(out_dir, f"{arch}_{opt_name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.write(f"DONE {json.dumps(summary)}\n")
    log.close()
    return summary


def _run_task(args):
    return run(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(ARCHITECTURES))
    ap.add_argument("--opts", nargs="+", default=list(fcnn.OPTIMIZERS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--tol", type=float, default=TOLERANCE)
    ap.add_argument("--out", default=RESULTS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    load_data()                                    # build the cache once
    tasks = [(arch, opt, a.max_epochs, a.tol, a.out) for arch in a.archs for opt in a.opts]
    with Pool(min(a.workers, len(tasks))) as pool:
        for s in pool.imap_unordered(_run_task, tasks):
            print(f"{s['arch']:>3} {s['optimizer']:>9}: epochs={s['epochs']:5d} "
                  f"converged={s['converged']} loss {s['initial_loss']:.4f}->{s['final_loss']:.4f} "
                  f"train={s['train_acc']:.4f} val={s['val_acc']:.4f} ({s['seconds']:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
