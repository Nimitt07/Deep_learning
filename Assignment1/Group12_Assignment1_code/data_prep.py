"""
Data extraction and train/test splitting.

IMPORTANT: each dataset is split into train (70%) and test (30%) exactly once, per class,
with a fixed random seed, and the resulting split is cached to processed_data/*.npz.
Every model-training script (classification_ls.py, classification_nls.py,
regression_univariate.py, regression_bivariate.py) loads the split via the loader
functions below -- if the cache already exists it is reused as-is, so switching models
never reshuffles or re-splits the data.

To force a fresh split (e.g. if you intentionally want to change the split policy),
delete the corresponding .npz file in processed_data/, or call the prepare_* function
with force=True.
"""

import os
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(THIS_DIR, "..", "Group12")
PROCESSED_DIR = os.path.join(THIS_DIR, "processed_data")

SEED = 42
TRAIN_FRACTION = 0.7


def _split_train_test(X, y, train_frac=TRAIN_FRACTION, seed=SEED):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_train = int(round(n * train_frac))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def _split_per_class_and_stack(X, y, seed_base):
    """Split each class independently (so the 70/30 ratio holds per-class), then stack."""
    Xtr_list, ytr_list, Xte_list, yte_list = [], [], [], []
    for i, label in enumerate(sorted(set(y.tolist()))):
        mask = y == label
        Xtr, ytr, Xte, yte = _split_train_test(X[mask], y[mask], seed=seed_base + i)
        Xtr_list.append(Xtr)
        ytr_list.append(ytr)
        Xte_list.append(Xte)
        yte_list.append(yte)
    return (
        np.vstack(Xtr_list),
        np.concatenate(ytr_list),
        np.vstack(Xte_list),
        np.concatenate(yte_list),
    )


# ---------------------------------------------------------------------------
# Classification: Dataset 1 - Linearly Separable (LS)
# ---------------------------------------------------------------------------

def prepare_ls_classification(force=False):
    out_path = os.path.join(PROCESSED_DIR, "classification_ls.npz")
    if os.path.exists(out_path) and not force:
        return
    class_files = ["Class1.txt", "Class2.txt", "Class3.txt"]
    X_parts, y_parts = [], []
    for label, fname in enumerate(class_files, start=1):
        path = os.path.join(RAW_DIR, "Classification", "LS_Group12", fname)
        data = np.loadtxt(path)
        X_parts.append(data)
        y_parts.append(np.full(data.shape[0], label))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    Xtr, ytr, Xte, yte = _split_per_class_and_stack(X, y, seed_base=SEED)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.savez(out_path, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)


def load_ls_classification():
    prepare_ls_classification()
    d = np.load(os.path.join(PROCESSED_DIR, "classification_ls.npz"))
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"]


# ---------------------------------------------------------------------------
# Classification: Dataset 2 - Nonlinearly Separable (NLS)
# ---------------------------------------------------------------------------

def prepare_nls_classification(force=False):
    out_path = os.path.join(PROCESSED_DIR, "classification_nls.npz")
    if os.path.exists(out_path) and not force:
        return
    path = os.path.join(RAW_DIR, "Classification", "NLS_Group12.txt")
    data = np.loadtxt(path, skiprows=1)  # first line is a text header, not data
    # First 500 rows = class1, next 500 = class2, last 500 = class3 (per dataset description)
    counts = [500, 500, 500]
    labels = np.concatenate([np.full(c, i + 1) for i, c in enumerate(counts)])
    assert labels.shape[0] == data.shape[0], "Unexpected NLS row count vs documented class sizes"
    Xtr, ytr, Xte, yte = _split_per_class_and_stack(data, labels, seed_base=SEED + 100)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.savez(out_path, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)


def load_nls_classification():
    prepare_nls_classification()
    d = np.load(os.path.join(PROCESSED_DIR, "classification_nls.npz"))
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"]


# ---------------------------------------------------------------------------
# Regression: Dataset 1 - Univariate
# ---------------------------------------------------------------------------

def prepare_univariate_regression(force=False):
    out_path = os.path.join(PROCESSED_DIR, "regression_univariate.npz")
    if os.path.exists(out_path) and not force:
        return
    path = os.path.join(RAW_DIR, "Regression", "UnivariateData", "12.csv")
    data = np.loadtxt(path, delimiter=",")
    X = data[:, :1]
    y = data[:, 1]
    Xtr, ytr, Xte, yte = _split_train_test(X, y, seed=SEED + 200)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.savez(out_path, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)


def load_univariate_regression():
    prepare_univariate_regression()
    d = np.load(os.path.join(PROCESSED_DIR, "regression_univariate.npz"))
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"]


# ---------------------------------------------------------------------------
# Regression: Dataset 2 - Bivariate
# ---------------------------------------------------------------------------

def prepare_bivariate_regression(force=False):
    out_path = os.path.join(PROCESSED_DIR, "regression_bivariate.npz")
    if os.path.exists(out_path) and not force:
        return
    path = os.path.join(RAW_DIR, "Regression", "BivariateData", "12.csv")
    data = np.loadtxt(path, delimiter=",")
    X = data[:, :2]
    y = data[:, 2]
    Xtr, ytr, Xte, yte = _split_train_test(X, y, seed=SEED + 300)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.savez(out_path, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)


def load_bivariate_regression():
    prepare_bivariate_regression()
    d = np.load(os.path.join(PROCESSED_DIR, "regression_bivariate.npz"))
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"]


if __name__ == "__main__":
    prepare_ls_classification()
    prepare_nls_classification()
    prepare_univariate_regression()
    prepare_bivariate_regression()
    for name, loader in [
        ("LS classification", load_ls_classification),
        ("NLS classification", load_nls_classification),
        ("Univariate regression", load_univariate_regression),
        ("Bivariate regression", load_bivariate_regression),
    ]:
        Xtr, ytr, Xte, yte = loader()
        print(f"{name}: X_train={Xtr.shape} y_train={ytr.shape} X_test={Xte.shape} y_test={yte.shape}")
