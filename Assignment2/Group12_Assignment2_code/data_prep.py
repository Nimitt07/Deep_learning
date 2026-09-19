"""
Data extraction and train / validation / test splitting for Assignment-2.

Assignment-2 requires a 60 / 20 / 20 train / validation / test split (Assignment-1 used a
70 / 30 train / test split), so the split is redone here and cached separately.

IMPORTANT: each dataset is split exactly once, per class (for classification) or over the
whole file (for regression), with a fixed random seed, and the resulting split is cached to
processed_data/*.npz.  Every experiment script loads the split through the loader functions
below; if the cache already exists it is reused as-is, so switching architectures or models
never reshuffles or re-splits the data.

To force a fresh split, delete the corresponding .npz file in processed_data/, or call the
prepare_* function with force=True.
"""

import os
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(THIS_DIR, "processed_data")

SEED = 42
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2
# the test fraction is the remainder (0.2)


RAW_DIR_CANDIDATES = [
    os.path.join(THIS_DIR, "Group12"),
    os.path.join(THIS_DIR, "..", "Group12"),
    os.path.join(THIS_DIR, "..", "..", "Group12"),
]


def find_raw_dir():
    """Locate the raw Group12 data folder.

    Searched, in order: <code>/Group12, <code>/../Group12 (the layout used in Assignment-1),
    <code>/../../Group12.  Keeping the search flexible means the same code runs from the
    submission zip, from the lab folder, or from Colab without editing any path.

    The lookup is deliberately lazy: it is only needed when a split has to be *built*. If
    processed_data/ already holds the cached splits, the experiments run without the raw data.
    """
    for c in RAW_DIR_CANDIDATES:
        if os.path.isdir(os.path.join(c, "Classification")):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "Could not locate the raw 'Group12' data folder (needed only when the cached split in "
        "processed_data/ is missing). Looked in: "
        + ", ".join(os.path.abspath(c) for c in RAW_DIR_CANDIDATES)
    )


# ---------------------------------------------------------------------------
# Generic 60 / 20 / 20 splitting helpers
# ---------------------------------------------------------------------------

def _split_train_val_test(X, y, seed):
    """Shuffle once with a fixed seed, then take 60% / 20% / 20% contiguous slices."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_train = int(round(n * TRAIN_FRACTION))
    n_val = int(round(n * VAL_FRACTION))
    tr = perm[:n_train]
    va = perm[n_train:n_train + n_val]
    te = perm[n_train + n_val:]
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def _split_per_class_and_stack(X, y, seed_base):
    """Split each class independently so the 60/20/20 ratio holds within every class."""
    parts = [[] for _ in range(6)]
    for i, label in enumerate(sorted(set(y.tolist()))):
        mask = y == label
        pieces = _split_train_val_test(X[mask], y[mask], seed=seed_base + i)
        for k in range(6):
            parts[k].append(pieces[k])
    return (
        np.vstack(parts[0]), np.concatenate(parts[1]),
        np.vstack(parts[2]), np.concatenate(parts[3]),
        np.vstack(parts[4]), np.concatenate(parts[5]),
    )


def _save(out_path, Xtr, ytr, Xva, yva, Xte, yte):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.savez(out_path,
             X_train=Xtr, y_train=ytr,
             X_val=Xva, y_val=yva,
             X_test=Xte, y_test=yte)


def _load(name):
    d = np.load(os.path.join(PROCESSED_DIR, name))
    return (d["X_train"], d["y_train"],
            d["X_val"], d["y_val"],
            d["X_test"], d["y_test"])


# ---------------------------------------------------------------------------
# Classification Dataset 1 - Linearly Separable (LS), 3 classes, 2-D
# ---------------------------------------------------------------------------

def prepare_ls_classification(force=False):
    out_path = os.path.join(PROCESSED_DIR, "classification_ls.npz")
    if os.path.exists(out_path) and not force:
        return
    raw_dir = find_raw_dir()
    X_parts, y_parts = [], []
    for label, fname in enumerate(["Class1.txt", "Class2.txt", "Class3.txt"], start=1):
        data = np.loadtxt(os.path.join(raw_dir, "Classification", "LS_Group12", fname))
        X_parts.append(data)
        y_parts.append(np.full(data.shape[0], label))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    _save(out_path, *_split_per_class_and_stack(X, y, seed_base=SEED))


def load_ls_classification():
    prepare_ls_classification()
    return _load("classification_ls.npz")


# ---------------------------------------------------------------------------
# Classification Dataset 2 - Nonlinearly Separable (NLS), 3 classes, 2-D
# ---------------------------------------------------------------------------

def prepare_nls_classification(force=False):
    out_path = os.path.join(PROCESSED_DIR, "classification_nls.npz")
    if os.path.exists(out_path) and not force:
        return
    path = os.path.join(find_raw_dir(), "Classification", "NLS_Group12.txt")
    data = np.loadtxt(path, skiprows=1)   # first line is a text header describing the ordering
    # Header states: first 500 rows = class1, next 500 = class2, last 500 = class3.
    counts = [500, 500, 500]
    labels = np.concatenate([np.full(c, i + 1) for i, c in enumerate(counts)])
    assert labels.shape[0] == data.shape[0], "Unexpected NLS row count vs documented class sizes"
    _save(out_path, *_split_per_class_and_stack(data, labels, seed_base=SEED + 100))


def load_nls_classification():
    prepare_nls_classification()
    return _load("classification_nls.npz")


# ---------------------------------------------------------------------------
# Regression Dataset 1 - Univariate (1-D input)
# ---------------------------------------------------------------------------

def prepare_univariate_regression(force=False):
    out_path = os.path.join(PROCESSED_DIR, "regression_univariate.npz")
    if os.path.exists(out_path) and not force:
        return
    data = np.loadtxt(os.path.join(find_raw_dir(), "Regression", "UnivariateData", "12.csv"),
                      delimiter=",")
    X, y = data[:, :1], data[:, 1]
    _save(out_path, *_split_train_val_test(X, y, seed=SEED + 200))


def load_univariate_regression():
    prepare_univariate_regression()
    return _load("regression_univariate.npz")


# ---------------------------------------------------------------------------
# Regression Dataset 2 - Bivariate (2-D input)
# ---------------------------------------------------------------------------

def prepare_bivariate_regression(force=False):
    out_path = os.path.join(PROCESSED_DIR, "regression_bivariate.npz")
    if os.path.exists(out_path) and not force:
        return
    data = np.loadtxt(os.path.join(find_raw_dir(), "Regression", "BivariateData", "12.csv"),
                      delimiter=",")
    X, y = data[:, :2], data[:, 2]
    _save(out_path, *_split_train_val_test(X, y, seed=SEED + 300))


def load_bivariate_regression():
    prepare_bivariate_regression()
    return _load("regression_bivariate.npz")


# ---------------------------------------------------------------------------
# Standardisation (statistics estimated on TRAINING data only)
# ---------------------------------------------------------------------------

class Standardizer:
    """z-score scaling: z = (v - mean) / std, with mean/std taken from the training split only.

    Sigmoid / tanh hidden units saturate when the raw inputs are large (the LS inputs live
    around x1 ~ 20, and the bivariate targets reach ~ 90), which stalls SGD. Standardising the
    inputs -- and, for regression, the targets -- keeps the units in their active range.
    Every reported RMSE / %RMSE is computed after inverse-transforming the predictions back to
    the original units, so the numbers remain directly comparable with Assignment-1.
    """

    def __init__(self, values):
        values = np.asarray(values, dtype=float)
        self.mean = values.mean(axis=0)
        self.std = values.std(axis=0)
        self.std = np.where(self.std == 0, 1.0, self.std)

    def transform(self, values):
        return (np.asarray(values, dtype=float) - self.mean) / self.std

    def inverse(self, values):
        return np.asarray(values, dtype=float) * self.std + self.mean


if __name__ == "__main__":
    loaders = [
        ("LS classification", load_ls_classification),
        ("NLS classification", load_nls_classification),
        ("Univariate regression", load_univariate_regression),
        ("Bivariate regression", load_bivariate_regression),
    ]
    print("Raw data directory: " + find_raw_dir())
    print()
    for name, loader in loaders:
        Xtr, ytr, Xva, yva, Xte, yte = loader()
        print(f"{name}:")
        print(f"    train = {Xtr.shape}, val = {Xva.shape}, test = {Xte.shape}")
        if ytr.dtype.kind in "iu":
            for c in sorted(set(ytr.tolist())):
                print(f"      class {c}: train={int((ytr == c).sum())} "
                      f"val={int((yva == c).sum())} test={int((yte == c).sum())}")
