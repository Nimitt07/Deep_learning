"""Loads the Group-12 MNIST subset (classes 0,1,2,3,7) and flattens every
28x28 image into a 784-dimensional vector scaled to [0, 1]."""
import os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "..", "Group_12", "Group_12")
CACHE = os.path.join(HERE, "data_cache.npz")


def _load_split(split):
    split_dir = os.path.join(DATA_ROOT, split)
    classes = sorted(os.listdir(split_dir), key=int)
    X, y = [], []
    for label_idx, cls in enumerate(classes):
        cls_dir = os.path.join(split_dir, cls)
        for fname in sorted(os.listdir(cls_dir)):
            img = Image.open(os.path.join(cls_dir, fname)).convert("L")
            X.append(np.asarray(img, dtype=np.float32).reshape(784) / 255.0)
            y.append(label_idx)
    return np.stack(X), np.array(y, dtype=np.int64), [int(c) for c in classes]


def load_data():
    """Returns dict with X_train, y_train, X_val, y_val, X_test, y_test, digits."""
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return {k: d[k] for k in d.files}
    out = {}
    for split in ("train", "val", "test"):
        X, y, digits = _load_split(split)
        out[f"X_{split}"], out[f"y_{split}"] = X, y
    out["digits"] = np.array(digits)
    np.savez_compressed(CACHE, **out)
    return out


if __name__ == "__main__":
    d = load_data()
    for k, v in d.items():
        print(k, v.shape, v.dtype)
