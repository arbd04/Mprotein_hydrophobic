import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import h5py
import numpy as np
from mprotein.models.stack_ensemble import contact_Regression

H5_PATH = "./data/maps/maps.h5"
SAVE_DIR = "./data/weights/"
SAMPLE_SIZE = 20
PENALTY_ = 0.15

X_train = []
y_train = []

with h5py.File(H5_PATH, "r") as f:
    all_keys = list(f.keys())
    if len(all_keys) > SAMPLE_SIZE:
        selected_keys = np.random.choice(all_keys, SAMPLE_SIZE, replace=False)
    else:
        selected_keys = all_keys

    for acc in selected_keys:
        amap = f[acc]["attention_map"][:, :, 1:-1, 1:-1]  # (Layer, Head, L, L)
        cmap = f[acc]["contact_map"][:]  # (L, L)

        n_layers, n_heads, L, _ = amap.shape
        amap_merge = amap.reshape(n_layers * n_heads, L, L)
        amap_flat = amap_merge.reshape(n_layers * n_heads, -1).T
        cmap_flat = cmap.flatten()

        X_train.append(amap_flat)
        y_train.append(cmap_flat)

X_final = np.vstack(X_train)
y_final = np.concatenate(y_train)

contact_Regression(X_final, y_final, PENALTY_, SAVE_DIR)
