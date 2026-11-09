"""Loading and preprocessing for the mmWave breathing-signal dataset.

Expected format: one row per sample, N signal columns followed by a final
label column (e.g. bradypnea, eupnea, kussmaul, biot). CSV or Excel.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Return (X, y_int, label_encoder). X has shape (n_samples, signal_len)."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    labels = df.iloc[:, -1].astype(str).to_numpy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    return X, y, encoder


def scale_and_reshape(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize using statistics from the training split only, then add a channel axis.

    Fitting the scaler on training data alone keeps test-set information out
    of preprocessing, so the evaluation is not optimistically biased.
    """
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)[..., np.newaxis]
    X_test = scaler.transform(X_test)[..., np.newaxis]
    return X_train.astype(np.float32), X_test.astype(np.float32)
