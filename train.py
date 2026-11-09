"""Train and evaluate the mmWave breathing-pattern 1D CNN.

Examples:
    # Repeated stratified 5-fold CV of the final model (recommended: small dataset)
    python train.py --data data/breath_detection_dataset.xlsx

    # Compare all five configurations
    python train.py --data data/breath_detection_dataset.xlsx --config all

    # Single 70/30 split, as in the original experiments
    python train.py --data data/breath_detection_dataset.xlsx --mode split
"""

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from tensorflow import keras

from mmwave_cnn import CONFIGS, CNNConfig, build_model, load_dataset, scale_and_reshape


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def fit_and_predict(cfg: CNNConfig, X_tr, y_tr, X_te, n_classes: int, seed: int) -> np.ndarray:
    """Train on (X_tr, y_tr) and return predicted class ids for X_te."""
    set_seed(seed)
    X_tr, X_te = scale_and_reshape(X_tr, X_te)

    # Stratified, shuffled validation split for early stopping. (Keras'
    # validation_split takes the *last* rows unshuffled, which can hold out a
    # single class when the data is ordered by label.)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=seed
    )

    callbacks = []
    if cfg.early_stopping_patience > 0:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=cfg.early_stopping_patience, restore_best_weights=True
            )
        )

    model = build_model(cfg, input_len=X_tr.shape[1], n_classes=n_classes)
    model.fit(
        X_fit,
        keras.utils.to_categorical(y_fit, n_classes),
        validation_data=(X_val, keras.utils.to_categorical(y_val, n_classes)),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return model.predict(X_te, verbose=0).argmax(axis=1)


def evaluate(cfg: CNNConfig, X, y, n_classes: int, args) -> dict:
    """Evaluate one configuration; return per-run metrics and a summed confusion matrix."""
    if args.mode == "split":
        splits = [
            train_test_split(
                np.arange(len(y)), test_size=args.test_size, stratify=y, random_state=args.seed
            )
        ]
    else:
        rskf = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)
        splits = list(rskf.split(X, y))

    accs, f1s = [], []
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i, (tr, te) in enumerate(splits):
        y_pred = fit_and_predict(cfg, X[tr], y[tr], X[te], n_classes, seed=args.seed + i)
        accs.append(accuracy_score(y[te], y_pred))
        f1s.append(f1_score(y[te], y_pred, average="macro"))
        cm += confusion_matrix(y[te], y_pred, labels=range(n_classes))
        print(f"  [{cfg.name}] run {i + 1}/{len(splits)}: acc={accs[-1]:.4f} macro-F1={f1s[-1]:.4f}")

    return {
        "config": cfg.name,
        "runs": len(splits),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="CSV/XLSX: signal columns then a label column")
    parser.add_argument("--config", default="gap_dropout", choices=[*CONFIGS, "all"])
    parser.add_argument("--mode", default="cv", choices=["cv", "split"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    X, y, encoder = load_dataset(args.data)
    n_classes = len(encoder.classes_)
    print(f"Loaded {X.shape[0]} samples x {X.shape[1]} features, classes: {list(encoder.classes_)}")

    names = list(CONFIGS) if args.config == "all" else [args.config]
    results = [evaluate(CONFIGS[n], X, y, n_classes, args) for n in names]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "confusion_matrix"} for r in results])
    summary.to_csv(out / "summary.csv", index=False)
    with open(out / "results.json", "w") as f:
        json.dump({"classes": list(encoder.classes_), "args": vars(args), "results": results}, f, indent=2)

    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved {out / 'summary.csv'} and {out / 'results.json'}")


if __name__ == "__main__":
    main()
