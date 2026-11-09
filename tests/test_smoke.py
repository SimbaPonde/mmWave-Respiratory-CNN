"""Smoke test: the full train/evaluate path runs on synthetic breathing-like signals.

Uses four sinusoidal "breathing rates" so the test needs no private data.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def make_synthetic(path: Path, per_class: int = 20, length: int = 600) -> None:
    rng = np.random.default_rng(0)
    t = np.linspace(0, 30, length)
    rows = []
    for label, freq in [("bradypnea", 0.15), ("eupnea", 0.25), ("kussmaul", 0.4), ("biot", 0.6)]:
        for _ in range(per_class):
            sig = np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi)) + 0.2 * rng.standard_normal(length)
            rows.append([*sig, label])
    pd.DataFrame(rows, columns=[*map(str, range(length)), "label"]).to_csv(path, index=False)


def test_train_cli_runs(tmp_path):
    data = tmp_path / "synthetic.csv"
    make_synthetic(data)
    out = tmp_path / "results"
    cmd = [sys.executable, "train.py", "--data", str(data), "--folds", "2", "--repeats", "1", "--out", str(out)]
    subprocess.run(cmd, cwd=ROOT, check=True)

    summary = pd.read_csv(out / "summary.csv")
    assert summary.loc[0, "config"] == "gap_dropout"
    assert summary.loc[0, "accuracy_mean"] > 0.5  # well above 4-class chance on separable data
