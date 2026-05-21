from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .metrics import expected_calibration_error


def plot_reliability(y_true, probs, n_bins=15, title="Reliability diagram", out_path=None):
    ece, rows = expected_calibration_error(y_true, probs, n_bins=n_bins)
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--")
    if not df.empty:
        ax.plot(df["confidence"], df["accuracy"], marker="o")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"{title}\nECE={ece:.3f}")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200)
    return fig, ax


def plot_model_selection(df: pd.DataFrame, out_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    ax.bar(x, df["val_macro_f1"], label="Macro-F1")
    if "val_balanced_accuracy" in df:
        ax.plot(x, df["val_balanced_accuracy"], marker="o", label="Balanced accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(df["run_tag"], rotation=30, ha="right")
    ax.set_ylabel("Validation metric")
    ax.set_title("Validation-only model selection")
    ax.legend(frameon=False)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200)
    return fig, ax
