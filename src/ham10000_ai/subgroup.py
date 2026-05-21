from __future__ import annotations

import numpy as np
import pandas as pd
from .metrics import evaluate_probs


def add_age_bins(df: pd.DataFrame, age_col: str = "age") -> pd.DataFrame:
    out = df.copy()
    out["age_bin"] = pd.cut(out[age_col], bins=[-np.inf, 30, 45, 60, 75, np.inf], labels=["<=30", "31-45", "46-60", "61-75", ">75"])
    return out


def subgroup_metrics(df: pd.DataFrame, y_true: np.ndarray, probs: np.ndarray, subgroup_vars: list[str], n_bins: int = 15, min_n: int = 30) -> pd.DataFrame:
    rows = []
    df = df.reset_index(drop=True)
    for var in subgroup_vars:
        for name, idx in df.groupby(var, dropna=False).groups.items():
            idx = np.array(list(idx))
            if len(idx) < min_n:
                continue
            m = evaluate_probs(y_true[idx], probs[idx], prefix="", n_bins=n_bins)
            m.update({"subgroup_variable": var, "subgroup": str(name), "n": int(len(idx))})
            rows.append(m)
    return pd.DataFrame(rows)


def subgroup_gaps(metrics_df: pd.DataFrame, metrics=("balanced_accuracy", "macro_f1", "brier", "ece")) -> pd.DataFrame:
    rows = []
    for var, g in metrics_df.groupby("subgroup_variable"):
        for metric in metrics:
            vals = g[metric].dropna()
            if len(vals) == 0:
                continue
            rows.append({"subgroup_variable": var, "metric": metric, "min": vals.min(), "max": vals.max(), "gap": vals.max() - vals.min()})
    return pd.DataFrame(rows)
