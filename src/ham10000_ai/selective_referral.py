from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def predictive_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def uncertainty_scores(probs: np.ndarray) -> dict[str, np.ndarray]:
    sorted_probs = -np.sort(-probs, axis=1)
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    return {
        "maxprob_uncertainty": 1.0 - probs.max(axis=1),
        "entropy": predictive_entropy(probs),
        "negative_margin": -margin,
    }


def selective_curve(y_true: np.ndarray, probs: np.ndarray, uncertainty: np.ndarray, positive_class: int | None = None) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    pred = probs.argmax(axis=1)
    order = np.argsort(uncertainty)
    n = len(y_true)
    rows = []
    for k in range(max(1, int(0.05 * n)), n + 1):
        keep = order[:k]
        kept_mask = np.zeros(n, dtype=bool)
        kept_mask[keep] = True
        row = {
            "coverage": k / n,
            "referral_rate": 1 - k / n,
            "selective_risk": 1 - accuracy_score(y_true[keep], pred[keep]),
            "balanced_accuracy": balanced_accuracy_score(y_true[keep], pred[keep]) if len(np.unique(y_true[keep])) > 1 else np.nan,
            "macro_f1": f1_score(y_true[keep], pred[keep], average="macro", zero_division=0),
        }
        if positive_class is not None:
            true_pos = y_true == positive_class
            if true_pos.sum() > 0:
                row["auto_sensitivity_for_class"] = float((true_pos & kept_mask & (pred == positive_class)).sum() / true_pos.sum())
                row["coverage_for_true_class"] = float(kept_mask[true_pos].mean())
            else:
                row["auto_sensitivity_for_class"] = np.nan
                row["coverage_for_true_class"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def operating_points(curve: pd.DataFrame, coverages=(0.95, 0.90, 0.80, 0.70, 0.50)) -> pd.DataFrame:
    rows = []
    for target in coverages:
        idx = (curve["coverage"] - target).abs().idxmin()
        row = curve.loc[idx].to_dict()
        row["target_coverage"] = target
        rows.append(row)
    return pd.DataFrame(rows)
