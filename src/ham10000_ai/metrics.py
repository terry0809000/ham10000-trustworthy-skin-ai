from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, log_loss, roc_auc_score


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z -= np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    onehot = np.eye(probs.shape[1])[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if hi > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        acc = correct[mask].mean()
        c = conf[mask].mean()
        w = mask.mean()
        ece += w * abs(acc - c)
        rows.append({"bin_low": lo, "bin_high": hi, "n": int(mask.sum()), "accuracy": float(acc), "confidence": float(c)})
    return float(ece), rows


def evaluate_probs(y_true: np.ndarray, probs: np.ndarray, prefix: str = "", n_bins: int = 15) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    pred = probs.argmax(axis=1)
    out = {
        prefix + "accuracy": accuracy_score(y_true, pred),
        prefix + "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        prefix + "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        prefix + "weighted_f1": f1_score(y_true, pred, average="weighted", zero_division=0),
        prefix + "macro_recall": recall_score(y_true, pred, average="macro", zero_division=0),
        prefix + "brier": multiclass_brier(y_true, probs),
        prefix + "nll": log_loss(y_true, probs, labels=list(range(probs.shape[1]))),
    }
    ece, _ = expected_calibration_error(y_true, probs, n_bins=n_bins)
    out[prefix + "ece"] = ece
    try:
        out[prefix + "ovr_roc_auc_macro"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except Exception:
        out[prefix + "ovr_roc_auc_macro"] = np.nan
    return {k: float(v) for k, v in out.items()}
