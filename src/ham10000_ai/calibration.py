from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
from .metrics import softmax_np


def fit_temperature_scaling(logits_val: np.ndarray, y_val: np.ndarray, bounds=(0.2, 5.0)) -> tuple[float, float]:
    """Fit scalar temperature on validation logits by minimising NLL."""
    y_val = np.asarray(y_val).astype(int)

    def objective(t: float) -> float:
        probs = softmax_np(logits_val, temperature=t)
        return float(log_loss(y_val, probs, labels=list(range(probs.shape[1]))))

    res = minimize_scalar(objective, bounds=bounds, method="bounded")
    return float(res.x), float(res.fun)
