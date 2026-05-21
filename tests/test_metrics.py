import numpy as np
from ham10000_ai.metrics import softmax_np, multiclass_brier, expected_calibration_error, evaluate_probs
from ham10000_ai.selective_referral import selective_curve, uncertainty_scores


def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]])
    probs = softmax_np(logits)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_brier_is_nonnegative():
    y = np.array([0, 1, 2])
    p = np.eye(3)
    assert multiclass_brier(y, p) == 0.0


def test_ece_range():
    y = np.array([0, 1, 1, 0])
    p = np.array([[0.9, 0.1], [0.3, 0.7], [0.6, 0.4], [0.8, 0.2]])
    ece, rows = expected_calibration_error(y, p, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_selective_sensitivity_bounded():
    y = np.array([1, 1, 0, 0])
    p = np.array([[0.2, 0.8], [0.7, 0.3], [0.9, 0.1], [0.6, 0.4]])
    unc = uncertainty_scores(p)["maxprob_uncertainty"]
    curve = selective_curve(y, p, unc, positive_class=1)
    vals = curve["auto_sensitivity_for_class"].dropna()
    assert (vals <= 1.0).all()
    assert (vals >= 0.0).all()
