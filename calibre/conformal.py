"""Split conformal prediction, implemented from scratch with NumPy.

Split conformal prediction wraps *any* trained model and turns its point
predictions into prediction sets (classification) or prediction intervals
(regression) that contain the true answer with probability >= 1 - alpha.
The only assumption is that calibration and future data are exchangeable
(roughly: drawn from the same distribution). No assumptions about the model.

Reference: Vovk, Gammerman, Shafer (2005); Angelopoulos & Bates (2021),
"A Gentle Introduction to Conformal Prediction and Distribution-Free
Uncertainty Quantification".
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1 - alpha) quantile of calibration scores.

    With n calibration scores, the correct level is ceil((n + 1)(1 - alpha)) / n,
    not simply 1 - alpha. This correction is what makes the coverage guarantee
    hold for small calibration sets.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        raise ValueError("Need at least one calibration score.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


# --------------------------------------------------------------------------
# Classification (least-ambiguous set-valued classifier, "LAC")
# --------------------------------------------------------------------------

def classification_scores(proba: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    """Nonconformity score: 1 - probability the model gave the true class."""
    return 1.0 - proba[np.arange(len(y_idx)), y_idx]


def classification_sets(proba: np.ndarray, qhat: float) -> np.ndarray:
    """Boolean matrix (n_samples, n_classes): which classes are in each set.

    A class is included when its probability is at least 1 - qhat. The model's
    top class is always kept, so a set is never empty (this can only make the
    sets larger, so coverage stays valid).
    """
    sets = proba >= 1.0 - qhat
    sets[np.arange(len(proba)), proba.argmax(axis=1)] = True
    return sets


def set_coverage(sets: np.ndarray, y_idx: np.ndarray) -> float:
    return float(sets[np.arange(len(y_idx)), y_idx].mean())


# --------------------------------------------------------------------------
# Regression (absolute-residual intervals)
# --------------------------------------------------------------------------

def regression_scores(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(y, dtype=float) - np.asarray(pred, dtype=float))


def regression_intervals(pred: np.ndarray, qhat: float):
    pred = np.asarray(pred, dtype=float)
    return pred - qhat, pred + qhat


def interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    return float(((y >= lo) & (y <= hi)).mean())
