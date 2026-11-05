"""Class-wise adaptive decision thresholds (same rule and objective as notebook Step 6.67)."""

import numpy as np
from sklearn.metrics import f1_score

from .metrics import macro_fpr

THRESHOLD_GRID = np.round(np.arange(0.05, 1.00, 0.05), 2)


def default_predictions(proba, labels):
    return np.asarray(labels)[np.argmax(proba, axis=1)]


def adaptive_predictions(proba, labels, thresholds):
    """Pick the most probable class among those whose probability reaches its threshold.

    If no class is eligible, fall back to the ordinary argmax.
    """
    thresholds = np.asarray(thresholds, dtype=float).reshape(1, -1)
    eligible = proba >= thresholds
    adjusted = np.where(eligible, proba, -np.inf)
    idx = np.argmax(adjusted, axis=1)
    none = ~eligible.any(axis=1)
    idx[none] = np.argmax(proba[none], axis=1)
    return np.asarray(labels)[idx]


def objective(y_true, proba, labels, thresholds, fpr_weight=2.0):
    pred = adaptive_predictions(proba, labels, thresholds)
    f1 = f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)
    fpr = macro_fpr(y_true, pred, labels)
    return {"macro_f1": float(f1), "macro_fpr": float(fpr), "objective": float((1 - f1) + fpr_weight * fpr)}


def optimize_thresholds(y_cal, proba_cal, labels, max_passes=3, fpr_weight=2.0):
    """Coordinate descent over a 0.05-0.95 grid, using calibration data only."""
    thresholds = np.full(len(labels), 0.5)
    best = objective(y_cal, proba_cal, labels, thresholds, fpr_weight)
    history = [{"Pass": 0, "Macro F1": best["macro_f1"], "Macro FPR": best["macro_fpr"], "Changes": 0}]
    for p in range(1, max_passes + 1):
        changes = 0
        for k in range(len(labels)):
            current = thresholds[k]
            for t in THRESHOLD_GRID:
                trial = thresholds.copy()
                trial[k] = t
                m = objective(y_cal, proba_cal, labels, trial, fpr_weight)
                if m["objective"] < best["objective"] - 1e-12:
                    best, current = m, t
            if current != thresholds[k]:
                thresholds[k] = current
                changes += 1
        history.append({"Pass": p, "Macro F1": best["macro_f1"], "Macro FPR": best["macro_fpr"], "Changes": changes})
        if changes == 0:
            break
    return thresholds, history
