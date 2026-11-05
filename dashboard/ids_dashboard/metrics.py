"""Evaluation metrics used throughout the dashboard (mirrors the notebook definitions)."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def macro_fpr(y_true, y_pred, labels) -> float:
    """Macro one-vs-rest false positive rate: mean of FP / (FP + TN) over classes."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    fprs = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        denom = fp + tn
        fprs.append(fp / denom if denom > 0 else 0.0)
    return float(np.mean(fprs))


def classification_metrics(y_true, y_pred, labels) -> dict:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro Precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "Macro Recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "Macro F1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "Weighted F1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "Macro FPR": macro_fpr(y_true, y_pred, labels),
    }
