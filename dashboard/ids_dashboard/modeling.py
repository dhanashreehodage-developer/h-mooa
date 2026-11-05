"""Leakage-safe splitting and Random Forest training/evaluation."""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from .metrics import classification_metrics

SPLIT_GROUP = "Group-aware (identical feature vectors kept together)"
SPLIT_STRATIFIED = "Stratified random"


def _feature_vector_groups(X: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(X.fillna(-9.99e99), index=False).to_numpy()


def _split_once(X, y, test_size, method, seed):
    if method == SPLIT_GROUP:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        a, b = next(splitter.split(X, y, groups=_feature_vector_groups(X)))
        return a, b
    stratify = y if y.value_counts().min() >= 2 else None
    idx = np.arange(len(y))
    a, b = train_test_split(idx, test_size=test_size, random_state=seed, stratify=stratify)
    return a, b


def three_way_split(X, y, test_size=0.2, val_size=0.2, method=SPLIT_GROUP, seed=42):
    """Train / validation / test split. Validation is carved out of the training portion.

    The test partition is never used for fitting, threshold calibration or model selection.
    """
    if len(y) < 50:
        raise ValueError("At least 50 labelled rows are needed to create train/validation/test splits.")
    trval, test = _split_once(X, y, test_size, method, seed)
    X_trval, y_trval = X.iloc[trval].reset_index(drop=True), y.iloc[trval].reset_index(drop=True)
    tr, val = _split_once(X_trval, y_trval, val_size, method, seed + 1)
    return {
        "X_train": X_trval.iloc[tr].reset_index(drop=True),
        "y_train": y_trval.iloc[tr].reset_index(drop=True),
        "X_val": X_trval.iloc[val].reset_index(drop=True),
        "y_val": y_trval.iloc[val].reset_index(drop=True),
        "X_test": X.iloc[test].reset_index(drop=True),
        "y_test": y.iloc[test].reset_index(drop=True),
        "method": method,
    }


def build_rf(n_estimators=100, max_depth=None, min_samples_leaf=1, max_features="sqrt", seed=42, n_jobs=-1):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=int(n_estimators),
                    max_depth=None if max_depth in (None, 0) else int(max_depth),
                    min_samples_leaf=int(min_samples_leaf),
                    max_features=max_features,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def measure_latency(model, X, repeats=3, batch=4096) -> float:
    """Median prediction time in milliseconds per 1,000 samples."""
    Xb = X.iloc[: min(batch, len(X))]
    if len(Xb) == 0:
        return float("nan")
    model.predict_proba(Xb)
    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict_proba(Xb)
        runs.append(time.perf_counter() - start)
    return float(np.median(runs)) / len(Xb) * 1_000_000


def fit_and_evaluate(model, X_train, y_train, X_eval, y_eval, labels=None):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    labels = list(model.classes_) if labels is None else labels
    start = time.perf_counter()
    proba = model.predict_proba(X_eval)
    predict_time = time.perf_counter() - start
    pred = model.classes_[np.argmax(proba, axis=1)]
    metrics = classification_metrics(y_eval, pred, labels)
    metrics["Training Time (s)"] = train_time
    metrics["Prediction Time (s)"] = predict_time
    metrics["Latency (ms / 1,000 samples)"] = predict_time / max(len(X_eval), 1) * 1_000_000
    return {"model": model, "proba": proba, "pred": pred, "metrics": metrics, "labels": labels}


def binary_view(y_true, y_pred, benign_label):
    def to_bin(values):
        return np.where(np.asarray(values) == benign_label, "Benign", "Attack")

    return to_bin(y_true), to_bin(y_pred)
