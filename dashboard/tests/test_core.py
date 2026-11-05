import numpy as np
import pandas as pd
import pytest

from ids_dashboard import data_utils as D
from ids_dashboard import results as R
from ids_dashboard.cuckoo import CuckooSearchDemo, levy_steps
from ids_dashboard.firefly_pso import FireflyPSODemo
from ids_dashboard.metrics import macro_fpr
from ids_dashboard.modeling import SPLIT_GROUP, build_rf, fit_and_evaluate, three_way_split
from ids_dashboard.pareto import dominates, non_dominated_sort
from ids_dashboard.thresholds import adaptive_predictions, optimize_thresholds


@pytest.fixture(scope="module")
def prepared():
    df = D.make_synthetic_demo(3000)
    target = D.target_candidates(df)[0]
    feats = D.numeric_feature_columns(df, target, D.default_excluded_columns(df, target))
    X, y = D.prepare_xy(df, target, feats)
    return df, target, X, y


def test_target_detected_without_assuming_name(prepared):
    df, target, X, y = prepared
    assert target == "traffic_category"
    assert "timestamp" in D.default_excluded_columns(df, target)
    assert target not in X.columns and y.nunique() == 8


def test_group_split_has_no_shared_feature_vectors(prepared):
    _, _, X, y = prepared
    Xd = pd.concat([X, X.iloc[:300]], ignore_index=True)
    yd = pd.concat([y, y.iloc[:300]], ignore_index=True)
    sp = three_way_split(Xd, yd, method=SPLIT_GROUP)
    def h(d):
        return set(pd.util.hash_pandas_object(d.fillna(-1), index=False))

    assert not (h(sp["X_train"]) & h(sp["X_test"]))
    assert not (h(sp["X_val"]) & h(sp["X_test"]))


def test_rf_and_metrics(prepared):
    _, _, X, y = prepared
    sp = three_way_split(X, y)
    out = fit_and_evaluate(build_rf(30), sp["X_train"], sp["y_train"], sp["X_test"], sp["y_test"], sorted(y.unique()))
    assert 0.5 < out["metrics"]["Macro F1"] <= 1
    assert 0 <= out["metrics"]["Macro FPR"] < 0.2


def test_macro_fpr_perfect():
    assert macro_fpr(["a", "b"], ["a", "b"], ["a", "b"]) == 0.0


def test_pareto():
    assert dominates([1, 1], [2, 2]) and not dominates([1, 2], [2, 1])
    assert non_dominated_sort([[1, 2], [2, 1], [3, 3]]) == [[0, 1], [2]]


def test_thresholds_rule_and_fallback():
    proba = np.array([[0.6, 0.4], [0.45, 0.55]])
    assert list(adaptive_predictions(proba, ["a", "b"], [0.7, 0.3])) == ["b", "b"]
    assert list(adaptive_predictions(proba, ["a", "b"], [0.9, 0.9])) == ["a", "b"]
    th, hist = optimize_thresholds(np.array(["a", "b"]), proba, ["a", "b"])
    assert len(th) == 2 and hist[0]["Pass"] == 0


def test_optimizer_demos_run(prepared):
    _, _, X, y = prepared
    sp = three_way_split(X, y)
    fp = FireflyPSODemo(sp["X_train"], sp["y_train"], sp["X_val"], sp["y_val"], population=4, iterations=1, trees=10).run()
    assert fp["archive"] and all(a["feature_count"] >= 3 for a in fp["archive"])
    bounds = {"n_estimators": (10, 30), "max_depth": (3, 10), "min_samples_leaf": (1, 4)}
    cs = CuckooSearchDemo(sp["X_train"], sp["y_train"], sp["X_val"], sp["y_val"], bounds, nests=3, iterations=1).run()
    assert cs["archive"] and len(levy_steps(10)) == 10


def test_recorded_results_match_reported_baseline():
    df = R.FINAL_TEST_METRICS.set_index("Model")
    assert round(df.loc["RF_71", "Accuracy"] * 100, 2) == 97.10
    assert round(df.loc["RF_9", "Macro F1"] * 100, 2) == 95.79
    assert round(df.loc["RF_6", "Accuracy"] * 100, 2) == 95.83
    assert round(df.loc["RF_9", "Feature Reduction (%)"], 2) == 87.32
    assert len(R.RF_71_FEATURES) == 71 and set(R.RF_6_FEATURES) <= set(R.RF_9_FEATURES)


def test_bad_csv_rejected(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(D.DatasetError):
        D.read_csv_sampled(str(p), 100)
    with pytest.raises(D.DatasetError):
        D.read_csv_sampled(str(tmp_path / "missing.csv"), 100)
