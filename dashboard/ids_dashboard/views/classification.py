import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from .. import data_utils as D
from .. import results as R
from ..metrics import classification_metrics
from ..modeling import SPLIT_GROUP, SPLIT_STRATIFIED, binary_view, build_rf, fit_and_evaluate, three_way_split
from ..state import is_synthetic, require_prepared
from ..ui import badge, metric_cards, pct, style_fig
from .feature_selection import subsample

MAX_FEATURE_OPTIONS = ["sqrt", "log2", 0.5, 0.8, 1.0]


def confusion_fig(y_true, y_pred, labels, normalize, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    z = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    fig = px.imshow(z, x=labels, y=labels, text_auto=".2f" if normalize else True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Share" if normalize else "Count"), aspect="auto")
    return style_fig(fig, 460, title)


def _feature_sets(X):
    sets = {"All numeric features": list(X.columns)}
    for name, feats in (("Research RF_9 subset", R.RF_9_FEATURES), ("Research RF_6 subset", R.RF_6_FEATURES)):
        if all(f in X.columns for f in feats):
            sets[name] = feats
    demo = st.session_state.get("demo_features")
    if demo and all(f in X.columns for f in demo):
        sets["Firefly-PSO demo subset"] = demo
    sets["Custom selection"] = None
    return sets


def _settings(X, y):
    st.markdown("##### Experiment settings")
    sets = _feature_sets(X)
    c1, c2 = st.columns([1, 2])
    choice = c1.selectbox("Feature set", list(sets))
    feats = sets[choice]
    if feats is None:
        feats = c2.multiselect("Features", list(X.columns), default=list(X.columns)[: min(10, X.shape[1])])
    else:
        c2.caption(f"{len(feats)} features: " + ", ".join(feats[:12]) + (" ..." if len(feats) > 12 else ""))

    c1, c2, c3, c4 = st.columns(4)
    method = c1.radio("Split method", [SPLIT_GROUP, SPLIT_STRATIFIED],
                      help="Group-aware splitting keeps identical feature vectors in one partition, as in the notebook, "
                           "so duplicated rows cannot leak from training into testing.")
    test_size = c2.slider("Test share", 0.1, 0.4, 0.2, 0.05)
    val_size = c3.slider("Validation share (of remaining)", 0.1, 0.4, 0.2, 0.05)
    max_train = c4.select_slider("Max training rows", [10_000, 25_000, 50_000, 100_000, 250_000, 500_000], value=50_000,
                                 help="Training is subsampled for speed on a laptop; the test split is not subsampled.")

    p = st.session_state.get("rf_params", {})
    st.markdown("**Random Forest hyperparameters** (default = final research configuration: 100 trees, balanced class weights)")
    c1, c2, c3, c4 = st.columns(4)
    n_est = c1.number_input("Number of trees", 10, 500, int(p.get("n_estimators", 100)), 10)
    depth = c2.number_input("Maximum depth (0 = unlimited)", 0, 100, int(p.get("max_depth") or 0))
    leaf = c3.number_input("Min samples per leaf", 1, 50, int(p.get("min_samples_leaf", 1)))
    mf_default = p.get("max_features", "sqrt")
    mf = c4.selectbox("Maximum features", MAX_FEATURE_OPTIONS,
                      index=MAX_FEATURE_OPTIONS.index(mf_default) if mf_default in MAX_FEATURE_OPTIONS else 0, format_func=str)
    classes = sorted(y.unique())
    detected = D.detect_benign_label(classes)
    benign = st.selectbox("Benign (normal traffic) class for the benign-vs-attack view", classes,
                          index=classes.index(detected) if detected in classes else 0)
    return dict(feats=feats, choice=choice, method=method, test_size=test_size, val_size=val_size,
                max_train=max_train, params=dict(n_estimators=n_est, max_depth=depth, min_samples_leaf=leaf, max_features=mf),
                benign=benign)


def _train(X, y, cfg):
    if not cfg["feats"]:
        st.error("Select at least one feature.")
        return
    try:
        with st.spinner("Splitting data and training the Random Forest..."):
            sp = three_way_split(X[cfg["feats"]], y, cfg["test_size"], cfg["val_size"], cfg["method"])
            missing = set(y.unique()) - set(sp["y_train"].unique())
            if missing:
                st.warning(f"Classes absent from the training split: {sorted(missing)}. Load more rows.")
            Xtr, ytr = subsample(sp["X_train"], sp["y_train"], cfg["max_train"])
            labels = sorted(y.unique())
            out = fit_and_evaluate(build_rf(**cfg["params"]), Xtr, ytr, sp["X_test"], sp["y_test"], labels)
            out["val_proba"] = out["model"].predict_proba(sp["X_val"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Training failed: {exc}")
        return
    out.update(splits=sp, config=cfg, n_train=len(Xtr), features=cfg["feats"])
    st.session_state["rf_result"] = out
    st.session_state.pop("threshold_result", None)
    m = out["metrics"]
    st.session_state.setdefault("live_runs", []).append(
        {"Run": len(st.session_state.get("live_runs", [])) + 1, "Feature set": cfg["choice"], "Features": len(cfg["feats"]),
         "Feature Reduction (%)": (1 - len(cfg["feats"]) / X.shape[1]) * 100, **m}
    )


def _results(res):
    sp, m, labels = res["splits"], res["metrics"], res["labels"]
    badge("live", f"Random Forest trained on {res['n_train']:,} training rows, evaluated on {len(sp['y_test']):,} "
                  f"held-out test rows ({sp['method'].split(' (')[0].lower()} split).")
    if is_synthetic():
        badge("synthetic")
    metric_cards([("Accuracy", pct(m["Accuracy"]), "test split"), ("Macro F1", pct(m["Macro F1"]), "multiclass"),
                  ("Macro FPR", pct(m["Macro FPR"], 3), "one-vs-rest"), ("Training time", f"{m['Training Time (s)']:.2f} s", ""),
                  ("Latency", f"{m['Latency (ms / 1,000 samples)']:.2f} ms", "per 1,000 samples")])
    if len(labels) != R.N_CLASSES and not is_synthetic():
        st.warning(f"This target has {len(labels)} classes; the research dataset has {R.N_CLASSES}.")
    y_test = sp["y_test"].to_numpy()
    t1, t2, t3 = st.tabs(["Multiclass", "Benign vs attack", "Predictions"])
    with t1:
        norm = st.toggle("Normalise confusion matrix by actual class", value=True)
        st.plotly_chart(confusion_fig(y_test, res["pred"], labels, norm, f"Confusion matrix ({len(labels)} classes)"))
        rep = pd.DataFrame(classification_report(y_test, res["pred"], labels=labels, output_dict=True, zero_division=0)).T
        st.markdown("**Classification report**")
        st.dataframe(rep.round(4))
    with t2:
        benign = res["config"]["benign"]
        bt, bp = binary_view(y_test, res["pred"], benign)
        st.caption(f"Derived from the multiclass predictions: '{benign}' = Benign, every other class = Attack.")
        bm = classification_metrics(bt, bp, ["Attack", "Benign"])
        cm = confusion_matrix(bt, bp, labels=["Attack", "Benign"])
        missed = cm[0, 1] / max(cm[0].sum(), 1)
        false_alarm = cm[1, 0] / max(cm[1].sum(), 1)
        metric_cards([("Binary accuracy", pct(bm["Accuracy"]), ""), ("Attack detection rate", pct(1 - missed), "attacks flagged"),
                      ("Attacks classified as benign", pct(missed), "missed attacks"),
                      ("False alarm rate", pct(false_alarm), "benign flagged as attack")])
        st.plotly_chart(confusion_fig(bt, bp, ["Attack", "Benign"], False, "Benign vs attack confusion matrix"))
    with t3:
        proba = res["proba"]
        pred_df = pd.DataFrame({"Actual": y_test, "Predicted": res["pred"], "Confidence": proba.max(axis=1),
                                "Correct": y_test == res["pred"]})
        only_err = st.checkbox("Show misclassified rows only")
        st.dataframe((pred_df[~pred_df["Correct"]] if only_err else pred_df).head(500).round(4), hide_index=True)
        st.download_button("Download test predictions (CSV)", pred_df.to_csv(index=False), "test_predictions.csv")


def _batch_predict(res):
    st.markdown("##### Classify another CSV (offline batch)")
    st.caption("Applies the trained model to a file with the same feature columns. This is offline batch "
               "classification, not live traffic monitoring.")
    up = st.file_uploader("CSV to classify", type=["csv"], key="batch_csv")
    if up is None:
        return
    try:
        new = D.read_uploaded(up, 500_000)
        missing = [f for f in res["features"] if f not in new.columns]
        if missing:
            st.error(f"Missing feature columns: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
            return
        Xn = new[res["features"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        proba = res["model"].predict_proba(Xn)
        out = new.copy()
        out["predicted_class"] = res["model"].classes_[proba.argmax(axis=1)]
        out["confidence"] = proba.max(axis=1)
        st.dataframe(out.head(200))
        fig = px.histogram(out, x="predicted_class", color="predicted_class")
        st.plotly_chart(style_fig(fig, 320, "Predicted class counts"))
        st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not classify the file: {exc}")


def render():
    st.subheader("Random Forest Intrusion Classification")
    X, y = require_prepared()
    if X is None:
        return
    cfg = _settings(X, y)
    if st.button("Train and evaluate Random Forest", type="primary"):
        _train(X, y, cfg)
    res = st.session_state.get("rf_result")
    if res:
        st.divider()
        _results(res)
        st.divider()
        _batch_predict(res)
