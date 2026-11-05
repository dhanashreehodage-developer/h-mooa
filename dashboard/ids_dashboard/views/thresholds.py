import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

from .. import results as R
from ..metrics import classification_metrics
from ..state import is_synthetic
from ..thresholds import adaptive_predictions, default_predictions, optimize_thresholds
from ..ui import badge, metric_cards, source, style_fig

EXPLORATORY = ("Adaptive thresholding is **exploratory**. In our current assessment it did **not** produce a "
               "meaningful overall improvement: assessment Macro F1 went from 0.8963 (default argmax) to 0.8952, "
               "and Macro FPR from 0.017573 to 0.017565. Gains seen on the calibration data did not transfer.")


def _explain():
    st.markdown(
        "A Random Forest outputs one probability per class. The default rule picks the most probable class. "
        "The adaptive rule gives every class its own threshold: a class is **eligible** only if its probability "
        "reaches its threshold, and the most probable eligible class wins (falling back to argmax if none is "
        "eligible). Thresholds are tuned by coordinate descent on **calibration** data to minimise "
        "(1 - Macro F1) + 2 x Macro FPR, then checked on separate **assessment** data."
    )
    st.warning(EXPLORATORY)


def _actual():
    badge("preliminary", "Measured outputs of the notebook's threshold experiment (RF_9, 2,500 calibration / 2,500 "
                         "assessment samples). Held-out test data was not used.")
    c1, c2 = st.columns([1, 2])
    c1.dataframe(pd.DataFrame(R.THRESHOLD_SPLIT.items(), columns=["Setting", "Value"]).astype(str), hide_index=True)
    fig = px.bar(R.THRESHOLDS, x="Class", y="Optimized Threshold", text="Optimized Threshold", color="Class")
    fig.add_hline(y=0.5, line_dash="dash", annotation_text="default 0.50")
    c2.plotly_chart(style_fig(fig, 340, "Optimised class-specific thresholds"))
    c1, c2 = st.columns(2)
    c1.markdown("**Assessment data (independent)**")
    c1.dataframe(R.THRESHOLD_ASSESSMENT, hide_index=True)
    c2.markdown("**Calibration data (used for tuning, optimistic)**")
    c2.dataframe(R.THRESHOLD_CALIBRATION, hide_index=True)
    fig = px.bar(R.THRESHOLD_RECALL, x="Class", y="Recall Change", color="Recall Change", color_continuous_scale="RdBu",
                 hover_data=["Support", "Default Recall", "Adaptive Recall"])
    st.plotly_chart(style_fig(fig, 320, "Class-wise recall change on assessment data (adaptive - default)"))
    source(R.SOURCE_THRESH)


def _live():
    res = st.session_state.get("rf_result")
    if res is None:
        st.info("Train a model in **Random Forest Classification** first.")
        return
    badge("live", "Validation split divided into non-overlapping calibration and assessment halves. "
                  "The test split is not used.")
    if is_synthetic():
        badge("synthetic")
    labels = list(res["model"].classes_)
    y_val = res["splits"]["y_val"].to_numpy()
    proba = res["val_proba"]
    if st.button("Calibrate thresholds", type="primary"):
        idx = np.arange(len(y_val))
        strat = y_val if pd.Series(y_val).value_counts().min() >= 2 else None
        cal, ass = train_test_split(idx, test_size=0.5, random_state=42, stratify=strat)
        with st.spinner("Coordinate descent over class thresholds..."):
            th, hist = optimize_thresholds(y_val[cal], proba[cal], labels)
        st.session_state["threshold_result"] = dict(th=th, hist=hist, cal=cal, ass=ass)
    tr = st.session_state.get("threshold_result")
    if not tr:
        return
    ass, cal = tr["ass"], tr["cal"]
    st.markdown("**Class-specific thresholds** (adjust manually to see the effect)")
    cols = st.columns(min(len(labels), 8))
    manual = np.array([cols[i % len(cols)].slider(str(c), 0.05, 0.95, float(tr["th"][i]), 0.05, key=f"th_{c}_{i}")
                       for i, c in enumerate(labels)])
    rows = []
    for name, idx in (("Calibration", cal), ("Assessment", ass)):
        for rule, pred in (("Default argmax", default_predictions(proba[idx], labels)),
                           ("Adaptive thresholds", adaptive_predictions(proba[idx], labels, manual))):
            rows.append({"Data": name, "Rule": rule, **classification_metrics(y_val[idx], pred, labels)})
    comp = pd.DataFrame(rows)
    a = comp[comp["Data"] == "Assessment"].set_index("Rule")
    d_f1 = a.loc["Adaptive thresholds", "Macro F1"] - a.loc["Default argmax", "Macro F1"]
    d_fpr = a.loc["Adaptive thresholds", "Macro FPR"] - a.loc["Default argmax", "Macro FPR"]
    metric_cards([("Assessment Macro F1 (default)", f"{a.loc['Default argmax', 'Macro F1']:.4f}", ""),
                  ("Assessment Macro F1 (adaptive)", f"{a.loc['Adaptive thresholds', 'Macro F1']:.4f}", f"change {d_f1:+.4f}"),
                  ("Assessment Macro FPR (default)", f"{a.loc['Default argmax', 'Macro FPR']:.5f}", ""),
                  ("Assessment Macro FPR (adaptive)", f"{a.loc['Adaptive thresholds', 'Macro FPR']:.5f}", f"change {d_fpr:+.5f}")])
    st.dataframe(comp.round(5), hide_index=True)
    st.dataframe(pd.DataFrame(tr["hist"]).round(5), hide_index=True)

    st.markdown("##### Inspect one assessment sample")
    k = st.slider("Assessment sample", 0, len(ass) - 1, 0)
    i = ass[k]
    p = proba[i]
    default = labels[int(np.argmax(p))]
    adaptive = adaptive_predictions(p.reshape(1, -1), labels, manual)[0]
    fig = go.Figure(go.Bar(x=labels, y=p, name="Predicted probability",
                           marker_color=["#00B4FF" if pi >= ti else "#475569" for pi, ti in zip(p, manual)]))
    fig.add_trace(go.Scatter(x=labels, y=manual, mode="markers", name="Class threshold",
                             marker=dict(symbol="line-ew-open", size=28, color="#F59E0B", line=dict(width=3))))
    st.plotly_chart(style_fig(fig, 340, "Class probabilities (blue = reaches its threshold)"))
    metric_cards([("Actual class", str(y_val[i]), ""), ("Default decision", default, "argmax"),
                  ("Adaptive decision", adaptive, "eligible argmax")])


def render():
    st.subheader("Adaptive Threshold Optimization")
    _explain()
    t1, t2 = st.tabs(["Research outputs", "Live demonstration on trained model"])
    with t1:
        _actual()
    with t2:
        _live()
