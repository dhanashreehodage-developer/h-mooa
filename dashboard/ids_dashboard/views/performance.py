import pandas as pd
import plotly.express as px
import streamlit as st

from .. import results as R
from ..state import is_synthetic
from ..ui import badge, metric_cards, pct, source, style_fig

METRICS = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Macro FPR"]


def _actual():
    badge("actual", f"Final held-out test evaluation of the three Random Forest models ({R.TEST_SAMPLES:,} samples).")
    df = R.FINAL_TEST_METRICS
    model = st.radio("Model", list(df["Model"]), horizontal=True)
    r = df[df["Model"] == model].iloc[0]
    metric_cards([("Accuracy", pct(r["Accuracy"]), ""), ("Macro Precision", pct(r["Macro Precision"]), ""),
                  ("Macro Recall", pct(r["Macro Recall"]), ""), ("Macro F1", pct(r["Macro F1"]), "")])
    st.write("")
    metric_cards([("Macro FPR", pct(r["Macro FPR"], 3), "one-vs-rest"),
                  ("Feature reduction", f"{r['Feature Reduction (%)']:.2f}%", f"{int(r['Feature Count'])} of 71 features"),
                  ("Training time", f"{r['Training Time (s)']:.2f} s", "full training set"),
                  ("Prediction latency", f"{r['Latency (ms / 1,000 samples)']:.2f} ms", "per 1,000 samples (derived)")])
    st.caption("Latency is derived from the measured batch prediction time over the 109,771 test samples.")

    long = df.melt(id_vars="Model", value_vars=METRICS[:4], var_name="Metric", value_name="Value")
    c1, c2 = st.columns(2)
    fig = px.bar(long, x="Metric", y="Value", color="Model", barmode="group", text_auto=".4f")
    fig.update_yaxes(range=[0.9, 1.0])
    c1.plotly_chart(style_fig(fig, 380, "Detection metrics (y-axis starts at 0.90)"))
    fig = px.bar(df, x="Model", y="Macro FPR", color="Model", text_auto=".5f")
    c2.plotly_chart(style_fig(fig, 380, "Macro false positive rate (lower is better)"))
    c1, c2 = st.columns(2)
    fig = px.bar(df, x="Model", y=["Training Time (s)", "Batch Prediction Time (s)"], barmode="group", text_auto=".2f")
    c1.plotly_chart(style_fig(fig, 360, "Computational cost"))
    fig = px.scatter(df, x="Feature Count", y="Macro F1", text="Model", size="Training Time (s)", color="Model")
    fig.update_traces(textposition="top center")
    c2.plotly_chart(style_fig(fig, 360, "Feature count vs Macro F1"))
    st.dataframe(df.round(6), hide_index=True)
    source(f"{R.SOURCE_FINAL}; {R.SOURCE_TIMING}")

    st.markdown("##### Class-wise F1")
    long = R.CLASSWISE_F1.melt(id_vars="Class", var_name="Model", value_name="F1")
    fig = px.bar(long, x="Class", y="F1", color="Model", barmode="group")
    fig.update_yaxes(range=[0.85, 1.0])
    st.plotly_chart(style_fig(fig, 360))
    c1, c2 = st.columns(2)
    c1.markdown("**Attacks classified as benign**")
    c1.dataframe(R.ATTACK_TO_BENIGN, hide_index=True)
    c2.markdown("**McNemar tests (Holm-adjusted)**")
    c2.dataframe(R.MCNEMAR, hide_index=True)
    st.markdown("**Bootstrap 95% confidence intervals**")
    st.dataframe(R.BOOTSTRAP, hide_index=True)


def _live():
    runs = st.session_state.get("live_runs")
    if not runs:
        st.info("Train a model in **Random Forest Classification** to compare your runs here.")
        return
    badge("live", "Runs trained in this session on the loaded dataset. These are not the research results.")
    if is_synthetic():
        badge("synthetic")
    df = pd.DataFrame(runs)
    last = df.iloc[-1]
    metric_cards([("Accuracy", pct(last["Accuracy"]), f"run {int(last['Run'])}"), ("Macro F1", pct(last["Macro F1"]), ""),
                  ("Macro FPR", pct(last["Macro FPR"], 3), ""), ("Feature reduction", f"{last['Feature Reduction (%)']:.1f}%", ""),
                  ("Training time", f"{last['Training Time (s)']:.2f} s", ""),
                  ("Latency", f"{last['Latency (ms / 1,000 samples)']:.2f} ms", "per 1,000")], columns=6)
    long = df.melt(id_vars=["Run", "Feature set"], value_vars=METRICS[:4], var_name="Metric", value_name="Value")
    long["Run"] = long["Run"].astype(str) + ": " + long["Feature set"]
    fig = px.bar(long, x="Metric", y="Value", color="Run", barmode="group")
    st.plotly_chart(style_fig(fig, 360, "Session runs"))
    st.dataframe(df.drop(columns=["Prediction Time (s)"]).round(5), hide_index=True)


def render():
    st.subheader("Performance Evaluation")
    t1, t2 = st.tabs(["Research results", "Session runs"])
    with t1:
        _actual()
    with t2:
        _live()
