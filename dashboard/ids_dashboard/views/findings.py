import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import results as R
from ..ui import badge, info_cards, metric_cards, style_fig


def render():
    st.subheader("Research Findings")
    st.markdown("#### 1. Baseline comparison (final held-out test set)")
    badge("actual", f"Measured on {R.TEST_SAMPLES:,} held-out test samples never used for training or selection.")
    df = R.FINAL_TEST_METRICS
    table = pd.DataFrame({
        "Model": ["RF using 71 features", "RF using 9 features", "RF using 6 features"],
        "Accuracy": [f"{v * 100:.2f}%" for v in df["Accuracy"]],
        "Macro F1": [f"{v * 100:.2f}%" for v in df["Macro F1"]],
        "Feature reduction": [f"{v:.2f}%" for v in df["Feature Reduction (%)"]],
        "Macro FPR": [f"{v * 100:.3f}%" for v in df["Macro FPR"]],
        "Training time": [f"{v:.1f} s" for v in df["Training Time (s)"]],
    })
    st.dataframe(table, hide_index=True)
    metric_cards([("Feature reduction (RF_9)", "87.32%", "71 -> 9 features"),
                  ("Accuracy change", f"{(df['Accuracy'][1] - df['Accuracy'][0]) * 100:+.2f} pts", "RF_9 vs RF_71"),
                  ("Macro F1 change", f"{(df['Macro F1'][1] - df['Macro F1'][0]) * 100:+.2f} pts", "RF_9 vs RF_71"),
                  ("Training time saved", "27.2%", "RF_9 vs RF_71")])
    st.markdown(
        "Using **9 features** reduces the original feature count by approximately **87.32%** (71 -> 9). "
        "This comes with **some reduction in detection performance**: accuracy falls from 97.10% to 96.32% and "
        "Macro F1 from 97.08% to 95.79%. Reducing further to 6 features (91.55% reduction) lowers Macro F1 to 95.08%. "
        "All pairwise differences are statistically significant (bootstrap CIs exclude zero; McNemar p < 0.05), "
        "so the compact models trade a small but real amount of accuracy for a much smaller feature set."
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Model"], y=df["Accuracy"] * 100, name="Accuracy (%)", text=[f"{v * 100:.2f}" for v in df["Accuracy"]]))
    fig.add_trace(go.Bar(x=df["Model"], y=df["Macro F1"] * 100, name="Macro F1 (%)", text=[f"{v * 100:.2f}" for v in df["Macro F1"]]))
    fig.add_trace(go.Scatter(x=df["Model"], y=df["Feature Reduction (%)"], name="Feature reduction (%)", yaxis="y2",
                             mode="lines+markers", line=dict(color="#F59E0B", width=3)))
    fig.update_layout(barmode="group", yaxis=dict(title="%", range=[93, 98]),
                      yaxis2=dict(title="Feature reduction (%)", overlaying="y", side="right", range=[0, 100]))
    st.plotly_chart(style_fig(fig, 400, "Accuracy and Macro F1 vs feature reduction"))

    st.markdown("#### 2. Preliminary optimization experiments (H-MOOA)")
    badge("preliminary", "Small-scale smoke tests (20,000 training / 5,000 validation samples, 2 iterations). "
                         "They validate the implementation and are not comparable to the baseline test results.")
    st.dataframe(pd.DataFrame([
        {"Experiment": "Binary Firefly-PSO feature selection", "Scale": "Population 8, 2 iterations, 24 evaluations",
         "Outcome": "11 non-dominated feature subsets (9-71 features); validation Macro F1 0.897-0.921"},
        {"Experiment": "Cuckoo Search RF hyperparameters", "Scale": "6 nests, 2 iterations, 22 evaluations",
         "Outcome": "4 Pareto configurations; best validation Macro F1 0.8956 (75 trees, depth 25, leaf 1, log2)"},
        {"Experiment": "Adaptive class-wise thresholds", "Scale": "2,500 calibration / 2,500 assessment",
         "Outcome": "No meaningful overall improvement: assessment Macro F1 0.8963 -> 0.8952"},
    ]), hide_index=True)

    st.markdown("#### 3. Key takeaways and limitations")
    info_cards([
        ("Findings", "<ul><li>A 9-feature Random Forest keeps 95.79% Macro F1 with 87.32% fewer features.</li>"
                     "<li>The 9-feature subset is stable: selected exactly in 4 of 5 seeds (mean Jaccard 0.945).</li>"
                     "<li>Time-delta, window-size and IP-flag statistics dominate feature importance.</li></ul>"),
        ("Limitations", "<ul><li>Optimization experiments are preliminary smoke tests.</li>"
                        "<li>Adaptive thresholding is exploratory and gave no meaningful gain.</li>"
                        "<li>Evaluation is offline on a recorded dataset, not real-time detection.</li></ul>"),
    ])
