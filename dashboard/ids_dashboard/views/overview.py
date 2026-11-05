import streamlit as st

from .. import results as R
from ..ui import badge, info_cards, metric_cards, workflow


def render():
    st.subheader("Project Overview")
    st.markdown(f"#### {R.PROJECT_TITLE}")
    badge("actual", "Project facts and dataset statistics come from the research notebook.")

    st.markdown(
        "**Objective.** Build a compact, accurate multiclass intrusion-detection model for IIoT network traffic. "
        "The project uses multi-objective optimization to reduce the 71 original features to a small subset while "
        "keeping detection performance (Macro F1) high and the false-positive rate (Macro FPR) low. Training time "
        "and prediction latency are tracked as computational-cost objectives."
    )

    metric_cards(
        [
            ("Dataset", "CIC DataSenseIoT 2025", "CIC-IIoT-2025 combined_dataset.csv"),
            ("Original features", f"{R.ORIGINAL_FEATURES}", "numeric network + log features"),
            ("Traffic classes", f"{R.N_CLASSES}", ", ".join(R.CLASS_NAMES)),
            ("Records analysed", f"{R.TOTAL_RECORDS:,}", f"held-out test set: {R.TEST_SAMPLES:,}"),
        ]
    )
    st.write("")
    st.markdown("##### Main algorithms")
    info_cards(
        [
            ("Hybrid Firefly-PSO", "<p>Binary PSO moves feature masks toward personal and Pareto-archive leaders; "
             "firefly attraction pulls weaker solutions toward better-ranked ones.</p>"),
            ("Cuckoo Search", "<p>Levy-flight exploration of Random Forest hyperparameters (trees, depth, "
             "min samples per leaf, max features) with nest abandonment and a Pareto archive.</p>"),
            ("Random Forest", "<p>Final classifier: 100 trees, balanced class weights, evaluated on a "
             "held-out test set of 109,771 samples.</p>"),
            ("Adaptive thresholds", "<p>Class-wise probability thresholds tuned by coordinate descent on "
             "calibration data and checked on separate assessment data (exploratory).</p>"),
        ]
    )
    st.write("")
    st.markdown("##### Research methodology")
    workflow(
        [
            "Dataset & cleaning",
            "Leakage-safe split",
            "Firefly-PSO feature selection",
            "Cuckoo Search tuning",
            "Random Forest",
            "Adaptive thresholds",
            "Evaluation",
        ]
    )

    st.markdown("##### How to read this dashboard")
    for kind in ["actual", "preliminary", "live", "illustrative", "synthetic"]:
        badge(kind)
    st.info(
        "Scope: this app classifies **offline CSV datasets** (batch analysis). "
        "It does not capture or monitor live network traffic, so it is not a real-time intrusion detection system."
    )
