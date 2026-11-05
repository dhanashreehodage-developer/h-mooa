"""Session-state helpers shared by the dashboard views."""

import streamlit as st

DOWNSTREAM_KEYS = ["splits", "rf_result", "fpso_result", "cuckoo_result", "threshold_result", "hp_trials", "live_runs"]


def reset_downstream():
    for k in DOWNSTREAM_KEYS:
        st.session_state.pop(k, None)


def dataset():
    return st.session_state.get("df")


def is_synthetic():
    return st.session_state.get("data_source") == "synthetic"


def prepared():
    """Return (X, y) once a target has been configured in the Dataset Analysis module."""
    return st.session_state.get("X"), st.session_state.get("y")


def require_prepared(message="Load a dataset and choose its target column in **Dataset Analysis** first."):
    X, y = prepared()
    if X is None or y is None:
        st.info(message)
        return None, None
    return X, y


def go_to(page):
    st.session_state["nav"] = page
