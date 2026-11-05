"""Hybrid Optimization-Based Intrusion Detection System - Streamlit entry point.

Run with:  streamlit run app.py
"""

import streamlit as st

from ids_dashboard import results as R
from ids_dashboard.ui import BADGES, header, inject_css
from ids_dashboard.views import (
    classification,
    cuckoo_search,
    dataset,
    feature_selection,
    findings,
    guide,
    overview,
    performance,
    thresholds,
)

st.set_page_config(page_title=R.DASHBOARD_TITLE, page_icon=":material/shield:", layout="wide")

PAGES = {
    "Guided Demonstration": guide,
    "Project Overview": overview,
    "Dataset Analysis": dataset,
    "Feature Selection": feature_selection,
    "Cuckoo Search": cuckoo_search,
    "Random Forest Classification": classification,
    "Adaptive Thresholds": thresholds,
    "Performance Evaluation": performance,
    "Research Findings": findings,
}


def sidebar():
    with st.sidebar:
        st.markdown(f'<div class="sidebar-title">{R.DASHBOARD_TITLE}</div>', unsafe_allow_html=True)
        st.caption("Multi-objective feature selection with hybrid global-local optimization")
        st.session_state.setdefault("nav", "Project Overview")
        page = st.radio("Navigation", list(PAGES), key="nav", label_visibility="collapsed")
        st.divider()
        status = st.empty()
        st.divider()
        with st.expander("Result labels"):
            for label, color, note in BADGES.values():
                st.markdown(f'<span class="badge" style="background:{color}">{label}</span><br>'
                            f'<span class="badge-note">{note}</span>', unsafe_allow_html=True)
    return page, status


def sidebar_status(status):
    with status.container():
        df = st.session_state.get("df")
        if df is None:
            st.caption("Dataset: not loaded")
        else:
            src = "SYNTHETIC demo data" if st.session_state.get("data_source") == "synthetic" else st.session_state.get("data_name")
            st.caption(f"Dataset: {src}\n\n{len(df):,} rows, {df.shape[1]} columns")
            if st.session_state.get("X") is not None:
                st.caption(f"Target: `{st.session_state['target']}` | features: {len(st.session_state.get('features', []))}")
            if st.session_state.get("rf_result"):
                st.caption("Random Forest: trained")


def main():
    inject_css()
    page, status = sidebar()
    header(R.DASHBOARD_TITLE, R.PROJECT_TITLE)
    try:
        PAGES[page].render()
    except Exception as exc:  # noqa: BLE001 - keep the demo running and show what failed
        st.error(f"Something went wrong in '{page}': {exc}")
        st.exception(exc)
    sidebar_status(status)


main()
