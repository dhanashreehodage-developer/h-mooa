import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .. import results as R
from ..cuckoo import MAX_FEATURES_OPTIONS, CuckooSearchDemo, levy_steps
from ..modeling import build_rf, fit_and_evaluate
from ..state import is_synthetic, require_prepared
from ..ui import badge, info_cards, source, style_fig, workflow
from .feature_selection import get_splits, subsample


def _explain():
    workflow(["Candidate nests", "Levy flight search", "Candidate evaluation", "Nest updates", "Pareto archive"])
    info_cards(
        [
            ("Candidate nests", "<p>Each nest is one Random Forest configuration: number of trees, maximum depth, "
                                "minimum samples per leaf and maximum features.</p>"),
            ("Levy flight search", "<p>New eggs are generated with heavy-tailed Levy steps: mostly small local moves "
                                   "with occasional long jumps that escape local optima.</p>"),
            ("Candidate evaluation", "<p>Each configuration trains a Random Forest on the RF_9 features and is scored "
                                     "on validation data (Macro F1, Macro FPR, latency).</p>"),
            ("Nest updates", "<p>A better egg replaces a randomly chosen nest; a fraction pa of the worst nests is "
                             "abandoned and rebuilt randomly. A Pareto archive keeps non-dominated settings.</p>"),
        ],
        columns=2,
    )
    with st.expander("Levy flight (Mantegna's algorithm)"):
        st.latex(r"s = \frac{u}{|v|^{1/\beta}},\quad u \sim N(0,\sigma_u^2),\; v \sim N(0,1)")
        st.latex(r"\sigma_u = \left[\frac{\Gamma(1+\beta)\sin(\pi\beta/2)}{\Gamma\!\left(\frac{1+\beta}{2}\right)\beta\,2^{(\beta-1)/2}}\right]^{1/\beta}")
        badge("illustrative", "Random walk generated now to show the shape of Levy flights; not experimental data.")
        c1, c2 = st.columns([1, 3])
        beta = c1.slider("beta", 1.1, 1.9, 1.5, 0.1)
        steps = c1.slider("Steps", 50, 500, 200, 50)
        rng = np.random.default_rng(3)
        levy = np.cumsum(np.column_stack([levy_steps(steps, beta, rng), levy_steps(steps, beta, rng)]) * 0.1, axis=0)
        gauss = np.cumsum(rng.normal(0, 0.1, (steps, 2)), axis=0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=levy[:, 0], y=levy[:, 1], mode="lines+markers", name="Levy flight", marker=dict(size=3)))
        fig.add_trace(go.Scatter(x=gauss[:, 0], y=gauss[:, 1], mode="lines", name="Gaussian random walk"))
        c2.plotly_chart(style_fig(fig, 360, "Levy flight vs Gaussian walk"))


def _actual():
    badge("preliminary", "Measured outputs of the notebook's Cuckoo Search smoke test (RF_9 features, 20,000 "
                         "training / 5,000 validation samples, 6 nests, 2 iterations). Not final model performance.")
    c1, c2 = st.columns([1, 2])
    c1.dataframe(pd.DataFrame(R.CUCKOO_CONFIG.items(), columns=["Setting", "Value"]).astype(str), hide_index=True)
    ev = R.CUCKOO_EVALUATIONS.copy()
    ev["Pareto archive"] = ev["Evaluation"].isin(R.CUCKOO_ARCHIVE_IDS)
    fig = px.scatter(ev, x="Macro FPR", y="Macro F1", color="Stage", symbol="Pareto archive", size="n_estimators",
                     hover_data=["n_estimators", "max_depth", "min_samples_leaf", "max_features"])
    c2.plotly_chart(style_fig(fig, 380, "All 22 hyperparameter evaluations (validation)"))
    st.markdown("**Final Pareto archive (4 configurations)**")
    st.dataframe(ev[ev["Pareto archive"]].drop(columns=["Pareto archive"]), hide_index=True)
    with st.expander("All evaluations"):
        st.dataframe(ev, hide_index=True)
    st.caption("Highest validation Macro F1 in the archive: 75 trees, max_depth 25, min_samples_leaf 1, max_features "
               "log2 (0.8956). The final test models used n_estimators=100 with balanced class weights.")
    source(R.SOURCE_CUCKOO)


def _explorer(X, y):
    st.markdown("##### Random Forest hyperparameter explorer")
    badge("live", "Trains one Random Forest on the training split and scores it on the validation split.")
    if is_synthetic():
        badge("synthetic")
    c1, c2, c3, c4, c5 = st.columns(5)
    n_est = c1.slider("Number of trees", 10, 300, 100, 10)
    depth = c2.slider("Maximum depth (0 = unlimited)", 0, 50, 0)
    leaf = c3.slider("Min samples per leaf", 1, 20, 1)
    mf = c4.selectbox("Maximum features", MAX_FEATURES_OPTIONS, format_func=str)
    n_fit = c5.select_slider("Training rows", [5_000, 10_000, 20_000, 50_000], value=10_000)
    if st.button("Train and evaluate this configuration", type="primary"):
        try:
            sp = get_splits(X, y)
            Xf, yf = subsample(sp["X_train"], sp["y_train"], n_fit)
            labels = sorted(y.unique())
            with st.spinner("Training..."):
                out = fit_and_evaluate(build_rf(n_est, depth, leaf, mf), Xf, yf, sp["X_val"], sp["y_val"], labels)
            row = {"n_estimators": n_est, "max_depth": depth or "None", "min_samples_leaf": leaf,
                   "max_features": str(mf), **{k: out["metrics"][k] for k in
                                               ["Macro F1", "Macro FPR", "Accuracy", "Training Time (s)",
                                                "Latency (ms / 1,000 samples)"]}}
            st.session_state.setdefault("hp_trials", []).append(row)
            st.session_state["rf_params"] = {"n_estimators": n_est, "max_depth": depth, "min_samples_leaf": leaf,
                                             "max_features": mf}
        except Exception as exc:  # noqa: BLE001
            st.error(f"Training failed: {exc}")
    trials = st.session_state.get("hp_trials")
    if trials:
        df = pd.DataFrame(trials)
        df.insert(0, "Trial", range(1, len(df) + 1))
        st.dataframe(df.round(5), hide_index=True)
        fig = px.scatter(df, x="Macro FPR", y="Macro F1", text="Trial", size="Training Time (s)",
                         hover_data=["n_estimators", "max_depth", "min_samples_leaf", "max_features"])
        fig.update_traces(textposition="top center")
        st.plotly_chart(style_fig(fig, 340, "Your trials (validation)"))
        st.caption("The latest configuration is pre-filled in the Random Forest Classification module.")


def _demo_search(X, y):
    st.markdown("##### Simplified Cuckoo Search on loaded data")
    badge("illustrative", "Reduced Cuckoo Search (few nests and iterations, two objectives). It is a demonstration, "
                          "not the complete original implementation.")
    c1, c2, c3, c4 = st.columns(4)
    nests = c1.slider("Nests", 3, 8, 4)
    iters = c2.slider("Iterations", 1, 4, 2)
    pa = c3.slider("Abandonment rate pa", 0.1, 0.5, 0.25, 0.05)
    n_fit = c4.select_slider("Fitting rows ", [2_000, 5_000, 10_000], value=5_000)
    t_lo, t_hi = st.slider("Number of trees range", 10, 300, (30, 120), 10)
    d_lo, d_hi = st.slider("Maximum depth range", 2, 50, (5, 30))
    l_lo, l_hi = st.slider("Min samples per leaf range", 1, 20, (1, 8))
    if st.button("Run Cuckoo Search demonstration"):
        try:
            sp = get_splits(X, y)
            Xf, yf = subsample(sp["X_train"], sp["y_train"], n_fit)
            Xv, yv = subsample(sp["X_val"], sp["y_val"], max(1_000, n_fit // 4), seed=1)
            bounds = {"n_estimators": (t_lo, t_hi), "max_depth": (d_lo, d_hi), "min_samples_leaf": (l_lo, l_hi)}
            bar = st.progress(0.0, text="Searching...")
            demo = CuckooSearchDemo(Xf, yf, Xv, yv, bounds, nests=nests, iterations=iters, pa=pa)
            st.session_state["cuckoo_result"] = demo.run(progress=lambda p: bar.progress(p, text="Searching..."))
            bar.empty()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Cuckoo Search demo failed: {exc}")
    res = st.session_state.get("cuckoo_result")
    if not res:
        return
    keys = {a["key"] for a in res["archive"]}
    df = pd.DataFrame([{**r["params"], "max_features": str(r["params"]["max_features"]), "Stage": r["stage"],
                        "Macro F1": r["macro_f1"], "Macro FPR": r["macro_fpr"], "Pareto archive": r["key"] in keys}
                       for r in res["evaluations"]])
    fig = px.scatter(df, x="Macro FPR", y="Macro F1", color="Stage", symbol="Pareto archive",
                     hover_data=["n_estimators", "max_depth", "min_samples_leaf", "max_features"])
    st.plotly_chart(style_fig(fig, 360, "Demo evaluations (validation)"))
    st.dataframe(df.round(5), hide_index=True)
    for line in res["log"]:
        st.caption(line)
    best = max(res["archive"], key=lambda a: a["macro_f1"])
    if st.button(f"Use best demo configuration {best['params']} in the Random Forest module"):
        st.session_state["rf_params"] = dict(best["params"])
        st.success("Saved.")


def render():
    st.subheader("Cuckoo Search Optimization")
    _explain()
    t1, t2 = st.tabs(["Research outputs", "Interactive exploration on loaded data"])
    with t1:
        _actual()
    with t2:
        X, y = require_prepared()
        if X is not None:
            _explorer(X, y)
            st.divider()
            _demo_search(X, y)
