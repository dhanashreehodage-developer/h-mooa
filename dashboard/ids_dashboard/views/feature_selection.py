import pandas as pd
import plotly.express as px
import streamlit as st

from .. import results as R
from ..firefly_pso import FireflyPSODemo
from ..modeling import SPLIT_GROUP, three_way_split
from ..state import is_synthetic, require_prepared
from ..ui import badge, info_cards, source, style_fig, workflow


def get_splits(X, y):
    if "splits" not in st.session_state:
        st.session_state["splits"] = three_way_split(X, y, method=SPLIT_GROUP)
    return st.session_state["splits"]


def subsample(X, y, n, seed=0):
    if len(X) <= n:
        return X, y
    idx = X.sample(n, random_state=seed).index
    return X.loc[idx], y.loc[idx]


def _explain():
    workflow(["71 Original Features", "Candidate Feature Subsets", "Hybrid Firefly-PSO Exploration",
              "Pareto Archive", "Selected Feature Subset"])
    info_cards(
        [
            ("PSO-inspired particle movement",
             "<p>Each particle is a binary mask over the 71 features. Its velocity mixes inertia, a pull toward its "
             "own best mask and a pull toward a leader drawn from the Pareto archive. A sigmoid turns velocity into "
             "the probability of selecting each feature.</p>"),
            ("Firefly-inspired attraction",
             "<p>Solutions on a better Pareto front are 'brighter'. A weaker particle is attracted toward a brighter "
             "one with strength beta0 * exp(-gamma * d^2), where d is the Hamming distance between masks.</p>"),
            ("Candidate evaluation",
             "<p>Every mask trains a Random Forest on the fitting split and is scored on validation data with four "
             "minimised objectives: detection error (1 - Macro F1), Macro FPR, feature ratio, prediction latency.</p>"),
            ("Pareto-based preservation",
             "<p>Non-dominated solutions go into a bounded archive. Crowding distance keeps the archive diverse "
             "when it is full, so good trade-offs are never lost between iterations.</p>"),
        ],
        columns=2,
    )
    with st.expander("Update equations"):
        st.latex(r"v_i \leftarrow w\,v_i + c_1 r_1 (p_i - x_i) + c_2 r_2 (g - x_i) + \beta_0 e^{-\gamma d_{ij}^2}(x_j - x_i)")
        st.latex(r"x_{i,k} = 1 \;\text{ if } \; u < \sigma(v_{i,k}) = \frac{1}{1+e^{-v_{i,k}}}, \text{ else } 0")
        st.latex(r"\min \; \big(1-\text{MacroF1},\; \text{MacroFPR},\; |S|/71,\; \text{latency}\big)")


def _actual():
    badge("preliminary", "Measured outputs of the notebook's Firefly-PSO smoke test (20,000 training / 5,000 "
                         "validation samples, population 8, 2 iterations). Not final model performance.")
    c1, c2 = st.columns([1, 2])
    c1.dataframe(pd.DataFrame(R.FPSO_CONFIG.items(), columns=["Setting", "Value"]).astype(str), hide_index=True, height=420)
    with c2:
        st.dataframe(R.FPSO_ITERATIONS, hide_index=True)
        st.caption(f"Initial population feature counts: {R.FPSO_INITIAL_COUNTS}")
        arch = R.FPSO_ARCHIVE.copy()
        arch["Label"] = arch["Candidate"].map(lambda c: "RF_71 (all)" if c == 0 else ("RF_9 subset" if c == 1 else f"C{c}"))
        fig = px.scatter(arch, x="Feature Count", y="Macro F1", color="Macro FPR", text="Label", size_max=14,
                         color_continuous_scale="Blues_r", hover_data=["Feature Reduction (%)", "Latency (ms / 1,000 samples)"])
        fig.update_traces(marker=dict(size=13, line=dict(width=1, color="white")), textposition="top center")
        st.plotly_chart(style_fig(fig, 380, "Final Pareto archive (all 11 candidates are non-dominated)"))
    st.dataframe(R.FPSO_ARCHIVE, hide_index=True)
    source(R.SOURCE_FPSO)

    st.markdown("##### Selected feature subsets")
    badge("actual", "Feature lists of the final models evaluated on the held-out test set.")
    st.markdown(
        "RF_9 came from the earlier multi-objective feature-selection phase (GA-based search followed by an "
        "importance-guided local search) and was confirmed by the five-seed robustness study. RF_6 is the result "
        "of further iterative reduction. The 9-feature candidate in the Firefly-PSO archive is this same RF_9 subset."
    )
    c1, c2 = st.columns(2)
    c1.markdown("**RF_9 (87.32% reduction)**")
    c1.dataframe(pd.DataFrame({"Feature": R.RF_9_FEATURES}), hide_index=True)
    c2.markdown("**RF_6 (91.55% reduction)**")
    c2.dataframe(pd.DataFrame({"Feature": R.RF_6_FEATURES}), hide_index=True)

    st.markdown("##### Five-seed stability of the selected subset")
    fig = px.bar(R.FEATURE_FREQUENCY, x="Selection Frequency (%)", y="Feature", orientation="h",
                 color="Selection Frequency (%)", color_continuous_scale="Blues")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style_fig(fig, 380, "Feature selection frequency across 5 seeds"))
    st.dataframe(R.FIVE_SEED_BEST, hide_index=True)
    st.caption("Exact match with RF_9 in 4 of 5 seeds; mean Jaccard similarity 0.945. Macro F1 here is validation "
               "performance on the development sample, not held-out test performance.")
    source(R.SOURCE_SEEDS)


def _live():
    X, y = require_prepared()
    if X is None:
        return
    badge("illustrative", "Simplified Firefly-PSO run on your data: small population, few iterations, three "
                          "objectives (no latency), small Random Forests. Not the original experiment.")
    if is_synthetic():
        badge("synthetic")
    c1, c2, c3, c4 = st.columns(4)
    pop = c1.slider("Population", 4, 12, 6)
    iters = c2.slider("Iterations", 1, 6, 3)
    trees = c3.slider("RF trees per evaluation", 10, 80, 30, 10)
    n_fit = c4.select_slider("Fitting rows", [2_000, 5_000, 10_000, 20_000], value=5_000)
    min_feat = st.slider("Minimum features per subset", 1, max(1, min(10, X.shape[1] - 1)), min(3, X.shape[1] - 1))
    st.caption(f"Up to {pop * (iters + 1)} Random Forest evaluations. Uses training and validation splits only; "
               "the test split is never touched.")
    if st.button("Run Firefly-PSO demonstration", type="primary"):
        try:
            sp = get_splits(X, y)
            Xf, yf = subsample(sp["X_train"], sp["y_train"], n_fit)
            Xv, yv = subsample(sp["X_val"], sp["y_val"], max(1_000, n_fit // 4), seed=1)
            bar = st.progress(0.0, text="Searching...")
            demo = FireflyPSODemo(Xf, yf, Xv, yv, population=pop, iterations=iters, trees=trees, min_features=min_feat)
            st.session_state["fpso_result"] = demo.run(progress=lambda p: bar.progress(p, text="Searching..."))
            bar.empty()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Feature-selection demo failed: {exc}")
    res = st.session_state.get("fpso_result")
    if not res:
        return
    hist = pd.DataFrame([{"Iteration": h["iteration"], "Archive size": h["archive_size"],
                          "Unique evaluations": h["unique_evaluations"],
                          "Mean population features": sum(h["population_counts"]) / len(h["population_counts"]),
                          "Best population Macro F1": max(h["population_f1"])} for h in res["history"]])
    evals = pd.DataFrame([{"Features": e["feature_count"], "Macro F1": e["macro_f1"], "Macro FPR": e["macro_fpr"],
                           "In archive": any(e["key"] == a["key"] for a in res["archive"])} for e in res["evaluations"]])
    c1, c2 = st.columns(2)
    fig = px.scatter(evals, x="Features", y="Macro F1", color="In archive", hover_data=["Macro FPR"],
                     color_discrete_map={True: "#00B4FF", False: "#64748B"})
    c1.plotly_chart(style_fig(fig, 360, "All evaluated subsets (validation)"))
    fig = px.line(hist, x="Iteration", y=["Archive size", "Mean population features"], markers=True)
    c2.plotly_chart(style_fig(fig, 360, "Search progress"))
    arch = sorted(res["archive"], key=lambda a: (-a["macro_f1"], a["feature_count"]))
    table = pd.DataFrame([{"Solution": i, "Features": a["feature_count"], "Macro F1": round(a["macro_f1"], 4),
                           "Macro FPR": round(a["macro_fpr"], 5), "Selected features": ", ".join(a["features"])}
                          for i, a in enumerate(arch)])
    st.markdown("**Pareto archive (demo)**")
    st.dataframe(table, hide_index=True)
    pick = st.selectbox("Use an archive solution in the Random Forest module", table["Solution"],
                        format_func=lambda i: f"Solution {i}: {arch[i]['feature_count']} features, F1 {arch[i]['macro_f1']:.4f}")
    if st.button("Use this feature subset"):
        st.session_state["demo_features"] = arch[pick]["features"]
        st.success("Saved. Choose 'Firefly-PSO demo subset' in Random Forest Classification.")


def render():
    st.subheader("Feature Selection: Hybrid Firefly-PSO")
    _explain()
    t1, t2 = st.tabs(["Research outputs", "Live demonstration on loaded data"])
    with t1:
        _actual()
    with t2:
        _live()
