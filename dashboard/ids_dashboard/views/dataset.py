import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

from .. import data_utils as D
from .. import results as R
from ..state import dataset, is_synthetic, reset_downstream
from ..ui import badge, metric_cards, style_fig


def _load_panel():
    with st.expander("Load dataset", expanded=dataset() is None):
        c1, c2 = st.columns(2)
        max_rows = c1.number_input("Maximum rows to load", 1_000, 5_000_000, 200_000, step=10_000,
                                   help="Large files are read in chunks so memory stays bounded.")
        frac = c2.slider("Sample fraction per chunk", 0.01, 1.0, 1.0, 0.01,
                         help="Use < 1.0 to take a random sample across the whole file.")
        tab_up, tab_path, tab_syn = st.tabs(["Upload CSV", "Local file path", "Synthetic demo data"])
        with tab_up:
            up = st.file_uploader("CSV dataset", type=["csv"])
            if st.button("Load uploaded file", type="primary", disabled=up is None):
                _load(lambda: D.read_uploaded(up, int(max_rows), frac), up.name, "uploaded")
        with tab_path:
            path = st.text_input("Path to a CSV on this computer",
                                 placeholder=r"C:\data\cic-iiot-2025\combined_dataset.csv")
            st.caption("Recommended for the full multi-GB dataset: no upload needed, rows are streamed from disk.")
            if st.button("Load from path", disabled=not path):
                _load(lambda: D.read_csv_sampled(path.strip().strip('"'), int(max_rows), frac), path, "path")
        with tab_syn:
            st.warning("Synthetic data is randomly generated for trying the app. It is NOT CIC-IIoT-2025 data "
                       "and any metric computed on it is meaningless for the research.")
            n = st.slider("Synthetic rows", 2_000, 30_000, 6_000, 1_000)
            if st.button("Generate synthetic demo dataset"):
                _load(lambda: D.make_synthetic_demo(n), "synthetic demo", "synthetic")


def _load(fn, name, source):
    try:
        with st.spinner("Loading dataset..."):
            df = fn()
    except D.DatasetError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001 - surface any unexpected parsing problem to the user
        st.error(f"Unexpected error while loading the dataset: {exc}")
        return
    reset_downstream()
    for k in ("X", "y", "target", "features"):
        st.session_state.pop(k, None)
    st.session_state.update(df=df, data_name=name, data_source=source)
    st.success(f"Loaded {len(df):,} rows x {df.shape[1]} columns from {name}.")


def _configure_target(df):
    st.markdown("##### Target label and leakage control")
    candidates = D.target_candidates(df)
    if not candidates:
        st.warning("No column looks like a class label (2-50 distinct values). Pick the target manually.")
    options = candidates + [c for c in df.columns if c not in candidates]
    prev = st.session_state.get("target")
    idx = options.index(prev) if prev in options else 0
    c1, c2 = st.columns([1, 2])
    target = c1.selectbox("Target-label column", options, index=idx,
                          help="Detected from the uploaded data; candidates are ranked first. The name is never assumed.")
    nunique = df[target].nunique(dropna=True)
    if nunique > D.MAX_TARGET_CLASSES:
        c1.error(f"'{target}' has {nunique:,} distinct values; choose a categorical label column.")
        return
    default_ex = D.default_excluded_columns(df, target)
    excluded = c2.multiselect(
        "Excluded columns (labels, identifiers, timestamps)", [c for c in df.columns if c != target],
        default=default_ex,
        help="Other label columns and identifiers would leak the answer into the model, so they are excluded.",
    )
    features = D.numeric_feature_columns(df, target, excluded)
    try:
        X, y = D.prepare_xy(df, target, features)
    except D.DatasetError as exc:
        st.error(str(exc))
        return
    signature = (target, tuple(X.columns), len(df))
    if st.session_state.get("signature") != signature:
        reset_downstream()
        st.session_state["signature"] = signature
    st.session_state.update(target=target, features=list(X.columns), X=X, y=y)
    st.caption(f"Model features: {X.shape[1]} numeric columns. Non-numeric columns are not used as features.")


def render():
    st.subheader("Dataset Analysis")
    _load_panel()
    df = dataset()
    if df is None:
        st.info("Upload a CSV, point to a local file, or generate synthetic demo data to begin.")
        return
    if is_synthetic():
        badge("synthetic")
    else:
        badge("live", f"Analysis of '{st.session_state.get('data_name')}'.")

    metric_cards(
        [
            ("Rows", f"{len(df):,}", "loaded"),
            ("Columns", f"{df.shape[1]}", "including labels/metadata"),
            ("Missing cells", f"{int(df.isna().sum().sum()):,}", f"{df.isna().mean().mean() * 100:.2f}% of all cells"),
            ("Duplicate rows", f"{int(df.duplicated().sum()):,}", "exact duplicates"),
        ]
    )
    st.markdown("##### Preview")
    st.dataframe(df.head(50), height=260)

    _configure_target(df)
    X, y = st.session_state.get("X"), st.session_state.get("y")

    t_cls, t_miss, t_feat, t_corr = st.tabs(
        ["Class distribution", "Missing values", "Feature statistics", "Correlation heatmap"]
    )
    with t_miss:
        table = D.missing_value_table(df)
        st.dataframe(table, height=320)
        top = table[table["Missing"] > 0].head(25)
        if top.empty:
            st.success("No missing values in the loaded rows.")
        else:
            fig = px.bar(top, x=top.index, y="Missing (%)", labels={"x": "Column"})
            st.plotly_chart(style_fig(fig, 360, "Columns with missing values"))

    if y is None:
        return

    with t_cls:
        counts = y.value_counts()
        dist = pd.DataFrame({"Class": counts.index, "Count": counts.values,
                             "Percentage (%)": counts.values / counts.sum() * 100})
        metric_cards(
            [
                ("Classes", f"{len(counts)}", "in the selected target"),
                ("Largest class", str(counts.index[0]), f"{counts.iloc[0]:,} rows"),
                ("Smallest class", str(counts.index[-1]), f"{counts.iloc[-1]:,} rows"),
                ("Imbalance ratio", f"{counts.iloc[0] / counts.iloc[-1]:.1f}:1", "largest / smallest"),
            ]
        )
        if len(counts) != R.N_CLASSES and not is_synthetic():
            st.warning(f"The selected target has {len(counts)} classes; the research dataset has {R.N_CLASSES}.")
        c1, c2 = st.columns([3, 2])
        fig = px.bar(dist, x="Class", y="Count", color="Class", text="Count")
        c1.plotly_chart(style_fig(fig, 380, "Class counts"))
        fig = px.pie(dist, names="Class", values="Count", hole=0.45)
        c2.plotly_chart(style_fig(fig, 380, "Class share"))
        st.dataframe(dist.round(2), hide_index=True)
        with st.expander("Reference: class distribution of the full research dataset (actual)"):
            badge("actual")
            st.dataframe(R.CLASS_PERCENTAGES, hide_index=True)
            st.caption(f"Total records: {R.TOTAL_RECORDS:,}. Largest: {R.LARGEST_CLASS[0]} ({R.LARGEST_CLASS[1]:,}); "
                       f"smallest: {R.SMALLEST_CLASS[0]} ({R.SMALLEST_CLASS[1]:,}); ratio {R.IMBALANCE_RATIO}:1. "
                       f"Source: {R.SOURCE_EDA}")

    with t_feat:
        st.markdown(f"**{X.shape[1]} numeric features** available for modelling.")
        present = [f for f in R.RF_71_FEATURES if f in X.columns]
        st.caption(f"{len(present)} of the 71 original research features are present in this dataset.")
        st.dataframe(X.describe().T.round(4), height=360)
        feat = st.selectbox("Inspect a feature by class", list(X.columns))
        fig = px.box(pd.DataFrame({feat: X[feat], "Class": y}), x="Class", y=feat, color="Class", points=False)
        st.plotly_chart(style_fig(fig, 380, f"{feat} by class"))

    with t_corr:
        variances = X.var(numeric_only=True).sort_values(ascending=False)
        default = list(variances.index[: min(15, len(variances))])
        chosen = st.multiselect("Features (max 40)", list(X.columns), default=default, max_selections=40)
        method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
        if len(chosen) < 2:
            st.info("Select at least two features.")
        else:
            sample = X[chosen].sample(min(len(X), 50_000), random_state=42)
            corr = sample.corr(method=method)
            fig, ax = plt.subplots(figsize=(min(1 + 0.55 * len(chosen), 18), min(0.45 * len(chosen) + 1, 14)))
            fig.patch.set_facecolor("#0A1628")
            ax.set_facecolor("#0A1628")
            sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax,
                        annot=len(chosen) <= 15, fmt=".2f", annot_kws={"size": 7},
                        cbar_kws={"shrink": 0.7})
            ax.tick_params(colors="#C9DDF2", labelsize=8)
            plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
            ax.set_title(f"{method.title()} correlation", color="#E6F1FF")
            st.pyplot(fig)
            plt.close(fig)
            upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
            strong = upper[upper.abs() >= 0.9].sort_values(key=np.abs, ascending=False)
            if len(strong):
                st.caption(f"{len(strong)} feature pairs with |r| >= 0.9 (redundancy that feature selection can remove).")
