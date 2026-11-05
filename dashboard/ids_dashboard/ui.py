"""Shared styling and layout helpers (dark-blue cybersecurity theme)."""

import html

import plotly.graph_objects as go
import streamlit as st

ACCENT = "#00B4FF"
PALETTE = ["#00B4FF", "#22D3EE", "#A78BFA", "#F59E0B", "#34D399", "#F472B6", "#60A5FA", "#FB7185"]

BADGES = {
    "actual": ("ACTUAL RESULT", "#16A34A", "Measured values from the saved outputs of the research notebook."),
    "preliminary": ("PRELIMINARY EXPERIMENT", "#D97706", "Real measured outputs from small-scale (smoke-test) optimization runs; not final results."),
    "live": ("LIVE COMPUTATION", "#0284C7", "Computed now, in this app, on the dataset you loaded."),
    "illustrative": ("ILLUSTRATIVE DEMO", "#7C3AED", "Simplified demonstration of the method; not the original experiment and not a research result."),
    "synthetic": ("SYNTHETIC DATA", "#DB2777", "Randomly generated demo data; not CIC-IIoT-2025."),
}

CSS = f"""
<style>
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3 {{ letter-spacing: 0.2px; }}
.app-header {{
    background: linear-gradient(120deg, #0B1E3F 0%, #0F2D5C 55%, #0A4C7A 100%);
    border: 1px solid #1E3A66; border-radius: 14px; padding: 18px 24px; margin-bottom: 18px;
    box-shadow: 0 0 24px rgba(0,180,255,0.10);
}}
.app-header h1 {{ margin: 0; font-size: 1.75rem; color: #E6F1FF; }}
.app-header p {{ margin: 4px 0 0 0; color: #9CC3E6; font-size: 0.95rem; }}
.metric-card {{
    background: linear-gradient(160deg, #10264A 0%, #0D1E3A 100%);
    border: 1px solid #1E3A66; border-left: 4px solid {ACCENT}; border-radius: 12px;
    padding: 14px 16px; height: 100%; min-height: 96px;
}}
.metric-card .label {{ color: #8FB3D9; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.6px; }}
.metric-card .value {{ color: #E6F1FF; font-size: 1.55rem; font-weight: 700; margin-top: 2px; }}
.metric-card .sub {{ color: #7FA6CC; font-size: 0.78rem; margin-top: 2px; }}
.badge {{
    display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.72rem;
    font-weight: 700; letter-spacing: 0.6px; color: white; margin: 2px 6px 6px 0;
}}
.badge-note {{ color: #9CC3E6; font-size: 0.82rem; }}
.flow {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 8px 0 14px 0; }}
.flow .step {{
    background: #10264A; border: 1px solid #1E3A66; border-radius: 10px; padding: 10px 14px;
    color: #E6F1FF; font-size: 0.88rem; text-align: center; min-width: 120px;
}}
.flow .step.active {{ border-color: {ACCENT}; box-shadow: 0 0 12px rgba(0,180,255,0.35); }}
.flow .step .num {{ color: {ACCENT}; font-weight: 700; font-size: 0.75rem; display: block; }}
.flow .arrow {{ color: {ACCENT}; font-size: 1.2rem; }}
.info-card {{
    background: #0F2242; border: 1px solid #1E3A66; border-radius: 12px; padding: 14px 18px; height: 100%;
}}
.info-card h4 {{ margin: 0 0 6px 0; color: {ACCENT}; font-size: 1rem; }}
.info-card p, .info-card li {{ color: #C9DDF2; font-size: 0.9rem; }}
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #081429 0%, #0B1E3F 100%); }}
section[data-testid="stSidebar"] .sidebar-title {{ color: #E6F1FF; font-weight: 700; font-size: 1.05rem; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def header(title, subtitle=""):
    st.markdown(
        f'<div class="app-header"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def badge(kind, note=None):
    label, color, default_note = BADGES[kind]
    text = default_note if note is None else note
    st.markdown(
        f'<span class="badge" style="background:{color}">{label}</span>'
        f'<span class="badge-note">{html.escape(text)}</span>',
        unsafe_allow_html=True,
    )


def metric_cards(items, columns=None):
    """items: list of (label, value, sub) tuples."""
    cols = st.columns(columns or len(items))
    for i, item in enumerate(items):
        label, value, sub = (list(item) + [""])[:3]
        with cols[i % len(cols)]:
            st.markdown(
                f'<div class="metric-card"><div class="label">{html.escape(str(label))}</div>'
                f'<div class="value">{html.escape(str(value))}</div>'
                f'<div class="sub">{html.escape(str(sub))}</div></div>',
                unsafe_allow_html=True,
            )


def info_cards(items, columns=None):
    """items: list of (title, body_html)."""
    cols = st.columns(columns or len(items))
    for i, (title, body) in enumerate(items):
        with cols[i % len(cols)]:
            st.markdown(f'<div class="info-card"><h4>{html.escape(title)}</h4>{body}</div>', unsafe_allow_html=True)


def workflow(steps, active=None):
    parts = []
    for i, s in enumerate(steps):
        cls = "step active" if active is not None and i == active else "step"
        parts.append(f'<div class="{cls}"><span class="num">STEP {i + 1}</span>{html.escape(s)}</div>')
        if i < len(steps) - 1:
            parts.append('<div class="arrow">&#10140;</div>')
    st.markdown(f'<div class="flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def source(text):
    st.caption(f"Source: {text}")


def style_fig(fig: go.Figure, height=420, title=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(16,38,74,0.35)",
        height=height,
        margin=dict(l=20, r=20, t=50 if title else 20, b=20),
        font=dict(color="#C9DDF2"),
        colorway=PALETTE,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        title=title,
    )
    return fig


def pct(x, digits=2):
    return f"{x * 100:.{digits}f}%"
