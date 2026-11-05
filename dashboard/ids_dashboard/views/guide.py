import streamlit as st

from ..state import go_to
from ..ui import workflow

STEPS = [
    ("Load the dataset", "Dataset Analysis", lambda s: s.get("df") is not None),
    ("Explore the traffic classes", "Dataset Analysis", lambda s: s.get("y") is not None),
    ("Examine the original features", "Dataset Analysis", lambda s: s.get("X") is not None),
    ("Demonstrate feature selection", "Feature Selection", lambda s: s.get("fpso_result") is not None),
    ("Explore Cuckoo Search hyperparameters", "Cuckoo Search", lambda s: bool(s.get("hp_trials")) or s.get("cuckoo_result") is not None),
    ("Train the Random Forest classifier", "Random Forest Classification", lambda s: s.get("rf_result") is not None),
    ("Demonstrate adaptive thresholding", "Adaptive Thresholds", lambda s: s.get("threshold_result") is not None),
    ("Evaluate the classification performance", "Performance Evaluation", lambda s: s.get("rf_result") is not None),
    ("Display the final findings", "Research Findings", lambda s: False),
]


def render():
    st.subheader("Guided Demonstration")
    st.markdown(
        "Follow these nine steps during a presentation. Each button opens the relevant module; "
        "a step is ticked once its result exists in this session."
    )
    done = [check(st.session_state) for _, _, check in STEPS]
    next_idx = next((i for i, d in enumerate(done) if not d), len(STEPS) - 1)
    workflow([s for s, _, _ in STEPS], active=next_idx)
    st.progress(sum(done) / len(STEPS), text=f"{sum(done)} of {len(STEPS)} steps completed")

    for i, (title, page, _) in enumerate(STEPS):
        c1, c2, c3 = st.columns([0.08, 0.62, 0.30])
        c1.markdown(f"**{'✓' if done[i] else i + 1}**")
        c2.markdown(f"**{title}**  \nModule: *{page}*")
        c3.button(f"Open {page}", key=f"guide_{i}", on_click=go_to, args=(page,))
    st.caption("Tip: no dataset at hand? In Dataset Analysis you can load a clearly labelled synthetic demo dataset.")
