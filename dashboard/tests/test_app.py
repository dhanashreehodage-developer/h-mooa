from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _frame(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "f1": rng.normal(size=n), "f2": rng.normal(size=n), "f3": rng.normal(size=n),
        "alternate_class": np.where(np.arange(n) % 2, "negative", "positive"),
        "one_class": "only",
    })


def _app_on(page):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["df"] = _frame()
    at.session_state["data_name"] = "fixture.csv"
    at.session_state["data_source"] = "upload"
    at.session_state["nav"] = page
    return at.run()


def _target_box(at):
    return next(s for s in at.selectbox if s.label == "Target-label column")


def test_every_page_renders():
    at = _app_on("Project Overview")
    for page in ["Guided Demonstration", "Dataset Analysis", "Feature Selection", "Cuckoo Search",
                 "Random Forest Classification", "Adaptive Thresholds", "Performance Evaluation", "Research Findings"]:
        at.sidebar.radio(key="nav").set_value(page).run()
        assert not at.exception, page


def test_invalid_target_clears_prepared_data():
    at = _app_on("Dataset Analysis")
    _target_box(at).set_value("alternate_class").run()
    assert at.session_state["X"] is not None
    _target_box(at).set_value("one_class").run()
    assert any("at least two classes" in e.value for e in at.error)
    assert "X" not in at.session_state and "y" not in at.session_state
    at.sidebar.radio(key="nav").set_value("Random Forest Classification").run()
    assert not [b for b in at.button if b.label.startswith("Train and evaluate Random")]
    assert not any("Target:" in c.value for c in at.sidebar.caption)
