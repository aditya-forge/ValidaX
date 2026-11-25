"""Console helpers and a headless smoke test of the pages that need only outputs/."""
import hashlib
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

REPO = Path(__file__).resolve().parents[1]
PROTECTED = [REPO / "app" / "model.pkl", *sorted((REPO / "outputs").glob("*"))]


def test_masked_reasons_are_reduced_to_family_labels(console):
    assert console.safe_reason("masked counting feature (C14) (3.0) increased risk") == \
        "masked counting feature (C14) increased risk"
    assert console.safe_reason("V70 (0.97) increased risk") == "masked Vesta-engineered feature (V70) increased risk"
    assert console.safe_reason("hour of day (9) increased risk").startswith("hour of the relative 24-hour")


def test_first_seen_sentinel_is_spelled_out(console):
    assert console.humanize_seconds(-1) == console.NO_PRIOR_TXN == "No prior transaction (first seen)"
    assert console.humanize_seconds(3600 * 30) == "1.2 days"
    reason = console.investigation_reason("amount_vs_card_avg_ratio", 0.3, 300.0, first_seen=True)
    assert "not meaningful" in reason and "300" not in reason


def test_segment_view_hides_placeholders(console):
    table = pd.DataFrame({"card_is_first_seen": [1.0, 0.0], "seconds_since_card_last_txn": [-1.0, 7200.0],
                          "amount_vs_card_avg_ratio": [179.8, 1.36]})
    view = console.segment_view(table)
    assert view["seconds_since_card_last_txn"].tolist() == [console.NO_PRIOR_TXN, "2.0 h"]
    assert view["amount_vs_card_avg_ratio"].iloc[0].startswith("n/a")


def test_classification_table_counts(console):
    scored = pd.DataFrame({"fraud_probability": [0.9, 0.5, 0.2, 0.1], "isFraud": [1, 0, 1, 0]})
    m = console.cost_at_threshold(scored, 0.4)
    table = console.classification_table(m)
    assert table["Flagged for review"].tolist() == [1, 1]
    assert table["Not flagged"].tolist() == [1, 1]
    assert m["estimated_cost"] == 500 + 10


PAGE_SCRIPT = """
import importlib.util
import streamlit as st
spec = importlib.util.spec_from_file_location("console", r"{app}")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
st.html(app.STYLE)
app.{render}()
"""


def run_page(render: str) -> AppTest:
    script = PAGE_SCRIPT.format(app=REPO / "app" / "app.py", render=render)
    return AppTest.from_string(script, default_timeout=300).run()


def test_app_boots_with_top_navigation():
    """The full script: brand header + st.navigation(position='top') + default Overview page."""
    at = AppTest.from_file(str(REPO / "app" / "app.py"), default_timeout=300).run()
    assert not at.exception, [e.message for e in at.exception]
    bodies = [h.proto.body for h in at.get("html")]
    assert any("vx-status" in b and "Threshold 0.24" in b for b in bodies)      # top-right status badge
    assert any("vx-strip" in b and "&#36;566,840" in b for b in bodies)          # telemetry strip


@pytest.mark.skipif(not (REPO / "outputs" / "scored_transactions.csv").exists(), reason="outputs/ not present")
@pytest.mark.parametrize("render", ["render_overview", "render_classify", "render_alerts",
                                    "render_investigate", "render_simulator"])
def test_pages_render_without_errors(render):
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in PROTECTED}
    at = run_page(render)
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert any("not a determination of fraud" in h.proto.body for h in at.get("html"))
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in PROTECTED}
    assert before == after


def test_overview_has_callout_actions_and_strip():
    at = run_page("render_overview")
    bodies = " ".join(h.proto.body for h in at.get("html"))
    assert "Launch Workspace" in bodies and 'href="#model-benchmarks"' in bodies
    assert "streamlit run app/app.py" in bodies and "&#36;" in bodies and "$ " not in bodies
    assert bodies.count('class="vx-cell"') == 5


def _result_values(at):
    import re
    body = next(h.proto.body for h in at.get("html") if "vx-result" in h.proto.body)
    return [re.sub(r"<[^>]+>", "", v) for v in re.findall(r'<div class="v">(.*?)</div>', body)], body


def test_classify_form_submits_and_predicts():
    at = run_page("render_classify")
    [b for b in at.button if b.label == "Classify transaction"][0].click().run()
    assert not at.exception, [e.message for e in at.exception]
    values, body = _result_values(at)
    assert values[0] in ("Flag for review", "No alert") and len(values[1]) == 6      # 4-decimal score
    assert "&#36;500" in body and "&#36;10" in body and "500 alerts/day" in body
    reasons = next(h.proto.body for h in at.get("html") if 'class="reasons"' in h.proto.body)
    assert reasons.count("<li") == 5


@pytest.mark.parametrize("preset, decision", [("Typical repeat card", "No alert"),
                                              ("First-seen card", "Flag for review"),
                                              ("High-risk spike", "Flag for review")])
def test_presets_classify_in_one_click(preset, decision):
    at = run_page("render_classify")
    at.button(key=f"preset_{preset}").click().run()
    assert not at.exception and not at.warning
    values, _ = _result_values(at)
    assert values[0] == decision and values[3] == f"Preset: {preset}"
