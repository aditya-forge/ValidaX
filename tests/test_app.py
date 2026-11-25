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


@pytest.mark.skipif(not (REPO / "outputs" / "scored_transactions.csv").exists(), reason="outputs/ not present")
@pytest.mark.parametrize("page", ["Fraud overview", "Alert queue", "Threshold simulator"])
def test_pages_render_without_errors(page):
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in PROTECTED}
    at = AppTest.from_file(str(REPO / "app" / "app.py"), default_timeout=300).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert any('class="ctx"' in h.proto.body and "not a determination of fraud" in h.proto.body
               for h in at.get("html"))
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in PROTECTED}
    assert before == after
