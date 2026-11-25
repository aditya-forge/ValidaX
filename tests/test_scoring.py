import re

import pytest

import scoring

FAMILY_LABEL = re.compile(r"^masked [A-Za-z/ -]+ \((?:[CDMV]\d+|id_\w+)\)$")


@pytest.mark.parametrize("column, family", [
    ("C1", "masked counting feature"), ("C14", "masked counting feature"),
    ("D15", "masked time-delta feature"), ("M4", "masked match flag"),
    ("V339", "masked Vesta-engineered feature"), ("id_31", "masked identity/device signal"),
])
def test_masked_columns_get_family_labels(column, family):
    assert scoring.is_masked_column(column)
    assert scoring.label_for_feature(column) == f"{family} ({column})"
    assert FAMILY_LABEL.match(scoring.label_for_feature(column))


@pytest.mark.parametrize("column", ["DeviceInfo", "DeviceType", "card_max_amount_prior", "c_missing_count", "card6"])
def test_documented_columns_are_not_treated_as_masked(column):
    assert not scoring.is_masked_column(column)
    assert not scoring.label_for_feature(column).startswith("masked ")


@pytest.mark.parametrize("column, value", [("V70", 0.97), ("C14", 3.0), ("M4", "M2"), ("id_31", "chrome 63.0")])
def test_masked_reason_codes_never_show_values(column, value):
    reason = scoring.reason_text(column, 0.4, value)
    assert str(value) not in reason
    assert reason.endswith("increased risk")


def test_documented_reason_codes_show_formatted_values():
    assert scoring.reason_text("TransactionAmt", 0.2, 117.0) == "transaction amount ($117.00) increased risk"
    assert "1.85x this card's average" in scoring.reason_text("amount_vs_card_avg_ratio", 0.1, 1.8532)
    assert "no prior transaction" in scoring.reason_text("seconds_since_card_last_txn", 0.1, -1.0)
    assert scoring.reason_text("P_emaildomain", -0.3, "gmail.com").endswith("(gmail.com) decreased risk")


def test_time_labels_are_relative_not_calendar():
    for col in ("transaction_hour", "transaction_dow", "is_night", "is_weekend"):
        assert "relative" in scoring.label_for_feature(col)
