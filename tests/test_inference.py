import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import features
import inference as inf

REPO = Path(__file__).resolve().parents[1]
VELOCITY = ["card_prior_txn_count", "card_avg_amount_prior", "card_max_amount_prior",
            "seconds_since_card_last_txn", "amount_vs_card_avg_ratio", "card_is_first_seen"]
EMPTY_HISTORY = pd.DataFrame(columns=["count", "amount_sum", "amount_max", "last_dt"]).rename_axis("card1")


def test_classify_labels_at_threshold():
    assert inf.classify([0.23, 0.24, 0.9], 0.24).tolist() == ["Legitimate", "Fraudulent", "Fraudulent"]


def test_normalize_columns_maps_kaggle_test_identity_names():
    out = inf.normalize_columns(pd.DataFrame(columns=["TransactionID", "id-01", "id-31", "DeviceType"]))
    assert list(out.columns) == ["TransactionID", "id_01", "id_31", "DeviceType"]


def test_velocity_without_history_matches_feature_pipeline(transactions):
    expected = features.add_card_velocity_features(transactions).set_index("TransactionID")
    got = inf.add_velocity_with_history(transactions, EMPTY_HISTORY).set_index("TransactionID")
    for col in VELOCITY:
        assert np.allclose(got.loc[expected.index, col].to_numpy(float), expected[col].to_numpy(float)), col


def test_history_plus_batch_equals_full_run(transactions):
    """Velocity for later rows = history summary of earlier rows + earlier rows of the batch."""
    full = features.add_card_velocity_features(transactions).set_index("TransactionID")
    cut = transactions["TransactionDT"].quantile(0.6)
    earlier = transactions[transactions["TransactionDT"] <= cut]
    later = transactions[transactions["TransactionDT"] > cut]
    got = inf.add_velocity_with_history(later, inf.build_card_history(earlier)).set_index("TransactionID")
    for col in VELOCITY:
        assert np.allclose(got[col].to_numpy(float), full.loc[got.index, col].to_numpy(float)), col


def test_pin_categories_uses_training_order_and_drops_unseen():
    frame = pd.DataFrame({"card4": ["visa", "unknown-network", None]})
    out = inf.pin_categories(frame, {"card4": ["american express", "visa"]})
    assert list(out["card4"].cat.categories) == ["american express", "visa"]
    assert out["card4"].cat.codes.tolist() == [1, -1, -1]


@pytest.fixture(scope="module")
def metadata():
    return inf.load_metadata()


def test_artifacts_are_consistent(metadata):
    assert len(metadata["raw_columns"]) == 433 and len(metadata["categories"]) == 31
    assert set(metadata["categories"]) <= set(features.CATEGORICAL_COLUMNS)
    history = inf.load_card_history()
    assert history.index.name == "card1" and int(history["count"].sum()) == metadata["history_rows"] == 590_540
    summary = json.loads((inf.ARTIFACTS_DIR / "interpretable_model_summary.json").read_text())
    assert summary["features"] == len(features.INTERPRETABLE_MODEL_FEATURES)


def test_prepare_single_and_form_model(metadata):
    bundle = joblib.load(inf.ARTIFACTS_DIR / "interpretable_model.pkl")
    inputs = {"TransactionAmt": 117.0, "ProductCD": "W", "card4": "visa", "card6": "debit",
              "P_emaildomain": "gmail.com", "addr1": 299.0, "addr2": 87.0, "transaction_hour": 14,
              "transaction_dow": 2, "card_prior_txn_count": 12, "card_avg_amount_prior": 95.0,
              "card_max_amount_prior": 300.0, "seconds_since_card_last_txn": 30 * 3600}
    X = inf.prepare_single(inputs, metadata, bundle["feature_cols"])
    assert list(X.columns) == bundle["feature_cols"] and len(X) == 1
    assert X["amount_vs_card_avg_ratio"].iloc[0] == pytest.approx(117 / 96)
    first = inf.prepare_single({**inputs, "seconds_since_card_last_txn": None}, metadata, bundle["feature_cols"])
    assert first["card_is_first_seen"].iloc[0] == 1 and first["seconds_since_card_last_txn"].iloc[0] == -1
    proba = bundle["xgb_model"].predict_proba(pd.concat([X, first]))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_prepare_batch_rejects_bad_uploads(metadata):
    history = inf.load_card_history()
    with pytest.raises(ValueError, match="Missing required"):
        inf.prepare_batch(pd.DataFrame({"TransactionID": [1]}), metadata, history, ["TransactionAmt"])
    dup = pd.DataFrame({"TransactionID": [1, 1], "TransactionDT": [1, 2],
                        "TransactionAmt": [1.0, 2.0], "card1": [5, 5]})
    with pytest.raises(ValueError, match="unique"):
        inf.prepare_batch(dup, metadata, history, ["TransactionAmt"])


def test_prepare_batch_minimal_columns(metadata):
    """Only the required columns: everything else is treated as missing, as in training."""
    history = inf.load_card_history()
    raw = pd.DataFrame({"TransactionID": [9_000_001, 9_000_002], "TransactionDT": [16_000_000, 16_000_100],
                        "TransactionAmt": [50.0, 75.0], "card1": [999_999, 999_999]})
    cols = ["TransactionAmt", "card_prior_txn_count", "seconds_since_card_last_txn", "V70", "M4", "id_31"]
    X = inf.prepare_batch(raw, metadata, history, cols)
    assert X["card_prior_txn_count"].tolist() == [0, 1]            # unseen card, then its first transaction
    assert X["seconds_since_card_last_txn"].tolist() == [-1.0, 100.0]
    assert X["V70"].isna().all() and X["M4"].isna().all()
