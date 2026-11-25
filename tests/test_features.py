import numpy as np
import pandas as pd

import features

VELOCITY = [
    "card_prior_txn_count", "card_avg_amount_prior", "card_max_amount_prior",
    "seconds_since_card_last_txn", "amount_vs_card_avg_ratio", "card_is_first_seen",
]


def test_velocity_uses_only_earlier_transactions(transactions):
    out = features.add_card_velocity_features(transactions)
    for _, card in out.sort_values(features.CHRONOLOGICAL_ORDER).groupby("card1"):
        amounts = card["TransactionAmt"].to_numpy()
        assert card["card_prior_txn_count"].tolist() == list(range(len(card)))
        expected_max = [0.0] + [amounts[:i].max() for i in range(1, len(card))]
        assert np.allclose(card["card_max_amount_prior"], expected_max)
        expected_avg = [0.0] + [amounts[:i].mean() for i in range(1, len(card))]
        assert np.allclose(card["card_avg_amount_prior"], expected_avg)


def test_velocity_has_no_future_leakage(transactions):
    full = features.add_card_velocity_features(transactions)
    cut = transactions["TransactionDT"].quantile(0.7)
    past_only = features.add_card_velocity_features(transactions[transactions["TransactionDT"] <= cut])
    same_rows = full[full["TransactionDT"] <= cut].reset_index(drop=True)
    for col in VELOCITY:
        assert np.allclose(past_only[col].to_numpy(float), same_rows[col].to_numpy(float)), col


def test_first_seen_card_uses_sentinel(transactions):
    out = features.add_card_velocity_features(transactions)
    first = out[out["card_is_first_seen"] == 1]
    assert len(first) == transactions["card1"].nunique()
    assert (first["seconds_since_card_last_txn"] == -1).all()
    assert (first["card_prior_txn_count"] == 0).all()


def test_same_second_ties_break_by_transaction_id():
    df = pd.DataFrame({"TransactionID": [3, 1, 2], "TransactionDT": [100, 100, 100],
                       "card1": [7, 7, 7], "TransactionAmt": [30.0, 10.0, 20.0]})
    out = features.add_card_velocity_features(df).set_index("TransactionID")
    assert out.loc[[1, 2, 3], "card_prior_txn_count"].tolist() == [0, 1, 2]


def test_relative_cycle_buckets():
    df = pd.DataFrame({
        "TransactionAmt": [10.0, 20.0], "transaction_hour": [3, 15], "transaction_dow": [5, 1],
        "P_emaildomain": ["gmail.com", None], "R_emaildomain": [None, None],
        "DeviceType": ["mobile", None], "dist1": [1.0, None], "dist2": [None, None],
    })
    out = features.add_interpretable_features(df)
    assert out["is_night"].tolist() == [1, 0]
    assert out["is_weekend"].tolist() == [1, 0]
    assert out["purchaser_email_is_free"].tolist() == [1, 0]


def test_model_feature_columns_separate_masked_families():
    cols = ["TransactionAmt", "card6", "C1", "D2", "M4", "V70", "id_31", "c_missing_count", "isFraud"]
    selected = features.get_model_feature_columns(pd.DataFrame(columns=cols))
    assert "isFraud" not in selected
    assert selected.index("TransactionAmt") < selected.index("C1")
    assert {"C1", "D2", "M4", "V70", "id_31", "c_missing_count"} <= set(selected)
    assert "C1" not in features.get_model_feature_columns(pd.DataFrame(columns=cols), include_masked=False)
