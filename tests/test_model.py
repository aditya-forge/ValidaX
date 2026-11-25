import numpy as np
import pandas as pd
import pytest

import model as m


def test_operating_assumptions():
    assert (m.FALSE_NEGATIVE_COST, m.FALSE_POSITIVE_COST, m.DAILY_REVIEW_CAPACITY) == (500, 10, 500)
    assert m.PROMOTE_TUNED_MODEL is False


def test_chronological_split_keeps_time_order():
    df = pd.DataFrame({"TransactionDT": np.arange(100)[::-1], "transaction_day": 0, "isFraud": 0})
    train, test = m.chronological_split(df)
    assert (len(train), len(test)) == (80, 20)
    assert train["TransactionDT"].max() < test["TransactionDT"].min()


def test_cost_sensitive_threshold_matches_cost_formula():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.3, 0.35, 0.1, 0.05, 0.6])
    curve = m.cost_sensitive_threshold(y, p)
    row = curve[np.isclose(curve["threshold"], 0.32)].iloc[0]
    assert row["false_negatives"] == 1 and row["false_positives"] == 2
    assert row["total_cost"] == 1 * 500 + 2 * 10


def test_capacity_at_k():
    y = np.array([1, 0, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.1, 0.7, 0.2])
    cap = m.capacity_at_k(y, p, k=2)
    assert cap == {"k": 2, "fraud_captured": 1, "fraud_total": 2, "precision_at_k": 0.5, "recall_at_k": 0.5}


def test_risk_tiers():
    assert [m.risk_tier(p) for p in (0.95, 0.7, 0.4, 0.1)] == ["Critical", "High", "Medium", "Low"]


def test_best_params_from_search_returns_native_types():
    row = {k: v[0] for k, v in m.XGB_SEARCH_SPACE.items()}
    search = pd.DataFrame([{**row, "best_iteration": np.int64(99), "val_pr_auc": 0.5}])
    params = m.best_params_from_search(search, val_frac=0.2)
    assert isinstance(params["max_depth"], int) and params["n_estimators"] == 125


def test_bundle_save_refuses_to_overwrite(tmp_path):
    path = tmp_path / "model.pkl"
    bundle = m.build_model_bundle("xgb", "lr", ["TransactionAmt", "C1"], 0.24)
    m.save_model_bundle(bundle, path)
    with pytest.raises(FileExistsError):
        m.save_model_bundle(bundle, path)


def test_tuned_model_is_not_promoted_by_default(tmp_path):
    path = tmp_path / "model.pkl"
    assert m.promote_tuned_model({"xgb_model": "tuned"}, path) is False
    assert not path.exists()


def test_cluster_features_are_interpretable_only():
    import scoring
    assert not [c for c in m.CLUSTER_FEATURES if scoring.is_masked_column(c)]
