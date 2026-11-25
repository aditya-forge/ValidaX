"""Train the form model: XGBoost on the 29 interpretable features only.

Used by the app's "Classify a transaction" page, where a person types in one
transaction. The 416 masked C/D/M/V/id_ columns can't be entered by hand, so
this model sees only documented fields and the card1 velocity features.

Protocol (same as the main model):
  - chronological 80/20 split on TransactionDT (472,432 train / 118,108 test);
  - tree count by early stopping, and the cost-optimal threshold ($500 missed
    fraud / $10 false positive, scenario assumptions) chosen on a chronological
    validation slice (last 20% of the training window);
  - refit on the full training window, reported once on the test window.

Writes artifacts/interpretable_model.pkl and artifacts/interpretable_model_summary.json.
Never touches app/model.pkl or outputs/. Run: python src/interpretable_model.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import inference as inf
import model as m
from features import INTERPRETABLE_MODEL_FEATURES

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "transactions_features.parquet"
BUNDLE_PATH = inf.ARTIFACTS_DIR / "interpretable_model.pkl"
SUMMARY_PATH = inf.ARTIFACTS_DIR / "interpretable_model_summary.json"
VAL_FRAC = 0.2


def _spw(y) -> float:
    y = np.asarray(y)
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def _operating_point(y, proba, threshold) -> dict:
    row = m.metrics_at_threshold(y, proba, threshold)
    row["total_cost"] = row["false_negatives"] * m.FALSE_NEGATIVE_COST + row["false_positives"] * m.FALSE_POSITIVE_COST
    return {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in row.items()}


def train(features_path: Path = FEATURES_PATH, overwrite: bool = False) -> dict:
    cols = list(INTERPRETABLE_MODEL_FEATURES)
    metadata = inf.load_metadata()
    df = pd.read_parquet(features_path, columns=["TransactionID", "TransactionDT", "transaction_day", "isFraud", *cols])
    df = inf.pin_categories(df, {c: v for c, v in metadata["categories"].items() if c in cols})
    train_df, test_df = m.chronological_split(df)
    fit_df, val_df = m.chronological_split(train_df, train_frac=1 - VAL_FRAC)

    # Tree count and threshold from the validation slice only.
    probe = m.make_xgb(_spw(fit_df["isFraud"]), n_estimators=2000, early_stopping_rounds=50)
    probe.fit(fit_df[cols], fit_df["isFraud"], eval_set=[(val_df[cols], val_df["isFraud"])], verbose=False)
    val_proba = probe.predict_proba(val_df[cols])[:, 1]
    val_curve = m.cost_sensitive_threshold(val_df["isFraud"].to_numpy(), val_proba)
    threshold = round(float(val_curve.loc[val_curve["total_cost"].idxmin(), "threshold"]), 2)
    n_estimators = int(round((probe.best_iteration + 1) / (1 - VAL_FRAC)))

    final = m.make_xgb(_spw(train_df["isFraud"]), n_estimators=n_estimators)
    final.fit(train_df[cols], train_df["isFraud"], verbose=False)
    y_test = test_df["isFraud"].to_numpy()
    proba = final.predict_proba(test_df[cols])[:, 1]
    daily = m.daily_capacity_simulation(test_df, proba, capacity=m.DAILY_REVIEW_CAPACITY)

    summary = {
        "model": "XGBoost, interpretable features only",
        "features": len(cols),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "n_estimators": n_estimators,
        "threshold": threshold,
        "threshold_source": "cost-optimal on the chronological validation slice ($500 FN / $10 FP)",
        "validation_pr_auc": float(m.score_probabilities(val_df["isFraud"], val_proba)["pr_auc"]),
        "test": {
            **m.score_probabilities(y_test, proba),
            "at_threshold": _operating_point(y_test, proba, threshold),
            "at_0.5": _operating_point(y_test, proba, 0.5),
            "avg_daily_recall_at_500": float(daily["recall"].mean()),
            "avg_daily_precision_at_500": float(daily["precision"].mean()),
        },
    }
    bundle = {
        "xgb_model": final,
        "feature_cols": cols,
        "categories": {c: v for c, v in metadata["categories"].items() if c in cols},
        "threshold": threshold,
        "summary": summary,
    }
    m.save_model_bundle(bundle, BUNDLE_PATH, overwrite=overwrite)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing artifacts/interpretable_model.pkl")
    train(overwrite=parser.parse_args().overwrite)
