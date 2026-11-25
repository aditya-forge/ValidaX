"""Produce the investigator-facing risk scoring table.

Scores every transaction in the (chronological) test split, assigns a risk
tier, and -- for anything at or above the Medium tier, i.e. what would
actually enter an alert queue -- generates plain-English SHAP reason codes.

Reason-code labels are written by hand per interpretable feature. Masked
C1-C14 / D1-D15 / M1-M9 / V1-V339 / id_* columns get a generic, family-level
label (e.g. "masked Vesta-engineered feature (V70)"), never an invented
specific meaning, and their raw values are never shown in a reason code --
consistent with src/features.py and the competition's own data description.

Time-derived features are labeled as positions in the *relative*
TransactionDT cycle, because TransactionDT is a seconds offset from an
undisclosed reference point, not a calendar timestamp.
"""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from model import DATA_PATH, MODEL_PATH, chronological_split, risk_tier

OUTPUTS_DIR = Path("outputs")
SCORED_PATH = OUTPUTS_DIR / "scored_transactions.csv"
REASON_TIER_THRESHOLD = 0.3  # Medium and above get SHAP reason codes
TOP_N_REASONS = 3

FEATURE_LABELS = {
    "TransactionAmt": "transaction amount",
    "log_amount": "transaction amount (log-scaled)",
    "transaction_hour": "hour of the relative 24-hour time cycle",
    "transaction_dow": "day of the relative 7-day time cycle",
    "is_night": "hours 0-5 of the relative 24-hour cycle",
    "is_weekend": "days 5-6 of the relative 7-day cycle",
    "ProductCD": "product code",
    "card4": "card network",
    "card6": "card type (credit/debit)",
    "addr1": "billing address code",
    "addr2": "billing address code (secondary)",
    "dist1": "distance feature",
    "dist2": "distance feature (secondary)",
    "has_dist1": "distance feature present",
    "has_dist2": "distance feature present (secondary)",
    "P_emaildomain": "purchaser email domain",
    "R_emaildomain": "recipient email domain",
    "purchaser_email_is_free": "purchaser uses a free/consumer email domain",
    "has_recipient_email": "recipient email domain present",
    "email_domain_match": "purchaser/recipient email domains match",
    "DeviceType": "device type",
    "DeviceInfo": "device/browser string",
    "has_identity_data": "identity/device data present",
    "is_mobile_device": "mobile device",
    "card_prior_txn_count": "this card's prior transaction count",
    "card_avg_amount_prior": "this card's average past transaction amount",
    "card_max_amount_prior": "this card's largest past transaction amount",
    "seconds_since_card_last_txn": "time since this card's last transaction",
    "amount_vs_card_avg_ratio": "amount vs. this card's typical amount",
    "card_is_first_seen": "first transaction seen for this card",
    # Missingness meta-features: they describe how much masked data is
    # present for a transaction, not what that data means.
    "v_features_missing_count": "amount of Vesta-engineered data available",
    "v_features_missing_pct": "amount of Vesta-engineered data available",
    "id_features_missing_count": "amount of identity data available",
    "c_missing_count": "number of masked counting features missing",
    "d_missing_count": "number of masked time-delta features missing",
    "m_missing_count": "number of masked match flags missing",
}

MASKED_FAMILY_LABELS = {
    "C": "masked counting feature",
    "D": "masked time-delta feature",
    "M": "masked match flag",
    "V": "masked Vesta-engineered feature",
    "id_": "masked identity/device signal",
}

# Raw masked columns: C1-C14, D1-D15, M1-M9, V1-V339 and id_*.
_MASKED_COLUMN = re.compile(r"^(?P<family>[CDMV])\d+$|^(?P<identity>id_)\w+$")

DOLLAR_FEATURES = {"TransactionAmt", "card_avg_amount_prior", "card_max_amount_prior"}


def masked_family(name: str):
    """Family prefix ("C", "D", "M", "V" or "id_") for a raw masked column, else None."""
    match = _MASKED_COLUMN.match(name)
    if match is None:
        return None
    return match.group("family") or match.group("identity")


def is_masked_column(name: str) -> bool:
    return masked_family(name) is not None


def label_for_feature(name: str) -> str:
    """Investigator-facing label; masked columns only ever get their family label."""
    family = masked_family(name)
    if family is not None:
        return f"{MASKED_FAMILY_LABELS[family]} ({name})"
    return FEATURE_LABELS.get(name, name)


def _format_value(feature: str, value) -> str:
    if pd.isna(value):
        return "missing"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if feature in DOLLAR_FEATURES:
        return f"${number:,.2f}"
    if feature == "amount_vs_card_avg_ratio":
        return f"{number:,.2f}x this card's average"
    if feature == "seconds_since_card_last_txn":
        return "no prior transaction" if number < 0 else f"{number:,.0f} s"
    if feature == "v_features_missing_pct":
        return f"{number:.0%} missing"
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.2f}"


def reason_text(feature: str, shap_value: float, raw_value=None) -> str:
    """One reason code. Masked columns get a family label and never show a value."""
    label = label_for_feature(feature)
    direction = "increased" if shap_value > 0 else "decreased"
    show_value = (
        raw_value is not None
        and not is_masked_column(feature)
        and feature in FEATURE_LABELS
    )
    if show_value:
        return f"{label} ({_format_value(feature, raw_value)}) {direction} risk"
    return f"{label} {direction} risk"


def build_reason_codes(shap_values, feature_names: list, raw_df: pd.DataFrame, top_n: int = TOP_N_REASONS) -> list:
    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    reasons = []
    for row_i, row in enumerate(values):
        idx = np.argsort(-np.abs(row))[:top_n]
        raw_row = raw_df.iloc[row_i]
        reasons.append([
            reason_text(feature_names[i], row[i], raw_row[feature_names[i]]) for i in idx
        ])
    return reasons


def main():
    bundle = joblib.load(MODEL_PATH)
    xgb_model = bundle["xgb_model"]
    feature_cols = bundle["feature_cols"]

    df = pd.read_parquet(DATA_PATH)
    _, test = chronological_split(df)

    X_test = test[feature_cols]
    fraud_probability = xgb_model.predict_proba(X_test)[:, 1]

    scored = test[[
        "TransactionID", "TransactionDT", "transaction_day", "isFraud",
        "TransactionAmt", "ProductCD", "card4", "card6", "P_emaildomain",
        "DeviceType",
    ]].copy()
    scored["fraud_probability"] = fraud_probability
    scored["risk_tier"] = scored["fraud_probability"].apply(risk_tier)

    scored["top_reason_1"] = ""
    scored["top_reason_2"] = ""
    scored["top_reason_3"] = ""

    alert_mask = scored["fraud_probability"] >= REASON_TIER_THRESHOLD
    n_alerts = int(alert_mask.sum())
    print(f"Generating SHAP reason codes for {n_alerts:,} alert-tier transactions "
          f"(fraud_probability >= {REASON_TIER_THRESHOLD})...")

    if n_alerts > 0:
        explainer = shap.TreeExplainer(xgb_model)
        X_alerts = X_test.loc[alert_mask]
        shap_values = explainer(X_alerts)
        reasons = build_reason_codes(shap_values, feature_cols, X_alerts)
        reason_cols = pd.DataFrame(
            reasons, index=X_alerts.index,
            columns=["top_reason_1", "top_reason_2", "top_reason_3"],
        )
        scored.loc[alert_mask, ["top_reason_1", "top_reason_2", "top_reason_3"]] = reason_cols

    scored = scored.sort_values("fraud_probability", ascending=False).reset_index(drop=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    scored.to_csv(SCORED_PATH, index=False)
    print(f"Saved {len(scored):,} scored transactions -> {SCORED_PATH}")
    print(scored["risk_tier"].value_counts())

    return scored


if __name__ == "__main__":
    main()
