"""Feature engineering for the IEEE-CIS fraud model.

Two tiers of features:

1. Interpretable business features: built from columns whose real-world
   meaning IS documented by the competition (amount, time offset, product,
   card network/type, email domain, device, address/distance) plus genuinely
   new features derived from them (free-email flag, email-domain match,
   relative time-cycle buckets, card-level transaction velocity). These are
   the features used in investigator-facing narrative text (dashboard reason
   codes, SHAP summaries).

   Time: TransactionDT is a count of seconds from an undisclosed reference
   point, NOT a calendar timestamp. Every time-derived feature here is a
   bucket of that relative offset. `transaction_hour` is (offset // 3600) mod
   24 and `transaction_dow` is (offset // 86400) mod 7, so `is_night` and
   `is_weekend` mark fixed positions in those relative cycles. They do not
   claim to be real local night-time or a real Saturday/Sunday; the column
   names are kept because the trained model bundle depends on them.

2. Masked engineered features: the raw C1-C14 / D1-D15 / M1-M9 / V1-V339 /
   id_* columns, used as opaque numeric/categorical model inputs. Per the
   competition's own data description their exact real-world meaning is
   masked by Vesta, and this project does NOT invent specific interpretations
   for any individual column in that family. The only features WE derive from
   them are meta-features about missingness (e.g. "how many V-block columns
   are populated for this transaction"), which describes data availability,
   not the masked content itself.
"""
import numpy as np
import pandas as pd

FREE_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "aol.com",
    "icloud.com",
    "live.com",
    "msn.com",
}

MASKED_FAMILIES = {
    "C": [f"C{i}" for i in range(1, 15)],
    "D": [f"D{i}" for i in range(1, 16)],
    "M": [f"M{i}" for i in range(1, 10)],
}

# Buckets of the relative TransactionDT cycles (see module docstring).
RELATIVE_NIGHT_HOURS = range(0, 6)   # hours 0-5 of the relative 24-hour cycle
RELATIVE_WEEKEND_DAYS = (5, 6)       # positions 5-6 of the relative 7-day cycle

# Ordering key for point-in-time features. TransactionID breaks ties between
# transactions recorded in the same second (it increases with TransactionDT),
# so "prior" always means "arrived earlier" and the order is reproducible.
CHRONOLOGICAL_ORDER = ["TransactionDT", "TransactionID"]


def _existing(df: pd.DataFrame, cols: list) -> list:
    return [c for c in cols if c in df.columns]


def add_interpretable_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["log_amount"] = np.log1p(df["TransactionAmt"])
    # Relative-cycle buckets of TransactionDT, not calendar night/weekend.
    df["is_night"] = df["transaction_hour"].isin(RELATIVE_NIGHT_HOURS).astype("int8")
    df["is_weekend"] = df["transaction_dow"].isin(RELATIVE_WEEKEND_DAYS).astype("int8")

    df["purchaser_email_is_free"] = (
        df["P_emaildomain"].isin(FREE_EMAIL_DOMAINS)
    ).astype("int8")
    df["has_recipient_email"] = df["R_emaildomain"].notna().astype("int8")
    df["email_domain_match"] = (
        df["P_emaildomain"].notna()
        & (df["P_emaildomain"] == df["R_emaildomain"])
    ).astype("int8")

    df["has_identity_data"] = df["DeviceType"].notna().astype("int8")
    df["is_mobile_device"] = (df["DeviceType"] == "mobile").astype("int8")

    df["has_dist1"] = df["dist1"].notna().astype("int8")
    df["has_dist2"] = df["dist2"].notna().astype("int8")

    for col, prefix in [("M1", "m1"), ("M4", "m4")]:
        if col in df.columns:
            df[f"{prefix}_is_match"] = (df[col] == "T").astype("int8")

    return df


def add_card_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time transaction velocity, grouped by `card1`.

    card1 is a high-cardinality, low-missingness numeric card-attribute code
    that the fraud-detection community commonly uses as a pseudo account
    identifier for exactly this purpose. We don't claim to know what card1
    literally encodes, only that grouping by it and ordering by
    TransactionDT gives a meaningful "this card's recent activity" signal.

    Leakage control: rows are put in a stable (card1, TransactionDT,
    TransactionID) order, and every statistic is an expanding window that
    excludes the current row (cumulative op, then subtract or shift by one
    *within the card*). A transaction therefore only sees transactions of the
    same card that arrived before it. Same-second transactions are ordered by
    TransactionID. Computing over train and test together is safe, because a
    test transaction only looks backwards, the same history it would have in
    production.

    `seconds_since_card_last_txn` is a difference of relative offsets
    (seconds), never a calendar date; -1 means "no prior transaction".
    """
    df = df.sort_values(["card1", *CHRONOLOGICAL_ORDER], kind="mergesort").copy()
    grp = df.groupby("card1", sort=False)

    cum_count = grp.cumcount()  # number of strictly earlier transactions
    df["card_prior_txn_count"] = cum_count

    cum_amount_sum = grp["TransactionAmt"].cumsum() - df["TransactionAmt"]
    avg_prior = cum_amount_sum / cum_count.replace(0, np.nan)
    df["card_avg_amount_prior"] = avg_prior.fillna(0.0)

    # Running max up to and including each row, shifted one row within the
    # same card so it covers earlier transactions only.
    running_max = grp["TransactionAmt"].cummax()
    prior_max = running_max.groupby(df["card1"], sort=False).shift(1)
    df["card_max_amount_prior"] = prior_max.fillna(0.0)

    prior_dt = grp["TransactionDT"].shift(1)
    seconds_since = df["TransactionDT"] - prior_dt
    df["seconds_since_card_last_txn"] = seconds_since.fillna(-1)

    df["amount_vs_card_avg_ratio"] = df["TransactionAmt"] / (
        df["card_avg_amount_prior"] + 1
    )
    df["card_is_first_seen"] = (df["card_prior_txn_count"] == 0).astype("int8")

    df = df.sort_values("TransactionID", kind="mergesort").reset_index(drop=True)
    return df


def add_masked_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Missingness meta-features over masked column families.

    These describe how much Vesta-engineered / counting / timedelta / match
    data was populated for a transaction, not what that data means.
    """
    df = df.copy()
    v_cols = [c for c in df.columns if c.startswith("V")]
    id_cols = [c for c in df.columns if c.startswith("id_")]

    if v_cols:
        df["v_features_missing_count"] = df[v_cols].isna().sum(axis=1).astype("int32")
        df["v_features_missing_pct"] = df["v_features_missing_count"] / len(v_cols)
    if id_cols:
        df["id_features_missing_count"] = df[id_cols].isna().sum(axis=1).astype("int32")

    for family, cols in MASKED_FAMILIES.items():
        present = _existing(df, cols)
        if present:
            df[f"{family.lower()}_missing_count"] = df[present].isna().sum(axis=1).astype("int32")

    return df


def encode_categoricals_for_xgboost(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


CATEGORICAL_COLUMNS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
] + [f"M{i}" for i in range(1, 10)] + [
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
    "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
]

INTERPRETABLE_MODEL_FEATURES = [
    "TransactionAmt",
    "log_amount",
    "transaction_hour",
    "transaction_dow",
    "is_night",
    "is_weekend",
    "ProductCD",
    "card4",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "has_dist1",
    "has_dist2",
    "P_emaildomain",
    "R_emaildomain",
    "purchaser_email_is_free",
    "has_recipient_email",
    "email_domain_match",
    "DeviceType",
    "has_identity_data",
    "is_mobile_device",
    "card_prior_txn_count",
    "card_avg_amount_prior",
    "card_max_amount_prior",
    "seconds_since_card_last_txn",
    "amount_vs_card_avg_ratio",
    "card_is_first_seen",
]

MASKED_MODEL_FEATURE_FAMILIES = ("C", "D", "M", "V", "id_")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_interpretable_features(df)
    df = add_card_velocity_features(df)
    df = add_masked_meta_features(df)
    df = encode_categoricals_for_xgboost(df, CATEGORICAL_COLUMNS)
    return df


def get_model_feature_columns(df: pd.DataFrame, include_masked: bool = True) -> list:
    cols = [c for c in INTERPRETABLE_MODEL_FEATURES if c in df.columns]
    meta_cols = [c for c in df.columns if c.endswith("_missing_count") or c.endswith("_missing_pct")]
    cols += [c for c in meta_cols if c not in cols]
    if include_masked:
        masked_cols = [
            c for c in df.columns
            if c.startswith(MASKED_MODEL_FEATURE_FAMILIES) and c not in cols
        ]
        cols += masked_cols
    return cols


if __name__ == "__main__":
    from pathlib import Path

    in_path = Path("data/processed/transactions_clean.parquet")
    out_path = Path("data/processed/transactions_features.parquet")

    df = pd.read_parquet(in_path)
    df = build_features(df)
    df.to_parquet(out_path, index=False)
    feats = get_model_feature_columns(df)
    print(f"Built {len(feats)} model features for {len(df):,} rows -> {out_path}")
