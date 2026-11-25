"""Turn new transactions into model features and fraudulent/legitimate predictions.

Two entry points:

- ``prepare_batch``: raw IEEE-CIS rows (same columns as ``train_transaction.csv``,
  optionally merged with ``train_identity.csv``) -> the 451 features of the main
  model in ``app/model.pkl``.
- ``prepare_single``: one transaction typed into a form (amount, product, card,
  email, device, relative time, card history) -> the 29 interpretable features of
  the form model in ``artifacts/interpretable_model.pkl``.

Both reuse src/features.py and pin categorical columns to the exact categories
seen in training (``artifacts/inference_metadata.json``). XGBoost encodes
categoricals by their position in that list, so a different order would
silently change predictions.

Card-history features need the card's earlier transactions. For uploads they come
from ``artifacts/card_history.csv`` (per-``card1`` count, amount sum, max and last
``TransactionDT`` over the labeled IEEE-CIS window) plus earlier rows in the
same upload. That is point-in-time correct for transactions that happen after the
labeled window, such as Kaggle's ``test_transaction.csv``.

Run ``python src/inference.py`` to rebuild the metadata and history files from
data/processed/ (needs the Kaggle-derived parquet files; never touches outputs/).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import features as ft

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
METADATA_PATH = ARTIFACTS_DIR / "inference_metadata.json"
CARD_HISTORY_PATH = ARTIFACTS_DIR / "card_history.csv"

TIME_COLUMNS = ("transaction_day", "transaction_hour", "transaction_dow")
REQUIRED_BATCH_COLUMNS = ("TransactionID", "TransactionDT", "TransactionAmt", "card1")
FRAUDULENT, LEGITIMATE = "Fraudulent", "Legitimate"


# Section: Helpers
def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Kaggle's test_identity.csv uses ``id-01``; training used ``id_01``."""
    return frame.rename(columns=lambda c: c.replace("id-", "id_") if c.startswith("id-") else c)


def classify(probability, threshold: float):
    """Fraudulent if the score is at or above the operating threshold, else legitimate."""
    probability = np.asarray(probability, dtype=float)
    return np.where(probability >= threshold, FRAUDULENT, LEGITIMATE)


def add_time_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    """Same relative-offset buckets as data_processing.clean (TransactionDT is not a date)."""
    frame["transaction_day"] = (frame["TransactionDT"] // 86_400).astype("int32")
    frame["transaction_hour"] = ((frame["TransactionDT"] // 3_600) % 24).astype("int8")
    frame["transaction_dow"] = (frame["transaction_day"] % 7).astype("int8")
    return frame


def pin_categories(frame: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """Cast categorical columns to the training categories; unseen values become missing."""
    for col, cats in categories.items():
        if col in frame:
            values = frame[col].astype("object").where(frame[col].notna(), None)
            frame[col] = pd.Categorical(values, categories=cats)
    return frame


def add_velocity_with_history(frame: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Card velocity features combining stored history with earlier rows of this batch.

    Mirrors features.add_card_velocity_features: rows are ordered by
    (card1, TransactionDT, TransactionID) and every statistic covers earlier
    transactions only. ``history`` is indexed by card1 with columns
    count, amount_sum, amount_max, last_dt.
    """
    out = frame.sort_values(["card1", *ft.CHRONOLOGICAL_ORDER], kind="mergesort").copy()
    grp = out.groupby("card1", sort=False, dropna=False)
    amount = out["TransactionAmt"].astype("float64")

    within_count = grp.cumcount()
    within_sum = grp["TransactionAmt"].cumsum().astype("float64") - amount
    within_prev_max = grp["TransactionAmt"].cummax().groupby(out["card1"], sort=False, dropna=False).shift(1)
    within_prev_dt = grp["TransactionDT"].shift(1)

    hist = history.reindex(out["card1"].to_numpy()).astype("float64")
    hist_count = hist["count"].fillna(0).to_numpy()
    hist_sum = hist["amount_sum"].fillna(0.0).to_numpy()

    prior_count = hist_count + within_count.to_numpy()
    prior_sum = hist_sum + within_sum.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        avg_prior = np.where(prior_count > 0, prior_sum / np.where(prior_count > 0, prior_count, 1), 0.0)
    max_prior = np.fmax(hist["amount_max"].to_numpy(), within_prev_max.astype("float64").to_numpy())
    last_dt = within_prev_dt.astype("float64").to_numpy()
    last_dt = np.where(np.isnan(last_dt), hist["last_dt"].to_numpy(), last_dt)

    out["card_prior_txn_count"] = prior_count.astype("int64")
    out["card_avg_amount_prior"] = avg_prior
    out["card_max_amount_prior"] = np.nan_to_num(max_prior, nan=0.0)
    out["seconds_since_card_last_txn"] = np.where(np.isnan(last_dt), -1.0, out["TransactionDT"].to_numpy() - last_dt)
    out["amount_vs_card_avg_ratio"] = amount.to_numpy() / (out["card_avg_amount_prior"].to_numpy() + 1)
    out["card_is_first_seen"] = (out["card_prior_txn_count"] == 0).astype("int8")
    return out.sort_values("TransactionID", kind="mergesort")


# Section: Artifacts
def load_metadata(path: Path = METADATA_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_card_history(path: Path = CARD_HISTORY_PATH) -> pd.DataFrame:
    return pd.read_csv(path).set_index("card1")


def build_card_history(reference: pd.DataFrame) -> pd.DataFrame:
    """Per-card1 summary of the reference window: count, amount sum/max, last TransactionDT."""
    return (
        reference.groupby("card1")
        .agg(count=("TransactionAmt", "size"), amount_sum=("TransactionAmt", "sum"),
             amount_max=("TransactionAmt", "max"), last_dt=("TransactionDT", "max"))
        .astype({"count": "int64", "amount_sum": "float64", "amount_max": "float64", "last_dt": "int64"})
    )


# Section: Batch (CSV)
def read_batch_csv(source, metadata: dict, nrows: int | None = None) -> pd.DataFrame:
    """Read an uploaded IEEE-CIS CSV with the training dtypes (text vs float)."""
    header = pd.read_csv(source, nrows=0)
    if hasattr(source, "seek"):
        source.seek(0)
    normalized = {c: normalize_columns(pd.DataFrame(columns=[c])).columns[0] for c in header.columns}
    text = set(metadata["string_columns"])
    dtypes = {}
    for original, name in normalized.items():
        if name in ("TransactionID", "TransactionDT"):
            dtypes[original] = "int64"
        elif name in text:
            dtypes[original] = "object"
        elif name in metadata["raw_columns"] or name == "isFraud":
            dtypes[original] = "float32" if name != "isFraud" else "float64"
    return normalize_columns(pd.read_csv(source, dtype=dtypes, nrows=nrows, low_memory=False))


def prepare_batch(raw: pd.DataFrame, metadata: dict, history: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Raw IEEE-CIS rows -> main-model features, indexed by TransactionID (input order kept)."""
    raw = normalize_columns(raw)
    missing = [c for c in REQUIRED_BATCH_COLUMNS if c not in raw]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. "
                         "Upload rows in the IEEE-CIS train_transaction format.")
    if raw["TransactionID"].duplicated().any():
        raise ValueError("TransactionID values must be unique.")

    frame = raw.copy()
    for col in metadata["raw_columns"]:
        if col not in frame:
            frame[col] = np.nan
    text = set(metadata["string_columns"])
    for col in metadata["raw_columns"]:
        if col in text:
            frame[col] = frame[col].astype("object").where(frame[col].notna(), None)
        elif col not in ("TransactionID", "TransactionDT"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32")
    frame["TransactionID"] = frame["TransactionID"].astype("int64")
    frame["TransactionDT"] = frame["TransactionDT"].astype("int64")

    frame = add_time_buckets(frame)
    frame = ft.add_interpretable_features(frame)
    frame = add_velocity_with_history(frame, history)
    frame = ft.add_masked_meta_features(frame[[*metadata["raw_columns"], *TIME_COLUMNS,
                                               *[c for c in frame.columns if c not in metadata["raw_columns"]
                                                 and c not in TIME_COLUMNS]]])
    frame = pin_categories(frame, metadata["categories"])
    order = raw["TransactionID"].astype("int64").to_numpy()
    return frame.set_index("TransactionID", drop=False).loc[order, feature_cols]


# Section: Single (form)
def prepare_single(inputs: dict, metadata: dict, feature_cols: list) -> pd.DataFrame:
    """One form entry -> the interpretable model's features (one-row DataFrame).

    ``inputs`` keys: TransactionAmt, ProductCD, card4, card6, P_emaildomain,
    R_emaildomain, DeviceType, addr1, addr2, dist1, dist2, transaction_hour,
    transaction_dow, card_prior_txn_count, card_avg_amount_prior,
    card_max_amount_prior, seconds_since_card_last_txn (None or < 0 = first seen).
    Missing or None values are treated as missing, as in training.
    """
    row = {k: inputs.get(k) for k in ("TransactionAmt", "ProductCD", "card4", "card6", "P_emaildomain",
                                       "R_emaildomain", "DeviceType", "addr1", "addr2", "dist1", "dist2",
                                       "transaction_hour", "transaction_dow")}
    frame = pd.DataFrame([row])
    for col in ("TransactionAmt", "addr1", "addr2", "dist1", "dist2"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32")
    frame["transaction_hour"] = int(inputs.get("transaction_hour") or 0)
    frame["transaction_dow"] = int(inputs.get("transaction_dow") or 0)
    frame = ft.add_interpretable_features(frame)

    prior = int(inputs.get("card_prior_txn_count") or 0)
    gap = inputs.get("seconds_since_card_last_txn")
    first_seen = prior == 0 or gap is None or float(gap) < 0
    avg = 0.0 if first_seen else float(inputs.get("card_avg_amount_prior") or 0.0)
    frame["card_prior_txn_count"] = 0 if first_seen else prior
    frame["card_avg_amount_prior"] = avg
    frame["card_max_amount_prior"] = 0.0 if first_seen else float(inputs.get("card_max_amount_prior") or 0.0)
    frame["seconds_since_card_last_txn"] = -1.0 if first_seen else float(gap)
    frame["amount_vs_card_avg_ratio"] = frame["TransactionAmt"].astype("float64") / (avg + 1)
    frame["card_is_first_seen"] = np.int8(1 if first_seen else 0)

    frame = pin_categories(frame, {c: v for c, v in metadata["categories"].items() if c in feature_cols})
    return frame[feature_cols]


# Section: Build
def build_artifacts(clean_path: Path, features_path: Path, out_dir: Path = ARTIFACTS_DIR) -> dict:
    """Write inference_metadata.json and card_history.csv from the processed parquet files."""
    import pyarrow.parquet as pq

    clean_cols = pq.read_schema(clean_path).names
    raw_columns = [c for c in clean_cols if c not in TIME_COLUMNS and c != "isFraud"]
    feature_schema = pq.read_schema(features_path)
    categories = {}
    for col in ft.CATEGORICAL_COLUMNS:
        if col in feature_schema.names:
            series = pd.read_parquet(features_path, columns=[col])[col]
            categories[col] = [str(c) for c in series.cat.categories]
    sample = pd.read_parquet(clean_path, columns=raw_columns).head(1)
    string_columns = [c for c in raw_columns if sample[c].dtype == object or str(sample[c].dtype).startswith("str")]

    reference = pd.read_parquet(clean_path, columns=["card1", "TransactionAmt", "TransactionDT"])
    history = build_card_history(reference)

    # Form defaults: the most common billing address codes. A missing address is a
    # strong fraud signal in this data, so an empty default would bias every example.
    address = pd.read_parquet(clean_path, columns=["addr1", "addr2"])
    form_defaults = {c: float(address[c].mode().iloc[0]) for c in ("addr1", "addr2")}

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "raw_columns": raw_columns,
        "string_columns": string_columns,
        "categories": categories,
        "history_end_dt": int(reference["TransactionDT"].max()),
        "history_rows": int(len(reference)),
        "form_defaults": form_defaults,
    }
    (out_dir / "inference_metadata.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    history.reset_index().to_csv(out_dir / "card_history.csv", index=False)
    print(f"wrote {out_dir / 'inference_metadata.json'} ({len(raw_columns)} raw columns, "
          f"{len(categories)} categorical) and {out_dir / 'card_history.csv'} ({len(history):,} cards)")
    return metadata


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    build_artifacts(root / "data/processed/transactions_clean.parquet",
                    root / "data/processed/transactions_features.parquet")
