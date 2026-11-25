"""Load, merge, and clean the IEEE-CIS Fraud Detection dataset.

Source: IEEE-CIS Fraud Detection (Kaggle competition, data provided by Vesta
Corporation). Only the labeled `train_transaction.csv` + `train_identity.csv`
files are used: Kaggle's `test_*.csv` files have no isFraud ground truth
(they exist only for competition leaderboard scoring), so this project
carves its own chronological train/test split out of the labeled data
instead (see src/model.py).

IMPORTANT: many columns in this dataset are intentionally masked by Vesta.
Per the competition's own data description, C1-C14 are "counting" features,
D1-D15 are "timedelta" features, M1-M9 are "match" flags, V1-V339 are
Vesta-engineered features, and most id_* columns are anonymized identity/
device signals, but the exact real-world meaning of any specific column is
not disclosed. This project never invents specific meanings for those
columns (e.g. never claims "C3 = number of phone numbers on file"). They are
used as opaque numeric/categorical model inputs and referred to generically
(by column name and family) everywhere, including in SHAP explanations and
the dashboard.

TIME: TransactionDT is seconds elapsed from an undisclosed reference point,
not a calendar timestamp. The derived `transaction_day`, `transaction_hour`
and `transaction_dow` columns are buckets of that relative offset (day index,
hour within a relative 24-hour cycle, position within a relative 7-day
cycle), never real dates, local hours or weekdays.

This module only writes data/processed/transactions_clean.parquet; it never
touches outputs/ or app/model.pkl.
"""
import gc
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_PATH = Path("data/processed/transactions_clean.parquet")
TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"

# Expected shape of the full labeled training set and of the chronological
# 80/20 split used by src/model.py (first floor(0.8 * n) rows by time = train).
EXPECTED_ROWS = 590_540
TRAIN_FRAC = 0.8
EXPECTED_TRAIN_ROWS = 472_432
EXPECTED_TEST_ROWS = 118_108

# Ordering key: TransactionDT, with TransactionID breaking same-second ties.
CHRONOLOGICAL_ORDER = ["TransactionDT", "TransactionID"]

REQUIRED_TRANSACTION_COLUMNS = ("TransactionID", "isFraud", "TransactionDT", "TransactionAmt")
REQUIRED_IDENTITY_COLUMNS = ("TransactionID",)

# Columns whose real-world meaning IS documented by the competition and are
# safe to use in investigator-facing narrative text.
INTERPRETABLE_COLUMNS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card4",  # card network: visa / mastercard / amex / discover
    "card6",  # card type: credit / debit
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
]

# Masked column families: used as model inputs, never given invented
# semantic labels in any narrative/explanation text.
MASKED_PREFIXES = ("C", "D", "M", "V", "id_")

# Columns read as text. Pinning them avoids mis-typing a string column whose
# first rows happen to be empty. Everything else is numeric and read as float32.
STRING_COLUMNS = {
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo",
    *(f"M{i}" for i in range(1, 10)),
    *(f"id_{i}" for i in (12, 15, 16, 23, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38)),
}
INTEGER_COLUMNS = {"TransactionID": "int64", "isFraud": "int8", "TransactionDT": "int64"}
_DTYPE_SNIFF_ROWS = 5_000


# Section: Loading
def _require_files(raw_dir: Path) -> tuple[Path, Path]:
    tx_path, id_path = raw_dir / TRANSACTION_FILE, raw_dir / IDENTITY_FILE
    missing = [p for p in (tx_path, id_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing IEEE-CIS raw file(s): "
            + ", ".join(str(p) for p in missing)
            + f". Download {TRANSACTION_FILE} and {IDENTITY_FILE} from "
            "https://www.kaggle.com/competitions/ieee-fraud-detection/data (accepting the "
            f"competition rules first) and place them in {raw_dir}/. Run this script from "
            "the repository root."
        )
    return tx_path, id_path


def _csv_dtypes(path: Path) -> dict:
    """float32 for numeric columns, object for text, fixed ints for keys.

    Reading numerics as float32 from the start (instead of float64, then
    downcasting) roughly halves peak memory for the 394-column transaction file.
    """
    sample = pd.read_csv(path, nrows=_DTYPE_SNIFF_ROWS)
    dtypes = {}
    for col, dtype in sample.dtypes.items():
        if col in INTEGER_COLUMNS:
            dtypes[col] = INTEGER_COLUMNS[col]
        elif col in STRING_COLUMNS or dtype.kind not in "biuf":
            dtypes[col] = "object"
        else:
            dtypes[col] = "float32"
    return dtypes


def _read_csv(path: Path, required: tuple, nrows: int = None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=_csv_dtypes(path), nrows=nrows, low_memory=True)
    absent = [c for c in required if c not in df.columns]
    if absent:
        raise ValueError(f"{path.name} is missing required column(s) {absent}; is it the IEEE-CIS file?")
    return df


def load_raw(raw_dir: Path = RAW_DIR, nrows: int = None) -> pd.DataFrame:
    """Read both raw tables and left-join identity onto transactions.

    `nrows` limits the transaction rows read (for quick checks only); the
    row-count and split checks in `run()` require the full file.
    """
    raw_dir = Path(raw_dir)
    tx_path, id_path = _require_files(raw_dir)
    tx = _read_csv(tx_path, REQUIRED_TRANSACTION_COLUMNS, nrows=nrows)
    idn = _read_csv(id_path, REQUIRED_IDENTITY_COLUMNS)

    matched = int(tx["TransactionID"].isin(idn["TransactionID"]).sum())
    df = tx.merge(idn, on="TransactionID", how="left", validate="one_to_one")
    print(f"Loaded {len(tx):,} transactions; {matched:,} have identity data "
          f"({matched / len(tx):.1%} match rate)")
    del tx, idn
    gc.collect()
    return df


# Section: Cleaning
def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32 and int64 -> the smallest integer type that fits (in place).

    Columns already given a deliberate narrower type (e.g. int32
    transaction_day) are left alone so the output schema stays stable.
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates(subset="TransactionID")
    print(f"Dropped {before - len(df):,} duplicate TransactionIDs")

    if df["TransactionDT"].isna().any():
        raise ValueError("TransactionDT has missing values; cannot order transactions in time.")

    # Relative-offset buckets of TransactionDT (see module docstring): the
    # offset is a valid ordering key, but not a calendar timestamp.
    df["transaction_day"] = (df["TransactionDT"] // (24 * 60 * 60)).astype("int32")
    df["transaction_hour"] = ((df["TransactionDT"] // 3600) % 24).astype("int8")
    df["transaction_dow"] = (df["transaction_day"] % 7).astype("int8")

    df["isFraud"] = df["isFraud"].astype("int8")
    df = _downcast_numeric(df)
    df = df.sort_values(CHRONOLOGICAL_ORDER, kind="mergesort").reset_index(drop=True)
    return df


# Section: Checks
def check_chronological_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    expect_full_dataset: bool = True,
) -> dict:
    """Verify the chronological 80/20 split that src/model.py will make.

    Checks that rows are in time order, that the train window ends no later
    than the test window starts, and (for the full dataset) that the sizes are
    exactly 590,540 = 472,432 train + 118,108 test. Raises ValueError otherwise.
    """
    n_rows = len(df)
    cutoff = int(n_rows * train_frac)
    offsets = df["TransactionDT"].to_numpy()

    if not np.all(np.diff(offsets) >= 0):
        raise ValueError("Rows are not sorted by TransactionDT; run clean() first.")
    if not 0 < cutoff < n_rows:
        raise ValueError(f"Cannot split {n_rows:,} rows with train_frac={train_frac}.")
    if offsets[cutoff - 1] > offsets[cutoff]:
        raise ValueError("Train window overlaps the test window in TransactionDT.")

    summary = {
        "rows": n_rows,
        "train_rows": cutoff,
        "test_rows": n_rows - cutoff,
        "train_offset_range": (int(offsets[0]), int(offsets[cutoff - 1])),
        "test_offset_range": (int(offsets[cutoff]), int(offsets[-1])),
        "train_fraud_rate": float(df["isFraud"].iloc[:cutoff].mean()),
        "test_fraud_rate": float(df["isFraud"].iloc[cutoff:].mean()),
    }
    expected = (EXPECTED_ROWS, EXPECTED_TRAIN_ROWS, EXPECTED_TEST_ROWS)
    if expect_full_dataset and (n_rows, cutoff, n_rows - cutoff) != expected:
        raise ValueError(
            f"Split is {n_rows:,} = {cutoff:,} train + {n_rows - cutoff:,} test; expected "
            f"{EXPECTED_ROWS:,} = {EXPECTED_TRAIN_ROWS:,} + {EXPECTED_TEST_ROWS:,} for the full "
            "IEEE-CIS training set. Check that the complete Kaggle files were used."
        )
    print(
        f"Chronological split OK: {summary['train_rows']:,} train (offsets "
        f"{summary['train_offset_range'][0]:,}-{summary['train_offset_range'][1]:,} s, fraud "
        f"{summary['train_fraud_rate']:.2%}) / {summary['test_rows']:,} test (offsets "
        f"{summary['test_offset_range'][0]:,}-{summary['test_offset_range'][1]:,} s, fraud "
        f"{summary['test_fraud_rate']:.2%})"
    )
    return summary


# Section: Main
def run(raw_dir: Path = RAW_DIR, out_path: Path = PROCESSED_PATH,
        expect_full_dataset: bool = True) -> pd.DataFrame:
    df = clean(load_raw(raw_dir))
    check_chronological_split(df, expect_full_dataset=expect_full_dataset)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved cleaned data ({df.shape[0]:,} rows x {df.shape[1]} cols) to {out_path}")
    return df


if __name__ == "__main__":
    run()
