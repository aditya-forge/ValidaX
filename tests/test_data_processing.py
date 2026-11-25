import numpy as np
import pandas as pd
import pytest

import data_processing as dp


def _clean_frame(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": np.sort(rng.integers(86_400, 86_400 * 60, n)),
        "isFraud": rng.integers(0, 2, n),
    })


def test_split_check_passes_on_sorted_data():
    summary = dp.check_chronological_split(_clean_frame(), expect_full_dataset=False)
    assert (summary["train_rows"], summary["test_rows"]) == (800, 200)
    assert summary["train_offset_range"][1] <= summary["test_offset_range"][0]


def test_split_check_rejects_unsorted_data():
    with pytest.raises(ValueError, match="not sorted"):
        dp.check_chronological_split(_clean_frame().sample(frac=1, random_state=0), expect_full_dataset=False)


def test_split_check_requires_full_dataset_sizes():
    with pytest.raises(ValueError, match="472,432"):
        dp.check_chronological_split(_clean_frame())


def test_expected_split_constants():
    assert dp.EXPECTED_TRAIN_ROWS + dp.EXPECTED_TEST_ROWS == dp.EXPECTED_ROWS == 590_540
    assert int(dp.EXPECTED_ROWS * dp.TRAIN_FRAC) == dp.EXPECTED_TRAIN_ROWS


def test_missing_raw_files_give_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle.com/competitions/ieee-fraud-detection"):
        dp.load_raw(tmp_path)


def test_clean_adds_relative_time_buckets_and_downcasts():
    raw = pd.DataFrame({
        "TransactionID": np.array([2, 1], dtype="int64"),
        "TransactionDT": np.array([90_000 + 3_600 * 5, 86_400], dtype="int64"),
        "isFraud": [0, 1],
        "TransactionAmt": np.array([12.5, 7.0], dtype="float64"),
    })
    out = dp.clean(raw)
    assert out["TransactionID"].tolist() == [1, 2]                       # time order
    assert out["transaction_day"].tolist() == [1, 1]
    assert out["transaction_hour"].tolist() == [0, 6]
    assert out["TransactionAmt"].dtype == np.float32
    assert out["transaction_day"].dtype == np.int32                      # deliberate type kept
