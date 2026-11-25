"""Shared fixtures. src/ is on the import path via pyproject.toml (pytest pythonpath)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def transactions() -> pd.DataFrame:
    """Small synthetic transaction table with the columns the velocity code needs."""
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame({
        "TransactionID": np.arange(1000, 1000 + n),
        "TransactionDT": np.sort(rng.integers(86_400, 86_400 * 30, n)),
        "card1": rng.choice([111, 222, 333, 444], n),
        "TransactionAmt": rng.lognormal(4, 1, n).round(2),
    })


@pytest.fixture(scope="session")
def console():
    """app/app.py imported as a module (main() is not run)."""
    spec = importlib.util.spec_from_file_location("console", REPO / "app" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
