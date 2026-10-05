"""Tests for pipeline/split.py: periods are ordered and no target crosses a boundary.

Run from the project root:
    python -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.split import make_splits  # noqa: E402
from pipeline.targets import add_targets  # noqa: E402


@pytest.fixture(scope="module", params=[1, 10, 60])
def splits(request):
    """Full-length synthetic minute grid (same span as UCI) with targets."""
    H = request.param
    idx = pd.date_range("2006-12-16 17:24", "2010-11-26 21:02", freq="1min")
    rng = np.random.default_rng(0)
    load = pd.DataFrame({"load_w": rng.gamma(2.0, 500.0, len(idx)).astype("float32")}, index=idx)
    load.iloc[rng.choice(len(idx), 5000, replace=False)] = np.nan
    return H, make_splits(add_targets(load, H), H)


def test_val_and_test_never_earlier_than_last_train(splits):
    _, (train, val, test) = splits
    last_train = train.index.max()
    assert val.index.min() > last_train
    assert test.index.min() > last_train
    assert test.index.min() > val.index.max()


def test_no_target_window_crosses_into_next_period(splits):
    H, (train, val, test) = splits
    horizon = pd.Timedelta(minutes=H)
    assert train.index.max() + horizon < pd.Timestamp("2010-01-01")
    assert val.index.max() + horizon < pd.Timestamp("2010-07-01")


def test_periods_and_no_missing_values(splits):
    _, (train, val, test) = splits
    for part in (train, val, test):
        assert len(part) > 0
        assert not part.isna().any().any()
        assert part.index.is_monotonic_increasing
    assert train.index.min() >= pd.Timestamp("2006-12-01")
    assert val.index.min() >= pd.Timestamp("2010-01-01")
    assert test.index.min() >= pd.Timestamp("2010-07-01")
    assert test.index.max() < pd.Timestamp("2010-12-01")
