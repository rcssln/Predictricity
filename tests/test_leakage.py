"""Leakage tests for pipeline/features.py and pipeline/targets.py.

1. Features at time t depend only on data at or before t:
   rewriting every value after t leaves feature rows <= t bit-for-bit identical.
2. Targets at time t depend only on data after t:
   rewriting every value at or before t leaves target rows >= t bit-for-bit identical.

Each check also runs against a deliberately leaky function to prove the test
would catch a leak, not just pass vacuously.

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

from pipeline.features import add_features  # noqa: E402
from pipeline.targets import add_targets  # noqa: E402

N = 3 * 1440  # 3 days, so same_time_yesterday has history to use
HORIZONS = [1, 5, 15, 60]
# Cut points around every lag / window / horizon boundary, plus start and end.
CUTS = [0, 1, 4, 5, 14, 15, 59, 60, 61, 777, 1439, 1440, 1441, 2500, N - 61, N - 60, N - 2]


def make_load(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-09-29", periods=N, freq="1min")
    load = pd.Series(rng.gamma(2.0, 500.0, N), index=idx, name="load_w")
    load.iloc[rng.choice(N, 40, replace=False)] = np.nan  # scattered missing minutes
    load.iloc[2000:2010] = np.nan  # one longer gap
    return load.to_frame()


def rewrite(df, positions, seed):
    """Replace load at `positions` with different values, including some NaNs.

    New values sit far above the original range so any dependence on them
    (even through a rolling max) is guaranteed to change the output.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    new = 1e6 + rng.gamma(2.0, 900.0, len(positions))
    new[rng.random(len(positions)) < 0.05] = np.nan
    out.iloc[positions, out.columns.get_loc("load_w")] = new
    return out


def future_changes_leak_into_features(fn, df, t):
    base = fn(df)
    changed = fn(rewrite(df, np.arange(t + 1, len(df)), seed=t))
    feature_cols = [c for c in base.columns if c != "load_w"]
    try:
        pd.testing.assert_frame_equal(
            base[feature_cols].iloc[: t + 1], changed[feature_cols].iloc[: t + 1], check_exact=True
        )
        return False
    except AssertionError:
        return True


def past_changes_leak_into_targets(fn, df, t):
    base = fn(df)
    changed = fn(rewrite(df, np.arange(0, t + 1), seed=t))
    target_cols = [c for c in base.columns if c.startswith("y_")]
    try:
        pd.testing.assert_frame_equal(
            base[target_cols].iloc[t:], changed[target_cols].iloc[t:], check_exact=True
        )
        return False
    except AssertionError:
        return True


@pytest.fixture(scope="module")
def load():
    return make_load()


# --- 1. Features never see the future ---------------------------------------

@pytest.mark.parametrize("t", CUTS)
def test_features_ignore_values_after_t(load, t):
    assert not future_changes_leak_into_features(add_features, load, t)


def test_feature_check_catches_a_leak(load):
    def leaky(df):
        out = add_features(df)
        out["centered_mean_5"] = df["load_w"].rolling(5, center=True).mean()
        return out

    assert future_changes_leak_into_features(leaky, load, t=777)


# --- 2. Targets never see the present or past -------------------------------

@pytest.mark.parametrize("H", HORIZONS)
@pytest.mark.parametrize("t", CUTS)
def test_targets_ignore_values_at_or_before_t(load, t, H):
    assert not past_changes_leak_into_targets(lambda df: add_targets(df, H), load, t)


@pytest.mark.parametrize("H", HORIZONS)
def test_target_check_catches_a_leak(load, H):
    def leaky(df):
        out = add_targets(df, H)
        # Off-by-one: window t .. t+H-1 includes the current minute.
        out[f"y_peak_{H}"] = df["load_w"].rolling(
            pd.api.indexers.FixedForwardWindowIndexer(window_size=H), min_periods=H
        ).max()
        return out

    assert past_changes_leak_into_targets(leaky, load, t=777)


# --- Target definitions ------------------------------------------------------

def test_target_values_and_missing_rule():
    idx = pd.date_range("2008-01-01", periods=8, freq="1min")
    df = pd.DataFrame({"load_w": [1.0, 5.0, 2.0, np.nan, 4.0, 3.0, 9.0, 0.0]}, index=idx)
    out = add_targets(df, 2)

    expected_next = [2.0, np.nan, 4.0, 3.0, 9.0, 0.0, np.nan, np.nan]
    # max of (t+1, t+2); NaN if either is missing or past the end
    expected_peak = [5.0, np.nan, np.nan, 4.0, 9.0, 9.0, np.nan, np.nan]
    np.testing.assert_array_equal(out["y_next_2"].to_numpy(), expected_next)
    np.testing.assert_array_equal(out["y_peak_2"].to_numpy(), expected_peak)
