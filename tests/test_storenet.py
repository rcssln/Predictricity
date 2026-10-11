"""Tests for the Step 16 StoreNet replication data: cleaned files and split periods.

The file-based tests are skipped when data/processed/storenet/ is missing (run
pipeline/clean_storenet.py first), so CI and teammates without the data still pass.

Run from the project root:
    python -m pytest tests
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.replicate_storenet import STORENET_PERIODS  # noqa: E402
from pipeline.clean_storenet import HOUSES, OUT_DIR  # noqa: E402

needs_data = pytest.mark.skipif(not OUT_DIR.exists(),
                                reason="data/processed/storenet/ missing; run clean_storenet.py")


def test_split_periods_do_not_overlap_and_stay_inside_2020():
    spans = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in
             (STORENET_PERIODS[k] for k in ("train", "val", "test"))]
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end < start
    assert spans[0][0] >= pd.Timestamp("2020-01-01")
    assert spans[-1][1] <= pd.Timestamp("2020-12-31 23:59")


@needs_data
@pytest.mark.parametrize("house", HOUSES)
def test_cleaned_file_has_strict_minute_index(house):
    df = pd.read_parquet(OUT_DIR / f"{house}_1min.parquet")
    idx = df.index
    assert isinstance(idx, pd.DatetimeIndex)
    assert not idx.has_duplicates
    assert idx.is_monotonic_increasing
    assert (idx[1:] - idx[:-1] == pd.Timedelta("1min")).all()
    assert idx[0] == pd.Timestamp("2020-01-01 00:00")
    assert idx[-1] == pd.Timestamp("2020-12-31 23:59")
    assert list(df.columns) == ["load_w"]


@needs_data
@pytest.mark.parametrize("house", HOUSES)
def test_load_is_never_negative(house):
    load = pd.read_parquet(OUT_DIR / f"{house}_1min.parquet")["load_w"]
    assert (load.dropna() >= 0).all()
