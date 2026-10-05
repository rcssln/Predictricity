"""Chronological train / val / test split for the UCI 1-minute data.

    train: Dec 2006 - Dec 2009
    val:   Jan 2010 - Jun 2010
    test:  Jul 2010 - Nov 2010

The last H rows of train and of val are dropped so no target window
(t+1 .. t+H) reaches into the next period. Rows with any missing feature
or target are then dropped.

Run from anywhere (builds features + targets from uci_1min.parquet):
    python pipeline/split.py          # H = 10
    python pipeline/split.py 60
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.features import add_features  # noqa: E402
from pipeline.targets import add_targets  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "uci_1min.parquet"

PERIODS = {
    "train": ("2006-12-01", "2009-12-31 23:59"),
    "val": ("2010-01-01", "2010-06-30 23:59"),
    "test": ("2010-07-01", "2010-11-30 23:59"),
}


def make_splits(data: pd.DataFrame, H: int):
    """Split features+targets on a strict 1-minute index into (train, val, test).

    Trimming happens on the full minute grid, before missing rows are dropped,
    so "last H rows" always means the last H minutes of the period.
    """
    parts = {}
    for name, (start, end) in PERIODS.items():
        part = data.loc[start:end]
        if name != "test":
            part = part.iloc[:-H] if H > 0 else part
        parts[name] = part.dropna()
    return parts["train"], parts["val"], parts["test"]


def build_dataset(H: int) -> pd.DataFrame:
    load = pd.read_parquet(DATA_PATH)
    return add_targets(add_features(load), H)


def main(H: int = 10):
    data = build_dataset(H)
    train, val, test = make_splits(data, H)

    print(f"H = {H} min | {data.shape[1]} columns "
          f"(targets: y_next_{H}, y_peak_{H})\n")
    print(f"{'period':<6} {'rows':>10}   {'first timestamp':<20} {'last timestamp':<20}")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"{name:<6} {len(part):>10,}   {str(part.index[0]):<20} {str(part.index[-1]):<20}")
    total = len(train) + len(val) + len(test)
    print(f"{'total':<6} {total:>10,}   ({len(data) - total:,} of {len(data):,} rows dropped)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
