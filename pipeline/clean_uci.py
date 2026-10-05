"""Clean the UCI Individual Household Electric Power Consumption dataset.

Input:  data/raw/household_power_consumption.txt  (';'-separated, '?' or blank = missing)
Output: data/processed/uci_1min.parquet           (timestamp index, one column: load_w)

Steps:
    - Combine Date (dd/mm/yyyy) and Time into a timestamp index
    - Keep Global_active_power and convert kW -> W as load_w
    - Reindex to a strict 1-minute frequency (missing minutes become NaN)
    - Linearly interpolate gaps of MAX_GAP_MIN minutes or less; longer gaps stay NaN

Run from anywhere:
    python pipeline/clean_uci.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "raw" / "household_power_consumption.txt"
OUT_PATH = ROOT / "data" / "processed" / "uci_1min.parquet"

MAX_GAP_MIN = 5


def fill_short_gaps(s, max_gap):
    """Interpolate only NaN runs of length <= max_gap that have data on both sides.

    interpolate(limit=n) alone would also fill the first n minutes of longer
    gaps, so mask by each run's total length instead.
    """
    is_na = s.isna()
    run_id = (is_na != is_na.shift()).cumsum()
    run_len = is_na.groupby(run_id).transform("size")
    short_gap = is_na & (run_len <= max_gap)
    filled = s.interpolate(limit_area="inside")
    return s.where(~short_gap, filled)


def main():
    raw = pd.read_csv(IN_PATH, sep=";", na_values=["?", ""], low_memory=False)
    raw["timestamp"] = pd.to_datetime(
        raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df = raw.set_index("timestamp")[["Global_active_power"]].astype("float32")
    df["load_w"] = df["Global_active_power"] * 1000

    df = df[["load_w"]].sort_index()
    dupes = df.index.duplicated().sum()
    if dupes:
        df = df[~df.index.duplicated(keep="first")]

    observed_na = int(df["load_w"].isna().sum())
    df = df.asfreq("1min")
    before = int(df["load_w"].isna().sum())
    df["load_w"] = fill_short_gaps(df["load_w"], MAX_GAP_MIN)
    after = int(df["load_w"].isna().sum())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH)

    print(f"Wrote {OUT_PATH.relative_to(ROOT)}\n")
    print(f"Raw rows:                    {len(raw):,}")
    print(f"Duplicate timestamps dropped: {dupes:,}")
    print(f"Range:                       {df.index[0]} -> {df.index[-1]}")
    print(f"Rows at 1-min frequency:     {len(df):,}")
    print(f"Missing in source ('?'/blank): {observed_na:,}")
    print(f"Missing after reindex:       {before:,}")
    print(f"Filled (gaps <= {MAX_GAP_MIN} min):      {before - after:,}")
    print(f"Left empty (longer gaps):    {after:,}")
    print(f"\n{df['load_w'].describe().to_string()}")


if __name__ == "__main__":
    main()
