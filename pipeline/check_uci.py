"""Summarize the cleaned UCI data and save a Markdown table.

Input:  data/processed/uci_1min.parquet            (from pipeline/clean_uci.py)
        data/raw/household_power_consumption.txt   (to count missing rows before cleaning)
Output: results/tables/step03_data_summary.md

"Missing before cleaning" counts minutes with no load value on the strict
1-minute grid before interpolation: '?'/blank readings plus any minutes
absent from the raw file. "Missing after cleaning" counts NaNs left in the
parquet. The longest gap is the longest NaN run after cleaning.

Run from anywhere:
    python pipeline/check_uci.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT / "data" / "processed" / "uci_1min.parquet"
RAW_PATH = ROOT / "data" / "raw" / "household_power_consumption.txt"
OUT_PATH = ROOT / "results" / "tables" / "step03_data_summary.md"


def missing_before_cleaning(path):
    raw = pd.read_csv(
        path, sep=";", na_values=["?", ""], usecols=["Date", "Time", "Global_active_power"],
        low_memory=False,
    )
    ts = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
    s = pd.Series(raw["Global_active_power"].to_numpy(), index=ts).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return int(s.asfreq("1min").isna().sum())


def longest_gap_minutes(s):
    is_na = s.isna()
    if not is_na.any():
        return 0
    run_id = (is_na != is_na.shift()).cumsum()
    return int(is_na.groupby(run_id).sum().max())


def main():
    load = pd.read_parquet(CLEAN_PATH)["load_w"]

    rows = [
        ("Row count", f"{len(load):,}"),
        ("First timestamp", str(load.index[0])),
        ("Last timestamp", str(load.index[-1])),
        ("Missing rows before cleaning", f"{missing_before_cleaning(RAW_PATH):,}"),
        ("Missing rows after cleaning", f"{int(load.isna().sum()):,}"),
        ("Longest gap (minutes)", f"{longest_gap_minutes(load):,}"),
        ("Mean load_w (W)", f"{load.mean():.1f}"),
        ("Median load_w (W)", f"{load.median():.1f}"),
        ("95th percentile load_w (W)", f"{load.quantile(0.95):.1f}"),
        ("99th percentile load_w (W)", f"{load.quantile(0.99):.1f}"),
    ]

    width = max(len(name) for name, _ in rows)
    lines = [
        "# Step 03: UCI data summary",
        "",
        f"Source: `{CLEAN_PATH.relative_to(ROOT).as_posix()}`. "
        "Statistics exclude missing values.",
        "",
        f"| {'Metric':<{width}} | Value |",
        f"| {'-' * width} | ----: |",
    ]
    lines += [f"| {name:<{width}} | {value} |" for name, value in rows]
    md = "\n".join(lines) + "\n"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(md, encoding="utf-8")

    print(md)
    print(f"Saved {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
