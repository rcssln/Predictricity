"""Clean the StoreNet 2020 households into the UCI format (Step 16).

Input:  data/raw/storenet/H{n}_W.csv          (StoreNet, 1-minute, power in W)
Output: data/processed/storenet/H{n}_1min.parquet   (timestamp index on a strict 1-minute
        grid over calendar 2020, one float32 column load_w, like uci_1min.parquet)
        results/tables/step16_storenet_summary.md

Steps, as pre-registered in docs/step16_replication.md:
    - parse the timestamp and keep the consumption column, "Consumption(W)", as load_w
    - keep the first copy of any repeated minute
    - reindex to every minute of calendar 2020
    - set every run of >= ZERO_RUN_MIN consecutive minutes of exactly 0 W to missing
      (logging outages recorded as zero, e.g. 28-30 April 2020)
    - fill gaps of <= 5 minutes with fill_short_gaps() from clean_uci.py; longer gaps stay

Houses with negative load or long zero runs are flagged in the summary, never dropped.

Run from anywhere:
    python pipeline/clean_storenet.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.clean_uci import MAX_GAP_MIN, fill_short_gaps  # noqa: E402

RAW_DIR = ROOT / "data" / "raw" / "storenet"
OUT_DIR = ROOT / "data" / "processed" / "storenet"
SUMMARY_PATH = ROOT / "results" / "tables" / "step16_storenet_summary.md"
UCI_SUMMARY = ROOT / "results" / "tables" / "step03_data_summary.md"

# Pre-registered: every house with >= 99% of 2020's minutes holding a valid consumption value.
HOUSES = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11", "H12", "H13",
          "H15", "H17", "H18", "H19", "H20"]
CONSUMPTION = "Consumption(W)"
PRODUCTION = "Production(W)"
ZERO_RUN_MIN = 60
GRID = pd.date_range("2020-01-01 00:00", "2020-12-31 23:59", freq="min", name="timestamp")


def runs_of(mask):
    """Length of the run each element belongs to (runs of equal values)."""
    run_id = (mask != mask.shift()).cumsum()
    return mask.groupby(run_id).transform("size")


def longest_gap_minutes(s):
    is_na = s.isna()
    return int(runs_of(is_na)[is_na].max()) if is_na.any() else 0


def clean_house(house):
    raw = pd.read_csv(RAW_DIR / f"{house}_W.csv", skipinitialspace=True,
                      usecols=["date", CONSUMPTION, PRODUCTION])
    raw["timestamp"] = pd.to_datetime(raw["date"], format="%Y-%m-%d %H:%M:%S")
    raw = raw.set_index("timestamp").sort_index()
    dupes = int(raw.index.duplicated().sum())
    raw = raw[~raw.index.duplicated(keep="first")]
    has_pv = bool((raw[PRODUCTION] > 0).any())

    load = raw[CONSUMPTION].astype("float32").reindex(GRID)
    missing_before = int(load.isna().sum())
    negatives = int((load < 0).sum())

    zero = load == 0
    long_zero = zero & (runs_of(zero) >= ZERO_RUN_MIN)
    zero_minutes = int(long_zero.sum())
    load = load.mask(long_zero)

    load = fill_short_gaps(load, MAX_GAP_MIN)
    out = load.to_frame("load_w")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / f"{house}_1min.parquet")

    flags = []
    if negatives:
        flags.append(f"{negatives} negative minutes")
    if zero_minutes:
        flags.append("long zero runs set missing")
    return {
        "house": house, "rows": len(out), "duplicates_dropped": dupes,
        "missing_before": missing_before, "zero_run_minutes": zero_minutes,
        "missing_after": int(out["load_w"].isna().sum()),
        "longest_gap": longest_gap_minutes(out["load_w"]),
        "mean": float(out["load_w"].mean()), "median": float(out["load_w"].median()),
        "p95": float(out["load_w"].quantile(0.95)), "pv": "yes" if has_pv else "no",
        "flags": "; ".join(flags) or "none",
    }


def uci_row():
    """The UCI values from the Step 3 summary table, for comparison."""
    text = UCI_SUMMARY.read_text(encoding="utf-8")
    get = lambda label: next(  # noqa: E731
        line.split("|")[2].strip() for line in text.splitlines() if line.startswith(f"| {label}"))
    return (f"| UCI (Step 3, 2006–2010) | {get('Row count')} | – "
            f"| {get('Missing rows before cleaning')} | – | {get('Missing rows after cleaning')} "
            f"| {get('Longest gap (minutes)')} | {get('Mean load_w (W)')} "
            f"| {get('Median load_w (W)')} | {get('95th percentile load_w (W)')} | no | – |")


def main():
    rows = []
    for house in HOUSES:
        rows.append(clean_house(house))
        print(f"{house}: done", flush=True)
    t = pd.DataFrame(rows)
    lines = [
        "# Step 16: StoreNet data summary (calendar 2020, after cleaning)", "",
        "Source: data/processed/storenet/H{n}_1min.parquet (18 pre-registered houses). "
        "Missing before cleaning counts minutes of 2020 with no valid consumption value; "
        f"zero-run minutes are runs of >= {ZERO_RUN_MIN} min at exactly 0 W, set to missing; "
        f"gaps of <= {MAX_GAP_MIN} min are then filled. Load statistics exclude missing values. "
        "Most zero-run minutes fall on the 5th of each month from January to May (almost every "
        "house reads 0 W from 00:00 for about a day) and on 28-30 April 2020: logging artifacts "
        "found while cleaning, which the pre-registered rule treats as missing.",
        "",
        "| House | Rows | Duplicates dropped | Missing before cleaning | Zero-run minutes set "
        "missing | Missing after cleaning | Longest gap (min) | Mean load (W) | Median load (W) "
        "| 95th percentile (W) | PV | Flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['house']} | {r['rows']:,} | {r['duplicates_dropped']} | {r['missing_before']:,} "
            f"| {r['zero_run_minutes']:,} | {r['missing_after']:,} | {r['longest_gap']:,} "
            f"| {r['mean']:.1f} | {r['median']:.1f} | {r['p95']:.1f} | {r['pv']} | {r['flags']} |")
    lines.append(f"| **Median of 18 houses** | | | | | | | {t['mean'].median():.1f} "
                 f"| {t['median'].median():.1f} | {t['p95'].median():.1f} | {(t['pv'] == 'yes').sum()} of 18 | |")
    lines.append(uci_row())
    md = "\n".join(lines) + "\n"
    SUMMARY_PATH.write_text(md, encoding="utf-8")
    sys.stdout.reconfigure(errors="replace")
    print("\n" + md)
    print(f"Saved {SUMMARY_PATH.relative_to(ROOT)} and {len(rows)} files in "
          f"{OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
