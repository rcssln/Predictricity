"""Build model features from the raw per-minute appliance readings.

Input:  data/raw/simulated_power.csv  (long format: one row per appliance per minute)
Output: data/processed/features.csv   (wide format: one row per timestamp)

Columns:
    timestamp
    <appliance>_watts, <appliance>_state     for each appliance
    total_watts                              sum of appliance watts
    lag_1, lag_5, lag_15                     total_watts N minutes earlier
    rolling_mean_5, rolling_std_5            over the last 5 minutes (incl. current)
    rolling_mean_15                          over the last 15 minutes (incl. current)
    rate_of_change                           total_watts - lag_1 (W per minute)
    hour_sin, hour_cos                       cyclical time of day (fractional hour)
    day_of_week                              0 = Monday ... 6 = Sunday

Rows with any missing value (the first 15 minutes, which lack full lag
history, or any timestamp missing an appliance reading) are dropped.

Standard library only. Run from anywhere:
    python pipeline/feature_engineering.py
"""

import csv
import math
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "raw" / "simulated_power.csv"
OUT_PATH = ROOT / "data" / "processed" / "features.csv"

APPLIANCES = ["refrigerator", "water_heater", "electric_fan", "washing_machine"]
LAGS = [1, 5, 15]
TS_FORMAT = "%Y-%m-%d %H:%M:%S"

COLUMNS = (
    ["timestamp"]
    + [f"{a}_watts" for a in APPLIANCES]
    + [f"{a}_state" for a in APPLIANCES]
    + ["total_watts"]
    + [f"lag_{k}" for k in LAGS]
    + ["rolling_mean_5", "rolling_std_5", "rolling_mean_15", "rate_of_change"]
    + ["hour_sin", "hour_cos", "day_of_week"]
)


def pivot(path):
    """Long rows -> {timestamp: {"<appliance>_watts": float, "<appliance>_state": int}}."""
    wide = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rec = wide.setdefault(row["timestamp"], {})
            rec[f"{row['appliance']}_watts"] = float(row["watts"])
            rec[f"{row['appliance']}_state"] = int(row["appliance_state"])
    return wide


def window(values, i, size):
    """The last `size` values ending at index i, or None if history is short or incomplete."""
    if i + 1 < size:
        return None
    vals = values[i + 1 - size : i + 1]
    return None if None in vals else vals


def build_features(wide):
    timestamps = sorted(wide)
    records = []
    for ts in timestamps:
        rec = {"timestamp": ts}
        for a in APPLIANCES:
            rec[f"{a}_watts"] = wide[ts].get(f"{a}_watts")
            rec[f"{a}_state"] = wide[ts].get(f"{a}_state")
        watts = [rec[f"{a}_watts"] for a in APPLIANCES]
        rec["total_watts"] = None if None in watts else sum(watts)
        records.append(rec)

    total = [r["total_watts"] for r in records]
    for i, rec in enumerate(records):
        for k in LAGS:
            rec[f"lag_{k}"] = total[i - k] if i >= k else None

        w5, w15 = window(total, i, 5), window(total, i, 15)
        rec["rolling_mean_5"] = statistics.fmean(w5) if w5 else None
        rec["rolling_std_5"] = statistics.stdev(w5) if w5 else None  # sample std, like pandas
        rec["rolling_mean_15"] = statistics.fmean(w15) if w15 else None

        prev = rec["lag_1"]
        rec["rate_of_change"] = (
            None if prev is None or total[i] is None else total[i] - prev
        )

        dt = datetime.strptime(rec["timestamp"], TS_FORMAT)
        hour = dt.hour + dt.minute / 60
        rec["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        rec["hour_cos"] = math.cos(2 * math.pi * hour / 24)
        rec["day_of_week"] = dt.weekday()

    return [r for r in records if all(r[c] is not None for c in COLUMNS)]


def fmt(col, value):
    if col.endswith("_state") or col in ("timestamp", "day_of_week"):
        return value
    if col.startswith("hour_"):
        return f"{value:.6f}"
    return f"{value:.3f}"


def main():
    rows = build_features(pivot(IN_PATH))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for r in rows:
            writer.writerow([fmt(c, r[c]) for c in COLUMNS])

    print(f"Wrote {OUT_PATH.relative_to(ROOT)}\n")
    print(f"Shape: ({len(rows)}, {len(COLUMNS)})\n")
    print("Columns:")
    for c in COLUMNS:
        print(f"  {c}")
    # 24 columns is too wide for a terminal, so show the first rows transposed.
    print("\nFirst 3 rows:")
    head = rows[:3]
    for c in COLUMNS:
        print(f"  {c:<22}" + "".join(f"{str(fmt(c, r[c])):>22}" for r in head))


if __name__ == "__main__":
    main()
