"""Naive baselines scored on the validation set.

For each horizon H in (5, 10, 15) and each target (y_next_H, y_peak_H), score
persistence, same-time-yesterday and a 15-minute moving average with MAE and
RMSE in watts. Saves results/tables/step07_baselines_val.csv.

Predictions are computed on the full strict 1-minute grid (shift counts rows,
so it must run before any rows are dropped), then restricted to the val rows
from make_splits. All baselines are scored on the same rows: val rows where
every baseline has a prediction.

Run from anywhere:
    python models/baselines.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.features import add_features  # noqa: E402
from pipeline.split import DATA_PATH, make_splits  # noqa: E402
from pipeline.targets import add_targets  # noqa: E402

OUT_PATH = ROOT / "results" / "tables" / "step07_baselines_val.csv"
HORIZONS = [5, 10, 15]


def baseline_predictions(d, H):
    return {
        "persistence": d["now"],
        "same_time_yesterday": d["load_w"].shift(1440 - H),
        "moving_avg_15": d["mean_15"],
    }


def score(y_true, y_pred):
    err = y_pred - y_true
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))


def main():
    features = add_features(pd.read_parquet(DATA_PATH))

    rows = []
    for H in HORIZONS:
        data = add_targets(features, H)
        _, val, _ = make_splits(data, H)
        preds = pd.DataFrame(baseline_predictions(data, H)).loc[val.index]
        preds = preds.dropna()
        for target in (f"y_next_{H}", f"y_peak_{H}"):
            y = val.loc[preds.index, target].to_numpy("float64")
            for name in preds.columns:
                mae, rmse = score(y, preds[name].to_numpy("float64"))
                rows.append({
                    "H": H, "target": target.rsplit("_", 1)[0], "baseline": name,
                    "mae_w": round(mae, 1), "rmse_w": round(rmse, 1), "n_rows": len(y),
                })
        print(f"H={H}: scored {len(preds):,} of {len(val):,} val rows")

    table = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {OUT_PATH.relative_to(ROOT)}\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
