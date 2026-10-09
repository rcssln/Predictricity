"""Step 15: feature ablation for the H = 10 LightGBM models, on VALIDATION only (Objective 1).

Retrains the H = 10 LightGBM model for y_next and y_peak with one feature group removed at a
time (or only a small group kept), using the setting Step 8 selected for each target (read
from results/tables/step08_tuning_log.csv) and Step 8's training procedure (2,000 trees max,
learning rate 0.03, subsampling 0.8, early stopping on validation MAE). Every run uses the
same train and validation rows (those with all 24 features present), so only the features
differ. The test set is not touched: Table 4.1 does not change.

    All 24 features     the reference
    Without calendar    tod_sin, tod_cos, dow, is_weekend removed
    Without yesterday   same_time_yesterday removed
    Without rolling     mean/std/max over 5, 15 and 60 min removed
    Without long lags   lag_15, lag_30, lag_60 removed
    Without diff        diff_1 removed
    Recent only         now, lag_1, lag_2, lag_3, lag_5 and diff_1 kept
    Now only            now kept

Validation scores are optimistic (early stopping uses validation, as in Step 8); the
comparison between runs is what matters.

Outputs:
    results/tables/step15_feature_ablation.csv   every run (written after each fit)
    results/tables/step15_feature_ablation.md    the table, with the MAE change vs all features

Run from anywhere (the full run takes about 45-60 min on an 8 GB laptop):
    python models/feature_ablation.py --max-trees 20   # smoke test, about a minute
    python models/feature_ablation.py
    python models/feature_ablation.py --resume         # continue an interrupted run
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import gc  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import train_demand  # noqa: E402
from pipeline.features import add_features  # noqa: E402
from pipeline.split import DATA_PATH, make_splits  # noqa: E402
from pipeline.targets import add_targets  # noqa: E402

H = 10
TARGETS = ["y_next", "y_peak"]
TUNING_LOG = ROOT / "results" / "tables" / "step08_tuning_log.csv"
CSV_PATH = ROOT / "results" / "tables" / "step15_feature_ablation.csv"
MD_PATH = ROOT / "results" / "tables" / "step15_feature_ablation.md"

CALENDAR = ["tod_sin", "tod_cos", "dow", "is_weekend"]
ROLLING = [f"{s}_{w}" for w in (5, 15, 60) for s in ("mean", "std", "max")]
LONG_LAGS = ["lag_15", "lag_30", "lag_60"]
RECENT = ["now", "lag_1", "lag_2", "lag_3", "lag_5", "diff_1"]


def runs(all_features):
    """(run name, what changes, feature list) for every ablation run."""
    drop = lambda cols: [c for c in all_features if c not in cols]  # noqa: E731
    keep = lambda cols: [c for c in all_features if c in cols]      # noqa: E731
    return [
        ("All 24 features", "the reference", all_features),
        ("Without calendar", "time of day, day of week, weekend flag removed", drop(CALENDAR)),
        ("Without yesterday", "load at this time yesterday removed", drop(["same_time_yesterday"])),
        ("Without rolling", "5/15/60-min mean, std, max removed", drop(ROLLING)),
        ("Without long lags", "lags 15, 30, 60 removed", drop(LONG_LAGS)),
        ("Without diff", "the 1-minute change removed", drop(["diff_1"])),
        ("Recent only", "now, lags 1-5 and diff_1 kept", keep(RECENT)),
        ("Now only", "the current load only", keep(["now"])),
    ]


def step8_setting(target):
    """The LightGBM setting Step 8 selected for this target at H = 10 (lowest val MAE)."""
    log = pd.read_csv(TUNING_LOG)
    rows = log[(log["H"] == H) & (log["target"] == target) & (log["algo"] == "lightgbm")]
    best = rows.loc[rows["val_mae_w"].idxmin()]
    return {"objective": best["objective"], best["param"]: int(best["value"])}


def markdown(table, max_trees):
    lines = ["**Feature ablation, H = 10 LightGBM, validation set (Jan–Jun 2010)**", "",
             "| Run | Change | Features | y_next val MAE (W) | Δ vs all | y_peak val MAE (W) "
             "| Δ vs all |", "|---|---|---|---|---|---|---|"]
    ref = {t: table[(table["run"] == "All 24 features") & (table["target"] == t)]["val_mae_w"].iloc[0]
           for t in TARGETS}
    for run in table["run"].unique():
        r = {t: table[(table["run"] == run) & (table["target"] == t)].iloc[0] for t in TARGETS}
        cells = []
        for t in TARGETS:
            d = r[t]["val_mae_w"] - ref[t]
            cells.append(f"{r[t]['val_mae_w']:.1f} | {d:+.1f} ({d / ref[t]:+.1%})")
        lines.append(f"| {run} | {r[TARGETS[0]]['change']} | {r[TARGETS[0]]['n_features']} "
                     f"| {' | '.join(cells)} |")
    best = {t: table[table["target"] == t].loc[lambda x: x["val_mae_w"].idxmin(), "run"]
            for t in TARGETS}
    note = (f"Lowest validation MAE: {best['y_next']} (y_next), {best['y_peak']} (y_peak). Each "
            "run uses the Step 8 setting for its target and the same rows; early stopping uses "
            "validation, so the scores are optimistic and only the comparison between runs "
            "matters. The test set was not used.")
    if max_trees:
        note = f"SMOKE TEST ({max_trees} trees max): these numbers mean nothing. " + note
    return "\n".join(lines + ["", note]) + "\n"


def main():
    sys.stdout.reconfigure(errors="replace")
    args = sys.argv[1:]
    max_trees = int(args[args.index("--max-trees") + 1]) if "--max-trees" in args else None
    done = pd.DataFrame()
    if "--resume" in args and CSV_PATH.exists():
        done = pd.read_csv(CSV_PATH)
        print(f"Resuming: {len(done)} runs already done")
    else:
        CSV_PATH.unlink(missing_ok=True)
    if max_trees:
        train_demand.N_ESTIMATORS = max_trees

    features = add_features(pd.read_parquet(DATA_PATH)).astype("float32")
    all_features = [c for c in features.columns if c != "load_w"]
    train, val, _ = make_splits(add_targets(features, H), H)   # test is discarded
    del features
    gc.collect()

    rows = done.to_dict("records")
    for target in TARGETS:
        params = step8_setting(target)
        y_train = train[f"{target}_{H}"].to_numpy("float32")
        y_val = val[f"{target}_{H}"].to_numpy("float32")
        for name, change, cols in runs(all_features):
            if any(r["run"] == name and r["target"] == target for r in rows):
                continue
            t0 = time.perf_counter()
            model = train_demand.make_model("lightgbm", params)
            n_trees = train_demand.fit("lightgbm", model, train[cols].to_numpy("float32"), y_train,
                                       val[cols].to_numpy("float32"), y_val)
            pred = model.predict(val[cols].to_numpy("float32"))
            mae = float(np.mean(np.abs(pred - y_val)))
            rmse = float(np.sqrt(np.mean((pred - y_val) ** 2)))
            rows.append({"run": name, "change": change, "target": target, "n_features": len(cols),
                         "params": str(params), "best_n_trees": n_trees, "val_mae_w": round(mae, 2),
                         "val_rmse_w": round(rmse, 2),
                         "fit_seconds": round(time.perf_counter() - t0, 1)})
            pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
            print(f"{target:<6} {name:<18} features={len(cols):<2} trees={n_trees:<5} "
                  f"val MAE={mae:7.2f} W  RMSE={rmse:7.2f} W  ({rows[-1]['fit_seconds']:.0f}s)",
                  flush=True)
            del model
            gc.collect()

    table = pd.DataFrame(rows)
    order = [r[0] for r in runs(all_features)]
    table = table.sort_values(["target", "run"], key=lambda s: s.map(
        {n: i for i, n in enumerate(order)}) if s.name == "run" else s).reset_index(drop=True)
    table.to_csv(CSV_PATH, index=False)
    md = markdown(table, max_trees)
    MD_PATH.write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Saved {CSV_PATH.relative_to(ROOT)}\nSaved {MD_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
