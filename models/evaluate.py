"""Score every saved model and baseline on the TEST set, once (Step 9).

For each H in (5, 10, 15) and target (y_next, y_peak):
    - build float32 features and targets exactly as in training, split with make_splits()
    - baselines from models/baselines.py (computed on the full minute grid, then
      restricted to test rows); the best LightGBM / XGBoost from models/artifacts/
    - all five are scored on the same rows: test rows where every baseline has a prediction
    - MAE, RMSE (W) and skill vs persistence = 1 - MAE_model / MAE_persistence

Outputs:
    results/tables/step09_table_4_1.csv          every H / target / model
    results/tables/step09_table_4_1.md           Table 4.1, one table per target
    data/processed/step09_test_preds.parquet     every prediction (Steps 10 and 11)
    results/handoff/step09_test_preds.parquet    1-minute grid over the test period with
        load_w, y_peak_10, yhat_peak_10_lgbm, yhat_peak_10_xgb (Step 12 simulation)

The test set is scored once. Before that, check the code on validation, which writes
nothing and should reproduce step07_baselines_val.csv and step08_tuning_log.csv:
    python models/evaluate.py --check-val
    python models/evaluate.py
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import gc  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.baselines import baseline_predictions, score  # noqa: E402
from pipeline.features import add_features  # noqa: E402
from pipeline.split import DATA_PATH, PERIODS, make_splits  # noqa: E402
from pipeline.targets import add_targets  # noqa: E402

ARTIFACT_DIR = ROOT / "models" / "artifacts"
TABLE_DIR = ROOT / "results" / "tables"
PREDS_PATH = ROOT / "data" / "processed" / "step09_test_preds.parquet"
HANDOFF_PATH = ROOT / "results" / "handoff" / "step09_test_preds.parquet"

HORIZONS = [5, 10, 15]
TARGETS = ["y_next", "y_peak"]
MODELS = {"lightgbm": "lgbm", "xgboost": "xgb"}
ROW_NAMES = {
    "persistence": "Persistence",
    "same_time_yesterday": "Same time yesterday",
    "moving_avg_15": "Moving average 15",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
}


def predict_all(load, features, H, split):
    """Predictions of every baseline and model on one split, for one H."""
    data = add_targets(features, H)
    _, val, test = make_splits(data, H)
    part = val if split == "val" else test
    preds = pd.DataFrame(baseline_predictions(data, H)).loc[part.index].dropna()
    part = part.loc[preds.index]
    del data, val, test
    gc.collect()

    out = {"load_w": load.loc[part.index].astype("float32")}
    for target in TARGETS:
        col = f"{target}_{H}"
        out[col] = part[col]
        for name in preds.columns:
            out[f"yhat_{target.split('_')[1]}_{H}_{name}"] = preds[name].astype("float32")
        for algo, short in MODELS.items():
            art = joblib.load(ARTIFACT_DIR / f"{algo}_{target}_{H}.joblib")
            assert art["target"] == col and art["H"] == H
            X = part[art["features"]].to_numpy("float32")
            out[f"yhat_{target.split('_')[1]}_{H}_{short}"] = pd.Series(
                art["model"].predict(X), index=part.index, dtype="float32")
            del art, X
            gc.collect()
    return pd.DataFrame(out)


def score_table(preds, H):
    rows = []
    for target in TARGETS:
        y = preds[f"{target}_{H}"].to_numpy("float64")
        short = target.split("_")[1]
        maes = {}
        for name in ROW_NAMES:
            mae, rmse = score(y, preds[f"yhat_{short}_{H}_{name}"].to_numpy("float64"))
            maes[name] = mae
            rows.append({"H": H, "target": target, "model": name, "mae_w": mae,
                         "rmse_w": rmse, "n_rows": len(y)})
        for row in rows[-len(ROW_NAMES):]:
            row["skill_vs_persistence"] = 1 - row["mae_w"] / maes["persistence"]
    return rows


def markdown_table(table, target):
    t = table[table["target"] == target].set_index(["model", "H"])
    lines = [
        f"**Table 4.1{'a' if target == 'y_next' else 'b'}. Test-set errors, `{target}` "
        f"({t['n_rows'].min():,}–{t['n_rows'].max():,} test minutes per H)**",
        "",
        "| Model | MAE 10 min (W) | RMSE 10 min (W) | Skill vs persistence (10 min) "
        "| MAE 5 min (W) | MAE 15 min (W) |",
        "|---|---|---|---|---|---|",
    ]
    for name, label in ROW_NAMES.items():
        r = {H: t.loc[(name, H)] for H in HORIZONS}
        lines.append(
            f"| {label} | {r[10]['mae_w']:.1f} | {r[10]['rmse_w']:.1f} "
            f"| {r[10]['skill_vs_persistence']:.1%} | {r[5]['mae_w']:.1f} | {r[15]['mae_w']:.1f} |")
    return "\n".join(lines)


def main(split):
    load = pd.read_parquet(DATA_PATH)
    features = add_features(load).astype("float32")
    load = load["load_w"]

    rows, frames = [], []
    for H in HORIZONS:
        preds = predict_all(load, features, H, split)
        rows += score_table(preds, H)
        frames.append(preds)
        print(f"H={H}: scored {len(preds):,} {split} rows", flush=True)

    table = pd.DataFrame(rows)
    show = table.assign(mae_w=table["mae_w"].round(1), rmse_w=table["rmse_w"].round(1),
                        skill_vs_persistence=table["skill_vs_persistence"].round(3))
    print()
    print(show.to_string(index=False))

    if split == "val":
        print("\nValidation check only: nothing saved.")
        return

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    show.to_csv(TABLE_DIR / "step09_table_4_1.csv", index=False)
    md = "\n\n".join(markdown_table(table, t) for t in TARGETS)
    (TABLE_DIR / "step09_table_4_1.md").write_text(
        md + "\n\nSkill vs persistence = 1 − MAE_model / MAE_persistence. "
        "All rows for one H are scored on the same test minutes.\n", encoding="utf-8")

    # One wide frame of every prediction; the shared load_w column is kept once.
    all_preds = pd.concat([f.drop(columns="load_w") for f in frames], axis=1)
    all_preds.insert(0, "load_w", load.loc[all_preds.index].astype("float32"))
    PREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_preds.to_parquet(PREDS_PATH)

    # Hand-off for the simulation: every test minute, NaN where nothing was scored.
    start, end = PERIODS["test"]
    grid = pd.date_range(start, end, freq="min", name=all_preds.index.name)
    handoff = pd.DataFrame({"load_w": load.reindex(grid).astype("float32")})
    for col in ["y_peak_10", "yhat_peak_10_lgbm", "yhat_peak_10_xgb"]:
        handoff[col] = all_preds[col].reindex(grid)
    HANDOFF_PATH.parent.mkdir(parents=True, exist_ok=True)
    handoff.to_parquet(HANDOFF_PATH)

    print(f"\n{md}\n")
    for p in [TABLE_DIR / "step09_table_4_1.csv", TABLE_DIR / "step09_table_4_1.md",
              PREDS_PATH, HANDOFF_PATH]:
        print(f"Saved {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main("val" if "--check-val" in sys.argv else "test")
