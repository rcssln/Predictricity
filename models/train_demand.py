"""Train and tune LightGBM and XGBoost demand forecasters on train, select on val.

For each H in (5, 10, 15) and target in (y_next, y_peak):
    - build features (pipeline/features.py) and targets (pipeline/targets.py)
    - split with pipeline/split.py make_splits()  -- the test set is never used
    - LightGBM: num_leaves in (31, 63, 127); XGBoost: max_depth in (6, 8, 10)
      2000 trees max, learning rate 0.03, early stopping on validation MAE
    - keep the setting with the best validation MAE per algorithm

Outputs:
    results/tables/step08_tuning_log.csv          every setting and its val score
    models/artifacts/{algo}_{target}_{H}.joblib   best model per algorithm/target/H,
        saved as a dict with the model, feature list, H, target, params and dates

Note: early stopping and model selection both use the validation set, so the
validation scores here are optimistic; the held-out test set gives the unbiased
estimate later.

Run from anywhere (takes a while: 36 fits on ~1.6M rows):
    python models/train_demand.py
"""

import os

# Keep BLAS from grabbing one large buffer per core (fails on low-memory machines).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import gc  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.features import add_features  # noqa: E402
from pipeline.split import DATA_PATH, make_splits  # noqa: E402
from pipeline.targets import add_targets  # noqa: E402

LOG_PATH = ROOT / "results" / "tables" / "step08_tuning_log.csv"
ARTIFACT_DIR = ROOT / "models" / "artifacts"

HORIZONS = [5, 10, 15]
TARGETS = ["y_next", "y_peak"]
N_ESTIMATORS = 2000
LEARNING_RATE = 0.03
EARLY_STOPPING_ROUNDS = 100
SEED = 42
GRID = {
    "lightgbm": [{"num_leaves": n} for n in (31, 63, 127)],
    "xgboost": [{"max_depth": d} for d in (6, 8, 10)],
}


def make_model(algo, params):
    if algo == "lightgbm":
        return lgb.LGBMRegressor(
            n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, metric="l1",
            random_state=SEED, verbose=-1, **params,
        )
    return xgb.XGBRegressor(
        n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, tree_method="hist",
        eval_metric="mae", early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=SEED, **params,
    )


def fit(algo, model, X_train, y_train, X_val, y_val):
    if algo == "lightgbm":
        model.fit(
            X_train, y_train, eval_X=(X_val,), eval_y=(y_val,),
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        return model.best_iteration_
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model.best_iteration + 1


def display(path):
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def append_log(row):
    pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.unlink(missing_ok=True)

    load = pd.read_parquet(DATA_PATH)
    features = add_features(load).astype("float32")
    feature_cols = [c for c in features.columns if c != "load_w"]
    print(f"{len(feature_cols)} features: {', '.join(feature_cols)}\n")

    for H in HORIZONS:
        train, val, _ = make_splits(add_targets(features, H), H)  # test is discarded
        X_train = train[feature_cols].to_numpy("float32")
        X_val = val[feature_cols].to_numpy("float32")
        dates = {
            "train_start": str(train.index[0]), "train_end": str(train.index[-1]),
            "val_start": str(val.index[0]), "val_end": str(val.index[-1]),
            "n_train": len(train), "n_val": len(val),
        }
        ys = {t: (train[f"{t}_{H}"].to_numpy("float32"), val[f"{t}_{H}"].to_numpy("float32"))
              for t in TARGETS}
        del train, val
        gc.collect()

        for target in TARGETS:
            y_train, y_val = ys[target]
            for algo, grid in GRID.items():
                best = None
                for params in grid:
                    t0 = time.perf_counter()
                    model = make_model(algo, params)
                    n_trees = fit(algo, model, X_train, y_train, X_val, y_val)
                    pred = model.predict(X_val)
                    mae = float(np.mean(np.abs(pred - y_val)))
                    rmse = float(np.sqrt(np.mean((pred - y_val) ** 2)))
                    secs = time.perf_counter() - t0

                    (name, value), = params.items()
                    append_log({
                        "H": H, "target": target, "algo": algo, "param": name, "value": value,
                        "best_n_trees": n_trees, "val_mae_w": round(mae, 2),
                        "val_rmse_w": round(rmse, 2), "fit_seconds": round(secs, 1),
                    })
                    print(f"H={H:<2} {target:<6} {algo:<8} {name}={value:<4} "
                          f"trees={n_trees:<5} val MAE={mae:7.2f} W  RMSE={rmse:7.2f} W  ({secs:.0f}s)",
                          flush=True)
                    if best is None or mae < best["val_mae_w"]:
                        best = {"model": model, "params": params, "best_n_trees": n_trees,
                                "val_mae_w": mae, "val_rmse_w": rmse}
                    else:
                        del model
                    gc.collect()

                path = ARTIFACT_DIR / f"{algo}_{target}_{H}.joblib"
                joblib.dump({
                    **best, "algo": algo, "H": H, "target": f"{target}_{H}",
                    "features": feature_cols, "learning_rate": LEARNING_RATE, **dates,
                }, path)
                print(f"  -> best {algo} {best['params']} saved to {display(path)}\n",
                      flush=True)

        del X_train, X_val, ys
        gc.collect()

    log = pd.read_csv(LOG_PATH)
    print(f"Saved {display(LOG_PATH)} ({len(log)} settings)")
    best = log.loc[log.groupby(["H", "target", "algo"])["val_mae_w"].idxmin()]
    print("\nBest per H / target / algorithm (validation):")
    print(best[["H", "target", "algo", "param", "value", "best_n_trees",
                "val_mae_w", "val_rmse_w"]].to_string(index=False))

if __name__ == "__main__":
    main()
