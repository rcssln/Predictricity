"""Train a LightGBM regressor to predict total_watts from engineered features.

Input:  data/processed/features.csv  (from pipeline/feature_engineering.py)
Output: models/lgbm_model.pkl        (joblib-serialized LGBMRegressor)

Split is chronological: the first 80% of rows train, the last 20% test.

Run from anywhere:
    python pipeline/train_model.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "processed" / "features.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "lgbm_model.pkl"

TARGET = "total_watts"
TRAIN_FRACTION = 0.8


def main():
    df = pd.read_csv(IN_PATH, parse_dates=["timestamp"]).sort_values("timestamp")

    X = df.drop(columns=["timestamp", TARGET])
    y = df[TARGET]

    split = int(len(df) * TRAIN_FRACTION)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LGBMRegressor(verbose=-1)  # defaults; just silence training logs
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    importance = (
        pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    importance.index += 1

    print(f"Saved model to {MODEL_PATH.relative_to(ROOT)}\n")
    print(f"Train size: {len(X_train)}  ({df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[split - 1]})")
    print(f"Test size:  {len(X_test)}  ({df['timestamp'].iloc[split]} -> {df['timestamp'].iloc[-1]})")
    print(f"MAE:  {mae:.3f} W")
    print(f"RMSE: {rmse:.3f} W\n")
    print("Top 10 features by importance (split count):")
    print(importance.to_string())


if __name__ == "__main__":
    main()
