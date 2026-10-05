"""Simulate one real-time inference step as it would run on the Raspberry Pi.

1. Load the trained model (models/lgbm_model.pkl).
2. Load the last 15 rows of data/processed/features.csv as recent sensor history.
3. Treat the most recent row as the current reading.
4. Predict the next minute's total_watts.
5. Ask decision/shed.py whether to shed an appliance.

NOTE: models/lgbm_model.pkl is currently trained with the same minute's
total_watts as its target, and its inputs include the per-appliance watts
that sum to it. Until the model is retrained on a next-minute target, the
"predicted next" value will closely mirror the current total.

Run from anywhere:
    python services/inference.py
"""

import sys
from collections import deque
from io import StringIO
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision.shed import CRITICAL, NON_CRITICAL, should_shed  # noqa: E402

MODEL_PATH = ROOT / "models" / "lgbm_model.pkl"
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
HISTORY_ROWS = 15


def load_recent_history(path, n):
    """Return the last n rows without holding the whole file in memory."""
    with path.open() as f:
        header = f.readline()
        tail = deque(f, maxlen=n)
    return pd.read_csv(StringIO(header + "".join(tail)), parse_dates=["timestamp"])


def main():
    model = joblib.load(MODEL_PATH)
    history = load_recent_history(FEATURES_PATH, HISTORY_ROWS)
    current = history.iloc[-1]

    # Feed features in exactly the order the model was trained on.
    X = history[model.feature_name_].iloc[[-1]]
    predicted = float(model.predict(X)[0])

    states = {a: int(current[f"{a}_state"]) for a in CRITICAL + NON_CRITICAL}
    decision = should_shed(predicted, states)

    print(f"History loaded:          {len(history)} rows "
          f"({history['timestamp'].iloc[0]} -> {history['timestamp'].iloc[-1]})")
    print(f"Current timestamp:       {current['timestamp']}")
    print(f"Current appliance states: {states}")
    print(f"Current total_watts:     {current['total_watts']:.1f} W")
    print(f"Predicted next total:    {predicted:.1f} W")
    print(f"Shedding decision:       {decision if decision else 'No shedding needed'}")


if __name__ == "__main__":
    main()
