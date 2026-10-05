import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer

def add_targets(df: pd.DataFrame, H: int, col: str = "load_w") -> pd.DataFrame:
    """Add forecast targets for horizon H minutes (expects a strict 1-minute index).

    y_next_{H}: load at t+H.
    y_peak_{H}: max load over t+1 .. t+H; NaN if any of those minutes is
                missing or the window runs past the end of the data.
    """
    out = df.copy()
    s = out[col]
    out[f"y_next_{H}"] = s.shift(-H)
    window = FixedForwardWindowIndexer(window_size=H)
    out[f"y_peak_{H}"] = s.shift(-1).rolling(window, min_periods=H).max()
    return out
