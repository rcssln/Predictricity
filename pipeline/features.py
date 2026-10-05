import numpy as np
import pandas as pd

def add_features(df: pd.DataFrame, col: str = "load_w") -> pd.DataFrame:
    out = df.copy()
    s = out[col]
    out["now"] = s
    for lag in [1, 2, 3, 5, 10, 15, 30, 60]:
        out[f"lag_{lag}"] = s.shift(lag)
    for w in [5, 15, 60]:
        out[f"mean_{w}"] = s.rolling(w).mean()
        out[f"std_{w}"] = s.rolling(w).std()
        out[f"max_{w}"] = s.rolling(w).max()
    out["diff_1"] = s.diff(1)
    out["same_time_yesterday"] = s.shift(1440)
    minute = out.index.hour * 60 + out.index.minute
    out["tod_sin"] = np.sin(2 * np.pi * minute / 1440)
    out["tod_cos"] = np.cos(2 * np.pi * minute / 1440)
    out["dow"] = out.index.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    return out
