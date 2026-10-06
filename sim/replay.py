"""Step 12.2: replay UCI test-period demand on the testbed's switched zones.

    1. Scale the UCI test-period load so its 99th percentile equals demand_p99_w.
    2. Each minute, choose the on/off combination of zones whose total is closest to the
       scaled demand. Critical comes first: a sheddable zone is only on when critical is,
       so the choices are "everything off" or critical plus any set of sheddable zones.
    3. Scale the saved H = 10 y_peak forecasts (LightGBM and XGBoost) by the same factor,
       if forecast_file in sim/config.yaml exists.

All settings: sim/config.yaml. Missing minutes stay NaN; the simulation only uses days
with every minute present.

Run from anywhere (plots one example day):
    python sim/replay.py
"""

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.split import DATA_PATH, PERIODS  # noqa: E402
from sim.config import FIG_DIR, load_config  # noqa: E402

FORECAST_COLS = {"yhat_peak_10_lgbm": "fc_lgbm_w", "yhat_peak_10_xgb": "fc_xgb_w"}


def zone_combinations(zones):
    """All allowed on/off sets: nothing, or critical plus any set of sheddable zones."""
    critical = [z for z, v in zones.items() if not v["sheddable"]]
    sheddable = [z for z, v in zones.items() if v["sheddable"]]
    combos = [()]
    for k in range(len(sheddable) + 1):
        for extra in itertools.combinations(sheddable, k):
            combos.append(tuple(critical) + extra)
    return combos


def assign_zones(demand_w, zones):
    """Boolean frame (one column per zone): the combination closest to each minute's demand.

    Ties go to the combination listed first (fewer zones). NaN demand -> all False.
    """
    combos = zone_combinations(zones)
    totals = np.array([sum(zones[z]["watts"] for z in c) for c in combos], dtype=float)
    d = demand_w.to_numpy(dtype=float)
    best = np.abs(d[:, None] - totals[None, :]).argmin(axis=1)
    out = pd.DataFrame(False, index=demand_w.index, columns=list(zones))
    for i, combo in enumerate(combos):
        rows = (best == i) & ~np.isnan(d)
        for z in combo:
            out.loc[rows, z] = True
    return out


def replay(cfg):
    """Scaled demand, zone states, zone load and scaled forecasts for the test period."""
    start, end = PERIODS["test"]
    load = pd.read_parquet(DATA_PATH)["load_w"].loc[start:end]
    scale = cfg["demand_p99_w"] / float(load.quantile(0.99))
    out = pd.DataFrame({"uci_load_w": load, "demand_w": load * scale})
    zones = assign_zones(out["demand_w"], cfg["zones"])
    out = out.join(zones)
    out["zone_load_w"] = sum(zones[z] * v["watts"] for z, v in cfg["zones"].items())
    out.loc[out["demand_w"].isna(), "zone_load_w"] = np.nan

    path = ROOT / cfg["forecast_file"]
    if path.exists():
        fc = pd.read_parquet(path)
        for src, dst in FORECAST_COLS.items():
            out[dst] = fc[src].reindex(out.index) * scale
    else:
        print(f"Note: {cfg['forecast_file']} not found; forecasts skipped "
              "(run models/evaluate.py, Step 9, to create it).")
    return out, scale


def main():
    cfg = load_config()
    data, scale = replay(cfg)
    complete = data["zone_load_w"].notna().groupby(data.index.normalize()).all()
    print(f"Scale factor {scale:.5f} (UCI p99 {cfg['demand_p99_w'] / scale:.0f} W -> "
          f"{cfg['demand_p99_w']} W); {int(complete.sum())} of {len(complete)} test days "
          "have every minute")
    print(f"Mean zone load {data['zone_load_w'].mean():.1f} W "
          f"({data['zone_load_w'].mean() * 24:.0f} Wh/day); share of minutes with "
          f"everything off {(data['zone_load_w'] == 0).mean():.1%}")
    on_share = data[list(cfg["zones"])].mean().round(3).to_dict()
    print(f"Share of minutes each zone is on: {on_share}")

    # Example day: the complete day whose energy is closest to the median complete day.
    energy = data["zone_load_w"].groupby(data.index.normalize()).sum()[complete] / 60
    day = (energy - energy.median()).abs().idxmin()
    d = data.loc[day: day + pd.Timedelta("1D") - pd.Timedelta("1min")]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(d.index, d["demand_w"], color="#898781", lw=1.0, label="Scaled UCI demand")
    ax.step(d.index, d["zone_load_w"], where="post", color="#2a78d6", lw=1.6,
            label="Replayed zone load")
    if "fc_lgbm_w" in d:
        ax.plot(d.index + pd.Timedelta(minutes=cfg["horizon_min"]), d["fc_lgbm_w"],
                color="#eb6834", lw=1.0, alpha=0.9, label="Scaled LightGBM 10-min peak forecast")
    ax.set_ylabel("Power (W)")
    ax.set_xlim(d.index[0], d.index[-1])
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 3)))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e1e0d9")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title(f"Demand replay on the testbed zones, median-energy test day "
                 f"{day:%a %d %b %Y} ({energy[day]:.0f} Wh)", loc="left", fontsize=11,
                 fontweight="bold")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "step12_replay_example_day.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
