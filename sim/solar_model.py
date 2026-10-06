"""Step 12.2: modeled solar output, minute by minute (stand-in until the real panel log).

For each day: a clear-sky curve, a half sine from sunrise to sunset peaking at
solar_peak_w, times a cloud factor in 0.2-1.0. The cloud factor is a seeded random walk
that drifts smoothly around a level drawn for the day (0.6-1.0 on most days, 0.2-0.4 on
the overcast_day_share of days that are fully overcast). All settings: sim/config.yaml.

Run from anywhere (plots one clear and one overcast example day):
    python sim/solar_model.py
"""

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

from pipeline.split import PERIODS  # noqa: E402
from sim.config import FIG_DIR, load_config  # noqa: E402


def _minutes(hhmm):
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def clear_sky(index, cfg):
    """Clear-sky output (W) at each timestamp: half sine between sunrise and sunset."""
    rise, sset = _minutes(cfg["sunrise"]), _minutes(cfg["sunset"])
    minute = index.hour * 60 + index.minute
    phase = (minute - rise) / (sset - rise)
    out = cfg["solar_peak_w"] * np.sin(np.pi * phase)
    return pd.Series(np.where((phase >= 0) & (phase <= 1), out, 0.0), index=index)


def cloud_factor(index, cfg):
    """Smooth seeded cloud factor in [0.2, 1.0], one random walk per day."""
    rng = np.random.default_rng(cfg["seed"])
    out = np.empty(len(index))
    days = index.normalize()
    for day in days.unique():
        pos = np.flatnonzero(days == day)
        overcast = rng.random() < cfg["overcast_day_share"]
        lo, hi = (0.2, 0.4) if overcast else (0.2, 1.0)
        level = rng.uniform(0.2, 0.35) if overcast else rng.uniform(0.6, 1.0)
        steps = rng.normal(0, cfg["cloud_step_sd"], len(pos))
        x = level
        for i, step in enumerate(steps):
            x = min(max(x + 0.05 * (level - x) + step, lo), hi)
            out[pos[i]] = x
    return pd.Series(out, index=index)


def solar(index, cfg):
    """DataFrame with clear_sky_w, cloud and pv_w (modeled panel output) per minute."""
    cs = clear_sky(index, cfg)
    cloud = cloud_factor(index, cfg)
    return pd.DataFrame({"clear_sky_w": cs, "cloud": cloud, "pv_w": cs * cloud})


def test_period_index():
    start, end = PERIODS["test"]
    return pd.date_range(start, end, freq="min", name="timestamp")


def main():
    cfg = load_config()
    pv = solar(test_period_index(), cfg)
    daily_wh = pv["pv_w"].resample("D").sum() / 60
    print(f"Modeled solar over {len(daily_wh)} test days: mean {daily_wh.mean():.0f} Wh/day "
          f"(min {daily_wh.min():.0f}, max {daily_wh.max():.0f}); "
          f"clear-sky {pv['clear_sky_w'].resample('D').sum().mean() / 60:.0f} Wh/day")

    days = {"Sunny day": daily_wh.idxmax(), "Overcast day": daily_wh.idxmin()}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (label, day) in zip(axes, days.items()):
        d = pv.loc[day: day + pd.Timedelta("1D") - pd.Timedelta("1min")]
        ax.plot(d.index, d["clear_sky_w"], color="#898781", lw=1.2, ls="--", label="Clear sky")
        ax.plot(d.index, d["pv_w"], color="#eda100", lw=1.8, label="Modeled output")
        ax.set_title(f"{label}: {day:%d %b %Y} ({daily_wh[day]:.0f} Wh)", loc="left",
                     fontsize=10, fontweight="bold")
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 6)))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e1e0d9")
    axes[0].set_ylabel("Solar output (W)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(f"Modeled solar: a {cfg['solar_peak_w']} W half-sine scaled by a smooth "
                 "random cloud factor", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "step12_solar_example_days.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
