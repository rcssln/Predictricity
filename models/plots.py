"""Step 10: the four forecasting figures for Chapter 4.

Uses only the Step 9 test predictions (data/processed/step09_test_preds.parquet) and the
saved H = 10 LightGBM y_peak model (for feature importance); no model is refit.

Saves to results/figures/:
    step10_pred_vs_actual.png      actual load vs LightGBM y_next forecast, H = 10,
                                   one weekday and one weekend day
    step10_feature_importance.png  top 15 LightGBM features by gain, H = 10, y_peak
    step10_error_by_hour.png       MAE by hour of day, LightGBM vs persistence, H = 10
    step10_error_by_horizon.png    MAE at H = 5, 10, 15 for every model and baseline

The example days are not hand-picked: among test days with every minute present, each
is the day whose LightGBM MAE is closest to the median for its day type.

Run from anywhere:
    python models/plots.py
"""

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PREDS_PATH = ROOT / "data" / "processed" / "step09_test_preds.parquet"
ARTIFACT_DIR = ROOT / "models" / "artifacts"
FIG_DIR = ROOT / "results" / "figures"

HORIZONS = [5, 10, 15]
# Fixed categorical order (validated palette); text stays in ink colours.
PREDICTORS = {
    "lgbm": ("LightGBM", "#2a78d6", "o"),
    "xgb": ("XGBoost", "#eb6834", "s"),
    "persistence": ("Persistence", "#1baf7a", "^"),
    "moving_avg_15": ("Moving average 15", "#eda100", "D"),
    "same_time_yesterday": ("Same time yesterday", "#e87ba4", "v"),
}
ACTUAL = "#52514e"
INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "font.size": 10, "legend.frameon": False, "lines.linewidth": 2,
    "figure.titlesize": 13, "figure.titleweight": "bold",
})


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def suptitle(fig, title, subtitle, bottom=0.0):
    """Finding as the title, what is shown as the subtitle; lays out the axes below them."""
    h = fig.get_figheight()
    # Plain figure text: tight_layout would otherwise reserve room for a suptitle twice.
    fig.text(0.01, 1 - 0.08 / h, title, ha="left", va="top", fontsize=13, fontweight="bold")
    fig.text(0.01, 1 - 0.4 / h, subtitle, ha="left", va="top", color=INK_2, fontsize=10)
    fig.tight_layout(rect=(0, bottom, 1, 1 - 0.62 / h))


def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.relative_to(ROOT)}")


def pick_days(preds):
    """Median-error complete weekday and weekend day for the H = 10 y_next forecast."""
    p = preds[["y_next_10", "yhat_next_10_lgbm"]].dropna()
    err = (p["yhat_next_10_lgbm"] - p["y_next_10"]).abs()
    by_day = err.groupby(err.index.normalize()).agg(["mean", "size"])
    by_day = by_day[by_day["size"] == 1440]
    days = {}
    for label, is_weekend in [("Weekday", False), ("Weekend day", True)]:
        group = by_day[(by_day.index.dayofweek >= 5) == is_weekend]["mean"]
        days[label] = (group - group.median()).abs().idxmin()
    return days, by_day


def pred_vs_actual(preds):
    days, by_day = pick_days(preds)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharey=True)
    for ax, (label, day) in zip(axes, days.items()):
        window = slice(day, day + pd.Timedelta("1D") - pd.Timedelta("1min"))
        actual = preds.loc[window, "load_w"]
        # The forecast made at t is for t + 10 min, so draw it at its target time.
        fc = preds["yhat_next_10_lgbm"].shift(10, freq="min").loc[window]
        ax.plot(actual.index, actual / 1000, color=ACTUAL, lw=1.0, label="Actual load")
        ax.plot(fc.index, fc / 1000, color=PREDICTORS["lgbm"][1], lw=1.6,
                label="LightGBM forecast (made 10 min earlier)")
        ax.set_title(f"{label}: {day:%a %d %b %Y}  (MAE {by_day.loc[day, 'mean']:.0f} W)")
        ax.set_ylabel("Load (kW)")
        ax.set_xlim(actual.index[0], actual.index[-1])
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 3)))
    axes[0].legend(loc="upper left", ncol=2)
    suptitle(fig, "The forecast tracks the daily load shape but catches sudden spikes only "
             "after they start",
             "Household load and the H = 10 min LightGBM y_next forecast on two typical test days")
    save(fig, "step10_pred_vs_actual.png")


def feature_importance():
    art = joblib.load(ARTIFACT_DIR / "lightgbm_y_peak_10.joblib")
    gain = pd.Series(art["model"].booster_.feature_importance(importance_type="gain"),
                     index=art["features"])
    share = (gain / gain.sum() * 100).sort_values(ascending=False)
    top = share.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color=PREDICTORS["lgbm"][1], height=0.7)
    for y, v in enumerate(top.values):
        ax.text(v + share.max() * 0.01, y, f"{v:.1f}%", va="center", fontsize=9, color=INK_2)
    ax.set_xlabel("Share of total gain (%)")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlim(0, share.max() * 1.12)
    suptitle(fig, f"Recent load drives the peak forecast: top feature '{share.index[0]}' "
             f"has {share.iloc[0]:.0f}% of the gain",
             f"Top 15 of {len(share)} LightGBM features by gain, y_peak, H = 10 min "
             f"(the 15 shown hold {share.head(15).sum():.0f}%)")
    save(fig, "step10_feature_importance.png")


def error_by_hour(preds):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    gains = {}
    for ax, target in zip(axes, ["next", "peak"]):
        y = f"y_{target}_10"
        p = preds[[y, f"yhat_{target}_10_lgbm", f"yhat_{target}_10_persistence"]].dropna()
        hour = p.index.hour
        for key in ["persistence", "lgbm"]:
            name, color, marker = PREDICTORS[key]
            err = (p[f"yhat_{target}_10_{key}"] - p[y]).abs().groupby(hour).mean()
            ax.plot(err.index, err.values, color=color, marker=marker, ms=5, label=name)
        ax.set_title(f"y_{target}")
        ax.set_xlabel("Hour of day (forecast time)")
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlim(-0.5, 23.5)
        lg = (p[f"yhat_{target}_10_lgbm"] - p[y]).abs().groupby(hour).mean()
        ps = (p[f"yhat_{target}_10_persistence"] - p[y]).abs().groupby(hour).mean()
        gains[target] = ((ps - lg) > 0).sum()
    axes[0].set_ylabel("MAE (W)")
    axes[0].legend(loc="upper left")
    wins = " and ".join(f"{n} of 24 (y_{t})" for t, n in gains.items())
    suptitle(fig, "Errors are highest in the evening; LightGBM beats persistence in every hour"
             if min(gains.values()) == 24 else
             f"Errors are highest in the evening; LightGBM beats persistence in {wins} hours",
             "Test-set MAE by hour of day, H = 10 min")
    save(fig, "step10_error_by_hour.png")


def error_by_horizon(preds):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    yesterday = {}
    for ax, target in zip(axes, ["next", "peak"]):
        for key, (name, color, marker) in PREDICTORS.items():
            vals = []
            for H in HORIZONS:
                p = preds[[f"y_{target}_{H}", f"yhat_{target}_{H}_{key}"]].dropna()
                vals.append(mae(p.iloc[:, 1], p.iloc[:, 0]))
            if key == "same_time_yesterday":  # 640-780 W: off the scale, given in the subtitle
                yesterday[target] = vals
                continue
            # XGBoost sits almost on top of LightGBM, so it is dashed with hollow markers.
            style = dict(ls="--", mfc=SURFACE, zorder=3) if key == "xgb" else dict(zorder=2)
            ax.plot(HORIZONS, vals, color=color, marker=marker, ms=7, label=name, **style)
        ax.set_title(f"y_{target}")
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Forecast horizon H (min)")
    axes[0].set_ylim(0, max(line.get_ydata().max() for ax in axes for line in ax.lines) * 1.08)
    axes[0].set_ylabel("MAE (W)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0))
    rng = "; y_peak ".join(f"{min(v):.0f} W at every H" if round(min(v)) == round(max(v))
                           else f"{min(v):.0f}–{max(v):.0f} W" for v in yesterday.values())
    suptitle(fig, "The models' lead over persistence grows with the horizon",
             "Test-set MAE at H = 5, 10 and 15 min. Not shown: same time yesterday "
             f"(y_next {rng})", bottom=0.08)
    save(fig, "step10_error_by_horizon.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    preds = pd.read_parquet(PREDS_PATH)
    pred_vs_actual(preds)
    feature_importance()
    error_by_hour(preds)
    error_by_horizon(preds)


if __name__ == "__main__":
    sys.exit(main())
