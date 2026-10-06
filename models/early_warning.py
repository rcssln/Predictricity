"""Step 11: early warning of demand spikes, predictive vs reactive.

Definitions (Initial Results Runbook):
    T                  95th percentile of TRAIN load (top 5% of demand), from train only
    spike event        load rises above T after at least 10 minutes at or below it;
                       the event starts at that first minute above T
    predictive alert   at minute t, the H = 10 LightGBM y_peak forecast (max load over
                       t+1 .. t+10) exceeds T - margin
    reactive alert     at minute t, the current load exceeds T
    early-warning rate share of events with an alert 1-10 minutes before the start
    recall             the same: events "preceded" by an alert within 10 minutes. An alert
                       at the start minute itself is not counted: the forecast made then
                       already sees the high load and exceeds T for every event, so
                       counting it makes recall 100% at any margin
    detection recall   share of events with an alert 0-10 minutes before the start
                       (reported so the reactive rule, which never warns, is not all zeros)
    lead time          minutes from the first alert 1-10 minutes before an event to its
                       start; median over warned events (0 when none are warned)
    precision          share of alert episodes (runs of consecutive alert minutes)
                       followed by an event start within 0-10 minutes of the episode start
    F1                 2 * precision * recall / (precision + recall)

Events are found on the full 1-minute load series, never on the split tables with
dropped rows: a minute next to a gap is never an event start, because the 10 minutes
before it must all be present and at or below T. Both rules are scored on the same
minutes (those with a forecast) and the same events (those with a forecast for every
minute from 10 minutes before the start up to the start).

The margin is chosen on VALIDATION (best F1, with recall = early warning, over 0, 2.5, ..., 50% of T); the TEST set
is then scored once, at that margin.

Outputs:
    results/tables/step11_margin_sweep_val.csv
    results/tables/step11_early_warning.md
    results/figures/step11_precision_recall.png

Run from anywhere (--check-val stops after the validation sweep, before the test set):
    python models/early_warning.py --check-val
    python models/early_warning.py
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.features import add_features  # noqa: E402
from pipeline.split import DATA_PATH, PERIODS, make_splits  # noqa: E402
from pipeline.targets import add_targets  # noqa: E402

H = 10
QUIET = 10        # minutes at or below T before an event start
WINDOW = 10       # minutes before an event in which an alert counts
QUANTILE = 0.95
MARGIN_STEPS = np.arange(0, 50.01, 2.5)   # % of T
MODEL_PATH = ROOT / "models" / "artifacts" / f"lightgbm_y_peak_{H}.joblib"
SWEEP_PATH = ROOT / "results" / "tables" / "step11_margin_sweep_val.csv"
TABLE_PATH = ROOT / "results" / "tables" / "step11_early_warning.md"
FIG_PATH = ROOT / "results" / "figures" / "step11_precision_recall.png"


def event_starts(load, T, quiet=QUIET):
    """True at each minute where load first exceeds T after `quiet` minutes at or below it.

    `load` must be on a strict 1-minute grid with NaN for missing minutes; a missing
    minute counts as neither above nor below T, so it can never complete a quiet run.
    """
    below = (load <= T).astype("float64")
    quiet_before = below.shift(1).rolling(quiet, min_periods=quiet).sum() == quiet
    return (load > T) & quiet_before


def episode_starts(alert):
    """True at the first minute of each run of consecutive alert minutes."""
    return alert & ~alert.shift(1, fill_value=False)


def _any_within(flags, lo, hi):
    """At each minute t: is `flags` True at any minute t-hi .. t-lo (offsets are >= 0)?"""
    out = pd.Series(False, index=flags.index)
    for k in range(lo, hi + 1):
        out |= flags.shift(k, fill_value=False)
    return out


def measures(alert, events, scored):
    """The four Step 11 measures for one alert rule on one period.

    alert, events: boolean series on the full minute grid.
    scored: boolean series, True at minutes that belong to the period and have a forecast.
    """
    alert = alert & scored
    covered = scored.astype("float64").rolling(WINDOW + 1, min_periods=WINDOW + 1).min() == 1
    events = events & covered
    n_events = int(events.sum())

    early = _any_within(alert, 1, WINDOW)[events]
    detected = _any_within(alert, 0, WINDOW)[events]

    # Lead time: the largest k in 1..WINDOW with an alert k minutes before the start.
    lead = pd.Series(np.nan, index=events.index[events])
    for k in range(WINDOW, 0, -1):
        hit = alert.shift(k, fill_value=False)[events] & lead.isna()
        lead[hit] = k

    starts = episode_starts(alert)
    followed = _any_within(events[::-1], 0, WINDOW)[::-1]   # an event start in t .. t+10
    n_episodes = int(starts.sum())

    precision = float(followed[starts].mean()) if n_episodes else np.nan
    recall = float(early.mean()) if n_events else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "n_events": n_events,
        "n_alert_episodes": n_episodes,
        "early_warning_rate": recall,
        "median_lead_min": float(lead.median()) if lead.notna().any() else 0.0,
        "precision": precision,
        "recall": recall,
        "detection_recall": float(detected.mean()) if n_events else np.nan,
        "f1": f1,
    }


def forecasts(features, load_index, model):
    """H = 10 y_peak forecasts on the val and test rows, on the full minute grid."""
    _, val, test = make_splits(add_targets(features, H), H)
    out = {}
    for name, part in [("val", val), ("test", test)]:
        yhat = model["model"].predict(part[model["features"]].to_numpy("float32"))
        out[name] = pd.Series(yhat, index=part.index).reindex(load_index)
    return out


def in_period(index, name):
    start, end = PERIODS[name]
    return pd.Series((index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end)), index=index)


def plot_sweep(sweep, chosen, reactive):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e1e0d9", "axes.edgecolor": "#c3c2b7",
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "savefig.facecolor": "#fcfcfb", "xtick.color": "#898781",
                         "ytick.color": "#898781", "axes.labelcolor": "#52514e"})
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(sweep["recall"], sweep["precision"], color="#2a78d6", marker="o", ms=5, lw=2,
            label="Predictive rule, margin 0–50% of T")
    for _, r in sweep.iloc[::4].iterrows():
        ax.annotate(f"{r['margin_pct']:.0f}%", (r["recall"], r["precision"]), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color="#52514e")
    ax.plot(chosen["recall"], chosen["precision"], marker="o", ms=12, mfc="none",
            mec="#0b0b0b", mew=1.5, ls="none",
            label=f"Chosen: margin {chosen['margin_pct']:.1f}% (best F1 {chosen['f1']:.2f})")
    ax.plot(reactive["recall"], reactive["precision"], marker="^", ms=9, color="#1baf7a",
            ls="none", label="Reactive rule (load > T now)")
    ax.set_xlabel("Recall = early-warning rate (share of spikes alerted 1–10 min ahead)")
    ax.set_ylabel("Precision (share of alerts followed by a spike)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.text(0.01, 0.985, "A wider margin catches more spikes but raises more false alarms",
             ha="left", va="top", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.935, "Precision vs recall of the predictive alert across margins, validation "
             "(Jan–Jun 2010)", ha="left", va="top", fontsize=9.5, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    sys.stdout.reconfigure(errors="replace")   # the Windows console cannot print "−"
    raw = pd.read_parquet(DATA_PATH)
    load = raw["load_w"]
    full_grid = pd.date_range(load.index[0], load.index[-1], freq="min")
    assert load.index.equals(full_grid), "load must be on a strict 1-minute grid"

    train_start, train_end = PERIODS["train"]
    T = float(load.loc[train_start:train_end].quantile(QUANTILE))
    events = event_starts(load, T)
    print(f"T = {T:.0f} W (95th percentile of train load); "
          f"{int(events.sum()):,} spike events on the full series")

    model = joblib.load(MODEL_PATH)
    features = add_features(raw).astype("float32")   # exactly as in training
    yhat = forecasts(features, load.index, model)
    del features, raw

    def score(split, margin_w=None):
        scored = yhat[split].notna() & in_period(load.index, split)
        alert = (load > T) if margin_w is None else (yhat[split] > T - margin_w)
        return measures(alert.fillna(False), events, scored)

    rows = []
    for pct in MARGIN_STEPS:
        rows.append({"margin_pct": pct, "margin_w": T * pct / 100,
                     **score("val", T * pct / 100)})
    sweep = pd.DataFrame(rows)
    SWEEP_PATH.parent.mkdir(parents=True, exist_ok=True)
    sweep.round(4).to_csv(SWEEP_PATH, index=False)
    chosen = sweep.loc[sweep["f1"].idxmax()]
    reactive_val = score("val")
    print("\nValidation sweep:")
    print(sweep.round(3).to_string(index=False))
    print(f"\nChosen margin on validation: {chosen['margin_pct']:.1f}% of T "
          f"= {chosen['margin_w']:.0f} W (F1 {chosen['f1']:.3f})")
    plot_sweep(sweep, chosen, reactive_val)
    if "--check-val" in sys.argv:
        print(f"Validation only: saved {SWEEP_PATH.relative_to(ROOT)} and "
              f"{FIG_PATH.relative_to(ROOT)}; the test set was not scored.")
        return

    # The test set, once, at the chosen margin.
    test = {"Reactive (load > T now)": score("test"),
            f"Predictive (forecast peak > T − {chosen['margin_w']:.0f} W)":
                score("test", chosen["margin_w"])}
    lines = [
        "**Table 4.2. Early warning of demand spikes, test set (Jul–Nov 2010)**",
        "",
        "| Rule | Early-warning rate | Median lead time (min) | Precision | Recall "
        "| F1 | Detection recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for rule, m in test.items():
        lines.append(f"| {rule} | {m['early_warning_rate']:.1%} | {m['median_lead_min']:.0f} "
                     f"| {m['precision']:.1%} | {m['recall']:.1%} | {m['f1']:.2f} "
                     f"| {m['detection_recall']:.1%} |")
    n = next(iter(test.values()))
    lines += [
        "",
        f"T = {T:.0f} W, the 95th percentile of training load. {n['n_events']:,} spike events "
        f"(load above T after at least {QUIET} minutes at or below it) in the test period, "
        f"each with a forecast for the {WINDOW} minutes before it. The margin "
        f"({chosen['margin_pct']:.1f}% of T) was chosen by best F1 on validation, before "
        "the test set was scored. Early-warning rate = recall = share of events with an "
        "alert 1–10 min before the start; lead time is the median over warned events; "
        "precision = share of alert episodes followed by an event start within 0–10 min; "
        "detection recall also counts an alert at the start minute. "
        f"Alert episodes on test: reactive {test['Reactive (load > T now)']['n_alert_episodes']:,}, "
        f"predictive {list(test.values())[1]['n_alert_episodes']:,}.",
    ]
    TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    for p in [SWEEP_PATH, TABLE_PATH, FIG_PATH]:
        print(f"Saved {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
