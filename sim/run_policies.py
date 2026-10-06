"""Step 12.3: simulate the four load-shedding policies over every complete test day.

Every day starts at battery_start_pct and is simulated minute by minute with the same
rules for every policy; only the information each policy acts on differs:

    No control                   nothing is shed
    Reactive                     current demand  vs current solar  + battery allowance
    Predictive, demand only      forecast peak   vs current solar  + battery allowance
    Predictive, demand + solar   forecast peak   vs forecast solar + battery allowance

"Forecast peak" is the H = 10 LightGBM y_peak forecast scaled like the demand (Step 9
hand-off file). "Forecast solar" is clear-sky persistence: clear-sky output 10 minutes
ahead times the current ratio of actual to clear-sky output (ratio 1 before sunrise).

Shared rules (all settings in sim/config.yaml):
    budget      = solar + allowance(state of charge)
    shed        sheddable zones are shed in priority order (highest number first:
                zoneC, zoneA, zoneB) until need - shed watts <= budget; critical is never
                shed. When the battery is empty, every sheddable zone is shed.
    restore     one zone at a time, last shed first, once need - shed watts + restore_margin_w
                <= budget has held for restore_after_min minutes in a row
    min switch  no zone changes state within min_switch_min of the last change

Measures per day and policy:
    battery-empty minutes   minutes where the battery and solar cannot serve the load that
                            is switched on (that load goes unserved)
    deficit events          from the No-control run: demand above solar + allowance after
                            at least 10 minutes without such a deficit (the same events
                            are used for every policy)
    early-warning rate      share of deficit events with a shedding episode STARTING 1-10 min
                            before (shedding that was already on for longer is not a warning)
    median lead time        minutes from the earliest such start to the event, over warned
                            events (0 when none are warned)
    shed precision          share of shedding episodes followed by a deficit event start
                            within 0-10 min of the episode start
    shed recall             share of deficit events that were warned or met with shedding
                            already on at the start minute
    comfort cost            zone-minutes a sheddable zone was wanted but shed
    critical interruptions  minutes the controller switched the critical zone off (must be 0)
    switches                zone state changes made by the controller

Outputs:
    results/tables/step12_per_day.csv
    results/tables/step12_table_4_3_preliminary.md
    results/tables/step12_wilcoxon.csv
    results/figures/step12_soc_sample_day.png
    results/tables/step12_sensitivity.md   the same policies with the `sensitivity` settings
                                           (a larger battery and array) from config.yaml

Run from anywhere:
    python sim/run_policies.py
"""

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.battery import allowance_w, can_serve, step_soc  # noqa: E402
from sim.config import FIG_DIR, TABLE_DIR, load_config  # noqa: E402
from sim.replay import replay  # noqa: E402
from sim.solar_model import solar  # noqa: E402

POLICIES = {
    "no_control": "No control",
    "reactive": "Reactive",
    "pred_demand": "Predictive, demand only",
    "pred_both": "Predictive, demand + solar",
}
COLORS = {"no_control": "#e87ba4", "reactive": "#1baf7a", "pred_demand": "#2a78d6",
          "pred_both": "#eb6834"}
QUIET = 10
WINDOW = 10


def solar_forecast(pv, cs, horizon):
    """Clear-sky persistence: clear-sky output `horizon` min ahead x current pv/clear-sky."""
    ratio = np.where(cs >= 1.0, pv / np.maximum(cs, 1e-9), 1.0)
    return cs.shift(-horizon, fill_value=0.0) * ratio


def simulate(day, policy, cfg):
    """Minute-by-minute run of one policy on one day. Returns a dict of per-minute arrays."""
    zones = cfg["zones"]
    order = sorted([z for z, v in zones.items() if v["sheddable"]],
                   key=lambda z: -zones[z]["priority"])
    cum = np.cumsum([0] + [zones[z]["watts"] for z in order])
    cap = cfg["battery_usable_wh"]
    eff = dict(charge_eff=cfg["charge_eff"], inverter_eff=cfg["inverter_eff"])
    n = len(day)
    want = {z: day[z].to_numpy() for z in zones}
    pv = day["pv_w"].to_numpy()
    demand = day["zone_load_w"].to_numpy()
    need_info = day["fc_w"].to_numpy() if policy.startswith("pred") else demand
    supply_info = day["pv_fc_w"].to_numpy() if policy == "pred_both" else pv

    soc = cap * cfg["battery_start_pct"] / 100
    k, ok_run, last_switch = 0, 0, -10**9
    out = {key: np.zeros(n) for key in ["soc_pct", "served_w", "shed_k", "empty",
                                        "comfort", "critical_off", "switches", "deficit"]}
    for t in range(n):
        budget = supply_info[t] + allowance_w(100 * soc / cap, cfg["battery_allowance_w"])
        # Deficit on the information of the reactive rule (used from the No-control run).
        out["deficit"][t] = demand[t] > pv[t] + allowance_w(100 * soc / cap,
                                                            cfg["battery_allowance_w"])
        if policy != "no_control":
            need = need_info[t]
            k_need = next((i for i, c in enumerate(cum) if need - c <= budget), len(order))
            k_ok = next((i for i, c in enumerate(cum)
                         if need - c + cfg["restore_margin_w"] <= budget), len(order))
            if soc <= 0:
                k_need = len(order)
            can_switch = t - last_switch >= cfg["min_switch_min"]
            new_k = k
            if k_need > k:
                ok_run = 0
                if can_switch:
                    new_k = k_need
            elif k_ok < k:
                ok_run += 1
                if ok_run >= cfg["restore_after_min"] and can_switch:
                    new_k, ok_run = k - 1, 0
            else:
                ok_run = 0
            if new_k != k:
                out["switches"][t] = abs(new_k - k)
                k, last_switch = new_k, t
        shed = set(order[:k])
        served = sum(zones[z]["watts"] for z in zones if want[z][t] and z not in shed)
        out["comfort"][t] = sum(1 for z in shed if want[z][t])
        out["critical_off"][t] = any(not zones[z]["sheddable"] for z in shed)
        out["empty"][t] = served > 0 and not can_serve(soc, pv[t], served, **eff)
        out["served_w"][t] = 0.0 if out["empty"][t] else served
        out["shed_k"][t] = k
        soc = step_soc(soc, pv[t], served, cap, **eff)
        out["soc_pct"][t] = 100 * soc / cap
    return out


def starts_after_quiet(flag, quiet=QUIET):
    """True where `flag` turns on after at least `quiet` minutes off (within one day)."""
    f = np.asarray(flag, dtype=bool)
    out = np.zeros_like(f)
    for t in range(quiet, len(f)):
        out[t] = f[t] and not f[t - quiet:t].any()
    return out


def warning_measures(active, events):
    """Event-based measures of one policy's shedding against the day's deficit events."""
    active = np.asarray(active, dtype=bool)
    ev = np.flatnonzero(events)
    onset = active & ~np.r_[False, active[:-1]]
    leads, n_detected = [], 0
    for s in ev:
        before = [k for k in range(1, WINDOW + 1) if s - k >= 0 and onset[s - k]]
        if before:
            leads.append(max(before))
        if before or active[s]:
            n_detected += 1
    eps = np.flatnonzero(onset)
    n_true = sum(1 for a in eps if events[a:a + WINDOW + 1].any())
    return {"n_events": len(ev), "n_warned": len(leads), "n_detected": n_detected,
            "n_episodes": len(eps), "n_true_episodes": n_true, "leads": leads}


def load_days(cfg):
    data, scale = replay(cfg)
    sol = solar(data.index, cfg)
    data = data.join(sol)
    data["fc_w"] = data["fc_lgbm_w"]
    data["pv_fc_w"] = solar_forecast(data["pv_w"], data["clear_sky_w"], cfg["horizon_min"])
    need = ["zone_load_w", "fc_w", "pv_w"]
    complete = data[need].notna().all(axis=1).groupby(data.index.normalize()).all()
    days = [d for d in complete.index[complete] if (data.index.normalize() == d).sum() == 1440]
    return data, days, scale


def run(cfg):
    """Simulate every policy on every complete test day."""
    data, days, scale = load_days(cfg)
    rows, leads, traces = [], {p: [] for p in POLICIES}, {}
    for day in days:
        d = data.loc[day: day + pd.Timedelta("1D") - pd.Timedelta("1min")]
        runs = {p: simulate(d, p, cfg) for p in POLICIES}
        events = starts_after_quiet(runs["no_control"]["deficit"])
        traces[day] = {p: r["soc_pct"] for p, r in runs.items()}
        for p, r in runs.items():
            w = warning_measures(r["shed_k"] > 0, events)
            leads[p] += w.pop("leads")
            rows.append({
                "date": day.date(), "policy": p,
                "battery_empty_min": int(r["empty"].sum()),
                "deficit_events": w["n_events"],
                "early_warning_rate": w["n_warned"] / w["n_events"] if w["n_events"] else np.nan,
                "shed_precision": (w["n_true_episodes"] / w["n_episodes"]
                                   if w["n_episodes"] else np.nan),
                "shed_recall": w["n_detected"] / w["n_events"] if w["n_events"] else np.nan,
                "comfort_cost_min": int(r["comfort"].sum()),
                "critical_interruptions": int(r["critical_off"].sum()),
                "switches": int(r["switches"].sum()),
                "end_soc_pct": round(float(r["soc_pct"][-1]), 1),
                "solar_wh": round(float(d["pv_w"].sum() / 60), 1),
                "demand_wh": round(float(d["zone_load_w"].sum() / 60), 1),
                **w,
            })
    return pd.DataFrame(rows), leads, traces, scale


def summarize(per_day, leads):
    """Table 4.3 values: mean per day for minutes and counts; rates pooled over all events."""
    summary = []
    for p in POLICIES:
        g = per_day[per_day["policy"] == p]
        ev, eps = g["n_events"].sum(), g["n_episodes"].sum()
        summary.append({
            "policy": p,
            "battery_empty_min": g["battery_empty_min"].mean(),
            "early_warning_rate": g["n_warned"].sum() / ev if ev else np.nan,
            "median_lead_min": float(np.median(leads[p])) if leads[p] else 0.0,
            "shed_precision": g["n_true_episodes"].sum() / eps if eps else np.nan,
            "shed_recall": g["n_detected"].sum() / ev if ev else np.nan,
            "comfort_cost_min": g["comfort_cost_min"].mean(),
            "critical_interruptions": int(g["critical_interruptions"].sum()),
            "switches": g["switches"].mean(),
        })
    return summary


def table_md(summary, title, note):
    pct = (lambda x: "–" if np.isnan(x) else f"{x:.1%}")
    lines = [
        title, "",
        "| Policy | Battery-empty min/day | Early-warning rate | Median lead time (min) "
        "| Shed precision | Shed recall | Comfort cost (zone-min/day) "
        "| Critical interruptions | Switches/day |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        lines.append(
            f"| {POLICIES[s['policy']]} | {s['battery_empty_min']:.1f} "
            f"| {pct(s['early_warning_rate'])} | {s['median_lead_min']:.0f} "
            f"| {pct(s['shed_precision'])} | {pct(s['shed_recall'])} "
            f"| {s['comfort_cost_min']:.1f} | {s['critical_interruptions']} "
            f"| {s['switches']:.1f} |")
    return "\n".join(lines + ["", note]) + "\n"


def setting_note(cfg, per_day):
    days = per_day["date"].nunique()
    n_events = int(per_day.loc[per_day["policy"] == "no_control", "n_events"].sum())
    zero = int((per_day.loc[per_day["policy"] == "no_control", "battery_empty_min"] == 0).sum())
    return (
        f"{days} complete test days, battery {cfg['battery_usable_wh']} Wh usable starting "
        f"at {cfg['battery_start_pct']}% each midnight, modeled solar {cfg['solar_peak_w']} W "
        "peak (half-sine with random clouds) until the real panel log exists. Without control, "
        f"{zero} of {days} days have no battery-empty minutes. Battery-empty minutes, comfort "
        "cost and switches are means per day; the warning measures are pooled over all "
        f"{n_events:,} deficit events (from the No-control run, the same for every policy) and "
        "all shedding episodes. Early warning = a shedding episode starts 1–10 min before a "
        "deficit; shed recall also counts shedding already on at the start minute.")


def main():
    sys.stdout.reconfigure(errors="replace")   # the Windows console cannot print "–"
    cfg = load_config()
    per_day, leads, traces, scale = run(cfg)
    days = sorted(traces)
    print(f"{len(days)} complete test days; demand scale {scale:.5f}")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    per_day.to_csv(TABLE_DIR / "step12_per_day.csv", index=False)
    summary = summarize(per_day, leads)
    md = table_md(
        summary,
        "**Table 4.3 (preliminary). Load-shedding policies in simulation, testbed settings**",
        setting_note(cfg, per_day))
    (TABLE_DIR / "step12_table_4_3_preliminary.md").write_text(md, encoding="utf-8")

    # Sensitivity: the same run with a larger battery and array.
    sens_cfg = {**cfg, **cfg["sensitivity"]}
    sens_day, sens_leads, _, _ = run(sens_cfg)
    sens_md = table_md(
        summarize(sens_day, sens_leads),
        "**Table 4.3b (sensitivity). The same policies with a larger battery and array "
        f"({sens_cfg['battery_usable_wh']} Wh, {sens_cfg['solar_peak_w']} W)**",
        setting_note(sens_cfg, sens_day))
    (TABLE_DIR / "step12_sensitivity.md").write_text(sens_md, encoding="utf-8")

    # Wilcoxon signed-rank tests, paired by day.
    tests = []
    wide = {m: per_day.pivot(index="date", columns="policy", values=m)
            for m in ["battery_empty_min", "comfort_cost_min"]}
    for m, table in wide.items():
        for a, b in itertools.combinations(POLICIES, 2):
            diff = table[a] - table[b]
            try:
                stat, p = wilcoxon(table[a], table[b])
            except ValueError:   # every paired difference is zero
                stat, p = np.nan, np.nan
            tests.append({"measure": m, "policy_a": a, "policy_b": b,
                          "mean_a": table[a].mean(), "mean_b": table[b].mean(),
                          "median_diff": diff.median(), "n_days_differ": int((diff != 0).sum()),
                          "statistic": stat, "p_value": p})
    tests = pd.DataFrame(tests)
    tests.round(6).to_csv(TABLE_DIR / "step12_wilcoxon.csv", index=False)

    # Battery charge on a sample day: the No-control day closest to the median of days that
    # have any battery-empty minutes (not hand-picked).
    nc = per_day[per_day["policy"] == "no_control"].set_index("date")["battery_empty_min"]
    pool = nc[nc > 0] if (nc > 0).any() else nc
    sample = (pool - pool.median()).abs().idxmin()
    day = pd.Timestamp(sample)
    idx = pd.date_range(day, periods=1440, freq="min")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for p, label in POLICIES.items():
        empty = per_day[(per_day["policy"] == p) & (per_day["date"] == sample)][
            "battery_empty_min"].iloc[0]
        ax.plot(idx, traces[day][p], color=COLORS[p], lw=1.8,
                ls="--" if p == "pred_both" else "-", label=f"{label} ({empty} min empty)")
    ax.set_ylabel("Battery charge (%)")
    ax.set_ylim(-2, 102)
    ax.set_xlim(idx[0], idx[-1])
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 3)))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e1e0d9")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title(f"Battery charge under the four policies, {day:%a %d %b %Y} "
                 "(median No-control day with battery-empty minutes)", loc="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "step12_soc_sample_day.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n" + md + "\n" + sens_md)
    print(tests[["measure", "policy_a", "policy_b", "mean_a", "mean_b", "n_days_differ",
                 "p_value"]].round(4).to_string(index=False))
    print("\nChecks (12.4):")
    s = {r["policy"]: r for r in summary}
    print(f"  critical interruptions all 0: {all(r['critical_interruptions'] == 0 for r in summary)}")
    print(f"  No control has the most battery-empty minutes: "
          f"{max(s, key=lambda p: s[p]['battery_empty_min']) == 'no_control'}")
    print(f"  reactive lead 0, predictive lead > 0: {s['reactive']['median_lead_min'] == 0} / "
          f"{s['pred_demand']['median_lead_min'] > 0 and s['pred_both']['median_lead_min'] > 0}")
    same = per_day[per_day.policy == "pred_demand"].reset_index(drop=True)[
        ["battery_empty_min", "comfort_cost_min", "switches"]].equals(
        per_day[per_day.policy == "pred_both"].reset_index(drop=True)[
            ["battery_empty_min", "comfort_cost_min", "switches"]])
    print(f"  two predictive policies differ: {not same}")
    for f in ["step12_per_day.csv", "step12_table_4_3_preliminary.md", "step12_wilcoxon.csv",
              "step12_sensitivity.md"]:
        print(f"Saved results/tables/{f}")
    print("Saved results/figures/step12_soc_sample_day.png")


if __name__ == "__main__":
    main()
