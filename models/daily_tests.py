"""Step 14: day-level tests, effect sizes and bootstrap confidence intervals for Step 12.

Reads only the committed Step 12 per-day files (results/tables/step12_{scenario}_per_day.csv),
so it needs neither the UCI data nor the models. For every scenario, outcome (battery-empty
minutes and comfort cost per day) and pair of policies, paired by simulated day:

    mean_diff_a_minus_b   mean of (policy a - policy b) over days
    ci95_low, ci95_high   95% percentile bootstrap CI of that mean (days resampled with
                          replacement, 10,000 resamples, seed 42)
    median_diff           median of the daily differences
    a_lower_days, b_lower_days, tied_days
    rank_biserial         matched-pairs rank-biserial correlation, in [-1, 1]: -1 means
                          every non-tied day favours a (a lower), +1 every day favours b
    p_value               Wilcoxon signed-rank test, two-sided (zero differences dropped),
                          the same test as Step 12
    p_holm                Holm-adjusted within each scenario and outcome, for every
                          comparison except the pre-registered primary one (scenario B,
                          battery-empty minutes, Reactive vs Predictive demand only)

Outputs:
    results/tables/step14_daily_tests.csv   every comparison
    results/tables/step14_daily_tests.md    the key comparisons per scenario

Run from anywhere:
    python models/daily_tests.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = ROOT / "results" / "tables"
SCENARIOS = {"evening_sized": "B, evening-sized (primary)", "original": "A, original",
             "sunset_full": "C, sunset-full"}
POLICIES = {"no_control": "No control", "reactive": "Reactive",
            "pred_demand": "Predictive, demand only", "pred_both": "Predictive, demand + solar"}
OUTCOMES = {"battery_empty_min": "Battery-empty min/day", "comfort_cost_min": "Comfort cost"}
PRIMARY = ("evening_sized", "battery_empty_min", "reactive", "pred_demand")
KEY_PAIRS = [("reactive", "pred_demand"), ("pred_demand", "pred_both"),
             ("reactive", "pred_both"), ("no_control", "reactive"),
             ("no_control", "pred_demand")]
N_BOOT = 10_000
SEED = 42


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation of daily differences d = a - b (zeros dropped):
    (sum of ranks of positive d - sum of ranks of negative d) / total rank sum."""
    d = d[d != 0]
    if len(d) == 0:
        return np.nan
    ranks = rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def bootstrap_ci(d, rng):
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(p):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    running = 0.0
    for rank, i in enumerate(ok[np.argsort(p[ok])]):
        running = max(running, min(1.0, (len(ok) - rank) * p[i]))
        out[i] = running
    return out


def compare(scenario, per_day, rng):
    rows = []
    for outcome in OUTCOMES:
        wide = per_day.pivot(index="date", columns="policy", values=outcome)
        for a, b in itertools.combinations(POLICIES, 2):
            d = (wide[a] - wide[b]).to_numpy(dtype=float)
            try:
                p = float(wilcoxon(wide[a], wide[b]).pvalue)
            except ValueError:   # every daily difference is zero
                p = np.nan
            lo, hi = bootstrap_ci(d, rng)
            rows.append({
                "scenario": scenario, "outcome": outcome, "policy_a": a, "policy_b": b,
                "n_days": len(d), "mean_a": wide[a].mean(), "mean_b": wide[b].mean(),
                "mean_diff_a_minus_b": d.mean(), "ci95_low": lo, "ci95_high": hi,
                "median_diff": float(np.median(d)), "a_lower_days": int((d < 0).sum()),
                "b_lower_days": int((d > 0).sum()), "tied_days": int((d == 0).sum()),
                "rank_biserial": rank_biserial(d), "p_value": p,
                "primary": (scenario, outcome, a, b) == PRIMARY,
            })
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for outcome in OUTCOMES:
        sec = (out["outcome"] == outcome) & ~out["primary"]
        out.loc[sec, "p_holm"] = holm(out.loc[sec, "p_value"])
    return out


def p_text(p):
    return "–" if pd.isna(p) else "< 0.001" if p < 0.001 else f"{p:.3f}"


def markdown(table):
    lines = ["# Step 14: day-level tests, effect sizes and bootstrap CIs (Step 12 simulation)", "",
             "Paired by simulated day. Difference = first policy − second policy, so a negative "
             "value means the first policy has less. 95% CI: percentile bootstrap over days "
             f"({N_BOOT:,} resamples). Rank-biserial r: −1 = every non-tied day favours the "
             "first policy, +1 = every day favours the second. p: Wilcoxon signed-rank, "
             "two-sided; Holm-adjusted within scenario and outcome except the pre-registered "
             "primary comparison (scenario B, battery-empty minutes, Reactive vs Predictive "
             "demand only), which is marked *.", ""]
    for scenario, label in SCENARIOS.items():
        t = table[table["scenario"] == scenario]
        lines += [f"## Scenario {label} ({int(t['n_days'].iloc[0])} days)", "",
                  "| Outcome | Comparison | Means (first / second) | Difference (95% CI) "
                  "| First lower / second lower / tied days | Rank-biserial r | p |",
                  "|---|---|---|---|---|---|---|"]
        for outcome, olabel in OUTCOMES.items():
            for a, b in KEY_PAIRS:
                r = t[(t["outcome"] == outcome) & (t["policy_a"] == a) & (t["policy_b"] == b)].iloc[0]
                p = r["p_value"] if r["primary"] else r["p_holm"]
                lines.append(
                    f"| {olabel} | {POLICIES[a]} vs {POLICIES[b]}{' *' if r['primary'] else ''} "
                    f"| {r['mean_a']:.1f} / {r['mean_b']:.1f} "
                    f"| {r['mean_diff_a_minus_b']:+.1f} ({r['ci95_low']:+.1f} to {r['ci95_high']:+.1f}) "
                    f"| {r['a_lower_days']} / {r['b_lower_days']} / {r['tied_days']} "
                    f"| {r['rank_biserial']:+.2f} | {p_text(p)} |")
        lines.append("")
    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(errors="replace")
    rng = np.random.default_rng(SEED)
    tables = []
    for scenario in SCENARIOS:
        per_day = pd.read_csv(TABLE_DIR / f"step12_{scenario}_per_day.csv")
        tables.append(compare(scenario, per_day, rng))
    table = pd.concat(tables, ignore_index=True)
    rounded = table.round({c: 4 for c in ["mean_a", "mean_b", "mean_diff_a_minus_b",
                                          "ci95_low", "ci95_high", "median_diff",
                                          "rank_biserial"]})
    rounded.to_csv(TABLE_DIR / "step14_daily_tests.csv", index=False)
    md = markdown(table)
    (TABLE_DIR / "step14_daily_tests.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print("\nSaved results/tables/step14_daily_tests.csv")
    print("Saved results/tables/step14_daily_tests.md")


if __name__ == "__main__":
    main()
