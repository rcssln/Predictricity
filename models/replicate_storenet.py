"""Step 16: replicate the UCI forecasting results on StoreNet, with no new tuning.

Follows docs/step16_replication.md. For each pre-registered house, H in (5, 10, 15) and
target in (y_next, y_peak):
    - features, targets and the chronological split with the same functions as UCI
      (add_features, add_targets, make_splits with the StoreNet periods)
    - the three baselines from models/baselines.py
    - LightGBM and XGBoost with the best Step 8 setting for that H, target and algorithm
      (lowest val_mae_w in results/tables/step08_tuning_log.csv), built with make_model() and
      trained with fit() from models/train_demand.py; early stopping on the house's
      validation rows only
    - the test rows scored once: MAE, RMSE and skill = 1 - MAE / MAE(persistence) for both
      models and all three baselines, on the same test minutes (rows where every baseline
      has a prediction)

Outputs:
    results/tables/step16_storenet_per_house.csv   house, H, target, predictor, MAE, RMSE,
                                                   skill, n_rows (written after each house)
    results/tables/step16_replication.md           UCI skill next to StoreNet median and
                                                   min-max skill, houses beating persistence
    results/tables/step16_replication_tests.md     R1: Wilcoxon signed-rank across houses on
                                                   house-level test MAE, model vs persistence
    results/figures/step16_skill_by_horizon.png    skill vs horizon, UCI and StoreNet

Run from anywhere:
    python models/replicate_storenet.py --max-trees 20   # smoke test; numbers mean nothing
    python models/replicate_storenet.py                  # full run, 216 fits
    python models/replicate_storenet.py --resume         # skip houses already done
    python models/replicate_storenet.py --tables-only    # rebuild tables, figure and verdict
                                                         # from the saved results
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import gc  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import train_demand  # noqa: E402
from models.baselines import baseline_predictions, score  # noqa: E402
from models.daily_tests import rank_biserial  # noqa: E402
from pipeline.clean_storenet import HOUSES, OUT_DIR  # noqa: E402
from pipeline.features import add_features  # noqa: E402
from pipeline.split import make_splits  # noqa: E402
from pipeline.targets import add_targets  # noqa: E402

STORENET_PERIODS = {
    "train": ("2020-01-01", "2020-08-31 23:59"),
    "val": ("2020-09-01", "2020-10-31 23:59"),
    "test": ("2020-11-01", "2020-12-31 23:59"),
}
HORIZONS = [5, 10, 15]
TARGETS = ["y_next", "y_peak"]
MODELS = {"lightgbm": "lgbm", "xgboost": "xgb"}
BASELINES = ["persistence", "same_time_yesterday", "moving_avg_15"]
TABLES = ROOT / "results" / "tables"
PER_HOUSE = TABLES / "step16_storenet_per_house.csv"
REPLICATION = TABLES / "step16_replication.md"
TESTS = TABLES / "step16_replication_tests.md"
FIGURE = ROOT / "results" / "figures" / "step16_skill_by_horizon.png"
UCI_TABLE = TABLES / "step09_table_4_1.csv"
LABELS = {"lgbm": "LightGBM", "xgb": "XGBoost", "persistence": "Persistence",
          "same_time_yesterday": "Same time yesterday", "moving_avg_15": "Moving average 15"}


def step8_settings():
    """{(H, target, algo): params} from the Step 8 tuning log (lowest validation MAE)."""
    log = pd.read_csv(TABLES / "step08_tuning_log.csv")
    best = log.loc[log.groupby(["H", "target", "algo"])["val_mae_w"].idxmin()]
    return {(int(r.H), r.target, r.algo): {"objective": r.objective, r.param: int(r.value)}
            for r in best.itertuples()}


def run_house(house, settings):
    load = pd.read_parquet(OUT_DIR / f"{house}_1min.parquet")
    features = add_features(load).astype("float32")
    cols = [c for c in features.columns if c != "load_w"]
    rows = []
    for H in HORIZONS:
        data = add_targets(features, H)
        train, val, test = make_splits(data, H, periods=STORENET_PERIODS)
        base = pd.DataFrame(baseline_predictions(data, H)).loc[test.index].dropna()
        test = test.loc[base.index]
        X_train, X_val = train[cols].to_numpy("float32"), val[cols].to_numpy("float32")
        X_test = test[cols].to_numpy("float32")
        for target in TARGETS:
            col = f"{target}_{H}"
            y = test[col].to_numpy("float64")
            preds = {name: base[name].to_numpy("float64") for name in BASELINES}
            trees = {}
            for algo, short in MODELS.items():
                model = train_demand.make_model(algo, settings[(H, target, algo)])
                trees[short] = train_demand.fit(algo, model, X_train, train[col].to_numpy("float32"),
                                                X_val, val[col].to_numpy("float32"))
                preds[short] = model.predict(X_test).astype("float64")
                del model
            mae_p = score(y, preds["persistence"])[0]
            for name, p in preds.items():
                mae, rmse = score(y, p)
                rows.append({"house": house, "H": H, "target": target, "predictor": name,
                             "mae_w": round(mae, 3), "rmse_w": round(rmse, 3),
                             "skill": round(1 - mae / mae_p, 5), "n_rows": len(y),
                             "n_train": len(train), "best_n_trees": trees.get(name, "")})
        print(f"  {house} H={H}: {len(train):,} train, {len(val):,} val, {len(test):,} test rows",
              flush=True)
        del data, train, val, test, X_train, X_val, X_test
        gc.collect()
    return rows


ABLATION = TABLES / "step16_ablation_per_house.csv"
SINGLE_GROUPS = ["Without calendar", "Without yesterday", "Without rolling", "Without long lags",
                 "Without diff"]
# One sentence per criterion that is not met, written after the results were seen. It explains
# the result; it does not change the criteria in docs/step16_replication.md.
REASONS = {}


def verdict(per_house):
    """R1-R3 verdict lines, worded as in docs/step16_replication.md; None if the ablation
    results (Prompt 5) are not complete yet."""
    if not ABLATION.exists():
        return None
    ab = pd.read_csv(ABLATION)
    n = per_house["house"].nunique()
    if ab.groupby("house").size().reindex(per_house["house"].unique()).fillna(0).lt(16).any():
        return None
    sk = per_house.pivot_table(index="house", columns=["H", "target", "predictor"], values="skill")

    # R1: median skill > 0 for both models, every H and target, and the H = 10 Wilcoxon tests.
    medians = {(H, t, m): sk[(H, t, m)].median() for H in HORIZONS for t in TARGETS
               for m in MODELS.values()}
    worst = min(medians, key=medians.get)
    pvals = {}
    for t in TARGETS:
        g = per_house[(per_house.H == 10) & (per_house.target == t)]
        pers = g[g.predictor == "persistence"].set_index("house")["mae_w"]
        for m in MODELS.values():
            pvals[(t, m)] = wilcoxon(g[g.predictor == m].set_index("house")["mae_w"].loc[pers.index],
                                     pers).pvalue
    r1 = all(v > 0 for v in medians.values()) and all(p < 0.05 for p in pvals.values())
    r1_houses = int((sk[[c for c in sk.columns if c[2] in MODELS.values()]] > 0).all(axis=1).sum())
    pmax = max(pvals.values())
    r1_text = (f"{'met' if r1 else 'not met'}: lowest median skill {medians[worst]:.1%} "
               f"({LABELS[worst[2]]}, {worst[1]}, H = {worst[0]}); H = 10 Wilcoxon p "
               f"{'< 0.001' if pmax < 0.001 else f'≤ {pmax:.3f}'} for all four tests")

    # R2: median LightGBM y_next skill higher at H = 15 than at H = 5.
    s5, s15 = sk[(5, "y_next", "lgbm")], sk[(15, "y_next", "lgbm")]
    r2 = s15.median() > s5.median()
    r2_text = (f"{'met' if r2 else 'not met'}: median skill {s5.median():.1%} at H = 5, "
               f"{s15.median():.1%} at H = 15")
    r2_houses = int((s15 > s5).sum())

    # R3: "recent only" raises median validation MAE more than removing any single group.
    r3_parts, r3_ok, r3_houses = [], [], []
    for t in TARGETS:
        g = ab[ab.target == t].pivot(index="house", columns="run", values="val_mae_w")
        d = g.sub(g["All 24 features"], axis=0)
        recent = d["Recent only"].median()
        single = d[SINGLE_GROUPS].median()
        r3_ok.append(recent > single.max())
        r3_houses.append(int((d["Recent only"] > d[SINGLE_GROUPS].max(axis=1)).sum()))
        r3_parts.append(f"{t}: recent only {recent:+.1f} W vs largest single group "
                        f"{single.max():+.1f} W ({single.idxmax().lower()})")
    r3 = all(r3_ok)
    r3_text = f"{'met' if r3 else 'not met'} (both targets): " + "; ".join(r3_parts)

    rows = [
        ("R1: models beat persistence", r1_text, f"{r1_houses} of {n} (both models beat "
         "persistence at every H and target)", "R1", r1),
        ("R2: skill grows with horizon", r2_text, f"{r2_houses} of {n}", "R2", r2),
        ("R3: recent readings carry the accuracy", r3_text,
         f"y_next {r3_houses[0]} of {n}; y_peak {r3_houses[1]} of {n}", "R3", r3),
    ]
    lines = ["", "## Replication verdict (criteria as pre-registered in docs/step16_replication.md)",
             "", "| Criterion | Result | Houses meeting it |", "|---|---|---|"]
    for name, text, houses, key, ok in rows:
        reason = "" if ok or key not in REASONS else f" {REASONS[key]}"
        lines.append(f"| {name} | {text}.{reason} | {houses} |")
    lines += ["", "R1 is checked on all 12 median skills (2 models × 3 horizons × 2 targets) and "
              "the four H = 10 Wilcoxon tests (2 models × 2 targets). The pre-registration does "
              "not name a target for R3, so it is checked on both and counts as met only if it "
              "holds for both."]
    return "\n".join(lines)


def summary_tables(per_house):
    uci = pd.read_csv(UCI_TABLE).set_index(["H", "target", "model"])["skill_vs_persistence"]
    n = per_house["house"].nunique()
    lines = ["# Step 16: replication on StoreNet", "",
             f"Skill vs persistence = 1 − MAE_model / MAE_persistence on each house's test set "
             f"(Nov–Dec 2020). StoreNet: median and min–max across {n} houses; "
             "UCI: the single UCI household, Table 4.1 (test set, Jul–Nov 2010). Same pipeline "
             "and Step 8 settings, no new tuning.", "",
             "| H | Target | Model | UCI skill | StoreNet median skill | StoreNet min–max "
             "| Houses beating persistence |", "|---|---|---|---|---|---|---|"]
    for H in HORIZONS:
        for target in TARGETS:
            for short in MODELS.values():
                s = per_house[(per_house.H == H) & (per_house.target == target)
                              & (per_house.predictor == short)]["skill"]
                lines.append(f"| {H} | {target} | {LABELS[short]} | {uci[(H, target, short)]:.1%} "
                             f"| {s.median():.1%} | {s.min():.1%} to {s.max():.1%} "
                             f"| {(s > 0).sum()} of {n} |")
    v = verdict(per_house)
    if v:
        lines.append(v)
    REPLICATION.write_text("\n".join(lines) + "\n", encoding="utf-8")

    t = ["# Step 16: R1 tests, model vs persistence across houses", "",
         f"Wilcoxon signed-rank test, two-sided, paired by house ({n} houses), on house-level "
         "test MAE. Median difference = median over houses of (MAE_model − MAE_persistence), "
         "in W; negative means the model has the lower error. Rank-biserial r: −1 = every house "
         "favours the model. The pre-registered R1 test is H = 10; H = 5 and 15 are extra.", "",
         "| H | Target | Model | W | p | Median difference (W) | Rank-biserial r | Role |",
         "|---|---|---|---|---|---|---|---|"]
    for H in HORIZONS:
        for target in TARGETS:
            g = per_house[(per_house.H == H) & (per_house.target == target)]
            pers = g[g.predictor == "persistence"].set_index("house")["mae_w"]
            for short in MODELS.values():
                m = g[g.predictor == short].set_index("house")["mae_w"].loc[pers.index]
                d = (m - pers).to_numpy()
                res = wilcoxon(m, pers)
                p = "< 0.001" if res.pvalue < 0.001 else f"{res.pvalue:.3f}"
                t.append(f"| {H} | {target} | {LABELS[short]} | {res.statistic:.1f} | {p} "
                         f"| {np.median(d):+.1f} | {rank_biserial(d):+.2f} "
                         f"| {'R1 (pre-registered)' if H == 10 else 'extra'} |")
    TESTS.write_text("\n".join(t) + "\n", encoding="utf-8")
    return "\n".join(lines), "\n".join(t)


def figure(per_house):
    uci = pd.read_csv(UCI_TABLE).set_index(["H", "target", "model"])["skill_vs_persistence"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, target in zip(axes, TARGETS):
        g = per_house[(per_house.target == target) & (per_house.predictor == "lgbm")]
        stats = g.groupby("H")["skill"].agg(["median", "min", "max"]).reindex(HORIZONS) * 100
        ax.fill_between(HORIZONS, stats["min"], stats["max"], color="#2a78d6", alpha=0.18,
                        linewidth=0, label=f"StoreNet min–max ({g.house.nunique()} houses)")
        ax.plot(HORIZONS, stats["median"], color="#2a78d6", marker="o", lw=2,
                label="StoreNet median")
        ax.plot(HORIZONS, [uci[(H, target, "lgbm")] * 100 for H in HORIZONS], color="#0b0b0b",
                marker="s", lw=2, ls="--", label="UCI (one household)")
        ax.axhline(0, color="#898781", lw=1)
        ax.set_title(f"{target}", loc="left", fontweight="bold")
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Forecast horizon H (min)")
        ax.grid(color="#e1e0d9")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("LightGBM skill vs persistence (%)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("LightGBM skill vs persistence by horizon: UCI and StoreNet (test sets)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    sys.stdout.reconfigure(errors="replace")
    args = sys.argv[1:]
    if "--max-trees" in args:
        train_demand.N_ESTIMATORS = int(args[args.index("--max-trees") + 1])
        print(f"SMOKE TEST: {train_demand.N_ESTIMATORS} trees max; the numbers mean nothing")
    if "--tables-only" in args:   # rebuild the tables, figure and verdict from saved results
        per_house = pd.read_csv(PER_HOUSE)
        rep, tests = summary_tables(per_house)
        figure(per_house)
        print(rep + "\n\n" + tests)
        return
    done = pd.read_csv(PER_HOUSE) if "--resume" in args and PER_HOUSE.exists() else pd.DataFrame()
    finished = set(done["house"]) if len(done) else set()
    if "--resume" not in args:
        PER_HOUSE.unlink(missing_ok=True)
    settings = step8_settings()
    rows = done.to_dict("records")
    for house in HOUSES:
        if house in finished:
            print(f"{house}: already done, skipped")
            continue
        t0 = time.perf_counter()
        print(f"{house}:", flush=True)
        rows += run_house(house, settings)
        pd.DataFrame(rows).to_csv(PER_HOUSE, index=False)
        print(f"{house}: done in {time.perf_counter() - t0:.0f} s", flush=True)
    per_house = pd.DataFrame(rows)
    rep, tests = summary_tables(per_house)
    figure(per_house)
    print("\n" + rep + "\n\n" + tests)
    for p in (PER_HOUSE, REPLICATION, TESTS, FIGURE):
        print(f"Saved {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
