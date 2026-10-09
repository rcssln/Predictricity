"""Reproduce every table and figure in results/ with one command (Step 13).

Runs Steps 2, 3, 6, 7, 8, 9, 10, 11, 12, 14 and 15 in order, each as its own script. Needs
data/raw/household_power_consumption.txt (Step 1) and the packages in requirements.txt.

    python run_initial_results.py                   # everything, retraining included
                                                    # (about 2.5 hours on an 8 GB laptop)
    python run_initial_results.py --skip-training   # reuse the 12 saved models in
                                                    # models/artifacts/ (about an hour, most
                                                    # of it the Step 15 feature ablation)

The test set is only re-scored with the frozen models and settings, so a run reproduces
the committed numbers; it does not tune anything.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "household_power_consumption.txt"
ARTIFACTS = [ROOT / "models" / "artifacts" / f"{algo}_{target}_{H}.joblib"
             for algo in ("lightgbm", "xgboost") for target in ("y_next", "y_peak")
             for H in (5, 10, 15)]

STEPS = [
    ("2", "Clean the UCI data", ["pipeline/clean_uci.py"]),
    ("3", "Data summary table", ["pipeline/check_uci.py"]),
    ("6", "Chronological split check", ["pipeline/split.py", "10"]),
    ("7", "Baselines on validation", ["models/baselines.py"]),
    ("8", "Train and tune LightGBM and XGBoost", ["models/train_demand.py"]),
    ("9", "Score the test set (Table 4.1)", ["models/evaluate.py"]),
    ("10", "Forecasting figures", ["models/plots.py"]),
    ("11", "Early warning (Table 4.2)", ["models/early_warning.py"]),
    ("12", "Modeled solar", ["sim/solar_model.py"]),
    ("12", "Demand replay", ["sim/replay.py"]),
    ("12", "Policy simulation, scenario A original (first run's files too)",
     ["sim/run_policies.py", "--scenario", "original"]),
    ("12", "Policy simulation, scenario B evening_sized (primary Table 4.3)",
     ["sim/run_policies.py", "--scenario", "evening_sized"]),
    ("12", "Policy simulation, scenario C sunset_full",
     ["sim/run_policies.py", "--scenario", "sunset_full"]),
    ("12", "Switching sensitivity on scenario B", ["sim/run_policies.py", "--switching"]),
    ("14", "Day-level tests, effect sizes and bootstrap CIs", ["models/daily_tests.py"]),
    ("15", "Feature ablation on validation (about 45-60 min)", ["models/feature_ablation.py"]),
]


def main():
    skip_training = "--skip-training" in sys.argv
    if not RAW.exists():
        sys.exit(f"Missing {RAW.relative_to(ROOT)}: download the UCI dataset first (Step 1).")
    if skip_training and not all(p.exists() for p in ARTIFACTS):
        missing = [p.name for p in ARTIFACTS if not p.exists()]
        sys.exit(f"--skip-training needs all 12 saved models; missing: {', '.join(missing)}")

    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8"}
    start = time.perf_counter()
    for step, name, cmd in STEPS:
        if step == "8" and skip_training:
            print(f"\n=== Step {step}: {name} - skipped (--skip-training, saved models used)")
            continue
        print(f"\n=== Step {step}: {name}  (python {' '.join(cmd)})", flush=True)
        t0 = time.perf_counter()
        result = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env)
        if result.returncode != 0:
            sys.exit(f"Step {step} failed (exit code {result.returncode}); stopping.")
        print(f"=== Step {step} done in {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"\nAll steps finished in {(time.perf_counter() - start) / 60:.1f} min. "
          "Tables are in results/tables/, figures in results/figures/.")


if __name__ == "__main__":
    main()
