# Predictricity

## Step 0. Set up Python

Requires Python 3.12+ (developed on 3.14). From the project root:

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (Git Bash)
source .venv/Scripts/activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

All commands below assume the venv is active and you are in the project root.
(Every script finds the project root by itself, so they also work from other folders.)

## Step 1. Download the UCI dataset

1. Go to <https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption>
   and click **Download** (≈ 20 MB zip).
2. Unzip it and put `household_power_consumption.txt` (≈ 127 MB) here:

```
data/raw/household_power_consumption.txt
```

The raw file is `;`-separated, has 2,075,259 rows, and marks missing readings with `?`.

## Step 2. Clean the data → `uci_1min.parquet`

```bash
python pipeline/clean_uci.py
```

- Combines `Date` + `Time` into a timestamp, keeps `Global_active_power`, converts kW → W as `load_w`.
- Forces a strict 1-minute grid.
- Interpolates **only gaps of ≤ 5 minutes**; longer gaps stay empty (NaN).

Output: `data/processed/uci_1min.parquet`

## Step 3. Data summary table

```bash
python pipeline/check_uci.py
```

Output: `results/tables/step03_data_summary.md`

Expected values: 2,075,259 rows · 2006-12-16 17:24 → 2010-11-26 21:02 ·
25,979 missing before cleaning, 25,903 after · longest gap 7,226 min ·
mean 1,091.6 W, median 602.0 W, p95 3,264.0 W, p99 4,850.0 W.

## Step 4. Exploratory figures

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/step04_explore.ipynb
```

(or open `notebooks/step04_explore.ipynb` in Jupyter / VS Code and *Run All*).

Outputs in `results/figures/`:

| File | Shows |
|---|---|
| `step04_daily_profile.png` | Average load by time of day, weekdays vs weekends |
| `step04_one_week.png` | One ordinary week (29 Sep – 5 Oct 2008) of raw 1-minute load |
| `step04_distribution.png` | Histogram of `load_w`, log-scaled counts |
| `step04_monthly_mean.png` | Mean load per month, Dec 2006 – Nov 2010 |

## Step 5. Prove there is no data leakage

```bash
python -m pytest tests
```

Expected: **100 passed**. This proves:

- `pipeline/features.py` — features at time *t* never change when any value **after** *t* changes.
- `pipeline/targets.py` — targets at time *t* never change when any value **at or before** *t* changes.
- Each check also catches a deliberately leaky version, so the tests are not passing by accident.
- `pipeline/split.py` — no validation/test timestamp is earlier than the last training timestamp,
  and no target window crosses into the next period.

## Step 6. Chronological train / val / test split

```bash
python pipeline/split.py        # H = 10 (default)
python pipeline/split.py 15     # any other horizon
```

| Period | Dates | Rows (H = 10) |
|---|---|---|
| train | Dec 2006 – Dec 2009 | 1,585,707 |
| val | Jan – Jun 2010 | 252,574 |
| test | Jul – Nov 2010 | 199,010 |

The last H minutes of train and val are dropped so no target looks into the next period,
then rows with any missing feature or target are dropped. **The test set is held out —
do not use it until the final model is chosen.**

## Step 7. Baselines on the validation set

```bash
python models/baselines.py
```

Output: `results/tables/step07_baselines_val.csv` — MAE and RMSE (W) for
persistence, same-time-yesterday and 15-minute moving average, for H = 5, 10, 15
and both targets. **Persistence is the baseline to beat** (e.g. H = 10, `y_next`:
MAE 331.3 W, RMSE 655.2 W).

## Step 8. Train and tune LightGBM and XGBoost

```bash
python models/train_demand.py
```

Takes about 2.5 hours on an 8 GB laptop (72 fits on ~1.6M training rows). For each H in
5, 10, 15 and each target (`y_next`, `y_peak`):

- Builds float32 features and targets, splits with `make_splits()`. **The test set is not used.**
- Trains LightGBM (`num_leaves` 31, 63, 127) and XGBoost (`max_depth` 6, 8, 10), each with a
  squared-error and an absolute-error objective (`l2`/`l1`, `reg:squarederror`/`reg:absoluteerror`),
  up to 2,000 trees at learning rate 0.03, row and column subsampling 0.8, with early stopping
  on validation MAE (patience 100).
- Keeps the setting with the lowest validation MAE for each algorithm.

If a run is interrupted (for example by running out of memory), continue it with
`python models/train_demand.py --resume`. Finished H / target / algorithm groups are kept, and
a group cut off partway is redone in full.

Outputs:

- `results/tables/step08_tuning_log.csv`: every setting with its tree count, validation
  MAE/RMSE (W) and fit time.
- `models/artifacts/{algo}_{target}_{H}.joblib`: the best model per algorithm, target and H,
  saved as a dict with `model`, `features`, `H`, `target`, `params`, `best_n_trees`,
  `val_mae_w`, `val_rmse_w` and the train/val start and end dates and row counts
  (12 files, ~365 MB in total).

Best validation MAE (W) vs persistence:

| H | Target | LightGBM | XGBoost | Persistence |
|---|---|---|---|---|
| 5 | `y_next` | 202.9 | **201.5** | 226.9 |
| 5 | `y_peak` | **168.2** | 170.3 | 175.4 |
| 10 | `y_next` | **274.7** | 275.1 | 331.3 |
| 10 | `y_peak` | **264.7** | 267.3 | 286.9 |
| 15 | `y_next` | **314.7** | 315.9 | 396.1 |
| 15 | `y_peak` | **340.5** | 344.4 | 385.1 |

The models now beat persistence on both targets at every H. The absolute-error objective won
in all 12 groups: squared error pulls forecasts of the spiky load upward and costs MAE (an
earlier squared-error-only grid lost to persistence on `y_peak`). The gain over persistence
grows with H (`y_next`: 11% at H = 5, 21% at H = 15) and is smallest for `y_peak` at H = 5
(4%). The trade-off is RMSE: absolute-error models aim at the median and miss large spikes by
more, e.g. H = 15 `y_peak` LightGBM RMSE 676.8 W vs 638.2 W for squared-error
XGBoost at depth 8. These validation scores are optimistic because the same data drove early stopping
and model selection; the test set gives the unbiased estimate.

## Step 9. Score the test set once

```bash
python models/evaluate.py --check-val   # optional: reruns validation, saves nothing
python models/evaluate.py
```

Scores the five predictors (three baselines, the best LightGBM and the best XGBoost) on the
test set (Jul–Nov 2010) for each H and target. All five are scored on the same test minutes
(~199,000 per H). `--check-val` runs the same code on validation and reproduces the Step 7
and Step 8 numbers exactly; it was run first, so the test set was scored only once.
**After this step the models are frozen: no more tuning.**

Outputs:

- `results/tables/step09_table_4_1.csv` and `.md`: MAE, RMSE (W) and skill vs persistence
  (1 − MAE_model / MAE_persistence) for every H, target and predictor (Table 4.1).
- `data/processed/step09_test_preds.parquet`: every test prediction, for Steps 10 and 11.
- `results/handoff/step09_test_preds.parquet`: every test minute (220,320) with `load_w`,
  `y_peak_10`, `yhat_peak_10_lgbm`, `yhat_peak_10_xgb`, for the Step 12 simulation. Minutes
  with missing data are NaN (8.4% of `load_w`, 9.7% of the forecasts).

Test MAE (W), best baseline (persistence) vs the models:

| H | Target | Persistence | LightGBM | XGBoost | Best skill |
|---|---|---|---|---|---|
| 5 | `y_next` | 221.2 | 192.7 | **191.3** | 13.5% |
| 5 | `y_peak` | 163.8 | **156.3** | 158.7 | 4.6% |
| 10 | `y_next` | 316.4 | **257.1** | 258.1 | 18.8% |
| 10 | `y_peak` | 270.4 | **247.4** | 250.2 | 8.5% |
| 15 | `y_next` | 369.7 | **292.4** | 294.7 | 20.9% |
| 15 | `y_peak` | 364.6 | **319.4** | 323.7 | 12.4% |

Both models beat every baseline at every H on the test set, by a little more than on
validation. The gain grows with H, and is smallest for `y_peak` at H = 5 (4.6%). LightGBM
and XGBoost are within 1% of each other; XGBoost has the lower RMSE in every case.

---

## Step 10. Forecasting figures

```bash
python models/plots.py
```

Draws Chapter 4's four forecasting figures from the Step 9 test predictions and the saved
H = 10 LightGBM `y_peak` model (for feature importance). Nothing is refit. Saved to
`results/figures/`, with 2–3 sentence captions in `results/figures/step10_captions.md`:

| Figure | Shows |
|---|---|
| `step10_pred_vs_actual.png` | Actual load vs the H = 10 LightGBM `y_next` forecast on a median-error weekday and weekend day |
| `step10_feature_importance.png` | Top 15 LightGBM features by gain, H = 10, `y_peak` (`now` + `lag_1` hold 83%) |
| `step10_error_by_hour.png` | MAE by hour of day, LightGBM vs persistence, H = 10 (LightGBM lower in every hour) |
| `step10_error_by_horizon.png` | MAE at H = 5, 10, 15 for both models and the baselines |

---

## Step 11. Early warning of demand spikes

```bash
python -m pytest tests/test_early_warning.py   # hand-made series with known event starts
python models/early_warning.py --check-val     # optional: validation sweep only
python models/early_warning.py
```

A spike event is load rising above T = 3,378 W (the 95th percentile of training load)
after at least 10 minutes at or below it. Events are found on the full 1-minute series, so
a gap never creates a fake event start. Two alert rules are compared on the same minutes
and events:

- **Reactive:** current load > T. It can never warn before a spike starts.
- **Predictive:** the H = 10 LightGBM `y_peak` forecast > T − margin.

The margin is chosen on validation by best F1 over 0–50% of T in 2.5% steps, then the test
set is scored once. Recall counts only alerts 1–10 minutes *before* a spike, so it equals
the early-warning rate. An alert at the start minute is not counted: the forecast made then
already sees the high load and exceeds T for every event, which would make recall 100% at
any margin and push F1 to a margin of 0. "Detection recall" (alerts 0–10 min before) is
also reported.

Outputs: `results/tables/step11_margin_sweep_val.csv` (every margin, validation),
`results/tables/step11_early_warning.md` (Table 4.2, test) and
`results/figures/step11_precision_recall.png` (precision vs recall across margins,
validation).

Test set (Jul–Nov 2010, 425 events), margin 12.5% of T = 422 W:

| Rule | Early-warning rate | Median lead time (min) | Precision | F1 |
|---|---|---|---|---|
| Reactive (load > T now) | 0.0% | 0 | 51.8% | 0.00 |
| Predictive (forecast peak > T − 422 W) | **57.9%** | **3** | 41.1% | **0.48** |

The predictive rule warns 1–10 minutes ahead of 58% of spikes, with a median lead of
3 minutes, at the cost of more false alarms (precision 41% vs 52%). Both rules detect
every spike by its start minute. Test results are close to validation (57% warned,
precision 44%).

---

## Step 12. Load-shedding policies in simulation (preliminary)

```bash
pip install -r requirements.txt                       # adds scipy and pyyaml
python sim/solar_model.py                             # 12.2: modeled solar, example-day plot
python sim/replay.py                                  # 12.2: demand replay on the zones, example-day plot
python sim/run_policies.py                            # 12.3: primary scenario B (evening_sized)
python sim/run_policies.py --scenario original        # scenario A (also the first run's files)
python sim/run_policies.py --scenario sunset_full     # scenario C
python sim/run_policies.py --switching                # min switch time 1, 5, 10 on scenario B
python -m pytest tests/test_sim.py                    # simulator tests on a hand-made 2-day input
```

All settings are in `sim/config.yaml` (report every value in Chapter 3). Demand is the UCI
test-period load scaled so its 99th percentile is 110 W, replayed on the testbed's four
switched zones (critical 25 W, never shed; zones A 40 W, B 20 W, C 25 W). Each minute the
replay picks the zone combination closest to the scaled demand, with critical first.
Solar is a **modeled** half-sine with a smooth random cloud factor, so the results are
preliminary until the real panel log exists.

The four policies use the same shedding rules (shed zone C, then A, then B until the need
fits the budget; restore one zone after 3 minutes with 10 W to spare; shed everything when
the battery is empty). Only the information differs:

| Policy | Need | Supply |
|---|---|---|
| No control | – | – |
| Reactive | current demand | current solar + battery allowance |
| Predictive, demand only | H = 10 LightGBM peak forecast | current solar + battery allowance |
| Predictive, demand + solar | H = 10 LightGBM peak forecast | clear-sky persistence solar forecast + allowance |

### Scenarios

The scenarios are the `scenarios:` section of `sim/config.yaml`, chosen with `--scenario`
(default `evening_sized`). They were pre-registered in `docs/step12_scenarios.md` before they
were run, together with the primary comparison and a feasibility check on the No-control run.

| Scenario | Battery | Day runs | Start charge | Solar peak | Role |
|---|---|---|---|---|---|
| A. `original` | 43 Wh | 00:00 → 00:00 | 60% | 80 W | Sensitivity (the first run) |
| B. `evening_sized` | 220 Wh | 00:00 → 00:00 | 60% | 104 W | **Primary: Table 4.3** |
| C. `sunset_full` | 43 Wh | 17:00 → 17:00 | 100% | 80 W | Sensitivity |

A simulated day is the 1,440 minutes from `day_start`, labelled by its start date, and is used
only if every minute is present; per-day measures, deficit events and the Wilcoxon pairing use
it. `min_switch_min` can be set per scenario (1 in all three).

**Outputs**, per scenario: `results/tables/step12_{scenario}_table_4_3.md`,
`step12_{scenario}_per_day.csv`, `step12_{scenario}_wilcoxon.csv` (with a `primary` flag and
Holm-adjusted p-values for the secondary comparisons) and
`results/figures/step12_{scenario}_soc_sample_day.png`. `--switching` writes
`step12_switching_sensitivity.md`. `--scenario original` also rewrites the first run's files,
unchanged: `step12_table_4_3_preliminary.md`, `step12_per_day.csv`, `step12_wilcoxon.csv`,
`step12_sensitivity.md` (86 Wh, 120 W) and `step12_soc_sample_day.png`. The example-day
plots are `step12_solar_example_days.png` and `step12_replay_example_day.png`.

### Results

Table 4.3, scenario B (136 complete test days):

| Policy | Battery-empty min/day | Early warning | Lead (min) | Shed precision | Comfort cost (zone-min/day) | Switches/day | Critical interruptions |
|---|---|---|---|---|---|---|---|
| No control | 205.8 | 0.0% | 0 | – | 0.0 | 0.0 | 0 |
| Reactive | 110.1 | 0.3% | 2 | 70.6% | 249.6 | 86.9 | 0 |
| Predictive, demand only | **103.1** | **15.6%** | 2 | 61.3% | 260.1 | 75.5 | 0 |
| Predictive, demand + solar | 103.3 | 15.4% | 2 | 61.1% | 259.5 | 75.8 | 0 |

- **Primary comparison:** predictive (demand only) has fewer battery-empty minutes than
  reactive, 103.1 vs 110.1 min/day (Wilcoxon signed-rank, paired by day, p < 0.001; the
  policies differ on 70 of 136 days).
- **The cost is comfort:** predictive sheds more (260.1 vs 249.6 zone-min/day) with lower
  precision (61.3% vs 70.6%), but it warns ahead of more deficits (15.6% vs 0.3%) and
  switches less (75.5 vs 86.9 per day).
- Every shedding policy roughly halves battery-empty time compared with no control.
- **The solar forecast adds nothing measurable** on battery-empty minutes (p = 0.75).
- Reactive's 0.3% early warning comes from its own battery reaching empty: the cutoff rule
  then sheds everything a few minutes before the No-control run's deficit.

**Scenario A (the first run)** used the testbed's 43 Wh battery. Without control the battery
was empty on all 136 days (488.8 min/day), and the shedding policies were almost identical
(409.4 reactive, 407.3 and 407.1 predictive). That is why scenario B was added.
**Scenario C** behaves the same way: empty every day, with 426.5 min/day for reactive vs 424.2
for predictive. **Switching sensitivity** (scenario B): with a minimum switch time of 5 and 10
minutes, predictive stays ahead of reactive (105.1 vs 113.3 and 106.7 vs 117.3 min/day) and
switches far less (43.8 and 30.7 per day). The primary setting stays 1 minute.

Early warning counts a shedding episode that *starts* 1–10 minutes before a deficit event;
shedding that was already on for hours is not a warning. Deficit events come from the
No-control run (demand above solar + allowance after 10 minutes without a deficit) and are
the same for every policy. Checks (12.4), in every scenario: no critical interruptions; No
control has the most battery-empty minutes; predictive policies warn ahead (lead 2 min) and
reactive essentially does not; the two predictive policies differ.

---

## Step 13. Reproduce every table and figure with one command

```bash
pip install -r requirements.txt
# place data/raw/household_power_consumption.txt first (Step 1)
python run_initial_results.py --skip-training   # reuse the 12 saved models: about 6 min
python run_initial_results.py                   # retrain too: about 2.5 h on an 8 GB laptop
```

`run_initial_results.py` runs Steps 2, 3, 6, 7, 8, 9, 10, 11 and 12 (all three scenarios
and the switching sensitivity) in order and stops at the first failure. It rebuilds everything in `results/tables/` and `results/figures/`, plus
`results/handoff/` and `data/processed/`. The test set is only re-scored with the frozen
models and settings, so nothing is tuned. Checked on 2026-10-06, and again on 2026-10-07 with the scenarios, from an empty
`data/processed/` with `--skip-training`: every committed table and figure came out
byte-identical, and so did the regenerated data files.

Not included: Step 4's notebook (exploration only) and the Step 5 tests; run them with
`python -m nbconvert --to notebook --execute --inplace notebooks/step04_explore.ipynb` and
`python -m pytest tests`. The one-page summary for the adviser is
`docs/initial_results_summary.md`.

### Step by step (the same, by hand)

```bash
pip install -r requirements.txt
# place data/raw/household_power_consumption.txt first (Step 1)
python pipeline/clean_uci.py
python pipeline/check_uci.py
python -m nbconvert --to notebook --execute --inplace notebooks/step04_explore.ipynb
python -m pytest tests
python pipeline/split.py 10
python models/baselines.py
python models/train_demand.py
python models/evaluate.py
python models/plots.py
python models/early_warning.py
python sim/solar_model.py
python sim/replay.py
python sim/run_policies.py --scenario original
python sim/run_policies.py --scenario evening_sized
python sim/run_policies.py --scenario sunset_full
python sim/run_policies.py --switching
```

## Files that make up the study

| File | Role |
|---|---|
| `run_initial_results.py` | Step 13: rebuild every table and figure with one command |
| `docs/initial_results_summary.md` | Step 13: one-page summary for the adviser |
| `pipeline/clean_uci.py` | Step 2: raw UCI → clean 1-minute parquet |
| `pipeline/check_uci.py` | Step 3: data summary table |
| `notebooks/step04_explore.ipynb` | Step 4: exploratory figures |
| `pipeline/features.py` | `add_features()`: lags, rolling stats, time-of-day, day-of-week (past-only) |
| `pipeline/targets.py` | `add_targets()`: `y_next_H`, `y_peak_H` (future-only) |
| `pipeline/split.py` | Step 6: `make_splits()` chronological split |
| `models/baselines.py` | Step 7: baseline scores on validation |
| `models/train_demand.py` | Step 8: tune LightGBM / XGBoost on validation, save best models |
| `models/evaluate.py` | Step 9: score everything on the test set once (Table 4.1) |
| `models/plots.py` | Step 10: the four forecasting figures |
| `models/early_warning.py` | Step 11: predictive vs reactive spike alerts (Table 4.2) |
| `sim/config.yaml`, `sim/config.py` | Step 12: every simulation setting, and the loader |
| `sim/solar_model.py`, `sim/replay.py` | Step 12.2: modeled solar; demand replay on the zones |
| `sim/battery.py`, `sim/run_policies.py` | Step 12.3: battery model; four-policy simulation (Table 4.3) |
| `docs/step12_scenarios.md` | Step 12: pre-registered scenarios, feasibility check and comparisons |
| `tests/test_sim.py` | Step 12: simulator tests on a hand-made 2-day input |
| `tests/test_leakage.py`, `tests/test_split.py` | Step 5: leakage and split proofs |
| `tests/test_early_warning.py` | Step 11: event and alert measures on known series |

## Early prototype (simulated data, not part of the study)

These files were the first end-to-end sketch of the system on simulated appliance data.
They are **not** needed to reproduce the study results above.

| File | What it does |
|---|---|
| `sim/generate_data.py` | Simulates 4 weeks of per-minute readings for 4 appliances → `data/raw/simulated_power.csv` |
| `pipeline/feature_engineering.py` | Pivots and builds features → `data/processed/features.csv` |
| `pipeline/train_model.py` | Trains LightGBM → `models/lgbm_model.pkl` (not in git; run this to rebuild it) |
| `decision/shed.py` | `should_shed()`: picks a non-critical appliance to turn off above 1,500 W |
| `services/inference.py` | Simulates one real-time step on the Raspberry Pi |

Known limitation: the prototype model's inputs include the per-appliance watts that sum
to its target, so it reproduces the current load rather than forecasting. The UCI pipeline
above (`features.py` + `targets.py`, verified by `tests/test_leakage.py`) is the leakage-free
replacement.
