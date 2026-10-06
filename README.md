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

## Quick reproduce (everything, in order)

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
```

## Files that make up the study

| File | Role |
|---|---|
| `pipeline/clean_uci.py` | Step 2: raw UCI → clean 1-minute parquet |
| `pipeline/check_uci.py` | Step 3: data summary table |
| `notebooks/step04_explore.ipynb` | Step 4: exploratory figures |
| `pipeline/features.py` | `add_features()`: lags, rolling stats, time-of-day, day-of-week (past-only) |
| `pipeline/targets.py` | `add_targets()`: `y_next_H`, `y_peak_H` (future-only) |
| `pipeline/split.py` | Step 6: `make_splits()` chronological split |
| `models/baselines.py` | Step 7: baseline scores on validation |
| `models/train_demand.py` | Step 8: tune LightGBM / XGBoost on validation, save best models |
| `models/evaluate.py` | Step 9: score everything on the test set once (Table 4.1) |
| `tests/test_leakage.py`, `tests/test_split.py` | Step 5: leakage and split proofs |

## Early prototype (simulated data, not part of the study)

These files were the first end-to-end sketch of the system on simulated appliance data.
They are **not** needed to reproduce the study results above.

| File | What it does |
|---|---|
| `sim/generate_data.py` | Simulates 4 weeks of per-minute readings for 4 appliances → `data/raw/simulated_power.csv` |
| `pipeline/feature_engineering.py` | Pivots and builds features → `data/processed/features.csv` |
| `pipeline/train_model.py` | Trains LightGBM → `models/lgbm_model.pkl` |
| `decision/shed.py` | `should_shed()`: picks a non-critical appliance to turn off above 1,500 W |
| `services/inference.py` | Simulates one real-time step on the Raspberry Pi |

Known limitation: the prototype model's inputs include the per-appliance watts that sum
to its target, so it reproduces the current load rather than forecasting. The UCI pipeline
above (`features.py` + `targets.py`, verified by `tests/test_leakage.py`) is the leakage-free
replacement.
