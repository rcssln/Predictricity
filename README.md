# Predictricity

Short-horizon household load forecasting for peak-aware load shedding.

The study uses the **UCI Individual Household Electric Power Consumption** dataset
(one French household, 1-minute readings, Dec 2006 – Nov 2010) to forecast
**the load H minutes ahead** (`y_next_H`) and **the peak load over the next H minutes**
(`y_peak_H`) for H = 5, 10 and 15. Every result in `results/` can be regenerated
by following the steps below in order.

---

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
