# Step 16: StoreNet replication, pre-registration

*Written 11 October 2026, after inspecting the data format (Prompt 1) and before any model
was trained on StoreNet. This file is committed on its own, before any training, so that
the houses, split, method and criteria are on record before results exist. They will not
be changed after results are seen.*

## Purpose

The study tests whether the UCI forecasting results (Steps 1–15) hold on newer data. The
existing pipeline is run unchanged, with no new tuning, on StoreNet. UCI stays the main
dataset, and no Step 1–15 result changes.

## 1. Dataset

- **Source:** StoreNet 2020, 1-minute resolution: 20 households in Dingle, Ireland, each with
  a 10 kWh battery; 10 of them also have rooftop solar PV.
- **Citation:** Khadem et al. (2024), *Comprehensive Dataset on Electrical Load Profiles for
  Energy Community in Ireland*, Scientific Data. The dataset is at
  https://doi.org/10.6084/m9.figshare.c.6829134 (CC0).
- **Files:** `data/raw/storenet/H{n}_W.csv`. Only the consumption column is used. In the files
  it is named `Consumption(W)`, with a leading space in the header, not `Consumption_W` as in
  the paper. It is already in watts.

## 2. Houses

The rule is every house with at least 99% of 2020's 527,040 minutes holding a **valid
consumption value**. The timestamps themselves are almost always present (99.99%); the gaps
are missing consumption values inside them. Counting valid values reproduces the paper's
completeness figures.

**18 houses: H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13, H15, H17, H18, H19, H20.**

| Excluded house | Valid consumption | Paper |
|---|---|---|
| H14 | 92.25% (data ends 11 December 2020) | 92.25% |
| H16 | 96.40% | 96.41% |

Every eligible house is used. No house is dropped later for any reason, including its
results.

## 3. Split

The split is chronological, never shuffled:

| Set | Period |
|---|---|
| Training | 2020-01-01 00:00 to 2020-08-31 23:59 |
| Validation | 2020-09-01 00:00 to 2020-10-31 23:59 |
| Test | 2020-11-01 00:00 to 2020-12-31 23:59 |

As for UCI, the last H minutes of training and of validation are dropped, so that no target
window crosses into the next set. Rows with any missing feature or target are removed.

## 4. Method

- **Cleaning** (`pipeline/clean_storenet.py`):
  1. Parse the timestamps and keep the consumption column as `load_w`.
  2. Keep the first copy of any repeated minute. None were found.
  3. Set every run of **60 or more consecutive minutes of exactly 0 W** to missing. The
     inspection found such runs in most houses, mainly a site-wide outage around 28–30 April
     2020 (up to 2,199 minutes), plus single-house runs (H7 on 5 March, H20 on 5 January, H1
     on 13 January). A home's standby load never drops to exactly 0 W for an hour, so these
     are treated as missing data recorded as zero. Shorter zero runs stay as recorded.
  4. Reindex to every minute of calendar 2020.
  5. Fill gaps of 5 minutes or less with `fill_short_gaps()` from `pipeline/clean_uci.py`;
     longer gaps stay missing. This is the same rule as for UCI.
- **Features and targets:** the same 24 features (`pipeline/features.py`) and the same two
  targets, `y_next` and `y_peak` (`pipeline/targets.py`), at H = 5, 10 and 15 minutes.
- **Baselines:** the same three, with the existing code in `models/baselines.py`:
  persistence, same time yesterday, and the 15-minute moving average.
- **Models:** LightGBM and XGBoost with the best Step 8 setting for each H, target and
  algorithm, read from `results/tables/step08_tuning_log.csv` (lowest validation MAE).
  **No new tuning.** For the record, those settings are:

  | Model | Setting | Used for |
  |---|---|---|
  | LightGBM | absolute-error (l1) objective, num_leaves 127 | every H and target |
  | XGBoost | absolute-error objective, max_depth 10 | every H and target except below |
  | XGBoost | absolute-error objective, max_depth 8 | y_peak at H = 10 and H = 15 |

- **Fixed settings** (via `make_model()` and `fit()` in `models/train_demand.py`): up to 2,000
  trees, learning rate 0.03, row and column subsampling 0.8, early stopping after 100 rounds
  on the house's **StoreNet validation** rows.
- **One model per house**, horizon, target and algorithm: 18 × 3 × 2 × 2 = 216 fits.
- **Each house's test set is scored once**: MAE, RMSE, and skill = 1 − MAE_model /
  MAE_persistence, for both models and all three baselines, on the same test minutes.
- **Feature ablation (for R3):** the Step 15 runs are reused unchanged: all 24 features;
  without calendar, yesterday, rolling, long lags and diff; recent only; and now only. They
  are run for every house at H = 10, LightGBM only, scored on **validation only**.

## 5. Replication criteria (fixed now)

| # | Criterion | Counts as replicated if |
|---|---|---|
| R1 | Models beat persistence | Median skill vs persistence across houses > 0 for LightGBM and XGBoost at H = 5, 10, 15 and both targets, **and** a two-sided Wilcoxon signed-rank test across houses on house-level test MAE (model vs persistence, same house) gives p < 0.05 at H = 10. |
| R2 | Skill grows with horizon | Median LightGBM skill for y_next is higher at H = 15 than at H = 5. |
| R3 | Recent readings carry the accuracy | In the H = 10 ablation (validation only), "recent only" raises median validation MAE more than removing any single feature group. |

The R1 test pairs each house's model error with that same house's persistence error. It
does not compare UCI with StoreNet, which cannot be paired.

## 6. Reporting

Results are reported whatever they show, with the number of houses meeting each criterion.
The criteria will not be changed after results are seen. A criterion that is not met is
reported plainly, with one sentence on the likely reason.

## 7. Known limitations

- **Data length:** one year of data per house (8 months of training), against about 4 years
  for UCI (3 years of training).
- **COVID-19:** 2020 includes the COVID-19 lockdowns, which may have changed household
  routines.
- **Gaps and zeros:**
  - missing consumption values: 0.2–1.0% of minutes in the eligible houses
  - the logging outage around 28–30 April, recorded as 0 W and handled by the zero-run rule
    above
  - any gap handling the dataset authors applied before release
- **Time zone:** the timestamps are **not local Irish time with daylight saving**: there is
  no missing hour on 29 March 2020, no repeated hour on 25 October 2020, and no duplicate
  timestamps.
  - The files run from 2020-01-01 01:00 to 2021-01-01 00:59.
  - PV production centres around 12:41–13:06 in January for most PV houses, close to
    Dingle's solar noon of about 12:45 UTC. That points to a fixed clock close to UTC.
  - The calendar-2020 grid therefore misses the first hour of 1 January, and drops the
    files' last hour, which falls in 2021.
- **Location:** the houses are Irish, so they are no closer to Philippine households than the
  French UCI home; no suitable public Philippine 1-minute household dataset was available.
