# Predictricity: initial results

*7 October 2026. Every number below comes from `results/tables/`; all of them can be rebuilt with `python run_initial_results.py`.*

## 1. Objective

To test whether 5–15 minute household demand forecasts let a small solar–battery system shed non-critical loads earlier and protect its battery better than a controller that reacts only to current demand.

## 2. Data

- **Source:** UCI *Individual Household Electric Power Consumption*, one household in France, 1-minute active power, 16 Dec 2006 – 26 Nov 2010 (2,075,259 minutes).
- **Cleaning:** reindexed to a strict 1-minute grid; gaps of up to 5 minutes interpolated; longer gaps left missing (25,903 missing minutes after cleaning).
- **Chronological split, no shuffling:**
  - **Train:** Dec 2006 – Dec 2009 (1,585,707 rows)
  - **Validation:** Jan – Jun 2010 (252,574)
  - **Test:** Jul – Nov 2010 (199,010)
- **Leakage controls:** features use only data up to time t, and targets only data after t. The last H minutes of train and validation are dropped. pytest checks both rules.

## 3. Method

- **Targets** at horizons H = 5, 10 and 15 min:
  - `y_next`: the load at t + H
  - `y_peak`: the maximum load over t+1 … t+H
- **Baselines:** persistence, same time yesterday, and a 15-minute moving average.
- **Models:** LightGBM and XGBoost with 24 lag, rolling and calendar features. Each was tuned on validation only (72 fits, squared- and absolute-error objectives). The test set was scored once, after a validation-only dry run reproduced the validation scores exactly.
- **Early warning:**
  - **Spike:** load rising above T = 3,378 W (the 95th percentile of training load) after at least 10 quiet minutes.
  - **Predictive alert:** the H = 10 `y_peak` forecast exceeds T − margin. The margin was chosen by best F1 on validation.
  - **Reactive alert:** the current load exceeds T.
  - **Recall:** counts only alerts 1–10 min *before* a spike.
- **Policy simulation (preliminary):**
  - **Setup:** UCI test demand scaled to the testbed's four zones (critical 25 W, never shed; 40, 20 and 25 W sheddable), with modeled solar (a half-sine with random clouds).
  - **Policies:** No control, Reactive, Predictive (demand only) and Predictive (demand + solar). All four use the same shedding rules; only the information they act on differs.
  - **Scenarios**, pre-registered in [docs/step12_scenarios.md](step12_scenarios.md) before they were run:
    - **B (primary):** a 220 Wh battery starting each midnight at 60%, and a 104 W array.
    - **A:** the first run, with the testbed's 43 Wh battery and 80 W.
    - **C:** 43 Wh, starting full at 17:00, with each day running from 17:00 to 17:00.
  - **Primary comparison:** Predictive (demand only) vs Reactive on battery-empty minutes in B, using a Wilcoxon signed-rank test paired by day.

## 4. Results

**Table 4.1 (test, H = 10 min).** MAE and RMSE in W; skill = 1 − MAE / MAE of persistence.

| Model | `y_next` MAE | RMSE | Skill | `y_peak` MAE | RMSE | Skill |
|---|---|---|---|---|---|---|
| Persistence | 316.4 | 595.9 | 0.0% | 270.4 | 563.1 | 0.0% |
| Same time yesterday | 639.4 | 992.0 | −102.1% | 734.8 | 1120.8 | −171.8% |
| Moving average 15 | 347.3 | 604.2 | −9.8% | 386.0 | 671.0 | −42.8% |
| LightGBM | **257.1** | 513.0 | **18.8%** | **247.4** | 518.8 | **8.5%** |
| XGBoost | 258.1 | 508.3 | 18.4% | 250.2 | 508.3 | 7.5% |

Both models beat every baseline at every horizon. The gain over persistence grows with H, from 13.5% at 5 min to 20.9% at 15 min for `y_next`. It is smallest for `y_peak` at 5 min (4.6%).

**Table 4.2 (test, 425 spikes).**

| Rule | Early-warning rate | Median lead (min) | Precision | F1 |
|---|---|---|---|---|
| Reactive | 0.0% | 0 | 51.8% | 0.00 |
| Predictive (margin 422 W) | 57.9% | 3 | 41.1% | 0.48 |

The forecast warns before 57.9% of spikes, a median of 3 min ahead, in exchange for more false alarms.

**Table 4.3 (PRELIMINARY: solar is modeled). Policy simulation, scenario B, 136 days.**

| Policy | Battery-empty min/day | Early warning | Shed precision | Comfort cost (zone-min/day) | Switches/day | Critical interruptions |
|---|---|---|---|---|---|---|
| No control | 205.8 | 0.0% | – | 0.0 | 0.0 | 0 |
| Reactive | 110.1 | 0.3% | 70.6% | 249.6 | 86.9 | 0 |
| Predictive, demand | **103.1** | **15.6%** | 61.3% | 260.1 | 75.5 | 0 |
| Predictive, demand + solar | 103.3 | 15.4% | 61.1% | 259.5 | 75.8 | 0 |

- **Primary result: predictive beats reactive.** Battery-empty minutes are 103.1 vs 110.1 per day (Wilcoxon, p < 0.001); the two differ on 70 of 136 days.
- **The cost is comfort:** predictive sheds more (260.1 vs 249.6 zone-min/day, Holm-adjusted p < 0.001) and its shedding is less precise (61.3% vs 70.6%). It also starts shedding before more deficits (15.6% vs 0.3%).
- **All three shedding policies roughly halve empty-battery time** compared with no control (205.8 min/day; Holm-adjusted p < 0.001).
- **The solar forecast adds nothing measurable** (103.3 vs 103.1 min/day, p = 0.75).

**Why the first scenario was replaced.** The first run (scenario A) used the testbed's 43 Wh battery. Without control the battery was empty on every one of the 136 days, for 488.8 min/day, and the three shedding policies came out almost identical (407.1–409.4 min/day). An empty battery leaves nothing for a better policy to protect. Scenario B was therefore defined and recorded in [docs/step12_scenarios.md](step12_scenarios.md) before it was run. Without control it leaves 32 of 136 days with no battery-empty minutes, so policies can be told apart.

**Sensitivity.**
- **Scenario C** (43 Wh, starting full at 17:00) behaves like A: the battery empties every day, and predictive edges reactive by 424.2 vs 426.5 min/day (p < 0.001).
- **Switching:** in scenario B, raising the minimum switch time from 1 to 5 to 10 minutes keeps predictive ahead of reactive at every setting (105.1 vs 113.3, then 106.7 vs 117.3 min/day). Switches per day drop to 43.8 and 30.7 for predictive.

## 5. Limitations

- **One household:** a single French household from 2006–2010, so the forecast results may not transfer to other households or load profiles.
- **Modeled solar:** solar is modeled, not measured, which is why Table 4.3 is preliminary.
- **Simplified demand:** demand is scaled and discretised onto four fixed zones.
- **Battery sizing:** the testbed's 43 Wh battery cannot carry evening demand: it is empty on every simulated day in scenarios A and C. The primary scenario B assumes a 220 Wh battery that the testbed does not have.
- **Shedding rules:** the shed order, restore rule and cutoff were designed for the simulation, not taken from the literature.

## 6. Next steps

1. Log the real panel output (E2), replace the modeled solar, and re-run Step 12.
2. Decide the testbed battery size: scenario B (220 Wh) is the setting in which forecasting made a measurable difference.
3. Run live sessions on the testbed (E4) with the frozen models.
