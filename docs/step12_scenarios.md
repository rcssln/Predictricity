# Step 12 scenarios: pre-registration

*Written 7 October 2026, before any of the new scenarios was simulated. This file is committed
before the new runs so that the change of scenario is on record as a design decision, not a
search for a better result.*

## Why the first run was uninformative

The first Step 12 run (`results/tables/step12_table_4_3_preliminary.md`) used the testbed's 43 Wh
battery, starting at 60% each midnight, with 80 W of modeled solar. Without control, **0 of 136**
days had no battery-empty minutes, and the battery was empty for **488.8 min/day**. Shedding
lowered that to 409.4 (Reactive), 407.3 (Predictive, demand only) and 407.1 min/day
(Predictive, demand + solar).

The battery was empty overnight and in the evening under every policy, because 43 Wh cannot
cover evening demand (about 195 Wh from 18:00 to 24:00). Once the battery is empty there is
nothing left to protect, so the three shedding policies end up almost identical (2.1–2.3
min/day apart), and the solar-forecast ablation cannot show an effect. That run answers
"can a 43 Wh battery carry this household?" (no), not "does forecasting improve shedding?".

## Scenarios

Everything not listed here stays exactly as in the current `sim/config.yaml`: zones, demand
scaling and replay, the LightGBM forecasts, the solar model's shape and clouds, the shedding
rules, the restore rule, efficiencies and the allowance table.

| Scenario | Battery (usable) | Day runs | Start charge | Solar peak | Role |
|---|---|---|---|---|---|
| A. `original` | 43 Wh | 00:00 → 00:00 | 60% | 80 W | Sensitivity (the first run, kept unchanged) |
| B. `evening_sized` | 220 Wh | 00:00 → 00:00 | 60% | 104 W | **Primary: reported as Table 4.3** |
| C. `sunset_full` | 43 Wh | 17:00 → 17:00 next day | 100% | 80 W | Sensitivity |

- **B's battery (220 Wh)** is about the size of the evening's demand: about 195 Wh from 18:00
  to 24:00 on average, or about 229 Wh drawn from the battery once the 85% inverter
  efficiency is included. Heavy evenings can therefore still empty it, which is what lets
  shedding make a difference.
- **B's array (104 W)** follows a fixed rule, not a result: mean daily solar equals mean daily
  replayed demand over the 136 simulated days. That is 552.6 Wh/day of demand against 424.7
  Wh/day of solar at 80 W, so the peak is scaled by 552.6 / 424.7, which gives 104.1 W,
  rounded to 104 W.
- **C** keeps the testbed battery but starts each simulated day full at 17:00, just before the
  evening peak. A day is the 1,440 minutes from 17:00 to 16:59 the next day, and only
  windows with every minute present are used. Per-day metrics, deficit events and the
  Wilcoxon pairing all use this day definition.
- **min_switch_min** stays at 1 in every scenario.

## Feasibility check, before any policy is compared

For each scenario, the No-control run alone is checked first. A scenario can only separate
policies if some days end with battery-empty minutes and some do not. If No control has zero
empty minutes on almost every day, or empty minutes on every day, the scenario is reported as
unable to separate policies and the work stops for a decision. No setting is tuned to make a
scenario pass.

## Comparisons

- **Primary outcome:** battery-empty minutes per day.
- **Primary comparison:** Predictive (demand only) vs Reactive, in scenario B.
- **Test:** Wilcoxon signed-rank, paired by day, two-sided, α = 0.05.
- **Secondary comparisons**, all in scenario B, on battery-empty minutes and on comfort cost
  (zone-minutes shed):
  - Predictive (demand + solar) vs Predictive (demand only): the solar-forecast ablation
  - Predictive (demand + solar) vs Reactive
  - each of the three shedding policies vs No control
- **Multiple comparisons:** the secondary comparisons are reported with raw and
  Holm-adjusted p-values, within each outcome.
- **Also reported, without tests:** early-warning rate, median lead time, shed precision and
  recall, switches per day, and critical interruptions (must be 0).
- **Scenarios A and C** are reported with the same tables and tests, as sensitivity results.
- **Switching sensitivity:** scenario B is re-run with min_switch_min = 1, 5 and 10. It is
  reported as is, and the primary setting (1) does not change because of it.

Whatever the results show, they will be reported. If Predictive does not beat Reactive in
scenario B, that is the finding.
