# Step 14: day-level tests, effect sizes and bootstrap CIs (Step 12 simulation)

Paired by simulated day. Difference = first policy − second policy, so a negative value means the first policy has less. 95% CI: percentile bootstrap over days (10,000 resamples). Rank-biserial r: −1 = every non-tied day favours the first policy, +1 = every day favours the second. p: Wilcoxon signed-rank, two-sided; Holm-adjusted within scenario and outcome except the pre-registered primary comparison (scenario B, battery-empty minutes, Reactive vs Predictive demand only), which is marked *.

## Scenario B, evening-sized (primary) (136 days)

| Outcome | Comparison | Means (first / second) | Difference (95% CI) | First lower / second lower / tied days | Rank-biserial r | p |
|---|---|---|---|---|---|---|
| Battery-empty min/day | Reactive vs Predictive, demand only * | 110.1 / 103.1 | +6.9 (+5.1 to +8.9) | 3 / 67 / 66 | +0.97 | < 0.001 |
| Battery-empty min/day | Predictive, demand only vs Predictive, demand + solar | 103.1 / 103.3 | -0.2 (-0.7 to +0.3) | 19 / 16 / 101 | -0.06 | 0.748 |
| Battery-empty min/day | Reactive vs Predictive, demand + solar | 110.1 / 103.3 | +6.7 (+4.8 to +8.8) | 6 / 66 / 64 | +0.92 | < 0.001 |
| Battery-empty min/day | No control vs Reactive | 205.8 / 110.1 | +95.7 (+80.4 to +111.0) | 0 / 104 / 32 | +1.00 | < 0.001 |
| Battery-empty min/day | No control vs Predictive, demand only | 205.8 / 103.1 | +102.6 (+86.6 to +119.5) | 0 / 104 / 32 | +1.00 | < 0.001 |
| Comfort cost | Reactive vs Predictive, demand only | 249.6 / 260.1 | -10.5 (-12.7 to -8.4) | 106 / 6 / 24 | -0.96 | < 0.001 |
| Comfort cost | Predictive, demand only vs Predictive, demand + solar | 260.1 / 259.5 | +0.6 (+0.2 to +1.1) | 17 / 41 / 78 | +0.43 | 0.004 |
| Comfort cost | Reactive vs Predictive, demand + solar | 249.6 / 259.5 | -9.8 (-12.0 to -7.8) | 105 / 6 / 25 | -0.95 | < 0.001 |
| Comfort cost | No control vs Reactive | 0.0 / 249.6 | -249.6 (-291.8 to -208.5) | 130 / 0 / 6 | -1.00 | < 0.001 |
| Comfort cost | No control vs Predictive, demand only | 0.0 / 260.1 | -260.1 (-302.8 to -218.6) | 130 / 0 / 6 | -1.00 | < 0.001 |

## Scenario A, original (136 days)

| Outcome | Comparison | Means (first / second) | Difference (95% CI) | First lower / second lower / tied days | Rank-biserial r | p |
|---|---|---|---|---|---|---|
| Battery-empty min/day | Reactive vs Predictive, demand only | 409.4 / 407.3 | +2.1 (+1.2 to +3.2) | 12 / 49 / 75 | +0.75 | < 0.001 |
| Battery-empty min/day | Predictive, demand only vs Predictive, demand + solar | 407.3 / 407.1 | +0.1 (-0.1 to +0.4) | 14 / 18 / 104 | +0.13 | 0.503 |
| Battery-empty min/day | Reactive vs Predictive, demand + solar | 409.4 / 407.1 | +2.3 (+1.3 to +3.4) | 11 / 51 / 74 | +0.76 | < 0.001 |
| Battery-empty min/day | No control vs Reactive | 488.8 / 409.4 | +79.3 (+65.2 to +94.5) | 0 / 116 / 20 | +1.00 | < 0.001 |
| Battery-empty min/day | No control vs Predictive, demand only | 488.8 / 407.3 | +81.5 (+67.0 to +96.3) | 0 / 118 / 18 | +1.00 | < 0.001 |
| Comfort cost | Reactive vs Predictive, demand only | 390.9 / 393.4 | -2.5 (-3.3 to -1.8) | 78 / 22 / 36 | -0.73 | < 0.001 |
| Comfort cost | Predictive, demand only vs Predictive, demand + solar | 393.4 / 393.3 | +0.1 (-0.1 to +0.3) | 16 / 25 / 95 | +0.22 | 0.219 |
| Comfort cost | Reactive vs Predictive, demand + solar | 390.9 / 393.3 | -2.4 (-3.2 to -1.6) | 76 / 25 / 35 | -0.67 | < 0.001 |
| Comfort cost | No control vs Reactive | 0.0 / 390.9 | -390.9 (-438.9 to -344.2) | 136 / 0 / 0 | -1.00 | < 0.001 |
| Comfort cost | No control vs Predictive, demand only | 0.0 / 393.4 | -393.4 (-441.5 to -345.9) | 136 / 0 / 0 | -1.00 | < 0.001 |

## Scenario C, sunset-full (135 days)

| Outcome | Comparison | Means (first / second) | Difference (95% CI) | First lower / second lower / tied days | Rank-biserial r | p |
|---|---|---|---|---|---|---|
| Battery-empty min/day | Reactive vs Predictive, demand only | 426.5 / 424.2 | +2.2 (+1.3 to +3.3) | 12 / 55 / 68 | +0.75 | < 0.001 |
| Battery-empty min/day | Predictive, demand only vs Predictive, demand + solar | 424.2 / 424.1 | +0.2 (-0.1 to +0.5) | 14 / 15 / 106 | +0.16 | 0.458 |
| Battery-empty min/day | Reactive vs Predictive, demand + solar | 426.5 / 424.1 | +2.4 (+1.4 to +3.7) | 11 / 52 / 72 | +0.79 | < 0.001 |
| Battery-empty min/day | No control vs Reactive | 505.7 / 426.5 | +79.2 (+64.5 to +94.7) | 0 / 115 / 20 | +1.00 | < 0.001 |
| Battery-empty min/day | No control vs Predictive, demand only | 505.7 / 424.2 | +81.4 (+66.4 to +97.5) | 0 / 116 / 19 | +1.00 | < 0.001 |
| Comfort cost | Reactive vs Predictive, demand only | 391.7 / 394.8 | -3.0 (-4.0 to -2.2) | 78 / 19 / 38 | -0.76 | < 0.001 |
| Comfort cost | Predictive, demand only vs Predictive, demand + solar | 394.8 / 394.8 | -0.0 (-0.3 to +0.2) | 19 / 25 / 91 | +0.02 | 0.901 |
| Comfort cost | Reactive vs Predictive, demand + solar | 391.7 / 394.8 | -3.0 (-4.0 to -2.1) | 78 / 20 / 37 | -0.73 | < 0.001 |
| Comfort cost | No control vs Reactive | 0.0 / 391.7 | -391.7 (-440.5 to -344.8) | 135 / 0 / 0 | -1.00 | < 0.001 |
| Comfort cost | No control vs Predictive, demand only | 0.0 / 394.8 | -394.8 (-442.9 to -348.4) | 135 / 0 / 0 | -1.00 | < 0.001 |

