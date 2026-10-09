**Feature ablation, H = 10 LightGBM, validation set (Jan–Jun 2010)**

| Run | Change | Features | y_next val MAE (W) | Δ vs all | y_peak val MAE (W) | Δ vs all |
|---|---|---|---|---|---|---|
| All 24 features | the reference | 24 | 274.7 | +0.0 (+0.0%) | 264.7 | +0.0 (+0.0%) |
| Without calendar | time of day, day of week, weekend flag removed | 20 | 277.7 | +3.0 (+1.1%) | 266.7 | +2.0 (+0.8%) |
| Without yesterday | load at this time yesterday removed | 23 | 274.5 | -0.2 (-0.1%) | 265.1 | +0.4 (+0.1%) |
| Without rolling | 5/15/60-min mean, std, max removed | 15 | 277.0 | +2.3 (+0.8%) | 265.9 | +1.2 (+0.5%) |
| Without long lags | lags 15, 30, 60 removed | 21 | 276.2 | +1.5 (+0.5%) | 265.6 | +0.9 (+0.3%) |
| Without diff | the 1-minute change removed | 23 | 275.9 | +1.2 (+0.4%) | 266.0 | +1.3 (+0.5%) |
| Recent only | now, lags 1-5 and diff_1 kept | 6 | 293.1 | +18.4 (+6.7%) | 271.1 | +6.4 (+2.4%) |
| Now only | the current load only | 1 | 310.1 | +35.4 (+12.9%) | 276.4 | +11.8 (+4.4%) |

Lowest validation MAE: Without yesterday (y_next), All 24 features (y_peak). Each run uses the Step 8 setting for its target and the same rows; early stopping uses validation, so the scores are optimistic and only the comparison between runs matters. The test set was not used.
