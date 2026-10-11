# Step 16: feature ablation on StoreNet (H = 10, LightGBM, validation only)

Change in validation MAE when a feature group is removed (or only a small group kept), relative to all 24 features. StoreNet: median across 18 houses of each house's change, in W and %. UCI: the single household, from results/tables/step15_feature_ablation.md. Validation only; no test set is used.

## y_next

| Run | UCI change (W) | UCI change (%) | StoreNet median change (W) | StoreNet median change (%) | Houses where MAE rises |
|---|---|---|---|---|---|
| Without calendar | +3.0 | +1.1% | +1.7 | +1.2% | 15 of 18 |
| Without yesterday | -0.2 | -0.1% | -0.2 | -0.1% | 5 of 18 |
| Without rolling | +2.3 | +0.8% | +1.2 | +0.7% | 16 of 18 |
| Without long lags | +1.5 | +0.5% | +1.1 | +0.3% | 17 of 18 |
| Without diff | +1.2 | +0.4% | +0.3 | +0.1% | 14 of 18 |
| Recent only | +18.4 | +6.7% | +12.4 | +4.8% | 17 of 18 |
| Now only | +35.4 | +12.9% | +28.9 | +12.1% | 18 of 18 |

## y_peak

| Run | UCI change (W) | UCI change (%) | StoreNet median change (W) | StoreNet median change (%) | Houses where MAE rises |
|---|---|---|---|---|---|
| Without calendar | +2.0 | +0.8% | +0.9 | +0.3% | 14 of 18 |
| Without yesterday | +0.4 | +0.1% | -0.1 | -0.1% | 8 of 18 |
| Without rolling | +1.2 | +0.5% | +2.0 | +0.7% | 18 of 18 |
| Without long lags | +0.9 | +0.3% | +0.8 | +0.3% | 14 of 18 |
| Without diff | +1.3 | +0.5% | +1.4 | +0.5% | 18 of 18 |
| Recent only | +6.4 | +2.4% | +6.2 | +2.7% | 18 of 18 |
| Now only | +11.8 | +4.4% | +27.6 | +11.7% | 18 of 18 |

