# Step 16: replication on StoreNet

Skill vs persistence = 1 − MAE_model / MAE_persistence on each house's test set (Nov–Dec 2020). StoreNet: median and min–max across 18 houses; UCI: the single UCI household, Table 4.1 (test set, Jul–Nov 2010). Same pipeline and Step 8 settings, no new tuning.

| H | Target | Model | UCI skill | StoreNet median skill | StoreNet min–max | Houses beating persistence |
|---|---|---|---|---|---|---|
| 5 | y_next | LightGBM | 12.8% | 13.8% | 4.9% to 22.1% | 18 of 18 |
| 5 | y_next | XGBoost | 13.5% | 13.6% | -0.5% to 21.7% | 17 of 18 |
| 5 | y_peak | LightGBM | 4.6% | 8.4% | 3.5% to 20.8% | 18 of 18 |
| 5 | y_peak | XGBoost | 3.1% | 6.3% | -8.4% to 19.8% | 17 of 18 |
| 10 | y_next | LightGBM | 18.8% | 19.4% | 3.8% to 29.7% | 18 of 18 |
| 10 | y_next | XGBoost | 18.4% | 18.8% | -10.5% to 29.4% | 17 of 18 |
| 10 | y_peak | LightGBM | 8.5% | 11.8% | -7.5% to 25.0% | 17 of 18 |
| 10 | y_peak | XGBoost | 7.5% | 11.0% | -12.1% to 24.6% | 17 of 18 |
| 15 | y_next | LightGBM | 20.9% | 19.9% | -0.2% to 33.4% | 17 of 18 |
| 15 | y_next | XGBoost | 20.3% | 19.5% | -21.1% to 33.2% | 17 of 18 |
| 15 | y_peak | LightGBM | 12.4% | 14.9% | 1.3% to 26.3% | 18 of 18 |
| 15 | y_peak | XGBoost | 11.2% | 13.4% | -12.0% to 25.7% | 17 of 18 |
