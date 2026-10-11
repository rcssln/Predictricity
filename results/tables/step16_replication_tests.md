# Step 16: R1 tests, model vs persistence across houses

Wilcoxon signed-rank test, two-sided, paired by house (18 houses), on house-level test MAE. Median difference = median over houses of (MAE_model − MAE_persistence), in W; negative means the model has the lower error. Rank-biserial r: −1 = every house favours the model. The pre-registered R1 test is H = 10; H = 5 and 15 are extra.

| H | Target | Model | W | p | Median difference (W) | Rank-biserial r | Role |
|---|---|---|---|---|---|---|---|
| 5 | y_next | LightGBM | 0.0 | < 0.001 | -33.2 | -1.00 | extra |
| 5 | y_next | XGBoost | 1.0 | < 0.001 | -32.1 | -0.99 | extra |
| 5 | y_peak | LightGBM | 0.0 | < 0.001 | -19.6 | -1.00 | extra |
| 5 | y_peak | XGBoost | 4.0 | < 0.001 | -16.0 | -0.95 | extra |
| 10 | y_next | LightGBM | 0.0 | < 0.001 | -60.0 | -1.00 | R1 (pre-registered) |
| 10 | y_next | XGBoost | 1.0 | < 0.001 | -57.7 | -0.99 | R1 (pre-registered) |
| 10 | y_peak | LightGBM | 1.0 | < 0.001 | -48.7 | -0.99 | R1 (pre-registered) |
| 10 | y_peak | XGBoost | 3.0 | < 0.001 | -43.2 | -0.96 | R1 (pre-registered) |
| 15 | y_next | LightGBM | 1.0 | < 0.001 | -75.6 | -0.99 | extra |
| 15 | y_next | XGBoost | 1.0 | < 0.001 | -75.0 | -0.99 | extra |
| 15 | y_peak | LightGBM | 0.0 | < 0.001 | -68.1 | -1.00 | extra |
| 15 | y_peak | XGBoost | 1.0 | < 0.001 | -57.9 | -0.99 | extra |
