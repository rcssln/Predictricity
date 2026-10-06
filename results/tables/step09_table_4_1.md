**Table 4.1a. Test-set errors, `y_next` (198,995–199,025 test minutes per H)**

| Model | MAE 10 min (W) | RMSE 10 min (W) | Skill vs persistence (10 min) | MAE 5 min (W) | MAE 15 min (W) |
|---|---|---|---|---|---|
| Persistence | 316.4 | 595.9 | 0.0% | 221.2 | 369.7 |
| Same time yesterday | 639.4 | 992.0 | -102.1% | 639.5 | 639.3 |
| Moving average 15 | 347.3 | 604.2 | -9.8% | 298.6 | 380.9 |
| LightGBM | 257.1 | 513.0 | 18.8% | 192.7 | 292.4 |
| XGBoost | 258.1 | 508.3 | 18.4% | 191.3 | 294.7 |

**Table 4.1b. Test-set errors, `y_peak` (198,995–199,025 test minutes per H)**

| Model | MAE 10 min (W) | RMSE 10 min (W) | Skill vs persistence (10 min) | MAE 5 min (W) | MAE 15 min (W) |
|---|---|---|---|---|---|
| Persistence | 270.4 | 563.1 | 0.0% | 163.8 | 364.6 |
| Same time yesterday | 734.8 | 1120.8 | -171.8% | 686.4 | 782.3 |
| Moving average 15 | 386.0 | 671.0 | -42.8% | 304.6 | 461.4 |
| LightGBM | 247.4 | 518.8 | 8.5% | 156.3 | 319.4 |
| XGBoost | 250.2 | 508.3 | 7.5% | 158.7 | 323.7 |

Skill vs persistence = 1 − MAE_model / MAE_persistence. All rows for one H are scored on the same test minutes.
