# Step 16: StoreNet data summary (calendar 2020, after cleaning)

Source: data/processed/storenet/H{n}_1min.parquet (18 pre-registered houses). Missing before cleaning counts minutes of 2020 with no valid consumption value; zero-run minutes are runs of >= 60 min at exactly 0 W, set to missing; gaps of <= 5 min are then filled. Load statistics exclude missing values. Most zero-run minutes fall on the 5th of each month from January to May (almost every house reads 0 W from 00:00 for about a day) and on 28-30 April 2020: logging artifacts found while cleaning, which the pre-registered rule treats as missing.

| House | Rows | Duplicates dropped | Missing before cleaning | Zero-run minutes set missing | Missing after cleaning | Longest gap (min) | Mean load (W) | Median load (W) | 95th percentile (W) | PV | Flags |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| H1 | 527,040 | 0 | 2,016 | 4,202 | 5,415 | 605 | 365.7 | 30.0 | 2518.0 | yes | long zero runs set missing |
| H2 | 527,040 | 0 | 1,404 | 4,768 | 5,364 | 605 | 560.4 | 243.0 | 2838.0 | yes | long zero runs set missing |
| H3 | 527,040 | 0 | 1,154 | 4,760 | 5,124 | 590 | 606.3 | 286.0 | 3055.0 | yes | long zero runs set missing |
| H4 | 527,040 | 0 | 2,569 | 144 | 2,043 | 1,440 | 923.3 | 309.0 | 3473.0 | yes | long zero runs set missing |
| H5 | 527,040 | 0 | 2,039 | 4,988 | 5,880 | 636 | 280.9 | 186.0 | 896.0 | yes | long zero runs set missing |
| H6 | 527,040 | 0 | 4,995 | 10,126 | 13,971 | 3,431 | 726.0 | 401.0 | 2809.0 | no | long zero runs set missing |
| H7 | 527,040 | 0 | 1,988 | 13,859 | 14,979 | 3,496 | 762.1 | 345.0 | 2954.0 | yes | long zero runs set missing |
| H8 | 527,040 | 0 | 3,092 | 10,655 | 12,553 | 3,480 | 790.8 | 299.0 | 3557.0 | no | long zero runs set missing |
| H9 | 527,040 | 0 | 2,258 | 10,212 | 11,636 | 3,397 | 447.7 | 263.0 | 1503.0 | no | long zero runs set missing |
| H10 | 527,040 | 0 | 2,377 | 13,186 | 14,785 | 3,525 | 931.6 | 505.0 | 3299.0 | yes | long zero runs set missing |
| H11 | 527,040 | 0 | 3,316 | 10,297 | 12,664 | 3,522 | 706.6 | 287.0 | 2928.0 | yes | long zero runs set missing |
| H12 | 527,040 | 0 | 4,522 | 10,057 | 13,148 | 3,337 | 339.3 | 233.0 | 1104.0 | no | long zero runs set missing |
| H13 | 527,040 | 0 | 3,545 | 10,228 | 13,040 | 3,457 | 322.0 | 173.0 | 1333.0 | yes | long zero runs set missing |
| H15 | 527,040 | 0 | 2,144 | 10,006 | 11,349 | 3,333 | 678.4 | 353.0 | 2573.0 | no | long zero runs set missing |
| H17 | 527,040 | 0 | 2,311 | 10,198 | 11,686 | 3,484 | 212.9 | 92.0 | 1066.0 | yes | long zero runs set missing |
| H18 | 527,040 | 0 | 2,552 | 10,105 | 11,597 | 3,461 | 667.9 | 373.0 | 2501.0 | no | long zero runs set missing |
| H19 | 527,040 | 0 | 2,971 | 0 | 2,187 | 861 | 579.9 | 306.0 | 2381.0 | no | none |
| H20 | 527,040 | 0 | 2,195 | 10,060 | 11,293 | 3,393 | 393.0 | 165.0 | 1870.0 | no | long zero runs set missing |
| **Median of 18 houses** | | | | | | | 593.1 | 286.5 | 2545.5 | 10 of 18 | |
| UCI (Step 3, 2006–2010) | 2,075,259 | – | 25,979 | – | 25,903 | 7,226 | 1091.6 | 602.0 | 3264.0 | no | – |
