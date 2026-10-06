# Step 10 figure captions (Chapter 4)

**Figure 4.1. `step10_pred_vs_actual.png`.** Actual household load (grey) and the LightGBM `y_next` forecast made 10 minutes earlier (blue) on a test weekday (Fri 22 Oct 2010) and weekend day (Sun 15 Aug 2010). Each is the complete test day of its type whose forecast error is closest to the median, so neither is hand-picked (MAE 250 W and 264 W). The forecast follows the slow daily shape closely, but sudden appliance switch-ons show up in it only after they start.

**Figure 4.2. `step10_feature_importance.png`.** Share of total split gain for the 15 most important of 24 features in the H = 10 min LightGBM `y_peak` model. The current load (`now`, 68.9%) and the load one minute earlier (`lag_1`, 13.9%) hold 83% of the gain; short-window statistics come next, and the time-of-day features hold 1.5%. The model leans on the most recent readings, consistent with its modest gain over persistence at short horizons.

**Figure 4.3. `step10_error_by_hour.png`.** Test-set MAE of LightGBM and persistence by hour of day at H = 10 min, for `y_next` (left) and `y_peak` (right). Errors follow household activity: LightGBM's `y_next` MAE averages about 150 W from 00:00 to 05:59 and about 390 W from 18:00 to 20:59. LightGBM is lower than persistence in every hour, by 36–84 W for `y_next` and 13–46 W for `y_peak`.

**Figure 4.4. `step10_error_by_horizon.png`.** Test-set MAE at H = 5, 10 and 15 min for both models, persistence and the 15-minute moving average. Same time yesterday is left off because it is far worse (639 W for `y_next`, 686–782 W for `y_peak`). LightGBM and XGBoost are almost identical, and their lead over persistence widens from 30 W at H = 5 to 77 W at H = 15 for `y_next`, and from 8 W to 45 W for `y_peak`.
