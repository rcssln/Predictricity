**Table 4.2. Early warning of demand spikes, test set (Jul–Nov 2010)**

| Rule | Early-warning rate | Median lead time (min) | Precision | Recall | F1 | Detection recall |
|---|---|---|---|---|---|---|
| Reactive (load > T now) | 0.0% | 0 | 51.8% | 0.0% | 0.00 | 100.0% |
| Predictive (forecast peak > T − 422 W) | 57.9% | 3 | 41.1% | 57.9% | 0.48 | 100.0% |

T = 3378 W, the 95th percentile of training load. 425 spike events (load above T after at least 10 minutes at or below it) in the test period, each with a forecast for the 10 minutes before it. The margin (12.5% of T) was chosen by best F1 on validation, before the test set was scored. Early-warning rate = recall = share of events with an alert 1–10 min before the start; lead time is the median over warned events; precision = share of alert episodes followed by an event start within 0–10 min; detection recall also counts an alert at the start minute. Alert episodes on test: reactive 820, predictive 1,196.
