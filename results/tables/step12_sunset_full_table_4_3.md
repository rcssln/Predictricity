**Table 4.3 (preliminary). Load-shedding policies in simulation, scenario C, sunset-full (43 Wh, 80 W, 100% at 17:00, days 17:00–17:00)**

| Policy | Battery-empty min/day | Early-warning rate | Median lead time (min) | Shed precision | Shed recall | Comfort cost (zone-min/day) | Critical interruptions | Switches/day |
|---|---|---|---|---|---|---|---|---|
| No control | 505.7 | 0.0% | 0 | – | 0.0% | 0.0 | 0 | 0.0 |
| Reactive | 426.5 | 0.6% | 6 | 72.6% | 97.8% | 391.7 | 0 | 366.7 |
| Predictive, demand only | 424.2 | 6.0% | 2 | 57.5% | 94.3% | 394.8 | 0 | 353.2 |
| Predictive, demand + solar | 424.1 | 6.8% | 2 | 58.8% | 94.6% | 394.8 | 0 | 354.3 |

135 complete test days, battery 43 Wh usable starting at 100% at 17:00 each day (a simulated day runs 17:00 to 17:00 the next day, labelled by its start date), modeled solar 80 W peak (half-sine with random clouds) until the real panel log exists. Without control, 0 of 135 days have no battery-empty minutes. Battery-empty minutes, comfort cost and switches are means per day; the warning measures are pooled over all 1,460 deficit events (from the No-control run, the same for every policy) and all shedding episodes. Early warning = a shedding episode starts 1–10 min before a deficit; shed recall also counts shedding already on at the start minute.
