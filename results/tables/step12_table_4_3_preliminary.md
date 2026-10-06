**Table 4.3 (preliminary). Load-shedding policies in simulation, testbed settings**

| Policy | Battery-empty min/day | Early-warning rate | Median lead time (min) | Shed precision | Shed recall | Comfort cost (zone-min/day) | Critical interruptions | Switches/day |
|---|---|---|---|---|---|---|---|---|
| No control | 488.8 | 0.0% | 0 | – | 0.0% | 0.0 | 0 | 0.0 |
| Reactive | 409.4 | 0.7% | 6 | 75.0% | 97.2% | 390.9 | 0 | 300.7 |
| Predictive, demand only | 407.3 | 7.0% | 2 | 57.8% | 93.9% | 393.4 | 0 | 282.8 |
| Predictive, demand + solar | 407.1 | 7.5% | 2 | 58.7% | 94.1% | 393.3 | 0 | 283.7 |

136 complete test days, battery 43 Wh usable starting at 60% each midnight, modeled solar 80 W peak (half-sine with random clouds) until the real panel log exists. Without control, 0 of 136 days have no battery-empty minutes. Battery-empty minutes, comfort cost and switches are means per day; the warning measures are pooled over all 1,418 deficit events (from the No-control run, the same for every policy) and all shedding episodes. Early warning = a shedding episode starts 1–10 min before a deficit; shed recall also counts shedding already on at the start minute.
