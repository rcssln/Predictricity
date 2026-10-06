**Table 4.3b (sensitivity). The same policies with a larger battery and array (86 Wh, 120 W)**

| Policy | Battery-empty min/day | Early-warning rate | Median lead time (min) | Shed precision | Shed recall | Comfort cost (zone-min/day) | Critical interruptions | Switches/day |
|---|---|---|---|---|---|---|---|---|
| No control | 327.7 | 0.0% | 0 | – | 0.0% | 0.0 | 0 | 0.0 |
| Reactive | 256.2 | 1.0% | 4 | 77.0% | 95.4% | 297.5 | 0 | 177.6 |
| Predictive, demand only | 252.4 | 11.4% | 2 | 61.2% | 91.5% | 303.3 | 0 | 162.7 |
| Predictive, demand + solar | 252.5 | 11.6% | 2 | 61.0% | 90.8% | 302.7 | 0 | 163.9 |

136 complete test days, battery 86 Wh usable starting at 60% each midnight, modeled solar 120 W peak (half-sine with random clouds) until the real panel log exists. Without control, 4 of 136 days have no battery-empty minutes. Battery-empty minutes, comfort cost and switches are means per day; the warning measures are pooled over all 1,057 deficit events (from the No-control run, the same for every policy) and all shedding episodes. Early warning = a shedding episode starts 1–10 min before a deficit; shed recall also counts shedding already on at the start minute.
