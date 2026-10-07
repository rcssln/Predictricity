**Table 4.3 (preliminary). Load-shedding policies in simulation, scenario B, evening-sized (220 Wh, 104 W, 60% at midnight), primary**

| Policy | Battery-empty min/day | Early-warning rate | Median lead time (min) | Shed precision | Shed recall | Comfort cost (zone-min/day) | Critical interruptions | Switches/day |
|---|---|---|---|---|---|---|---|---|
| No control | 205.8 | 0.0% | 0 | – | 0.0% | 0.0 | 0 | 0.0 |
| Reactive | 110.1 | 0.3% | 2 | 70.6% | 87.9% | 249.6 | 0 | 86.9 |
| Predictive, demand only | 103.1 | 15.6% | 2 | 61.3% | 83.6% | 260.1 | 0 | 75.5 |
| Predictive, demand + solar | 103.3 | 15.4% | 2 | 61.1% | 83.1% | 259.5 | 0 | 75.8 |

136 complete test days, battery 220 Wh usable starting at 60% each midnight, modeled solar 104 W peak (half-sine with random clouds) until the real panel log exists. Without control, 32 of 136 days have no battery-empty minutes. Battery-empty minutes, comfort cost and switches are means per day; the warning measures are pooled over all 986 deficit events (from the No-control run, the same for every policy) and all shedding episodes. Early warning = a shedding episode starts 1–10 min before a deficit; shed recall also counts shedding already on at the start minute.
