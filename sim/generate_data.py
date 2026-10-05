"""Simulate 4 weeks of per-minute power readings for household appliances.

Output: data/raw/simulated_power.csv with columns
    timestamp, appliance, watts, appliance_state

Appliances:
    refrigerator    - always on at 120-180 W, with an hourly compressor kick
                      to ~200 W lasting 10-15 minutes
    water_heater    - ~3 kW bursts, mostly morning and evening showers
    electric_fan    - daytime usage driven by a per-day "warmth" factor
    washing_machine - 1-2 cycles/day of 45-90 min at 400-500 W between
                      07:00 and 20:00; exactly 0 W when off

Standard library only. Run from anywhere:
    python sim/generate_data.py
"""

import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 42
START = datetime(2026, 9, 7)  # a Monday
DAYS = 28
MINUTES_PER_DAY = 24 * 60
TOTAL_MINUTES = DAYS * MINUTES_PER_DAY

APPLIANCES = ["refrigerator", "water_heater", "electric_fan", "washing_machine"]

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "raw" / "simulated_power.csv"


def minute_of(day, hour, minute=0):
    return day * MINUTES_PER_DAY + hour * 60 + minute


def sim_refrigerator(rng):
    """Continuous 120-180 W load with an hourly ~200 W compressor kick.

    Baseline wanders on a ~40 min cycle with noise and door-open bumps,
    clamped to 120-180 W. Once per hour the compressor kicks in and holds
    ~200 W for 10-15 minutes before dropping back to the baseline.
    """
    watts = [0.0] * TOTAL_MINUTES
    state = [1] * TOTAL_MINUTES
    phase = rng.uniform(0, 2 * math.pi)
    period = 40.0
    door_boost = [0.0] * TOTAL_MINUTES
    kick = [0.0] * TOTAL_MINUTES

    for hour_idx in range(DAYS * 24):
        start = hour_idx * 60 + rng.randint(0, 45)
        level = rng.uniform(195, 205)
        for m in range(start, min(start + rng.randint(10, 15), TOTAL_MINUTES)):
            kick[m] = level

    # Door openings cluster around breakfast, lunch, dinner, late snack.
    for day in range(DAYS):
        for center_h, prob in [(7, 0.9), (12, 0.7), (19, 0.95), (22, 0.4)]:
            if rng.random() < prob:
                start = minute_of(day, center_h) + int(rng.gauss(0, 30))
                length = rng.randint(5, 20)
                bump = rng.uniform(10, 25)
                for m in range(max(start, 0), min(start + length, TOTAL_MINUTES)):
                    door_boost[m] = max(door_boost[m], bump)

    for m in range(TOTAL_MINUTES):
        # Period drifts slightly so cycles are not perfectly regular.
        period += rng.gauss(0, 0.05)
        period = min(max(period, 34.0), 46.0)
        phase += 2 * math.pi / period
        hour = (m % MINUTES_PER_DAY) / 60
        # Slightly harder work in the warm afternoon.
        ambient = 4 * math.sin(2 * math.pi * (hour - 9) / 24)
        if kick[m]:
            watts[m] = kick[m] + rng.gauss(0, 2.0)
        else:
            w = 150 + 22 * math.sin(phase) + ambient + door_boost[m] + rng.gauss(0, 3.0)
            watts[m] = min(max(w, 120.0), 180.0)
    return watts, state


def sim_water_heater(rng):
    """~3 kW element bursts at shower times plus occasional short reheats."""
    watts = [0.0] * TOTAL_MINUTES
    state = [0] * TOTAL_MINUTES
    rated = rng.uniform(2900, 3100)

    for day in range(DAYS):
        weekend = (START + timedelta(days=day)).weekday() >= 5
        events = []
        # Morning shower: later and slightly less certain on weekends.
        if rng.random() < (0.75 if weekend else 0.92):
            hour = rng.uniform(7.5, 10.0) if weekend else rng.uniform(5.75, 7.75)
            events.append((hour, rng.randint(15, 35)))
        # Evening shower.
        if rng.random() < 0.8:
            events.append((rng.uniform(18.5, 22.0), rng.randint(15, 30)))
        # Midday dishwashing / standby-loss reheat.
        if rng.random() < 0.35:
            events.append((rng.uniform(11.5, 15.0), rng.randint(4, 10)))

        for hour, length in events:
            start = minute_of(day, int(hour), int((hour % 1) * 60))
            for m in range(start, min(start + length, TOTAL_MINUTES)):
                watts[m] = rated + rng.gauss(0, 25)
                state[m] = 1
    return watts, state


def sim_electric_fan(rng):
    """On/off Markov chain whose turn-on rate follows a daytime profile."""
    watts = [0.0] * TOTAL_MINUTES
    state = [0] * TOTAL_MINUTES
    speeds = {"low": 38.0, "medium": 52.0, "high": 68.0}

    def daytime_weight(hour):
        # Ramps up mid-morning, peaks ~14:00-15:00, fades late evening.
        if hour < 8 or hour >= 23:
            return 0.02
        return max(0.0, math.sin(math.pi * (hour - 8) / 15)) ** 1.5

    on = False
    speed = "medium"
    warmth = 1.0
    for m in range(TOTAL_MINUTES):
        if m % MINUTES_PER_DAY == 0:
            warmth = rng.uniform(0.5, 1.3)  # day-to-day weather variation
        hour = (m % MINUTES_PER_DAY) / 60
        w = daytime_weight(hour) * warmth
        if on:
            # Sessions last longer during the warm part of the day.
            p_off = 0.04 - 0.035 * min(w, 1.0)
            if rng.random() < p_off:
                on = False
            elif rng.random() < 0.003:
                speed = rng.choice(list(speeds))
        else:
            if rng.random() < 0.02 * w:
                on = True
                speed = "high" if w > 0.9 else rng.choice(["low", "medium", "medium"])
        if on:
            watts[m] = speeds[speed] + rng.gauss(0, 1.2)
            state[m] = 1
    return watts, state


WASH_WINDOW = (7 * 60, 20 * 60)  # cycles run entirely within 07:00-20:00
WASH_MIN_GAP = 30  # minutes between back-to-back loads


def sim_washing_machine(rng):
    """1-2 cycles per day of 45-90 min at 400-500 W; exactly 0 W when off."""
    watts = [0.0] * TOTAL_MINUTES
    state = [0] * TOTAL_MINUTES
    win_start, win_end = WASH_WINDOW

    for day in range(DAYS):
        durations = [rng.randint(45, 90) for _ in range(rng.randint(1, 2))]
        earliest = win_start
        for i, length in enumerate(durations):
            # Leave room in the window for any loads still to come.
            needed_after = sum(d + WASH_MIN_GAP for d in durations[i + 1:])
            start = rng.randint(earliest, win_end - length - needed_after)
            level = rng.uniform(430, 470)
            base = day * MINUTES_PER_DAY + start
            for m in range(base, base + length):
                watts[m] = min(max(level + rng.gauss(0, 15), 400.0), 500.0)
                state[m] = 1
            earliest = start + length + WASH_MIN_GAP
    return watts, state


def main():
    rng = random.Random(SEED)
    sims = {
        "refrigerator": sim_refrigerator(rng),
        "water_heater": sim_water_heater(rng),
        "electric_fan": sim_electric_fan(rng),
        "washing_machine": sim_washing_machine(rng),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestamp", "appliance", "watts", "appliance_state"]
    preview = []
    rows_written = 0
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for m in range(TOTAL_MINUTES):
            ts = (START + timedelta(minutes=m)).strftime("%Y-%m-%d %H:%M:%S")
            for name in APPLIANCES:
                watts, state = sims[name]
                row = [ts, name, f"{watts[m]:.1f}", state[m]]
                writer.writerow(row)
                rows_written += 1
                if len(preview) < 10:
                    preview.append(row)

    print(f"Wrote {OUT_PATH.relative_to(ROOT)}\n")
    print(f"{'timestamp':<20} {'appliance':<16} {'watts':>8} {'appliance_state':>16}")
    for ts, name, w, s in preview:
        print(f"{ts:<20} {name:<16} {w:>8} {s:>16}")
    print(f"\nTotal rows: {rows_written}")


if __name__ == "__main__":
    main()
