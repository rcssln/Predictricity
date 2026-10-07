"""Tests for the Step 12 simulation (sim/run_policies.py) on a tiny hand-made 2-day input.

Run from the project root:
    python -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.config import load_config  # noqa: E402
from sim.run_policies import POLICIES, day_windows, simulate  # noqa: E402

SCENARIOS = list(load_config()["scenarios"])
NEED = ["zone_load_w", "fc_w", "pv_w"]


def two_days(pv_w=30.0):
    """2 x 1,440 minutes from 2010-07-01 00:00. Zones switch on and off in a fixed pattern;
    a sheddable zone is only on when critical is (as in the replay)."""
    idx = pd.date_range("2010-07-01", periods=2 * 1440, freq="min")
    rng = np.random.default_rng(0)
    on = rng.random(len(idx)) < 0.6
    data = pd.DataFrame(index=idx)
    data["critical"] = on
    data["zoneA"] = on & (rng.random(len(idx)) < 0.4)
    data["zoneB"] = on & (rng.random(len(idx)) < 0.5)
    data["zoneC"] = on & (rng.random(len(idx)) < 0.3)
    zones = load_config()["zones"]
    data["zone_load_w"] = sum(data[z] * v["watts"] for z, v in zones.items()).astype(float)
    data["fc_w"] = data["zone_load_w"].rolling(10, min_periods=1).max().shift(-9).ffill()
    data["pv_w"] = pv_w
    data["pv_fc_w"] = pv_w
    return data


def first_day(data):
    return data.iloc[:1440]


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("policy", list(POLICIES))
def test_critical_is_never_shed(scenario, policy):
    cfg = {**load_config(scenario), "battery_start_pct": 5}   # a hard day: little charge
    out = simulate(first_day(two_days(pv_w=0.0)), policy, cfg)
    assert out["critical_off"].sum() == 0


def test_day_starting_at_17_uses_the_right_minutes():
    data = two_days()
    starts = day_windows(data, "17:00", NEED)
    # Only 2010-07-01 17:00 has a full 1,440 minutes; the window from 07-02 17:00 runs
    # past the end of the data.
    assert starts == [pd.Timestamp("2010-07-01 17:00")]
    window = data.loc[starts[0]: starts[0] + pd.Timedelta("1D") - pd.Timedelta("1min")]
    assert len(window) == 1440
    assert window.index[0] == pd.Timestamp("2010-07-01 17:00")
    assert window.index[-1] == pd.Timestamp("2010-07-02 16:59")
    # Midnight days: both dates are complete.
    assert day_windows(data, "00:00", NEED) == [pd.Timestamp("2010-07-01"),
                                                pd.Timestamp("2010-07-02")]


def test_day_with_a_missing_minute_is_dropped():
    data = two_days()
    data.loc["2010-07-02 03:00", "zone_load_w"] = np.nan   # inside the 17:00 window
    assert day_windows(data, "17:00", NEED) == []
    assert day_windows(data, "00:00", NEED) == [pd.Timestamp("2010-07-01")]


@pytest.mark.parametrize("policy", list(POLICIES))
def test_battery_large_enough_for_all_demand_never_empties(policy):
    cfg = {**load_config("evening_sized"), "battery_usable_wh": 1e6, "battery_start_pct": 100}
    out = simulate(first_day(two_days(pv_w=0.0)), policy, cfg)
    assert out["empty"].sum() == 0


@pytest.mark.parametrize("battery", [
    {"battery_start_pct": 0},                                # a normal battery, at 0%
    {"battery_usable_wh": 1e-9, "battery_start_pct": 100},   # no battery (0 Wh would divide by 0)
])
@pytest.mark.parametrize("policy", list(POLICIES))
def test_empty_battery_and_no_solar_is_empty_whenever_load_is_on(policy, battery):
    cfg = {**load_config("evening_sized"), **battery}
    day = first_day(two_days(pv_w=0.0))
    out = simulate(day, policy, cfg)
    np.testing.assert_array_equal(out["empty"].astype(bool), day["zone_load_w"].to_numpy() > 0)
