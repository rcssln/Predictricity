"""Tests for models/early_warning.py on hand-made series where the answers are known.

Run from the project root:
    python -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.early_warning import episode_starts, event_starts, measures  # noqa: E402

T = 1000.0
LOW, HIGH = 500.0, 2000.0


def series(n, high=(), missing=()):
    """n minutes at LOW, with HIGH at the given minutes and NaN at the missing ones."""
    s = pd.Series(LOW, index=pd.date_range("2010-01-01", periods=n, freq="min"))
    s.iloc[list(high)] = HIGH
    s.iloc[list(missing)] = np.nan
    return s


def starts(flags):
    return list(np.flatnonzero(flags.to_numpy()))


def test_event_needs_ten_quiet_minutes_before():
    # 20: first spike after 20 quiet minutes -> event. 25: only 4 quiet minutes -> not.
    # 40: 14 quiet minutes (26-39) -> event. 45-47: only 4 quiet minutes (41-44) -> not.
    load = series(60, high=[20, 21, 25, 40, 45, 46, 47])
    assert starts(event_starts(load, T)) == [20, 40]


def test_event_needs_strictly_above_T():
    load = series(30)
    load.iloc[15] = T                     # equal to T is not above it
    assert starts(event_starts(load, T)) == []
    load.iloc[15] = T + 1
    assert starts(event_starts(load, T)) == [15]


def test_gap_never_creates_an_event_start():
    # Minutes 10-14 are missing; 15 is high. Its 10 previous minutes include the gap,
    # so it is not an event start, even though no known minute before it is above T.
    load = series(40, high=[15, 35], missing=range(10, 15))
    assert starts(event_starts(load, T)) == [35]
    # A high minute inside the data right after a gap of only quiet minutes is fine
    # once 10 present quiet minutes have passed again.
    load = series(40, high=[26], missing=range(10, 15))
    assert starts(event_starts(load, T)) == [26]


def test_episode_starts():
    alert = pd.Series([False, True, True, False, True, False, False, True])
    assert starts(episode_starts(alert)) == [1, 4, 7]


def flags(n, at):
    s = pd.Series(False, index=pd.date_range("2010-01-01", periods=n, freq="min"))
    s.iloc[list(at)] = True
    return s


def test_measures_known_answer():
    n = 100
    events = flags(n, [30, 60, 90])
    scored = flags(n, range(n))
    # Event 30: alert run 24-30 (warned 6 min ahead). Event 60: alert only at 60
    # (detected, not warned). Event 90: no alert. False alarm run 10-11.
    alert = flags(n, [10, 11, *range(24, 31), 60])
    m = measures(alert, events, scored)
    assert m["n_events"] == 3
    assert m["n_alert_episodes"] == 3                  # starts at 10, 24, 60
    assert np.isclose(m["early_warning_rate"], 1 / 3)  # only event 30
    assert np.isclose(m["recall"], 1 / 3)              # recall = early warning
    assert np.isclose(m["detection_recall"], 2 / 3)    # events 30 and 60
    assert np.isclose(m["precision"], 2 / 3)           # episodes at 24 and 60
    assert m["median_lead_min"] == 6.0                 # only event 30 is warned, 6 min ahead


def test_reactive_rule_never_warns_early():
    load = series(60, high=[20, 21, 40])
    events = event_starts(load, T)
    m = measures(load > T, events, flags(60, range(60)))
    assert m["early_warning_rate"] == 0
    assert m["median_lead_min"] == 0
    assert m["recall"] == 0
    assert m["detection_recall"] == 1
    assert m["precision"] == 1                            # every crossing was an event


def test_events_without_full_forecast_window_are_dropped():
    n = 50
    events = flags(n, [20, 40])
    scored = flags(n, [i for i in range(n) if i != 35])   # minute 35 has no forecast
    m = measures(flags(n, []), events, scored)
    assert m["n_events"] == 1                             # 40 needs 30-40 scored
