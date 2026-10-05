"""Load-shedding decision: pick an appliance to turn off when demand is forecast high.

Run from anywhere:
    python decision/shed.py
"""

THRESHOLD_WATTS = 1500  # forecast demand above this triggers shedding

# Shed order: first ON appliance in this list is turned off.
NON_CRITICAL = ["water_heater", "washing_machine", "electric_fan"]
CRITICAL = ["refrigerator"]  # never shed


def should_shed(forecast_watts, current_states):
    """Return the appliance to shed, or None.

    forecast_watts: predicted total demand in watts.
    current_states: {appliance_name: 0 or 1}; missing appliances count as OFF.
    """
    if forecast_watts > THRESHOLD_WATTS:
        for appliance in NON_CRITICAL:
            if current_states.get(appliance, 0) == 1:
                return appliance
    return None


if __name__ == "__main__":
    all_on = {name: 1 for name in CRITICAL + NON_CRITICAL}
    fridge_only = {name: int(name in CRITICAL) for name in CRITICAL + NON_CRITICAL}

    cases = [
        ("Case 1: 1600 W, all ON", 1600, all_on, "water_heater"),
        ("Case 2: 1200 W, all ON", 1200, all_on, None),
        ("Case 3: 1600 W, only refrigerator ON", 1600, fridge_only, None),
    ]
    for label, forecast, states, expected in cases:
        result = should_shed(forecast, states)
        status = "PASS" if result == expected else "FAIL"
        print(f"{label:<40} -> {result!r:<16} (expected {expected!r}) {status}")
