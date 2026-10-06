"""Step 12.3: battery state of charge, one minute at a time (Initial Results Runbook)."""


def step_soc(soc_wh, pv_w, load_w, cap_wh, dt_h=1 / 60,
             charge_eff=0.85, inverter_eff=0.85):
    soc_wh += pv_w * charge_eff * dt_h          # energy in from the panel
    soc_wh -= load_w / inverter_eff * dt_h      # energy out to the loads
    return min(max(soc_wh, 0.0), cap_wh)


def can_serve(soc_wh, pv_w, load_w, dt_h=1 / 60, charge_eff=0.85, inverter_eff=0.85):
    """True if the battery plus this minute's solar covers this minute's load."""
    return soc_wh + pv_w * charge_eff * dt_h >= load_w / inverter_eff * dt_h - 1e-9


def allowance_w(soc_pct, allowance):
    """Power the controller lets the battery supply, by state of charge (config table)."""
    if soc_pct > 60:
        return allowance["above_60"]
    if soc_pct > 40:
        return allowance["above_40"]
    return allowance["else"]
