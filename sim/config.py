"""Shared helpers for the Step 12 simulation: settings and paths."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "sim" / "config.yaml"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"


def load_config(scenario=None, path=CONFIG_PATH):
    """Settings from config.yaml; with a scenario name, its overrides applied on top."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if scenario is None:
        return cfg
    if scenario not in cfg["scenarios"]:
        raise KeyError(f"Unknown scenario {scenario!r}; choose from {list(cfg['scenarios'])}")
    return {**cfg, **(cfg["scenarios"][scenario] or {}), "scenario": scenario}
