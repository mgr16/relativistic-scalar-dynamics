"""Unit and artifact checks for the R1 kinetic-domination measurement."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts/oracle_energy_split.py"
SPEC = importlib.util.spec_from_file_location("oracle_energy_split", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
energy_split = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = energy_split
SPEC.loader.exec_module(energy_split)


def test_contiguous_crossover_rejects_disconnected_outer_island():
    radii = np.asarray([0.01, 0.02, 0.04, 0.08, 0.16])
    ratio = np.asarray([1e-4, 1e-3, 2e-2, 1e-3, 1e-4])
    assert energy_split._contiguous_crossover(radii, ratio, 1e-2) == 0.02


def test_log_slope_recovers_power_law():
    radii = np.geomspace(0.01, 1.0, 100)
    ratio = 0.2 * radii**2.8
    got = energy_split._log_slope(radii, ratio, (0.05, 0.5))
    assert got == pytest.approx(2.8, abs=1e-12)


def test_versioned_outputs_are_self_consistent():
    assert energy_split.DEFAULT_DATA.is_file()
    assert energy_split.DEFAULT_JSON.is_file()
    arrays = energy_split._load_npz(energy_split.DEFAULT_DATA)
    payload = energy_split.build_payload(arrays, energy_split.DEFAULT_DATA)
    checked = json.loads(energy_split.DEFAULT_JSON.read_text(encoding="utf-8"))

    assert payload == checked
    assert checked["summary"]["strong_snapshot_count"] > 0
    assert np.all(arrays["kinetic_energy_density"] >= 0.0)
    assert np.all(arrays["potential_energy_density"] >= 0.0)
    assert checked["data"]["path"].startswith("docs/research/phase3/data/")
