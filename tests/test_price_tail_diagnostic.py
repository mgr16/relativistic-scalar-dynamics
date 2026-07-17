"""Checks for the versioned R2.4 Price-tail failure diagnosis."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts/price_tail_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("price_tail_diagnostic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnostic
SPEC.loader.exec_module(diagnostic)


def test_versioned_diagnosis_reconstructs_and_measures_a_floor():
    arrays = diagnostic._load(diagnostic.DEFAULT_DATA)
    payload = diagnostic.build_payload(arrays, diagnostic.DEFAULT_DATA)
    checked = json.loads(diagnostic.DEFAULT_JSON.read_text(encoding="utf-8"))
    assert payload == checked

    fit = checked["original_test_fit"]
    assert fit["exponent"] == pytest.approx(-1.983424032768612)
    assert fit["r2"] < 0.1
    assert max(row["r2"] for row in checked["window_sweep"]) < 0.4
    rms = [row["rms"] for row in checked["rms_bins"]]
    assert rms[-1] > 2.0 * min(rms)
    floor = checked["amplitude_floor"]
    assert floor["last_over_first_rms_bin"] > (
        5.0 * floor["price_prediction_last_over_first"]
    )
    assert checked["late_floor_lines"]["dw1"] >= 0.05


def test_check_mode_is_read_only_and_current():
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (diagnostic.DEFAULT_DATA, diagnostic.DEFAULT_JSON)
    }
    assert diagnostic.main(["--check"]) == 0
    after = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (diagnostic.DEFAULT_DATA, diagnostic.DEFAULT_JSON)
    }
    assert after == before
