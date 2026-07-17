"""Artifact and reconstruction checks for the R2 sensitivity grid."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts/oracle_sensitivity_scan.py"
SPEC = importlib.util.spec_from_file_location("oracle_sensitivity_scan", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sensitivity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sensitivity
SPEC.loader.exec_module(sensitivity)


def test_versioned_grid_is_complete_and_self_consistent():
    arrays = sensitivity._load_npz(sensitivity.DEFAULT_DATA)
    payload = sensitivity.build_payload(arrays, sensitivity.DEFAULT_DATA)
    checked = json.loads(sensitivity.DEFAULT_JSON.read_text(encoding="utf-8"))

    assert payload == checked
    assert checked["protocol"]["lambdas"] == list(sensitivity.LAMBDAS)
    assert checked["protocol"]["amplitudes"] == list(sensitivity.AMPLITUDES)
    assert checked["summary"]["cell_count"] == (
        len(sensitivity.LAMBDAS) * len(sensitivity.AMPLITUDES)
    )
    assert len({cell["id"] for cell in checked["cells"]}) == 12
    assert checked["summary"]["discovery_threshold_exceeded"] is True
    assert checked["summary"]["max_abs_dev_from_unity"] > 0.15
    assert checked["data"]["path"].startswith("docs/research/phase3/data/")


def test_audit_cell_reconstructs_imported_production_discriminator():
    checked = json.loads(sensitivity.DEFAULT_JSON.read_text(encoding="utf-8"))
    audit = checked["audit_spot_check"]

    def curve(name: str) -> dict[str, np.ndarray]:
        item = audit[name]
        return {
            "t": np.asarray(item["t"], dtype=float),
            "a_primary": np.asarray(item["a_primary"], dtype=float),
            "strong_mask": np.asarray(item["strong_mask"], dtype=bool),
        }

    reconstructed = sensitivity.raw_discriminator(curve("linear"), curve("mexhat"))
    assert reconstructed["l2_ratio"] == pytest.approx(
        audit["reported_D_oracle"], rel=0.0, abs=1e-15
    )
    cell = next(cell for cell in checked["cells"] if cell["id"] == audit["cell_id"])
    assert reconstructed["l2_ratio"] == pytest.approx(
        cell["D_oracle"], rel=0.0, abs=1e-15
    )
    assert reconstructed["n_samples"] == cell["support"]["n_samples"]


def test_check_mode_is_read_only_and_current():
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (sensitivity.DEFAULT_DATA, sensitivity.DEFAULT_JSON)
    }
    assert sensitivity.main(["--check"]) == 0
    after = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (sensitivity.DEFAULT_DATA, sensitivity.DEFAULT_JSON)
    }
    assert after == before
