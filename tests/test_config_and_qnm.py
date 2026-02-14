import json
from pathlib import Path

import numpy as np
import pytest

from psyop.config import load_config, validate_config
from psyop.analysis.qnm import compute_qnm, estimate_qnm_prony_modes


def test_config_validation_and_json_loading(tmp_path: Path):
    cfg = {
        "mesh": {"R": 10.0, "lc": 1.0},
        "metric": {"type": "flat"},
        "solver": {"cfl": 0.2, "degree": 1, "bc_type": "characteristic"},
        "initial_conditions": {"type": "gaussian"},
        "evolution": {"t_end": 1.0},
        "output": {"dir": "out"},
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    loaded = load_config(str(p))
    validated = validate_config(loaded)
    assert validated["solver"]["bc_type"] == "characteristic"


def test_qnm_detrend_and_extended_prony():
    t = np.linspace(0, 10, 500)
    signal = np.cos(2 * np.pi * 1.2 * t) * np.exp(-0.1 * t) + 0.01
    freqs, spec = compute_qnm(signal, t[1] - t[0], window="tukey", detrend=True)
    assert freqs.shape == spec.shape
    modes = estimate_qnm_prony_modes(signal, t[1] - t[0], modes=1)
    assert modes and "frequency" in modes[0] and "score" in modes[0]


def test_potential_numpy_api_aliases():
    potential_module = pytest.importorskip("psyop.physics.potential")
    HiggsPotential = potential_module.HiggsPotential
    pot = HiggsPotential(m_squared=1.0, lambda_coupling=0.1)
    x = np.array([0.0, 1.0])
    np.testing.assert_allclose(pot.evaluate_np(x), pot.evaluate_numpy(x))
    np.testing.assert_allclose(pot.derivative_np(x), pot.derivative_numpy(x))
