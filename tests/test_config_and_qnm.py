import json
from pathlib import Path

import pytest
np = pytest.importorskip("numpy")

from rsd.config import load_config, validate_config
from rsd.analysis.qnm import compute_qnm, estimate_qnm_prony, estimate_qnm_prony_modes

pytestmark = pytest.mark.requires_numpy


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


def test_config_legacy_aliases_are_normalized(tmp_path: Path):
    cfg = {
        "mesh": {"mesh_type": "gmsh", "R": 10.0, "lc": 1.0},
        "metric": {"type": "flat"},
        "solver": {"cfl": 0.2, "degree": 1, "sommerfeld": False},
        "initial_conditions": {"type": "gaussian"},
        "evolution": {"t_end": 1.0, "output_every": 1},
        "output": {"results_dir": "out"},
    }
    p = tmp_path / "legacy_cfg.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    validated = validate_config(load_config(str(p)))

    assert validated["mesh"]["type"] == "gmsh"
    assert validated["solver"]["enable_sommerfeld"] is False
    assert validated["output"]["dir"] == "out"


def test_black_hole_metrics_require_excision():
    cfg = {
        "mesh": {"R": 10.0, "lc": 1.0},
        "metric": {"type": "schwarzschild", "M": 1.0},
        "solver": {"cfl": 0.2, "degree": 1},
        "initial_conditions": {"type": "gaussian"},
        "evolution": {"t_end": 1.0},
        "output": {"dir": "out"},
    }
    with pytest.raises(ValueError, match="r_inner"):
        validate_config(cfg)

    cfg["mesh"]["r_inner"] = 0.5
    validated = validate_config(cfg)
    assert validated["mesh"]["r_inner"] == 0.5


def test_kerr_spin_bound_is_validated():
    cfg = {
        "mesh": {"R": 10.0, "lc": 1.0, "r_inner": 1.0},
        "metric": {"type": "kerr", "M": 1.0, "a": 1.5},
        "solver": {"cfl": 0.2, "degree": 1},
        "initial_conditions": {"type": "gaussian"},
        "evolution": {"t_end": 1.0},
        "output": {"dir": "out"},
    }
    with pytest.raises(ValueError, match="naked"):
        validate_config(cfg)


def test_intractable_mesh_resolution_is_rejected():
    cfg = {
        "mesh": {"R": 15.0, "lc": 0.001},
        "metric": {"type": "flat"},
        "solver": {"cfl": 0.2, "degree": 1},
        "initial_conditions": {"type": "gaussian"},
        "evolution": {"t_end": 1.0},
        "output": {"dir": "out"},
    }
    with pytest.raises(ValueError, match="intractable"):
        validate_config(cfg)


def test_qnm_detrend_and_extended_prony():
    t = np.linspace(0, 10, 500)
    signal = np.cos(2 * np.pi * 1.2 * t) * np.exp(-0.1 * t) + 0.01
    freqs, spec = compute_qnm(signal, t[1] - t[0], window="tukey", detrend=True)
    assert freqs.shape == spec.shape
    modes = estimate_qnm_prony_modes(signal, t[1] - t[0], modes=1)
    assert modes and "frequency" in modes[0] and "score" in modes[0]
    assert "stability" in modes[0]


def test_prony_recovers_frequency_decay_and_amplitude():
    dt = 0.02
    t = np.arange(0, 10, dt)
    f0, tau, amp = 1.2, 0.15, 2.0
    signal = amp * np.cos(2 * np.pi * f0 * t) * np.exp(-tau * t)
    modes = estimate_qnm_prony_modes(signal, dt, modes=1)
    assert modes
    dominant = modes[0]
    assert abs(dominant["frequency"]) == pytest.approx(f0, rel=0.05)
    assert dominant["decay"] == pytest.approx(tau, rel=0.10)
    # La amplitud ya no es un placeholder: debe aproximar amp/2 (un coseno
    # real se reparte en dos exponenciales complejas conjugadas)
    assert dominant["amplitude"] == pytest.approx(amp / 2.0, rel=0.15)


def test_prony_orders_modes_by_amplitude():
    dt = 0.02
    t = np.arange(0, 10, dt)
    strong = 3.0 * np.cos(2 * np.pi * 0.8 * t) * np.exp(-0.1 * t)
    weak = 0.3 * np.cos(2 * np.pi * 2.0 * t) * np.exp(-0.1 * t)
    # estimate_qnm_prony devuelve los modos dominantes primero (antes el
    # orden era el arbitrario de np.linalg.eigvals)
    pairs = estimate_qnm_prony(strong + weak, dt, modes=4, svd_rank=4)
    assert pairs
    assert abs(pairs[0][0]) == pytest.approx(0.8, rel=0.1)


def test_potential_numpy_api_aliases():
    potential_module = pytest.importorskip("rsd.physics.potential")
    HiggsPotential = potential_module.HiggsPotential
    pot = HiggsPotential(m_squared=1.0, lambda_coupling=0.1)
    x = np.array([0.0, 1.0])
    np.testing.assert_allclose(pot.evaluate_np(x), pot.evaluate_numpy(x))
    np.testing.assert_allclose(pot.derivative_np(x), pot.derivative_numpy(x))
