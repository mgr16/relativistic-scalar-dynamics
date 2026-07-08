"""
Ventana admisible de excisión esférica en Kerr-Schild
(docs/math/excision_window.md): la esfera cartesiana debe caber en la
región atrapada r₋ < r < r₊ para que el borde "do-nothing" sea consistente.
"""

import numpy as np
import pytest

from rsd.config import validate_config
from rsd.physics.metrics import kerr_excision_window

pytestmark = [pytest.mark.requires_numpy]


def test_window_values_match_table():
    # casos de la tabla de docs/math/excision_window.md (M=1)
    for a, lo_ref, hi_ref in [
        (0.0, 0.0, 2.0),
        (0.5, 0.51764, 1.86603),
        (0.9, 1.06218, 1.43589),
    ]:
        lo, hi = kerr_excision_window(1.0, a)
        assert lo == pytest.approx(lo_ref, abs=2e-5)
        assert hi == pytest.approx(hi_ref, abs=2e-5)


def test_window_closes_at_high_spin():
    lo, hi = kerr_excision_window(1.0, 0.98)
    assert lo >= hi, "por encima de a≈0.9718 la ventana debe cerrarse"
    # justo debajo del spin crítico la ventana sigue abierta
    lo, hi = kerr_excision_window(1.0, 0.97)
    assert lo < hi


def test_window_scales_with_mass():
    lo1, hi1 = kerr_excision_window(1.0, 0.6)
    lo2, hi2 = kerr_excision_window(2.0, 1.2)
    assert lo2 == pytest.approx(2.0 * lo1, rel=1e-12)
    assert hi2 == pytest.approx(2.0 * hi1, rel=1e-12)


def test_naked_singularity_rejected():
    with pytest.raises(ValueError):
        kerr_excision_window(1.0, 1.1)


def _kerr_cfg(a: float, r_inner: float) -> dict:
    return {
        "mesh": {"type": "gmsh", "R": 20.0, "lc": 1.5, "r_inner": r_inner},
        "metric": {"type": "kerr", "M": 1.0, "a": a},
        "solver": {"degree": 1, "cfl": 0.3, "potential_type": "zero"},
        "initial_conditions": {"type": "gaussian", "A": 1e-3, "r0": 8.0, "w": 2.0, "v0": 0.0},
        "evolution": {"t_end": 10.0, "output_every": 10},
        "output": {"dir": "results"},
    }


def test_validate_config_enforces_window():
    # dentro de la ventana para a=0.9: (1.062, 1.436)
    validate_config(_kerr_cfg(0.9, 1.25))
    # el valor histórico 1.0 queda fuera (casquete ecuatorial bajo r₋)
    with pytest.raises(ValueError, match="excision\\s+window"):
        validate_config(_kerr_cfg(0.9, 1.0))
    # por encima del horizonte tampoco
    with pytest.raises(ValueError, match="excision\\s+window"):
        validate_config(_kerr_cfg(0.9, 1.5))
    # spin supercrítico: ninguna esfera es válida
    with pytest.raises(ValueError, match="spheroidal"):
        validate_config(_kerr_cfg(0.99, 1.2))


def test_validate_config_schwarzschild_ks_unaffected():
    # a=0: ventana (0, 2M), el uso histórico r_inner=1.0 sigue válido
    validate_config(_kerr_cfg(0.0, 1.0))
    with pytest.raises(ValueError):
        validate_config(_kerr_cfg(0.0, 2.5))


def test_ringdown_default_picks_window_midpoint():
    lo, hi = kerr_excision_window(1.0, 0.9)
    assert 0.5 * (lo + hi) == pytest.approx(1.249, abs=1e-3)
    assert np.isfinite(lo) and np.isfinite(hi)
