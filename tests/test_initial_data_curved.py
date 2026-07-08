#!/usr/bin/env python3
"""
Datos iniciales entrantes consistentes con el fondo curvo
(direction="ingoing_curved"): reducen el transitorio espurio frente a la
relación de espacio plano (cuantificado en el piloto de Fase 0 con el
oráculo 1D; aquí se valida el cableado 3D).
"""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

try:
    from mpi4py import MPI

    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False

pytestmark = [
    pytest.mark.requires_numpy,
    pytest.mark.requires_dolfinx,
    pytest.mark.skipif(not HAS_DOLFINX, reason="DOLFINx not available"),
]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(scope="module")
def box_mesh():
    from dolfinx.mesh import create_box

    return create_box(MPI.COMM_WORLD, [[-12, -12, -12], [12, 12, 12]], [10, 10, 10])


def test_curved_requires_background(box_mesh):
    from rsd.physics.initial_conditions import GaussianBump

    with pytest.raises(ValueError, match="background"):
        GaussianBump(box_mesh, A=1e-3, r0=8.0, w=2.0, v0=0.0,
                     direction="ingoing_curved")


def test_curved_reduces_to_flat_on_flat_background(box_mesh):
    from rsd.physics.initial_conditions import GaussianBump
    from rsd.physics.metrics import FlatBackgroundCoeffs

    flat = GaussianBump(box_mesh, A=1e-3, r0=8.0, w=2.0, v0=0.0,
                        direction="ingoing")
    curved = GaussianBump(box_mesh, A=1e-3, r0=8.0, w=2.0, v0=0.0,
                          direction="ingoing_curved",
                          background=FlatBackgroundCoeffs())
    d = np.max(np.abs(flat.get_momentum().x.array - curved.get_momentum().x.array))
    assert d < 1e-14, f"en fondo plano ambas relaciones deben coincidir (Δ={d:.2e})"


def test_curved_momentum_differs_on_kerr_schild(box_mesh):
    from rsd.physics.initial_conditions import GaussianBump
    from rsd.physics.metrics import KerrSchildCoeffs

    flat = GaussianBump(box_mesh, A=1e-3, r0=8.0, w=2.0, v0=0.0,
                        direction="ingoing")
    curved = GaussianBump(box_mesh, A=1e-3, r0=8.0, w=2.0, v0=0.0,
                          direction="ingoing_curved",
                          background=KerrSchildCoeffs(M=1.0, a=0.0))
    rel = (
        np.max(np.abs(flat.get_momentum().x.array - curved.get_momentum().x.array))
        / np.max(np.abs(flat.get_momentum().x.array))
    )
    assert rel > 0.05, "sobre Kerr-Schild la relación curva debe diferir de la plana"


def test_radial_factors_consistency():
    """Los factores radiales reproducen la identidad c_in = 1 de Kerr-Schild."""
    from rsd.physics.metrics import KerrSchildCoeffs

    r = np.linspace(0.3, 30.0, 200)
    alpha, beta_r, sqrt_grr = KerrSchildCoeffs(M=1.0, a=0.0).radial_factors_np(r)
    c_in = beta_r + alpha * sqrt_grr
    assert np.allclose(c_in, 1.0, atol=1e-12), "rayo nulo entrante KS: dr/dt = -1"


def test_config_accepts_new_direction():
    from rsd.config import DEFAULT_CONFIG, validate_config

    cfg = {**DEFAULT_CONFIG}
    cfg["initial_conditions"] = {
        **DEFAULT_CONFIG["initial_conditions"], "direction": "ingoing_curved"
    }
    validate_config(cfg)
    cfg["initial_conditions"]["direction"] = "sideways"
    with pytest.raises(ValueError):
        validate_config(cfg)
