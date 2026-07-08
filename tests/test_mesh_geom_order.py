#!/usr/bin/env python3
"""
Mallas con geometría curva de 2.º orden (mesh.geom_order = 2): el error
de volumen de la bola debe caer drásticamente frente a facetas planas,
que es exactamente el error geométrico O(h²) que domina con P2+.
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
    pytest.mark.requires_mesh,
    pytest.mark.skipif(not HAS_DOLFINX, reason="DOLFINx not available"),
]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

R = 5.0
LC = 1.5


def _ball_volume(geom_order: int) -> float:
    import dolfinx.fem as fem
    import ufl

    from rsd.mesh.gmsh import build_ball_mesh

    mesh, _, _ = build_ball_mesh(R=R, lc=LC, comm=MPI.COMM_WORLD,
                                 geom_order=geom_order)
    vol = fem.assemble_scalar(
        fem.form(1.0 * ufl.dx(domain=mesh, metadata={"quadrature_degree": 4}))
    )
    return float(mesh.comm.allreduce(vol))


def test_curved_cells_reduce_geometric_error():
    v_exact = 4.0 / 3.0 * np.pi * R**3
    err1 = abs(_ball_volume(1) - v_exact) / v_exact
    err2 = abs(_ball_volume(2) - v_exact) / v_exact
    # facetas planas subestiman el volumen a O(h²); las celdas curvas deben
    # mejorar el error geométrico en al menos un orden de magnitud
    assert err2 < 0.1 * err1, (
        f"geom_order=2 debería reducir el error de volumen ≥10x: "
        f"err1={err1:.3e}, err2={err2:.3e}"
    )


def test_invalid_geom_order_rejected():
    from rsd.mesh.gmsh import build_ball_mesh

    with pytest.raises(ValueError):
        build_ball_mesh(R=R, lc=LC, comm=MPI.COMM_WORLD, geom_order=3)


def test_config_validates_geom_order():
    from rsd.config import DEFAULT_CONFIG, validate_config

    cfg = {**DEFAULT_CONFIG}
    cfg["mesh"] = {**DEFAULT_CONFIG["mesh"], "geom_order": 2}
    validate_config(cfg)
    cfg["mesh"]["geom_order"] = 4
    with pytest.raises(ValueError):
        validate_config(cfg)
