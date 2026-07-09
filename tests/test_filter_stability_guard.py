#!/usr/bin/env python3
"""
test_filter_stability_guard.py

La cota de estabilidad del filtro espectral (ε·dt·λmax < 2,
docs/math/dissipation.md) depende de la malla: λmax(M⁻¹K) ~ 1/h_min².
Cruzarla no "disipa más": amplifica exponencialmente (un barrido real con
ε=0.05 divergió a ~1e148 sin ningún aviso). El solver debe rechazarla con
RuntimeError informativo — que reporta el ε_max de la malla — en vez de
producir basura en silencio, para ambos órdenes del filtro.
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

R = 10.0
R_INNER = 1.0
LC = 2.5


@pytest.fixture(scope="module")
def shell_mesh():
    from rsd.mesh.gmsh import build_ball_mesh

    comm = MPI.COMM_WORLD
    mesh, _, facet_tags = build_ball_mesh(R=R, lc=LC, comm=comm, r_inner=R_INNER)
    if facet_tags is None:
        pytest.skip("mesh without facet tags (gmsh unavailable)")
    return mesh, facet_tags


def _make_solver(mesh, facet_tags, eps, order):
    from rsd.physics.metrics import KerrSchildCoeffs
    from rsd.solvers.first_order import FirstOrderKGSolver

    solver = FirstOrderKGSolver(
        mesh=mesh, domain_radius=R, degree=1, potential_type="zero",
        cfl_factor=0.3, filter_strength=eps, filter_order=order,
    )
    bg = KerrSchildCoeffs(M=1.0, a=0.0).build(mesh)
    solver.set_background(*bg, rebuild=False)
    solver.enable_sommerfeld(facet_tags, outer_tag=2, rebuild=False)
    solver.enable_excision(facet_tags, inner_tag=3, rebuild=False)
    solver.rebuild_operators()
    return solver


@pytest.mark.parametrize("order", [2, 4])
def test_guard_rejects_unstable_eps(shell_mesh, order):
    """ε absurdo ⇒ RuntimeError con el ε_max de la malla, no basura 1e148."""
    mesh, facet_tags = shell_mesh
    solver = _make_solver(mesh, facet_tags, eps=50.0, order=order)
    dt = solver.compute_adaptive_dt()
    with pytest.raises(RuntimeError, match="filter_strength <"):
        solver.ssp_rk3_step(dt)


def test_guard_allows_stable_eps(shell_mesh):
    """ε pequeño pasa el guard y evoluciona finito (ambos órdenes cubiertos
    por la misma cota; basta orden 2 aquí)."""
    mesh, facet_tags = shell_mesh
    solver = _make_solver(mesh, facet_tags, eps=1e-4, order=2)
    dt = solver.compute_adaptive_dt()
    for _ in range(3):
        solver.ssp_rk3_step(dt)
    assert np.all(np.isfinite(solver.u.x.array))


def test_lambda_max_estimated_for_order_2(shell_mesh):
    """λmax debe existir también en orden 2 (antes sólo se estimaba para el
    orden 4): es lo que hace verificable la cota."""
    mesh, facet_tags = shell_mesh
    solver = _make_solver(mesh, facet_tags, eps=1e-4, order=2)
    lam = solver._filter_lambda_max
    assert np.isfinite(lam) and lam > 0.0
