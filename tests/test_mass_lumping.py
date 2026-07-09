#!/usr/bin/env python3
"""
test_mass_lumping.py

Guardián de la opción `optimization.mass_lumping` (row-sum, sólo P1): la
resolución de masa pasa de un KSP (CG) a una escala puntual por M_L⁻¹, que
es el coste dominante tras el fast path. Verifica:

  1. M_L⁻¹ existe, es finita y estrictamente positiva (P1 ⇒ row-sum > 0),
     y no se crea ningún KSP de masa.
  2. A/B: la evolución lumped reproduce la consistente dentro de la
     perturbación O(h²) esperada (misma ecuación, masa distinta) y ambas
     son estables.
  3. degree > 1 se rechaza en el constructor (row-sum puede dar entradas
     ≤ 0 fuera de P1).
  4. El filtro espectral (iteración de potencias + aplicación) funciona
     sobre la ruta lumped.
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
N_STEPS = 10


@pytest.fixture(scope="module")
def shell_mesh():
    from rsd.mesh.gmsh import build_ball_mesh

    comm = MPI.COMM_WORLD
    mesh, _, facet_tags = build_ball_mesh(R=R, lc=LC, comm=comm, r_inner=R_INNER)
    if facet_tags is None:
        pytest.skip("mesh without facet tags (gmsh unavailable)")
    return mesh, facet_tags


def _make_solver(mesh, facet_tags, mass_lumping, **extra):
    from rsd.physics.metrics import KerrSchildCoeffs
    from rsd.solvers.first_order import FirstOrderKGSolver

    solver = FirstOrderKGSolver(
        mesh=mesh,
        domain_radius=R,
        degree=1,
        potential_type="zero",
        cfl_factor=0.3,
        cfg={"optimization": {"preassemble": True, "mass_lumping": mass_lumping}},
        **extra,
    )
    bg = KerrSchildCoeffs(M=1.0, a=0.0).build(mesh)
    solver.set_background(*bg, rebuild=False)
    solver.enable_sommerfeld(facet_tags, outer_tag=2, rebuild=False)
    solver.enable_excision(facet_tags, inner_tag=3, rebuild=False)
    solver.rebuild_operators()
    return solver


def _set_pulse(solver, mesh, w=1.5):
    from rsd.physics.initial_conditions import GaussianBump

    phi0 = GaussianBump(mesh=mesh, V=solver.V_scalar, A=1e-2, r0=5.0, w=w,
                        v0=0.0, direction="ingoing")
    solver.set_initial_conditions(phi0.get_function(), phi0.get_momentum())


def _evolve(mesh, facet_tags, mass_lumping, n_steps=N_STEPS, w=1.5, **extra):
    solver = _make_solver(mesh, facet_tags, mass_lumping, **extra)
    _set_pulse(solver, mesh, w=w)
    dt = 0.5 * solver.compute_adaptive_dt()
    for _ in range(n_steps):
        solver.ssp_rk3_step(dt)
    return solver.u.x.array.copy()


def test_lumped_inverse_positive_and_no_ksp(shell_mesh):
    mesh, facet_tags = shell_mesh
    solver = _make_solver(mesh, facet_tags, mass_lumping=True)
    assert solver.mass_solver is None, "lumping no debe crear KSP de masa"
    inv = solver._mass_lumped_inv.getArray(readonly=True)
    assert np.all(np.isfinite(inv))
    assert np.all(inv > 0.0), "row-sum de masa P1 con √γ>0 debe ser > 0"


def test_lumping_ab_vs_consistent(shell_mesh):
    """A/B sobre un pulso RESUELTO por la malla (w ≫ lc no se puede aquí;
    w=4 ≈ 1.6·lc es lo honesto en la malla gruesa del suite rápido): misma
    ecuación con masa lumped vs consistente ⇒ diferencia O(h²) pequeña en
    norma global, pero NO cero (cero delataría ruta lumped inactiva). El
    A/B a resolución de producción vive en
    benchmarks/benchmark_mass_lumping.py. Con pulsos sub-resueltos ambas
    masas difieren O(1) en L∞ — eso no es un fallo del lumping."""
    mesh, facet_tags = shell_mesh
    u_cons = _evolve(mesh, facet_tags, mass_lumping=False, w=4.0)
    u_lump = _evolve(mesh, facet_tags, mass_lumping=True, w=4.0)

    assert np.all(np.isfinite(u_cons))
    assert np.all(np.isfinite(u_lump))
    scale = float(np.sqrt(np.mean(u_cons**2))) or 1.0
    diff = float(np.sqrt(np.mean((u_lump - u_cons) ** 2))) / scale
    assert 1e-13 < diff < 0.15, f"lumped vs consistente: rel diff L2 {diff:.3e}"


def test_lumping_long_run_stays_bounded(shell_mesh):
    """El riesgo real del lumping es inestabilidad, no inexactitud."""
    mesh, facet_tags = shell_mesh
    u = _evolve(mesh, facet_tags, mass_lumping=True, n_steps=40)
    assert np.all(np.isfinite(u))
    assert float(np.max(np.abs(u))) < 1.0, "amplitud creció sin control"


def test_lumping_rejects_degree_above_1(shell_mesh):
    from rsd.solvers.first_order import FirstOrderKGSolver

    mesh, _ = shell_mesh
    with pytest.raises(ValueError, match="mass_lumping"):
        FirstOrderKGSolver(
            mesh=mesh, domain_radius=R, degree=2,
            potential_type="zero",
            cfg={"optimization": {"mass_lumping": True}},
        )


def test_filter_runs_on_lumped_path(shell_mesh):
    """Iteración de potencias de λmax + aplicación del filtro vía M_L⁻¹."""
    mesh, facet_tags = shell_mesh
    u = _evolve(mesh, facet_tags, mass_lumping=True, n_steps=5,
                filter_strength=0.005, filter_order=4)
    assert np.all(np.isfinite(u))
