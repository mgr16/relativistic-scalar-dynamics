#!/usr/bin/env python3
"""
test_operator_fastpath.py

A/B de exactitud: la ruta rápida (operador lineal completo preensamblado,
RHS = A·u + resto cúbico) debe reproducir la ruta lenta (forma no lineal
ensamblada por etapa) hasta el redondeo, para cada potencial y con toda la
física activada (Kerr-Schild + excisión + Sommerfeld + esponja).

Este test es el guardián del refactor de rendimiento de la Fase 1: si las
rutas divergen, el preensamblado dejó de representar la misma ecuación.
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
N_STEPS = 5

POTENTIALS = [
    ("zero", {}),
    ("quadratic", {"m_squared": 0.5}),
    ("higgs", {"m_squared": 0.5, "lambda_coupling": 0.2}),
    ("mexican_hat", {"lambda_coupling": 0.2, "vacuum_value": 1.0}),
]


@pytest.fixture(scope="module")
def shell_mesh():
    from psyop.mesh.gmsh import build_ball_mesh

    comm = MPI.COMM_WORLD
    mesh, _, facet_tags = build_ball_mesh(R=R, lc=LC, comm=comm, r_inner=R_INNER)
    if facet_tags is None:
        pytest.skip("mesh without facet tags (gmsh unavailable)")
    return mesh, facet_tags


def _evolve(mesh, facet_tags, potential_type, potential_params, preassemble):
    from psyop.physics.initial_conditions import GaussianBump
    from psyop.physics.metrics import KerrSchildCoeffs
    from psyop.solvers.first_order import FirstOrderKGSolver

    v0 = 1.0 if potential_type == "mexican_hat" else 0.0
    solver = FirstOrderKGSolver(
        mesh=mesh,
        domain_radius=R,
        degree=1,
        potential_type=potential_type,
        potential_params=potential_params,
        cfl_factor=0.3,
        cfg={"optimization": {"preassemble": preassemble}},
        sponge={"enabled": True, "width": 2.0, "strength": 0.5},
    )
    bg = KerrSchildCoeffs(M=1.0, a=0.0).build(mesh)
    solver.set_background(*bg, rebuild=False)
    solver.enable_sommerfeld(facet_tags, outer_tag=2, rebuild=False)
    solver.enable_excision(facet_tags, inner_tag=3, rebuild=False)
    solver.rebuild_operators()

    phi0 = GaussianBump(mesh=mesh, V=solver.V_scalar, A=1e-2, r0=5.0, w=1.5,
                        v0=v0, direction="ingoing")
    solver.set_initial_conditions(phi0.get_function(), phi0.get_momentum())

    dt = 0.5 * solver.compute_adaptive_dt()
    for _ in range(N_STEPS):
        solver.ssp_rk3_step(dt)
    return solver.u.x.array.copy()


@pytest.mark.parametrize("potential_type,potential_params", POTENTIALS)
def test_fast_path_matches_slow_path(shell_mesh, potential_type, potential_params):
    mesh, facet_tags = shell_mesh
    u_fast = _evolve(mesh, facet_tags, potential_type, potential_params, True)
    u_slow = _evolve(mesh, facet_tags, potential_type, potential_params, False)

    scale = float(np.max(np.abs(u_slow))) or 1.0
    diff = float(np.max(np.abs(u_fast - u_slow))) / scale
    assert diff < 1e-10, (
        f"{potential_type}: fast/slow paths diverged (rel diff {diff:.3e})"
    )


def test_fully_linear_potentials_skip_per_step_assembly(shell_mesh):
    """zero/quadratic: la ruta rápida no debe ensamblar nada por paso."""
    from psyop.physics.metrics import KerrSchildCoeffs
    from psyop.solvers.first_order import FirstOrderKGSolver

    mesh, facet_tags = shell_mesh
    solver = FirstOrderKGSolver(
        mesh=mesh, domain_radius=R, degree=1,
        potential_type="quadratic", potential_params={"m_squared": 1.0},
        cfg={"optimization": {"preassemble": True}},
    )
    bg = KerrSchildCoeffs(M=1.0, a=0.0).build(mesh)
    solver.set_background(*bg, rebuild=False)
    solver.enable_sommerfeld(facet_tags, outer_tag=2, rebuild=False)
    solver.enable_excision(facet_tags, inner_tag=3, rebuild=False)
    solver.rebuild_operators()
    assert solver._nonlinear_form_compiled is None

    solver_nl = FirstOrderKGSolver(
        mesh=mesh, domain_radius=R, degree=1,
        potential_type="higgs",
        potential_params={"m_squared": 1.0, "lambda_coupling": 0.1},
        cfg={"optimization": {"preassemble": True}},
    )
    solver_nl.set_background(*bg, rebuild=False)
    solver_nl.enable_sommerfeld(facet_tags, outer_tag=2, rebuild=False)
    solver_nl.enable_excision(facet_tags, inner_tag=3, rebuild=False)
    solver_nl.rebuild_operators()
    assert solver_nl._nonlinear_form_compiled is not None
