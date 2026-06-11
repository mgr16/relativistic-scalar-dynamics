"""
Tests del monitor de Cowling: la métrica de validez ζ = 8πρ/√K debe escalar
con A² (ρ es cuadrática en el campo) y disparar el warning solo para
amplitudes donde el backreaction dejaría de ser despreciable.
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
    pytest.mark.requires_dolfinx,
    pytest.mark.skipif(not HAS_DOLFINX, reason="DOLFINx not available"),
]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _make_solver(A: float):
    from psyop.mesh.gmsh import build_ball_mesh
    from psyop.physics.initial_conditions import GaussianBump
    from psyop.physics.metrics import make_background
    from psyop.solvers.first_order import FirstOrderKGSolver

    metric_cfg = {"type": "kerr", "M": 1.0, "a": 0.0}
    mesh, _, facet_tags = build_ball_mesh(
        R=10.0, lc=1.5, comm=MPI.COMM_WORLD, r_inner=1.0
    )
    bg = make_background(metric_cfg)
    alpha, beta, gammaInv, sqrtg, K = bg.build(mesh)
    solver = FirstOrderKGSolver(
        mesh=mesh, domain_radius=10.0, degree=1,
        potential_type="zero", cfl_factor=0.3,
    )
    solver.set_background(
        alpha=alpha, beta=beta, gammaInv=gammaInv, sqrtg=sqrtg, K=K, rebuild=False
    )
    solver.rebuild_operators()
    bump = GaussianBump(mesh=mesh, V=solver.V_scalar, A=A, r0=5.0, w=1.5, v0=0.0)
    solver.set_initial_conditions(bump.get_function(), bump.get_momentum())
    return solver, metric_cfg


def test_zeta_scales_quadratically_with_amplitude():
    from psyop.analysis.cowling import CowlingMonitor

    solver1, metric_cfg = _make_solver(A=1e-3)
    m1 = CowlingMonitor(solver1, metric_cfg).evaluate()

    solver2, _ = _make_solver(A=1e-2)
    m2 = CowlingMonitor(solver2, metric_cfg).evaluate()

    # rho ~ A^2: subir A x10 debe subir zeta y E x100
    assert m2["zeta_max"] == pytest.approx(100.0 * m1["zeta_max"], rel=1e-6)
    assert m2["energy_ratio"] == pytest.approx(100.0 * m1["energy_ratio"], rel=1e-6)
    # Amplitud chica: test-field claramente valido
    assert m1["zeta_max"] < 1e-2
    assert m1["energy_ratio"] < 1e-2


def test_check_warns_for_large_amplitude(caplog):
    from psyop.analysis.cowling import CowlingMonitor

    solver, metric_cfg = _make_solver(A=10.0)
    monitor = CowlingMonitor(solver, metric_cfg)
    with caplog.at_level("WARNING"):
        result = monitor.check(t=0.0)
        monitor.check(t=1.0)  # el warning se emite una sola vez
    assert result["zeta_max"] > 1e-2
    warnings = [r for r in caplog.records if "Cowling" in r.message]
    assert len(warnings) == 1


def test_flat_background_uses_domain_scale():
    from psyop.analysis.cowling import CowlingMonitor
    from psyop.mesh.gmsh import build_ball_mesh
    from psyop.physics.initial_conditions import GaussianBump
    from psyop.physics.metrics import make_background
    from psyop.solvers.first_order import FirstOrderKGSolver

    metric_cfg = {"type": "flat", "M": 1.0}
    mesh, _, _ = build_ball_mesh(R=8.0, lc=1.5, comm=MPI.COMM_WORLD)
    bg = make_background(metric_cfg)
    alpha, beta, gammaInv, sqrtg, K = bg.build(mesh)
    solver = FirstOrderKGSolver(
        mesh=mesh, domain_radius=8.0, degree=1,
        potential_type="zero", cfl_factor=0.3,
    )
    solver.set_background(
        alpha=alpha, beta=beta, gammaInv=gammaInv, sqrtg=sqrtg, K=K, rebuild=False
    )
    solver.rebuild_operators()
    bump = GaussianBump(mesh=mesh, V=solver.V_scalar, A=1e-3, r0=4.0, w=1.5, v0=0.0)
    solver.set_initial_conditions(bump.get_function(), bump.get_momentum())

    result = CowlingMonitor(solver, metric_cfg).evaluate()
    assert np.isfinite(result["zeta_max"]) and result["zeta_max"] > 0
    assert result["zeta_max"] < 1e-2
