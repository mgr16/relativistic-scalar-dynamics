"""
Balance de energía de Killing (docs/math/killing_energy.md).

En fondos estacionarios (Kerr-Schild) la energía de Killing E_K satisface
una ley de conservación con flujos puros de superficie:

    E_K(t) − E_K(0) + ∫ (F_inner + F_outer) dt = 0 ,

sin los términos de volumen β/K que impiden cerrar el balance euleriano.
Escalera de validación:
1. Oráculo 1D, plano: E_K = E y el balance cierra cuando el pulso sale.
2. Oráculo 1D, Schwarzschild-KS: un pulso entrante cruza el horizonte y el
   flujo por r_min recupera TODA la energía (∫F_in ≈ E_K0); el residual
   converge a 2.º orden.
3. Solver 3D con excisión: el residual de Killing es pequeño y mucho menor
   que el residual euleriano naive en la misma corrida.
"""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from psyop.reference import SphericalOracle1D  # noqa: E402

pytestmark = [pytest.mark.requires_numpy]

try:
    from mpi4py import MPI  # noqa: F401

    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False


def _cumulative_flux(ts, flux):
    return np.concatenate(
        [[0.0], np.cumsum(0.5 * (flux[1:] + flux[:-1]) * np.diff(ts))]
    )


def _killing_residual(out):
    f_total = out.flux_inner_killing + out.flux_outer_killing
    cum = _cumulative_flux(out.ts, f_total)
    return out.energies_killing - out.energies_killing[0] + cum


# ----------------------------------------------------------------------
# 1. oráculo 1D — plano
# ----------------------------------------------------------------------

def test_killing_equals_eulerian_flat_and_balance_closes():
    oracle = SphericalOracle1D(
        M=0.0, l=0, r_min=2.0, r_max=60.0, n_points=1200,
        grid="uniform", ko_eps=0.0,
    )
    oracle.set_initial_gaussian(A=1e-3, r0=30.0, width=3.0, direction="outgoing")
    out = oracle.evolve(t_end=40.0, output_every=5)

    # con M=0: α=1, β=0 ⇒ E_K ≡ E
    assert np.allclose(out.energies_killing, out.energies, rtol=1e-12)

    E0 = out.energies_killing[0]
    res = _killing_residual(out)
    assert np.max(np.abs(res)) / E0 < 2e-3
    # el pulso salió del dominio
    assert out.energies_killing[-1] / E0 < 1e-6


# ----------------------------------------------------------------------
# 2. oráculo 1D — a través del horizonte (Schwarzschild-KS)
# ----------------------------------------------------------------------

def _horizon_run(n_points: int):
    oracle = SphericalOracle1D(
        M=1.0, l=0, r_min=0.5, r_max=40.0, n_points=n_points,
        grid="log", ko_eps=0.0,
    )
    oracle.set_initial_gaussian(
        A=1e-3, r0=6.0, width=1.0, direction="ingoing_curved"
    )
    return oracle.evolve(t_end=12.0, output_every=10)


@pytest.fixture(scope="module")
def horizon_runs():
    return {n: _horizon_run(n) for n in (800, 1600)}


def test_killing_balance_closes_through_horizon(horizon_runs):
    out = horizon_runs[1600]
    E0 = out.energies_killing[0]
    res = _killing_residual(out)
    assert np.max(np.abs(res)) / E0 < 1e-3

    # el flujo por r_min recupera toda la energía absorbida
    absorbed = _cumulative_flux(out.ts, out.flux_inner_killing)[-1]
    assert abs(absorbed / E0 - 1.0) < 5e-3


def test_killing_residual_second_order(horizon_runs):
    res = {
        n: np.max(np.abs(_killing_residual(out))) / out.energies_killing[0]
        for n, out in horizon_runs.items()
    }
    ratio = res[800] / res[1600]
    assert ratio > 2.5, f"convergencia insuficiente del residual: ×{ratio:.2f}"


# ----------------------------------------------------------------------
# 3. solver 3D con excisión (Kerr-Schild a=0)
# ----------------------------------------------------------------------

def _run_3d(lc: float, lc_inner: float, t_end: float = 12.0):
    from psyop.mesh.gmsh import build_ball_mesh
    from psyop.physics.initial_conditions import GaussianBump
    from psyop.physics.metrics import KerrSchildCoeffs
    from psyop.solvers.first_order import FirstOrderKGSolver

    comm = MPI.COMM_WORLD
    mesh, _, facet_tags = build_ball_mesh(
        R=10.0, lc=lc, comm=comm, r_inner=1.0, lc_inner=lc_inner
    )
    if facet_tags is None:
        pytest.skip("mesh without facet tags (gmsh unavailable)")

    solver = FirstOrderKGSolver(
        mesh=mesh, domain_radius=10.0, degree=1,
        potential_type="zero", cfl_factor=0.3,
    )
    bg = KerrSchildCoeffs(M=1.0, a=0.0).build(mesh)
    solver.set_background(*bg, rebuild=False)
    solver.enable_sommerfeld(facet_tags, outer_tag=2, rebuild=False)
    solver.enable_excision(facet_tags, inner_tag=3, rebuild=False)
    solver.rebuild_operators()

    ic = GaussianBump(mesh=mesh, V=solver.V_scalar, A=1e-3, r0=5.0, w=1.2,
                      v0=0.0, direction="ingoing")
    solver.set_initial_conditions(ic.get_function(), ic.get_momentum())

    dt = 0.5 * solver.compute_adaptive_dt()
    ts, e_naive, f_naive, e_k, fk_out, fk_in = [], [], [], [], [], []
    t = 0.0
    while t < t_end:
        ts.append(t)
        e_naive.append(solver.energy())
        f_naive.append(solver.boundary_flux() + solver.inner_flux())
        e_k.append(solver.energy_killing())
        fk_out.append(solver.killing_flux())
        fk_in.append(solver.killing_inner_flux())
        solver.ssp_rk3_step(dt)
        t += dt

    ts = np.asarray(ts)
    e_k = np.asarray(e_k)
    res_naive = (
        np.asarray(e_naive) - e_naive[0]
        + _cumulative_flux(ts, np.asarray(f_naive))
    )
    fk_total = np.asarray(fk_out) + np.asarray(fk_in)
    res_k = e_k - e_k[0] + _cumulative_flux(ts, fk_total)
    absorbed = _cumulative_flux(ts, np.asarray(fk_in))[-1]
    return {
        "E0": float(e_k[0]),
        "E_final_rel": float(e_k[-1] / e_k[0]),
        "res_k_rel": float(np.max(np.abs(res_k)) / e_k[0]),
        "res_naive_rel": float(abs(res_naive[-1]) / e_naive[0]),
        "absorbed_rel": float(absorbed / e_k[0]),
    }


@pytest.mark.requires_dolfinx
@pytest.mark.requires_mesh
@pytest.mark.skipif(not HAS_DOLFINX, reason="DOLFINx not available")
def test_killing_3d_absorption_smoke():
    """Malla gruesa: firma y magnitud correctas del flujo de absorción.

    Un error de transcripción UFL (signo, factor α/β) rompería estas cotas
    por completo; la convergencia fina del residual es el test `slow`.
    """
    out = _run_3d(lc=1.6, lc_inner=0.5)
    # el flujo interior recupera la energía absorbida (≈E0, malla gruesa)
    assert 0.6 < out["absorbed_rel"] < 1.05, out
    # el campo abandonó el dominio y el residual es moderado incluso grueso
    assert out["E_final_rel"] < 0.15, out
    assert out["res_k_rel"] < 0.25, out


@pytest.mark.slow
@pytest.mark.requires_dolfinx
@pytest.mark.requires_mesh
@pytest.mark.skipif(not HAS_DOLFINX, reason="DOLFINx not available")
def test_killing_3d_residual_converges_and_beats_naive():
    """Refinando ~×1.8 el residual de Killing cae ~×3 (≈2.º orden) y queda
    por debajo del residual euleriano, que converge a su término de volumen
    (medido: 13.0%→3.9% vs naive 4.3%→8.6%)."""
    coarse = _run_3d(lc=1.6, lc_inner=0.5)
    fine = _run_3d(lc=0.9, lc_inner=0.18)
    assert fine["res_k_rel"] < 0.5 * coarse["res_k_rel"], (coarse, fine)
    assert fine["res_k_rel"] < fine["res_naive_rel"], fine
    assert abs(fine["absorbed_rel"] - 1.0) < 0.1, fine
