#!/usr/bin/env python3
"""
test_sommerfeld_reflection.py

Test A/B: la condición característica saliente (Sommerfeld) debe absorber
energía cuando el pulso llega al borde, frente al caso sin BC (Neumann
natural, reflectante), que la conserva aproximadamente.
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

R = 12.0
T_END = 8.0
MESH_LC = 3.0


def run_case(use_sommerfeld: bool):
    """Evoluciona un pulso gaussiano y devuelve (E0, E_final, flujo_medio)."""
    from rsd.mesh.gmsh import build_ball_mesh, get_outer_tag
    from rsd.physics.initial_conditions import GaussianBump
    from rsd.solvers.first_order import FirstOrderKGSolver
    from rsd.utils.utils import compute_dt_cfl

    comm = MPI.COMM_WORLD
    mesh, _, facet_tags = build_ball_mesh(R=R, lc=MESH_LC, comm=comm)
    dt = compute_dt_cfl(mesh, cfl=0.2, c_max=1.0)

    solver = FirstOrderKGSolver(
        mesh=mesh,
        domain_radius=R,
        degree=1,
        potential_type="quadratic",
        potential_params={"m_squared": 0.1},
        cfl_factor=0.2,
    )

    if use_sommerfeld:
        assert facet_tags is not None, "mesh generation must provide facet tags"
        solver.enable_sommerfeld(facet_tags, get_outer_tag(facet_tags, default=2))

    # Pulso centrado lejos del borde, sin offset de vacío (v0=0):
    # toda la energía es radiativa y debe salir del dominio
    phi0 = GaussianBump(mesh=mesh, V=solver.V_scalar, A=0.1, r0=R / 3, w=R / 8, v0=0.0)
    solver.set_initial_conditions(phi0.get_function())

    energy_initial = solver.energy()
    fluxes = []
    t = 0.0
    while t < T_END:
        step_dt = min(dt, T_END - t)
        solver.ssp_rk3_step(step_dt)
        t += step_dt
        if use_sommerfeld:
            fluxes.append(solver.boundary_flux())

    energy_final = solver.energy()
    mean_flux = float(np.mean(fluxes)) if fluxes else 0.0
    return energy_initial, energy_final, mean_flux


def test_sommerfeld_reduces_reflection():
    """Con Sommerfeld el dominio debe retener menos energía que sin BC."""
    E0_off, Ef_off, _ = run_case(use_sommerfeld=False)
    E0_on, Ef_on, mean_flux = run_case(use_sommerfeld=True)

    assert E0_off > 0 and E0_on > 0, "initial energy must be positive"
    # Misma condición inicial en ambos casos
    assert E0_on == pytest.approx(E0_off, rel=1e-8)

    loss_off = (E0_off - Ef_off) / E0_off
    loss_on = (E0_on - Ef_on) / E0_on

    # La BC absorbente debe perder estrictamente más energía que el caso
    # reflectante una vez que el pulso alcanza el borde
    assert loss_on > loss_off, (
        f"Sommerfeld should absorb energy: loss_on={loss_on:.4%} "
        f"vs loss_off={loss_off:.4%}"
    )

    # El flujo medio saliente debe ser positivo (convención: saliente > 0)
    assert mean_flux > 0.0, f"outgoing mean flux should be positive, got {mean_flux:.3e}"
