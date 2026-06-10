"""
Convergencia temporal: mide el ORDEN observado del integrador SSP-RK3
(antes solo se exigía err_fino <= 1.1·err_grueso, que pasa con casi
cualquier esquema). Calibrado: p ≈ 3.1 en esta malla.
"""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
pytestmark = [pytest.mark.slow, pytest.mark.requires_numpy, pytest.mark.requires_dolfinx]


def test_ssp_rk3_observed_temporal_order():
    pytest.importorskip("dolfinx")
    from mpi4py import MPI
    from psyop.mesh.gmsh import build_ball_mesh
    from psyop.physics.initial_conditions import GaussianBump
    from psyop.solvers.first_order import FirstOrderKGSolver

    # La misma malla para todas las corridas: el error espacial es idéntico
    # y la comparación entre dt mide convergencia puramente temporal
    mesh, _, _ = build_ball_mesh(R=4.0, lc=1.2, comm=MPI.COMM_WORLD)
    t_end = 0.48

    def evolve(dt: float):
        solver = FirstOrderKGSolver(
            mesh=mesh,
            domain_radius=4.0,
            degree=1,
            potential_type="quadratic",
            potential_params={"m_squared": 1.0},
            cfl_factor=0.2,
        )
        ic = GaussianBump(mesh, V=solver.V_scalar, A=0.01, r0=1.4, w=0.8, v0=0.0)
        solver.set_initial_conditions(ic.get_function())
        n_steps = round(t_end / dt)
        for _ in range(n_steps):
            solver.ssp_rk3_step(dt)
        return solver.u.x.array.copy()

    ref = evolve(0.005)
    err_coarse = np.linalg.norm(evolve(0.04) - ref)
    err_fine = np.linalg.norm(evolve(0.02) - ref)

    assert err_fine > 0 and err_coarse > 0, "errors must be nonzero to measure order"
    observed_order = float(np.log2(err_coarse / err_fine))
    # SSP-RK3 es de 3.er orden; margen para contaminación de la referencia
    assert 2.5 < observed_order < 4.0, (
        f"observed temporal order {observed_order:.2f} outside [2.5, 4.0] "
        f"(err_coarse={err_coarse:.3e}, err_fine={err_fine:.3e})"
    )
