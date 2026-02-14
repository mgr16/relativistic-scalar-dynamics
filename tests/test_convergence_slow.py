import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.mark.slow
def test_temporal_refinement_reduces_solution_error():
    pytest.importorskip("dolfinx")
    from mpi4py import MPI
    from psyop.mesh.gmsh import build_ball_mesh
    from psyop.physics.initial_conditions import GaussianBump
    from psyop.solvers.first_order import FirstOrderKGSolver

    mesh, _, _ = build_ball_mesh(R=4.0, lc=2.0, comm=MPI.COMM_WORLD)

    def evolve(dt: float):
        solver = FirstOrderKGSolver(
            mesh=mesh,
            domain_radius=4.0,
            degree=1,
            potential_type="zero",
            cfl_factor=0.2,
        )
        ic = GaussianBump(mesh, A=0.01, r0=1.4, w=0.8, v0=0.0)
        solver.set_initial_conditions(ic.get_function())
        t = 0.0
        while t < 0.12 - 1e-12:
            solver.ssp_rk3_step(dt)
            t += dt
        phi, _ = solver.get_fields()
        return phi.x.array.copy()

    ref = evolve(0.005)
    coarse = evolve(0.02)
    fine = evolve(0.01)
    err_coarse = np.linalg.norm(coarse - ref)
    err_fine = np.linalg.norm(fine - ref)
    assert err_fine <= 1.1 * err_coarse
