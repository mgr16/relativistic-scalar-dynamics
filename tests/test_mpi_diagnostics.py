import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _run_case(nproc: int) -> dict:
    repo_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    code = r"""
import json
from mpi4py import MPI
from psyop.mesh.gmsh import build_ball_mesh, get_outer_tag
from psyop.physics.initial_conditions import GaussianBump
from psyop.solvers.first_order import FirstOrderKGSolver

comm = MPI.COMM_WORLD
mesh, _, facet_tags = build_ball_mesh(R=4.0, lc=2.0, comm=comm)
solver = FirstOrderKGSolver(mesh=mesh, domain_radius=4.0, degree=1, potential_type="quadratic", potential_params={"m_squared": 1.0}, cfl_factor=0.2)
if facet_tags is not None:
    solver.enable_sommerfeld(facet_tags, outer_tag=get_outer_tag(facet_tags, default=2))
ic = GaussianBump(mesh, A=0.01, r0=1.5, w=0.8, v0=0.0)
solver.set_initial_conditions(ic.get_function())
for _ in range(3):
    solver.ssp_rk3_step(0.01)
out = {"energy": solver.energy(), "flux": solver.boundary_flux()}
if comm.rank == 0:
    print(json.dumps(out))
"""
    if nproc == 1:
        cmd = [sys.executable, "-c", code]
    else:
        cmd = [shutil.which("mpiexec"), "-n", str(nproc), sys.executable, "-c", code]
    out = subprocess.check_output(cmd, cwd=str(repo_root), env=env, text=True)
    lines = [line.strip() for line in out.splitlines() if line.strip().startswith("{")]
    return json.loads(lines[-1])


@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="mpiexec not available")
@pytest.mark.mpi
def test_mpi_global_diagnostics_consistent_between_1_and_2_ranks():
    pytest.importorskip("dolfinx")
    one = _run_case(1)
    two = _run_case(2)
    assert one["energy"] == pytest.approx(two["energy"], rel=2e-2, abs=1e-8)
    assert one["flux"] == pytest.approx(two["flux"], rel=2e-2, abs=1e-8)
