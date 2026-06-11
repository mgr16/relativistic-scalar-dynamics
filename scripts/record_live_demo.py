#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
record_live_demo.py – Genera docs/media/live_demo.gif para el README.

Corre una simulación corta (pulso gaussiano saliente en fondo plano) con el
LiveViewer en modo off-screen y arma un GIF con un frame cada N pasos.
Requiere pyvista e imageio (extra `viz` + imageio).

Uso (macOS necesita CC=/usr/bin/clang para el JIT de DOLFINx):
    CC=/usr/bin/clang python scripts/record_live_demo.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import imageio.v2 as imageio
from mpi4py import MPI

from psyop.mesh.gmsh import build_ball_mesh, get_outer_tag
from psyop.physics.initial_conditions import GaussianBump
from psyop.physics.metrics import make_background
from psyop.solvers.first_order import FirstOrderKGSolver
from psyop.utils.live_view import LiveViewer
from psyop.utils.utils import compute_dt_cfl

R = 12.0
T_END = 16.0
FRAME_EVERY = 2  # pasos entre frames
FPS = 12

OUT = ROOT / "docs" / "media" / "live_demo.gif"


def main() -> int:
    mesh, _, facet_tags = build_ball_mesh(R=R, lc=1.0, comm=MPI.COMM_WORLD)
    if mesh.comm.size > 1:
        raise SystemExit("record_live_demo.py debe correrse en serie (1 rank)")

    bg = make_background({"type": "flat", "M": 1.0})
    alpha, beta, gammaInv, sqrtg, K = bg.build(mesh)

    solver = FirstOrderKGSolver(
        mesh=mesh,
        domain_radius=R,
        degree=1,
        potential_type="higgs",
        potential_params={"m_squared": 1.0, "lambda_coupling": 0.1},
        cfl_factor=0.3,
    )
    solver.set_background(
        alpha=alpha, beta=beta, gammaInv=gammaInv, sqrtg=sqrtg, K=K, rebuild=False
    )
    solver.enable_sommerfeld(
        facet_tags, outer_tag=get_outer_tag(facet_tags, default=2), rebuild=False
    )
    solver.rebuild_operators()

    phi0 = GaussianBump(
        mesh=mesh, V=solver.V_scalar, A=1.0, r0=4.0, w=1.2, v0=0.0,
        direction="outgoing",
    )
    solver.set_initial_conditions(phi0.get_function(), phi0.get_momentum())

    viewer = LiveViewer(solver.V_phi, off_screen=True, window_size=(640, 480))
    frames = []

    def capture(t: float) -> None:
        phi, _ = solver.get_fields()
        viewer.update(phi, t)
        if viewer.failed:
            raise RuntimeError("LiveViewer falló durante la captura")
        frames.append(viewer.plotter.screenshot(return_img=True))

    dt = compute_dt_cfl(mesh, cfl=0.3, c_max=1.0, degree=1)
    t, step = 0.0, 0
    capture(t)
    while t < T_END:
        step_dt = min(dt, T_END - t)
        solver.ssp_rk3_step(step_dt)
        t += step_dt
        step += 1
        if step % FRAME_EVERY == 0:
            capture(t)
    viewer.close()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(OUT, frames, fps=FPS, loop=0)
    size_mb = OUT.stat().st_size / 1e6
    print(f"GIF escrito: {OUT} ({len(frames)} frames, {size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
