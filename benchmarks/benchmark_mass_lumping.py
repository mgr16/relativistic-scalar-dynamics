#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark de mass lumping (M_L⁻¹ puntual) vs masa consistente (KSP CG).

Tras el fast path los solves de masa dominan el paso; el lumping los
elimina. Caso representativo: Kerr-Schild + excisión + Sommerfeld,
potencial lineal (donde el paso es 100% matvecs + solves de masa).

Uso: python benchmarks/benchmark_mass_lumping.py [--lc 0.8] [--steps 20]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mpi4py import MPI  # noqa: E402

from rsd.mesh.gmsh import build_ball_mesh  # noqa: E402
from rsd.physics.initial_conditions import GaussianBump  # noqa: E402
from rsd.physics.metrics import KerrSchildCoeffs  # noqa: E402
from rsd.solvers.first_order import FirstOrderKGSolver  # noqa: E402


def run(mesh, facet_tags, R, mass_lumping, steps):
    solver = FirstOrderKGSolver(
        mesh=mesh, domain_radius=R, degree=1,
        potential_type="zero",
        cfg={"optimization": {"preassemble": True,
                              "mass_lumping": mass_lumping}},
    )
    bg = KerrSchildCoeffs(M=1.0, a=0.0).build(mesh)
    solver.set_background(*bg, rebuild=False)
    solver.enable_sommerfeld(facet_tags, outer_tag=2, rebuild=False)
    solver.enable_excision(facet_tags, inner_tag=3, rebuild=False)
    solver.rebuild_operators()
    # w=4: pulso RESUELTO por la malla — la desviación reportada abajo es la
    # del régimen convergente O(h^p). Con frentes sub-resueltos (w ~ lc) las
    # masas lumped/consistente difieren O(1) en L∞ por diseño espectral; ese
    # número no mide la calidad del lumping.
    phi0 = GaussianBump(mesh=mesh, V=solver.V_scalar, A=1e-3, r0=6.0, w=4.0,
                        v0=0.0, direction="ingoing")
    solver.set_initial_conditions(phi0.get_function(), phi0.get_momentum())
    dt = solver.compute_adaptive_dt()

    solver.ssp_rk3_step(dt)  # calentamiento (JIT, caches)
    t0 = time.perf_counter()
    for _ in range(steps):
        solver.ssp_rk3_step(dt)
    wall = time.perf_counter() - t0
    return wall / steps, solver.u.x.array.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lc", type=float, default=0.8)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    R, r_inner = 15.0, 1.0
    comm = MPI.COMM_WORLD
    mesh, _, facet_tags = build_ball_mesh(
        R=R, lc=args.lc, comm=comm, r_inner=r_inner, lc_inner=args.lc / 3
    )
    n_cells = mesh.topology.index_map(mesh.topology.dim).size_global
    print(f"malla: lc={args.lc}, {n_cells} celdas, {args.steps} pasos")

    t_cons, u_cons = run(mesh, facet_tags, R, False, args.steps)
    t_lump, u_lump = run(mesh, facet_tags, R, True, args.steps)

    sc_inf = float(np.max(np.abs(u_cons))) or 1.0
    sc_l2 = float(np.sqrt(np.mean(u_cons**2))) or 1.0
    d_inf = float(np.max(np.abs(u_lump - u_cons))) / sc_inf
    d_l2 = float(np.sqrt(np.mean((u_lump - u_cons) ** 2))) / sc_l2
    print(f"consistente: {t_cons * 1e3:8.2f} ms/paso")
    print(f"lumped:      {t_lump * 1e3:8.2f} ms/paso   "
          f"(speedup ×{t_cons / t_lump:.2f})")
    print(f"desviación (pulso resuelto w=4): rel_linf={d_inf:.3e}  "
          f"rel_l2={d_l2:.3e}")


if __name__ == "__main__":
    main()
