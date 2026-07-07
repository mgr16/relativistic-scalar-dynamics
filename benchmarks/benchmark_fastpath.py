#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark de la ruta rápida (operador preensamblado) vs la ruta lenta.

Caso representativo del estudio de convergencia: Kerr-Schild + excisión +
Sommerfeld, potenciales lineal (zero) y no lineal (higgs).

Uso: python benchmarks/benchmark_fastpath.py [--lc 0.8] [--steps 20]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mpi4py import MPI  # noqa: E402

from psyop.mesh.gmsh import build_ball_mesh  # noqa: E402
from psyop.physics.initial_conditions import GaussianBump  # noqa: E402
from psyop.physics.metrics import KerrSchildCoeffs  # noqa: E402
from psyop.solvers.first_order import FirstOrderKGSolver  # noqa: E402


def run(mesh, facet_tags, R, potential_type, potential_params, preassemble, steps):
    solver = FirstOrderKGSolver(
        mesh=mesh, domain_radius=R, degree=1,
        potential_type=potential_type, potential_params=potential_params,
        cfg={"optimization": {"preassemble": preassemble}},
    )
    bg = KerrSchildCoeffs(M=1.0, a=0.0).build(mesh)
    solver.set_background(*bg, rebuild=False)
    solver.enable_sommerfeld(facet_tags, outer_tag=2, rebuild=False)
    solver.enable_excision(facet_tags, inner_tag=3, rebuild=False)
    solver.rebuild_operators()
    phi0 = GaussianBump(mesh=mesh, V=solver.V_scalar, A=1e-3, r0=6.0, w=1.5,
                        v0=0.0, direction="ingoing")
    solver.set_initial_conditions(phi0.get_function(), phi0.get_momentum())
    dt = solver.compute_adaptive_dt()

    solver.ssp_rk3_step(dt)  # calentamiento (JIT, caches)
    t0 = time.perf_counter()
    for _ in range(steps):
        solver.ssp_rk3_step(dt)
    wall = time.perf_counter() - t0
    return wall / steps


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
    print(f"malla: lc={args.lc}, {n_cells} celdas", flush=True)

    for ptype, params in [("zero", {}), ("higgs", {"m_squared": 1.0, "lambda_coupling": 0.1})]:
        t_slow = run(mesh, facet_tags, R, ptype, params, False, args.steps)
        t_fast = run(mesh, facet_tags, R, ptype, params, True, args.steps)
        print(
            f"  {ptype:8s}: lenta {t_slow * 1e3:8.1f} ms/paso | "
            f"rápida {t_fast * 1e3:8.1f} ms/paso | speedup ×{t_slow / t_fast:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
