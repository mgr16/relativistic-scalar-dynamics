#!/usr/bin/env python3
"""Performance benchmarks for PSYOP."""

import time
import sys
from pathlib import Path
from typing import List, Tuple
import argparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from mpi4py import MPI
    import dolfinx
    HAS_DOLFINX = True
except ImportError:
    print("ERROR: DOLFINx is required for benchmarks")
    sys.exit(1)

from psyop.solvers.first_order import FirstOrderKGSolver
from psyop.mesh.gmsh import build_ball_mesh
from psyop.physics.initial_conditions import GaussianBump
from psyop.utils.logger import setup_logger

logger = setup_logger("benchmark", level="INFO")


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self, mesh_size, n_cells, degree, cfl, n_steps, 
                 total_time, time_per_step, cells_per_second):
        self.mesh_size = mesh_size
        self.n_cells = n_cells
        self.degree = degree
        self.cfl = cfl
        self.n_steps = n_steps
        self.total_time = total_time
        self.time_per_step = time_per_step
        self.cells_per_second = cells_per_second
    
    def __str__(self):
        return (f"R={self.mesh_size[0]:.1f} lc={self.mesh_size[1]:.1f} "
                f"cells={self.n_cells:6d} degree={self.degree} CFL={self.cfl:.2f} | "
                f"time/step={self.time_per_step:.4f}s cells/s={self.cells_per_second:.0f}")


def benchmark_solver(R, lc, degree=1, cfl=0.3, n_steps=100, warmup_steps=10):
    """Benchmark solver performance."""
    logger.info(f"Benchmarking: R={R}, lc={lc}, degree={degree}, CFL={cfl}")
    
    mesh, _, facet_tags = build_ball_mesh(R=R, lc=lc, comm=MPI.COMM_WORLD)
    n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    
    logger.info(f"Mesh has {n_cells} cells")
    
    solver = FirstOrderKGSolver(
        mesh=mesh, domain_radius=R, degree=degree,
        potential_type="quadratic", potential_params={"m_squared": 1.0},
        cfl_factor=cfl,
    )
    
    ic = GaussianBump(mesh, A=0.01, r0=R/2, w=R/10, v0=0.0)
    solver.set_initial_conditions(ic.get_function())
    
    from psyop.utils.utils import compute_dt_cfl
    dt = compute_dt_cfl(mesh, cfl=cfl, c_max=1.0)
    
    logger.info(f"Warming up ({warmup_steps} steps)...")
    for _ in range(warmup_steps):
        solver.ssp_rk3_step(dt)
    
    logger.info(f"Benchmarking ({n_steps} steps)...")
    start_time = time.perf_counter()
    for _ in range(n_steps):
        solver.ssp_rk3_step(dt)
    elapsed = time.perf_counter() - start_time
    
    time_per_step = elapsed / n_steps
    cells_per_second = n_cells / time_per_step
    
    result = BenchmarkResult(
        mesh_size=(R, lc), n_cells=n_cells, degree=degree, cfl=cfl,
        n_steps=n_steps, total_time=elapsed, time_per_step=time_per_step,
        cells_per_second=cells_per_second,
    )
    
    logger.info(f"Result: {result}")
    return result


def main():
    """Run all benchmarks."""
    parser = argparse.ArgumentParser(description="PSYOP Performance Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    args = parser.parse_args()
    
    if args.quick:
        configs = [(5.0, 2.5), (10.0, 2.5)]
        n_steps = 20
    else:
        configs = [(5.0, 3.0), (5.0, 2.0), (10.0, 3.0), (10.0, 2.0)]
        n_steps = 50
    
    logger.info("=" * 70)
    logger.info("PSYOP PERFORMANCE BENCHMARKS")
    logger.info("=" * 70)
    
    results = []
    for R, lc in configs:
        result = benchmark_solver(R=R, lc=lc, n_steps=n_steps)
        results.append(result)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    for r in results:
        logger.info(str(r))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
