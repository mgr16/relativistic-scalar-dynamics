#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for PSYOP scalar field simulator.

This version requires DOLFINx (FEniCS legacy support removed).
"""

import json
import os
import time
import argparse
import logging
import sys
from typing import Dict, Any, Tuple, Optional

import numpy as np

# DOLFINx imports (required)
try:
    import dolfinx.fem as fem
    import dolfinx.io
    import dolfinx.geometry
    import ufl
    from mpi4py import MPI
except ImportError as e:
    print("ERROR: DOLFINx is required but not installed.")
    print("Install with: conda install -c conda-forge dolfinx")
    raise ImportError("DOLFINx is required to run PSYOP") from e

# Import logging utilities
from psyop.utils.logger import setup_logger, get_logger
from psyop.config import load_config, validate_config


def create_example_config(filename: str = "config_example.json") -> None:
    """Create example configuration file."""
    cfg = {
        "mesh": {"R": 30.0, "lc": 1.5, "mesh_type": "gmsh"},
        "solver": {
            "degree": 1,
            "cfl": 0.3,
            "potential_type": "quadratic",
            "potential_params": {"m_squared": 1.0},
            "ko_eps": 0.0,
            "bc_type": "characteristic",
            "outer_tag": 2,
            "enable_sommerfeld": True,
        },
        "metric": {"type": "flat", "M": 1.0},
        "initial_conditions": {"type": "gaussian", "A": 0.01, "r0": 10.0, "w": 3.0, "v0": 1.0},
        "evolution": {"t_end": 50.0, "output_every": 10, "verbose": True},
        "output": {"dir": "results", "qnm_analysis": True}
    }
    logger = get_logger()
    with open(filename, 'w') as f:
        json.dump(cfg, f, indent=2)
    logger.info(f"Example configuration created: {filename}")


def sample_phi_at_point(phi_fn, point: np.ndarray, mesh) -> Optional[float]:
    """
    Sample field value at a specific point.
    
    Args:
        phi_fn: DOLFINx function
        point: 3D point coordinates
        mesh: Mesh object
        
    Returns:
        Field value at point, or None if point not in mesh
    """
    point_reshaped = point.reshape(1, 3)
    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, point_reshaped)
    cells = dolfinx.geometry.compute_colliding_cells(mesh, candidates, point_reshaped)
    
    if len(cells.links(0)) == 0:
        return None
    
    cell = cells.links(0)[0]
    val = phi_fn.eval(point_reshaped[0], cell)
    return float(val[0]) if np.ndim(val) > 0 else float(val)


def main() -> int:
    """Main entry point for PSYOP simulation."""
    parser = argparse.ArgumentParser(
        description="PSYOP: Scalar field simulation with Sommerfeld boundary conditions"
    )
    parser.add_argument("--config", type=str, default=None, 
                        help="Configuration JSON file")
    parser.add_argument("--output", type=str, default="results", 
                        help="Output directory")
    parser.add_argument("--create-config", action="store_true", 
                        help="Create example configuration file")
    parser.add_argument("--test", action="store_true", 
                        help="Run basic test without FEM")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logger("psyop", level=log_level)
    
    if args.create_config:
        create_example_config()
        return 0
    
    # Test mode
    if args.test:
        logger.info("=== TEST MODE ===")
        logger.info(f"NumPy version: {np.__version__}")
        logger.info("DOLFINx: Available")
        logger.info("Ready to run simulations")
        return 0


    # Import components
    from psyop.solvers.first_order import FirstOrderKGSolver
    from psyop.mesh.gmsh import build_ball_mesh, get_outer_tag
    from psyop.physics.initial_conditions import GaussianBump
    from psyop.physics.metrics import make_background
    from psyop.utils.utils import compute_dt_cfl
    from psyop.analysis.qnm import compute_qnm, estimate_peak
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    else:
        config_file = 'config_example.json'
        if not os.path.exists(config_file):
            logger.warning(f"Config file not found: {config_file}")
            logger.info("Creating default configuration...")
            create_example_config(config_file)
        cfg = load_config(config_file)
        logger.info(f"Loaded default configuration from: {config_file}")
    
    # Normalize config
    if isinstance(cfg, dict) and isinstance(cfg.get("output"), dict):
        out = cfg["output"]
        if "dir" not in out and "results_dir" in out:
            out["dir"] = out["results_dir"]
    
    cfg = validate_config(cfg)
    logger.info("Configuration validated successfully")
    
    # Build mesh and metric
    logger.info(f"Building mesh: R={cfg['mesh']['R']}, lc={cfg['mesh']['lc']}")
    mesh, cell_tags, facet_tags = build_ball_mesh(
        R=cfg["mesh"]["R"], 
        lc=cfg["mesh"]["lc"], 
        comm=MPI.COMM_WORLD
    )
    logger.info(f"Mesh created with {mesh.topology.index_map(mesh.topology.dim).size_local} cells")
    
    bg = make_background(cfg["metric"])
    alpha_f, beta_f, gammaInv_f, sqrtg_f, K_f = bg.build(mesh)
    c_max = getattr(bg, 'max_characteristic_speed', lambda m: 1.0)(mesh)
    logger.info(f"Background metric set: {cfg['metric']['type']}")
    
    # Setup output directory
    dt = compute_dt_cfl(mesh, cfl=cfg["solver"]["cfl"], c_max=c_max)
    logger.info(f"CFL timestep: dt = {dt:.6e}")
    
    outdir = os.path.join(
        cfg.get("output", {}).get("dir", args.output), 
        time.strftime("run_%Y%m%d_%H%M%S")
    )
    if mesh.comm.rank == 0:
        os.makedirs(outdir, exist_ok=True)
    logger.info(f"Output directory: {outdir}")
    
    # Save configuration
    if mesh.comm.rank == 0:
        with open(os.path.join(outdir, 'config.json'), 'w') as g:
            json.dump(cfg, g, indent=2)
        manifest = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version,
            "mpi_size": mesh.comm.size,
        }
        with open(os.path.join(outdir, 'manifest.json'), 'w') as g:
            json.dump(manifest, g, indent=2)


    # Initialize solver
    logger.info("Initializing FirstOrderKGSolver...")
    solver = FirstOrderKGSolver(
        mesh=mesh,
        domain_radius=cfg["mesh"]["R"],
        degree=cfg["solver"]["degree"],
        potential_type=cfg["solver"]["potential_type"],
        potential_params=cfg["solver"].get("potential_params", {}),
        cfl_factor=cfg["solver"]["cfl"],
        cfg=cfg
    )
    solver.set_background(alpha=alpha_f, beta=beta_f, gammaInv=gammaInv_f, sqrtg=sqrtg_f, K=K_f)
    
    if cfg["solver"].get("enable_sommerfeld", True):
        if facet_tags is None:
            raise ValueError("enable_sommerfeld=True requires facet_tags from mesh generation")
        outer_tag = get_outer_tag(facet_tags, default=2)
        solver.enable_sommerfeld(facet_tags, outer_tag=cfg["solver"].get("outer_tag", outer_tag))
        logger.info(f"Sommerfeld boundary conditions enabled (tag={outer_tag})")
    
    # Set initial conditions
    ic = cfg["initial_conditions"]
    if ic.get("type") == "gaussian":
        logger.info(f"Setting Gaussian initial conditions: A={ic['A']}, r0={ic['r0']}, w={ic['w']}")
        phi0 = GaussianBump(
            mesh=mesh, 
            V=solver.V_scalar if hasattr(solver, 'V_scalar') else None,
            A=ic["A"], 
            r0=ic["r0"], 
            w=ic["w"], 
            v0=ic["v0"]
        )
        solver.set_initial_conditions(phi0.get_function())
    
    # Setup sampling
    sample_cfg = cfg.get("analysis", {})
    sample_point = np.array(
        sample_cfg.get("sample_point", [ic.get("r0", 0.0), 0.0, 0.0]), 
        dtype=float
    )
    time_series = []
    energy_series = []
    flux_series = []
    
    # Evolution parameters
    t, step = 0.0, 0
    t_end = cfg["evolution"]["t_end"]
    output_every = cfg["evolution"]["output_every"]
    diagnostics = cfg.get("output", {}).get("diagnostics", True)
    
    logger.info(f"Starting evolution: t_end={t_end}, dt={dt:.6e}, output_every={output_every}")
    
    # Main evolution loop
    with dolfinx.io.XDMFFile(mesh.comm, os.path.join(outdir, "phi_evolution.xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)
        
        while t < t_end:
            solver.ssp_rk3_step(dt)
            t += dt
            step += 1
            
            if step % output_every == 0:
                phi, Pi = solver.get_fields()
                xdmf.write_function(phi, t)
                
                # Sample field
                sample_val = sample_phi_at_point(phi, sample_point, mesh)
                if sample_val is not None:
                    time_series.append((t, sample_val))
                
                # Diagnostics
                if diagnostics:
                    energy_series.append((t, solver.energy()))
                    flux_series.append((t, solver.boundary_flux()))
                
                if step % (output_every * 10) == 0:
                    logger.info(f"Progress: t={t:.3f}/{t_end:.3f} (step {step})")
    
    logger.info(f"Evolution complete: final time t={t:.3f}")

    # Save output data
    if time_series and mesh.comm.rank == 0:
        ts_path = os.path.join(outdir, "time_series.txt")
        with open(ts_path, "w") as f:
            for t_val, phi_val in time_series:
                f.write(f"{t_val:.12e} {phi_val:.12e}\n")
        logger.info(f"Time series saved: {len(time_series)} samples")
    
    if diagnostics and energy_series and mesh.comm.rank == 0:
        energy_path = os.path.join(outdir, "energy_series.txt")
        with open(energy_path, "w") as f:
            for t_val, e_val in energy_series:
                f.write(f"{t_val:.12e} {e_val:.12e}\n")
        logger.info(f"Energy series saved: {len(energy_series)} samples")
    
    if diagnostics and flux_series and mesh.comm.rank == 0:
        flux_path = os.path.join(outdir, "flux_series.txt")
        with open(flux_path, "w") as f:
            for t_val, f_val in flux_series:
                f.write(f"{t_val:.12e} {f_val:.12e}\n")
        logger.info(f"Flux series saved: {len(flux_series)} samples")
    
    # QNM analysis
    if cfg.get("output", {}).get("qnm_analysis", False) and len(time_series) > 8:
        logger.info("Performing QNM analysis...")
        dt_sample = time_series[1][0] - time_series[0][0]
        signal = [v for _, v in time_series]
        qnm_method = cfg.get("analysis", {}).get("qnm_method", "fft").lower()
        
        if qnm_method == "prony":
            from psyop.analysis.qnm import estimate_qnm_prony
            modes = int(cfg.get("analysis", {}).get("qnm_modes", 1))
            prony_results = estimate_qnm_prony(signal, dt_sample, modes=modes)
            if prony_results:
                np.savetxt(
                    os.path.join(outdir, "qnm_prony.txt"),
                    np.array(prony_results),
                    header="freq_real decay_rate", 
                    comments=""
                )
                logger.info(f"QNM Prony analysis saved ({modes} modes)")
        else:
            freqs, spec = compute_qnm(signal, dt_sample)
            f_peak, s_peak = estimate_peak(freqs, spec)
            np.savetxt(
                os.path.join(outdir, "qnm_spectrum.txt"), 
                np.column_stack([freqs, spec])
            )
            with open(os.path.join(outdir, "qnm_peak.txt"), "w") as f:
                f.write(f"{f_peak:.12e} {s_peak:.12e}\n")
            logger.info(f"QNM FFT analysis saved (peak at f={f_peak:.6e})")
    
    logger.info(f"Simulation completed successfully!")
    logger.info(f"Results saved to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
