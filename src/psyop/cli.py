import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from psyop.analysis.qnm import compute_qnm, estimate_peak, estimate_qnm_prony_modes
from psyop.config import load_config, validate_config
from psyop.utils.logger import get_logger, setup_logger


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
        "output": {"dir": "results", "qnm_analysis": True},
    }
    logger = get_logger()
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    logger.info(f"Example configuration created: {filename}")


def sample_phi_at_point(phi_fn, point: np.ndarray, mesh) -> Optional[float]:
    """Sample field value at a specific point."""
    import dolfinx.geometry

    point_reshaped = point.reshape(1, 3)
    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, point_reshaped)
    cells = dolfinx.geometry.compute_colliding_cells(mesh, candidates, point_reshaped)

    if len(cells.links(0)) == 0:
        return None

    cell = cells.links(0)[0]
    val = phi_fn.eval(point_reshaped[0], cell)
    return float(val[0]) if np.ndim(val) > 0 else float(val)


def _build_run_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=str, default=None, help="Configuration JSON file")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument(
        "--create-config", action="store_true", help="Create example configuration file"
    )
    parser.add_argument("--test", action="store_true", help="Run basic test without FEM")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser


def run_main(argv=None) -> int:
    run_parser = _build_run_parser(
        argparse.ArgumentParser(
            description="PSYOP: Scalar field simulation with Sommerfeld boundary conditions"
        )
    )
    args = run_parser.parse_args(argv)

    # DOLFINx imports (required)
    try:
        import dolfinx
        import dolfinx.io
        from mpi4py import MPI
    except ImportError as e:
        print("ERROR: DOLFINx is required but not installed.")
        print("Install with: conda install -c conda-forge dolfinx")
        raise ImportError("DOLFINx is required to run PSYOP") from e

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
    from psyop.analysis.qnm import compute_qnm, estimate_peak
    from psyop.mesh.gmsh import build_ball_mesh, get_outer_tag
    from psyop.physics.initial_conditions import GaussianBump
    from psyop.physics.metrics import make_background
    from psyop.solvers.first_order import FirstOrderKGSolver
    from psyop.utils.utils import compute_dt_cfl

    # Load configuration
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    else:
        config_file = "config_example.json"
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
        R=cfg["mesh"]["R"], lc=cfg["mesh"]["lc"], comm=MPI.COMM_WORLD
    )
    logger.info(f"Mesh created with {mesh.topology.index_map(mesh.topology.dim).size_local} cells")

    bg = make_background(cfg["metric"])
    alpha_f, beta_f, gammaInv_f, sqrtg_f, K_f = bg.build(mesh)
    c_max = getattr(bg, "max_characteristic_speed", lambda m: 1.0)(mesh)
    logger.info(f"Background metric set: {cfg['metric']['type']}")

    # Setup output directory
    dt = compute_dt_cfl(mesh, cfl=cfg["solver"]["cfl"], c_max=c_max)
    logger.info(f"CFL timestep: dt = {dt:.6e}")

    outdir = os.path.join(
        cfg.get("output", {}).get("dir", args.output), time.strftime("run_%Y%m%d_%H%M%S")
    )
    series_dir = os.path.join(outdir, "series")
    fields_dir = os.path.join(outdir, "fields")
    plots_dir = os.path.join(outdir, "plots")
    if mesh.comm.rank == 0:
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(series_dir, exist_ok=True)
        os.makedirs(fields_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"Output directory: {outdir}")

    # Save configuration
    if mesh.comm.rank == 0:
        with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as g:
            json.dump(cfg, g, indent=2)
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=10
            ).strip()
        except Exception:
            git_commit = "unknown"
        try:
            host_name = os.uname().nodename if hasattr(os, "uname") else "unknown"
        except (AttributeError, OSError):
            host_name = "unknown"
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "python": sys.version,
            "mpi_size": mesh.comm.size,
            "dolfinx": getattr(dolfinx, "__version__", "unknown"),
            "petsc4py": getattr(__import__("petsc4py"), "__version__", "unknown"),
            "host": host_name,
        }
        with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as g:
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
        cfg=cfg,
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
            V=solver.V_scalar if hasattr(solver, "V_scalar") else None,
            A=ic["A"],
            r0=ic["r0"],
            w=ic["w"],
            v0=ic["v0"],
        )
        solver.set_initial_conditions(phi0.get_function())

    # Setup sampling
    sample_cfg = cfg.get("analysis", {})
    sample_point = np.array(
        sample_cfg.get("sample_point", [ic.get("r0", 0.0), 0.0, 0.0]), dtype=float
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
    with dolfinx.io.XDMFFile(
        mesh.comm, os.path.join(fields_dir, "phi_evolution.xdmf"), "w"
    ) as xdmf:
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
        ts_csv_path = os.path.join(series_dir, "time_series.csv")
        ts_path = os.path.join(outdir, "time_series.txt")  # compatibility legacy
        with open(ts_csv_path, "w", encoding="utf-8") as fcsv:
            fcsv.write("t,phi\n")
            for t_val, phi_val in time_series:
                fcsv.write(f"{t_val:.12e},{phi_val:.12e}\n")
        with open(ts_path, "w", encoding="utf-8") as f:
            for t_val, phi_val in time_series:
                f.write(f"{t_val:.12e} {phi_val:.12e}\n")
        logger.info(f"Time series saved: {len(time_series)} samples")

    if diagnostics and energy_series and mesh.comm.rank == 0:
        energy_csv_path = os.path.join(series_dir, "energy.csv")
        energy_path = os.path.join(outdir, "energy_series.txt")  # compatibility legacy
        with open(energy_csv_path, "w", encoding="utf-8") as fcsv:
            fcsv.write("t,energy\n")
            for t_val, e_val in energy_series:
                fcsv.write(f"{t_val:.12e},{e_val:.12e}\n")
        with open(energy_path, "w", encoding="utf-8") as f:
            for t_val, e_val in energy_series:
                f.write(f"{t_val:.12e} {e_val:.12e}\n")
        logger.info(f"Energy series saved: {len(energy_series)} samples")

    if diagnostics and flux_series and mesh.comm.rank == 0:
        flux_csv_path = os.path.join(series_dir, "flux.csv")
        flux_path = os.path.join(outdir, "flux_series.txt")  # compatibility legacy
        with open(flux_csv_path, "w", encoding="utf-8") as fcsv:
            fcsv.write("t,flux\n")
            for t_val, f_val in flux_series:
                fcsv.write(f"{t_val:.12e},{f_val:.12e}\n")
        with open(flux_path, "w", encoding="utf-8") as f:
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
            from psyop.analysis.qnm import estimate_qnm_prony, estimate_qnm_prony_modes

            modes = int(cfg.get("analysis", {}).get("qnm_modes", 1))
            prony_results = estimate_qnm_prony(signal, dt_sample, modes=modes)
            prony_modes = estimate_qnm_prony_modes(signal, dt_sample, modes=modes)
            if prony_results:
                np.savetxt(
                    os.path.join(series_dir, "qnm_prony.csv"),
                    np.array(prony_results),
                    header="freq_real,decay_rate",
                    comments="",
                    delimiter=",",
                )
                with open(os.path.join(series_dir, "qnm_modes.json"), "w", encoding="utf-8") as f:
                    json.dump(prony_modes, f, indent=2)
                logger.info(f"QNM Prony analysis saved ({modes} modes)")
        else:
            freqs, spec = compute_qnm(signal, dt_sample)
            f_peak, s_peak = estimate_peak(freqs, spec)
            np.savetxt(
                os.path.join(series_dir, "qnm_spectrum.csv"),
                np.column_stack([freqs, spec]),
                delimiter=",",
                header="freq,spectrum",
                comments="",
            )
            with open(os.path.join(series_dir, "qnm_peak.json"), "w", encoding="utf-8") as f:
                json.dump({"frequency": f_peak, "spectrum": s_peak}, f, indent=2)
            logger.info(f"QNM FFT analysis saved (peak at f={f_peak:.6e})")

    logger.info("Simulation completed successfully!")
    logger.info(f"Results saved to: {outdir}")
    return 0


def postprocess_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Postprocess PSYOP run directory")
    parser.add_argument("--run", required=True, help="Run directory path")
    parser.add_argument("--qnm", action="store_true", help="Compute QNM spectrum")
    parser.add_argument("--plots", action="store_true", help="Reserved flag for plotting")
    parser.add_argument(
        "--window", default="hann", choices=["hann", "tukey"], help="Window for FFT"
    )
    parser.add_argument("--method", default="fft", choices=["fft", "prony"], help="QNM method")
    parser.add_argument("--modes", type=int, default=1, help="Number of modes for Prony (>=1)")
    args = parser.parse_args(argv)

    if not args.qnm:
        return 0

    series_dir = os.path.join(args.run, "series")
    os.makedirs(series_dir, exist_ok=True)
    plots_dir = os.path.join(args.run, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    ts_csv = os.path.join(series_dir, "time_series.csv")
    ts_txt = os.path.join(args.run, "time_series.txt")
    if os.path.exists(ts_csv):
        data = np.loadtxt(ts_csv, delimiter=",", skiprows=1)
    elif os.path.exists(ts_txt):
        data = np.loadtxt(ts_txt)
    else:
        raise FileNotFoundError(f"time_series not found in {ts_csv} or {ts_txt}")

    if data.ndim != 2 or data.shape[0] < 8 or data.shape[1] < 2:
        raise ValueError("time_series requires at least 8 samples with 2 columns")
    if args.modes < 1:
        raise ValueError("--modes must be >= 1")
    dt = float(np.mean(np.diff(data[:, 0])))
    signal = data[:, 1]
    if args.method == "prony":
        prony_modes = estimate_qnm_prony_modes(signal, dt, modes=args.modes)
        with open(os.path.join(series_dir, "qnm_modes.json"), "w", encoding="utf-8") as f:
            json.dump(prony_modes, f, indent=2)
        if prony_modes:
            rows = [
                [m["frequency"], m["decay"], m["amplitude"], m["phase"], m["score"]]
                for m in prony_modes
            ]
            np.savetxt(
                os.path.join(series_dir, "qnm_modes.csv"),
                np.asarray(rows, dtype=float),
                delimiter=",",
                header="frequency,decay,amplitude,phase,score",
                comments="",
            )
    else:
        freqs, spec = compute_qnm(signal, dt, window=args.window, detrend=True)
        f_peak, s_peak = estimate_peak(freqs, spec)
        np.savetxt(
            os.path.join(series_dir, "qnm_spectrum.csv"),
            np.column_stack([freqs, spec]),
            delimiter=",",
            header="freq,spectrum",
            comments="",
        )
        with open(os.path.join(series_dir, "qnm_peak.json"), "w", encoding="utf-8") as f:
            json.dump({"frequency": f_peak, "spectrum": s_peak}, f, indent=2)
        if args.plots:
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                ax.plot(freqs, spec)
                ax.set_xlabel("frequency")
                ax.set_ylabel("spectrum")
                ax.set_title("QNM spectrum")
                fig.savefig(
                    os.path.join(plots_dir, "qnm_spectrum.png"), dpi=150, bbox_inches="tight"
                )
                plt.close(fig)
            except Exception:
                pass
    return 0


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] not in {"run", "postprocess"}:
        return run_main(args)

    parser = argparse.ArgumentParser(prog="psyop", description="PSYOP command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _build_run_parser(subparsers.add_parser("run", help="Run simulation"))

    post = subparsers.add_parser("postprocess", help="Postprocess simulation outputs")
    post.add_argument("--run", required=True, help="Run directory path")
    post.add_argument("--qnm", action="store_true", help="Compute QNM spectrum")
    post.add_argument("--plots", action="store_true", help="Reserved flag for plotting")
    post.add_argument("--window", default="hann", choices=["hann", "tukey"], help="Window for FFT")
    post.add_argument("--method", default="fft", choices=["fft", "prony"], help="QNM method")
    post.add_argument("--modes", type=int, default=1, help="Number of modes for Prony (>=1)")

    ns = parser.parse_args(args)
    if ns.command == "run":
        return run_main(args[1:])
    return postprocess_main(args[1:])
