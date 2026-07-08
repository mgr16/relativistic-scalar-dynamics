import argparse
from copy import deepcopy
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from rsd.analysis.qnm import compute_qnm, estimate_peak, estimate_qnm_prony_modes
from rsd.config import DEFAULT_CONFIG, load_config, validate_config
from rsd.utils.logger import get_logger, setup_logger
from rsd.utils.units import units_from_config


def _write_physical_qnm(units, series_dir, modes=None, f_peak=None) -> None:
    """Escribe series/qnm_physical.json con el QNM en unidades astrofísicas."""
    physical = {"units": units.describe()}
    if modes:
        physical["modes"] = units.qnm_to_physical(modes)
    if f_peak is not None:
        physical["peak_frequency_Hz"] = units.frequency_Hz(f_peak)
    with open(os.path.join(series_dir, "qnm_physical.json"), "w", encoding="utf-8") as f:
        json.dump(physical, f, indent=2)
    logger = get_logger()
    if physical.get("modes"):
        m0 = physical["modes"][0]
        logger.info(
            f"Physical QNM (M={units.M_solar:g} M_sun): "
            f"f={m0['frequency_Hz']:.1f} Hz, tau={m0['damping_time_ms']:.2f} ms"
        )
    elif f_peak is not None:
        logger.info(
            f"Physical QNM (M={units.M_solar:g} M_sun): "
            f"f_peak={physical['peak_frequency_Hz']:.1f} Hz"
        )


def create_example_config(filename: str = "config_example.json") -> None:
    """Create example configuration file."""
    cfg = deepcopy(DEFAULT_CONFIG)
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
    parser.add_argument("--output", type=str, default=None, help="Override output directory")
    parser.add_argument(
        "--create-config", action="store_true", help="Create example configuration file"
    )
    parser.add_argument("--test", action="store_true", help="Run basic test without FEM")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Show live PyVista view of phi during evolution (serial only, demo/debug)",
    )
    parser.add_argument(
        "--live-every",
        type=int,
        default=None,
        metavar="N",
        help="Update live view every N steps (default: evolution.output_every)",
    )
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
            description="RSD: Scalar field simulation with Sommerfeld boundary conditions"
        )
    )
    args = run_parser.parse_args(argv)
    if args.live_every is not None and args.live_every < 1:
        run_parser.error("--live-every must be >= 1")

    # DOLFINx imports (required)
    try:
        import dolfinx
        import dolfinx.io
        from mpi4py import MPI
    except ImportError as e:
        sys.stderr.write("ERROR: DOLFINx is required but not installed.\n")
        sys.stderr.write("Install with: conda install -c conda-forge fenics-dolfinx\n")
        raise ImportError("DOLFINx is required to run RSD") from e

    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logger("rsd", level=log_level)

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
    from rsd.mesh.gmsh import INNER_BOUNDARY_TAG, build_ball_mesh, get_outer_tag
    from rsd.physics.initial_conditions import GaussianBump
    from rsd.physics.metrics import make_background
    from rsd.solvers.first_order import FirstOrderKGSolver
    from rsd.utils.utils import compute_dt_cfl

    # Load configuration
    if args.config:
        if not os.path.exists(args.config):
            # Fallar fuerte: caer en silencio a otra config es peor que abortar
            logger.error(f"Config file not found: {args.config}")
            return 2
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

    cfg = validate_config(cfg)
    if args.output is not None:
        cfg["output"]["dir"] = args.output
    logger.info("Configuration validated successfully")

    # Build mesh and metric
    r_inner = float(cfg["mesh"].get("r_inner", 0.0) or 0.0)
    lc_inner = cfg["mesh"].get("lc_inner")
    logger.info(
        f"Building mesh: R={cfg['mesh']['R']}, lc={cfg['mesh']['lc']}, "
        f"r_inner={r_inner}, lc_inner={lc_inner}"
    )
    mesh, cell_tags, facet_tags = build_ball_mesh(
        R=cfg["mesh"]["R"], lc=cfg["mesh"]["lc"], comm=MPI.COMM_WORLD,
        r_inner=r_inner, lc_inner=lc_inner,
        geom_order=int(cfg["mesh"].get("geom_order", 1)),
    )
    logger.info(f"Mesh created with {mesh.topology.index_map(mesh.topology.dim).size_local} cells")

    bg = make_background(cfg["metric"])
    alpha_f, beta_f, gammaInv_f, sqrtg_f, K_f = bg.build(mesh)
    c_max = getattr(bg, "max_characteristic_speed", lambda m: 1.0)(mesh)
    logger.info(f"Background metric set: {cfg['metric']['type']}")

    # Setup output directory
    dt = compute_dt_cfl(
        mesh, cfl=cfg["solver"]["cfl"], c_max=c_max, degree=cfg["solver"]["degree"]
    )
    logger.info(f"CFL timestep: dt = {dt:.6e}")

    outdir = os.path.join(cfg["output"]["dir"], time.strftime("run_%Y%m%d_%H%M%S"))
    series_dir = os.path.join(outdir, "series")
    fields_dir = os.path.join(outdir, "fields")
    plots_dir = os.path.join(outdir, "plots")
    if mesh.comm.rank == 0:
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(series_dir, exist_ok=True)
        os.makedirs(fields_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
    # Todos los ranks deben esperar a que existan los directorios antes de
    # abrir colectivamente el XDMF (evita carrera en arranque paralelo)
    mesh.comm.barrier()
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
    # rebuild=False: los operadores se reconstruyen una sola vez al final
    solver.set_background(
        alpha=alpha_f, beta=beta_f, gammaInv=gammaInv_f, sqrtg=sqrtg_f, K=K_f, rebuild=False
    )

    if cfg["solver"].get("enable_sommerfeld", True):
        if facet_tags is None:
            raise ValueError("enable_sommerfeld=True requires facet_tags from mesh generation")
        outer_tag = int(cfg["solver"].get("outer_tag", get_outer_tag(facet_tags, default=2)))
        solver.enable_sommerfeld(facet_tags, outer_tag=outer_tag, rebuild=False)
        logger.info(f"Sommerfeld boundary conditions enabled (tag={outer_tag})")

    if r_inner > 0:
        if facet_tags is None:
            raise ValueError("Excision (mesh.r_inner > 0) requires facet_tags from mesh generation")
        solver.enable_excision(facet_tags, inner_tag=INNER_BOUNDARY_TAG, rebuild=False)
        logger.info(f"Excision inner boundary enabled (tag={INNER_BOUNDARY_TAG}, r={r_inner})")

    solver.rebuild_operators()

    # Set initial conditions
    ic = cfg["initial_conditions"]
    if ic.get("type") == "gaussian":
        direction = str(ic.get("direction", "static")).lower()
        logger.info(
            f"Setting Gaussian initial conditions: A={ic['A']}, r0={ic['r0']}, "
            f"w={ic['w']}, direction={direction}"
        )
        phi0 = GaussianBump(
            mesh=mesh,
            V=solver.V_scalar if hasattr(solver, "V_scalar") else None,
            A=ic["A"],
            r0=ic["r0"],
            w=ic["w"],
            v0=ic["v0"],
            direction=direction,
            background=bg,
        )
        solver.set_initial_conditions(phi0.get_function(), phi0.get_momentum())

    # Setup sampling
    sample_cfg = cfg.get("analysis", {})
    sample_point = np.array(
        sample_cfg.get("sample_point", [ic.get("r0", 0.0), 0.0, 0.0]), dtype=float
    )
    time_series = []
    energy_series = []
    flux_series = []
    killing_series = []

    # Extracción multipolar (opcional): proyección sobre Y_lm en una esfera
    extractor = None
    multipole_series = []
    extraction_cfg = sample_cfg.get("extraction", {}) or {}
    if extraction_cfg.get("enabled", False):
        from rsd.analysis.extraction import MultipoleExtractor

        ext_radius = float(extraction_cfg.get("radius", 0.6 * float(cfg["mesh"]["R"])))
        ext_lmax = int(extraction_cfg.get("lmax", 2))
        extractor = MultipoleExtractor(mesh, radius=ext_radius, lmax=ext_lmax)
        logger.info(f"Multipole extraction enabled: R_ext={ext_radius}, lmax={ext_lmax}")

    # Evolution parameters
    t, step = 0.0, 0
    t_end = cfg["evolution"]["t_end"]
    output_every = cfg["evolution"]["output_every"]
    diagnostics = cfg.get("output", {}).get("diagnostics", True)
    save_series = cfg.get("output", {}).get("save_series", True)

    # Monitor de validez de Cowling: cuantifica la hipótesis de campo de prueba
    cowling_monitor = None
    cowling_series = []
    if diagnostics:
        from rsd.analysis.cowling import CowlingMonitor

        cowling_monitor = CowlingMonitor(solver, cfg["metric"])

    # Visualización en vivo (opcional): cero costo cuando --live no está activo
    live_viewer = None
    live_every = output_every
    if args.live:
        from rsd.utils.live_view import create_live_viewer

        live_every = args.live_every if args.live_every is not None else output_every
        live_viewer = create_live_viewer(solver.V_phi, comm=mesh.comm)
        if live_viewer is not None:
            # Primer frame con el estado inicial: calibra la barra de color
            phi_init, _ = solver.get_fields()
            live_viewer.update(phi_init, t=0.0)
            logger.info(f"Live view enabled: updating every {live_every} steps")

    logger.info(f"Starting evolution: t_end={t_end}, dt={dt:.6e}, output_every={output_every}")

    # Main evolution loop
    with dolfinx.io.XDMFFile(
        mesh.comm, os.path.join(fields_dir, "phi_evolution.xdmf"), "w"
    ) as xdmf:
        xdmf.write_mesh(mesh)

        while t < t_end:
            step_dt = min(dt, t_end - t)
            solver.ssp_rk3_step(step_dt)
            t += step_dt
            step += 1

            if step % output_every == 0:
                phi, Pi = solver.get_fields()
                xdmf.write_function(phi, t)

                if live_viewer is not None and step % live_every == 0:
                    live_viewer.update(phi, t)

                # Sample field: el punto vive en (a lo sumo) unos pocos ranks;
                # se recolecta en rank 0, que es quien escribe las series
                sample_val = sample_phi_at_point(phi, sample_point, mesh)
                gathered = mesh.comm.gather(sample_val, root=0)
                if mesh.comm.rank == 0 and gathered is not None:
                    found = [v for v in gathered if v is not None]
                    if found:
                        time_series.append((t, found[0]))

                # Multipoles (la extracción ya es global vía MPI)
                if extractor is not None:
                    coeffs = extractor.extract(phi)
                    multipole_series.append(
                        (t, [coeffs[mode] for mode in extractor.modes])
                    )

                # Diagnostics
                if diagnostics:
                    e_now = solver.energy()
                    energy_series.append((t, e_now))
                    # flujo total que abandona el dominio: borde exterior
                    # (radiación) + borde interior (absorción del agujero)
                    flux_series.append(
                        (t, solver.boundary_flux(), solver.inner_flux())
                    )
                    killing_series.append(
                        (t, solver.energy_killing(), solver.killing_flux(),
                         solver.killing_inner_flux())
                    )
                    cw = cowling_monitor.check(t, energy=e_now)
                    cowling_series.append((t, cw["zeta_max"], cw["energy_ratio"]))

                if step % (output_every * 10) == 0:
                    logger.info(f"Progress: t={t:.3f}/{t_end:.3f} (step {step})")

            elif live_viewer is not None and step % live_every == 0:
                # Paso solo de visualización (live_every != output_every)
                phi_live, _ = solver.get_fields()
                live_viewer.update(phi_live, t)

    if live_viewer is not None:
        live_viewer.close()

    logger.info(f"Evolution complete: final time t={t:.3f}")

    # Save output data
    if save_series and time_series and mesh.comm.rank == 0:
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

    if save_series and diagnostics and energy_series and mesh.comm.rank == 0:
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

    if save_series and diagnostics and flux_series and mesh.comm.rank == 0:
        flux_csv_path = os.path.join(series_dir, "flux.csv")
        flux_path = os.path.join(outdir, "flux_series.txt")  # compatibility legacy
        with open(flux_csv_path, "w", encoding="utf-8") as fcsv:
            fcsv.write("t,flux,flux_inner\n")
            for t_val, f_out, f_in in flux_series:
                fcsv.write(f"{t_val:.12e},{f_out:.12e},{f_in:.12e}\n")
        with open(flux_path, "w", encoding="utf-8") as f:
            for t_val, f_out, _ in flux_series:
                f.write(f"{t_val:.12e} {f_out:.12e}\n")
        logger.info(f"Flux series saved: {len(flux_series)} samples")

    if save_series and diagnostics and cowling_series and mesh.comm.rank == 0:
        with open(os.path.join(series_dir, "cowling.csv"), "w", encoding="utf-8") as fcsv:
            fcsv.write("t,zeta_max,energy_ratio\n")
            for t_val, z_val, e_val in cowling_series:
                fcsv.write(f"{t_val:.12e},{z_val:.12e},{e_val:.12e}\n")
        zeta_peak = max(z for _, z, _ in cowling_series)
        logger.info(
            f"Cowling validity: max zeta = {zeta_peak:.3e} "
            f"({'OK, test-field consistent' if zeta_peak < 1e-2 else 'MARGINAL'})"
        )

    # Balance de energía de Killing: en fondos estacionarios cierra con
    # flujos puros de superficie y el residual converge a 0 con la
    # resolución (docs/math/killing_energy.md) — es el balance de
    # referencia en runs con excisión.
    if save_series and diagnostics and len(killing_series) >= 2 and mesh.comm.rank == 0:
        tk_arr = np.array([row[0] for row in killing_series])
        ek_arr = np.array([row[1] for row in killing_series])
        fk_arr = np.array([row[2] + row[3] for row in killing_series])
        cum_k = np.concatenate(
            [[0.0], np.cumsum(0.5 * (fk_arr[1:] + fk_arr[:-1]) * np.diff(tk_arr))]
        )
        res_k = ek_arr + cum_k - ek_arr[0]
        with open(os.path.join(series_dir, "killing.csv"), "w", encoding="utf-8") as fcsv:
            fcsv.write("t,energy,flux,flux_inner,integrated_flux,balance_residual\n")
            for i, (t_val, e_val, fo_val, fi_val) in enumerate(killing_series):
                fcsv.write(
                    f"{t_val:.12e},{e_val:.12e},{fo_val:.12e},{fi_val:.12e},"
                    f"{cum_k[i]:.12e},{res_k[i]:.12e}\n"
                )
        scale_k = max(abs(ek_arr[0]), float(np.max(np.abs(cum_k))), 1e-300)
        logger.info(
            "Killing balance residual |E_K(t)+∫F_K−E_K(0)| / scale = "
            f"{abs(res_k[-1]) / scale_k:.3e}"
        )

    if save_series and multipole_series and mesh.comm.rank == 0:
        mp_csv_path = os.path.join(series_dir, "multipoles.csv")
        with open(mp_csv_path, "w", encoding="utf-8") as fcsv:
            fcsv.write("t," + ",".join(extractor.header()) + "\n")
            for t_val, coeff_row in multipole_series:
                row = ",".join(f"{c:.12e}" for c in coeff_row)
                fcsv.write(f"{t_val:.12e},{row}\n")
        logger.info(f"Multipole series saved: {len(multipole_series)} samples")

    # Balance de energía: E(t) + ∫F dt - E(0) ≈ 0 detecta inconsistencias
    # de discretización o de la BC (con flujo saliente positivo)
    if save_series and diagnostics and len(energy_series) >= 2 and mesh.comm.rank == 0:
        t_arr = np.array([t_val for t_val, _ in energy_series])
        e_arr = np.array([e_val for _, e_val in energy_series])
        # flujo total fuera del dominio: exterior (radiación) + interior
        # (absorción del agujero negro, exacta desde v3.1)
        f_arr = (
            np.array([f_out + f_in for _, f_out, f_in in flux_series])
            if flux_series else np.zeros_like(e_arr)
        )
        cum_flux = np.concatenate([[0.0], np.cumsum(0.5 * (f_arr[1:] + f_arr[:-1]) * np.diff(t_arr))])
        residual = e_arr + cum_flux - e_arr[0]
        with open(os.path.join(series_dir, "balance.csv"), "w", encoding="utf-8") as fcsv:
            fcsv.write("t,energy,integrated_flux,balance_residual\n")
            for i in range(len(t_arr)):
                fcsv.write(
                    f"{t_arr[i]:.12e},{e_arr[i]:.12e},{cum_flux[i]:.12e},{residual[i]:.12e}\n"
                )
        rel_residual = abs(residual[-1]) / max(e_arr[0], 1e-300)
        logger.info(f"Energy balance residual |E(t)+∫F-E0|/E0 = {rel_residual:.3e}")
        if rel_residual > 0.1:
            logger.warning(
                "Energy balance residual above 10%: check resolution, CFL or BC "
                "(a sponge layer breaks this balance by design; on stationary "
                "slicings (Kerr-Schild) the K/beta volume terms contribute "
                "O(field^2) — see docs/math/energy_stability.md)"
            )

    # QNM analysis (solo rank 0: es quien posee time_series y escribe archivos)
    if (
        mesh.comm.rank == 0
        and cfg.get("output", {}).get("qnm_analysis", False)
        and len(time_series) > 8
    ):
        logger.info("Performing QNM analysis...")
        dt_sample = time_series[1][0] - time_series[0][0]
        signal = [v for _, v in time_series]
        qnm_method = cfg.get("analysis", {}).get("qnm_method", "fft").lower()

        if qnm_method == "prony":
            from rsd.analysis.qnm import estimate_qnm_prony, estimate_qnm_prony_modes

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
                units = units_from_config(cfg)
                if units is not None:
                    _write_physical_qnm(units, series_dir, modes=prony_modes)
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
            units = units_from_config(cfg)
            if units is not None:
                _write_physical_qnm(units, series_dir, f_peak=f_peak)

    logger.info("Simulation completed successfully!")
    logger.info(f"Results saved to: {outdir}")
    return 0


def postprocess_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Postprocess RSD run directory")
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

    # Unidades físicas: la config exacta del run queda guardada en el outdir
    units = None
    run_cfg_path = os.path.join(args.run, "config.json")
    if os.path.exists(run_cfg_path):
        with open(run_cfg_path, encoding="utf-8") as f:
            units = units_from_config(json.load(f))

    if args.method == "prony":
        prony_modes = estimate_qnm_prony_modes(signal, dt, modes=args.modes)
        with open(os.path.join(series_dir, "qnm_modes.json"), "w", encoding="utf-8") as f:
            json.dump(prony_modes, f, indent=2)
        if units is not None and prony_modes:
            _write_physical_qnm(units, series_dir, modes=prony_modes)
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
        if units is not None:
            _write_physical_qnm(units, series_dir, f_peak=f_peak)
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
            except Exception as exc:
                get_logger(__name__).warning(f"Could not write QNM plot: {exc}")
    return 0


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] not in {"run", "postprocess"}:
        return run_main(args)

    parser = argparse.ArgumentParser(prog="rsd", description="RSD command line interface")
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
