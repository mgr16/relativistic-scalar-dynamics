#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 0 — piloto del sector esférico de la hipótesis H2 (oráculo 1D).

Produce datos y figuras en docs/research/phase0/:

1. interior_lineal: asintótica del campo lineal hacia r→0 sobre
   Schwarzschild-KS. Medimos el log-slope s(r,t) = r·∂_r u: si u ~ A·ln r + B
   cerca de la singularidad, s(r→0) tiende a una constante A ≠ 0.
2. dominacion_cinetica: razón R_pot(r) = |αV'(u)|_RMS / |términos cinéticos|_RMS
   para quadratic/higgs/mexican_hat y varias amplitudes; radio de crossover
   r* donde R_pot cae por debajo de 0.01.
3. cowling_interior: perfil ζ(r) = 8πρ/√Kretschmann hacia r→0 (¿mejora la
   aproximación de campo de prueba cerca de la singularidad?).
4. junk_inicial: comparación del transitorio espurio saliente entre la
   relación de momento plana ("ingoing") y la consistente con Kerr-Schild
   ("ingoing_curved") — insumo para la mejora de datos iniciales 3D.

Uso:  python scripts/pilot_phase0_oracle.py [--fast]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from rsd.reference import SphericalOracle1D  # noqa: E402

OUT = REPO / "docs" / "research" / "phase0"
FIG = OUT / "figures"
DATA = OUT / "data"

SQRT_KRETSCHMANN_COEF = 4.0 * np.sqrt(3.0)  # √(K_abcd K^abcd) = 4√3 M/r³


def _figure_backend():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def run_interior(
    l: int = 0,
    potential_type: str = "zero",
    potential_params: dict | None = None,
    A: float = 1e-3,
    u_infinity: float = 0.0,
    r_min: float = 0.02,
    n_points: int = 1600,
    t_end: float = 40.0,
    n_snapshots: int = 40,
):
    """Evolución interior estándar del piloto: pulso entrante desde r0=5."""
    oracle = SphericalOracle1D(
        M=1.0, l=l, r_min=r_min, r_max=60.0, n_points=n_points, grid="log",
        potential_type=potential_type, potential_params=potential_params,
        u_infinity=u_infinity, ko_eps=0.02,
    )
    oracle.set_initial_gaussian(A=A, r0=5.0, width=1.0, direction="ingoing_curved")
    dt = oracle.compute_dt()
    n_steps = int(np.ceil(t_end / dt))
    snap_every = max(1, n_steps // n_snapshots)
    t0 = time.perf_counter()
    out = oracle.evolve(
        t_end=t_end,
        probe_radii=[2.0 * r_min, 0.1, 0.5],
        output_every=200,
        snapshot_every=snap_every,
    )
    wall = time.perf_counter() - t0
    return oracle, out, wall


def study_linear_asymptotics(fast: bool) -> dict:
    """¿Crece el campo lineal logarítmicamente hacia r→0?"""
    plt = _figure_backend()
    summary = {}
    r_min = 0.02 if fast else 0.01
    n_points = 1600 if fast else 2600
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for col, l in enumerate((0, 1)):
        oracle, out, wall = run_interior(l=l, r_min=r_min, n_points=n_points)
        print(f"  lineal l={l}: {wall:.1f} s de pared", flush=True)

        ax = axes[0][col]
        n_snap = len(out.snapshot_ts)
        for k in range(0, n_snap, max(1, n_snap // 6)):
            ax.plot(oracle.r, out.snapshots_u[k] / 1e-3,
                    label=f"t={out.snapshot_ts[k]:.0f}M")
        ax.set_xscale("log")
        ax.set_xlim(r_min, 10)
        ax.set_xlabel("r/M")
        ax.set_ylabel("u / A")
        ax.set_title(f"perfil interior, l={l} (lineal)")
        ax.axvline(2.0, color="k", ls=":", lw=0.8)
        ax.legend(fontsize=7)

        # log-slope s(r,t) = r ∂_r u en el último snapshot y promedio final
        ax = axes[1][col]
        mask = oracle.r < 2.0
        slopes_final = []
        for k in range(max(0, n_snap - 5), n_snap):
            u_snap, Pi_snap = out.snapshots_u[k], out.snapshots_Pi[k]
            s = oracle.r * oracle._deriv_r(u_snap)
            slopes_final.append(s)
            ax.plot(oracle.r[mask], s[mask] / 1e-3, alpha=0.5, lw=0.8)
        s_mean = np.mean(slopes_final, axis=0)
        ax.plot(oracle.r[mask], s_mean[mask] / 1e-3, "k", lw=1.8, label="media últimos 5")
        ax.set_xscale("log")
        ax.set_xlabel("r/M")
        ax.set_ylabel("(r ∂_r u) / A")
        ax.set_title(f"log-slope, l={l}: constante ⇔ u ~ A·ln r")
        ax.legend(fontsize=7)

        # métrica cuantitativa: variación relativa del log-slope por década
        # en la zona profunda (r < 0.2) — ~0 si el perfil es logarítmico
        deep = (oracle.r > r_min * 1.5) & (oracle.r < 0.2)
        s_deep = s_mean[deep]
        flatness = float(np.std(s_deep) / (np.abs(np.mean(s_deep)) + 1e-30))
        summary[f"l{l}"] = {
            "log_slope_mean_over_A": float(np.mean(s_deep) / 1e-3),
            "log_slope_flatness": flatness,
            "u_at_rmin_over_A": float(out.snapshots_u[-1][0] / 1e-3),
            "wall_seconds": wall,
        }
        np.savez_compressed(
            DATA / f"linear_interior_l{l}.npz",
            r=oracle.r, snapshot_ts=np.asarray(out.snapshot_ts),
            snapshots_u=np.asarray(out.snapshots_u),
            snapshots_Pi=np.asarray(out.snapshots_Pi),
        )

    fig.tight_layout()
    fig.savefig(FIG / "linear_interior_asymptotics.png", dpi=150)
    plt.close(fig)
    return summary


def study_kinetic_domination(fast: bool) -> dict:
    """R_pot(r) y crossover r* para los tres potenciales y varias amplitudes."""
    plt = _figure_backend()
    cases = [
        ("quadratic", {"m_squared": 1.0}, 0.0),
        ("higgs", {"m_squared": 1.0, "lambda_coupling": 0.1}, 0.0),
        ("mexican_hat", {"lambda_coupling": 0.1, "vacuum_value": 1.0}, 1.0),
    ]
    amplitudes = [1e-3, 1e-2, 1e-1]
    n_points = 1200 if fast else 1600
    threshold = 0.01

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    summary = {}
    for ax, (ptype, params, u_inf) in zip(axes, cases):
        summary[ptype] = {}
        for A in amplitudes:
            oracle, out, wall = run_interior(
                l=0, potential_type=ptype, potential_params=params,
                A=A, u_infinity=u_inf, n_points=n_points,
            )
            # RMS de cada término sobre el último tercio de snapshots
            n_snap = len(out.snapshot_ts)
            ks = range(2 * n_snap // 3, n_snap)
            acc = None
            for k in ks:
                terms = oracle.rhs_term_breakdown(
                    out.snapshots_u[k], out.snapshots_Pi[k]
                )
                sq = {name: t**2 for name, t in terms.items()}
                acc = sq if acc is None else {
                    name: acc[name] + sq[name] for name in sq
                }
            rms = {name: np.sqrt(v / len(list(ks))) for name, v in acc.items()}
            kinetic = (
                rms["flux_div"] + rms["transport"] + rms["extrinsic"] + rms["angular"]
            )
            R_pot = rms["potential"] / (kinetic + 1e-300)

            ax.plot(oracle.r, R_pot, label=f"A={A:g}")
            # crossover: mayor radio por debajo del cual R_pot < threshold
            below = R_pot < threshold
            r_star = None
            for i in range(len(below) - 1, -1, -1):
                if not below[i]:
                    r_star = float(oracle.r[i + 1]) if i + 1 < len(below) else None
                    break
            else:
                r_star = float(oracle.r[0])
            summary[ptype][f"A={A:g}"] = {
                "r_star": r_star,
                "R_pot_at_rmin": float(R_pot[0]),
                "R_pot_max_inside_horizon": float(np.max(R_pot[oracle.r < 2.0])),
                "wall_seconds": wall,
            }
            np.savez_compressed(
                DATA / f"kinetic_{ptype}_A{A:g}.npz",
                r=oracle.r, R_pot=R_pot, **rms,
            )
            print(f"  {ptype} A={A:g}: r*={r_star}  R_pot(r_min)={R_pot[0]:.2e}"
                  f"  [{wall:.0f} s]", flush=True)
        ax.axhline(threshold, color="k", ls=":", lw=0.8)
        ax.axvline(2.0, color="k", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("r/M")
        ax.set_title(ptype)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("R_pot = |αV'| / |cinéticos| (RMS)")
    fig.suptitle("Dominación cinética hacia la singularidad (l=0, t∈[27,40]M)")
    fig.tight_layout()
    fig.savefig(FIG / "kinetic_domination.png", dpi=150)
    plt.close(fig)
    return summary


def study_cowling_interior(fast: bool) -> dict:
    """ζ(r) = 8πρ/√Kretschmann en el interior para el caso más exigente."""
    plt = _figure_backend()
    oracle, out, _ = run_interior(
        l=0, potential_type="higgs",
        potential_params={"m_squared": 1.0, "lambda_coupling": 0.1},
        A=1e-1, n_points=1200 if fast else 1600,
    )
    n_snap = len(out.snapshot_ts)
    zetas = []
    for k in range(2 * n_snap // 3, n_snap):
        rho = oracle.energy_density(out.snapshots_u[k], out.snapshots_Pi[k])
        sqrt_kret = SQRT_KRETSCHMANN_COEF * 1.0 / oracle.r**3
        zetas.append(8.0 * np.pi * rho / sqrt_kret)
    zeta = np.mean(zetas, axis=0)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(oracle.r, zeta)
    ax.axvline(2.0, color="k", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("r/M")
    ax.set_ylabel("ζ = 8πρ / √Kretschmann")
    ax.set_title("Validez de Cowling hacia la singularidad (higgs, A=0.1)")
    fig.tight_layout()
    fig.savefig(FIG / "cowling_interior.png", dpi=150)
    plt.close(fig)

    inside = oracle.r < 2.0
    return {
        "zeta_at_rmin": float(zeta[0]),
        "zeta_max_inside_horizon": float(np.max(zeta[inside])),
        "zeta_max_global": float(np.max(zeta)),
        "r_of_zeta_max": float(oracle.r[int(np.argmax(zeta))]),
    }


def study_initial_data_junk(fast: bool) -> dict:
    """Transitorio espurio saliente: relación plana vs consistente con KS."""
    plt = _figure_backend()
    n_points = 3000 if fast else 5000
    results = {}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for direction in ("ingoing", "ingoing_curved"):
        oracle = SphericalOracle1D(
            M=1.0, l=1, r_min=1.0, r_max=120.0, n_points=n_points,
            grid="uniform", ko_eps=0.02,
        )
        oracle.set_initial_gaussian(A=1e-3, r0=12.0, width=2.0, direction=direction)
        out = oracle.evolve(t_end=40.0, probe_radii=[30.0], output_every=5)
        ts, sig = out.ts, out.probes[30.0]
        # ventana de junk puro: lo que pasa por r=30 ANTES de que pueda
        # llegar el backscatter físico del pulso (que va hacia adentro)
        mask = (ts > 5.0) & (ts < 25.0)
        junk = float(np.max(np.abs(sig[mask])))
        results[direction] = junk
        ax.semilogy(ts, np.abs(sig) + 1e-16, label=direction)
    ax.axvspan(5, 25, alpha=0.1, color="gray")
    ax.set_xlabel("t/M")
    ax.set_ylabel("|u(r=30, t)|")
    ax.set_title("Radiación espuria saliente del dato inicial (l=1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "initial_data_junk.png", dpi=150)
    plt.close(fig)
    results["junk_suppression_factor"] = results["ingoing"] / max(
        results["ingoing_curved"], 1e-300
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="resoluciones reducidas (humo, ~5 min)")
    args = parser.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    summary = {"fast_mode": bool(args.fast)}

    print("[1/4] asintótica lineal interior...", flush=True)
    summary["linear_asymptotics"] = study_linear_asymptotics(args.fast)
    print("[2/4] dominación cinética...", flush=True)
    summary["kinetic_domination"] = study_kinetic_domination(args.fast)
    print("[3/4] Cowling interior...", flush=True)
    summary["cowling_interior"] = study_cowling_interior(args.fast)
    print("[4/4] junk de datos iniciales...", flush=True)
    summary["initial_data_junk"] = study_initial_data_junk(args.fast)

    summary["total_wall_seconds"] = time.perf_counter() - t0
    with open(OUT / "pilot_oracle_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nListo en {summary['total_wall_seconds']:.0f} s. "
          f"Figuras en {FIG}, datos en {DATA}.", flush=True)


if __name__ == "__main__":
    main()
