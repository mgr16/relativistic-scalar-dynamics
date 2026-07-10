#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 2 — calibración de la ventana radial del estimador interior a(t).

El diagnóstico interior 3D medirá a(t) (≡ A de Fournodavlos–Sbierski, ver
docs/research/phase2/literature.md) ajustando u contra la jerarquía
logarítmica truncada (rsd.analysis.interior) sobre [r_inner, ~0.5M]. La
ventana 3D es cara hacia adentro (r_inner chico ⇒ celdas diminutas), así
que la pregunta de diseño es: ¿cuánto sesgo de truncamiento tiene cada
ventana alcanzable, y el orden 1 (términos ζ₁·r·ln r + η₁·r) rescata la
ventana barata? El oráculo 1D da la respuesta con una verdad profunda
inalcanzable en 3D.

Protocolo, por dataset:
- verdad: fit orden 1 sobre (0.02, 0.2), con su propia estabilidad
  estimada moviendo la ventana de verdad ((0.015,0.15), (0.03,0.3));
- candidatas: (0.25,0.5), (0.15,0.5), (0.1,0.5), (0.1,0.3) × orden
  {0,1,2};
- sesgo relativo |a_win − a_truth|/|a_truth| en dos niveles de señal
  sobre t ≥ 4M (la fase activa es t ≈ 4–8M; después a(t) decae a una era
  tardía cuasi-estacionaria al 1–4 % del pico): FUERTE = snapshots con
  |a_truth| ≥ 0.3·max (la fase activa, donde vive la medición de H2) y
  MEDIBLE = ≥ 0.05·max (incluye los hombros del decaimiento); además la
  versión normalizada por la escala max|a_truth| sobre todo t ≥ 4.

Datasets:
- linear_interior_l{0,1}.npz del piloto F0 (reuso, n=2600, r_min=0.01);
- l=2 lineal (nueva, la producción 3D usa l=1,2);
- mexican_hat con u∞=v=1, A=0.1 (LA config H2; el piloto no guardó sus
  snapshots), fit sobre u (la constante b absorbe v);
- réplica l=0 lineal a n=1300 (estabilidad del sesgo en resolución).

Uso:  python scripts/interior_window_calibration.py [--force] [--fast]
      (--force re-corre aunque existan los npz; --fast baja resoluciones)
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

from rsd.analysis.interior import fit_log_profile_series  # noqa: E402
from rsd.reference import SphericalOracle1D  # noqa: E402

PHASE0_DATA = REPO / "docs" / "research" / "phase0" / "data"
OUT = REPO / "docs" / "research" / "phase2" / "interior"
DATA = OUT / "data"
FIG = OUT / "figures"

TRUTH_WINDOW = (0.02, 0.2)
TRUTH_SCAN = [(0.015, 0.15), (0.03, 0.3)]
CANDIDATES = [(0.25, 0.5), (0.15, 0.5), (0.1, 0.5), (0.1, 0.3)]
ORDERS = (0, 1, 2)
T_MIN = 4.0          # inicio de la fase activa (pulso r0=5, velocidad 1)
STRONG_FRACTION = 0.3     # fase activa: la medición de H2
MEASURABLE_FRACTION = 0.05  # + hombros del decaimiento


def run_oracle(
    l: int,
    n_points: int,
    potential_type: str = "zero",
    potential_params: dict | None = None,
    u_infinity: float = 0.0,
    A: float = 1e-3,
    t_end: float = 40.0,
    n_snapshots: int = 40,
) -> dict:
    """Evolución interior estándar del piloto F0 (pulso entrante desde r0=5)."""
    oracle = SphericalOracle1D(
        M=1.0, l=l, r_min=0.01, r_max=60.0, n_points=n_points, grid="log",
        potential_type=potential_type, potential_params=potential_params,
        u_infinity=u_infinity, ko_eps=0.02,
    )
    oracle.set_initial_gaussian(A=A, r0=5.0, width=1.0, direction="ingoing_curved")
    dt = oracle.compute_dt()
    snap_every = max(1, int(np.ceil(t_end / dt)) // n_snapshots)
    t0 = time.perf_counter()
    out = oracle.evolve(t_end=t_end, probe_radii=[0.1, 0.5],
                        output_every=200, snapshot_every=snap_every)
    wall = time.perf_counter() - t0
    return {
        "r": oracle.r,
        "snapshot_ts": np.asarray(out.snapshot_ts),
        "snapshots_u": np.asarray(out.snapshots_u),
        "snapshots_Pi": np.asarray(out.snapshots_Pi),
        "wall_seconds": wall,
    }


def load_or_run(path: Path, force: bool, **kwargs) -> dict:
    if path.exists() and not force:
        d = np.load(path)
        return {k: d[k] for k in d.files}
    res = run_oracle(**kwargs)
    np.savez_compressed(path, **{k: res[k] for k in
                                 ("r", "snapshot_ts", "snapshots_u", "snapshots_Pi")})
    print(f"  {path.name}: corrido en {res['wall_seconds']:.0f} s", flush=True)
    return res


def calibrate_dataset(name: str, data: dict) -> dict:
    """Sesgo por ventana/orden contra la verdad profunda, para un dataset."""
    r = data["r"]
    ts = data["snapshot_ts"]
    used = ts >= T_MIN
    snaps = [u for u, m in zip(data["snapshots_u"], used) if m]
    ts_used = ts[used]

    truth = fit_log_profile_series(r, snaps, TRUTH_WINDOW, order=1)
    a_truth = truth["a"]
    scale = float(np.max(np.abs(a_truth)))
    strong = np.abs(a_truth) >= STRONG_FRACTION * scale
    measurable = np.abs(a_truth) >= MEASURABLE_FRACTION * scale

    # piso de la verdad: spread al mover la ventana de verdad
    truth_spread = 0.0
    for win in TRUTH_SCAN:
        a_alt = fit_log_profile_series(r, snaps, win, order=1)["a"]
        truth_spread = max(truth_spread,
                           float(np.max(np.abs(a_alt - a_truth)[strong]
                                        / np.abs(a_truth)[strong])))

    result = {
        "ts": ts_used.tolist(),
        "a_truth": a_truth.tolist(),
        "a_truth_scale": scale,
        "truth_window": list(TRUTH_WINDOW),
        "truth_rel_spread_strong": truth_spread,
        "n_snapshots_used": int(len(snaps)),
        "n_strong": int(np.count_nonzero(strong)),
        "n_measurable": int(np.count_nonzero(measurable)),
        "windows": {},
    }
    for win in CANDIDATES:
        for order in ORDERS:
            fit = fit_log_profile_series(r, snaps, win, order=order)
            da = np.abs(fit["a"] - a_truth)
            rel_strong = da[strong] / np.abs(a_truth)[strong]
            rel_meas = da[measurable] / np.abs(a_truth)[measurable]
            key = f"[{win[0]:g},{win[1]:g}] o{order}"
            result["windows"][key] = {
                "bias_rel_median_strong": float(np.median(rel_strong)),
                "bias_rel_max_strong": float(np.max(rel_strong)),
                "bias_rel_median_measurable": float(np.median(rel_meas)),
                "bias_rel_max_measurable": float(np.max(rel_meas)),
                "bias_over_scale_max": float(np.max(da / scale)),
                "a_err_rel_median": float(np.median(
                    fit["a_err"][strong] / np.abs(a_truth)[strong])),
                "cond_median": float(np.median(fit["cond"])),
                "a_series": fit["a"].tolist(),
            }
    print(f"  {name}: escala a={scale:.3e}, "
          f"{int(np.count_nonzero(strong))} fuertes / "
          f"{int(np.count_nonzero(measurable))} medibles, "
          f"piso de la verdad {truth_spread:.2%}", flush=True)
    return result


def make_figure(all_results: dict, all_data: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(all_results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    for ax, name in zip(axes.ravel(), names):
        res = all_results[name]
        ts = np.array(res["ts"])
        ax.plot(ts, np.array(res["a_truth"]) / res["a_truth_scale"],
                "k-", lw=2.2, label=f"verdad {res['truth_window']} o1")
        for key, style in [("[0.25,0.5] o0", "C3-"), ("[0.25,0.5] o1", "C3--"),
                           ("[0.1,0.5] o1", "C0--"), ("[0.1,0.5] o2", "C0:")]:
            w = res["windows"][key]
            ax.plot(ts, np.array(w["a_series"]) / res["a_truth_scale"],
                    style, lw=1.1, alpha=0.85,
                    label=f"{key} (max {w['bias_rel_max_strong']:.1%})")
        ax.set_title(name)
        ax.axhline(0, color="gray", lw=0.5)
        ax.legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("t/M")
    for ax in axes[:, 0]:
        ax.set_ylabel("a(t) / max|a_verdad|")
    fig.suptitle("Calibración 1D de la ventana del estimador interior a(t) "
                 "(sesgo relativo máximo en snapshots fuertes)")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "window_calibration.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="re-correr el oráculo aunque existan los npz")
    parser.add_argument("--fast", action="store_true",
                        help="resoluciones reducidas (humo)")
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    n_hi = 1000 if args.fast else 1600
    n_lo = 800 if args.fast else 1300

    t0 = time.perf_counter()
    all_data: dict[str, dict] = {}

    print("[1/3] datasets del piloto F0 (reuso)...", flush=True)
    for l in (0, 1):
        d = np.load(PHASE0_DATA / f"linear_interior_l{l}.npz")
        all_data[f"lineal l={l} (F0, n=2600)"] = {k: d[k] for k in d.files}

    print("[2/3] corridas nuevas del oráculo...", flush=True)
    all_data[f"lineal l=2 (n={n_hi})"] = load_or_run(
        DATA / "linear_interior_l2.npz", args.force, l=2, n_points=n_hi)
    all_data[f"mexican_hat u∞=v, A=0.1 (n={n_hi})"] = load_or_run(
        DATA / "mexhat_vacuum_A0.1.npz", args.force, l=0, n_points=n_hi,
        potential_type="mexican_hat",
        potential_params={"lambda_coupling": 0.1, "vacuum_value": 1.0},
        u_infinity=1.0, A=0.1)
    lowres = load_or_run(
        DATA / "linear_interior_l0_lowres.npz", args.force, l=0, n_points=n_lo)

    print("[3/3] calibración...", flush=True)
    results = {name: calibrate_dataset(name, data)
               for name, data in all_data.items()}
    results[f"lineal l=0 (réplica n={n_lo})"] = calibrate_dataset(
        f"lineal l=0 (réplica n={n_lo})", lowres)

    make_figure({k: results[k] for k in list(all_data.keys())[:4]}, all_data)

    # tabla de decisión
    print("\n=== sesgo relativo de a(t) vs verdad profunda ===", flush=True)
    for name, res in results.items():
        print(f"\n{name}  [piso verdad {res['truth_rel_spread_strong']:.2%}; "
              f"{res['n_strong']} fuertes / {res['n_measurable']} medibles]")
        print(f"  {'ventana/orden':16s} {'FUERTE max/med':>16s} "
              f"{'MEDIBLE max/med':>17s} {'σ_a rel':>8s} {'cond':>9s}")
        for key, w in res["windows"].items():
            print(f"  {key:16s} {w['bias_rel_max_strong']:7.2%}/"
                  f"{w['bias_rel_median_strong']:7.2%} "
                  f"{w['bias_rel_max_measurable']:8.2%}/"
                  f"{w['bias_rel_median_measurable']:7.2%} "
                  f"{w['a_err_rel_median']:8.2%} {w['cond_median']:9.1f}")

    summary = {
        "protocol": {
            "truth_window": list(TRUTH_WINDOW), "truth_order": 1,
            "truth_scan": [list(w) for w in TRUTH_SCAN],
            "candidates": [list(w) for w in CANDIDATES],
            "t_min": T_MIN, "strong_fraction": STRONG_FRACTION,
            "fast_mode": bool(args.fast),
        },
        "datasets": {
            name: {k: v for k, v in res.items() if k != "ts"}
            for name, res in results.items()
        },
        "total_wall_seconds": time.perf_counter() - t0,
    }
    # las series largas van solo en la figura; el JSON guarda los números
    for res in summary["datasets"].values():
        res.pop("a_truth", None)
        for w in res["windows"].values():
            w.pop("a_series", None)
    with open(OUT / "window_calibration.json", "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\nListo en {summary['total_wall_seconds']:.0f} s. "
          f"JSON en {OUT / 'window_calibration.json'}, figura en {FIG}.",
          flush=True)


if __name__ == "__main__":
    main()
