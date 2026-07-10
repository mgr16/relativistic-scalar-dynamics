#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 2 — humo A/B del diagnóstico interior 3D: lineal vs sombrero mexicano
con dato idéntico.

Cierra el capítulo del diagnóstico interior (docs/research/phase2/interior/
note.md §4): dos corridas 3D con la MISMA perturbación gaussiana entrante
(A=0.1, r0=5, w=1, ingoing_curved) sobre su vacío respectivo —
    linear : V = 0,            φ = 0 + pulso
    mexhat : V = ¼λ(φ²−v²)²,   φ = v + pulso   (λ=0.1, v=1 — la config H2)
— con el banco interior de K=16 radios log en [0.1, 0.5] y el estimador
calibrado (orden 2 primario, orden 1 en [0.1, 0.3] de contraste).

Qué valida / mide:
 1. Pipeline 3D completo (banco → fit) contra el oráculo 1D: para campos
    esféricos c_00 = √(4π)·u, así que a_00(t) debe seguir a √(4π)·a_1D(t)
    de la verdad profunda del oráculo (o1 en [0.02, 0.2]) con el error de
    malla + sesgo de ventana calibrado.
 2. El primer número 3D del discriminador de H2: a_hat(t) vs a_lin(t) con
    dato idéntico (si la dominación cinética borra la estructura de vacío,
    los perfiles coinciden dentro del presupuesto de error).
 3. Salud del estimador con K=16 (cond, σ_a), mezcla de modos (fuga l>0),
    balance de Killing y monitor de Cowling.

Corridas crudas en results/phase2_interior_ab/ (gitignorado); resumen en
docs/research/phase2/interior/ab_smoke.json + figures/ab_smoke.png.

Uso:  python scripts/interior_ab_smoke.py [--force] [--fast] [--skip-runs]
      (--fast: piloto corto y grueso para validar el pipeline end-to-end)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(REPO / "src"))

from rsd.analysis.interior import fit_log_profile_multipole, fit_log_profile_series  # noqa: E402

OUT = REPO / "docs" / "research" / "phase2" / "interior"
DATA = OUT / "data"
FIG = OUT / "figures"
RESULTS = REPO / "results" / "phase2_interior_ab"
RSD_BIN = Path.home() / "miniforge3" / "envs" / "rsd-dolfinx" / "bin" / "rsd"

SQRT4PI = float(np.sqrt(4.0 * np.pi))

# Candidatos 3D (ventana, orden). La calibración 1D ordenó por SESGO
# (o2 [0.1,0.5] óptimo); en 3D el error de malla correlacionado se amplifica
# con el orden de la base (cond), así que el humo mide dev total vs verdad
# 1D por estimador y elige el primario por mediana en la corrida lineal.
ESTIMATORS = {
    "o0 [0.1,0.5]": ((0.1, 0.5), 0),
    "o1 [0.1,0.5]": ((0.1, 0.5), 1),
    "o1 [0.1,0.3]": ((0.1, 0.3), 1),
    "o2 [0.1,0.5]": ((0.1, 0.5), 2),
}
PHASE_ANCHOR = "o0 [0.1,0.5]"  # timing robusto (cond ~6): define la fase fuerte
TRUTH_WINDOW, TRUTH_ORDER = (0.02, 0.2), 1  # verdad profunda 1D (calibración)
T_MIN = 4.0
STRONG_FRACTION = 0.3

# el dato compartido del par A/B (la config H2 de la calibración 1D)
PULSE = {"A": 0.1, "r0": 5.0, "w": 1.0, "direction": "ingoing_curved"}

RUNS = {
    "linear": {
        "potential_type": "zero", "potential_params": {}, "v0": 0.0,
    },
    "mexhat": {
        "potential_type": "mexican_hat",
        "potential_params": {"lambda_coupling": 0.1, "vacuum_value": 1.0},
        "v0": 1.0,
    },
}


def base_cfg(t_end: float, lc_inner: float, output_every: int) -> dict:
    return {
        "mesh": {"type": "gmsh", "R": 15.0, "lc": 1.2,
                 "r_inner": 0.1, "lc_inner": lc_inner},
        "metric": {"type": "kerr", "M": 1.0, "a": 0.0},
        "solver": {
            "degree": 1, "cfl": 0.3, "bc_type": "characteristic",
            "enable_sommerfeld": True, "filter_strength": 0.0,
        },
        "initial_conditions": {"type": "gaussian", **PULSE},
        "analysis": {
            "sample_point": [0.2, 0.0, 0.0],
            "interior_profile": {
                "enabled": True, "r_lo": 0.1, "r_hi": 0.5,
                "n_radii": 32, "lmax": 2, "spacing": "log", "fit_order": 2,
            },
        },
        "evolution": {"t_end": t_end, "output_every": output_every, "verbose": False},
        "output": {"dir": "", "qnm_analysis": False,
                   "diagnostics": True, "save_series": True},
    }


def find_npz(outdir: Path) -> Path | None:
    hits = sorted(outdir.glob("run_*/series/interior_profiles.npz"))
    return hits[-1] if hits else None


def run_one(label: str, spec: dict, cfg_base: dict, force: bool) -> Path:
    outdir = RESULTS / f"run_{label}"
    existing = find_npz(outdir)
    if existing is not None and not force:
        print(f"  {label}: reuso {existing.relative_to(REPO)}", flush=True)
        return existing.parent.parent

    cfg = json.loads(json.dumps(cfg_base))
    cfg["solver"]["potential_type"] = spec["potential_type"]
    cfg["solver"]["potential_params"] = spec["potential_params"]
    cfg["initial_conditions"]["v0"] = spec["v0"]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(cfg, tf)
        cfgpath = tf.name

    outdir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CC": "/usr/bin/clang"}
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(RSD_BIN), "run", "--config", cfgpath, "--output", str(outdir)],
        capture_output=True, text=True, env=env,
    )
    wall = time.perf_counter() - t0
    (outdir / f"{label}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(
            f"run {label} falló (rc={proc.returncode}); log en {outdir / (label + '.log')}"
        )
    npz = find_npz(outdir)
    if npz is None:
        raise SystemExit(f"run {label}: no apareció interior_profiles.npz bajo {outdir}")
    print(f"  {label}: {wall:.0f} s → {npz.relative_to(REPO)}", flush=True)
    return npz.parent.parent


def load_run(rundir: Path) -> dict:
    d = np.load(rundir / "series" / "interior_profiles.npz")
    out = {k: d[k] for k in d.files}
    kcsv = rundir / "series" / "killing.csv"
    if kcsv.exists():
        k = np.loadtxt(kcsv, delimiter=",", skiprows=1)
        scale = max(abs(k[0, 1]), float(np.max(np.abs(k[:, 4]))), 1e-300)
        out["killing_residual_rel"] = float(abs(k[-1, 5]) / scale)
    ccsv = rundir / "series" / "cowling.csv"
    if ccsv.exists():
        c = np.loadtxt(ccsv, delimiter=",", skiprows=1)
        out["cowling_zeta_max"] = float(np.max(c[:, 1]))
    return out


def oracle_reference(label: str, force: bool, fast: bool) -> dict:
    """Verdad 1D del par A/B con EL MISMO dato y muestreo temporal denso
    (80 snapshots en 15M: el pico del mexhat es angosto y la referencia de
    la calibración, cada ~1M, lo submuestrea). Cacheada en DATA por label+n."""
    n = 800 if fast else 1600
    path = DATA / f"ab_smoke_ref_{label}_A0.1_n{n}.npz"
    if path.exists() and not force:
        d = np.load(path)
        return {k: d[k] for k in d.files}
    from rsd.reference import SphericalOracle1D

    spec = RUNS[label]
    oracle = SphericalOracle1D(
        M=1.0, l=0, r_min=0.01, r_max=60.0, n_points=n, grid="log",
        potential_type=spec["potential_type"],
        potential_params=spec["potential_params"] or None,
        u_infinity=spec["v0"], ko_eps=0.02,
    )
    oracle.set_initial_gaussian(A=PULSE["A"], r0=PULSE["r0"], width=PULSE["w"],
                                direction="ingoing_curved")
    dt = oracle.compute_dt()
    t_end = 15.0
    snap_every = max(1, int(np.ceil(t_end / dt)) // 80)
    t0 = time.perf_counter()
    res = oracle.evolve(t_end=t_end, probe_radii=[0.1, 0.5],
                        output_every=200, snapshot_every=snap_every)
    print(f"  oráculo {label} A=0.1 (n={n}): {time.perf_counter() - t0:.0f} s", flush=True)
    out = {"r": oracle.r, "snapshot_ts": np.asarray(res.snapshot_ts),
           "snapshots_u": np.asarray(res.snapshots_u)}
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def deep_truth(ref: dict) -> tuple[np.ndarray, np.ndarray]:
    """a_1D(t) de la verdad profunda (o1 [0.02, 0.2]), en la norma c_00 = √4π·u."""
    ts = ref["snapshot_ts"]
    fit = fit_log_profile_series(ref["r"], list(ref["snapshots_u"]),
                                 TRUTH_WINDOW, order=TRUTH_ORDER)
    return ts, SQRT4PI * fit["a"]


def _oracle_self_bias(ref_r, ref_snaps, ts_1d, a_1d, window, order) -> dict:
    """Sesgo de truncamiento 1D del estimador (mismo grid del oráculo):
    el yardstick que separa sesgo-de-ventana del término de malla 3D."""
    fit = fit_log_profile_series(ref_r, ref_snaps, window, order=order)
    a_win = SQRT4PI * fit["a"]
    scale = float(np.max(np.abs(a_1d)))
    strong = np.abs(a_1d) >= STRONG_FRACTION * scale
    dev = np.abs(a_win - a_1d)[strong] / scale
    return {"max": float(np.max(dev)), "median": float(np.median(dev))}


def analyze(runs: dict[str, dict], refs: dict[str, tuple[np.ndarray, np.ndarray]],
            raw_refs: dict[str, dict]) -> dict:
    summary: dict = {"runs": {}}
    a_series: dict = {}
    for label, data in runs.items():
        radii, t = data["radii"], data["t"]
        modes = [tuple(m) for m in data["modes"]]
        fits = {name: fit_log_profile_multipole(radii, data["u"], modes, win, order=o)
                for name, (win, o) in ESTIMATORS.items()}

        # fase fuerte anclada en el estimador de timing (o0: cond ~6, la
        # varianza de los órdenes altos en tiempos tardíos no debe definirla)
        a_anchor = fits[PHASE_ANCHOR][(0, 0)]["a"]
        late = t >= T_MIN
        scale_anchor = float(np.max(np.abs(a_anchor[late])))
        strong = late & (np.abs(a_anchor) >= STRONG_FRACTION * scale_anchor)

        ts_1d, a_1d = refs[label]
        strong &= t <= ts_1d[-1]
        a_ref = np.interp(t[strong], ts_1d, a_1d)
        scale_ref = float(np.max(np.abs(a_1d)))
        ref = raw_refs[label]

        est_report = {}
        for name, (win, order) in ESTIMATORS.items():
            f00 = fits[name][(0, 0)]
            dev = np.abs(f00["a"][strong] - a_ref) / scale_ref
            est_report[name] = {
                "window": list(win), "order": order,
                "dev_vs_1d_max": float(np.max(dev)),
                "dev_vs_1d_median": float(np.median(dev)),
                "bias_1d_only": _oracle_self_bias(
                    ref["r"], list(ref["snapshots_u"]), ts_1d, a_1d, win, order),
                "sigma_a_median_over_scale": float(
                    np.median(f00["a_err"][strong]) / scale_ref),
                "cond_median": float(np.median(f00["cond"][strong])),
            }

        junk = {}
        for lval in (1, 2):
            peak = max(float(np.max(np.abs(fits[PHASE_ANCHOR][m]["a"][strong])))
                       for m in modes if m[0] == lval)
            junk[f"l{lval}"] = peak / scale_ref

        a_series[label] = {"t": t, "strong": strong,
                           "fits": {n: fits[n][(0, 0)]["a"] for n in ESTIMATORS}}
        summary["runs"][label] = {
            "n_samples": int(t.size), "t_end": float(t[-1]),
            "n_strong": int(np.count_nonzero(strong)),
            "scale_1d_c00": scale_ref,
            "estimators": est_report,
            "junk_over_scale": junk,
            "killing_residual_rel": runs[label].get("killing_residual_rel"),
            "cowling_zeta_max_global": runs[label].get("cowling_zeta_max"),
        }

    # primario 3D: mediana mínima de dev vs verdad en la corrida LINEAL
    # (la verdad lineal es la más precisa; la elección se aplica a ambas)
    lin_est = summary["runs"]["linear"]["estimators"]
    primary = min(lin_est, key=lambda n: lin_est[n]["dev_vs_1d_median"])
    summary["primary_3d_estimator"] = primary

    # discriminador de H2: a_hat/a_lin con dato idéntico, fase fuerte común,
    # medido con el primario elegido + o0 de consistencia
    disc = {}
    for est_name, key in [(primary, "primary"), (PHASE_ANCHOR, "anchor_o0")]:
        sl, sh = a_series["linear"], a_series["mexhat"]
        a_l, a_h = sl["fits"][est_name], sh["fits"][est_name]
        a_h_on_l = np.interp(sl["t"], sh["t"], a_h)
        s_h_on_l = np.interp(sl["t"], sh["t"], sh["strong"].astype(float)) > 0.5
        both = sl["strong"] & s_h_on_l
        ratio = a_h_on_l[both] / a_l[both]
        disc[key] = {
            "estimator": est_name,
            "ratio_median": float(np.median(ratio)),
            "ratio_iqr": [float(np.percentile(ratio, 25)),
                          float(np.percentile(ratio, 75))],
            "n_samples": int(ratio.size),
        }
    # el mismo cociente en el oráculo 1D (la expectativa del humo)
    ts_rl, a_rl = refs["linear"]
    ts_rh, a_rh = refs["mexhat"]
    t_common = ts_rl[(ts_rl >= T_MIN) & (ts_rl <= min(ts_rl[-1], ts_rh[-1]))]
    r_l = np.interp(t_common, ts_rl, a_rl)
    r_h = np.interp(t_common, ts_rh, a_rh)
    strong_1d = np.abs(r_l) >= STRONG_FRACTION * np.max(np.abs(r_l))
    ratio_1d = r_h[strong_1d] / r_l[strong_1d]
    disc["oracle_1d"] = {
        "ratio_median": float(np.median(ratio_1d)),
        "ratio_iqr": [float(np.percentile(ratio_1d, 25)),
                      float(np.percentile(ratio_1d, 75))],
        "n_samples": int(ratio_1d.size),
    }
    summary["discriminator"] = {
        "definition": "a00_mexhat / a00_linear (mismo pulso), fase fuerte comun",
        **disc,
    }
    summary["_a_series"] = a_series  # solo para la figura; se poda antes del JSON
    return summary


def make_figure(summary: dict, runs: dict, refs: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # identidad fija por entidad en todos los paneles:
    # lineal = C0 (azul), mexhat = C3 (rojo), oráculo 1D = negro/gris
    styles = {"linear": ("C0", "lineal (V=0)"), "mexhat": ("C3", "mexhat u∞=v")}
    truth_styles = {"linear": "k", "mexhat": "0.45"}
    primary = summary["primary_3d_estimator"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    ax = axes[0, 0]
    for label, (color, name) in styles.items():
        s = summary["_a_series"][label]
        ax.plot(s["t"], s["fits"][primary], color=color, lw=1.6,
                label=f"3D {name} ({primary})")
        ts_1d, a_1d = refs[label]
        ax.plot(ts_1d, a_1d, truth_styles[label], ls="--", lw=1.2,
                label=f"oráculo 1D ×√4π ({name})")
    ax.set_xlim(0, None)
    ax.set_xlabel("t/M")
    ax.set_ylabel(r"$a_{00}(t)$")
    ax.set_title(f"validación 3D vs oráculo 1D (primario: {primary})")
    ax.axhline(0, color="gray", lw=0.5)
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    s = summary["_a_series"]["linear"]
    ts_1d, a_1d = refs["linear"]
    scale_ref = float(np.max(np.abs(a_1d)))
    for i, name in enumerate(ESTIMATORS):
        ax.plot(s["t"], s["fits"][name], color=f"C{i}", lw=1.2, label=name)
    ax.plot(ts_1d, a_1d, "k--", lw=1.6, label="verdad 1D ×√4π")
    ax.set_ylim(-3.0 * scale_ref, 3.0 * scale_ref)  # el o2 se sale: es el punto
    ax.set_xlabel("t/M")
    ax.set_ylabel(r"$a_{00}(t)$")
    ax.set_title("presupuesto de error (corrida lineal, y recortado a ±3 escalas)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    sl, sh = summary["_a_series"]["linear"], summary["_a_series"]["mexhat"]
    a_l = sl["fits"][primary]
    a_h_on_l = np.interp(sl["t"], sh["t"], sh["fits"][primary])
    s_h_on_l = np.interp(sl["t"], sh["t"], sh["strong"].astype(float)) > 0.5
    both = sl["strong"] & s_h_on_l
    with np.errstate(divide="ignore", invalid="ignore"):
        full_ratio = np.where(np.abs(a_l) > 0, a_h_on_l / a_l, np.nan)
    ax.plot(sl["t"][both], full_ratio[both], "C2-", lw=1.6,
            label=f"3D ({primary}, fase fuerte)")
    ts_rl, a_rl = refs["linear"]
    ts_rh, a_rh = refs["mexhat"]
    t_common = ts_rl[(ts_rl >= T_MIN) & (ts_rl <= min(ts_rl[-1], ts_rh[-1]))]
    r_l = np.interp(t_common, ts_rl, a_rl)
    r_h = np.interp(t_common, ts_rh, a_rh)
    strong_1d = np.abs(r_l) >= STRONG_FRACTION * np.max(np.abs(r_l))
    ax.plot(t_common[strong_1d], (r_h / r_l)[strong_1d], "k--", lw=1.2,
            label="oráculo 1D")
    ax.axhline(1, color="gray", lw=0.5)
    ax.set_xlabel("t/M")
    ax.set_ylabel(r"$a_{00}^{\rm hat}/a_{00}^{\rm lin}$")
    ax.set_title("discriminador de H2 (dato idéntico)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    anchor_win, anchor_order = ESTIMATORS[PHASE_ANCHOR]
    for label, (color, name) in styles.items():
        data = runs[label]
        radii = data["radii"]
        modes = [tuple(m) for m in data["modes"]]
        anchor = fit_log_profile_multipole(radii, data["u"], modes,
                                           anchor_win, order=anchor_order)
        t = data["t"]
        scale = summary["runs"][label]["scale_1d_c00"]
        for lval, ls in ((1, "-"), (2, ":")):
            peak = np.max(np.abs(np.stack(
                [anchor[m]["a"] for m in modes if m[0] == lval])), axis=0)
            ax.semilogy(t, peak / scale, color=color, ls=ls, lw=1.2,
                        label=f"{name} l={lval}")
    ax.set_xlabel("t/M")
    ax.set_ylabel(r"$\max_m |a_{lm}| / {\rm escala\ 1D}$")
    ax.set_title("fuga de modos l>0 (junk de malla, fit o0)")
    ax.legend(fontsize=7)

    fig.suptitle("Humo A/B del diagnóstico interior 3D: lineal vs mexhat, dato idéntico")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "ab_smoke.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="re-correr aunque existan resultados")
    parser.add_argument("--fast", action="store_true",
                        help="piloto corto y grueso (valida el pipeline)")
    parser.add_argument("--skip-runs", action="store_true",
                        help="solo post-proceso de corridas existentes")
    args = parser.parse_args()

    if args.fast:
        cfg = base_cfg(t_end=6.0, lc_inner=0.08, output_every=5)
    else:
        cfg = base_cfg(t_end=12.0, lc_inner=0.04, output_every=10)

    t0 = time.perf_counter()
    print("[1/4] corridas 3D...", flush=True)
    rundirs = {}
    for label, spec in RUNS.items():
        if args.skip_runs:
            npz = find_npz(RESULTS / f"run_{label}")
            if npz is None:
                raise SystemExit(f"--skip-runs pero no hay corrida {label}")
            rundirs[label] = npz.parent.parent
        else:
            rundirs[label] = run_one(label, spec, cfg, args.force)

    print("[2/4] referencias del oráculo 1D...", flush=True)
    raw_refs = {label: oracle_reference(label, force=False, fast=args.fast)
                for label in RUNS}
    refs = {label: deep_truth(ref) for label, ref in raw_refs.items()}

    print("[3/4] análisis...", flush=True)
    runs = {label: load_run(rd) for label, rd in rundirs.items()}
    summary = analyze(runs, refs, raw_refs)

    print("[4/4] figura + resumen...", flush=True)
    make_figure(summary, runs, refs)
    summary.pop("_a_series")
    summary["protocol"] = {
        "pulse": PULSE, "runs": RUNS,
        "estimators": {n: {"window": list(w), "order": o}
                       for n, (w, o) in ESTIMATORS.items()},
        "phase_anchor": PHASE_ANCHOR,
        "truth_1d": {"window": list(TRUTH_WINDOW), "order": TRUTH_ORDER,
                     "normalization": "sqrt(4pi)"},
        "mesh": cfg["mesh"], "t_end": cfg["evolution"]["t_end"],
        "fast_mode": bool(args.fast),
    }
    summary["total_wall_seconds"] = time.perf_counter() - t0
    with open(OUT / "ab_smoke.json", "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print("\n=== humo A/B: números clave ===")
    print(f"primario 3D elegido (mediana mínima vs verdad, corrida lineal): "
          f"{summary['primary_3d_estimator']}")
    for label, res in summary["runs"].items():
        print(f"\n{label}: escala 1D (c00) = {res['scale_1d_c00']:.3f}, "
              f"{res['n_strong']} snapshots fuertes")
        print(f"  {'estimador':14s} {'dev vs 1D max/med':>20s} "
              f"{'sesgo 1D max/med':>18s} {'σ_a med':>8s} {'cond':>8s}")
        for name, e in res["estimators"].items():
            b = e["bias_1d_only"]
            print(f"  {name:14s} {e['dev_vs_1d_max']:9.2%}/{e['dev_vs_1d_median']:8.2%} "
                  f"{b['max']:9.2%}/{b['median']:7.2%} "
                  f"{e['sigma_a_median_over_scale']:8.2%} {e['cond_median']:8.0f}")
        print(f"  junk l=1: {res['junk_over_scale']['l1']:.2%}, "
              f"l=2: {res['junk_over_scale']['l2']:.2%}; "
              f"Killing residual {res['killing_residual_rel']:.2e} "
              f"(cuadratura del flujo en r_inner=0.1: ver nota); "
              f"Cowling ζ_max global {res['cowling_zeta_max_global']:.2e} "
              f"(dominado por el exterior de curvatura débil)")
    disc = summary["discriminator"]
    print(f"\ndiscriminador a_hat/a_lin (fase fuerte común):")
    for key in ("primary", "anchor_o0", "oracle_1d"):
        e = disc[key]
        name = e.get("estimator", "verdad 1D")
        print(f"  {key:10s} ({name}): mediana {e['ratio_median']:.3f}, "
              f"IQR [{e['ratio_iqr'][0]:.3f}, {e['ratio_iqr'][1]:.3f}] "
              f"({e['n_samples']} muestras)")
    print(f"\nListo en {summary['total_wall_seconds']:.0f} s. "
          f"JSON en {OUT / 'ab_smoke.json'}, figura en {FIG / 'ab_smoke.png'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
