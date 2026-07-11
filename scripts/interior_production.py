#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 2 — corridas de producción del diagnóstico interior: lineal vs mexhat
con dato idéntico, modos l = 0, 1, 2, escalera de convergencia.

Extiende el protocolo del humo A/B (interior_ab_smoke.py; decisiones en
docs/research/phase2/interior/note.md §5) a la medición de producción:

  * Escalera l=0: lc_inner ∈ {0.056, 0.04, 0.028} × {lineal, mexhat} —
    baja/cuantifica el término de malla (10–15 % mediana en el humo) por
    auto-convergencia y da el número primario de H2 con presupuesto.
  * l=1 @ 0.04 y l=2 @ {0.04, 0.028}: a_l0(t) por modo — la dependencia
    angular A(t,ω) de Fournodavlos–Sbierski (valor añadido 3D). El dato es
    gaussiana × Y_l0 (idéntico al modo u_l del oráculo); banco lmax=4 en
    las corridas l=2 para ver el acoplamiento 2⊗2→{0,4} del mexhat
    (física SIN oráculo: la reducción 1D no es exacta con potencial no
    lineal y l>0).
  * Etapa 0 — A/B de mass lumping sobre la config de producción (la regla
    de F1: A/B antes de adoptar): lineal l=0 @ 0.04 lumped vs la corrida
    del humo (consistente). Se adopta para TODA la matriz solo si el
    desplazamiento de a00 en fase fuerte es mediana < 2 % y máx < 5 %
    (muy por debajo del término de malla); si no, la matriz corre sin
    lumping y el rung 0.04 de l=0 REUSA las corridas del humo.

Estimadores: primario 3D o1 [0.1, 0.5] + ancla de fase o0 (decisión del
humo — el o2 óptimo-en-sesgo es inusable en 3D por varianza).
Normalización vs oráculo: c_00 = √4π·u (dato l=0 sin factor Y_00);
c_l0 = u_l para l ≥ 1 (dato A·g·Y_l0 ⇒ c_l0(0) = A·g = u_l(0)).

Corridas crudas en results/phase2_production/ (gitignorado); resumen en
docs/research/phase2/production/production.json + figures/.

Uso:  python scripts/interior_production.py [--fast] [--force] [--skip-runs]
      [--workers N]
      (--fast: matriz reducida y gruesa para validar el pipeline end-to-end;
       escribe TODO bajo results/phase2_production_fast/ y no toca docs/)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(REPO / "src"))

from rsd.analysis.interior import fit_log_profile_multipole, fit_log_profile_series  # noqa: E402

RSD_BIN = Path.home() / "miniforge3" / "envs" / "rsd-dolfinx" / "bin" / "rsd"
SMOKE_RESULTS = REPO / "results" / "phase2_interior_ab"
SMOKE_DATA = REPO / "docs" / "research" / "phase2" / "interior" / "data"

SQRT4PI = float(np.sqrt(4.0 * np.pi))

# el dato compartido de TODOS los pares A/B (la config H2; l entra por corrida)
PULSE = {"A": 0.1, "r0": 5.0, "w": 1.0, "direction": "ingoing_curved"}
POTS = {
    "linear": {"potential_type": "zero", "potential_params": {}, "v0": 0.0},
    "mexhat": {"potential_type": "mexican_hat",
               "potential_params": {"lambda_coupling": 0.1, "vacuum_value": 1.0},
               "v0": 1.0},
}

# estimadores de producción (decisión del humo §5)
PRIMARY = ((0.1, 0.5), 1)   # o1 [0.1, 0.5]
ANCHOR = ((0.1, 0.5), 0)    # o0: timing robusto (cond ~6)
TRUTH_WINDOW, TRUTH_ORDER = (0.02, 0.2), 1
T_MIN = 4.0
# tope superior de la fase fuerte: en los canales l>0 el junk tardío crece
# hasta ~la escala de la señal después de t≈10M (dominio R=15 sin esponja);
# sin tope, la máscara |a|≥0.3·max lo incluye y contamina dev/discriminador
# (la regla de F1: ventanas ancladas, colas no convergentes fuera)
T_MAX = 10.0
STRONG_FRACTION = 0.3
LC_BASE, OUTPUT_EVERY_BASE = 0.04, 10  # muestreo ≈ 0.093M en todos los rungs

# criterio de adopción del lumping (vs término de malla 10-15 %)
ML_MEDIAN_TOL, ML_MAX_TOL = 0.02, 0.05


def matrix(fast: bool) -> list[dict]:
    """La matriz de corridas: (potencial, l, lc_inner, lmax del banco)."""
    if fast:
        cells = [(l, lc) for l in (0, 2) for lc in (0.12, 0.08)]
    else:
        cells = ([(0, lc) for lc in (0.056, 0.04, 0.028)]
                 + [(1, 0.04)]
                 + [(2, 0.04), (2, 0.028)])
    runs = []
    for pot in POTS:
        for l, lc in cells:
            runs.append({"pot": pot, "l": l, "lc": lc,
                         "lmax": 4 if l == 2 else 2,
                         "label": f"{pot}_l{l}_lc{lc:.3f}"})
    return runs


def base_cfg(t_end: float, lc_inner: float, l: int, lmax: int,
             lumping: bool) -> dict:
    return {
        "mesh": {"type": "gmsh", "R": 15.0, "lc": 1.2,
                 "r_inner": 0.1, "lc_inner": lc_inner},
        "metric": {"type": "kerr", "M": 1.0, "a": 0.0},
        "solver": {
            "degree": 1, "cfl": 0.3, "bc_type": "characteristic",
            "enable_sommerfeld": True, "filter_strength": 0.0,
        },
        "optimization": {"mass_lumping": bool(lumping)},
        "initial_conditions": {"type": "gaussian", **PULSE, "l": l, "m": 0},
        "analysis": {
            "sample_point": [0.2, 0.0, 0.0],
            "interior_profile": {
                "enabled": True, "r_lo": 0.1, "r_hi": 0.5,
                "n_radii": 32, "lmax": lmax, "spacing": "log", "fit_order": 1,
            },
        },
        "evolution": {
            "t_end": t_end,
            "output_every": max(1, round(OUTPUT_EVERY_BASE * LC_BASE / lc_inner)),
            "verbose": False,
        },
        "output": {"dir": "", "qnm_analysis": False,
                   "diagnostics": True, "save_series": True},
    }


def find_npz(outdir: Path) -> Path | None:
    hits = sorted(outdir.glob("run_*/series/interior_profiles.npz"))
    return hits[-1] if hits else None


def run_one(spec: dict, results_dir: Path, t_end: float, lumping: bool,
            force: bool) -> tuple[str, Path | None, float]:
    """Ejecuta (o reusa) una corrida; devuelve (label, rundir|None, wall_s)."""
    label = spec["label"] + ("_ml" if lumping else "")
    outdir = results_dir / f"run_{label}"
    existing = find_npz(outdir)
    if existing is not None and not force:
        print(f"  {label}: reuso {existing.relative_to(REPO)}", flush=True)
        return label, existing.parent.parent, 0.0

    cfg = base_cfg(t_end, spec["lc"], spec["l"], spec["lmax"], lumping)
    pot = POTS[spec["pot"]]
    cfg["solver"]["potential_type"] = pot["potential_type"]
    cfg["solver"]["potential_params"] = pot["potential_params"]
    cfg["initial_conditions"]["v0"] = pot["v0"]
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
        print(f"  {label}: FALLÓ (rc={proc.returncode}) tras {wall:.0f} s — "
              f"log en {outdir / (label + '.log')}", flush=True)
        return label, None, wall
    npz = find_npz(outdir)
    if npz is None:
        print(f"  {label}: sin interior_profiles.npz tras {wall:.0f} s", flush=True)
        return label, None, wall
    print(f"  {label}: {wall:.0f} s → {npz.relative_to(REPO)}", flush=True)
    return label, npz.parent.parent, wall


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


def fit_channel(data: dict, channel: tuple[int, int],
                est: tuple[tuple[float, float], int]) -> dict:
    win, order = est
    modes = [tuple(m) for m in data["modes"]]
    fits = fit_log_profile_multipole(data["radii"], data["u"], modes, win, order=order)
    return fits[channel]


def strong_mask(t: np.ndarray, a_anchor: np.ndarray) -> np.ndarray:
    window = (t >= T_MIN) & (t <= T_MAX)
    scale = float(np.max(np.abs(a_anchor[window]))) if np.any(window) else 0.0
    return window & (np.abs(a_anchor) >= STRONG_FRACTION * scale)


# ----------------------------------------------------------------------
# oráculo 1D
# ----------------------------------------------------------------------

def oracle_reference(pot: str, l: int, data_dir: Path, fast: bool) -> dict | None:
    """Verdad 1D densa del modo l (solo potencial lineal para l>0 — la
    reducción no es exacta con potencial no lineal). Cacheada por (pot, l, n).
    Para l=0 reusa el caché del humo si existe (mismo dato)."""
    if pot == "mexhat" and l > 0:
        return None
    n = 800 if fast else (2600 if l == 2 else 1600)
    if l == 0:
        smoke_path = SMOKE_DATA / f"ab_smoke_ref_{pot}_A0.1_n{n}.npz"
        if smoke_path.exists():
            d = np.load(smoke_path)
            return {k: d[k] for k in d.files}
    path = data_dir / f"prod_ref_{pot}_l{l}_A0.1_n{n}.npz"
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}
    from rsd.reference import SphericalOracle1D

    spec = POTS[pot]
    oracle = SphericalOracle1D(
        M=1.0, l=l, r_min=0.01, r_max=60.0, n_points=n, grid="log",
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
    print(f"  oráculo {pot} l={l} (n={n}): {time.perf_counter() - t0:.0f} s",
          flush=True)
    out = {"r": oracle.r, "snapshot_ts": np.asarray(res.snapshot_ts),
           "snapshots_u": np.asarray(res.snapshots_u)}
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def deep_truth(ref: dict, l: int) -> tuple[np.ndarray, np.ndarray]:
    """a_1D(t) profunda (o1 [0.02, 0.2]) en la norma del canal 3D:
    ×√4π para l=0 (dato esférico sin Y_00), ×1 para l ≥ 1."""
    norm = SQRT4PI if l == 0 else 1.0
    fit = fit_log_profile_series(ref["r"], list(ref["snapshots_u"]),
                                 TRUTH_WINDOW, order=TRUTH_ORDER)
    return ref["snapshot_ts"], norm * fit["a"]


# ----------------------------------------------------------------------
# etapa 0: A/B de mass lumping
# ----------------------------------------------------------------------

def lumping_ab(results_dir: Path, t_end: float, force: bool) -> dict:
    """Corre lineal l=0 @ LC_BASE con lumping y compara a00 (primario) contra
    la corrida consistente del humo. Devuelve el veredicto para la matriz."""
    smoke_npz = find_npz(SMOKE_RESULTS / "run_linear")
    if smoke_npz is None:
        print("  sin corrida del humo para comparar — lumping OFF por defecto",
              flush=True)
        return {"adopted": False, "reason": "no smoke baseline"}

    spec = {"pot": "linear", "l": 0, "lc": LC_BASE, "lmax": 2,
            "label": f"linear_l0_lc{LC_BASE:.3f}"}
    label, rundir, wall = run_one(spec, results_dir, t_end, lumping=True,
                                  force=force)
    if rundir is None:
        return {"adopted": False, "reason": "lumped run failed"}

    lumped = load_run(rundir)
    consistent = load_run(smoke_npz.parent.parent)
    fl = fit_channel(lumped, (0, 0), PRIMARY)
    fc = fit_channel(consistent, (0, 0), PRIMARY)
    anchor_c = fit_channel(consistent, (0, 0), ANCHOR)
    strong = strong_mask(consistent["t"], anchor_c["a"])
    a_l = np.interp(consistent["t"], lumped["t"], fl["a"])
    scale = float(np.max(np.abs(fc["a"][strong])))
    dev = np.abs(a_l - fc["a"])[strong] / scale
    med, mx = float(np.median(dev)), float(np.max(dev))
    adopted = med < ML_MEDIAN_TOL and mx < ML_MAX_TOL
    # el humo tardó 406 s consistente en esta config; wall≈0 si se reusó
    print(f"  lumping A/B: Δa00/escala mediana {med:.2%}, máx {mx:.2%} → "
          f"{'ADOPTADO' if adopted else 'RECHAZADO'} "
          f"(lumped {wall:.0f} s vs 406 s consistente)", flush=True)
    return {"adopted": adopted, "dev_median": med, "dev_max": mx,
            "wall_lumped_s": wall, "wall_consistent_smoke_s": 406.0,
            "criterion": {"median": ML_MEDIAN_TOL, "max": ML_MAX_TOL}}


# ----------------------------------------------------------------------
# análisis
# ----------------------------------------------------------------------

def analyze_run(label: str, data: dict, l: int,
                truth: tuple[np.ndarray, np.ndarray] | None) -> dict:
    """Fits del canal de señal (l, 0) + salud; dev vs verdad si la hay."""
    t = data["t"]
    ch = (l, 0)
    prim = fit_channel(data, ch, PRIMARY)
    anch = fit_channel(data, ch, ANCHOR)
    strong = strong_mask(t, anch["a"])
    out: dict = {
        "l": l, "n_samples": int(t.size), "t_end": float(t[-1]),
        "n_strong": int(np.count_nonzero(strong)),
        "killing_residual_rel": data.get("killing_residual_rel"),
        "cowling_zeta_max_global": data.get("cowling_zeta_max"),
        "sigma_a_median_primary": float(np.median(prim["a_err"][strong]))
        if np.any(strong) else None,
        "cond_median_primary": float(np.median(prim["cond"][strong]))
        if np.any(strong) else None,
    }
    # fuga/acoplamiento por canal ajeno: pico |a_l'm| / escala de la señal
    modes = [tuple(m) for m in data["modes"]]
    scale_sig = float(np.max(np.abs(anch["a"][strong]))) if np.any(strong) else 1.0
    other = {}
    for lv in sorted({m[0] for m in modes} - {l}):
        peak = max(float(np.max(np.abs(fit_channel(data, m, ANCHOR)["a"][strong])))
                   for m in modes if m[0] == lv) if np.any(strong) else 0.0
        other[f"l{lv}"] = peak / scale_sig
    out["offchannel_peak_over_signal"] = other

    if truth is not None:
        ts_1d, a_1d = truth
        ok = strong & (t <= ts_1d[-1])
        a_ref = np.interp(t[ok], ts_1d, a_1d)
        scale_ref = float(np.max(np.abs(a_1d)))
        for name, fit in (("primary", prim), ("anchor", anch)):
            dev = np.abs(fit["a"][ok] - a_ref) / scale_ref
            out[f"dev_vs_1d_{name}"] = {"max": float(np.max(dev)),
                                        "median": float(np.median(dev))}
        out["scale_1d"] = scale_ref
    return out


def discriminator(runs: dict[str, dict], pair: tuple[str, str], l: int) -> dict | None:
    """a_hat/a_lin del canal (l,0), fase fuerte común, primario + resúmenes
    robustos (mediana/IQR, cociente de picos y de normas L2)."""
    lab_l, lab_h = pair
    if lab_l not in runs or lab_h not in runs:
        return None
    dl, dh = runs[lab_l], runs[lab_h]
    ch = (l, 0)
    pl, ph = fit_channel(dl, ch, PRIMARY), fit_channel(dh, ch, PRIMARY)
    sl = strong_mask(dl["t"], fit_channel(dl, ch, ANCHOR)["a"])
    sh = strong_mask(dh["t"], fit_channel(dh, ch, ANCHOR)["a"])
    a_h = np.interp(dl["t"], dh["t"], ph["a"])
    s_h = np.interp(dl["t"], dh["t"], sh.astype(float)) > 0.5
    both = sl & s_h
    if not np.any(both):
        return None
    ratio = a_h[both] / pl["a"][both]
    peak_ratio = float(np.max(np.abs(a_h[both])) / np.max(np.abs(pl["a"][both])))
    l2_ratio = float(np.sqrt(np.sum(a_h[both] ** 2) / np.sum(pl["a"][both] ** 2)))
    return {"channel": list(ch), "n_samples": int(ratio.size),
            "ratio_median": float(np.median(ratio)),
            "ratio_iqr": [float(np.percentile(ratio, 25)),
                          float(np.percentile(ratio, 75))],
            "peak_ratio": peak_ratio, "l2_ratio": l2_ratio}


def ladder_convergence(runs: dict[str, dict], labels: list[str], l: int) -> dict | None:
    """Auto-convergencia del a(t) primario sobre la escalera (grueso→fino):
    diferencias sucesivas en la fase fuerte del rung medio + orden estimado.
    Caveat F1: remallado no anidado mete ruido — el orden es indicativo."""
    present = [lab for lab in labels if lab in runs]
    if len(present) < 2:
        return None
    mid = runs[present[min(1, len(present) - 1)]]
    ch = (l, 0)
    t_ref = mid["t"]
    strong = strong_mask(t_ref, fit_channel(mid, ch, ANCHOR)["a"])
    series = {}
    for lab in present:
        d = runs[lab]
        series[lab] = np.interp(t_ref, d["t"], fit_channel(d, ch, PRIMARY)["a"])
    scale = float(np.max(np.abs(series[present[-1]][strong])))
    diffs = {}
    for a, b in zip(present[:-1], present[1:]):
        rms = float(np.sqrt(np.mean((series[a] - series[b])[strong] ** 2)))
        diffs[f"{a} vs {b}"] = rms / scale
    out = {"diff_rms_over_scale": diffs, "scale": scale}
    if len(present) >= 3:
        vals = list(diffs.values())
        if vals[-1] > 0:
            q = vals[-2] / vals[-1]
            out["order_estimate"] = float(np.log(q) / np.log(1.4)) if q > 0 else None
    return out


# ----------------------------------------------------------------------
# figura
# ----------------------------------------------------------------------

def make_figure(runs: dict[str, dict], truths: dict, matrix_runs: list[dict],
                suffix: str, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) escalera l=0: primario por rung vs verdad, lineal y mexhat
    ax = axes[0, 0]
    colors = {"linear": ("C0", "Blues"), "mexhat": ("C3", "Reds")}
    lcs = sorted({r["lc"] for r in matrix_runs if r["l"] == 0}, reverse=True)
    for pot, (c, cmap) in colors.items():
        for i, lc in enumerate(lcs):
            lab = next((f"{pot}_l0_lc{lc:.3f}{s}" for s in (suffix, "")
                        if f"{pot}_l0_lc{lc:.3f}{s}" in runs), None)
            if lab is None:
                continue
            d = runs[lab]
            a = fit_channel(d, (0, 0), PRIMARY)["a"]
            ax.plot(d["t"], a, color=c, alpha=0.35 + 0.3 * i, lw=1.1,
                    label=f"{pot} lc={lc:g}")
        if (pot, 0) in truths:
            ts, a1 = truths[(pot, 0)]
            ax.plot(ts, a1, ls="--", color="k" if pot == "linear" else "0.5",
                    lw=1.3, label=f"oráculo {pot}")
    ax.set_xlabel("t/M"); ax.set_ylabel(r"$a_{00}(t)$ (o1 [0.1,0.5])")
    ax.set_title("escalera l=0 vs oráculo"); ax.axhline(0, color="gray", lw=0.5)
    ax.legend(fontsize=6); ax.set_xlim(0, None)

    # (b) modos l=1,2 (rung más fino disponible) vs oráculo lineal
    ax = axes[0, 1]
    for l, ls in ((1, "-"), (2, ":")):
        for pot, (c, _) in colors.items():
            labs = [f"{pot}_l{l}_lc{lc:.3f}{s}"
                    for lc in sorted({r["lc"] for r in matrix_runs if r["l"] == l})
                    for s in (suffix, "")]
            lab = next((x for x in labs if x in runs), None)
            if lab is None:
                continue
            d = runs[lab]
            a = fit_channel(d, (l, 0), PRIMARY)["a"]
            ax.plot(d["t"], a, color=c, ls=ls, lw=1.2, label=f"{pot} l={l}")
        if (l, "linear") and ("linear", l) in truths:
            ts, a1 = truths[("linear", l)]
            ax.plot(ts, a1, "k", ls=ls, lw=0.9, alpha=0.6,
                    label=f"oráculo lin l={l}")
    ax.set_xlabel("t/M"); ax.set_ylabel(r"$a_{l0}(t)$")
    ax.set_title("modos l>0 (canal (l,0))"); ax.axhline(0, color="gray", lw=0.5)
    ax.legend(fontsize=6); ax.set_xlim(0, None)

    # (c) discriminador por l (primario, fase fuerte común)
    ax = axes[1, 0]
    for l, c in ((0, "C2"), (1, "C4"), (2, "C5")):
        lcs_l = sorted({r["lc"] for r in matrix_runs if r["l"] == l})
        if not lcs_l:
            continue
        lc = lcs_l[0]
        pair = [next((f"{pot}_l{l}_lc{lc:.3f}{s}" for s in (suffix, "")
                      if f"{pot}_l{l}_lc{lc:.3f}{s}" in runs), None)
                for pot in ("linear", "mexhat")]
        if None in pair:
            continue
        dl, dh = runs[pair[0]], runs[pair[1]]
        ch = (l, 0)
        pl = fit_channel(dl, ch, PRIMARY)["a"]
        ph = np.interp(dl["t"], dh["t"], fit_channel(dh, ch, PRIMARY)["a"])
        sl = strong_mask(dl["t"], fit_channel(dl, ch, ANCHOR)["a"])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(pl) > 0, ph / pl, np.nan)
        ax.plot(dl["t"][sl], ratio[sl], color=c, lw=1.4,
                label=f"l={l} (lc={lc:g})")
    ax.axhline(1, color="gray", lw=0.5)
    ax.set_xlabel("t/M"); ax.set_ylabel(r"$a^{\rm hat}_{l0}/a^{\rm lin}_{l0}$")
    ax.set_title("discriminador de H2 por modo (dato idéntico)")
    ax.legend(fontsize=7); ax.set_ylim(0, 2)

    # (d) acoplamiento de modos del mexhat l=2: canales l=4 y l=0 vs lineal
    ax = axes[1, 1]
    lcs2 = sorted({r["lc"] for r in matrix_runs if r["l"] == 2})
    if lcs2:
        lc = lcs2[0]
        for pot, (c, _) in colors.items():
            lab = next((f"{pot}_l2_lc{lc:.3f}{s}" for s in (suffix, "")
                        if f"{pot}_l2_lc{lc:.3f}{s}" in runs), None)
            if lab is None:
                continue
            d = runs[lab]
            modes = [tuple(m) for m in d["modes"]]
            anch2 = fit_channel(d, (2, 0), ANCHOR)["a"]
            strong = strong_mask(d["t"], anch2)
            scale = float(np.max(np.abs(anch2[strong]))) if np.any(strong) else 1.0
            for lv, ls in ((4, "-"), (0, ":")):
                mm = [m for m in modes if m[0] == lv]
                if not mm:
                    continue
                peak = np.max(np.abs(np.stack(
                    [fit_channel(d, m, ANCHOR)["a"] for m in mm])), axis=0)
                ax.semilogy(d["t"], peak / scale, color=c, ls=ls, lw=1.2,
                            label=f"{pot} l={lv}")
        ax.set_xlabel("t/M")
        ax.set_ylabel(r"$\max_m |a_{lm}|/{\rm escala}\ l{=}2$")
        ax.set_title("acoplamiento 2⊗2 (mexhat) vs junk (lineal), corridas l=2")
        ax.legend(fontsize=7)

    fig.suptitle("F2 producción interior: escalera, modos y discriminador de H2")
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "production.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fast", action="store_true",
                        help="matriz reducida para validar el pipeline")
    parser.add_argument("--skip-runs", action="store_true",
                        help="solo post-proceso de corridas existentes")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    if args.fast:
        results_dir = REPO / "results" / "phase2_production_fast"
        out_dir = results_dir  # no tocar docs/ en modo fast
        t_end = 6.0
    else:
        results_dir = REPO / "results" / "phase2_production"
        out_dir = REPO / "docs" / "research" / "phase2" / "production"
        t_end = 12.0
    data_dir = out_dir / "data"
    fig_dir = out_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    mat = matrix(args.fast)

    # ---- etapa 0: lumping A/B (solo modo completo; fast valida pipeline) ----
    if args.fast:
        ml = {"adopted": False, "reason": "fast mode"}
    elif args.skip_runs:
        # re-análisis: conservar el veredicto del A/B de la corrida completa
        # (decide qué variante de corridas cargar y no debe perderse del JSON)
        try:
            prev = json.loads((out_dir / "production.json").read_text())
            ml = prev.get("mass_lumping_ab") or {"adopted": False,
                                                 "reason": "no previous verdict"}
        except (OSError, json.JSONDecodeError):
            ml = {"adopted": False, "reason": "no previous verdict"}
    else:
        print("[0/4] A/B de mass lumping en la config de producción...", flush=True)
        ml = lumping_ab(results_dir, t_end, args.force)
    lumping = bool(ml.get("adopted"))
    suffix = "_ml" if lumping else ""

    # ---- etapa 1: matriz de corridas (pool) ----
    print(f"[1/4] matriz: {len(mat)} corridas (lumping "
          f"{'ON' if lumping else 'OFF'}, workers={args.workers})...", flush=True)
    rundirs: dict[str, Path] = {}
    walls: dict[str, float] = {}
    if args.skip_runs:
        # cargar SOLO la variante que dicta el veredicto del lumping (una
        # corrida _ml rechazada no debe colarse como rung de producción) y
        # aplicar el mismo fallback de reuso del humo que el modo completo
        for spec in mat:
            npz = find_npz(results_dir / f"run_{spec['label']}{suffix}")
            if npz is not None:
                rundirs[spec["label"] + suffix] = npz.parent.parent
                continue
            if (not lumping and not args.fast and spec["l"] == 0
                    and abs(spec["lc"] - LC_BASE) < 1e-12):
                smoke_npz = find_npz(SMOKE_RESULTS / f"run_{spec['pot']}")
                if smoke_npz is not None:
                    rundirs[spec["label"]] = smoke_npz.parent.parent
    else:
        # si el lumping NO se adoptó, el rung base de l=0 reusa el humo
        pool_specs = []
        for spec in mat:
            if (not lumping and not args.fast and spec["l"] == 0
                    and abs(spec["lc"] - LC_BASE) < 1e-12):
                smoke_npz = find_npz(SMOKE_RESULTS / f"run_{spec['pot']}")
                if smoke_npz is not None:
                    rundirs[spec["label"]] = smoke_npz.parent.parent
                    print(f"  {spec['label']}: reuso corrida del humo", flush=True)
                    continue
            pool_specs.append(spec)
        # más caras primero (makespan): costo ~ pot × lc^-4
        def est_cost(s: dict) -> float:
            return (3.0 if s["pot"] == "mexhat" else 1.0) * s["lc"] ** -4
        pool_specs.sort(key=est_cost, reverse=True)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futs = [pool.submit(run_one, spec, results_dir, t_end, lumping,
                                args.force) for spec in pool_specs]
            for fut in futs:
                label, rundir, wall = fut.result()
                walls[label] = wall
                if rundir is not None:
                    rundirs[label] = rundir

    # ---- etapa 2: referencias 1D ----
    print("[2/4] referencias del oráculo 1D...", flush=True)
    ls_present = sorted({r["l"] for r in mat})
    truths: dict = {}
    for pot in POTS:
        for l in ls_present:
            ref = oracle_reference(pot, l, data_dir, args.fast)
            if ref is not None:
                truths[(pot, l)] = deep_truth(ref, l)

    # ---- etapa 3: análisis ----
    print("[3/4] análisis...", flush=True)
    runs = {lab: load_run(rd) for lab, rd in rundirs.items()}
    summary: dict = {"runs": {}, "walls_s": walls, "mass_lumping_ab": ml,
                     "lumping_adopted": lumping}
    for spec in mat:
        lab = spec["label"] + suffix if spec["label"] + suffix in runs \
            else spec["label"]
        if lab not in runs:
            summary["runs"][spec["label"]] = {"status": "MISSING"}
            continue
        truth = truths.get((spec["pot"], spec["l"]))
        summary["runs"][lab] = analyze_run(lab, runs[lab], spec["l"], truth)

    # discriminador por (l, lc)
    summary["discriminator"] = {}
    for l in ls_present:
        for lc in sorted({r["lc"] for r in mat if r["l"] == l}):
            pair_labels = []
            for pot in ("linear", "mexhat"):
                base = f"{pot}_l{l}_lc{lc:.3f}"
                pair_labels.append(base + suffix if base + suffix in runs else base)
            disc = discriminator(runs, tuple(pair_labels), l)
            if disc is not None:
                summary["discriminator"][f"l{l}_lc{lc:.3f}"] = disc
    # el mismo cociente en el oráculo (solo l=0: mexhat l>0 no tiene oráculo)
    if ("linear", 0) in truths and ("mexhat", 0) in truths:
        ts_l, a_l = truths[("linear", 0)]
        ts_h, a_h = truths[("mexhat", 0)]
        tt = ts_l[(ts_l >= T_MIN) & (ts_l <= min(ts_l[-1], ts_h[-1]))]
        rl = np.interp(tt, ts_l, a_l)
        rh = np.interp(tt, ts_h, a_h)
        st = np.abs(rl) >= STRONG_FRACTION * np.max(np.abs(rl))
        ratio = rh[st] / rl[st]
        summary["discriminator"]["oracle_1d_l0"] = {
            "ratio_median": float(np.median(ratio)),
            "peak_ratio": float(np.max(np.abs(rh[st])) / np.max(np.abs(rl[st]))),
            "l2_ratio": float(np.sqrt(np.sum(rh[st] ** 2) / np.sum(rl[st] ** 2))),
        }

    # escalera de convergencia por potencial (l=0; y l=2 si hay 2 rungs)
    summary["ladder"] = {}
    for pot in POTS:
        for l in ls_present:
            lcs = sorted({r["lc"] for r in mat if r["l"] == l}, reverse=True)
            if len(lcs) < 2:
                continue
            labels = []
            for lc in lcs:
                base = f"{pot}_l{l}_lc{lc:.3f}"
                labels.append(base + suffix if base + suffix in runs else base)
            conv = ladder_convergence(runs, labels, l)
            if conv is not None:
                summary["ladder"][f"{pot}_l{l}"] = conv

    # ---- etapa 4: figura + resumen ----
    print("[4/4] figura + resumen...", flush=True)
    try:
        make_figure(runs, truths, mat, suffix, fig_dir)
    except Exception as exc:  # la figura no debe tumbar los números
        print(f"  figura falló: {exc}", flush=True)
    summary["protocol"] = {
        "pulse": PULSE, "pots": POTS,
        "primary": {"window": list(PRIMARY[0]), "order": PRIMARY[1]},
        "anchor": {"window": list(ANCHOR[0]), "order": ANCHOR[1]},
        "truth_1d": {"window": list(TRUTH_WINDOW), "order": TRUTH_ORDER},
        "normalization": "c00 = sqrt(4pi)*u (l=0); c_l0 = u_l (l>=1)",
        "matrix": mat, "t_end": t_end, "fast_mode": bool(args.fast),
    }
    summary["total_wall_seconds"] = time.perf_counter() - t0
    with open(out_dir / "production.json", "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    # ---- resumen en consola ----
    print("\n=== producción interior: números clave ===")
    print(f"lumping: {'ADOPTADO' if lumping else 'OFF'} "
          f"({ml.get('dev_median', float('nan')):.2%} mediana A/B)"
          if "dev_median" in ml else
          f"lumping: OFF ({ml.get('reason', '')})")
    for lab, res in summary["runs"].items():
        if res.get("status") == "MISSING":
            print(f"  {lab}: FALTA")
            continue
        dev = res.get("dev_vs_1d_primary")
        devtxt = (f"dev1D {dev['max']:.1%}/{dev['median']:.1%}"
                  if dev else "sin oráculo")
        off = res.get("offchannel_peak_over_signal", {})
        offtxt = " ".join(f"{k}:{v:.1%}" for k, v in off.items())
        print(f"  {lab}: {devtxt}; off-channel {offtxt}")
    print("\ndiscriminador a_hat/a_lin (canal (l,0), primario):")
    for key, d in summary["discriminator"].items():
        if key.startswith("oracle"):
            print(f"  {key}: mediana {d['ratio_median']:.3f}, "
                  f"picos {d['peak_ratio']:.3f}, L2 {d['l2_ratio']:.3f}")
        else:
            print(f"  {key}: mediana {d['ratio_median']:.3f} "
                  f"IQR [{d['ratio_iqr'][0]:.3f}, {d['ratio_iqr'][1]:.3f}], "
                  f"picos {d['peak_ratio']:.3f}, L2 {d['l2_ratio']:.3f}")
    print("\nescalera (rms de diferencias sucesivas / escala):")
    for key, c in summary["ladder"].items():
        diffs = ", ".join(f"{k}: {v:.1%}" for k, v in
                          c["diff_rms_over_scale"].items())
        order = c.get("order_estimate")
        print(f"  {key}: {diffs}" + (f" → orden ~{order:.1f}" if order else ""))
    print(f"\nListo en {summary['total_wall_seconds']:.0f} s. "
          f"JSON en {out_dir / 'production.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
