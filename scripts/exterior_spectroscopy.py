#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 2 — espectroscopía exterior de producción (l=2): escalera de malla
sobre la config de trabajo del capítulo de cavidad + validación de dominio
R=40 con graduación apareada.

Diseño (docs/research/phase1/cavity/note.md §4, sin cambios):

  * Config de trabajo: l=2, r_ext=6, R=20, pulso r0=8/w=2/A=1e-3, ventanas
    ancladas al pico (abanico window=26M, offsets 0–8M), Prony modes=4,
    referencia Leaver l=2 n=0. El scatter del abanico es la incertidumbre
    de fit citable.
  * Escalera R=20: lc ∈ {1.4, 1.0, 0.7} con lc_inner = lc/3.75. Los rungs
    1.4 y 1.0 REUSAN los waveforms del capítulo de cavidad (config y camino
    de código idénticos — verificado: los commits posteriores solo tocan
    dato inicial config-driven y agregan MultiRadiusExtractor; el refactor
    del fit reproduce summary.json exactamente). El rung 0.7 es nuevo y
    prueba la predicción del diseño: err_re 3.5 % → ~1.5–2 % por ~2.º orden.
  * Validación de dominio R=40, esponja ancha (width 10, r>30) y
    GRADUACIÓN APAREADA: la ley de malla es lc(r) = lc_inner + (lc_out −
    lc_inner)·r/R, así que con lc_out = lc_inner + (lc − lc_inner)·(40/20)
    el perfil lc(r) en r < 20 es idéntico al del rung R=20 — el par aísla
    el efecto de dominio/cavidad del efecto de resolución (y cuesta ~×2,
    no ×8). El pozo barrera↔esponja se alarga (round-trip ~74M vs ~24M):
    el doblete de cavidad llega tarde y la ventana del ring queda limpia.
    Criterio: |ω(R40) − ω(R20)| dentro del scatter del abanico ⇒ el
    sistemático de dominio está acotado y el valor R=20 es citable.

Corridas en subprocesos (una por vez por default: la malla R=40 y la 0.7
conviven mal en 16 GB); idempotente: waveforms existentes se reusan,
--refit re-analiza sin evolucionar. Waveforms citables (pequeños) en
docs/research/phase2/exterior/data/; logs crudos en results/ (gitignorado).

Uso:  python scripts/exterior_spectroscopy.py [--fast] [--force] [--refit]
      [--workers 1] [--extra-lcs 0.5]
      (--fast: matriz diminuta para validar el pipeline end-to-end; escribe
       TODO bajo results/phase2_exterior_fast/ y no toca docs/)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

CAVITY = REPO / "docs" / "research" / "phase1" / "cavity"

# protocolo del capítulo de cavidad — NO cambiar sin recalibrar (canario:
# tests/test_cavity_mode_slow.py)
LC_INNER_RATIO = 3.75
WINDOW, OFFSETS, T_SEARCH = 26.0, (0.0, 2.0, 4.0, 6.0, 8.0), 12.0
TAIL_TMIN, FLOOR_TMIN = 40.0, 55.0
R_BASE = 20.0
PULSE_W, R_EXT = 2.0, 6.0
# barrido tardío en dominios grandes: ventana corta y offsets hasta antes
# del eco de esponja (~t=60 en R=40); pooled sobre off ≥ 10 (overtone < 5 %)
LATE_WINDOW, LATE_MIN_OFFSET = 16.0, 10.0


def matched_lc_out(lc: float, R: float) -> float:
    """lc exterior que reproduce en r < R_BASE la MISMA ley de graduación
    lc(r) = lc_inner + (lc − lc_inner)·r/R_BASE del rung base."""
    lci = lc / LC_INNER_RATIO
    return lci + (lc - lci) * (R / R_BASE)


def matrix(fast: bool, extra_lcs: list[float],
           r40_lcs: list[float]) -> list[dict]:
    if fast:
        lcs, r_val, t_end = [2.8], 28.0, 45.0
    else:
        lcs, r_val, t_end = [1.4, 1.0, 0.7], 40.0, 70.0
    runs = []
    for lc in lcs + sorted(extra_lcs, reverse=True):
        runs.append({
            "name": f"R{R_BASE:g}_lc{lc:g}", "role": "rung",
            "R": R_BASE, "lc": lc, "lc_out": lc,
            "lc_inner": lc / LC_INNER_RATIO,
            "sponge_width": 5.0, "t_end": t_end,
        })
    # validación de dominio: graduación apareada a cada rung pedido (en el
    # dominio grande la ventana del ring queda limpia de suelo de cavidad,
    # así que estos rungs son además la escalera limpia para −Im);
    # esponja ancha proporcional al dominio extra
    for lc_ref in ([lcs[0]] if fast else sorted(r40_lcs, reverse=True)):
        runs.append({
            "name": f"R{r_val:g}_lc{lc_ref:g}match", "role": "domain_check",
            "match_rung": f"R{R_BASE:g}_lc{lc_ref:g}",
            "R": r_val, "lc": lc_ref, "lc_out": matched_lc_out(lc_ref, r_val),
            "lc_inner": lc_ref / LC_INNER_RATIO,
            "sponge_width": 10.0 if not fast else 7.0, "t_end": t_end,
        })
    # reuso de los waveforms del capítulo de cavidad (config idéntica)
    if not fast:
        reuse = {"R20_lc1.4": CAVITY / "waveform_l2_lc1.4.npz",
                 "R20_lc1": CAVITY / "waveform_l2_lc1.npz"}
        for spec in runs:
            if spec["name"] in reuse:
                spec["reuse"] = str(reuse[spec["name"]])
    return runs


# ----------------------------------------------------------------------
# worker (subproceso): una evolución → un npz
# ----------------------------------------------------------------------

def run_worker(spec_json: str) -> int:
    spec = json.loads(spec_json)
    from rsd.analysis.ringdown import evolve_kerr_ringdown

    t0 = time.perf_counter()
    ts, sig = evolve_kerr_ringdown(
        a=0.0, l=2, m_abs=0, R=spec["R"], lc=spec["lc_out"],
        lc_inner=spec["lc_inner"], t_end=spec["t_end"],
        w=PULSE_W, r_ext=R_EXT, sponge_width=spec["sponge_width"],
    )
    wall = time.perf_counter() - t0
    out = Path(spec["out"])
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp, ts=ts, c20=sig, wall_seconds=wall, R=spec["R"],
        lc=spec["lc"], lc_out=spec["lc_out"], lc_inner=spec["lc_inner"],
        sponge_width=spec["sponge_width"], t_end=spec["t_end"],
    )
    os.replace(tmp, out)
    print(f"worker {spec['name']}: {wall:.0f} s, {ts.size} muestras → {out}",
          flush=True)
    return 0


def acquire_waveform(spec: dict, data_dir: Path, log_dir: Path, force: bool,
                     refit: bool) -> tuple[str, Path | None, float | None]:
    """Devuelve (name, ruta npz | None, wall_s | None). Idempotente:
    1) npz propio existente, 2) reuso del capítulo de cavidad, 3) worker."""
    name = spec["name"]
    own = data_dir / f"wf_{name}.npz"
    if own.exists() and not force:
        with np.load(own) as d:
            wall = float(d["wall_seconds"]) if "wall_seconds" in d.files else None
        print(f"  {name}: reuso {own.relative_to(REPO)}", flush=True)
        return name, own, wall
    reuse = spec.get("reuse")
    if reuse and Path(reuse).exists() and not force:
        print(f"  {name}: reuso {Path(reuse).relative_to(REPO)} "
              f"(capítulo de cavidad)", flush=True)
        return name, Path(reuse), None
    if refit:
        print(f"  {name}: FALTA (refit no evoluciona)", flush=True)
        return name, None, None

    print(f"  {name}: evolucionando (R={spec['R']:g}, lc_out={spec['lc_out']:.3f}, "
          f"lc_inner={spec['lc_inner']:.3f}, esponja {spec['sponge_width']:g}, "
          f"t_end={spec['t_end']:g})...", flush=True)
    wspec = {**spec, "out": str(own)}
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker",
         json.dumps(wspec)],
        capture_output=True, text=True,
        env={**os.environ, "CC": "/usr/bin/clang"},
    )
    wall = time.perf_counter() - t0
    (log_dir / f"{name}.log").write_text(
        proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    if proc.returncode != 0 or not own.exists():
        print(f"  {name}: FALLÓ (rc={proc.returncode}) tras {wall:.0f} s — "
              f"log en {log_dir / (name + '.log')}", flush=True)
        return name, None, wall
    print(f"  {name}: {wall:.0f} s → {own.relative_to(REPO)}", flush=True)
    return name, own, wall


# ----------------------------------------------------------------------
# análisis
# ----------------------------------------------------------------------

def analyze_run(spec: dict, npz: Path, ref: complex,
                wall: float | None) -> dict:
    from rsd.analysis.ringdown import fit_anchored_windows, fit_tail_lines

    with np.load(npz) as d:
        ts, sig = np.asarray(d["ts"]), np.asarray(d["c20"])
    rec: dict = {
        "role": spec["role"], "R": spec["R"], "lc": spec["lc"],
        "lc_out": spec["lc_out"], "lc_inner": spec["lc_inner"],
        "sponge_width": spec["sponge_width"],
        "waveform": str(npz.relative_to(REPO)),
        "wall_seconds": wall, "n_samples": int(ts.size),
        "dt_sample": float(np.mean(np.diff(ts))), "t_end": float(ts[-1]),
    }
    rec.update(fit_anchored_windows(ts, sig, ref, window=WINDOW,
                                    offsets=OFFSETS, t_search=T_SEARCH))
    i0 = int(np.searchsorted(ts, T_SEARCH))
    peak = float(np.max(np.abs(sig[i0:])))
    rec["ring_peak"] = peak
    m_floor = ts > FLOOR_TMIN
    if np.count_nonzero(m_floor) >= 8:
        floor = float(np.median(np.abs(sig[m_floor])))
        rec["tail_floor"] = floor
        if floor > 0:
            rec["peak_over_floor"] = peak / floor
            rec["usable_efolds"] = float(np.log(peak / floor))
    if float(ts[-1]) >= TAIL_TMIN + 10.0:
        rec["tail_lines"] = fit_tail_lines(ts, sig, TAIL_TMIN)
    # dominios grandes (sin suelo de cavidad): barrido de ventanas tardías.
    # El Prony NO separa el overtone n=1 (Δω 4 %, decaimiento ×3: el modo
    # dominante absorbe la mezcla y sesga −Im +14–16 % en ventanas
    # tempranas, independiente de la resolución); las ventanas con
    # off ≥ LATE_MIN_OFFSET lo dejan decaer (e^{-10/3.4} ≈ 5 %). Solo es
    # medible a R > R_BASE: en R=20 esas ventanas caen en el suelo.
    if spec["role"] == "domain_check":
        # una ventana por offset (mantiene el mapa offset↔fit aunque
        # alguna ventana no tenga muestras suficientes)
        offs, wres, wims = [], [], []
        for off in np.arange(0.0, 23.0, 2.0):
            one = fit_anchored_windows(ts, sig, ref, window=LATE_WINDOW,
                                       offsets=(float(off),),
                                       t_search=T_SEARCH)
            if one.get("n_windows", 0) == 1:
                offs.append(float(off))
                wres.append(one["omega_re_windows"][0])
                wims.append(one["omega_im_windows"][0])
        if offs:
            rec["late_window_sweep"] = {
                "window": LATE_WINDOW, "offsets": offs,
                "omega_re": wres, "omega_im": wims,
            }
            late = [(o, wr, wi) for o, wr, wi in zip(offs, wres, wims)
                    if o >= LATE_MIN_OFFSET]
            if late:
                wr = np.array([x[1] for x in late])
                wi = np.array([-x[2] for x in late])  # −Im > 0
                rec["late_pooled"] = {
                    "n_windows": len(late), "min_offset": LATE_MIN_OFFSET,
                    "omega_re": float(wr.mean()),
                    "omega_re_std": float(wr.std()),
                    "omega_im_neg": float(wi.mean()),
                    "omega_im_neg_std": float(wi.std()),
                    "err_re_signed": float((wr.mean() - ref.real) / ref.real),
                    "err_im_signed": float((wi.mean() - (-ref.imag))
                                           / (-ref.imag)),
                }
    return rec


def signed_errors(rec: dict, ref: complex) -> tuple[float, float]:
    """(err_re, err_im) FIRMADOS sobre (Re ω, −Im ω) vs Leaver."""
    e_re = (rec["omega_re"] - ref.real) / ref.real
    e_im = ((-rec["omega_im"]) - (-ref.imag)) / (-ref.imag)
    return float(e_re), float(e_im)


def _richardson3(hs: list[float], ws: list[float]) -> dict | None:
    """ω(h) = ω0 + C·h^p con 3 puntos (h descendente). None si las
    diferencias sucesivas cambian de signo (fuera de régimen asintótico)."""
    (h1, h2, h3), (w1, w2, w3) = hs, ws
    d12, d23 = w1 - w2, w2 - w3
    if d12 * d23 <= 0:
        return None
    target = d12 / d23

    def f(p: float) -> float:
        return (h1**p - h2**p) / (h2**p - h3**p) - target

    lo, hi = 0.3, 8.0
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    p = 0.5 * (lo + hi)
    C = d23 / (h2**p - h3**p)
    return {"p": float(p), "omega0": float(w3 - C * h3**p)}


def ladder_analysis(rungs: list[tuple[dict, dict]], ref: complex) -> dict:
    """rungs: [(spec, rec)] gruesa→fina, solo R=20 con fit válido."""
    rows = []
    for spec, rec in rungs:
        if "omega_re" not in rec:
            continue
        e_re, e_im = signed_errors(rec, ref)
        rows.append({"lc": spec["lc"], "err_re_signed": e_re,
                     "err_im_signed": e_im,
                     "omega_re": rec["omega_re"],
                     "omega_im_neg": -rec["omega_im"],
                     "fan_std_re": rec["omega_re_std"],
                     "fan_std_im": rec["omega_im_std"]})
    out: dict = {"rungs": rows}
    # orden vs la verdad EXTERNA (Leaver) por pares sucesivos: con
    # referencia exacta dos rungs bastan para un exponente efectivo
    for comp in ("re", "im"):
        pairs = []
        for a, b in zip(rows[:-1], rows[1:]):
            ea, eb = abs(a[f"err_{comp}_signed"]), abs(b[f"err_{comp}_signed"])
            if ea > 0 and eb > 0:
                pairs.append({"lcs": [a["lc"], b["lc"]],
                              "p": float(np.log(ea / eb)
                                         / np.log(a["lc"] / b["lc"]))})
        out[f"order_vs_leaver_{comp}"] = pairs
    # Richardson de 3 puntos (secundario; requiere monotonía)
    if len(rows) >= 3:
        last = rows[-3:]
        hs = [r["lc"] for r in last]
        for comp, key_val, truth in (("re", "omega_re", ref.real),
                                     ("im", "omega_im_neg", -ref.imag)):
            rich = _richardson3(hs, [r[key_val] for r in last])
            key = f"richardson_{comp}"
            if rich is None:
                out[key] = {"status": "no aplicable: diferencias sucesivas "
                                      "no monótonas (rung grueso fuera de "
                                      "régimen asintótico)"}
            else:
                rich["err_extrap_vs_leaver"] = float(
                    (rich["omega0"] - truth) / truth)
                out[key] = rich
    return out


def domain_check(rec20: dict, rec40: dict, ref: complex) -> dict:
    out: dict = {"match_rung_lc": rec20["lc"], "R": rec40["R"]}
    if "omega_re" not in rec20 or "omega_re" not in rec40:
        out["status"] = "INCOMPLETO (algún fit sin ventanas)"
        return out
    d_re = rec40["omega_re"] - rec20["omega_re"]
    d_im = rec40["omega_im"] - rec20["omega_im"]
    scat_re = max(rec20["omega_re_std"], rec40["omega_re_std"])
    scat_im = max(rec20["omega_im_std"], rec40["omega_im_std"])
    out.update({
        "delta_omega_re": float(d_re), "delta_omega_im": float(d_im),
        "fan_scatter_re": float(scat_re), "fan_scatter_im": float(scat_im),
        "within_fan_scatter": bool(abs(d_re) <= scat_re
                                   and abs(d_im) <= scat_im),
        "err_vs_leaver_R40": {"re": rec40["err_re"], "im": rec40["err_im"]},
        "err_vs_leaver_R20": {"re": rec20["err_re"], "im": rec20["err_im"]},
    })
    if "tail_floor" in rec20 and "tail_floor" in rec40:
        out["tail_floor_ratio_R40_over_R20"] = (
            rec40["tail_floor"] / rec20["tail_floor"])
        out["peak_over_floor"] = {"R20": rec20.get("peak_over_floor"),
                                  "R40": rec40.get("peak_over_floor")}
    for tag, rec in (("R20", rec20), ("R40", rec40)):
        if "tail_lines" in rec:
            tl = rec["tail_lines"]
            out[f"tail_lines_{tag}"] = {k: tl[k] for k in
                                        ("w1", "w2", "dw1", "dw2",
                                         "rms0", "rms2")}
    return out


# ----------------------------------------------------------------------
# figura
# ----------------------------------------------------------------------

def make_figure(recs: dict[str, dict], data: dict[str, tuple], ref: complex,
                fig_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))

    # (a) waveforms
    ax = axes[0, 0]
    for name, (ts, sig) in data.items():
        style = "--" if recs[name]["role"] == "domain_check" else "-"
        ax.semilogy(ts, np.abs(sig), style, lw=1.1,
                    label=f"{name} (lc_out={recs[name]['lc_out']:.2f})")
    ax.axvline(FLOOR_TMIN, color="gray", ls=":", lw=0.8)
    ax.set(xlabel="t/M", ylabel="|c_20(t)|",
           title="ring l=2: escalera R=20 + validación de dominio")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (b)/(c) frecuencias vs lc
    for ax, comp, truth, title in (
            (axes[0, 1], "re", ref.real, "Re Mω vs lc (abanico ± std)"),
            (axes[1, 0], "im", -ref.imag, "−Im Mω vs lc (abanico ± std)")):
        seen = set()
        for name, rec in recs.items():
            if "omega_re" not in rec:
                continue
            val = rec["omega_re"] if comp == "re" else -rec["omega_im"]
            std = rec[f"omega_{comp}_std"]
            if rec["role"] == "rung":
                ax.errorbar([rec["lc"]], [val], yerr=[std], fmt="o", ms=5,
                            capsize=3, color="C0",
                            label=None if "r" in seen else
                            "R=20 (ventanas de diseño)")
                seen.add("r")
            else:
                ax.errorbar([rec["lc"]], [val], yerr=[std], fmt="s", ms=6,
                            capsize=3, color="C3",
                            label=None if "d" in seen else
                            f"R={rec['R']:g} apareado (ventanas de diseño)")
                seen.add("d")
                lp = rec.get("late_pooled")
                if lp:
                    lval = (lp["omega_re"] if comp == "re"
                            else lp["omega_im_neg"])
                    lstd = lp[f"omega_{'re' if comp == 're' else 'im_neg'}_std"]
                    ax.errorbar([rec["lc"] * 1.02], [lval], yerr=[lstd],
                                fmt="^", ms=6, capsize=3, color="C2",
                                label=None if "l" in seen else
                                f"R={rec['R']:g} ventanas tardías "
                                f"(off≥{LATE_MIN_OFFSET:g})")
                    seen.add("l")
        ax.axhline(truth, color="k", ls="--", lw=1.0, label="Leaver l=2 n=0")
        ax.set(xlabel="lc del rung (R=20 equivalente)", title=title)
        ax.invert_xaxis()
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # (d) cola: FFT display + líneas ajustadas
    ax = axes[1, 1]
    shown = False
    for name, (ts, sig) in data.items():
        rec = recs[name]
        if "tail_lines" not in rec:
            continue
        m = ts >= TAIL_TMIN
        s = sig[m] - sig[m].mean()
        dt = float(np.mean(np.diff(ts[m])))
        n_pad = 8 * len(s)
        freqs = np.fft.rfftfreq(n_pad, dt) * 2 * np.pi
        amp = np.abs(np.fft.rfft(s * np.hanning(len(s)), n=n_pad))
        color = "C3" if rec["role"] == "domain_check" else "C0"
        ax.semilogy(freqs, amp, lw=1.2, color=color, label=f"{name} (cola)")
        tl = rec["tail_lines"]
        ax.axvline(tl["w1"], ls=":", color=color, lw=1.0)
        ax.axvline(tl["w2"], ls=":", color=color, lw=0.7)
        shown = True
    if shown:
        ax.axvline(ref.real, ls="--", color="k", lw=1.0, label="Leaver Re")
        ax.set(xlabel="Mω", ylabel="|FFT| (zero-pad ×8, display)",
               title=f"cola t>{TAIL_TMIN:g}M: líneas de cavidad "
                     "(ajuste continuo, punteadas)", xlim=(0, 0.8))
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    else:
        ax.set_axis_off()

    fig.suptitle("F2 espectroscopía exterior de producción (l=2, r_ext=6)")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--fast", action="store_true",
                        help="matriz diminuta para validar el pipeline")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refit", action="store_true",
                        help="re-analiza waveforms existentes sin evolucionar")
    parser.add_argument("--workers", type=int, default=1,
                        help="corridas simultáneas (default 1: R=40 y "
                             "lc=0.7 no conviven en 16 GB)")
    parser.add_argument("--extra-lcs", type=float, nargs="*", default=[],
                        help="rungs R=20 adicionales (p.ej. 0.5 para la "
                             "vía sub-1 %% si el paper la exige)")
    parser.add_argument("--r40-lcs", type=float, nargs="*", default=[1.0],
                        help="rungs base a aparear en R=40 (escalera limpia "
                             "de −Im + chequeos de dominio)")
    args = parser.parse_args()

    if args.worker is not None:
        return run_worker(args.worker)

    from rsd.analysis.leaver import schwarzschild_qnm

    ref = schwarzschild_qnm(l=2, n=0)

    if args.fast:
        out_dir = REPO / "results" / "phase2_exterior_fast"
        data_dir = log_dir = out_dir
        fig_path = out_dir / "spectroscopy.png"
    else:
        out_dir = REPO / "docs" / "research" / "phase2" / "exterior"
        data_dir = out_dir / "data"
        fig_path = out_dir / "figures" / "spectroscopy.png"
        log_dir = REPO / "results" / "phase2_exterior"
    for d in (out_dir, data_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    mat = matrix(args.fast, args.extra_lcs, args.r40_lcs)

    # ---- corridas (pool de subprocesos; secuencial por default) ----
    print(f"[1/3] waveforms: {len(mat)} corridas (workers={args.workers})...",
          flush=True)
    mat_sorted = sorted(mat, key=lambda s: (s["R"] / s["lc_out"]) ** 3,
                        reverse=True)
    paths: dict[str, tuple[Path, float | None]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(acquire_waveform, spec, data_dir, log_dir,
                            args.force, args.refit) for spec in mat_sorted]
        for fut in futs:
            name, npz, wall = fut.result()
            if npz is not None:
                paths[name] = (npz, wall)

    # ---- análisis ----
    print("[2/3] análisis...", flush=True)
    recs: dict[str, dict] = {}
    data: dict[str, tuple] = {}
    for spec in mat:
        if spec["name"] not in paths:
            recs[spec["name"]] = {"status": "MISSING", "role": spec["role"]}
            continue
        npz, wall = paths[spec["name"]]
        recs[spec["name"]] = analyze_run(spec, npz, ref, wall)
        with np.load(npz) as d:
            data[spec["name"]] = (np.asarray(d["ts"]), np.asarray(d["c20"]))

    rungs = [(s, recs[s["name"]]) for s in mat
             if s["role"] == "rung" and recs[s["name"]].get("status") != "MISSING"]
    ladder = ladder_analysis(rungs, ref) if len(rungs) >= 2 else None

    # escalera limpia R=40 (sin suelo de cavidad en la ventana del ring):
    # los rungs apareados ordenados grueso→fino, con lc = rung equivalente
    r40_rungs = sorted(
        [(s, recs[s["name"]]) for s in mat if s["role"] == "domain_check"
         and recs[s["name"]].get("status") != "MISSING"],
        key=lambda t: -t[0]["lc"])
    ladder_r40 = (ladder_analysis(r40_rungs, ref)
                  if len(r40_rungs) >= 2 else None)

    dchecks: dict[str, dict] = {}
    for dspec, rec40 in r40_rungs:
        base = recs.get(dspec["match_rung"])
        if base is not None and base.get("status") != "MISSING":
            dchecks[dspec["name"]] = domain_check(base, rec40, ref)

    # −Im de producción: pooled de ventanas tardías sobre TODOS los rungs
    # R=40 (la dispersión entre ventanas y rungs ES la sistemática de
    # protocolo — citarla, no elegir el abanico más bonito)
    late_re, late_im = [], []
    for _, rec40 in r40_rungs:
        sw = rec40.get("late_window_sweep")
        if not sw:
            continue
        for o, wr, wi in zip(sw["offsets"], sw["omega_re"], sw["omega_im"]):
            if o >= LATE_MIN_OFFSET:
                late_re.append(wr)
                late_im.append(-wi)
    late_pooled_all = None
    if late_re:
        arr_re, arr_im = np.array(late_re), np.array(late_im)
        late_pooled_all = {
            "n_windows": int(arr_re.size),
            "min_offset": LATE_MIN_OFFSET, "window": LATE_WINDOW,
            "omega_re": float(arr_re.mean()),
            "omega_re_std": float(arr_re.std()),
            "omega_im_neg": float(arr_im.mean()),
            "omega_im_neg_std": float(arr_im.std()),
            "err_re_signed": float((arr_re.mean() - ref.real) / ref.real),
            "err_im_signed": float((arr_im.mean() - (-ref.imag))
                                   / (-ref.imag)),
        }

    summary = {
        "leaver_l2": [ref.real, ref.imag],
        "protocol": {
            "l": 2, "m": 0, "r_ext": R_EXT,
            "pulse": {"A": 1e-3, "r0": 8.0, "w": PULSE_W,
                      "direction": "ingoing"},
            "windows": {"window": WINDOW, "offsets": list(OFFSETS),
                        "t_search": T_SEARCH, "prony_modes": 4},
            "tail": {"t_min": TAIL_TMIN, "floor_t_min": FLOOR_TMIN},
            "late_windows": {
                "window": LATE_WINDOW, "offset_grid": "0–22 paso 2",
                "pooled_min_offset": LATE_MIN_OFFSET,
                "rationale": "el Prony no separa el overtone n=1 (Δω 4 %, "
                             "decaimiento ×3): el modo dominante absorbe la "
                             "mezcla y sesga −Im +14–16 % en ventanas "
                             "tempranas, independiente de la resolución; "
                             "off ≥ 10 lo deja decaer a < 5 %. Solo medible "
                             "a R > 20: en R=20 esas ventanas caen en el "
                             "suelo de cavidad",
            },
            "lc_inner_ratio": LC_INNER_RATIO,
            "r40_matching": "lc(r) = lc_inner + (lc_out − lc_inner)·r/R con "
                            "lc_out = lc_inner + (lc − lc_inner)·R/20: "
                            "graduación idéntica en r<20 al rung base — el "
                            "par aísla dominio de resolución",
            "provenance": {
                "R20_lc1.4": "reusado de docs/research/phase1/cavity/"
                             "waveform_l2_lc1.4.npz (2026-07-09)",
                "R20_lc1": "reusado de docs/research/phase1/cavity/"
                           "waveform_l2_lc1.npz (2026-07-09)",
                "code_check": "camino solver/extracción sin cambios desde "
                              "el capítulo de cavidad; fit compartido "
                              "reproduce summary.json exactamente",
            },
            "fast_mode": bool(args.fast),
        },
        "runs": recs,
        "ladder_R20": ladder,
        "ladder_R40_clean": ladder_r40,
        "domain_checks": dchecks,
        "late_pooled_all_R40": late_pooled_all,
    }

    # ---- figura + salida ----
    print("[3/3] figura + resumen...", flush=True)
    try:
        make_figure({k: v for k, v in recs.items() if v.get("status") != "MISSING"},
                    data, ref, fig_path)
    except Exception as exc:  # la figura no debe tumbar los números
        print(f"  figura falló: {exc}", flush=True)
    summary["total_wall_seconds"] = time.perf_counter() - t0
    json_path = out_dir / "spectroscopy.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== espectroscopía exterior l=2: números clave ===")
    print(f"Leaver l=2 n=0: Mω = {ref.real:.5f} {ref.imag:+.5f}i")
    for name, rec in recs.items():
        if rec.get("status") == "MISSING":
            print(f"  {name}: FALTA")
            continue
        if "omega_re" not in rec:
            print(f"  {name}: sin ventanas útiles (t_end={rec['t_end']:g})")
            continue
        e_re, e_im = signed_errors(rec, ref)
        pf = rec.get("peak_over_floor")
        pf_txt = f"  pico/suelo {pf:.1f}" if pf else ""
        print(f"  {name}: Re {rec['omega_re']:.5f}±{rec['omega_re_std']:.5f} "
              f"({e_re:+.2%})  −Im {-rec['omega_im']:.5f}"
              f"±{rec['omega_im_std']:.5f} ({e_im:+.2%})"
              f"  [{rec['n_windows']} ventanas]{pf_txt}")
    for lab, lad in (("R=20", ladder), ("R=40 limpia", ladder_r40)):
        if not lad:
            continue
        print(f"\nescalera {lab} (err firmado vs Leaver):")
        for row in lad["rungs"]:
            print(f"  lc={row['lc']:g}: Re {row['err_re_signed']:+.2%}, "
                  f"−Im {row['err_im_signed']:+.2%}")
        for comp in ("re", "im"):
            for pr in lad.get(f"order_vs_leaver_{comp}", []):
                print(f"  orden {comp} ({pr['lcs'][0]:g}→{pr['lcs'][1]:g}): "
                      f"p ≈ {pr['p']:.1f}")
    for name, dc in dchecks.items():
        if "delta_omega_re" not in dc:
            continue
        verdict = "DENTRO" if dc["within_fan_scatter"] else "FUERA"
        print(f"\ndominio {name} (vs lc={dc['match_rung_lc']:g}): "
              f"Δω_re = {dc['delta_omega_re']:+.5f} "
              f"(scatter {dc['fan_scatter_re']:.5f}), "
              f"Δω_im = {dc['delta_omega_im']:+.5f} "
              f"(scatter {dc['fan_scatter_im']:.5f}) → {verdict} del scatter")
        fr = dc.get("tail_floor_ratio_R40_over_R20")
        if fr is not None:
            print(f"  suelo de cola R40/R20 = {fr:.2f}; pico/suelo "
                  f"{dc['peak_over_floor']['R20']:.1f} → "
                  f"{dc['peak_over_floor']['R40']:.1f}")
    if late_pooled_all:
        lp = late_pooled_all
        print(f"\nventanas tardías R=40 pooled (w={lp['window']:g}, "
              f"off ≥ {lp['min_offset']:g}, {lp['n_windows']} ventanas × "
              f"{len(r40_rungs)} rungs):")
        print(f"  Re {lp['omega_re']:.5f}±{lp['omega_re_std']:.5f} "
              f"({lp['err_re_signed']:+.2%})  "
              f"−Im {lp['omega_im_neg']:.5f}±{lp['omega_im_neg_std']:.5f} "
              f"({lp['err_im_signed']:+.2%})")
    print(f"\nListo en {summary['total_wall_seconds']:.0f} s. "
          f"JSON en {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
