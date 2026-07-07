#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 1 — estudio de convergencia espacial hacia Leaver.

Escalera de mallas (lc decreciente, razón lc_inner/lc fija) sobre
Schwarzschild–Kerr-Schild con excisión; para cada resolución se evoluciona
un pulso Y_10 entrante, se extrae c_10(t) y se ajusta el modo fundamental
con Prony sobre un ABANICO de ventanas ancladas al pico del ringdown
(la dispersión entre ventanas es la incertidumbre del ajuste, práctica
estándar en ringdown).

Productos (docs/research/phase1/convergence/):
- ladder.json: ω medido ± σ por resolución, error vs Leaver, órdenes
  observados (contra referencia y por auto-convergencia de waveforms),
  extrapolación de Richardson.
- figuras: error vs h (log-log), waveforms superpuestos.

Metodología aprendida del oráculo 1D: en Kerr-Schild la velocidad saliente
cerca del horizonte es α²(1−2M/r) → el ring llega a la esfera de extracción
con retardo tipo tortuga; la ventana se ancla al pico detectado, no a un
tiempo fijo.

Uso:
  python scripts/convergence_study.py            # escalera estándar (4 mallas)
  python scripts/convergence_study.py --deep     # añade la malla más fina
  python scripts/convergence_study.py --lcs 2.0 1.4
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

from psyop.analysis.leaver import schwarzschild_qnm  # noqa: E402
from psyop.analysis.qnm import estimate_qnm_prony  # noqa: E402
from psyop.analysis.ringdown import evolve_kerr_ringdown  # noqa: E402

OUT = REPO / "docs" / "research" / "phase1" / "convergence"

L_MODE = 1
LC_LADDER = [2.0, 1.4, 1.0, 0.7]
LC_DEEP = 0.5
LC_INNER_RATIO = 3.75          # lc_inner = lc / ratio (misma ley de graduación)
T_END = 70.0
WINDOW_OFFSETS = [0.0, 2.0, 4.0, 6.0, 8.0]
WINDOW_LENGTH = 26.0


def fit_with_uncertainty(ts: np.ndarray, sig: np.ndarray) -> dict:
    """Ajuste del modo oscilatorio dominante sobre un abanico de ventanas.

    Lecciones de la primera pasada (2026-07-07): el waveform extraído toca
    un suelo de discretización que NO decae ~×4 bajo el pico del ring, así
    que (a) las ventanas deben ser cortas (~1 período: el suelo domina las
    largas) y (b) el modo se selecciona por AMPLITUD filtrando transitorios
    de frecuencia ~0 (fit_ringdown_modes, como el test lento calibrado) —
    max|f| elige espurios.
    """
    from psyop.analysis.ringdown import fit_ringdown_modes

    i0 = np.searchsorted(ts, 12.0)
    t_pk = float(ts[int(np.argmax(np.abs(sig[i0:]))) + i0])

    omegas_re, omegas_im = [], []
    for off in WINDOW_OFFSETS:
        t_min = t_pk + off
        t_max = min(t_pk + off + WINDOW_LENGTH, float(ts[-1]) - 1.0)
        try:
            modes = fit_ringdown_modes(ts, sig, t_min, t_max, modes=4)
        except ValueError:
            continue
        if not modes:
            continue
        w_re, w_im_pos = modes[0]
        omegas_re.append(w_re)
        omegas_im.append(-w_im_pos)
    if not omegas_re:
        raise RuntimeError("ningún ajuste de ventana produjo modos")
    return {
        "t_peak": t_pk,
        "omega_re": float(np.mean(omegas_re)),
        "omega_re_std": float(np.std(omegas_re)),
        "omega_im": float(np.mean(omegas_im)),
        "omega_im_std": float(np.std(omegas_im)),
        "n_windows": len(omegas_re),
    }


def observed_order(errors: list, hs: list) -> list:
    """Orden p entre pares consecutivos de la escalera (contra referencia)."""
    orders = []
    for i in range(len(errors) - 1):
        if errors[i] <= 0 or errors[i + 1] <= 0:
            orders.append(None)
            continue
        orders.append(
            float(np.log(errors[i] / errors[i + 1]) / np.log(hs[i] / hs[i + 1]))
        )
    return orders


def self_convergence_order(waveforms: list, hs: list) -> list:
    """Orden por auto-convergencia: ||W_i − W_{i+1}|| sobre malla temporal común."""
    if len(waveforms) < 3:
        return []
    t_lo = max(w[0][0] for w in waveforms)
    t_hi = min(w[0][-1] for w in waveforms)
    t_common = np.linspace(t_lo, t_hi, 800)
    interps = [np.interp(t_common, ts, cs) for ts, cs in waveforms]
    diffs = [
        float(np.linalg.norm(interps[i] - interps[i + 1]))
        for i in range(len(interps) - 1)
    ]
    orders = []
    for i in range(len(diffs) - 1):
        if diffs[i] <= 0 or diffs[i + 1] <= 0:
            orders.append(None)
            continue
        orders.append(
            float(np.log(diffs[i] / diffs[i + 1]) / np.log(hs[i] / hs[i + 1]))
        )
    return orders


def richardson(omega_coarse: float, omega_fine: float, h_coarse: float,
               h_fine: float, p: float) -> float:
    """Extrapolación de Richardson a h→0 con orden p medido."""
    rp = (h_coarse / h_fine) ** p
    return float((rp * omega_fine - omega_coarse) / (rp - 1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true", help="añade lc=0.5")
    ap.add_argument("--lcs", nargs="+", type=float, default=None)
    ap.add_argument("--t-end", type=float, default=T_END)
    ap.add_argument("--refit", action="store_true",
                    help="re-ajusta waveform_lc*.npz existentes sin re-evolucionar")
    args = ap.parse_args()

    lcs = args.lcs or (LC_LADDER + ([LC_DEEP] if args.deep else []))
    OUT.mkdir(parents=True, exist_ok=True)

    ref = schwarzschild_qnm(l=L_MODE, n=0)
    ladder = []
    waveforms = []
    for lc in lcs:
        lc_inner = lc / LC_INNER_RATIO
        wf_path = OUT / f"waveform_lc{lc:g}.npz"
        if args.refit:
            if not wf_path.exists():
                print(f"[lc={lc}] sin waveform guardado; omitido", flush=True)
                continue
            data = np.load(wf_path)
            ts, sig = data["ts"], data["c10"]
            wall = 0.0
            print(f"[lc={lc}] re-ajustando waveform existente...", flush=True)
        else:
            print(f"[lc={lc}] evolucionando (lc_inner={lc_inner:.3f}, "
                  f"t_end={args.t_end})...", flush=True)
            t0 = time.perf_counter()
            ts, sig = evolve_kerr_ringdown(
                a=0.0, l=L_MODE, m_abs=0, lc=lc, lc_inner=lc_inner,
                t_end=args.t_end,
            )
            wall = time.perf_counter() - t0
        fit = fit_with_uncertainty(ts, sig)
        err_re = abs(fit["omega_re"] - ref.real) / abs(ref.real)
        err_im = abs(fit["omega_im"] - ref.imag) / abs(ref.imag)
        entry = {
            "lc": lc,
            "lc_inner": lc_inner,
            **fit,
            "err_re": float(err_re),
            "err_im": float(err_im),
            "wall_seconds": wall,
        }
        ladder.append(entry)
        waveforms.append((ts, sig))
        np.savez_compressed(OUT / f"waveform_lc{lc:g}.npz", ts=ts, c10=sig)
        print(f"  Mω = {fit['omega_re']:.5f}±{fit['omega_re_std']:.5f} "
              f"{fit['omega_im']:+.5f}±{fit['omega_im_std']:.5f}i | "
              f"err = ({100 * err_re:.2f}%, {100 * err_im:.2f}%) | "
              f"{wall:.0f} s", flush=True)
        # guardado incremental: la escalera parcial ya es usable
        _write_summary(ladder, waveforms, ref, lcs)

    _write_summary(ladder, waveforms, ref, lcs, final=True)
    _figures(ladder, waveforms, ref)
    print(f"Listo. Resultados en {OUT}", flush=True)


def _write_summary(ladder, waveforms, ref, lcs, final=False) -> None:
    hs = [e["lc"] for e in ladder]
    errs_re = [e["err_re"] for e in ladder]
    summary = {
        "l": L_MODE,
        "leaver_reference": {"re": ref.real, "im": ref.imag},
        "ladder": ladder,
        "orders_vs_reference_re": observed_order(errs_re, hs),
        "orders_self_convergence": self_convergence_order(waveforms, hs),
    }
    if len(ladder) >= 3 and final:
        p_obs = summary["orders_vs_reference_re"][-1]
        if p_obs and p_obs > 0.5:
            summary["richardson_omega_re"] = richardson(
                ladder[-2]["omega_re"], ladder[-1]["omega_re"],
                ladder[-2]["lc"], ladder[-1]["lc"], p_obs,
            )
    with open(OUT / "ladder.json", "w") as fh:
        json.dump(summary, fh, indent=2)


def _figures(ladder, waveforms, ref) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs = np.array([e["lc"] for e in ladder])
    err_re = np.array([e["err_re"] for e in ladder])
    err_im = np.array([e["err_im"] for e in ladder])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.loglog(hs, err_re, "o-", label="|Δ Re ω|/Re ω")
    ax1.loglog(hs, err_im, "s-", label="|Δ Im ω|/Im ω")
    for p, style in [(1, ":"), (2, "--")]:
        ax1.loglog(hs, err_re[-1] * (hs / hs[-1]) ** p, "k" + style,
                   lw=0.8, label=f"~h^{p}")
    ax1.set_xlabel("lc")
    ax1.set_ylabel("error relativo vs Leaver")
    ax1.set_title(f"Convergencia QNM l={L_MODE} (Schwarzschild-KS)")
    ax1.legend(fontsize=8)

    for (ts, sig), e in zip(waveforms, ladder):
        ax2.plot(ts, sig / 1e-3, lw=0.9, label=f"lc={e['lc']:g}")
    ax2.set_xlabel("t/M")
    ax2.set_ylabel("c₁₀ / A")
    ax2.set_title("waveforms extraídos")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
