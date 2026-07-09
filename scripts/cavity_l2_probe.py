#!/usr/bin/env python3
"""Fase 1 — mitigación del modo de cavidad: probe l=2.

El capítulo de convergencia (docs/research/phase1/convergence/note.md)
demostró que la espectroscopía l=1 a R=20 es fit-limited: el ring (período
21M, τ=10.2M) deja <1 ciclo utilizable sobre un modo de cavidad
cuasi-estacionario (Mω=0.209) atrapado entre la barrera (~3M) y la esponja
(r>15). Dos intentos de rescate por post-proceso (ajuste conjunto
ring+cavidad y sustracción de cavidad ajustada en la cola) fallan porque
la cavidad aún se está llenando durante la ventana del ring — no es
estacionaria ahí y extrapolarla la falsea.

Este probe mide la vía física barata: l=2 (período 13M ⇒ ~2× ciclos útiles
en la misma ventana; frecuencia 0.484 bien separada de la cavidad). Corre
la MISMA configuración de la escalera (R=20, r_ext=6, r0=8, w=2, t_end=70)
con Y_20, ajusta contra Leaver l=2 y caracteriza el suelo de la cola.

Uso:  python scripts/cavity_l2_probe.py [--lcs 1.4 1.0] [--t-end 70]
Salidas en docs/research/phase1/cavity/: waveform_l2_lc*.npz, summary.json,
l2_probe.png.
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
OUT = REPO / "docs" / "research" / "phase1" / "cavity"

LC_INNER_RATIO = 3.75  # misma ley de graduación que la escalera l=1


def tail_lines(ts, sig, t_min):
    """Líneas cuasi-estacionarias de la cola (ajuste continuo, sin bins de
    FFT). Implementación compartida: rsd.analysis.ringdown.fit_tail_lines."""
    from rsd.analysis.ringdown import fit_tail_lines

    return fit_tail_lines(ts, sig, t_min)


def fit_windows(ts, sig, ref, window=26.0, offsets=(0.0, 2.0, 4.0, 6.0, 8.0)):
    """Prony por abanico de ventanas ancladas al pico (como la escalera)."""
    from rsd.analysis.ringdown import fit_ringdown_modes

    i0 = np.searchsorted(ts, 12.0)
    t_pk = float(ts[int(np.argmax(np.abs(sig[i0:]))) + i0])
    ws, gs = [], []
    for off in offsets:
        t_min = t_pk + off
        t_max = min(t_pk + off + window, float(ts[-1]) - 1.0)
        try:
            modes = fit_ringdown_modes(ts, sig, t_min, t_max, modes=4)
        except ValueError:
            continue
        if modes:
            ws.append(modes[0][0])
            gs.append(modes[0][1])
    ws, gs = np.array(ws), np.array(gs)
    return {
        "t_peak": t_pk,
        "omega_re": float(ws.mean()), "omega_re_std": float(ws.std()),
        "omega_im": float(-gs.mean()), "omega_im_std": float(gs.std()),
        "err_re": float(abs(ws.mean() - ref.real) / abs(ref.real)),
        "err_im": float(abs(-gs.mean() - ref.imag) / abs(ref.imag)),
        "n_windows": int(len(ws)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lcs", type=float, nargs="+", default=[1.4, 1.0])
    ap.add_argument("--t-end", type=float, default=70.0)
    ap.add_argument("--w", type=float, default=2.0,
                    help="ancho del pulso (2.0 = escalera; 1.0 desplaza la "
                         "excitación a ω altas: más ring, menos cavidad)")
    ap.add_argument("--r-ext", type=float, default=6.0,
                    help="radio de extracción (6.0 = escalera; ~4 acerca la "
                         "extracción a la barrera, donde el modo atrapado "
                         "es más débil)")
    ap.add_argument("--tag", type=str, default="",
                    help="sufijo para los archivos de salida (p.ej. _w1)")
    ap.add_argument("--refit", action="store_true",
                    help="re-analiza waveforms guardados sin re-evolucionar")
    args = ap.parse_args()

    from rsd.analysis.leaver import schwarzschild_qnm
    from rsd.analysis.ringdown import evolve_kerr_ringdown

    ref = schwarzschild_qnm(l=2, n=0)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    summary = {"leaver_l2": [ref.real, ref.imag], "pulse_w": args.w,
               "runs": {}}

    for lc in args.lcs:
        lc_inner = lc / LC_INNER_RATIO
        wf_path = OUT / f"waveform_l2{tag}_lc{lc:g}.npz"
        if args.refit and wf_path.exists():
            d = np.load(wf_path)
            ts, sig = d["ts"], d["c20"]
            wall = 0.0
            print(f"[lc={lc}] re-ajustando waveform existente...", flush=True)
        else:
            print(f"[lc={lc}] evolucionando l=2 (lc_inner={lc_inner:.3f}, "
                  f"w={args.w}, t_end={args.t_end})...", flush=True)
            t0 = time.perf_counter()
            ts, sig = evolve_kerr_ringdown(
                a=0.0, l=2, m_abs=0, lc=lc, lc_inner=lc_inner,
                t_end=args.t_end, w=args.w, r_ext=args.r_ext,
            )
            wall = time.perf_counter() - t0
            np.savez_compressed(wf_path, ts=ts, c20=sig)

        fit = fit_windows(ts, sig, ref)
        peak = float(np.max(np.abs(sig[np.searchsorted(ts, 12.0):])))
        floor = float(np.median(np.abs(sig[ts > 55.0])))
        rec = {
            "lc": lc, "lc_inner": lc_inner, "wall_seconds": wall,
            **fit,
            "ring_peak": peak, "tail_floor": floor,
            "peak_over_floor": peak / floor if floor > 0 else np.inf,
            "usable_efolds": float(np.log(peak / floor)) if floor > 0 else np.inf,
            "tail_lines": tail_lines(ts, sig, 40.0),
        }
        summary["runs"][f"lc{lc:g}"] = rec
        print(f"  Mω = {fit['omega_re']:.5f}±{fit['omega_re_std']:.5f} "
              f"({100 * fit['err_re']:.2f}%)  "
              f"-Im = {-fit['omega_im']:.5f}±{fit['omega_im_std']:.5f} "
              f"({100 * fit['err_im']:.2f}%)  "
              f"pico/suelo = {rec['peak_over_floor']:.1f} "
              f"({rec['usable_efolds']:.1f} e-folds)  wall={wall:.0f}s",
              flush=True)

    (OUT / f"summary{tag}.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for lc in args.lcs:
        d = np.load(OUT / f"waveform_l2{tag}_lc{lc:g}.npz")
        ax[0].semilogy(d["ts"], np.abs(d["c20"]), lw=1.2,
                       label=f"l=2{tag}, lc={lc}")
    l1 = REPO / "docs/research/phase1/convergence/waveform_lc1.npz"
    if l1.exists():
        d1 = np.load(l1)
        ax[0].semilogy(d1["ts"], np.abs(d1["c10"]), lw=1.0, ls="--", c="gray",
                       label="l=1, lc=1.0 (escalera)")
    ax[0].set(xlabel="t / M", ylabel="|c_lm(t)|",
              title="Ring l=2 vs suelo de cavidad")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    dfin = np.load(OUT / f"waveform_l2{tag}_lc{args.lcs[-1]:g}.npz")
    m = dfin["ts"] >= 40.0
    s = dfin["c20"][m] - dfin["c20"][m].mean()
    dt = float(np.mean(np.diff(dfin["ts"][m])))
    # FFT con zero-padding ×8 SOLO para display suave; la resolución real
    # sigue siendo Δω = 2π/T_cola ≈ 0.21 — las líneas se estiman por el
    # ajuste continuo (tail_lines), no por picos de FFT.
    n_pad = 8 * len(s)
    freqs = np.fft.rfftfreq(n_pad, dt) * 2 * np.pi
    amp = np.abs(np.fft.rfft(s * np.hanning(len(s)), n=n_pad))
    ax[1].semilogy(freqs, amp, lw=1.4)
    tl = summary["runs"][f"lc{args.lcs[-1]:g}"]["tail_lines"]
    ax[1].axvline(tl["w1"], ls=":", c="gray",
                  label=f"línea 1 (ajuste): {tl['w1']:.3f}")
    ax[1].axvline(tl["w2"], ls=":", c="tab:orange",
                  label=f"línea 2 (ajuste): {tl['w2']:.3f}")
    ax[1].axvline(ref.real, ls="--", c="tab:red",
                  label=f"Leaver l=2 ({ref.real:.3f})")
    ax[1].set(xlabel="Mω", ylabel="|FFT| (zero-pad ×8)",
              title=f"Cola t>40M (lc={args.lcs[-1]}; Δω_FFT≈{2 * np.pi / (dfin['ts'][-1] - 40.0):.2f})",
              xlim=(0, 0.8))
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"l2_probe{tag}.png", dpi=130)
    print(f"salidas -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
