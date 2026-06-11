#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qnm_kerr_sweep.py – Barrido de QNM en spin: evolución FEM vs Leaver.

Para cada spin de la lista evoluciona un pulso entrante con estructura
Y_l|m| sobre Kerr, ajusta el ringdown con Prony y compara las frecuencias
extraídas contra la referencia de fracciones continuas (psyop.analysis.leaver).
Con |m| > 0 la señal real contiene los modos prógrado (m>0) y retrógrado
(m<0): se reportan ambos.

Uso (macOS: CC=/usr/bin/clang):
    python scripts/qnm_kerr_sweep.py --spins 0,0.3,0.6,0.9 --l 1 --m 1 \\
        --t-end 60 --out sweep.csv
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--spins", default="0,0.3,0.6,0.9",
                   help="Lista de spins a/M separados por coma")
    p.add_argument("--l", type=int, default=1)
    p.add_argument("--m", type=int, default=0, help="|m| del perfil angular")
    p.add_argument("--t-end", type=float, default=45.0)
    p.add_argument("--fit-window", default="12,40",
                   help="Ventana de ajuste Prony t_min,t_max")
    p.add_argument("--modes", type=int, default=4, help="Modos Prony")
    p.add_argument("--lc", type=float, default=1.5)
    p.add_argument("--lc-inner", type=float, default=0.4)
    p.add_argument("--out", default=None, help="CSV de salida (opcional)")
    args = p.parse_args()

    from psyop.analysis.leaver import kerr_qnm
    from psyop.analysis.ringdown import evolve_kerr_ringdown, fit_ringdown_modes

    spins = [float(s) for s in args.spins.split(",")]
    t_min, t_max = (float(x) for x in args.fit_window.split(","))
    l, m_abs = args.l, abs(args.m)

    rows = []
    for a in spins:
        print(f"=== a = {a}: evolucionando hasta t = {args.t_end} ===")
        ts, c_lm = evolve_kerr_ringdown(
            a=a, l=l, m_abs=m_abs, t_end=args.t_end,
            lc=args.lc, lc_inner=args.lc_inner,
        )
        fitted = fit_ringdown_modes(ts, c_lm, t_min, t_max, modes=args.modes)
        targets = sorted(
            {kerr_qnm(l, m_abs, a=a), kerr_qnm(l, -m_abs, a=a)},
            key=lambda w: -w.real,
        )
        print(f"  Leaver: " + ", ".join(f"{w:.5f}" for w in targets))
        print(f"  FEM   : " + ", ".join(f"{w:.4f}-{d:.4f}j" for w, d in fitted[:2]))
        for rank, (w_fem, d_fem) in enumerate(fitted[: len(targets)]):
            # emparejar cada modo FEM con el target Leaver más cercano
            w_ref = min(targets, key=lambda w: abs(w.real - w_fem))
            rows.append(
                (a, rank, w_fem, d_fem, w_ref.real, -w_ref.imag,
                 abs(w_fem - w_ref.real) / w_ref.real)
            )

    print("\n a   modo  Re(Mw)_FEM  -Im(Mw)_FEM  Re(Mw)_Leaver  -Im(Mw)_Leaver  err_rel")
    for r in rows:
        print(f"{r[0]:4.2f}  {r[1]}    {r[2]:.4f}      {r[3]:.4f}       "
              f"{r[4]:.5f}        {r[5]:.5f}      {r[6]:.1%}")

    if args.out:
        np.savetxt(
            args.out, np.array(rows, dtype=float), delimiter=",",
            header="a,mode_rank,omega_re_fem,omega_im_fem,omega_re_leaver,omega_im_leaver,rel_err",
            comments="",
        )
        print(f"\nGuardado: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
