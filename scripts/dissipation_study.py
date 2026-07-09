#!/usr/bin/env python3
"""Fase 1 — cuantificación del sesgo del filtro espectral sobre observables.

Corre una configuración interior Schwarzschild-Kerr-Schild fija (excisión en
r_inner = 0.25 M, pulso entrante) variando la disipación del filtro FEM
(`filter_strength`, `filter_order`) y mide cuánto se desplazan los
observables FÍSICOS respecto a la corrida sin filtro (ε = 0):

  - φ interior en el punto de muestreo r* (dentro del horizonte),
  - energía euleriana E(t),
  - energía de Killing E_K(t) (el balance de referencia bajo excisión).

Afirmación de honestidad que cuantifica: el filtro remueve basura de malla
de alta frecuencia sin mover (apreciablemente) la física. Un desplazamiento
pequeño de los observables pese a que el filtro sí extrae energía es el
resultado buscado. Además demuestra el guard de estabilidad (corre primero,
como fail-fast): ε=0.05 cruza ε·dt·λmax ≥ 2 en esta malla y debe ser
rechazado con RuntimeError — la primera versión de este barrido (2026-07-07,
resultados preservados en v1_diverged/) divergió ahí a ~1e148 sin aviso.

Uso:
  python scripts/dissipation_study.py            # barrido completo (t=20M)
  python scripts/dissipation_study.py --smoke     # validación rápida (t=3M, malla gruesa)
  python scripts/dissipation_study.py --tmax 30   # horizonte temporal a medida

Salidas en docs/research/phase1/dissipation/: summary.json, series.npz,
dissipation.png.
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
OUTDIR = REPO / "docs" / "research" / "phase1" / "dissipation"
RSD_BIN = Path.home() / "miniforge3" / "envs" / "rsd-dolfinx" / "bin" / "rsd"

# (etiqueta, filter_strength ε, filter_order)
SWEEP = [
    ("off", 0.0, 2),
    ("o2_e0.005", 0.005, 2),   # linealidad del sesgo en ε (orden 2)
    ("o2_e0.02", 0.02, 2),
    ("o4_e0.02", 0.02, 4),     # la config candidata a producción
]

# Demostración del guard de estabilidad (docs/math/dissipation.md): en esta
# malla ε=0.05 cruza ε·dt·λmax ≥ 2 — la primera versión del barrido divergió
# a ~1e148 aquí; ahora el solver debe RECHAZARLO con RuntimeError informativo.
GUARD_DEMO = ("o2_e0.05", 0.05, 2)


def base_cfg(tmax: float, lc: float, r_inner: float, lc_inner: float) -> dict:
    return {
        "mesh": {"type": "gmsh", "R": 15.0, "lc": lc,
                 "r_inner": r_inner, "lc_inner": lc_inner},
        "metric": {"type": "kerr", "M": 1.0, "a": 0.0},
        "solver": {
            "degree": 1, "cfl": 0.3, "potential_type": "zero",
            "bc_type": "characteristic", "enable_sommerfeld": True,
        },
        "initial_conditions": {
            "type": "gaussian", "A": 0.01, "r0": 5.0, "w": 1.0,
            "v0": 0.0, "direction": "ingoing_curved",
        },
        "analysis": {"sample_point": [0.4, 0.0, 0.0]},
        "evolution": {"t_end": tmax, "output_every": 5, "verbose": False},
        "output": {"dir": "", "qnm_analysis": False,
                   "diagnostics": True, "save_series": True},
    }


def load_csv(path: Path) -> dict:
    with open(path) as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(-1, len(header))
    return {name: data[:, i] for i, name in enumerate(header)}


def _grep_lines(text: str, needle: str) -> list:
    return [ln.strip() for ln in text.splitlines() if needle in ln]


def run_one(label: str, eps: float, order: int, cfg_base: dict, keep: Path,
            expect_reject: bool = False) -> dict:
    cfg = json.loads(json.dumps(cfg_base))
    cfg["solver"]["filter_strength"] = eps
    cfg["solver"]["filter_order"] = order
    outdir = keep / f"run_{label}"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(cfg, tf)
        cfgpath = tf.name
    env = dict(os.environ, CC="/usr/bin/clang")
    t0 = time.time()
    proc = subprocess.run(
        [str(RSD_BIN), "run", "--config", cfgpath, "--output", str(outdir)],
        capture_output=True, text=True, env=env,
    )
    wall = time.time() - t0
    os.unlink(cfgpath)
    both = proc.stdout + "\n" + proc.stderr
    if expect_reject:
        msgs = _grep_lines(both, "Filtro espectral inestable")
        if proc.returncode == 0 or not msgs:
            raise SystemExit(
                f"guard demo {label}: se esperaba rechazo por estabilidad "
                f"(rc={proc.returncode}, msgs={msgs!r})")
        print(f"  [{label}] eps={eps} order={order} RECHAZADO por el guard "
              f"({wall:.0f}s): {msgs[0][:120]}...")
        return {"label": label, "eps": eps, "order": order, "wall": wall,
                "rejected": True, "message": msgs[0]}
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise SystemExit(f"run {label} failed (rc={proc.returncode})")
    # la CLI crea un subdirectorio run_<timestamp> dentro de --output
    ts_files = sorted(outdir.glob("**/series/time_series.csv"))
    if not ts_files:
        raise SystemExit(f"run {label}: no time_series.csv under {outdir}")
    sd = ts_files[-1].parent
    out = {"label": label, "eps": eps, "order": order, "wall": wall}
    # número de amortiguación ε·dt·λmax y ε_max que reporta el solver
    guard = _grep_lines(both, "filtro espectral orden")
    if guard:
        out["guard_log"] = guard[0]
    out["ts"] = load_csv(sd / "time_series.csv")
    out["energy"] = load_csv(sd / "energy.csv")
    kfile = sd / "killing.csv"
    out["killing"] = load_csv(kfile) if kfile.exists() else None
    print(f"  [{label}] eps={eps} order={order} wall={wall:.0f}s "
          f"({len(out['ts']['t'])} samples)")
    return out


def rel_shift(t0, f0, t1, f1) -> dict:
    """Desplazamiento relativo de f1 vs f0 sobre la malla temporal común."""
    lo, hi = max(t0[0], t1[0]), min(t0[-1], t1[-1])
    m = (t0 >= lo) & (t0 <= hi)
    tg = t0[m]
    a = f0[m]
    b = np.interp(tg, t1, f1)
    scale = np.max(np.abs(a)) or 1.0
    linf = np.max(np.abs(b - a)) / scale
    l2 = np.sqrt(np.mean((b - a) ** 2)) / (np.sqrt(np.mean(a ** 2)) or 1.0)
    return {"rel_linf": float(linf), "rel_l2": float(l2)}


def analyze(runs: list) -> dict:
    base = next(r for r in runs if r["label"] == "off")
    t0 = base["ts"]["t"]
    summary = {"baseline": "off", "observables": {}, "runs": {}}
    E0 = base["energy"]["energy"]
    ke_name = None
    if base["killing"] is not None:
        # 2.ª columna = E_K (energía de Killing), sea cual sea su nombre
        ke_name = [k for k in base["killing"] if k != "t"][0]
    for r in runs:
        if r.get("rejected"):
            summary["runs"][r["label"]] = {
                "eps": r["eps"], "order": r["order"], "wall": r["wall"],
                "rejected_by_stability_guard": True,
                "message": r["message"],
            }
            continue
        rec = {"eps": r["eps"], "order": r["order"], "wall": r["wall"]}
        if r.get("guard_log"):
            rec["guard_log"] = r["guard_log"]
        if r["label"] != "off":
            rec["phi_interior"] = rel_shift(
                t0, base["ts"]["phi"], r["ts"]["t"], r["ts"]["phi"])
            rec["energy"] = rel_shift(
                t0, E0, r["energy"]["t"], r["energy"]["energy"])
            if ke_name and r["killing"] is not None:
                rec["killing_energy"] = rel_shift(
                    t0, base["killing"][ke_name],
                    r["killing"]["t"], r["killing"][ke_name])
        # cuánta energía retira el filtro al final (vs baseline final)
        rec["energy_end_ratio"] = float(
            r["energy"]["energy"][-1] / (E0[-1] or 1.0))
        summary["runs"][r["label"]] = rec
    return summary


def make_figure(runs: list, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for r in runs:
        if r.get("rejected"):
            continue
        lbl = f"{r['label']} (ε={r['eps']}, p{r['order']})"
        ax[0].plot(r["ts"]["t"], r["ts"]["phi"], lw=1.6, label=lbl)
        ax[1].plot(r["energy"]["t"], r["energy"]["energy"], lw=1.6, label=lbl)
    ax[0].set(xlabel="t / M", ylabel="φ(r*=0.4M, t)  [interior]",
              title="Campo interior en el punto de muestreo")
    ax[1].set(xlabel="t / M", ylabel="E(t)", title="Energía euleriana")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle("Sesgo del filtro espectral sobre observables interiores "
                 "(Schwarzschild-KS, l=1)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"figura -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="validación rápida: t=3M, malla gruesa, 2 corridas")
    ap.add_argument("--tmax", type=float, default=20.0)
    ap.add_argument("--keep", type=str, default=None,
                    help="directorio para las corridas (default: temporal)")
    args = ap.parse_args()

    if args.smoke:
        cfg = base_cfg(tmax=3.0, lc=2.0, r_inner=0.25, lc_inner=0.15)
        sweep = [("off", 0.0, 2), ("o4_e0.02", 0.02, 4)]
    else:
        cfg = base_cfg(tmax=args.tmax, lc=1.2, r_inner=0.25, lc_inner=0.08)
        sweep = SWEEP

    OUTDIR.mkdir(parents=True, exist_ok=True)
    keep_ctx = (Path(args.keep) if args.keep
                else Path(tempfile.mkdtemp(prefix="dissip_")))
    keep_ctx.mkdir(parents=True, exist_ok=True)
    print(f"=== dissipation study ({'SMOKE' if args.smoke else 'FULL'}), "
          f"runs in {keep_ctx} ===")

    runs = []
    if not args.smoke:
        # primero: valida fail-fast que el guard rechaza ε=0.05 (~2 min)
        lbl, eps, order = GUARD_DEMO
        runs.append(run_one(lbl, eps, order, cfg, keep_ctx, expect_reject=True))
    runs += [run_one(lbl, eps, order, cfg, keep_ctx) for lbl, eps, order in sweep]
    summary = analyze(runs)

    if args.smoke:
        print(json.dumps(summary["runs"], indent=2))
        return 0

    ok = [r for r in runs if not r.get("rejected")]
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(
        OUTDIR / "series.npz",
        **{f"{r['label']}_t": r["ts"]["t"] for r in ok},
        **{f"{r['label']}_phi": r["ts"]["phi"] for r in ok},
        **{f"{r['label']}_Et": r["energy"]["t"] for r in ok},
        **{f"{r['label']}_E": r["energy"]["energy"] for r in ok},
    )
    make_figure(runs, OUTDIR / "dissipation.png")
    print(f"\nsummary -> {OUTDIR/'summary.json'}")
    print(json.dumps(summary["runs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
