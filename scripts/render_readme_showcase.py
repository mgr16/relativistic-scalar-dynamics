#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera la imagen-vitrina del README (docs/media/research_showcase.png):

  (a) el campo simulado: corte y=0 de φ durante el ringdown l=2 sobre
      Schwarzschild-KS con excisión (evolución corta ad-hoc, ~3 min a
      lc=1.0; el slice se cachea en results/showcase/ para iterar);
  (b) el resultado interior de H2: a00(t) lineal vs mexhat con dato
      idéntico (rung fino de la producción, primario o1 [0.1, 0.5]);
  (c) el resultado exterior: |c20(t)| R=20 vs R=40 apareado contra el
      decaimiento de Leaver (el suelo de cavidad desaparece en R=40).

Los paneles (b)/(c) leen datos ya publicados (results/phase2_production/
y docs/research/phase2/exterior/data/); solo (a) evoluciona.

Uso:  python scripts/render_readme_showcase.py [--skip-sim] [--t-snap 17]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

OUT_PNG = REPO / "docs" / "media" / "research_showcase.png"
CACHE = REPO / "results" / "showcase" / "slice_l2.npz"
EXT_DATA = REPO / "docs" / "research" / "phase2" / "exterior" / "data"
PROD_RESULTS = REPO / "results" / "phase2_production"
ORACLE_DATA = REPO / "docs" / "research" / "phase2" / "interior" / "data"

# paleta de referencia validada (dataviz skill, modo claro)
C_LIN, C_HAT = "#2a78d6", "#1baf7a"   # slots categóricos 1 y 2
INK, INK2 = "#0b0b0b", "#52514e"
GRID = dict(alpha=0.25, lw=0.6)

R_DOM, R_INNER, LC, LC_INNER = 20.0, 1.0, 1.0, 1.0 / 3.75
GRID_N, GRID_HALF = 320, 19.0


# ----------------------------------------------------------------------
# (a) evolución corta + slice
# ----------------------------------------------------------------------

def simulate_slice(t_snap: float) -> dict:
    import dolfinx.fem as fem
    import dolfinx.geometry as geometry
    from mpi4py import MPI

    from rsd.mesh.gmsh import INNER_BOUNDARY_TAG, build_ball_mesh, get_outer_tag
    from rsd.physics.metrics import KerrSchildCoeffs
    from rsd.solvers.first_order import FirstOrderKGSolver
    from rsd.utils.utils import compute_dt_cfl

    print(f"[slice] malla R={R_DOM} lc={LC} + evolución hasta t={t_snap}M...",
          flush=True)
    mesh, _, facet_tags = build_ball_mesh(
        R=R_DOM, lc=LC, comm=MPI.COMM_WORLD, r_inner=R_INNER,
        lc_inner=LC_INNER)
    bg = KerrSchildCoeffs(M=1.0, a=0.0)
    solver = FirstOrderKGSolver(
        mesh=mesh, domain_radius=R_DOM, degree=1, potential_type="zero",
        cfl_factor=0.25,
        sponge={"enabled": True, "width": 5.0, "strength": 1.0})
    solver.set_background(*bg.build(mesh), rebuild=False)
    solver.enable_sommerfeld(facet_tags, get_outer_tag(facet_tags, default=2),
                             rebuild=False)
    solver.enable_excision(facet_tags, inner_tag=INNER_BOUNDARY_TAG,
                           rebuild=False)
    solver.rebuild_operators()

    A, r0, w = 1e-3, 8.0, 2.0

    def angular(x, y, z, r):
        return (3.0 * z**2 - r**2) / r**2  # ∝ Y_20 real

    def phi_profile(x):
        r = np.maximum(np.sqrt(x[0]**2 + x[1]**2 + x[2]**2), 1e-12)
        return A * angular(x[0], x[1], x[2], r) * np.exp(-((r - r0)**2) / w**2)

    def pi_profile(x):
        r = np.maximum(np.sqrt(x[0]**2 + x[1]**2 + x[2]**2), 1e-12)
        pert = A * angular(x[0], x[1], x[2], r) * np.exp(-((r - r0)**2) / w**2)
        return pert * (-2.0 * (r - r0) / w**2) + pert / r  # entrante

    phi0 = fem.Function(solver.V_scalar)
    phi0.interpolate(phi_profile)
    Pi0 = fem.Function(solver.V_scalar)
    Pi0.interpolate(pi_profile)
    solver.set_initial_conditions(phi0, Pi0)

    dt = compute_dt_cfl(mesh, cfl=0.25, c_max=bg.max_characteristic_speed(mesh))
    t, t0 = 0.0, time.perf_counter()
    while t < t_snap - 1e-12:
        step_dt = min(dt, t_snap - t)
        solver.ssp_rk3_step(step_dt)
        t += step_dt
    print(f"[slice] evolución lista en {time.perf_counter() - t0:.0f} s",
          flush=True)

    # muestreo del plano y=0 (el patrón Y_20 muestra sus lóbulos ahí)
    xs = np.linspace(-GRID_HALF, GRID_HALF, GRID_N)
    zs = np.linspace(-GRID_HALF, GRID_HALF, GRID_N)
    X, Z = np.meshgrid(xs, zs, indexing="xy")
    pts = np.zeros((GRID_N * GRID_N, 3))
    pts[:, 0], pts[:, 2] = X.ravel(), Z.ravel()
    rr = np.sqrt(pts[:, 0]**2 + pts[:, 2]**2)
    inside = (rr > R_INNER * 1.02) & (rr < R_DOM * 0.995)

    phi_f, _ = solver.get_fields()
    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts[inside])
    cells = geometry.compute_colliding_cells(mesh, cand, pts[inside])
    vals = np.full(pts.shape[0], np.nan)
    idx = np.where(inside)[0]
    ok_pts, ok_cells, ok_idx = [], [], []
    for k in range(len(idx)):
        links = cells.links(k)
        if len(links) > 0:
            ok_pts.append(pts[idx[k]])
            ok_cells.append(links[0])
            ok_idx.append(idx[k])
    vals[np.array(ok_idx)] = phi_f.eval(
        np.array(ok_pts), np.array(ok_cells, dtype=np.int32)).ravel()
    sl = vals.reshape(GRID_N, GRID_N)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, xs=xs, zs=zs, sl=sl, t_snap=t_snap)
    return {"xs": xs, "zs": zs, "sl": sl, "t_snap": t_snap}


# ----------------------------------------------------------------------
# (b) datos interiores
# ----------------------------------------------------------------------

def interior_series() -> dict | None:
    from rsd.analysis.interior import (fit_log_profile_multipole,
                                       fit_log_profile_series)

    out = {}
    for pot in ("linear", "mexhat"):
        hits = sorted((PROD_RESULTS / f"run_{pot}_l0_lc0.028").glob(
            "run_*/series/interior_profiles.npz"))
        if not hits:
            print(f"[interior] falta rung fino de {pot} — panel omitido",
                  flush=True)
            return None
        d = np.load(hits[-1])
        modes = [tuple(m) for m in d["modes"]]
        fit = fit_log_profile_multipole(d["radii"], d["u"], modes,
                                        (0.1, 0.5), order=1)[(0, 0)]
        out[pot] = (np.asarray(d["t"]), np.asarray(fit["a"]))
    # el cociente citado es el PUBLICADO (protocolo del discriminador de
    # producción — máscara de ancla o0 común), no una variante recalculada
    import json
    prod = json.loads((REPO / "docs" / "research" / "phase2" / "production"
                       / "production.json").read_text())
    out["l2_ratio"] = float(
        prod["discriminator"]["l0_lc0.028"]["l2_ratio"])
    return out


# ----------------------------------------------------------------------
# figura
# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sim", action="store_true",
                    help="reusa el slice cacheado en results/showcase/")
    ap.add_argument("--t-snap", type=float, default=17.0)
    args = ap.parse_args()

    if args.skip_sim and CACHE.exists():
        d = np.load(CACHE)
        sl = {"xs": d["xs"], "zs": d["zs"], "sl": d["sl"],
              "t_snap": float(d["t_snap"])}
        print(f"[slice] reuso {CACHE.relative_to(REPO)}", flush=True)
    else:
        sl = simulate_slice(args.t_snap)

    interior = interior_series()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7))
    fig.patch.set_facecolor("white")

    # ---- (a) slice del campo ----
    ax = axes[0]
    S = sl["sl"] * 1e3  # unidades ×10⁻³ para una colorbar legible
    vmax = float(np.nanquantile(np.abs(S), 0.995))
    im = ax.imshow(S, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[sl["xs"][0], sl["xs"][-1],
                           sl["zs"][0], sl["zs"][-1]],
                   interpolation="bilinear")
    ax.add_patch(Circle((0, 0), R_INNER, color="#111111", zorder=5))
    ax.add_patch(Circle((0, 0), 2.0, fill=False, ls="--", lw=1.0,
                        ec=INK2, zorder=6))
    ax.add_patch(Circle((0, 0), 6.0, fill=False, ls=":", lw=0.8,
                        ec=INK2, zorder=6, alpha=0.7))
    ax.annotate("horizon r=2M", xy=(1.5, -1.5), xytext=(6.0, -14.5),
                color=INK2, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.annotate(r"$r_{ext}=6M$", xy=(-4.3, 4.3), xytext=(-17.5, 13.5),
                color=INK2, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.set_title(f"l=2 ringdown on Schwarzschild — φ slice "
                 f"(t = {sl['t_snap']:g}M)", fontsize=9.5, color=INK)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(r"φ ($\times 10^{-3}$)", fontsize=9, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)

    # ---- (b) H2 interior ----
    ax = axes[1]
    if interior is not None:
        tl, al = interior["linear"]
        th, ah = interior["mexhat"]
        ax.axvspan(4.0, 10.0, color="#f0efe9", zorder=0)
        ax.plot(tl, al, color=C_LIN, lw=2.0, label="free field (V = 0)")
        ax.plot(th, ah, color=C_HAT, lw=2.0,
                label="Higgs vacuum (Mexican hat, u∞ = v)")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xlim(0, 12)
        ax.set_xlabel("t/M", fontsize=9, color=INK2)
        ax.set_ylabel(r"log-slope $a_{00}(t)$", fontsize=9, color=INK2)
        ax.set_title("interior: the Higgs vacuum does NOT change the "
                     "profile\n(identical data, r ∈ [0.1, 0.5]M)",
                     fontsize=9.5, color=INK)
        ax.annotate("strong phase (shaded):  "
                    r"$a^{hat}/a^{lin}$ (L2) = "
                    f"{interior['l2_ratio']:.2f}",
                    xy=(0.5, 0.06), xycoords="axes fraction", ha="center",
                    fontsize=8.5, color=INK)
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.grid(**GRID)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8, colors=INK2)
    else:
        ax.set_axis_off()

    # ---- (c) exterior QNM ----
    ax = axes[2]
    leaver_im = 0.09675877597828784
    styles = {"wf_R20_lc0.7.npz": (C_LIN, "-", "R = 20 (production)"),
              "wf_R40_lc0.7match.npz": (C_HAT, "--",
                                        "R = 40, matched grading")}
    t_pk = peak = None
    for fname, (c, ls, lab) in styles.items():
        d = np.load(EXT_DATA / fname)
        ts, sig = d["ts"], np.abs(d["c20"])
        ax.semilogy(ts, sig, color=c, lw=1.5, ls=ls, label=lab)
        if t_pk is None:
            i0 = np.searchsorted(ts, 12.0)
            j = int(np.argmax(sig[i0:])) + i0
            t_pk, peak = float(ts[j]), float(sig[j])
    tg = np.linspace(t_pk, 62.0, 50)
    ax.semilogy(tg, peak * np.exp(-leaver_im * (tg - t_pk)), ls=":",
                color=INK, lw=1.6, label="Leaver decay (l=2, n=0)")
    ax.annotate("cavity floor\n(R=20 domain artifact)",
                xy=(56, 1.35e-3), ha="center", fontsize=8, color=INK2)
    ax.set_xlim(0, 70)
    ax.set_ylim(2e-6, 4e-3)
    ax.set_xlabel("t/M", fontsize=9, color=INK2)
    ax.set_ylabel(r"$|c_{20}(t)|$ at $r_{ext} = 6M$", fontsize=9, color=INK2)
    ax.set_title("exterior: l=2 QNM vs Leaver\n"
                 "(Re −1.9 %, −Im +5 %; floor removed at R=40)",
                 fontsize=9.5, color=INK)
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    ax.grid(**GRID)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8, colors=INK2)

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=170, facecolor="white")
    print(f"figura → {OUT_PNG.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
