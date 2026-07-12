#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de ringdown sobre Kerr: evoluciona un cascarón gaussiano entrante
con estructura angular Y_lm real, extrae c_lm(t) en una esfera de extracción
y ajusta los modos QNM con Prony. Compartido por el barrido en spin
(scripts/qnm_kerr_sweep.py) y los tests de validación contra Leaver.

Física: con un campo REAL, una estructura Y_l|m| excita por igual los modos
co-rotantes (m>0) y contra-rotantes (m<0); en Kerr el frame dragging rompe
la degeneración y la señal extraída contiene AMBAS frecuencias (p.ej. para
l=1, |m|=1, a=0.9: Mω₊ ≈ 0.437 y Mω₋ ≈ 0.243, splitting del 80%). Medir el
splitting en una sola corrida cancela el error sistemático de discretización.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Estructuras angulares reales (sin normalizar: solo importa el contenido
# en frecuencia); ∝ Re(Y_l^|m|) en cartesianas
_ANGULAR_PROFILES: Dict[Tuple[int, int], Callable] = {
    (1, 0): lambda x, y, z, r: z / r,
    (1, 1): lambda x, y, z, r: x / r,
    (2, 0): lambda x, y, z, r: (3.0 * z**2 - r**2) / r**2,
    (2, 1): lambda x, y, z, r: x * z / r**2,
    (2, 2): lambda x, y, z, r: (x**2 - y**2) / r**2,
}


def evolve_kerr_ringdown(
    a: float = 0.0,
    l: int = 1,
    m_abs: int = 0,
    M: float = 1.0,
    R: float = 20.0,
    r_inner: Optional[float] = None,
    lc: float = 1.5,
    lc_inner: float = 0.4,
    r0: float = 8.0,
    w: float = 2.0,
    r_ext: float = 6.0,
    t_end: float = 45.0,
    A: float = 1e-3,
    cfl: float = 0.25,
    sample_every: int = 4,
    geom_order: int = 1,
    sponge_width: float = 5.0,
    comm=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evoluciona el pulso y devuelve (ts, c_lm) con c_lm real extraído.

    El pulso es entrante (Π = ∂ᵣφ + φ/r) para que caiga hacia el agujero
    negro y el ringdown quede limpio tras el tránsito.

    r_inner=None elige el punto medio de la ventana de outflow admisible
    (docs/math/excision_window.md): 1.0 para a=0, ≈1.25 para a=0.9.

    sponge_width es el ancho de la esponja absorbente pegada al borde
    exterior (r > R - sponge_width); dominios grandes (R ≥ 40) usan
    esponja ancha para que el pozo barrera↔esponja quede largo y el modo
    de cavidad llegue tarde (docs/research/phase1/cavity/note.md §4).
    """
    import dolfinx.fem as fem
    from mpi4py import MPI

    from rsd.analysis.extraction import MultipoleExtractor
    from rsd.mesh.gmsh import INNER_BOUNDARY_TAG, build_ball_mesh, get_outer_tag
    from rsd.physics.metrics import KerrSchildCoeffs, kerr_excision_window

    if r_inner is None:
        lo, hi = kerr_excision_window(M, a)
        if lo >= hi:
            raise ValueError(
                f"a={a}: Cartesian-sphere excision window is empty; "
                "spheroidal excision required"
            )
        r_inner = 0.5 * (lo + hi)
    from rsd.solvers.first_order import FirstOrderKGSolver
    from rsd.utils.utils import compute_dt_cfl

    key = (l, abs(m_abs))
    if key not in _ANGULAR_PROFILES:
        raise ValueError(
            f"angular profile (l={l}, |m|={m_abs}) not available; "
            f"choose from {sorted(_ANGULAR_PROFILES)}"
        )
    angular = _ANGULAR_PROFILES[key]
    comm = comm or MPI.COMM_WORLD

    mesh, _, facet_tags = build_ball_mesh(
        R=R, lc=lc, comm=comm, r_inner=r_inner, lc_inner=lc_inner,
        geom_order=geom_order,
    )
    bg = KerrSchildCoeffs(M=M, a=a)
    solver = FirstOrderKGSolver(
        mesh=mesh, domain_radius=R, degree=1, potential_type="zero",
        cfl_factor=cfl,
        sponge={"enabled": True, "width": sponge_width, "strength": 1.0},
    )
    solver.set_background(*bg.build(mesh), rebuild=False)
    solver.enable_sommerfeld(facet_tags, get_outer_tag(facet_tags, default=2), rebuild=False)
    solver.enable_excision(facet_tags, inner_tag=INNER_BOUNDARY_TAG, rebuild=False)
    solver.rebuild_operators()

    def phi_profile(x):
        r = np.maximum(np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2), 1e-12)
        return A * angular(x[0], x[1], x[2], r) * np.exp(-((r - r0) ** 2) / w**2)

    def pi_profile(x):
        r = np.maximum(np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2), 1e-12)
        pert = A * angular(x[0], x[1], x[2], r) * np.exp(-((r - r0) ** 2) / w**2)
        dpert = pert * (-2.0 * (r - r0) / w**2)
        return dpert + pert / r  # entrante

    phi0 = fem.Function(solver.V_scalar)
    phi0.interpolate(phi_profile)
    Pi0 = fem.Function(solver.V_scalar)
    Pi0.interpolate(pi_profile)
    solver.set_initial_conditions(phi0, Pi0)

    extractor = MultipoleExtractor(mesh, radius=r_ext, lmax=l)
    dt = compute_dt_cfl(mesh, cfl=cfl, c_max=bg.max_characteristic_speed(mesh))

    ts, c_lm = [], []
    t, step = 0.0, 0
    while t < t_end - 1e-12:
        step_dt = min(dt, t_end - t)
        solver.ssp_rk3_step(step_dt)
        t += step_dt
        step += 1
        if step % sample_every == 0:
            phi_f, _ = solver.get_fields()
            ts.append(t)
            c_lm.append(extractor.extract(phi_f)[(l, abs(m_abs))])
    return np.array(ts), np.array(c_lm)


def fit_ringdown_modes(
    ts: np.ndarray,
    signal: np.ndarray,
    t_min: float,
    t_max: float,
    modes: int = 4,
    min_omega: float = 0.1,
) -> List[Tuple[float, float]]:
    """Ajusta modos QNM en la ventana [t_min, t_max] vía Prony.

    Devuelve [(Re Mω, -Im Mω), ...] de los modos oscilatorios y decayentes,
    ordenados por amplitud descendente (los transitorios de frecuencia ~0
    se filtran con min_omega).
    """
    from rsd.analysis.qnm import estimate_qnm_prony

    mask = (ts >= t_min) & (ts <= t_max)
    if mask.sum() < 4 * modes:
        raise ValueError(
            f"ringdown window [{t_min}, {t_max}] has only {int(mask.sum())} samples"
        )
    dt_sample = float(np.mean(np.diff(ts[mask])))
    pairs = estimate_qnm_prony(signal[mask], dt_sample, modes=modes, svd_rank=modes)
    out = []
    for f, d in pairs:
        omega_re = 2.0 * np.pi * abs(f)
        if omega_re > min_omega and d > 0:
            out.append((float(omega_re), float(d)))
    return out


def fit_anchored_windows(
    ts: np.ndarray,
    signal: np.ndarray,
    ref: complex,
    window: float = 26.0,
    offsets: Tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0),
    t_search: float = 12.0,
    modes: int = 4,
) -> Dict[str, float]:
    """Prony por abanico de ventanas ancladas al pico del ring.

    El protocolo de la escalera de convergencia y del capítulo de cavidad
    (docs/research/phase1/{convergence,cavity}/note.md): en Kerr-Schild el
    ring llega con retardo tipo tortuga, así que cada ventana
    [t_pk + off, t_pk + off + window] se ancla al pico detectado a partir
    de t_search (el tránsito directo del pulso por la esfera de extracción
    queda antes y no debe capturarse). Se ajusta Prony (modos dominantes)
    por ventana y se reporta media/desviación del abanico: la desviación
    entre ventanas es la incertidumbre de fit citable. ``ref`` es la
    frecuencia compleja de referencia (Leaver, convención Im < 0).

    Devuelve t_peak, n_windows y — si alguna ventana ajustó — omega_re/
    omega_im (media), *_std (scatter del abanico), err_re/err_im relativos
    a ``ref`` y las listas por ventana omega_{re,im}_windows.
    """
    ts = np.asarray(ts, dtype=float)
    signal = np.asarray(signal, dtype=float)
    i0 = int(np.searchsorted(ts, t_search))
    if i0 >= ts.size:
        raise ValueError(f"t_search={t_search} beyond signal end {ts[-1]}")
    t_pk = float(ts[int(np.argmax(np.abs(signal[i0:]))) + i0])
    ws, gs = [], []
    for off in offsets:
        t_min = t_pk + off
        t_max = min(t_pk + off + window, float(ts[-1]) - 1.0)
        try:
            fitted = fit_ringdown_modes(ts, signal, t_min, t_max, modes=modes)
        except ValueError:
            continue
        if fitted:
            ws.append(fitted[0][0])
            gs.append(fitted[0][1])
    out: Dict[str, float] = {"t_peak": t_pk, "n_windows": int(len(ws))}
    if not ws:
        return out
    ws_a, gs_a = np.array(ws), np.array(gs)
    out.update({
        "omega_re": float(ws_a.mean()), "omega_re_std": float(ws_a.std()),
        "omega_im": float(-gs_a.mean()), "omega_im_std": float(gs_a.std()),
        "err_re": float(abs(ws_a.mean() - ref.real) / abs(ref.real)),
        "err_im": float(abs(-gs_a.mean() - ref.imag) / abs(ref.imag)),
        "omega_re_windows": [float(w) for w in ws],
        "omega_im_windows": [float(-g) for g in gs],
    })
    return out


def fit_tail_lines(
    ts: np.ndarray,
    signal: np.ndarray,
    t_min: float,
    w_lo: float = 0.08,
    w_hi: float = 0.48,
    w2_hi: float = 0.65,
    dw: float = 0.0005,
) -> Dict[str, float]:
    """Líneas cuasi-estacionarias de una cola por ajuste CONTINUO en
    frecuencia (VarPro: barrido de ω + amplitudes/fases por lstsq lineal).

    Motivación (docs/research/phase1/cavity/note.md): la FFT de una cola
    corta es engañosa — con T ≈ 30M su resolución es Δω = 2π/T ≈ 0.21 y
    los "picos" caen en la grilla de bins (así nació el espejismo de una
    escalera armónica 0.209/0.419 en la cola l=1, que en realidad son
    líneas en ≈0.20 y ≈0.34). Este ajuste no está limitado por bins.

    Devuelve dict con w1/w2 (línea dominante y segunda), amp1/amp2 y
    rms0/rms1/rms2 (residuo con 0, 1 y 2 líneas): rms2 ≪ rms0 indica una
    cola coherente de dos modos.
    """
    m = ts >= t_min
    t, s = ts[m], signal[m]
    if t.size < 16:
        raise ValueError(f"tail t >= {t_min} has only {int(t.size)} samples")
    rms0 = float(np.sqrt(np.mean((s - s.mean()) ** 2)))
    ones = np.ones_like(t)

    def solve(ws):
        cols = [ones]
        for w in ws:
            cols += [np.cos(w * t), np.sin(w * t)]
        G = np.stack(cols, axis=1)
        coef, *_ = np.linalg.lstsq(G, s, rcond=None)
        rms = float(np.sqrt(np.mean((G @ coef - s) ** 2)))
        return rms, coef

    # 1 línea: barrido fino 1D
    grid1 = np.arange(w_lo, w_hi, dw)
    rms1, w1s = min((solve([w])[0], w) for w in grid1)

    # 2 líneas: el barrido greedy (1 línea y luego la otra) queda SESGADO
    # cuando la separación baja de la resolución de Rayleigh 2π/T — la
    # regresión conjunta no: barrido 2D grueso + refinamiento local.
    coarse = np.arange(w_lo, w2_hi, 0.004)
    best = None
    for i, wa in enumerate(coarse):
        for wb in coarse[i + 1:]:
            if wb - wa < 0.02:
                continue
            r, _ = solve([wa, wb])
            if best is None or r < best[0]:
                best = (r, wa, wb)
    _, wa, wb = best
    # Refinamiento iterativo con re-centrado: la trinchera de rms es
    # DIAGONAL en (wa, wb) — desplazamientos correlacionados de ambas
    # líneas compensan parcialmente — así que el mínimo grueso puede caer
    # a varios pasos del verdadero; una caja fija se queda en el borde.
    for _ in range(8):
        fine_a = np.arange(wa - 0.006, wa + 0.006 + dw / 2, dw)
        fine_b = np.arange(wb - 0.006, wb + 0.006 + dw / 2, dw)
        best = None
        for fa in fine_a:
            for fb in fine_b:
                r, c = solve([fa, fb])
                if best is None or r < best[0]:
                    best = (r, fa, fb, c)
        rms2, wa_new, wb_new, coef = best
        on_edge = (
            abs(wa_new - fine_a[0]) < dw / 2 or abs(wa_new - fine_a[-1]) < dw / 2
            or abs(wb_new - fine_b[0]) < dw / 2 or abs(wb_new - fine_b[-1]) < dw / 2
        )
        wa, wb = wa_new, wb_new
        if not on_edge:
            break

    amp_a = float(np.hypot(coef[1], coef[2]))
    amp_b = float(np.hypot(coef[3], coef[4]))

    def half_width(w_free, w_other):
        """Semiancho de PERFIL (rms ≤ 1.05·rms2 re-optimizando la otra
        línea en cada punto): la incertidumbre honesta de cada línea. Con
        colas cortas y líneas sub-Rayleigh la verosimilitud es una
        trinchera DIAGONAL — barrer un eje con el otro fijo la
        subestima; el perfil sigue la trinchera."""
        others = np.arange(w_other - 0.02, w_other + 0.02 + dw, 2 * dw)
        span = 0.0
        for sign in (-1.0, 1.0):
            w = w_free
            while abs(w - w_free) < 0.05:
                w += sign * dw
                prof = min(solve([w, wo])[0] for wo in others)
                if prof > 1.05 * rms2:
                    break
            span = max(span, abs(w - w_free))
        return float(span)

    dwa = half_width(wa, wb)
    dwb = half_width(wb, wa)
    if amp_b > amp_a:  # w1 = línea dominante por amplitud
        wa, wb, amp_a, amp_b = wb, wa, amp_b, amp_a
        dwa, dwb = dwb, dwa
    return {
        "w1": float(wa), "w2": float(wb),
        "dw1": dwa, "dw2": dwb,
        "amp1": amp_a, "amp2": amp_b,
        "rms0": rms0, "rms1": float(rms1), "rms2": float(rms2),
    }
