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
    comm=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evoluciona el pulso y devuelve (ts, c_lm) con c_lm real extraído.

    El pulso es entrante (Π = ∂ᵣφ + φ/r) para que caiga hacia el agujero
    negro y el ringdown quede limpio tras el tránsito.

    r_inner=None elige el punto medio de la ventana de outflow admisible
    (docs/math/excision_window.md): 1.0 para a=0, ≈1.25 para a=0.9.
    """
    import dolfinx.fem as fem
    from mpi4py import MPI

    from psyop.analysis.extraction import MultipoleExtractor
    from psyop.mesh.gmsh import INNER_BOUNDARY_TAG, build_ball_mesh, get_outer_tag
    from psyop.physics.metrics import KerrSchildCoeffs, kerr_excision_window

    if r_inner is None:
        lo, hi = kerr_excision_window(M, a)
        if lo >= hi:
            raise ValueError(
                f"a={a}: Cartesian-sphere excision window is empty; "
                "spheroidal excision required"
            )
        r_inner = 0.5 * (lo + hi)
    from psyop.solvers.first_order import FirstOrderKGSolver
    from psyop.utils.utils import compute_dt_cfl

    key = (l, abs(m_abs))
    if key not in _ANGULAR_PROFILES:
        raise ValueError(
            f"angular profile (l={l}, |m|={m_abs}) not available; "
            f"choose from {sorted(_ANGULAR_PROFILES)}"
        )
    angular = _ANGULAR_PROFILES[key]
    comm = comm or MPI.COMM_WORLD

    mesh, _, facet_tags = build_ball_mesh(
        R=R, lc=lc, comm=comm, r_inner=r_inner, lc_inner=lc_inner
    )
    bg = KerrSchildCoeffs(M=M, a=a)
    solver = FirstOrderKGSolver(
        mesh=mesh, domain_radius=R, degree=1, potential_type="zero",
        cfl_factor=cfl, sponge={"enabled": True, "width": 5.0, "strength": 1.0},
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
    from psyop.analysis.qnm import estimate_qnm_prony

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
