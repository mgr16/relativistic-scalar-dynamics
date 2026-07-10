#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimador del observable interior de H2: pendiente logarítmica a(t).

Fournodavlos–Sbierski (arXiv:1804.01941; docs/research/phase2/literature.md)
demuestran que toda solución suave de la ecuación de onda en el interior de
Schwarzschild con decaimiento en el horizonte admite la expansión hacia la
singularidad

    u(t, r, ω) = A(t, ω)·ln r + B(t, ω) + Σ_{n≥1} [ζ_n rⁿ ln r + η_n rⁿ]

con A = L²-lim_{r→0} u/ln r. El observable de H2 es a(t) ≡ A (por modo
angular). Este módulo lo estima por regresión lineal de u contra la base
truncada de esa jerarquía sobre una ventana radial [r_lo, r_hi]:

    order=N:  u ≈ Σ_{n=0..N} [ζ_n rⁿ·ln r + η_n rⁿ]   (2(N+1) parámetros)

con a ≡ ζ₀ y b ≡ η₀ (y c ≡ ζ₁, d ≡ η₁ cuando existen). El sesgo de
truncamiento del orden N está dominado por el primer término omitido,
O(r^{N+1}·|ln r|) evaluado en la ventana. La calibración cuantitativa de
ese sesgo por ventana (con el oráculo 1D) vive en
scripts/interior_window_calibration.py y decide la ventana de producción
3D. Cuidado: sobre ventanas angostas las columnas de órdenes altos son
casi colineales (ver `cond`) y el precio del de-sesgo es varianza.

Honestidad de las incertidumbres: los errores 1σ reportados son OLS con
σ² = SSR/(n−p), que asume residuo blanco. Sobre perfiles FD suaves el
residuo está CORRELACIONADO (es el resto de la jerarquía, no ruido), así
que estos errores son indicativos, no cobertura garantizada — la
incertidumbre de programa se obtiene del scan de ventanas y resoluciones,
igual que hace fit_tail_lines con el perfil de verosimilitud.

Solo NumPy (ruta Core CI, sin DOLFINx).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

__all__ = ["LogProfileFit", "fit_log_profile", "fit_log_profile_series",
           "fit_log_profile_multipole", "local_log_slope"]

_SERIES_KEYS = ("a", "b", "c", "d", "a_err", "b_err", "c_err", "d_err",
                "resid_rms", "cond", "n_points")


MAX_ORDER = 2


@dataclass
class LogProfileFit:
    """Resultado de un ajuste u ≈ Σ_{n≤order} (ζ_n rⁿ ln r + η_n rⁿ).

    a, b, c, d son alias de ζ₀, η₀, ζ₁, η₁ (NaN cuando el orden no los
    incluye); `zetas`/`etas` traen todos los coeficientes del orden pedido.
    `cond` es el número de condición de la matriz de diseño con columnas
    normalizadas: sobre ventanas angostas las columnas de órdenes altos
    son casi colineales y `cond` grande avisa que los coeficientes
    individuales altos no son interpretables aunque `a` siga bien
    determinada (su error honesto es a_err + el scan de ventanas).
    """

    a: float
    b: float
    c: float
    d: float
    a_err: float
    b_err: float
    c_err: float
    d_err: float
    zetas: Tuple[float, ...]
    etas: Tuple[float, ...]
    zeta_errs: Tuple[float, ...]
    eta_errs: Tuple[float, ...]
    resid_rms: float
    cond: float
    n_points: int
    r_lo: float
    r_hi: float
    order: int


def _design_matrix(r: np.ndarray, order: int) -> np.ndarray:
    ln_r = np.log(r)
    cols = []
    for n in range(order + 1):
        rn = r**n
        cols += [rn * ln_r, rn]
    return np.column_stack(cols)


def fit_log_profile(
    r: np.ndarray,
    u: np.ndarray,
    r_window: Tuple[float, float],
    order: int = 0,
) -> LogProfileFit:
    """Ajusta la jerarquía logarítmica truncada sobre una ventana radial.

    Args:
        r: radios (positivos, cualquier orden/malla).
        u: perfil del campo en esos radios (mismo tamaño que r).
        r_window: (r_lo, r_hi) de la ventana de ajuste, extremos incluidos.
        order: N ∈ [0, MAX_ORDER] → base {rⁿ·ln r, rⁿ : n ≤ N}.

    Returns:
        LogProfileFit con coeficientes, errores OLS 1σ (indicativos — ver
        docstring del módulo), rms del residuo y diagnóstico de condición.
    """
    if order not in range(MAX_ORDER + 1):
        raise ValueError(f"order must be in [0, {MAX_ORDER}], got {order}")
    r = np.asarray(r, dtype=float)
    u = np.asarray(u, dtype=float)
    if r.shape != u.shape or r.ndim != 1:
        raise ValueError(f"r and u must be equal-length 1D, got {r.shape} vs {u.shape}")
    r_lo, r_hi = float(r_window[0]), float(r_window[1])
    if not (0.0 < r_lo < r_hi):
        raise ValueError(f"need 0 < r_lo < r_hi, got ({r_lo}, {r_hi})")

    mask = (r >= r_lo) & (r <= r_hi)
    n_par = 2 * (order + 1)
    n_pts = int(np.count_nonzero(mask))
    if n_pts < n_par + 2:
        raise ValueError(
            f"window [{r_lo}, {r_hi}] contains {n_pts} points; "
            f"need at least {n_par + 2} for order={order}"
        )

    X = _design_matrix(r[mask], order)
    y = u[mask]

    # columnas a rms unidad: condición numérica sana y `cond` comparable
    scale = np.sqrt(np.mean(X**2, axis=0))
    Xs = X / scale
    coef_s, _, _, _ = np.linalg.lstsq(Xs, y, rcond=None)
    coef = coef_s / scale

    resid = y - X @ coef
    dof = max(n_pts - n_par, 1)
    sigma2 = float(resid @ resid) / dof
    cov_s = sigma2 * np.linalg.inv(Xs.T @ Xs)
    errs = np.sqrt(np.diag(cov_s)) / scale

    full = np.full(4, np.nan)
    full_err = np.full(4, np.nan)
    full[: min(n_par, 4)] = coef[: min(n_par, 4)]
    full_err[: min(n_par, 4)] = errs[: min(n_par, 4)]

    return LogProfileFit(
        a=float(full[0]), b=float(full[1]), c=float(full[2]), d=float(full[3]),
        a_err=float(full_err[0]), b_err=float(full_err[1]),
        c_err=float(full_err[2]), d_err=float(full_err[3]),
        zetas=tuple(float(v) for v in coef[0::2]),
        etas=tuple(float(v) for v in coef[1::2]),
        zeta_errs=tuple(float(v) for v in errs[0::2]),
        eta_errs=tuple(float(v) for v in errs[1::2]),
        resid_rms=float(np.sqrt(np.mean(resid**2))),
        cond=float(np.linalg.cond(Xs)),
        n_points=n_pts, r_lo=r_lo, r_hi=r_hi, order=order,
    )


def fit_log_profile_series(
    r: np.ndarray,
    snapshots_u: Sequence[np.ndarray],
    r_window: Tuple[float, float],
    order: int = 0,
) -> Dict[str, np.ndarray]:
    """Aplica fit_log_profile a una pila de snapshots u(r) (p.ej. del oráculo).

    Returns:
        dict de arrays de longitud n_snapshots con claves
        a, b, c, d, a_err, b_err, c_err, d_err, resid_rms, cond, n_points.
    """
    fits = [fit_log_profile(r, u_k, r_window, order=order) for u_k in snapshots_u]
    return {key: np.array([getattr(f, key) for f in fits]) for key in _SERIES_KEYS}


def fit_log_profile_multipole(
    radii: np.ndarray,
    coeffs: np.ndarray,
    modes: Sequence[Tuple[int, int]],
    r_window: Tuple[float, float],
    order: int = 0,
) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    """Ajusta a_lm(t) sobre las series del banco de extracción 3D.

    Args:
        radii: los K radios de extracción (forma (K,)).
        coeffs: series c_lm(t, r_k) con forma (n_times, K, n_modes) — lo que
            produce MultiRadiusExtractor.extract apilado en el tiempo.
        modes: lista de (l, m) en el orden del eje n_modes.
        r_window, order: como en fit_log_profile.

    Returns:
        {(l, m): dict de series} con las mismas claves que
        fit_log_profile_series; a_lm(t) es la clave "a" de cada modo.

    Convención de normalización: c_lm es el coeficiente de proyección
    ∮ φ Y_lm dΩ, así que para un campo esféricamente simétrico φ = f(r) el
    modo (0,0) lleva un factor √(4π) respecto de f (Y_00 = 1/√(4π)). Los
    cocientes entre corridas con el mismo dato (el discriminador A/B de H2)
    son independientes de esta convención.
    """
    radii = np.asarray(radii, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)
    if radii.ndim != 1:
        raise ValueError(f"radii must be 1D, got shape {radii.shape}")
    if coeffs.ndim != 3 or coeffs.shape[1] != radii.size or coeffs.shape[2] != len(modes):
        raise ValueError(
            f"coeffs must have shape (n_times, {radii.size}, {len(modes)}), "
            f"got {coeffs.shape}"
        )
    return {
        (int(l), int(m)): fit_log_profile_series(radii, coeffs[:, :, j], r_window, order=order)
        for j, (l, m) in enumerate(modes)
    }


def local_log_slope(r: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Pendiente logarítmica puntual s(r) = du/d(ln r) = r·∂_r u.

    Diagnóstico sin ajuste (el estimador del piloto de F0): si u sigue la
    jerarquía, s(r→0) → a. Útil para inspeccionar dónde empieza la zona
    logarítmica; para números con incertidumbre usar fit_log_profile.
    """
    r = np.asarray(r, dtype=float)
    u = np.asarray(u, dtype=float)
    if r.shape != u.shape or r.ndim != 1:
        raise ValueError(f"r and u must be equal-length 1D, got {r.shape} vs {u.shape}")
    if np.any(r <= 0):
        raise ValueError("all radii must be positive")
    return np.gradient(u, np.log(r))
