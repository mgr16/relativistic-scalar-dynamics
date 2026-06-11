#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colas de Price: decaimiento tardío en ley de potencia tras el ringdown.

Para un escalar sin masa sobre Schwarzschild, una perturbación l-polar con
datos iniciales genéricos de soporte compacto decae a radio fijo como

    φ(t, r) ~ t^-(2l+3)        (Price 1972)

(l=1 → t^-5). La cola proviene del backscatter en la cola del potencial
curvo a r ~ t/2: para medirla limpia hasta t_end el borde exterior debe
estar causalmente desconectado de la esfera de extracción
(R ≳ (t_end + r_ext + r0)/2) — un borde absorbente cercano la suprime.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def fit_power_law_tail(
    ts: np.ndarray,
    signal: np.ndarray,
    t_min: float,
    t_max: float,
) -> Tuple[float, float]:
    """Ajusta |signal| ~ C·t^p en [t_min, t_max] por regresión log-log.

    Returns:
        (p, r2): exponente de la ley de potencia (negativo para decaimiento)
        y coeficiente de determinación R² del ajuste lineal en log-log
        (R² ≈ 1 ⇔ ley de potencia limpia; R² bajo ⇒ la ventana todavía
        contiene ringdown exponencial o ruido de piso).
    """
    ts = np.asarray(ts, dtype=float)
    signal = np.asarray(signal, dtype=float)
    mask = (ts >= t_min) & (ts <= t_max) & (np.abs(signal) > 0)
    if mask.sum() < 8:
        raise ValueError(
            f"tail window [{t_min}, {t_max}] has only {int(mask.sum())} usable samples"
        )
    logt = np.log(ts[mask])
    logphi = np.log(np.abs(signal[mask]))

    A = np.vstack([logt, np.ones_like(logt)]).T
    coeffs, residuals, _, _ = np.linalg.lstsq(A, logphi, rcond=None)
    p = float(coeffs[0])

    pred = A @ coeffs
    ss_res = float(np.sum((logphi - pred) ** 2))
    ss_tot = float(np.sum((logphi - np.mean(logphi)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return p, float(r2)


def price_exponent(l: int, static_moment: bool = False) -> int:
    """Exponente teórico de Price para el multipolo l.

    Args:
        l: número multipolar
        static_moment: True si los datos iniciales tienen momento estático
            (φ≠0, Π=0 en t=0), que decae un orden más lento
    """
    return -(2 * l + 2) if static_moment else -(2 * l + 3)
