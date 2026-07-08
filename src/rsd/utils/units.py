#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversión de unidades geométricas (G = c = 1) a unidades astrofísicas.

El código trabaja en unidades geométricas donde la masa del agujero negro
fija las escalas: con `metric.M = M_code`, una unidad de tiempo/longitud del
código equivale a (M_solar / M_code) veces la escala de una masa solar:

    T_sun = G·M_sun/c³ = 4.925490947641267e-6 s
    L_sun = G·M_sun/c² = 1.476625038050125 km

Así, el QNM fundamental l=2 de un agujero negro de 30 M_sun (Mω ≈ 0.4837)
cae en ~520 Hz: la banda de LIGO. Esta capa convierte los outputs del
pipeline QNM (frecuencias ordinarias f = ω/2π y tasas de decaimiento 1/τ,
ambas por unidad de tiempo del código) a Hz y ms.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

# Escalas de una masa solar en unidades geométricas (CODATA/IAU nominal):
# G·M_sun = 1.32712440018e20 m³/s², c = 299792458 m/s
GM_SUN_M3_S2 = 1.32712440018e20
C_M_S = 299792458.0
T_SUN_S = GM_SUN_M3_S2 / C_M_S**3  # 4.92549e-6 s
L_SUN_KM = GM_SUN_M3_S2 / C_M_S**2 / 1000.0  # 1.476625 km


class GeometricUnits:
    """Conversor entre unidades del código y unidades físicas.

    Args:
        M_solar: Masa del agujero negro en masas solares (> 0)
        M_code: Valor de `metric.M` en unidades del código (default 1.0);
            una unidad de tiempo del código = (M_solar/M_code)·T_sun
    """

    def __init__(self, M_solar: float, M_code: float = 1.0):
        if M_solar <= 0:
            raise ValueError(f"M_solar must be > 0, got {M_solar}")
        if M_code <= 0:
            raise ValueError(f"M_code must be > 0, got {M_code}")
        self.M_solar = float(M_solar)
        self.M_code = float(M_code)
        scale = self.M_solar / self.M_code
        self.T_unit_s = scale * T_SUN_S  # segundos por unidad de tiempo código
        self.L_unit_km = scale * L_SUN_KM  # km por unidad de longitud código

    # --- tiempo / longitud ---
    def time_s(self, t_code: float) -> float:
        return t_code * self.T_unit_s

    def time_ms(self, t_code: float) -> float:
        return self.time_s(t_code) * 1e3

    def length_km(self, x_code: float) -> float:
        return x_code * self.L_unit_km

    # --- pipeline QNM (f en ciclos por unidad de tiempo código) ---
    def frequency_Hz(self, f_code: float) -> float:
        return f_code / self.T_unit_s

    def angular_frequency_Hz(self, omega_code: float) -> float:
        """ω angular del código (p.ej. Mω de Leaver con M_code=1) → f en Hz."""
        return omega_code / (2.0 * np.pi) / self.T_unit_s

    def damping_time_ms(self, decay_code: float) -> float:
        """Tasa de decaimiento 1/τ del código → τ en milisegundos."""
        if decay_code <= 0:
            return float("inf")
        return (1.0 / decay_code) * self.T_unit_s * 1e3

    # --- reporte ---
    def describe(self) -> Dict[str, float]:
        return {
            "M_solar": self.M_solar,
            "M_code": self.M_code,
            "T_unit_s": self.T_unit_s,
            "L_unit_km": self.L_unit_km,
        }

    def qnm_to_physical(self, modes: list) -> list:
        """Convierte modos Prony [{frequency, decay, ...}] a unidades físicas."""
        out = []
        for m in modes:
            out.append(
                {
                    "frequency_Hz": self.frequency_Hz(float(m["frequency"])),
                    "damping_time_ms": self.damping_time_ms(float(m["decay"])),
                    **{
                        k: m[k]
                        for k in ("amplitude", "phase", "score")
                        if k in m
                    },
                }
            )
        return out


def units_from_config(cfg: Dict[str, Any]):
    """Crea GeometricUnits desde la config si `output.physical_units` está.

    Devuelve None si no está configurado (las corridas siguen siendo
    adimensionales por default). Espera:
        "output": {"physical_units": {"M_solar": 30.0}}
    y usa `metric.M` como masa del código.
    """
    pu = (cfg.get("output", {}) or {}).get("physical_units") or {}
    m_solar = pu.get("M_solar")
    if not m_solar:
        return None
    m_code = float((cfg.get("metric", {}) or {}).get("M", 1.0))
    return GeometricUnits(M_solar=float(m_solar), M_code=m_code)
