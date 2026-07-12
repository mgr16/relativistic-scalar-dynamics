#!/usr/bin/env python3
"""
test_ringdown_windows.py

Guardián de fit_anchored_windows (rsd.analysis.ringdown): el protocolo de
espectroscopía por abanico de ventanas ancladas al pico del ring, usado por
la escalera de convergencia (F1), el capítulo de cavidad y la espectroscopía
exterior de producción (F2). Reglas que fija:

  * el pico se busca a partir de t_search (el tránsito directo del pulso
    por la esfera de extracción NO debe anclar las ventanas);
  * cada ventana ajusta Prony y se citan media/scatter del abanico;
  * una señal demasiado corta no revienta: n_windows == 0 y sin claves
    de frecuencia (el llamador decide).
"""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

pytestmark = [pytest.mark.requires_numpy]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Ringdown sintético tipo l=2 de Schwarzschild (Leaver: 0.48364 - 0.09676i)
W_REF, G_REF = 0.48364, 0.09676
REF = complex(W_REF, -G_REF)
DT = 0.09  # muestreo típico de las corridas (sample_every=4)


def synth(t_end=70.0, t_ring=15.0, a_transit=2.0):
    """Tránsito espurio temprano (t≈6) + ring anclable en t_ring."""
    ts = np.arange(0.0, t_end, DT)
    sig = np.zeros_like(ts)
    m = ts >= t_ring
    sig[m] = np.exp(-G_REF * (ts[m] - t_ring)) * np.cos(W_REF * (ts[m] - t_ring))
    # tránsito directo ANTES de t_search=12: más alto que el ring para
    # verificar que el anclaje lo ignora
    sig += a_transit * np.exp(-((ts - 6.0) ** 2) / 1.5)
    return ts, sig


def test_recovers_reference_mode_with_fan_scatter():
    from rsd.analysis.ringdown import fit_anchored_windows

    ts, sig = synth()
    out = fit_anchored_windows(ts, sig, REF)
    assert out["n_windows"] == 5, out
    # el anclaje debe caer en el ring (t=15), no en el tránsito (t=6)
    assert abs(out["t_peak"] - 15.0) < 0.5, out
    assert abs(out["omega_re"] - W_REF) < 2e-3, out
    assert abs(-out["omega_im"] - G_REF) < 2e-3, out
    assert out["err_re"] < 5e-3 and out["err_im"] < 2e-2, out
    # señal limpia: el abanico debe ser coherente
    assert out["omega_re_std"] < 1e-3, out
    assert len(out["omega_re_windows"]) == 5


def test_short_signal_degrades_to_zero_windows():
    from rsd.analysis.ringdown import fit_anchored_windows

    # la señal termina justo tras el pico: ninguna ventana tiene los
    # 4*modes muestras mínimas del Prony
    ts, sig = synth(t_end=16.5)
    out = fit_anchored_windows(ts, sig, REF)
    assert out["n_windows"] == 0
    assert "omega_re" not in out
