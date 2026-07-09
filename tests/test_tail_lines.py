#!/usr/bin/env python3
"""
test_tail_lines.py

Guardián de fit_tail_lines (rsd.analysis.ringdown): extracción de líneas
cuasi-estacionarias por ajuste continuo en frecuencia. Nació de un error
real (docs/research/phase1/cavity/note.md): los picos de la FFT de una
cola de ~30M caen en la grilla de bins (Δω = 2π/T ≈ 0.21) y produjeron
una falsa "escalera armónica 0.209/0.419"; el ajuste continuo no está
limitado por bins y debe recuperar frecuencias arbitrarias entre bins.
"""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

pytestmark = [pytest.mark.requires_numpy]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Malla temporal como la de las colas reales: t ∈ [40, 69], dt ≈ 0.214
TS = np.arange(40.0, 69.0, 0.214)
RNG = np.random.default_rng(7)


def test_estimator_unbiased_at_high_snr():
    """Con información de sobra (ruido ínfimo) el estimador conjunto debe
    clavar dos líneas sub-Rayleigh (Δω=0.14 < 2π/T≈0.22) FUERA de la
    grilla de bins — esto es lo que el greedy secuencial y la FFT no
    pueden hacer."""
    from rsd.analysis.ringdown import fit_tail_lines

    w1, w2 = 0.201, 0.343
    sig = (3e-4 * np.cos(w1 * TS + 0.7)
           + 1.5e-4 * np.cos(w2 * TS - 1.1)
           + 5e-8 * RNG.standard_normal(TS.size))
    out = fit_tail_lines(TS, sig, t_min=40.0)
    assert abs(out["w1"] - w1) < 2e-3, out
    assert abs(out["w2"] - w2) < 2e-3, out
    assert out["amp1"] == pytest.approx(3e-4, rel=0.02)
    assert out["amp2"] == pytest.approx(1.5e-4, rel=0.02)
    assert out["rms2"] < 1e-2 * out["rms0"]


def test_identifiability_limit_reported_at_real_snr():
    """Al SNR de las colas reales la verosimilitud es una trinchera
    degenerada: el mínimo puntual puede desviarse ~0.01 y la función debe
    REPORTARLO (dw1/dw2 del orden de la desviación, no cero)."""
    from rsd.analysis.ringdown import fit_tail_lines

    w1, w2 = 0.201, 0.343
    sig = (3e-4 * np.cos(w1 * TS + 0.7)
           + 1.5e-4 * np.cos(w2 * TS - 1.1)
           + 5e-6 * RNG.standard_normal(TS.size))
    out = fit_tail_lines(TS, sig, t_min=40.0)
    # dentro del límite de identificabilidad…
    assert abs(out["w1"] - w1) < 0.015, out
    assert abs(out["w2"] - w2) < 0.015, out
    # …y la incertidumbre reportada es consistente con la desviación real
    assert out["dw1"] > 0.5 * abs(out["w1"] - w1), out
    assert out["dw2"] > 0.5 * abs(out["w2"] - w2), out
    assert out["rms2"] < 0.1 * out["rms0"]


def test_single_line_reports_weak_second():
    from rsd.analysis.ringdown import fit_tail_lines

    sig = 2e-4 * np.cos(0.27 * TS + 0.3) + 4e-6 * RNG.standard_normal(TS.size)
    out = fit_tail_lines(TS, sig, t_min=40.0)
    assert abs(out["w1"] - 0.27) < 5e-3
    # la "segunda línea" sólo recoge ruido: amplitud muy inferior
    assert out["amp2"] < 0.15 * out["amp1"]


def test_short_tail_raises():
    from rsd.analysis.ringdown import fit_tail_lines

    with pytest.raises(ValueError, match="tail"):
        fit_tail_lines(TS[:8], np.ones(8), t_min=0.0)
