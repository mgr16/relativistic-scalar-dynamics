#!/usr/bin/env python3
"""
test_cavity_mode_slow.py

Canario de los modos de cavidad del dominio R=20 (docs/research/phase1/
cavity/note.md): la cola tardía del ringdown está dominada por modos
atrapados entre la barrera de potencial y la esponja (r>15) — un artefacto
FÍSICO del dominio, no de la malla. Para l=2 la cola es un doblete
coherente limpio: w1 ≈ 0.351 ± 0.003 y w2 ≈ 0.560 ± 0.007 (ajuste continuo
fit_tail_lines; independiente de la excitación w=2→1 y estable en
lc=1.4→1.0). Si este test falla tras cambiar R, la esponja o las BC, el
espectro de cavidad se movió y toda calibración de espectroscopía
(ventanas, suelos, ratios pico/suelo) debe revisarse.

Notas de honestidad: (a) la cifra histórica "0.209 + armónico 0.419" del
capítulo de convergencia era un artefacto de bins de FFT (Δω = 2π/30M ≈
0.21); (b) la cola l=1 NO es un doblete limpio (mezcla una componente
secular lenta con líneas que vagan entre resoluciones) — por eso el
canario se define sobre l=2.
"""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

try:
    from mpi4py import MPI

    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False

pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_numpy,
    pytest.mark.requires_dolfinx,
    pytest.mark.requires_mesh,
    pytest.mark.skipif(not HAS_DOLFINX, reason="DOLFINx not available"),
]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Banda esperada del doblete l=2 en R=20 (medido: w1=0.3485–0.3545,
# w2=0.5555–0.5655 en lc∈{1.4, 1.0} y w∈{2, 1}; margen para remallados)
W1_LO, W1_HI = 0.32, 0.38
W2_LO, W2_HI = 0.52, 0.60


def test_cavity_doublet_l2_r20():
    from rsd.analysis.ringdown import evolve_kerr_ringdown, fit_tail_lines

    # lc=1.4: la corrida l=2 más barata donde el doblete ya es estable
    ts, sig = evolve_kerr_ringdown(
        a=0.0, l=2, m_abs=0, lc=1.4, lc_inner=1.4 / 3.75, t_end=70.0,
    )
    out = fit_tail_lines(ts, sig, t_min=40.0)

    assert W1_LO < out["w1"] < W1_HI, (
        f"línea fundamental de cavidad l=2 fuera de banda: "
        f"w1={out['w1']:.4f} (¿cambió R, la esponja o las BC?) {out}"
    )
    assert W2_LO < out["w2"] < W2_HI, (
        f"segunda línea de cavidad l=2 fuera de banda: "
        f"w2={out['w2']:.4f} {out}"
    )
    # Cola cuasi-estacionaria coherente: el doblete explica ≳85% del rms
    assert out["rms2"] < 0.15 * out["rms0"], out
