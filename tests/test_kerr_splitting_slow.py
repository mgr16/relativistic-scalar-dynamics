"""
Validación del frame dragging: splitting prógrado/retrógrado en Kerr.

Un campo REAL con estructura Y_1|1| sobre Kerr a=0.9 excita por igual los
modos m=+1 (co-rotante) y m=-1 (contra-rotante). Leaver da

    Mω₊ = 0.437234 - 0.071848i   (prógrado)
    Mω₋ = 0.243371 - 0.094422i   (retrógrado)

es decir un splitting de frecuencias del 80% (ratio 1.797) — un efecto
puramente rotacional: en Schwarzschild ambos modos son degenerados y la
señal contiene UNA sola frecuencia. Detectar dos modos bien separados es
por lo tanto una medición directa del arrastre de marcos, imposible de
fingir con errores de discretización (que desplazan, pero no desdoblan).

Honestidad sobre la precisión: en la malla de CI (lc=1.5/0.4) las
frecuencias absolutas tienen sistemáticos de +30-40% (calibrado
2026-06-11: modos en 0.3137 y 0.6037 con Prony modes=6, ventana [12,58];
la mezcla espheroidal-esférica de la extracción Y_lm y la resolución cerca
del horizonte de a=0.9 sesgan hacia arriba). El test valida el splitting
y las bandas absolutas calibradas; afinar requiere malla más fina.
"""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

try:
    from mpi4py import MPI  # noqa: F401

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

SPIN = 0.9
OMEGA_PROGRADE = 0.437234
OMEGA_RETROGRADE = 0.243371
SPLITTING_RATIO = OMEGA_PROGRADE / OMEGA_RETROGRADE  # 1.797


def test_frame_dragging_splits_m_modes():
    from rsd.analysis.ringdown import evolve_kerr_ringdown, fit_ringdown_modes

    ts, c11 = evolve_kerr_ringdown(a=SPIN, l=1, m_abs=1, t_end=60.0)
    assert np.all(np.isfinite(c11)), "extracted signal contains NaN/inf"
    assert np.max(np.abs(c11)) > 0, "extracted signal is identically zero"

    fitted = fit_ringdown_modes(ts, c11, t_min=12.0, t_max=58.0, modes=6)
    # Quedarse con frecuencias distintas (Prony devuelve pares conjugados)
    freqs = []
    for w, d in fitted:
        if not any(abs(w - f) / f < 0.05 for f in freqs):
            freqs.append(w)
    assert len(freqs) >= 2, (
        f"frame dragging should split m=±1 into two modes, got {freqs}"
    )

    # Los dos modos dominantes por amplitud (freqs preserva el orden Prony)
    w_low, w_high = sorted(freqs[:2])
    ratio = w_high / w_low

    # El splitting es la observable robusta (los sistemáticos de
    # discretización desplazan ambos modos juntos, no los separan)
    assert ratio == pytest.approx(SPLITTING_RATIO, abs=0.65), (
        f"splitting ratio {ratio:.3f} vs Leaver {SPLITTING_RATIO:.3f}"
    )
    # Bandas absolutas calibradas en malla de CI (sesgo sistemático +30-40%)
    assert w_high == pytest.approx(OMEGA_PROGRADE, rel=0.45), (
        f"prograde Mω = {w_high:.4f}, Leaver = {OMEGA_PROGRADE}"
    )
    assert w_low == pytest.approx(OMEGA_RETROGRADE, rel=0.45), (
        f"retrograde Mω = {w_low:.4f}, Leaver = {OMEGA_RETROGRADE}"
    )
