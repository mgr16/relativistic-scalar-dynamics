"""
Validación de colas de Price: tras el ringdown, un escalar SIN MASA sobre
Schwarzschild decae a radio fijo como t^-(2l+3) (Price 1972) — para l=1,
t^-5. La cola nace del backscatter en el potencial curvo a r ~ t/2, así
que el dominio debe ser causalmente limpio: con R=60, r0=8 y r_ext=6 el
eco del borde llega a t ≈ (60-8)+(60-6) = 106, después de la ventana de
ajuste.

Es el segundo benchmark clásico de campos sobre agujeros negros (el
primero es el ringdown QNM): valida la propagación de larga distancia,
el potencial efectivo curvo y la ausencia de reflexiones espurias.
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

PRICE_EXPONENT = -5  # l=1, datos genéricos: t^-(2l+3)


def test_massless_l1_price_tail():
    from rsd.analysis.ringdown import evolve_kerr_ringdown
    from rsd.analysis.tails import fit_power_law_tail, price_exponent

    assert price_exponent(1) == PRICE_EXPONENT

    ts, c10 = evolve_kerr_ringdown(
        a=0.0, l=1, m_abs=0, R=60.0, lc=3.0, lc_inner=0.5,
        r0=8.0, w=2.0, r_ext=6.0, t_end=110.0,
    )
    assert np.all(np.isfinite(c10)), "extracted signal contains NaN/inf"

    # Ventana calibrada: post-ringdown (e^-0.098t ya cayó >4 órdenes a t=55)
    # y pre-eco del borde (t=106)
    p, r2 = fit_power_law_tail(ts, c10, t_min=55.0, t_max=100.0)

    # Tolerancias calibradas en malla de CI (ver docs/validation/summary.md)
    assert p == pytest.approx(PRICE_EXPONENT, abs=1.0), (
        f"tail exponent {p:.2f}, Price = {PRICE_EXPONENT}"
    )
    assert r2 > 0.97, f"tail is not a clean power law (R² = {r2:.3f})"
