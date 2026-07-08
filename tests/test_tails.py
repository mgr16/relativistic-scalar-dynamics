"""
Tests del ajuste de colas de Price con señales sintéticas: el estimador
debe recuperar el exponente exacto de una ley de potencia pura y detectar
(vía R² bajo) cuando la ventana todavía está dominada por el ringdown.
"""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rsd.analysis.tails import fit_power_law_tail, price_exponent  # noqa: E402


def test_recovers_pure_power_law():
    ts = np.linspace(50.0, 200.0, 300)
    signal = 3.7e-6 * ts**-5
    p, r2 = fit_power_law_tail(ts, signal, 60.0, 190.0)
    assert p == pytest.approx(-5.0, abs=1e-10)
    assert r2 == pytest.approx(1.0, abs=1e-12)


def test_recovers_with_alternating_sign_noise():
    rng = np.random.default_rng(7)
    ts = np.linspace(50.0, 200.0, 400)
    signal = 1e-5 * ts**-3 * (1.0 + 0.05 * rng.standard_normal(ts.size))
    p, r2 = fit_power_law_tail(ts, signal, 55.0, 195.0)
    assert p == pytest.approx(-3.0, abs=0.05)
    assert r2 > 0.99


def test_exponential_window_has_low_r2_vs_power_law():
    # Un ringdown exponencial en ventana log-log curva hacia abajo: el
    # exponente local crece sin cota y R² del ajuste global cae
    ts = np.linspace(10.0, 100.0, 300)
    ringdown = np.exp(-0.1 * ts)
    _, r2_exp = fit_power_law_tail(ts, ringdown, 10.0, 100.0)
    _, r2_tail = fit_power_law_tail(ts, ts**-5, 10.0, 100.0)
    # El exponencial es convexo en log-log: R² cae bien por debajo de la
    # ley de potencia pura (medido: 0.927 vs 1.0)
    assert r2_tail > 0.999
    assert r2_exp < 0.95


def test_price_exponents():
    assert price_exponent(0) == -3
    assert price_exponent(1) == -5
    assert price_exponent(2) == -7
    assert price_exponent(1, static_moment=True) == -4


def test_window_validation():
    ts = np.linspace(0, 10, 20)
    with pytest.raises(ValueError, match="usable samples"):
        fit_power_law_tail(ts, np.ones_like(ts), 9.0, 9.5)
