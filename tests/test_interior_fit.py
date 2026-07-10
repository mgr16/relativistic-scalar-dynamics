#!/usr/bin/env python3
"""
test_interior_fit.py

Guardián de rsd.analysis.interior (estimador del observable a(t) de H2).
La base del ajuste es la jerarquía de Fournodavlos–Sbierski truncada
(docs/research/phase2/literature.md): u = a·ln r + b + ζ₁·r·ln r + η₁·r + …
Verifica con sintéticos: recuperación exacta cuando el modelo es completo,
escala del sesgo de truncamiento del orden 0 (≈ primer término omitido,
O(r·|ln r|) en la ventana), de-sesgo por orden 1, y cobertura razonable de
los errores OLS bajo ruido blanco.
"""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

pytestmark = [pytest.mark.requires_numpy]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# malla log como la del oráculo (r ∈ [0.01, 60])
R_GRID = np.geomspace(0.01, 60.0, 2600)
RNG = np.random.default_rng(11)

A_TRUE, B_TRUE = -0.37, 0.021
Z1_TRUE, E1_TRUE = 0.85, -0.44


def _hierarchy(r, a=A_TRUE, b=B_TRUE, z1=0.0, e1=0.0):
    return a * np.log(r) + b + z1 * r * np.log(r) + e1 * r


def test_exact_recovery_order0():
    from rsd.analysis.interior import fit_log_profile

    u = _hierarchy(R_GRID)
    fit = fit_log_profile(R_GRID, u, (0.05, 0.5), order=0)
    assert abs(fit.a - A_TRUE) < 1e-12
    assert abs(fit.b - B_TRUE) < 1e-12
    assert fit.resid_rms < 1e-13
    assert fit.order == 0 and np.isnan(fit.c) and np.isnan(fit.d)


def test_exact_recovery_order1():
    from rsd.analysis.interior import fit_log_profile

    u = _hierarchy(R_GRID, z1=Z1_TRUE, e1=E1_TRUE)
    fit = fit_log_profile(R_GRID, u, (0.05, 0.5), order=1)
    for got, want in ((fit.a, A_TRUE), (fit.b, B_TRUE),
                      (fit.c, Z1_TRUE), (fit.d, E1_TRUE)):
        assert abs(got - want) < 1e-9
    assert fit.resid_rms < 1e-11


def test_truncation_bias_scaling_and_order1_rescue():
    """Ley de escala del sesgo de truncamiento del orden 0 sobre ventanas
    auto-similares [w, 2w]: para un término omitido puro η₁·r el sesgo de
    `a` es ∝ w (ratio 2 exacto por cada halving); para ζ₁·r·ln r escala
    como w·|ln w| (ratio entre 2 y 3 en esta zona). Con términos mezclados
    de signos opuestos las proyecciones pueden cancelarse de forma distinta
    por ventana (sin monotonía garantizada), pero el orden 1 de-sesga
    siempre. Esto fija la aritmética con que la calibración 1D interpreta
    las ventanas de producción."""
    from rsd.analysis.interior import fit_log_profile

    windows = [(0.25, 0.5), (0.125, 0.25), (0.0625, 0.125)]

    def biases(u):
        return [abs(fit_log_profile(R_GRID, u, w, order=0).a - A_TRUE)
                for w in windows]

    # término omitido puro η₁·r → sesgo ∝ posición de la ventana
    b_e1 = biases(_hierarchy(R_GRID, e1=E1_TRUE))
    for k in range(len(b_e1) - 1):
        assert b_e1[k] / b_e1[k + 1] == pytest.approx(2.0, rel=0.1), b_e1

    # linealidad exacta de OLS: el sesgo inducido por un término omitido
    # g(r) es EXACTAMENTE el coeficiente `a` de ajustar g a solas — esta
    # identidad es la aritmética con que se interpreta la calibración
    u_mix = _hierarchy(R_GRID, z1=Z1_TRUE, e1=E1_TRUE)
    g_only = _hierarchy(R_GRID, a=0.0, b=0.0, z1=Z1_TRUE, e1=E1_TRUE)
    for win in windows:
        bias_mix = fit_log_profile(R_GRID, u_mix, win, order=0).a - A_TRUE
        bias_g = fit_log_profile(R_GRID, g_only, win, order=0).a
        assert bias_mix == pytest.approx(bias_g, rel=1e-9, abs=1e-12)

    # ζ₁·r·ln r: su pendiente logarítmica r·(ln r + 1) se ANULA en
    # r = 1/e ≈ 0.37 (dentro de [0.25, 0.5]): el sesgo ahí está suprimido
    # por coincidencia de posición y NO es monótono ventana a ventana; lo
    # garantizado es el decaimiento hacia ventanas profundas
    b_z1 = biases(_hierarchy(R_GRID, z1=Z1_TRUE))
    deep = abs(fit_log_profile(
        R_GRID, _hierarchy(R_GRID, z1=Z1_TRUE), (0.01, 0.02), order=0
    ).a - A_TRUE)
    assert deep < 0.5 * max(b_z1)
    assert deep / abs(A_TRUE) < 0.15

    # mezcla realista: el orden 1 elimina el sesgo en cualquier ventana
    for win in windows:
        fit1 = fit_log_profile(R_GRID, u_mix, win, order=1)
        assert abs(fit1.a - A_TRUE) < 1e-8, f"order=1 debe de-sesgar en {win}"


def test_exact_recovery_order2():
    from rsd.analysis.interior import fit_log_profile

    z2, e2 = 0.31, -0.18
    u = (_hierarchy(R_GRID, z1=Z1_TRUE, e1=E1_TRUE)
         + z2 * R_GRID**2 * np.log(R_GRID) + e2 * R_GRID**2)
    fit = fit_log_profile(R_GRID, u, (0.05, 0.5), order=2)
    assert abs(fit.a - A_TRUE) < 1e-7
    assert fit.zetas[2] == pytest.approx(z2, rel=1e-4)
    assert fit.etas[2] == pytest.approx(e2, rel=1e-4)
    assert len(fit.zetas) == len(fit.etas) == 3
    # los alias c, d siguen siendo ζ₁, η₁
    assert fit.c == pytest.approx(Z1_TRUE, rel=1e-4)
    assert fit.d == pytest.approx(E1_TRUE, rel=1e-4)


def test_ols_errors_reasonable_under_white_noise():
    from rsd.analysis.interior import fit_log_profile

    sigma = 1e-4
    u = _hierarchy(R_GRID) + sigma * RNG.standard_normal(R_GRID.size)
    fit = fit_log_profile(R_GRID, u, (0.05, 0.5), order=0)
    assert fit.a_err > 0
    assert abs(fit.a - A_TRUE) < 5.0 * fit.a_err
    # el rms del residuo debe reflejar el ruido inyectado
    assert 0.5 * sigma < fit.resid_rms < 2.0 * sigma


def test_series_matches_single_fits():
    from rsd.analysis.interior import fit_log_profile, fit_log_profile_series

    snaps = [_hierarchy(R_GRID, a=A_TRUE * k, z1=0.1 * k) for k in (1.0, 2.0, 3.0)]
    series = fit_log_profile_series(R_GRID, snaps, (0.1, 0.5), order=1)
    assert series["a"].shape == (3,)
    for k, u in enumerate(snaps):
        single = fit_log_profile(R_GRID, u, (0.1, 0.5), order=1)
        assert series["a"][k] == pytest.approx(single.a, abs=0.0)
        assert series["cond"][k] == pytest.approx(single.cond, abs=0.0)


def test_local_log_slope_constant_for_pure_log():
    from rsd.analysis.interior import local_log_slope

    u = _hierarchy(R_GRID)
    s = local_log_slope(R_GRID, u)
    inner = R_GRID < 1.0
    assert np.allclose(s[inner], A_TRUE, atol=1e-8)


def test_validation_errors():
    from rsd.analysis.interior import fit_log_profile

    u = _hierarchy(R_GRID)
    with pytest.raises(ValueError):
        fit_log_profile(R_GRID, u, (0.5, 0.1))            # ventana invertida
    with pytest.raises(ValueError):
        fit_log_profile(R_GRID, u, (0.05, 0.5), order=3)  # orden no soportado
    with pytest.raises(ValueError):
        # ventana sin puntos suficientes
        fit_log_profile(R_GRID[:5], u[:5], (0.011, 0.0111), order=1)
