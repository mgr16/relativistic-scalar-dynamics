"""
Tests del solver de Leaver: contra los valores publicados de Schwarzschild
(Berti, Cardoso & Starinets 2009) y contra valores de Kerr generados con el
paquete `qnm` (Stein 2019), verificados a precisión de máquina al momento
de escribir estos tests (err ~1e-15).
"""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rsd.analysis.leaver import (  # noqa: E402
    angular_eigenvalue,
    kerr_qnm,
    schwarzschild_qnm,
)

# Berti, Cardoso & Starinets (2009): escalar s=0, n=0
SCHWARZSCHILD_REFS = {
    0: 0.110455 - 0.104896j,
    1: 0.292936 - 0.097660j,
    2: 0.483644 - 0.096759j,
}

# Generados con qnm.modes_cache(s=0, ...) (paquete `qnm`, Stein 2019)
KERR_REFS = {
    (1, 0, 0.6): 0.30028277 - 0.09413041j,
    (2, 2, 0.6): 0.61736371 - 0.09124540j,
    (2, -2, 0.6): 0.41314648 - 0.09520441j,
    (1, 1, 0.9): 0.43723381 - 0.07184813j,
}


def test_schwarzschild_fundamental_modes():
    for l, ref in SCHWARZSCHILD_REFS.items():
        omega = schwarzschild_qnm(l)
        assert omega == pytest.approx(ref, abs=2e-6), f"l={l}"


def test_schwarzschild_first_overtone():
    omega = schwarzschild_qnm(2, n=1)
    assert omega == pytest.approx(0.463851 - 0.295604j, abs=2e-6)


def test_kerr_spin_modes():
    for (l, m, a), ref in KERR_REFS.items():
        omega = kerr_qnm(l=l, m=m, a=a)
        assert omega == pytest.approx(ref, abs=1e-7), f"l={l} m={m} a={a}"


def test_prograde_retrograde_splitting():
    # La rotación rompe la degeneración en m: el modo co-rotante sube en
    # frecuencia y decae más lento; el contra-rotante baja
    w0 = kerr_qnm(l=2, m=2, a=0.0)
    wp = kerr_qnm(l=2, m=2, a=0.7)
    wm = kerr_qnm(l=2, m=-2, a=0.7)
    assert wp.real > w0.real > wm.real
    assert abs(wp.imag) < abs(w0.imag)  # vive más


def test_angular_eigenvalue_limits():
    # c -> 0: armónico esférico exacto A = l(l+1)
    assert angular_eigenvalue(0.0, 2, 1) == pytest.approx(6.0, abs=1e-12)
    # c real chico: perturbación de segundo orden, A < l(l+1) para m=0
    A = angular_eigenvalue(0.3, 1, 0)
    assert A.real < 2.0 and abs(A.imag) < 1e-12


def test_input_validation():
    with pytest.raises(ValueError, match="spin"):
        kerr_qnm(l=2, m=0, a=1.5)
    with pytest.raises(ValueError, match="l="):
        kerr_qnm(l=1, m=2, a=0.3)
    with pytest.raises(ValueError, match="overtone"):
        kerr_qnm(l=1, m=0, n=-1)


@pytest.mark.slow
def test_cross_validation_against_qnm_package():
    qnm_pkg = pytest.importorskip("qnm")
    for (l, m, a) in [(2, 0, 0.95), (2, 1, 0.8)]:
        mode = qnm_pkg.modes_cache(s=0, l=l, m=m, n=0)
        w_ref, _, _ = mode(a=a)
        assert kerr_qnm(l=l, m=m, a=a) == pytest.approx(complex(w_ref), abs=1e-10)
