"""
Tests de la capa de unidades astrofísicas: las escalas geométricas de una
masa solar son constantes conocidas, y el QNM fundamental de un agujero
negro de 30 M_sun debe caer en la banda de LIGO (~100-1000 Hz).
"""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from psyop.utils.units import GeometricUnits, units_from_config  # noqa: E402


def test_solar_scales():
    u = GeometricUnits(M_solar=1.0)
    assert u.T_unit_s == pytest.approx(4.92549e-6, rel=1e-4)
    assert u.L_unit_km == pytest.approx(1.476625, rel=1e-4)


def test_ligo_band_for_30_msun():
    # Mω = 0.483644 (escalar l=2, n=0 Schwarzschild) para M = 30 M_sun
    u = GeometricUnits(M_solar=30.0)
    f_hz = u.angular_frequency_Hz(0.483644)
    assert 100.0 < f_hz < 1000.0  # banda LIGO
    assert f_hz == pytest.approx(521.0, rel=0.01)


def test_m_code_rescaling():
    # Con metric.M = 2, una unidad de código vale la mitad de masas solares
    u1 = GeometricUnits(M_solar=10.0, M_code=1.0)
    u2 = GeometricUnits(M_solar=10.0, M_code=2.0)
    assert u2.T_unit_s == pytest.approx(u1.T_unit_s / 2.0)


def test_damping_time():
    # Im(Mω) = 0.096759 (l=2, n=0): tau = M/0.096759 en unidades geométricas
    u = GeometricUnits(M_solar=30.0)
    tau_ms = u.damping_time_ms(0.096759 / (2.0 * np.pi) * (2.0 * np.pi))
    expected_ms = (1.0 / 0.096759) * 30.0 * 4.92549e-6 * 1e3
    assert tau_ms == pytest.approx(expected_ms, rel=1e-6)
    assert u.damping_time_ms(0.0) == float("inf")


def test_qnm_to_physical_roundtrip():
    u = GeometricUnits(M_solar=30.0)
    modes = [{"frequency": 0.0770, "decay": 0.0967, "amplitude": 1.0, "score": 0.9}]
    phys = u.qnm_to_physical(modes)
    assert phys[0]["frequency_Hz"] == pytest.approx(0.0770 / u.T_unit_s)
    assert phys[0]["amplitude"] == 1.0
    assert "phase" not in phys[0]


def test_units_from_config():
    cfg = {
        "metric": {"M": 1.0},
        "output": {"physical_units": {"M_solar": 62.0}},
    }
    u = units_from_config(cfg)
    assert u is not None and u.M_solar == 62.0
    assert units_from_config({"output": {}}) is None
    with pytest.raises(ValueError, match="M_solar"):
        GeometricUnits(M_solar=-1.0)
