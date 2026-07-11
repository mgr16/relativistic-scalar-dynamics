#!/usr/bin/env python3
"""
Dato inicial gaussiana × Y_lm (l > 0) en 3D: el prerrequisito de las
corridas de producción de F2 (modos l = 1, 2 con dato idéntico al modo
u_l del oráculo 1D).

Convención bajo test: la perturbación (y el momento — el ansatz radial
factoriza) se multiplica por la MISMA real_ylm ortonormal del extractor
multipolar, de modo que c_lm(r, t=0) = A·g(r) en el canal (l, m). Las
tolerancias FEM vienen medidas en la malla 16³ de este módulo (fuga de
canal ≤ 2.5 % en φ y ≤ 13 % en Π, convergente con h: 0.128 → 0.032 al
pasar de 16³ a 32³); los cocientes main/ref cancelan el error radial
común (~8.5 % en esta malla).
"""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

try:
    from mpi4py import MPI

    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False

pytestmark = [
    pytest.mark.requires_numpy,
    pytest.mark.requires_dolfinx,
    pytest.mark.skipif(not HAS_DOLFINX, reason="DOLFINx not available"),
]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SQ4PI = np.sqrt(4.0 * np.pi)
A, R0, W = 1e-3, 8.0, 2.0


@pytest.fixture(scope="module")
def box_mesh():
    from dolfinx.mesh import create_box

    return create_box(MPI.COMM_WORLD, [[-12, -12, -12], [12, 12, 12]], [16, 16, 16])


@pytest.fixture(scope="module")
def extractor(box_mesh):
    from rsd.analysis.extraction import MultipoleExtractor

    return MultipoleExtractor(box_mesh, radius=R0, lmax=3)


def test_default_is_spherical_l0(box_mesh):
    """El default (l, m) = (0, 0) reproduce exactamente el bump histórico."""
    from rsd.physics.initial_conditions import GaussianBump

    legacy = GaussianBump(box_mesh, A=A, r0=R0, w=W, v0=0.3, direction="ingoing")
    explicit = GaussianBump(box_mesh, A=A, r0=R0, w=W, v0=0.3, direction="ingoing",
                            l=0, m=0)
    assert np.array_equal(legacy.get_function().x.array,
                          explicit.get_function().x.array)
    assert np.array_equal(legacy.get_momentum().x.array,
                          explicit.get_momentum().x.array)


def test_angular_convention_matches_extraction(box_mesh):
    """Cartesiano→(θ, ϕ)→Y_lm del dato == real_ylm del extractor, exacto.

    El extractor construye sus puntos como R·(sinθcosϕ, sinθsinϕ, cosθ);
    si la inversión del dato divergiera (rango de atan2, definición de θ,
    signo de m), el canal (l, m) se rompería en silencio — este test lo
    pina sin error FEM de por medio.
    """
    from rsd.analysis.extraction import real_ylm
    from rsd.physics.initial_conditions import GaussianBump

    theta = np.linspace(0.1, np.pi - 0.1, 7)
    phi = np.linspace(0.0, 2 * np.pi, 9, endpoint=False)
    tg, pg = np.meshgrid(theta, phi, indexing="ij")
    tg, pg = tg.ravel(), pg.ravel()
    r = 5.0
    x = np.vstack([r * np.sin(tg) * np.cos(pg),
                   r * np.sin(tg) * np.sin(pg),
                   r * np.cos(tg)])
    for (l, m) in [(1, 0), (1, 1), (2, -1), (2, 2), (3, -3)]:
        bump = GaussianBump(box_mesh, A=A, r0=R0, w=W, v0=0.0, l=l, m=m)
        got = bump._angular_factor(x)
        want = real_ylm(l, m, tg, pg)
        assert np.max(np.abs(got - want)) < 1e-12, f"(l, m)=({l}, {m})"


def test_phi_channel_isolation(box_mesh, extractor):
    """c_lm(r0) cae en el canal pedido con la amplitud radial del l=0.

    main/ref usa como referencia c_00/√4π del bump esférico con el mismo
    perfil radial: cancela el error de interpolación radial común y aísla
    el angular (medido en 16³: |main/ref − 1| ≤ 1.4 %, fuga ≤ 2.5 %).
    El vacío v0 es esférico: debe quedar íntegro en (0, 0).
    """
    from rsd.physics.initial_conditions import GaussianBump

    ref_bump = GaussianBump(box_mesh, A=A, r0=R0, w=W, v0=0.0)
    ref = extractor.extract(ref_bump.get_function())[(0, 0)] / SQ4PI
    assert abs(ref / (A * 1.0) - 1.0) < 0.15, "referencia radial fuera de rango"

    for (l, m) in [(1, 0), (1, -1), (2, 0), (2, -1), (2, 2)]:
        bump = GaussianBump(box_mesh, A=A, r0=R0, w=W, v0=0.5, l=l, m=m)
        c = extractor.extract(bump.get_function())
        main = c[(l, m)]
        leak = max(abs(v) for k, v in c.items() if k not in ((l, m), (0, 0)))
        assert abs(main / ref - 1.0) < 0.05, f"(l, m)=({l}, {m}): main/ref={main/ref}"
        assert leak < 0.06 * abs(ref), f"(l, m)=({l}, {m}): fuga {leak/abs(ref):.3f}"
        assert abs(c[(0, 0)] / SQ4PI - 0.5) < 1e-4, "el vacío debe ser puro (0, 0)"


def test_momentum_channel_consistency(box_mesh, extractor):
    """Π lleva el mismo factor angular que φ (ingoing_curved sobre KS).

    Tolerancias medidas en 16³: |main/ref − 1| ≤ 6.7 %, fuga ≤ 12.8 %
    ((1, −1), la peor); la fuga converge con h (0.032 en 32³) — es
    interpolación P1 del perfil con estructura 1/r, no convención.
    """
    from rsd.physics.initial_conditions import GaussianBump
    from rsd.physics.metrics import KerrSchildCoeffs

    bg = KerrSchildCoeffs(M=1.0, a=0.0)
    ref_bump = GaussianBump(box_mesh, A=A, r0=R0, w=W, v0=0.0,
                            direction="ingoing_curved", background=bg)
    ref = extractor.extract(ref_bump.get_momentum())[(0, 0)] / SQ4PI

    for (l, m), leak_tol in [((2, 0), 0.12), ((1, -1), 0.25)]:
        bump = GaussianBump(box_mesh, A=A, r0=R0, w=W, v0=0.0,
                            direction="ingoing_curved", background=bg, l=l, m=m)
        c = extractor.extract(bump.get_momentum())
        main = c[(l, m)]
        leak = max(abs(v) for k, v in c.items() if k != (l, m))
        assert abs(main / ref - 1.0) < 0.15, f"(l, m)=({l}, {m}): main/ref={main/ref}"
        assert leak < leak_tol * abs(ref), f"(l, m)=({l}, {m}): fuga {leak/abs(ref):.3f}"


def test_invalid_l_m_raise(box_mesh):
    from rsd.physics.initial_conditions import GaussianBump

    for bad in ({"l": -1}, {"l": 1, "m": 2}, {"l": 1.5}, {"l": 0, "m": 1},
                {"l": 2, "m": -3}):
        with pytest.raises(ValueError):
            GaussianBump(box_mesh, A=A, r0=R0, w=W, v0=0.0, **bad)


def test_config_validates_l_m():
    from rsd.config import DEFAULT_CONFIG, validate_config

    def cfg_with(**ic):
        cfg = {**DEFAULT_CONFIG}
        cfg["initial_conditions"] = {**DEFAULT_CONFIG["initial_conditions"], **ic}
        return cfg

    validate_config(cfg_with(l=2, m=-2))
    validate_config(cfg_with())  # defaults (l, m) = (0, 0)
    for bad in ({"l": -1}, {"l": 2, "m": 3}, {"l": 1.5}, {"l": "dos"}):
        with pytest.raises(ValueError):
            validate_config(cfg_with(**bad))
