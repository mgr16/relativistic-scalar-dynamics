"""
Tests de la extracción multipolar: contra un campo analítico puro Y_10,
el coeficiente c_10 debe coincidir con el valor exacto y el resto de los
modos debe ser despreciable.
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
    pytest.mark.requires_numpy,
    pytest.mark.requires_dolfinx,
    pytest.mark.skipif(not HAS_DOLFINX, reason="DOLFINx not available"),
]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_extractor_recovers_pure_y10_field():
    import dolfinx.fem as fem

    from rsd.analysis.extraction import MultipoleExtractor
    from rsd.mesh.gmsh import build_ball_mesh

    mesh, _, _ = build_ball_mesh(R=8.0, lc=0.8, comm=MPI.COMM_WORLD)
    try:
        V = fem.functionspace(mesh, ("Lagrange", 1))
    except AttributeError:
        V = fem.FunctionSpace(mesh, ("Lagrange", 1))
    f = fem.Function(V)
    # φ = z·exp(-r/4) = g(r)·(z/r) con g = r·exp(-r/4); z/r = √(4π/3)·Y_10
    f.interpolate(
        lambda x: x[2] * np.exp(-np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) / 4.0)
    )

    R_ext = 5.0
    extractor = MultipoleExtractor(mesh, radius=R_ext, lmax=2)
    coeffs = extractor.extract(f)

    expected_c10 = R_ext * np.exp(-R_ext / 4.0) * np.sqrt(4.0 * np.pi / 3.0)
    # Calibrado: err ~0.7% (interpolación P1 en lc=0.8)
    assert coeffs[(1, 0)] == pytest.approx(expected_c10, rel=0.03)

    leakage = max(abs(v) for k, v in coeffs.items() if k != (1, 0))
    assert leakage < 0.02 * abs(coeffs[(1, 0)]), (
        f"mode leakage too large: {leakage:.2e} vs c_10={coeffs[(1, 0)]:.2e}"
    )


def test_extractor_validates_inputs():
    from rsd.analysis.extraction import MultipoleExtractor
    from rsd.mesh.gmsh import build_ball_mesh

    mesh, _, _ = build_ball_mesh(R=4.0, lc=2.0, comm=MPI.COMM_WORLD)
    with pytest.raises(ValueError, match="radius"):
        MultipoleExtractor(mesh, radius=-1.0)
    with pytest.raises(ValueError, match="lmax"):
        MultipoleExtractor(mesh, radius=2.0, lmax=-1)


def test_multi_radius_bank_matches_single_extractor():
    """El banco en K radios debe reproducir exactamente al extractor de un
    radio (misma cuadratura, mismo eval) — es la garantía de que a_lm(t) 3D
    hereda la maquinaria ya validada."""
    import dolfinx.fem as fem

    from rsd.analysis.extraction import MultipoleExtractor, MultiRadiusExtractor
    from rsd.mesh.gmsh import build_ball_mesh

    mesh, _, _ = build_ball_mesh(R=8.0, lc=0.8, comm=MPI.COMM_WORLD)
    try:
        V = fem.functionspace(mesh, ("Lagrange", 1))
    except AttributeError:
        V = fem.FunctionSpace(mesh, ("Lagrange", 1))
    f = fem.Function(V)
    f.interpolate(
        lambda x: x[2] * np.exp(-np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) / 4.0)
    )

    radii = [3.0, 4.0, 5.0]
    bank = MultiRadiusExtractor(mesh, radii, lmax=2)
    coeffs = bank.extract(f)
    assert coeffs.shape == (3, len(bank.modes))

    for k, r_k in enumerate(radii):
        single = MultipoleExtractor(mesh, radius=r_k, lmax=2)
        expected = single.extract(f)
        for j, mode in enumerate(bank.modes):
            assert coeffs[k, j] == pytest.approx(expected[mode], abs=1e-13)


def test_multi_radius_bank_recovers_log_slope():
    """Pipeline completo del diagnóstico interior sobre un campo sintético
    φ = a·ln r + b (puro l=0): banco de K=16 radios log + fit orden 0 →
    a_00 = √(4π)·a con error de interpolación P1, y fuga l>0 despreciable."""
    import dolfinx.fem as fem

    from rsd.analysis.extraction import MultiRadiusExtractor
    from rsd.analysis.interior import fit_log_profile_multipole
    from rsd.mesh.gmsh import build_ball_mesh

    a_true, b_true = 1.3, -0.7
    mesh, _, _ = build_ball_mesh(R=8.0, lc=0.8, comm=MPI.COMM_WORLD)
    try:
        V = fem.functionspace(mesh, ("Lagrange", 1))
    except AttributeError:
        V = fem.FunctionSpace(mesh, ("Lagrange", 1))
    f = fem.Function(V)
    # el piso en r evita el −∞ en el nodo del origen; no toca la ventana [2, 5]
    f.interpolate(
        lambda x: a_true * np.log(np.maximum(
            np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2), 1e-3)) + b_true
    )

    radii = np.geomspace(2.0, 5.0, 16)
    bank = MultiRadiusExtractor(mesh, radii, lmax=2)
    coeffs = bank.extract(f)[None, :, :]  # una "época"

    fits = fit_log_profile_multipole(radii, coeffs, bank.modes, (2.0, 5.0), order=0)
    a00 = fits[(0, 0)]["a"][0]
    expected = np.sqrt(4.0 * np.pi) * a_true
    assert a00 == pytest.approx(expected, rel=0.03)

    leakage = max(abs(fits[mode]["a"][0]) for mode in bank.modes if mode != (0, 0))
    assert leakage < 0.02 * abs(a00)


def test_multi_radius_bank_validates_coverage_and_inputs():
    from rsd.analysis.extraction import MultiRadiusExtractor
    from rsd.mesh.gmsh import build_ball_mesh

    mesh, _, _ = build_ball_mesh(R=4.0, lc=2.0, comm=MPI.COMM_WORLD)
    with pytest.raises(ValueError, match="radii"):
        MultiRadiusExtractor(mesh, [])
    with pytest.raises(ValueError, match="> 0"):
        MultiRadiusExtractor(mesh, [1.0, -2.0])
    with pytest.raises(ValueError, match="distinct"):
        MultiRadiusExtractor(mesh, [1.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="lmax"):
        MultiRadiusExtractor(mesh, [1.0], lmax=-1)
    # radio fuera del dominio: la cobertura debe fallar en la construcción,
    # no sesgar la cuadratura en silencio
    with pytest.raises(ValueError, match="fuera de la malla"):
        MultiRadiusExtractor(mesh, [1.0, 6.0])


def test_real_ylm_orthonormality():
    """Las Y_lm reales deben ser ortonormales bajo la cuadratura usada."""
    from rsd.analysis.extraction import real_ylm

    lmax = 3
    nodes, w = np.polynomial.legendre.leggauss(lmax + 2)
    theta = np.arccos(nodes)
    n_phi = 2 * lmax + 4
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    tg, pg = np.meshgrid(theta, phi, indexing="ij")
    wg = (np.repeat(w[:, None], n_phi, axis=1) * (2 * np.pi / n_phi)).ravel()
    tg, pg = tg.ravel(), pg.ravel()

    modes = [(l, m) for l in range(lmax + 1) for m in range(-l, l + 1)]
    for i, (l1, m1) in enumerate(modes):
        y1 = real_ylm(l1, m1, tg, pg)
        for (l2, m2) in modes[i:]:
            y2 = real_ylm(l2, m2, tg, pg)
            inner = float(np.dot(wg * y1, y2))
            expected = 1.0 if (l1, m1) == (l2, m2) else 0.0
            assert inner == pytest.approx(expected, abs=1e-10), (
                f"<Y_{l1}{m1}, Y_{l2}{m2}> = {inner}"
            )
