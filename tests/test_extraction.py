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
