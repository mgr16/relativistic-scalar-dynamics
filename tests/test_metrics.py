"""
Tests de los coeficientes 3+1 de fondo (métricas).

Verifica en particular que la curvatura extrínseca de Kerr-Schild ya no es
cero y reproduce el valor analítico de Schwarzschild-KS:
    K = 2M (r + 3M) / (r (r + 2M))^{3/2}   (Baumgarte & Shapiro)
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


def _interpolate_expression(expr, V):
    """Interpola una expresión UFL en V (compatible entre versiones DOLFINx)."""
    import dolfinx.fem as fem

    points = V.element.interpolation_points
    if callable(points):
        points = points()
    compiled = fem.Expression(expr, points)
    f = fem.Function(V)
    f.interpolate(compiled)
    return f


def _analytic_K_schwarzschild_ks(r, M):
    return 2.0 * M * (r + 3.0 * M) / (r * (r + 2.0 * M)) ** 1.5


def test_kerr_schild_trace_K_matches_schwarzschild_analytic():
    import dolfinx.fem as fem

    from rsd.mesh.gmsh import build_ball_mesh
    from rsd.physics.metrics import KerrSchildCoeffs

    M = 1.0
    mesh, _, _ = build_ball_mesh(R=8.0, lc=1.0, comm=MPI.COMM_WORLD, r_inner=1.5)

    coeffs = KerrSchildCoeffs(M=M, a=0.0)
    _, _, _, _, K_f = coeffs.build(mesh)

    try:
        V = fem.functionspace(mesh, ("Lagrange", 1))
    except AttributeError:
        V = fem.FunctionSpace(mesh, ("Lagrange", 1))
    K_h = _interpolate_expression(K_f, V)

    coords = V.tabulate_dof_coordinates()
    r = np.linalg.norm(coords[:, :3], axis=1)
    # Comparar en una banda intermedia, lejos de ambos bordes de la malla
    band = (r > 3.0) & (r < 6.0)
    assert band.sum() > 10, "test mesh should contain dofs in the comparison band"

    K_num = K_h.x.array[: len(r)][band]
    K_ref = _analytic_K_schwarzschild_ks(r[band], M)

    # K no debe ser idénticamente cero (bug previo) y debe seguir el perfil
    # analítico dentro de la tolerancia de interpolación P1 en malla gruesa
    assert np.max(np.abs(K_ref)) > 0
    rel_err = np.abs(K_num - K_ref) / np.abs(K_ref)
    assert np.median(rel_err) < 0.05, f"median rel err {np.median(rel_err):.3%}"


def test_flat_background_has_zero_K():
    from rsd.mesh.gmsh import build_ball_mesh
    from rsd.physics.metrics import FlatBackgroundCoeffs

    mesh, _, _ = build_ball_mesh(R=4.0, lc=2.0, comm=MPI.COMM_WORLD)
    _, _, _, _, K_f = FlatBackgroundCoeffs().build(mesh)
    assert float(K_f.value) == 0.0
