"""
Tests livianos del LiveViewer: construcción off_screen sobre una malla chica
y actualización de un frame sin error (pyvista soporta off_screen para CI),
más la degradación elegante de create_live_viewer en corridas paralelas.
"""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pyvista")

try:
    from mpi4py import MPI

    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False

pytestmark = [
    pytest.mark.requires_dolfinx,
    pytest.mark.skipif(not HAS_DOLFINX, reason="DOLFINx not available"),
]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_live_viewer_off_screen_updates_one_frame():
    import dolfinx.fem as fem

    from psyop.mesh.gmsh import build_ball_mesh
    from psyop.utils.live_view import LiveViewer

    mesh, _, _ = build_ball_mesh(R=4.0, lc=2.0, comm=MPI.COMM_WORLD)
    try:
        V = fem.functionspace(mesh, ("Lagrange", 1))
    except AttributeError:
        V = fem.FunctionSpace(mesh, ("Lagrange", 1))
    phi = fem.Function(V)
    phi.interpolate(lambda x: np.exp(-(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)))

    viewer = LiveViewer(V, off_screen=True)
    viewer.update(phi, t=0.0)

    # update() degrada a _failed en vez de propagar: el flag es el "sin error"
    assert not viewer.failed
    assert float(viewer.grid.point_data["phi"].max()) > 0.0
    viewer.close()


def test_create_live_viewer_declines_parallel_comm():
    from psyop.utils.live_view import create_live_viewer

    class FakeParallelComm:
        size = 2
        rank = 0

    assert create_live_viewer(None, comm=FakeParallelComm()) is None
