"""
Tests de generación de mallas: excisión (cáscara con borde interior
etiquetado) y graduación radial del tamaño de elemento.
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
    pytest.mark.requires_mesh,
    pytest.mark.skipif(not HAS_DOLFINX, reason="DOLFINx not available"),
]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _cell_sizes_and_radii(mesh):
    import dolfinx.mesh
    from dolfinx.cpp.mesh import h as cpp_h

    tdim = mesh.topology.dim
    n = mesh.topology.index_map(tdim).size_local
    cells = np.arange(n, dtype=np.int32)
    try:
        h = cpp_h(mesh, tdim, cells)
    except TypeError:
        h = cpp_h(mesh._cpp_object, tdim, cells)
    mid = dolfinx.mesh.compute_midpoints(mesh, tdim, cells)
    return h, np.linalg.norm(mid, axis=1)


def test_excised_mesh_has_inner_and_outer_tags():
    from psyop.mesh.gmsh import INNER_BOUNDARY_TAG, OUTER_BOUNDARY_TAG, build_ball_mesh

    mesh, _, facet_tags = build_ball_mesh(R=6.0, lc=1.5, comm=MPI.COMM_WORLD, r_inner=1.5)
    assert facet_tags is not None
    tags = set(np.unique(facet_tags.values))
    assert OUTER_BOUNDARY_TAG in tags, "outer boundary tag missing"
    assert INNER_BOUNDARY_TAG in tags, "inner (excision) boundary tag missing"

    # No debe haber celdas dentro del radio excisado
    _, r = _cell_sizes_and_radii(mesh)
    assert r.min() > 1.0, "cells found inside the excised region"


def test_graded_mesh_refines_center():
    from psyop.mesh.gmsh import build_ball_mesh

    mesh, _, facet_tags = build_ball_mesh(
        R=10.0, lc=1.5, comm=MPI.COMM_WORLD, lc_inner=0.6
    )
    assert facet_tags is not None
    h, r = _cell_sizes_and_radii(mesh)
    h_center = float(np.median(h[r < 3.0]))
    h_border = float(np.median(h[r > 7.0]))
    # Calibrado: ratio ~0.59 con lc_inner/lc = 0.4
    assert h_center < 0.75 * h_border, (
        f"graded mesh should refine the center: h_center={h_center:.3f} "
        f"h_border={h_border:.3f}"
    )


def test_lc_inner_is_validated():
    from psyop.mesh.gmsh import build_ball_mesh

    with pytest.raises(ValueError, match="lc_inner"):
        build_ball_mesh(R=6.0, lc=1.0, comm=MPI.COMM_WORLD, lc_inner=2.0)
