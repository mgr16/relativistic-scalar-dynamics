from typing import Tuple
import numpy as np
import ufl

HAS_DOLFINX = False
_DOLFINX_IMPORT_ERROR = None
try:
    import dolfinx
    import dolfinx.mesh as dmesh
    import dolfinx.fem as femx
    from mpi4py import MPI
    from petsc4py import PETSc
    HAS_DOLFINX = True
except Exception as e:
    _DOLFINX_IMPORT_ERROR = e

if not HAS_DOLFINX:
    raise ImportError("DOLFINx is required by psyop.backends.fem") from _DOLFINX_IMPORT_ERROR


def is_dolfinx() -> bool:
    return HAS_DOLFINX


def Constant(mesh, value):
    from petsc4py import PETSc
    return femx.Constant(mesh, PETSc.ScalarType(value))


def assemble_scalar(form):
    return float(femx.assemble_scalar(femx.form(form)))


def create_ds_with_outer_tag(mesh, R: float | None = None, atol: float = 0.1):
    """
    Crea meshtags para facetas de borde y devuelve (ds, tag_id, facet_tags).
    - Si R se provee: etiqueta facetas con radio ~ R (esfera).
    - Si R es None: etiqueta TODAS las facetas exteriores (caja u otras).
    Tag usado: 2
    """
    facet_dim = mesh.topology.dim - 1
    if R is not None:
        def boundary_marker(x):
            r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
            return np.isclose(r, R, atol=atol)
        facets = femx.locate_entities_boundary(mesh, facet_dim, boundary_marker)
    else:
        mesh.topology.create_connectivity(facet_dim, mesh.topology.dim)
        from dolfinx.mesh import exterior_facet_indices
        facets = exterior_facet_indices(mesh.topology)

    tags = np.full(facets.shape, 2, dtype=np.int32)
    facet_tags = femx.meshtags(mesh, facet_dim, facets, tags)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    return ds, 2, facet_tags

