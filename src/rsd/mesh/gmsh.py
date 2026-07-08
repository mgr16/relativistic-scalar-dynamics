#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gmsh.py

Funciones para generar mallas con Gmsh y etiquetar fronteras.
Crea geometrías esféricas con etiqueta "outer_boundary" para BC absorbentes.
"""

from typing import Optional, Tuple
import numpy as np
import dolfinx
import dolfinx.mesh as dmesh
from dolfinx.io import gmsh as dolfinx_gmsh
from mpi4py import MPI

from rsd.backends.fem import create_ds_with_outer_tag
from rsd.utils.logger import get_logger

logger = get_logger(__name__)
OUTER_BOUNDARY_TAG = 2
INNER_BOUNDARY_TAG = 3

# Importar Gmsh si está disponible
try:
    import gmsh
    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False
    logger.warning("Gmsh no disponible. Usando mallas cúbicas básicas.")

def build_ball_mesh(
    R: float,
    lc: float,
    comm: Optional[MPI.Comm] = None,
    r_inner: float = 0.0,
    lc_inner: Optional[float] = None,
    geom_order: int = 1,
) -> Tuple[dmesh.Mesh, Optional[dmesh.MeshTags], Optional[dmesh.MeshTags]]:
    """
    Crea una malla esférica con radio R y tamaño característico lc.

    Parámetros:
        R: Radio de la esfera
        lc: Tamaño característico de elemento (en el borde exterior)
        comm: Comunicador MPI (para DOLFINx)
        r_inner: Si > 0, excisa una esfera interior de ese radio (cáscara).
            El borde interior recibe la etiqueta INNER_BOUNDARY_TAG=3
            (requerido para fondos de agujero negro).
        lc_inner: Si se da (0 < lc_inner < lc), gradúa el tamaño de elemento
            radialmente: lc_inner en el centro/horizonte → lc en el borde
            (refinamiento donde la métrica varía más).
        geom_order: orden geométrico de las celdas (1 = facetas planas,
            2 = celdas curvas). Con elementos P2+ las facetas planas
            introducen un error geométrico O(h²) en las esferas de borde
            que domina sobre el error de aproximación; usar geom_order=2.

    Retorna:
        mesh: Malla del dominio
        cell_tags: Etiquetas de celdas (opcional)
        facet_tags: Etiquetas de facetas (outer=2, inner=3 si hay excisión)
    """
    r_inner = float(r_inner or 0.0)
    if r_inner < 0:
        raise ValueError(f"r_inner must be >= 0, got {r_inner}")
    if r_inner >= R:
        raise ValueError(f"r_inner ({r_inner}) must be smaller than R ({R})")
    if lc_inner is not None:
        lc_inner = float(lc_inner)
        if not (0 < lc_inner <= lc):
            raise ValueError(f"lc_inner must satisfy 0 < lc_inner <= lc, got {lc_inner}")
    geom_order = int(geom_order)
    if geom_order not in (1, 2):
        raise ValueError(f"geom_order must be 1 or 2, got {geom_order}")

    if not HAS_GMSH:
        if r_inner > 0 or lc_inner is not None or geom_order > 1:
            raise RuntimeError(
                "Excision/graded/curved meshes (r_inner/lc_inner/geom_order) "
                "require Gmsh. Install it with: conda install -c conda-forge gmsh"
            )
        logger.warning("Gmsh no disponible. Creando malla esférica simple...")
        return _create_simple_ball_mesh(R, comm, lc=lc)

    if comm is None:
        comm = MPI.COMM_WORLD

    # Inicializar Gmsh
    gmsh.initialize()
    gmsh.clear()

    try:
        # Crear modelo
        gmsh.model.add("ball")
        logger.debug(f"Creando malla esférica: R={R}, lc={lc}, r_inner={r_inner}")

        # Crear geometría: bola sólida o cáscara excisada
        outer_sphere = gmsh.model.occ.addSphere(0, 0, 0, R)
        if r_inner > 0:
            inner_sphere = gmsh.model.occ.addSphere(0, 0, 0, r_inner)
            gmsh.model.occ.cut([(3, outer_sphere)], [(3, inner_sphere)])
        gmsh.model.occ.synchronize()

        # Etiquetar superficies: la exterior por extensión de bounding box
        outer_surfaces = []
        inner_surfaces = []
        for dim, tag in gmsh.model.getEntities(2):
            bbox = gmsh.model.getBoundingBox(dim, tag)
            half_extent = max(
                bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]
            ) / 2.0
            if r_inner > 0 and abs(half_extent - r_inner) < abs(half_extent - R):
                inner_surfaces.append(tag)
            else:
                outer_surfaces.append(tag)
        if outer_surfaces:
            gmsh.model.addPhysicalGroup(2, outer_surfaces, OUTER_BOUNDARY_TAG)
        if inner_surfaces:
            gmsh.model.addPhysicalGroup(2, inner_surfaces, INNER_BOUNDARY_TAG)

        # Etiquetar volumen
        volumes = gmsh.model.getEntities(3)
        gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], 1)  # tag=1 para volumen

        # Configurar malla
        if lc_inner is not None and lc_inner < lc:
            # Graduación radial: lc_inner en r=0 (o el horizonte) → lc en r=R
            field = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(
                field,
                "F",
                f"{lc_inner} + ({lc} - {lc_inner}) * sqrt(x*x + y*y + z*z) / {R}",
            )
            gmsh.model.mesh.field.setAsBackgroundMesh(field)
            # El campo de fondo manda: desactivar otras fuentes de tamaño
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        else:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc/3)

        # Generar malla
        gmsh.model.mesh.generate(3)
        if geom_order > 1:
            # celdas curvas: los nodos de alto orden se proyectan a la
            # geometría CAD (esferas exactas en los bordes)
            gmsh.model.mesh.setOrder(geom_order)
        logger.info(f"Malla generada con Gmsh (orden geométrico {geom_order})")

        # Importar a DOLFINx
        mesh_data = dolfinx_gmsh.model_to_mesh(
            gmsh.model, comm, rank=0, gdim=3
        )
        if all(hasattr(mesh_data, attr) for attr in ("mesh", "cell_tags", "facet_tags")):
            mesh = mesh_data.mesh
            cell_tags = mesh_data.cell_tags
            facet_tags = mesh_data.facet_tags
        else:
            mesh, cell_tags, facet_tags = mesh_data

    finally:
        gmsh.finalize()

    return mesh, cell_tags, facet_tags

def _create_simple_ball_mesh(
    R: float,
    comm: Optional[MPI.Comm] = None,
    lc: Optional[float] = None,
) -> Tuple[dmesh.Mesh, Optional[dmesh.MeshTags], Optional[dmesh.MeshTags]]:
    """
    Fallback: crear malla cúbica simple sin Gmsh (resolución derivada de lc).
    """
    from dolfinx.mesh import create_box

    if lc and lc > 0:
        n = int(np.clip(round(2.0 * R / lc), 4, 48))
    else:
        n = 16
    logger.info(f"Creando malla cúbica como aproximación de esfera con R={R} ({n}^3 divisiones)")
    mesh = create_box(
        comm or MPI.COMM_WORLD,
        [[-R, -R, -R], [R, R, R]],
        [n, n, n]
    )
    cell_tags = None
    _, _, facet_tags = create_ds_with_outer_tag(mesh, R=None)

    return mesh, cell_tags, facet_tags

def build_box_mesh(
    x_min: np.ndarray, 
    x_max: np.ndarray, 
    nx: int, 
    ny: int, 
    nz: int, 
    comm: Optional[MPI.Comm] = None
) -> Tuple[dmesh.Mesh, Optional[dmesh.MeshTags], Optional[dmesh.MeshTags]]:
    """
    Crea una malla de caja rectangular con etiquetas de frontera.
    
    Parámetros:
        x_min: arrays con coordenadas mínimas [x, y, z]
        x_max: arrays con coordenadas máximas [x, y, z]
        nx, ny, nz: número de divisiones
        comm: Comunicador MPI
    
    Retorna:
        mesh, cell_tags, facet_tags
    """
    from dolfinx.mesh import create_box
    
    logger.info(f"Creando malla de caja: {x_min} to {x_max}, divisiones: [{nx}, {ny}, {nz}]")
    mesh = create_box(
        comm or MPI.COMM_WORLD,
        [x_min, x_max],
        [nx, ny, nz]
    )
    # Para caja, todas las fronteras son outer_boundary
    cell_tags = None
    _, _, facet_tags = create_ds_with_outer_tag(mesh, R=None)
    
    return mesh, cell_tags, facet_tags

def get_outer_tag(
    facet_tags: dmesh.MeshTags,
    name: str = "outer_boundary",
    default: int = 2,
) -> int:
    """Devuelve el id de la frontera externa (convención del proyecto: 2)."""
    return int(default if default is not None else OUTER_BOUNDARY_TAG)

if __name__ == "__main__":
    # Prueba básica
    logger.info("=== Prueba de gmsh_helpers.py ===")
    
    try:
        mesh, cell_tags, facet_tags = build_ball_mesh(R=5.0, lc=1.0)
        logger.info("✓ Malla esférica creada exitosamente")
        
        if hasattr(mesh, 'topology'):
            num_cells = mesh.topology.index_map(3).size_local
            logger.info(f"  Número de celdas: {num_cells}")
        
        if facet_tags is not None:
            logger.info("  ✓ Facet tags creados para BC absorbentes")
        
    except Exception as e:
        logger.error(f"✗ Error en prueba: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("Módulo gmsh_helpers.py completado.")
