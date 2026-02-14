#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gmsh_helpers.py

Funciones para generar mallas con Gmsh y etiquetar fronteras.
Crea geometrías esféricas con etiqueta "outer_boundary" para BC absorbentes.
"""

from typing import Optional, Tuple
import numpy as np
import dolfinx
import dolfinx.mesh as dmesh
from dolfinx.io import gmshio
from mpi4py import MPI

from psyop.backends.fem import create_ds_with_outer_tag
from psyop.utils.logger import get_logger

logger = get_logger(__name__)

# Importar Gmsh si está disponible
try:
    import gmsh
    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False
    logger.warning("Gmsh no disponible. Usando mallas de FEniCS básicas.")

def build_ball_mesh(
    R: float, 
    lc: float, 
    comm: Optional[MPI.Comm] = None
) -> Tuple[dmesh.Mesh, Optional[dmesh.MeshTags], Optional[dmesh.MeshTags]]:
    """
    Crea una malla esférica con radio R y tamaño característico lc.
    
    Parámetros:
        R: Radio de la esfera
        lc: Tamaño característico de elemento
        comm: Comunicador MPI (para DOLFINx)
    
    Retorna:
        mesh: Malla del dominio
        cell_tags: Etiquetas de celdas (opcional)
        facet_tags: Etiquetas de facetas con outer_boundary=2 (opcional)
    """
    if not HAS_GMSH:
        logger.warning("Gmsh no disponible. Creando malla esférica simple...")
        return _create_simple_ball_mesh(R, comm)
    
    if comm is None:
        comm = MPI.COMM_WORLD
    
    # Inicializar Gmsh
    gmsh.initialize()
    gmsh.clear()
    
    try:
        # Crear modelo
        gmsh.model.add("ball")
        logger.debug(f"Creando malla esférica: R={R}, lc={lc}")
        
        # Crear geometría esférica
        sphere = gmsh.model.occ.addSphere(0, 0, 0, R)
        gmsh.model.occ.synchronize()
        
        # Etiquetar superficies
        # outer_boundary = 2 (superficie exterior)
        surfaces = gmsh.model.getEntities(2)
        for surface in surfaces:
            gmsh.model.addPhysicalGroup(2, [surface[1]], 2)  # tag=2 para outer_boundary
        
        # Etiquetar volumen  
        volumes = gmsh.model.getEntities(3)
        for volume in volumes:
            gmsh.model.addPhysicalGroup(3, [volume[1]], 1)  # tag=1 para volumen
        
        # Configurar malla
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc/3)
        
        # Generar malla
        gmsh.model.mesh.generate(3)
        logger.info("Malla generada con Gmsh")
        
        # Importar a DOLFINx
        mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
            gmsh.model, comm, rank=0, gdim=3
        )
    
    finally:
        gmsh.finalize()
    
    return mesh, cell_tags, facet_tags

def _create_simple_ball_mesh(
    R: float, 
    comm: Optional[MPI.Comm] = None
) -> Tuple[dmesh.Mesh, Optional[dmesh.MeshTags], Optional[dmesh.MeshTags]]:
    """
    Fallback: crear malla esférica simple sin Gmsh.
    """
    from dolfinx.mesh import create_box
    
    logger.info(f"Creando malla cúbica como aproximación de esfera con R={R}")
    mesh = create_box(
        comm or MPI.COMM_WORLD,
        [[-R, -R, -R], [R, R, R]], 
        [16, 16, 16]
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
    facet_tags = None
    
    return mesh, cell_tags, facet_tags

def get_outer_tag(
    facet_tags: dmesh.MeshTags, 
    name: str = "outer_boundary", 
    default: int = 2
) -> int:
    """
    Devuelve el id de la frontera externa. 
    Si facet_tags trae nombres en metadata, busca 'outer_boundary'.
    """
    try:
        md = getattr(facet_tags, "metadata", None)
        if md and "names" in md:
            for k, v in md["names"].items():
                if v == name:
                    return int(k)
    except (AttributeError, KeyError, TypeError) as e:
        logger.debug(f"No se pudo obtener tag de metadata: {e}")
    return int(default)

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
