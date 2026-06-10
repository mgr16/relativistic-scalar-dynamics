#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo utils.py

Utilidades numéricas para la simulación:
  - Estimación del tamaño mínimo de celda (h_min).
  - Cálculo del paso de tiempo CFL (consciente del grado FEM).
"""

import numpy as np

from psyop.backends.fem import is_dolfinx


def _approx_hmin(mesh, sample: int = 4000) -> float:
    """
    Estima h_min muestreando longitudes de aristas locales. Suficiente para CFL.
    Estimación robusta para mallas DOLFINx.
    """
    try:
        if is_dolfinx() and hasattr(mesh, 'topology'):
            topo = mesh.topology
            topo.create_connectivity(1, 0)  # edges->vertices
            e2v = topo.connectivity(1, 0)
            X = mesh.geometry.x
            h_min = np.inf

            # Muestra al azar algunas aristas (o todas si son pocas)
            edges = np.arange(topo.index_map(1).size_local, dtype=np.int32)
            if edges.size > sample:
                rng = np.random.default_rng(12345)
                edges = rng.choice(edges, size=sample, replace=False)

            for e in edges:
                vs = e2v.links(e)
                if len(vs) >= 2:
                    h = np.linalg.norm(X[vs[1]] - X[vs[0]])
                    if h < h_min and h > 0:
                        h_min = h
            return float(h_min) if h_min != np.inf else 1.0

        else:
            return 1.0

    except Exception:
        # Fallback ultra-seguro
        return 1.0


def compute_dt_cfl(mesh, cfl: float = 0.3, c_max: float = 1.0, degree: int = 1) -> float:
    """
    dt = cfl * h_min / (c_max * degree²)

    - DOLFINx: h mínimo por celdas locales + reducción MPI.
    - El factor 1/degree² es el escalado estándar para elementos de orden
      alto con integración explícita (para degree=1 no cambia nada).
    """
    if is_dolfinx():
        try:
            from mpi4py import MPI
            from dolfinx.cpp.mesh import h as cpp_h
            tdim = mesh.topology.dim
            num_cells = mesh.topology.index_map(tdim).size_local
            h_vals = cpp_h(mesh, tdim, np.arange(num_cells, dtype=np.int32))
            hmin_local = float(np.min(h_vals)) if h_vals.size > 0 else 1.0
            try:
                hmin = float(mesh.comm.allreduce(hmin_local, op=MPI.MIN))
            except Exception:
                hmin = hmin_local
        except Exception:
            hmin = _approx_hmin(mesh)
    else:
        hmin = _approx_hmin(mesh)
    degree_factor = max(1, int(degree)) ** 2
    return float(cfl) * float(hmin) / (max(float(c_max), 1e-12) * degree_factor)
