#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo utils.py

Este módulo proporciona funciones utilitarias para la simulación, tales como:
  - Creación de mallas personalizadas.
  - Guardado de soluciones en formato PVD.
  - Cálculo de normas y energías.
  - Funciones de logging para registrar mensajes y diagnósticos.
  - Lectura de archivos de configuración (por ejemplo, en formato JSON).

Estas funciones ayudan a mantener el código principal limpio y a reutilizar código en otros proyectos.
"""

import numpy as np
import json
import datetime

from psyop.backends.fem import is_dolfinx

def create_mesh(domain_min=(-20, -20, -20), domain_max=(20, 20, 20), nx=15, ny=15, nz=15):
    """
    Crea una malla BoxMesh para un dominio rectangular definido por los puntos
    mínimo y máximo.

    Parámetros:
        domain_min (tuple): Coordenadas mínimas del dominio, e.g., (-20, -20, -20).
        domain_max (tuple): Coordenadas máximas del dominio, e.g., (20, 20, 20).
        nx, ny, nz (int): Número de divisiones en cada dirección.

    Retorna:
        mesh (Mesh): Malla del solver (helper legacy deshabilitado).
    """
    raise RuntimeError("create_mesh() legacy helper removed. Use psyop.mesh.gmsh/build_ball_mesh for DOLFINx meshes.")

def save_solution(solution, filename, time=None):
    """
    Guarda una solución del solver en disco (helper legacy deshabilitado).

    Parámetros:
        solution (Function): Solución a guardar.
        filename (str): Ruta base del archivo (sin extensión).
        time (float, opcional): Tiempo actual de la solución para etiquetar la salida.
    """
    raise RuntimeError("save_solution() legacy helper removed. Use DOLFINx XDMF output APIs.")

def compute_norm(solution, norm_type='L2'):
    """
    Calcula la norma de una solución (helper legacy deshabilitado).

    Parámetros:
        solution (Function): Función a evaluar.
        norm_type (str): Tipo de norma a calcular ('L2' o 'H1').

    Retorna:
        norm_value (float): Valor numérico de la norma.
    """
    raise RuntimeError("compute_norm() legacy helper removed. Use DOLFINx/PETSc vector norms.")

def compute_energy(phi, Vh, potential_params):
    """
    Calcula una estimación de la energía del campo escalar, considerando un
    término cinético y un potencial dado.

    Se utiliza una forma de energía del tipo:
        E = ∫ [ 0.5 |∇φ|² + V(φ) ] dx
    donde el potencial se puede definir, por ejemplo, como:
        V(φ) = lam*(φ**4/4 - v0²*φ**2/2)
    
    Parámetros:
        phi (Function): Campo escalar.
        Vh (FunctionSpace): Espacio de funciones asociado a phi.
        potential_params (dict): Diccionario que debe contener:
            - 'lam': coeficiente del potencial.
            - 'v0' : valor base del campo.
    
    Retorna:
        energy (float): Energía total calculada en el dominio.
    """
    raise RuntimeError("compute_energy() legacy helper removed. Use solver.energy() with DOLFINx forms.")

def log_message(message, logfile="simulation.log"):
    """
    Escribe un mensaje de log con timestamp en un archivo de log y lo muestra en consola.

    Parámetros:
        message (str): Mensaje a registrar.
        logfile (str): Ruta del archivo de log (por defecto "simulation.log").
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    # Escribir en el archivo de log
    with open(logfile, "a") as f:
        f.write(full_message + "\n")
    # Imprimir el mensaje en consola
    print(full_message)

def load_config(filename):
    """
    Carga parámetros de configuración desde un archivo JSON.

    Parámetros:
        filename (str): Ruta del archivo JSON de configuración.

    Retorna:
        config (dict): Diccionario con los parámetros de configuración.
    """
    with open(filename, "r") as f:
        config = json.load(f)
    return config

if __name__ == "__main__":
    log_message("utils.py: helpers legacy de FEniCS removidos; disponible compute_dt_cfl() para DOLFINx.")

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

def compute_dt_cfl(mesh, cfl=0.3, c_max=1.0):
    """
    dt = cfl * h_min / c_max
    - DOLFINx: h mínimo por celdas locales + reducción MPI
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
    return float(cfl) * float(hmin) / max(float(c_max), 1e-12)
