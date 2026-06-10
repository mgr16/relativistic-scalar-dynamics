#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
initial_conditions.py

Condiciones iniciales para campos escalares.
"""

from typing import List, Optional
import numpy as np
import dolfinx.fem as fem
import dolfinx.mesh as dmesh

from psyop.utils.logger import get_logger

logger = get_logger(__name__)


def _scalar_functionspace(mesh: dmesh.Mesh, degree: int = 1) -> fem.FunctionSpace:
    """Crea un espacio escalar Lagrange compatible con DOLFINx viejo y nuevo."""
    try:
        return fem.functionspace(mesh, ("Lagrange", degree))
    except AttributeError:  # DOLFINx < 0.7
        return fem.FunctionSpace(mesh, ("Lagrange", degree))


class GaussianBump:
    """
    Condición inicial tipo bump gaussiano para campo escalar.
    φ(r) = v0 * (1 + A * exp(-((r - r0)²)/w²))
    """
    
    def __init__(
        self, 
        mesh: dmesh.Mesh, 
        V: Optional[fem.FunctionSpace] = None, 
        A: float = 1e-3, 
        r0: float = 8.0, 
        w: float = 2.0, 
        v0: float = 1.0
    ):
        """
        Parámetros:
            mesh: Malla del dominio
            V: Espacio de funciones
            A: Amplitud de la perturbación
            r0: Centro radial de la perturbación
            w: Ancho de la perturbación
            v0: Valor de vacío del campo
        """
        self.mesh = mesh
        self.V = V if V is not None else _scalar_functionspace(mesh)
        self.A = float(A)
        self.r0 = float(r0)
        self.w = float(w)
        self.v0 = float(v0)

        logger.debug(f"Creando GaussianBump: A={A}, r0={r0}, w={w}, v0={v0}")
        self.phi = fem.Function(self.V, name="phi_initial")
        # interpolate() evalúa en los puntos correctos de cada elemento
        # (válido en paralelo y para cualquier grado, a diferencia de copiar
        # arrays evaluados en coordenadas de dofs)
        self.phi.interpolate(self._profile)
        logger.info("GaussianBump inicializado")

    def _profile(self, x: np.ndarray) -> np.ndarray:
        """Perfil gaussiano; x tiene forma (gdim, npuntos) según DOLFINx."""
        r = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        perturbation = self.A * np.exp(-((r - self.r0) ** 2) / (self.w ** 2))
        return self.v0 * (1.0 + perturbation)
    
    def get_function(self) -> fem.Function:
        """Retorna la función inicializada."""
        return self.phi

class PlaneWave:
    """
    Condición inicial de onda plana para pruebas.
    φ(x) = A * sin(k·x) donde k·x = kx*x + ky*y + kz*z
    """
    
    def __init__(
        self, 
        mesh: dmesh.Mesh, 
        V: fem.FunctionSpace, 
        A: float = 0.1, 
        k: Optional[List[float]] = None, 
        v0: float = 1.0
    ):
        """
        Parámetros:
            A: Amplitud
            k: Vector de onda [kx, ky, kz]
            v0: Valor base del campo
        """
        if k is None:
            k = [1, 0, 0]
        
        self.mesh = mesh
        self.V = V
        self.A = float(A)
        self.k = np.array(k, dtype=float)
        self.v0 = float(v0)
        
        logger.debug(f"Creando PlaneWave: A={A}, k={k}, v0={v0}")
        self.phi = fem.Function(V, name="phi_wave")
        self.phi.interpolate(self._profile)
        logger.info("PlaneWave inicializado")

    def _profile(self, x: np.ndarray) -> np.ndarray:
        """Onda plana; x tiene forma (gdim, npuntos) según DOLFINx."""
        k_dot_x = self.k[0] * x[0] + self.k[1] * x[1] + self.k[2] * x[2]
        return self.v0 + self.A * np.sin(k_dot_x)
    
    def get_function(self) -> fem.Function:
        """Retorna la función inicializada."""
        return self.phi

def create_zero_field(V: fem.FunctionSpace) -> fem.Function:
    """Crea un campo escalar con valor cero."""
    logger.debug("Creando campo cero")
    phi = fem.Function(V, name="zero_field")
    phi.x.array[:] = 0.0
    
    return phi

if __name__ == "__main__":
    # Prueba básica
    logger.info("=== Prueba de initial_conditions.py ===")
    
    try:
        from dolfinx.mesh import create_box
        from mpi4py import MPI
        mesh = create_box(MPI.COMM_WORLD, [[-5, -5, -5], [5, 5, 5]], [8, 8, 8])
        V = _scalar_functionspace(mesh)
        
        # Prueba Gaussian Bump
        gaussian = GaussianBump(mesh, V, A=0.1, r0=2.0, w=1.0)
        phi_gauss = gaussian.get_function()
        logger.info("✓ Gaussian Bump creado exitosamente")
        
        # Prueba Plane Wave
        wave = PlaneWave(mesh, V, A=0.1, k=[1, 1, 0])
        phi_wave = wave.get_function()
        logger.info("✓ Plane Wave creado exitosamente")
        
        # Prueba campo cero
        phi_zero = create_zero_field(V)
        logger.info("✓ Campo cero creado exitosamente")
        
        logger.info("Módulo initial_conditions.py completado exitosamente.")
        
    except Exception as e:
        logger.error(f"✗ Error en prueba: {e}")
        import traceback
        traceback.print_exc()
