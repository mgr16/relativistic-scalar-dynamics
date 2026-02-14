#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
initial_conditions.py

Condiciones iniciales para campos escalares.
"""

from typing import List
import numpy as np
import dolfinx.fem as fem
import ufl

from psyop.utils.logger import get_logger

logger = get_logger(__name__)

class GaussianBump:
    """
    Condición inicial tipo bump gaussiano para campo escalar.
    φ(r) = v0 * (1 + A * exp(-((r - r0)²)/w²))
    """
    
    def __init__(
        self, 
        mesh: fem.Mesh, 
        V: fem.FunctionSpace, 
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
        self.V = V
        self.A = float(A)
        self.r0 = float(r0)
        self.w = float(w)
        self.v0 = float(v0)
        
        logger.debug(f"Creando GaussianBump: A={A}, r0={r0}, w={w}, v0={v0}")
        self.phi = fem.Function(V, name="phi_initial")
        self._set_dolfinx_values()
    
    def _gaussian_expr(self, x: np.ndarray) -> np.ndarray:
        """Evaluación de la expresión gaussiana."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        r = np.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)
        perturbation = self.A * np.exp(-((r - self.r0)**2) / (self.w**2))
        return self.v0 * (1.0 + perturbation)
    
    def _set_dolfinx_values(self) -> None:
        """Configurar valores para DOLFINx."""
        # Obtener coordenadas de los DOFs
        if hasattr(self.V, 'tabulate_dof_coordinates'):
            dof_coords = self.V.tabulate_dof_coordinates()
        else:
            # Fallback para versiones más nuevas
            dof_coords = self.mesh.geometry.x
        
        # Evaluar la función gaussiana
        values = self._gaussian_expr(dof_coords)
        
        # Asignar valores
        self.phi.x.array[:] = values.astype(np.float64)
        logger.info("GaussianBump inicializado")
    
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
        mesh: fem.Mesh, 
        V: fem.FunctionSpace, 
        A: float = 0.1, 
        k: List[float] = None, 
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
        self._set_dolfinx_wave()
    
    def _wave_expr(self, x: np.ndarray) -> np.ndarray:
        """Evaluación de la onda plana."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # k·x
        k_dot_x = np.dot(x, self.k)
        return self.v0 + self.A * np.sin(k_dot_x)
    
    def _set_dolfinx_wave(self) -> None:
        """Configurar onda plana para DOLFINx."""
        if hasattr(self.V, 'tabulate_dof_coordinates'):
            dof_coords = self.V.tabulate_dof_coordinates()
        else:
            dof_coords = self.mesh.geometry.x
        
        values = self._wave_expr(dof_coords)
        self.phi.x.array[:] = values.astype(np.float64)
        logger.info("PlaneWave inicializado")
    
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
        V = fem.FunctionSpace(mesh, ("Lagrange", 1))
        
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
