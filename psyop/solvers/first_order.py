#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solver_first_order.py

Solver de primer orden para la ecuación de Klein-Gordon usando formulación (φ, Π).
Implementa SSP-RK3 con condiciones de frontera Sommerfeld.
DOLFINx-only implementation.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import time

import dolfinx.fem as fem
import dolfinx.fem.petsc
import ufl
from petsc4py import PETSc

from psyop.utils.logger import get_logger
from psyop.utils.utils import compute_dt_cfl
from psyop.physics.potential import get_potential
from psyop.physics.initial_conditions import GaussianBump, create_zero_field

logger = get_logger(__name__)

# Valid potential types
VALID_POTENTIAL_TYPES = ["quadratic", "higgs", "mexican_hat", "zero"]

class FirstOrderKGSolver:
    """
    Solver de primer orden para Klein-Gordon usando formulación (φ, Π).
    
    Sistema a resolver:
    Π := (1/α) (∂t φ - βⁱ∂ᵢφ)   (momento conjugado 3+1)
    ∂tφ = αΠ + βⁱ∂ᵢφ
    ∂tΠ = α DᵢDⁱφ + Dⁱα Dᵢφ + βⁱ∂ᵢΠ + αKΠ - αV'(φ)
    
    Con condiciones de frontera Sommerfeld: ∂φ/∂n + (1/r)φ = 0
    """
    
    def __init__(self, mesh, degree: int = 1, potential_type: str = "higgs", 
                 potential_params: Optional[Dict[str, Any]] = None,
                 cfl_factor: float = 0.5, domain_radius: float = 10.0, **kwargs):
        """
        Inicializa el solver.
        
        Args:
            mesh: Malla del dominio
            degree: Grado de los elementos finitos (1-5)
            potential_type: Tipo de potencial a usar
            potential_params: Parámetros del potencial
            cfl_factor: Factor CFL para time stepping adaptativo (0, 1]
            domain_radius: Radio del dominio (para BC Sommerfeld, > 0)
            
        Raises:
            ValueError: Si los parámetros de entrada no son válidos
        """
        # Input validation
        if not (0 < cfl_factor <= 1):
            raise ValueError(f"cfl_factor debe estar en (0, 1], got {cfl_factor}")
        
        if domain_radius <= 0:
            raise ValueError(f"domain_radius debe ser > 0, got {domain_radius}")
        
        if degree not in [1, 2, 3, 4, 5]:
            raise ValueError(f"degree debe estar en [1, 5], got {degree}")
        
        if potential_type not in VALID_POTENTIAL_TYPES:
            raise ValueError(f"Unknown potential type '{potential_type}'. options={VALID_POTENTIAL_TYPES}")
        
        self.mesh = mesh
        self.degree = degree
        self.cfl_factor = cfl_factor
        self.domain_radius = domain_radius
        self.cfg = kwargs.get("cfg", {})
        self.bc_type = kwargs.get("bc_type", self.cfg.get("solver", {}).get("bc_type", "characteristic")).lower()
        self._outer_tag = int(kwargs.get("outer_tag", self.cfg.get("solver", {}).get("outer_tag", 2)))
        self.ko_eps = float(kwargs.get("ko_eps", self.cfg.get("solver", {}).get("ko_eps", 0.0)))
        
        # Flags de BC y métricas
        self.has_sommerfeld = False
        self.alpha_f = None
        self.beta_f = None
        self.gammaInv_f = None
        self.sqrtg_f = None
        self.K_f = None

        # Configurar espacios de funciones
        self._setup_function_spaces()
        
        # Configurar potencial
        if potential_params is None:
            potential_params = {}
        self.potential = get_potential(potential_type, **potential_params)
        
        # Configurar formas variacionales
        self.preassemble_stiffness = bool(self.cfg.get("optimization", {}).get("preassemble", True))
        self._setup_variational_forms()
        
        # Configurar solver de matriz de masa
        self._setup_mass_matrix_solver()
        
        # Variables de estado
        self.current_time = 0.0
        self.current_dt = None
        # Función derivada temporal (du/dt)
        self.du = fem.Function(self.V, name="du")
        logger.info(f"FirstOrderKGSolver inicializado (grado={degree}, CFL={cfl_factor})")
    
    def _setup_function_spaces(self):
        """Configura los espacios de funciones."""
        # Espacio vectorial para (φ, Π)
        element = ufl.VectorElement("Lagrange", self.mesh.ufl_cell(), self.degree, dim=2)
        self.V = fem.FunctionSpace(self.mesh, element)
        
        # Espacio escalar auxiliar
        self.V_scalar = fem.FunctionSpace(self.mesh, ("Lagrange", self.degree))
        
        # Funciones de prueba
        v = ufl.TestFunction(self.V)
        self.test_phi, self.test_Pi = ufl.split(v)
        
        # Funciones de estado
        self.u = fem.Function(self.V, name="u")
        self.u_new = fem.Function(self.V, name="u_new")
        self.u1 = fem.Function(self.V, name="u1")
        self.u2 = fem.Function(self.V, name="u2")
        
        # Componentes actuales (dependen de self.u)
        self.phi_c, self.Pi_c = ufl.split(self.u)
        self.V_vector = self.V  # compatibilidad hacia atrás
        self.V_phi, self.phi_to_parent = self.V.sub(0).collapse()
        self.V_Pi, self.Pi_to_parent = self.V.sub(1).collapse()
    
    def _setup_variational_forms(self):
        """Configura las formas variacionales con métrica."""
        # Forma de masa con √γ
        dx = ufl.Measure("dx", domain=self.mesh)
        
        # Sistema de ecuaciones con métrica:
        # M * du/dt = F(u)
        # donde u = [φ, Π] y F(u) = [αΠ + β·∇φ, α∇·(γ⁻¹∇φ) + αKΠ - αV'(φ)]
        
        # Matriz de masa ponderada: ∫ √γ (test_φ * φ + test_Pi * Pi) dx
        u_trial = ufl.TrialFunction(self.V)
        v_test = ufl.TestFunction(self.V)
        phi_trial, Pi_trial = ufl.split(u_trial)
        test_phi_mass, test_Pi_mass = ufl.split(v_test)
        
        sqrtg = self._SQRTG()
        self.mass_form = (test_phi_mass * phi_trial + test_Pi_mass * Pi_trial) * sqrtg * dx
        
        if self.preassemble_stiffness:
            self.diffusion_form = self._diffusion_form()
            self.rhs_form = self._rhs_phi_form() + self._rhs_Pi_transport_form()
            self._setup_diffusion_matrix()
        else:
            # Forma del lado derecho (RHS) con métrica
            self.rhs_form = self._rhs_phi_form() + self._rhs_Pi_form()
    
    def _rhs_phi_form(self):
        """RHS para φ: ∂φ/∂t = αΠ + β·∇φ"""
        dx = ufl.Measure("dx", domain=self.mesh)
            
        alpha = self._ALPHA()
        beta  = self._BETA()
        sqrtg = self._SQRTG()

        term = (alpha * self.Pi_c) * self.test_phi * sqrtg * dx
        if beta is not None:
            term += ufl.dot(beta, ufl.grad(self.phi_c)) * self.test_phi * sqrtg * dx
        return term

    def _diffusion_form(self):
        """Forma bilineal para el término de difusión en Π."""
        dx = ufl.Measure("dx", domain=self.mesh)
        alpha = self._ALPHA()
        sqrtg = self._SQRTG()
        gammaInv = self._GAMMAINV()

        u_trial = ufl.TrialFunction(self.V)
        v_test = ufl.TestFunction(self.V)
        phi_trial, _ = ufl.split(u_trial)
        _, test_Pi = ufl.split(v_test)

        return - alpha * ufl.inner(ufl.dot(gammaInv, ufl.grad(phi_trial)), ufl.grad(test_Pi)) * sqrtg * dx

    def _rhs_Pi_transport_form(self):
        """RHS para Π sin el término de difusión."""
        dx = ufl.Measure("dx", domain=self.mesh)
            
        alpha = self._ALPHA()
        beta  = self._BETA()
        sqrtg = self._SQRTG()
        K = self._K()

        # Término de potencial
        Vp = self.potential.derivative_ufl(self.phi_c)

        # Transporte + curvatura extrínseca + potencial
        transport = (alpha*K*self.Pi_c - alpha*Vp) * self.test_Pi * sqrtg * dx
        if beta is not None:
            transport += ufl.dot(beta, ufl.grad(self.Pi_c)) * self.test_Pi * sqrtg * dx

        # Aporte de borde de Sommerfeld (si está habilitado)
        boundary_term = self._sommerfeld_boundary_term()

        return transport + boundary_term

    def _rhs_Pi_form(self):
        """RHS para Π: ∂Π/∂t = α∇·(γ⁻¹∇φ) + αKΠ - αV'(φ)"""
        dx = ufl.Measure("dx", domain=self.mesh)
            
        alpha = self._ALPHA()
        beta  = self._BETA()
        sqrtg = self._SQRTG()
        gammaInv = self._GAMMAINV()
        K = self._K()

        # Término de potencial
        Vp = self.potential.derivative_ufl(self.phi_c)

        # Difusión: - ∫ √γ * α * (γ^{ij} ∂_i φ ∂_j test_Pi) dx
        diffusion = - alpha * ufl.inner(ufl.dot(gammaInv, ufl.grad(self.phi_c)), ufl.grad(self.test_Pi)) * sqrtg * dx
        lapse_gradient = ufl.inner(ufl.dot(gammaInv, ufl.grad(alpha)), ufl.grad(self.phi_c)) * self.test_Pi * sqrtg * dx

        # Transporte + curvatura extrínseca + potencial
        transport = (alpha*K*self.Pi_c - alpha*Vp) * self.test_Pi * sqrtg * dx
        if beta is not None:
            transport += ufl.dot(beta, ufl.grad(self.Pi_c)) * self.test_Pi * sqrtg * dx

        # Aporte de borde de Sommerfeld (si está habilitado)
        boundary_term = self._sommerfeld_boundary_term()

        return diffusion + lapse_gradient + transport + boundary_term

    def _setup_diffusion_matrix(self):
        """Pre-ensambla la matriz de difusión si está habilitado."""
        self.diffusion_mat = fem.petsc.assemble_matrix(fem.form(self.diffusion_form))
        self.diffusion_mat.assemble()
        self.diffusion_out = self.diffusion_mat.createVecLeft()
    
    def _sommerfeld_boundary_term(self):
        """
        Término de Sommerfeld con velocidad característica saliente.
        
        Usamos condición característica: Π + c_out ∂n φ = 0
        con c_out = α - β·n.
        """
        if not self.has_sommerfeld:
            return ufl.Constant(self.mesh, 0.0)
        
        n = ufl.FacetNormal(self.mesh)
        alpha = self._ALPHA()
        beta = self._BETA()
        c_out = alpha if beta is None else alpha - ufl.dot(beta, n)
        if self.bc_type == "sommerfeld_spherical":
            r = ufl.sqrt(ufl.dot(ufl.SpatialCoordinate(self.mesh), ufl.SpatialCoordinate(self.mesh)) + 1.0e-15)
            flux_term = -(self.Pi_c + c_out * (ufl.dot(ufl.grad(self.phi_c), n) + self.phi_c / r)) * self.test_Pi * self.ds_outer
        else:
            flux_term = -(self.Pi_c + c_out * ufl.dot(ufl.grad(self.phi_c), n)) * self.test_Pi * self.ds_outer
        return flux_term
    
    def _setup_mass_matrix_solver(self):
        """Configura el solver de la matriz de masa con métrica."""
        # DOLFINx
        self.mass_matrix = fem.petsc.assemble_matrix(fem.form(self.mass_form))
        self.mass_matrix.assemble()
        
        # Configurar solver
        self.mass_solver = PETSc.KSP().create(self.mesh.comm)
        self.mass_solver.setOperators(self.mass_matrix)
        self.mass_solver.setType(PETSc.KSP.Type.CG)
        self.mass_solver.getPC().setType(PETSc.PC.Type.HYPRE)
        self.mass_solver.setTolerances(rtol=1e-12, atol=1e-15)
        # Vectores PETSc para resolver A w = b
        self.rhs_vec = self.mass_matrix.createVecRight()
        self.sol_vec = self.mass_matrix.createVecLeft()
    
    def set_initial_conditions(self, phi_init: Optional[fem.Function] = None, 
                               Pi_init: Optional[fem.Function] = None) -> None:
        """
        Establece condiciones iniciales.
        
        Args:
            phi_init: Función inicial para φ (si None, usa Gaussian bump)
            Pi_init: Función inicial para Π (si None, usa cero)
        """
        V_scalar = self.V_scalar
        if phi_init is None:
            phi_init = GaussianBump(self.mesh, V_scalar).get_function()
        if Pi_init is None:
            Pi_init = create_zero_field(V_scalar)
        self.u.x.array[self.phi_to_parent] = phi_init.x.array[:len(self.phi_to_parent)]
        self.u.x.array[self.Pi_to_parent] = Pi_init.x.array[:len(self.Pi_to_parent)]
        self.u.x.scatter_forward()
        logger.info("Condiciones iniciales establecidas")

    def set_background(self, alpha=None, beta=None, gammaInv=None, sqrtg=None, K=None):
        """Configura coeficientes de fondo (α, β, γ⁻¹, √γ, K)."""
        self.alpha_f = alpha
        self.beta_f = beta
        self.gammaInv_f = gammaInv
        self.sqrtg_f = sqrtg
        self.K_f = K

    def enable_sommerfeld(self, facet_tags, outer_tag: int = 2) -> None:
        """Habilita Sommerfeld usando facet_tags externos ya definidos."""
        if facet_tags is None:
            raise ValueError("facet_tags is required when enable_sommerfeld is active")
        self._outer_tag = int(outer_tag)
        self.facet_tags = facet_tags
        self.ds_outer = ufl.Measure("ds", domain=self.mesh, subdomain_data=facet_tags, subdomain_id=self._outer_tag)
        self.has_sommerfeld = True

    def get_fields(self) -> Tuple[fem.Function, fem.Function]:
        """Devuelve (phi, Pi) en espacio escalar."""
        phi = fem.Function(self.V_phi, name="phi")
        Pi = fem.Function(self.V_Pi, name="Pi")
        phi.x.array[:] = self.u.x.array[self.phi_to_parent]
        Pi.x.array[:] = self.u.x.array[self.Pi_to_parent]
        phi.x.scatter_forward()
        Pi.x.scatter_forward()
        return phi, Pi

    def compute_adaptive_dt(self, c_max: float = 1.0) -> float:
        """Calcula dt por CFL con velocidad característica máxima."""
        return compute_dt_cfl(self.mesh, cfl=self.cfl_factor, c_max=c_max)

    def evolve(self, t_final: float = 1.0, dt: Optional[float] = None, 
               output_every: Optional[int] = None, verbose: bool = False) -> float:
        """Evoluciona el sistema hasta t_final.
        
        Args:
            t_final: Tiempo final de la evolución
            dt: Paso de tiempo (si None, se calcula usando CFL)
            output_every: Cada cuántos pasos imprimir info (si None, no imprime)
            verbose: Si True, imprime información durante la evolución
            
        Returns:
            Tiempo final alcanzado
        """
        if dt is None:
            dt = self.compute_adaptive_dt()
        t = 0.0
        step = 0
        while t < t_final:
            self.ssp_rk3_step(dt)
            t += dt
            step += 1
            if verbose and output_every and step % output_every == 0:
                logger.info(f"t={t:.6f}")
        return t
    
    def _assemble_rhs_and_solve_du(self) -> None:
        """Ensamblar RHS(u) y resolver M du = RHS, dejando du en self.du."""
        rhs_form_with_bc = self.rhs_form  # _sommerfeld_boundary_term ya incluido
        self.rhs_vec.zeroEntries()
        fem.petsc.assemble_vector(self.rhs_vec, fem.form(rhs_form_with_bc))
        if self.preassemble_stiffness:
            try:
                self.u.x.scatter_forward()
                self.diffusion_mat.mult(self.u.x.petsc_vec, self.diffusion_out)
                self.rhs_vec.axpy(1.0, self.diffusion_out)
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Error usando matriz preensamblada, fallback a ensamblado: {e}")
                fem.petsc.assemble_vector(self.rhs_vec, fem.form(self._diffusion_form()))
        self.rhs_vec.assemble()
        self.mass_solver.solve(self.rhs_vec, self.sol_vec)
        # Copiar solución a self.du
        self.du.x.array[:] = self.sol_vec.getArray(readonly=True)
        self.du.x.scatter_forward()
    
    def ssp_rk3_step(self, dt: float) -> None:
        """
        Realiza un paso de integración SSP-RK3.
        
        Args:
            dt: Paso de tiempo
        
        u^(1) = u^n + dt * L(u^n)
        u^(2) = (3/4) * u^n + (1/4) * u^(1) + (1/4) * dt * L(u^(1))
        u^(n+1) = (1/3) * u^n + (2/3) * u^(2) + (2/3) * dt * L(u^(2))
        """
        # Etapa 1
        self._assemble_rhs_and_solve_du()
        self.u1.x.array[:] = self.u.x.array[:] + dt * self.du.x.array[:]
        # Etapa 2
        self.u.x.array[:] = self.u1.x.array[:]
        self._assemble_rhs_and_solve_du()
        self.u2.x.array[:] = (0.75 * self.u1.x.array[:] + 0.25 * self.u.x.array[:] + 0.25 * dt * self.du.x.array[:])
        # Etapa 3
        self.u.x.array[:] = self.u2.x.array[:]
        self._assemble_rhs_and_solve_du()
        self.u_new.x.array[:] = (1.0/3.0 * self.u1.x.array[:] + 2.0/3.0 * self.u2.x.array[:] + 2.0/3.0 * dt * self.du.x.array[:])
        self.u.x.array[:] = self.u_new.x.array[:]
        if self.ko_eps > 0:
            self.u.x.array[:] = (1.0 - self.ko_eps) * self.u.x.array[:] + self.ko_eps * self.u2.x.array[:]
        self.u.x.scatter_forward()
        self.current_time += dt
    
    def energy(self) -> float:
        """E = ∫ sqrtg [ 1/2 (γ^{ij}∂_i φ ∂_j φ) + 1/2 Π^2 + V(φ) ] dx"""
        sqrtg = self._SQRTG()
        gammaInv = self._GAMMAINV()
        gradphi = ufl.dot(gammaInv, ufl.grad(self.phi_c))
        # V(φ) usando el potencial configurado
        try:
            Vphi = self.potential.evaluate_ufl(self.phi_c)
        except (AttributeError, NotImplementedError):
            Vphi = 0.5 * self.phi_c * self.phi_c
        energy_density = (0.5 * ufl.inner(gradphi, ufl.grad(self.phi_c)) + 0.5 * self.Pi_c * self.Pi_c + Vphi) * sqrtg
        local_e = float(fem.assemble_scalar(fem.form(energy_density * ufl.dx)))
        return float(self.mesh.comm.allreduce(local_e))

    def boundary_flux(self) -> float:
        """Flujo aproximado en Γ_out: F ≈ ∫ Π (∂n φ) ds"""
        if not getattr(self, 'has_sommerfeld', False):
            return 0.0
        n = ufl.FacetNormal(self.mesh)
        alpha = self._ALPHA()
        beta = self._BETA()
        c_out = alpha if beta is None else alpha - ufl.dot(beta, n)
        flux_density = c_out * self.Pi_c * self.Pi_c
        formF = fem.form(flux_density * self.ds_outer)
        local_flux = float(fem.assemble_scalar(formF))
        return float(self.mesh.comm.allreduce(local_flux))

    def _ALPHA(self):
        if self.alpha_f is not None:
            return self.alpha_f
        return fem.Constant(self.mesh, 1.0)

    def _BETA(self):
        return self.beta_f

    def _GAMMAINV(self):
        if self.gammaInv_f is not None:
            return self.gammaInv_f
        dim = self.mesh.topology.dim
        return ufl.Identity(dim)

    def _SQRTG(self):
        if self.sqrtg_f is not None:
            return self.sqrtg_f
        return fem.Constant(self.mesh, 1.0)

    def _K(self):
        if self.K_f is not None:
            return self.K_f
        return fem.Constant(self.mesh, 0.0)

if __name__ == "__main__":
    # Prueba básica
    logger.info("=== Prueba de solver_first_order.py ===")
    
    try:
        # Crear mesh simple para prueba
        from dolfinx.mesh import create_box
        from mpi4py import MPI
        mesh = create_box(MPI.COMM_WORLD, [[-2, -2, -2], [2, 2, 2]], [4, 4, 4])
        
        # Crear solver
        solver = FirstOrderKGSolver(
            mesh=mesh,
            degree=1,
            potential_type="quadratic",
            potential_params={"m_squared": 1.0},
            cfl_factor=0.3,
            domain_radius=2.0
        )
        
        # Establecer condiciones iniciales
        solver.set_initial_conditions()
        
        # Evolución corta
        solver.evolve(t_final=0.1, verbose=True)
        
        # Obtener campos finales
        phi, Pi = solver.get_fields()
        
        logger.info("✓ Solver de primer orden completado exitosamente")
        
    except Exception as e:
        logger.error(f"✗ Error en prueba: {e}")
        import traceback
        traceback.print_exc()
