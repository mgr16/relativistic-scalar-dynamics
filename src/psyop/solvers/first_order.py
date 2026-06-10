#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
first_order.py

Solver de primer orden para la ecuación de Klein-Gordon usando formulación (φ, Π).
Implementa SSP-RK3 con condiciones de frontera Sommerfeld.
DOLFINx-only implementation.
"""

from typing import Optional, Tuple, Dict, Any
import math

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
            raise ValueError(f"CFL factor must be in (0, 1], got {cfl_factor}")
        
        if domain_radius <= 0:
            raise ValueError(f"domain_radius must be positive, got {domain_radius}")
        
        if degree not in [1, 2, 3, 4, 5]:
            raise ValueError(f"degree must be in [1, 5], got {degree}")
        
        if potential_type not in VALID_POTENTIAL_TYPES:
            raise ValueError(f"Unknown potential type '{potential_type}'. options={VALID_POTENTIAL_TYPES}")
        
        self.mesh = mesh
        self.degree = degree
        self.cfl_factor = cfl_factor
        self.domain_radius = domain_radius
        self.cfg = kwargs.get("cfg", {})
        self.bc_type = kwargs.get("bc_type", self.cfg.get("solver", {}).get("bc_type", "characteristic")).lower()
        self._outer_tag = int(kwargs.get("outer_tag", self.cfg.get("solver", {}).get("outer_tag", 2)))
        self._inner_tag = int(kwargs.get("inner_tag", self.cfg.get("solver", {}).get("inner_tag", 3)))
        self.ko_eps = float(kwargs.get("ko_eps", self.cfg.get("solver", {}).get("ko_eps", 0.0)))
        self.ko_order = int(kwargs.get("ko_order", self.cfg.get("solver", {}).get("ko_order", 2)))
        if self.bc_type not in {"characteristic", "sommerfeld_spherical"}:
            raise ValueError("bc_type must be characteristic or sommerfeld_spherical")
        if self.ko_eps < 0:
            raise ValueError(f"ko_eps must be >= 0, got {self.ko_eps}")
        if self.ko_order not in (2, 4):
            raise ValueError(f"ko_order must be 2 or 4, got {self.ko_order}")

        # Grado de cuadratura explícito: la estimación automática de UFL
        # diverge con coeficientes métricos no polinómicos (Kerr-Schild
        # contiene sqrt/divisiones anidadas y el grado estimado explota)
        self.quad_degree = int(
            kwargs.get(
                "quadrature_degree",
                self.cfg.get("solver", {}).get("quadrature_degree", 2 * degree + 2),
            )
        )
        if self.quad_degree < 1:
            raise ValueError(f"quadrature_degree must be >= 1, got {self.quad_degree}")

        # Capa esponja: amortigua Π en una cáscara junto al borde exterior
        sponge_cfg = kwargs.get("sponge", self.cfg.get("solver", {}).get("sponge", {})) or {}
        self.sponge_enabled = bool(sponge_cfg.get("enabled", False))
        self.sponge_width = float(sponge_cfg.get("width", 0.0))
        self.sponge_strength = float(sponge_cfg.get("strength", 1.0))
        if self.sponge_enabled:
            if not (0 < self.sponge_width < domain_radius):
                raise ValueError(
                    f"sponge.width must be in (0, domain_radius), got {self.sponge_width}"
                )
            if self.sponge_strength <= 0:
                raise ValueError(f"sponge.strength must be > 0, got {self.sponge_strength}")

        # Flags de BC y métricas
        self.has_sommerfeld = False
        self.has_excision = False
        self.facet_tags = None
        self.ds_outer = None
        self.ds_inner = None
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
        
        self.preassemble_stiffness = bool(self.cfg.get("optimization", {}).get("preassemble", True))
        self._setup_operators()
        
        # Variables de estado
        self.current_time = 0.0
        self.current_dt = None
        # Función derivada temporal (du/dt)
        self.du = fem.Function(self.V, name="du")
        logger.info(f"FirstOrderKGSolver inicializado (grado={degree}, CFL={cfl_factor})")

    def _setup_operators(self) -> None:
        """Rebuild forms, matrices and PETSc solvers from the current metric/BC state."""
        self._setup_variational_forms()
        self._setup_mass_matrix_solver()
        self._setup_filter_matrix()

    def rebuild_operators(self) -> None:
        """API pública para reconstruir operadores tras configurar fondo/BCs
        con rebuild=False (evita re-ensamblar varias veces durante el setup)."""
        self._setup_operators()

    def _dx(self):
        """Medida de volumen con grado de cuadratura fijo."""
        return ufl.Measure(
            "dx", domain=self.mesh, metadata={"quadrature_degree": self.quad_degree}
        )
    
    def _setup_function_spaces(self):
        """Configura los espacios de funciones."""
        # Espacio vectorial para (φ, Π)
        try:
            element = ufl.VectorElement("Lagrange", self.mesh.ufl_cell(), self.degree, dim=2)
            self.V = fem.FunctionSpace(self.mesh, element)
        except AttributeError:
            self.V = fem.functionspace(self.mesh, ("Lagrange", self.degree, (2,)))
        
        # Espacio escalar auxiliar
        self.V_scalar = fem.functionspace(self.mesh, ("Lagrange", self.degree))
        
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
        dx = self._dx()
        
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
            # Misma física que la ruta preensamblada: la difusión se obtiene
            # aplicando la forma bilineal al estado actual (ufl.action), de modo
            # que ambas rutas resuelven exactamente la misma ecuación.
            self.rhs_form = (
                self._rhs_phi_form()
                + ufl.action(self._diffusion_form(), self.u)
                + self._rhs_Pi_transport_form()
            )
    
    def _rhs_phi_form(self):
        """RHS para φ: ∂φ/∂t = αΠ + β·∇φ"""
        dx = self._dx()
            
        alpha = self._ALPHA()
        beta  = self._BETA()
        sqrtg = self._SQRTG()

        term = (alpha * self.Pi_c) * self.test_phi * sqrtg * dx
        if beta is not None:
            term += ufl.dot(beta, ufl.grad(self.phi_c)) * self.test_phi * sqrtg * dx
        return term

    def _diffusion_form(self):
        """
        Forma bilineal para el término de difusión en Π.

        La integración por partes de -∫ √γ α γ^{ij}∂_iφ ∂_j(test) dx genera
        simultáneamente α D_iD^iφ y D^iα D_iφ, por lo que NO debe añadirse un
        término de gradiente del lapse por separado.

        Si hay excisión, se re-añade el término natural de borde interior
        +∫ α √γ (γ^{ij}∂_iφ n_j) test ds(inner), lo que equivale a no imponer
        condición alguna ("do-nothing"): correcto cuando todas las
        características salen del dominio a través del borde excisado
        (foliaciones horizon-penetrating).
        """
        dx = self._dx()
        alpha = self._ALPHA()
        sqrtg = self._SQRTG()
        gammaInv = self._GAMMAINV()

        u_trial = ufl.TrialFunction(self.V)
        v_test = ufl.TestFunction(self.V)
        phi_trial, _ = ufl.split(u_trial)
        _, test_Pi = ufl.split(v_test)

        form = - alpha * ufl.inner(ufl.dot(gammaInv, ufl.grad(phi_trial)), ufl.grad(test_Pi)) * sqrtg * dx
        if self.has_excision:
            n = ufl.FacetNormal(self.mesh)
            form += alpha * sqrtg * ufl.dot(ufl.dot(gammaInv, ufl.grad(phi_trial)), n) * test_Pi * self.ds_inner
        return form

    def _rhs_Pi_transport_form(self):
        """RHS para Π sin el término de difusión."""
        dx = self._dx()
            
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

        # Capa esponja: -σ(r)·Π, con rampa cúbica suave junto al borde.
        # Absorbe las colas dispersivas (campo masivo, v_grupo < c) que la
        # BC característica no captura. Nota: rompe por diseño el balance
        # E + ∫F dt = E0 (absorbe energía en el volumen, no por el borde).
        sigma = self._sponge_sigma()
        if sigma is not None:
            transport += -sigma * self.Pi_c * self.test_Pi * sqrtg * dx

        # Aporte de borde de Sommerfeld (si está habilitado)
        boundary_term = self._sommerfeld_boundary_term()

        return transport + boundary_term

    def _sponge_sigma(self):
        """σ(r) = strength·q³ con q = clip((r - r_inicio)/width, 0, 1)."""
        if not self.sponge_enabled:
            return None
        x = ufl.SpatialCoordinate(self.mesh)
        r = ufl.sqrt(ufl.dot(x, x) + 1.0e-15)
        r_start = self.domain_radius - self.sponge_width
        q = ufl.max_value(0.0, ufl.min_value(1.0, (r - r_start) / self.sponge_width))
        return self.sponge_strength * q ** 3

    def _setup_diffusion_matrix(self):
        """Pre-ensambla la matriz de difusión si está habilitado."""
        self.diffusion_mat = fem.petsc.assemble_matrix(fem.form(self.diffusion_form))
        self.diffusion_mat.assemble()
        self.diffusion_out = self.diffusion_mat.createVecLeft()
    
    def _sommerfeld_boundary_term(self):
        """
        Condición característica saliente impuesta débilmente.

        Al integrar por partes la difusión aparece el término natural
        +∫ α √γ (γ^{ij}∂_iφ n_j) test ds. La condición saliente
        ∂_tφ + λ ∂_nφ = 0 con λ = α√(γ^{nn}) - β·n (despreciando derivadas
        tangenciales) equivale a Π = -√(γ^{nn}) ∂_nφ, de modo que el flujo
        conormal se sustituye por γ^{nn}∂_nφ → -√(γ^{nn}) Π. En la variante
        esférica se usa Π = -√(γ^{nn})(∂_nφ + φ/r).

        En espacio plano se reduce al término absorbente clásico -∫ Π v ds.
        """
        if not self.has_sommerfeld:
            return 0 * self.test_Pi * self._dx()

        n = ufl.FacetNormal(self.mesh)
        alpha = self._ALPHA()
        sqrtg = self._SQRTG()
        gammaInv = self._GAMMAINV()
        gnn = ufl.dot(n, ufl.dot(gammaInv, n))  # γ^{nn} = n_i γ^{ij} n_j
        c_n = ufl.sqrt(gnn)
        if self.bc_type == "sommerfeld_spherical":
            x = ufl.SpatialCoordinate(self.mesh)
            r = ufl.sqrt(ufl.dot(x, x) + 1.0e-15)
            conormal_flux = -(c_n * self.Pi_c + gnn * self.phi_c / r)
        else:
            conormal_flux = -c_n * self.Pi_c
        return alpha * sqrtg * conormal_flux * self.test_Pi * self.ds_outer
    
    def _setup_mass_matrix_solver(self):
        """Configura el solver de la matriz de masa con métrica."""
        # DOLFINx
        self.mass_matrix = fem.petsc.assemble_matrix(fem.form(self.mass_form))
        self.mass_matrix.assemble()
        
        # Configurar solver
        self.mass_solver = PETSc.KSP().create(self.mesh.comm)
        self.mass_solver.setOperators(self.mass_matrix)
        self.mass_solver.setType(PETSc.KSP.Type.CG)
        pc = self.mass_solver.getPC()
        try:
            has_hypre = bool(PETSc.Sys.hasExternalPackage("hypre"))
        except (AttributeError, PETSc.Error):
            has_hypre = False
        pc.setType(PETSc.PC.Type.HYPRE if has_hypre else PETSc.PC.Type.JACOBI)
        self.mass_solver.setTolerances(rtol=1e-10, atol=1e-14)
        # Vectores PETSc para resolver A w = b
        self.rhs_vec = self.mass_matrix.createVecRight()
        self.sol_vec = self.mass_matrix.createVecLeft()

    def _setup_filter_matrix(self):
        """Configure explicit diffusion matrix for controlled dissipation."""
        if self.ko_eps <= 0:
            # Sin disipación activa no se ensambla el filtro (ahorro de memoria)
            self.filter_mat = None
            return
        u_trial = ufl.TrialFunction(self.V)
        v_test = ufl.TestFunction(self.V)
        phi_trial, Pi_trial = ufl.split(u_trial)
        test_phi, test_Pi = ufl.split(v_test)
        gammaInv = self._GAMMAINV()
        sqrtg = self._SQRTG()
        dx = self._dx()
        self.filter_form = (
            ufl.inner(ufl.dot(gammaInv, ufl.grad(phi_trial)), ufl.grad(test_phi))
            + ufl.inner(ufl.dot(gammaInv, ufl.grad(Pi_trial)), ufl.grad(test_Pi))
        ) * sqrtg * dx
        self.filter_mat = fem.petsc.assemble_matrix(fem.form(self.filter_form))
        self.filter_mat.assemble()
        self.filter_rhs = self.filter_mat.createVecLeft()
        self.filter_du = self.mass_matrix.createVecLeft()
        if self.ko_order >= 4:
            self.filter_du2 = self.mass_matrix.createVecLeft()
            # λmax(M⁻¹K) por iteración de potencias: normaliza el filtro de
            # 4.º orden para que su condición de estabilidad sea la misma
            # que la del de 2.º orden (h² subestima λmax en mallas con
            # celdas de mala calidad y la sensibilidad es cuadrática)
            self._filter_lambda_max = self._estimate_filter_lambda_max()

    def _estimate_filter_lambda_max(self, iters: int = 15) -> float:
        """Estima λmax(M⁻¹K) del filtro por iteración de potencias."""
        v = self.mass_matrix.createVecLeft()
        w = self.mass_matrix.createVecLeft()
        v.setRandom()
        norm = v.norm()
        if norm == 0.0:
            v.set(1.0)
            norm = v.norm()
        v.scale(1.0 / norm)
        lam = 1.0
        for _ in range(iters):
            self.filter_mat.mult(v, self.filter_rhs)
            self.mass_solver.solve(self.filter_rhs, w)
            lam = w.norm()
            if lam <= 0.0:
                return 1.0
            w.copy(v)
            v.scale(1.0 / lam)
        # margen del 10% por convergencia inexacta de la iteración
        return float(1.1 * lam)
    
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
        try:
            # Ruta robusta: interpolación al subespacio (independiente del
            # orden interno de los dofs, válida en paralelo y para degree > 1)
            self.u.sub(0).interpolate(phi_init)
            self.u.sub(1).interpolate(Pi_init)
        except (RuntimeError, AttributeError, TypeError):
            # Fallback por copia directa de arrays (asume mismo layout de dofs)
            self.u.x.array[self.phi_to_parent] = phi_init.x.array[:len(self.phi_to_parent)]
            self.u.x.array[self.Pi_to_parent] = Pi_init.x.array[:len(self.Pi_to_parent)]
        self.u.x.scatter_forward()
        logger.info("Condiciones iniciales establecidas")

    def set_background(self, alpha=None, beta=None, gammaInv=None, sqrtg=None, K=None,
                       rebuild: bool = True):
        """
        Configura coeficientes de fondo (α, β, γ⁻¹, √γ, K).

        Args:
            rebuild: si False, pospone el re-ensamblado de operadores (útil
                cuando a continuación se llamará a enable_sommerfeld/
                enable_excision, que ya reconstruyen los operadores).
        """
        self.alpha_f = alpha
        self.beta_f = beta
        self.gammaInv_f = gammaInv
        self.sqrtg_f = sqrtg
        self.K_f = K
        if rebuild:
            self._setup_operators()

    def enable_sommerfeld(self, facet_tags, outer_tag: int = 2, rebuild: bool = True) -> None:
        """Habilita Sommerfeld usando facet_tags externos ya definidos."""
        if facet_tags is None:
            raise ValueError("facet_tags is required when enable_sommerfeld is active")
        self._outer_tag = int(outer_tag)
        self.facet_tags = facet_tags
        self.ds_outer = ufl.Measure(
            "ds", domain=self.mesh, subdomain_data=facet_tags,
            subdomain_id=self._outer_tag,
            metadata={"quadrature_degree": self.quad_degree},
        )
        self.has_sommerfeld = True
        if rebuild:
            self._setup_operators()

    def enable_excision(self, facet_tags, inner_tag: int = 3, rebuild: bool = True) -> None:
        """
        Habilita el borde interior excisado (agujero negro).

        Añade el término natural de borde ("do-nothing"), válido cuando todas
        las características salen del dominio por el borde interior, como
        ocurre dentro del horizonte en foliaciones horizon-penetrating
        (Kerr-Schild) o en el horizonte de coordenadas isotrópicas (α→0).
        """
        if facet_tags is None:
            raise ValueError("facet_tags is required when excision is active")
        self._inner_tag = int(inner_tag)
        self.facet_tags = facet_tags
        self.ds_inner = ufl.Measure(
            "ds", domain=self.mesh, subdomain_data=facet_tags,
            subdomain_id=self._inner_tag,
            metadata={"quadrature_degree": self.quad_degree},
        )
        self.has_excision = True
        if rebuild:
            self._setup_operators()

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
        """Calcula dt por CFL con velocidad característica máxima y grado FEM."""
        return compute_dt_cfl(self.mesh, cfl=self.cfl_factor, c_max=c_max, degree=self.degree)

    def evolve(self, t_final: float = 1.0, dt: Optional[float] = None,
               output_every: Optional[int] = None, verbose: bool = False) -> float:
        """Evoluciona el sistema hasta t_final (el último paso se recorta).

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
        while t < t_final - 1e-12:
            step_dt = min(dt, t_final - t)
            self.ssp_rk3_step(step_dt)
            t += step_dt
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
                # action() convierte la forma bilineal en lineal evaluada en u
                fem.petsc.assemble_vector(
                    self.rhs_vec, fem.form(ufl.action(self.diffusion_form, self.u))
                )
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
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError(f"dt must be a positive finite number, got {dt}")

        u0 = self.u.x.array.copy()

        # Etapa 1
        self._assemble_rhs_and_solve_du()
        self.u1.x.array[:] = u0 + dt * self.du.x.array[:]

        # Etapa 2
        self.u.x.array[:] = self.u1.x.array[:]
        self.u.x.scatter_forward()
        self._assemble_rhs_and_solve_du()
        self.u2.x.array[:] = 0.75 * u0 + 0.25 * (
            self.u1.x.array[:] + dt * self.du.x.array[:]
        )

        # Etapa 3
        self.u.x.array[:] = self.u2.x.array[:]
        self.u.x.scatter_forward()
        self._assemble_rhs_and_solve_du()
        self.u_new.x.array[:] = (1.0 / 3.0) * u0 + (2.0 / 3.0) * (
            self.u2.x.array[:] + dt * self.du.x.array[:]
        )
        self.u.x.array[:] = self.u_new.x.array[:]
        if self.ko_eps > 0:
            self._apply_ko_dissipation(dt)
        self.u.x.scatter_forward()
        self.current_time += dt

    def _apply_ko_dissipation(self, dt: float) -> None:
        """
        Disipación explícita controlada por ko_eps.

        ko_order=2: filtro laplaciano  u <- u - ε·dt·(M⁻¹K)u
        ko_order=4: filtro biarmónico  u <- u - (ε·dt/λmax)·(M⁻¹K)²u
        La normalización por λmax(M⁻¹K) hace que el modo de malla más rápido
        se amortigüe con la misma tasa ε·dt·λmax en ambos órdenes (idéntica
        condición de estabilidad, ε·dt·λmax < 2), pero el de 4.º orden
        amortigua los modos suaves con tasa ∝ (λ/λmax)·λ ≪ λ (estilo
        Kreiss-Oliger).
        """
        if self.filter_mat is None:
            return
        self.u.x.scatter_forward()
        self.filter_mat.mult(self.u.x.petsc_vec, self.filter_rhs)
        self.mass_solver.solve(self.filter_rhs, self.filter_du)
        if self.ko_order >= 4:
            self.filter_mat.mult(self.filter_du, self.filter_rhs)
            self.mass_solver.solve(self.filter_rhs, self.filter_du2)
            scale = self.ko_eps * dt / self._filter_lambda_max
            self.u.x.array[:] -= scale * self.filter_du2.getArray(readonly=True)
        else:
            self.u.x.array[:] -= self.ko_eps * dt * self.filter_du.getArray(readonly=True)
    
    def energy(self) -> float:
        """
        Energía física medida por observadores normales:
        E = ∫ sqrt(γ) ρ d^3x, con ρ = T_{μν} n^μ n^ν
          = 1/2 Π^2 + 1/2 D_iφ D^iφ + V(φ)
        """
        sqrtg = self._SQRTG()
        gammaInv = self._GAMMAINV()
        gradphi = ufl.dot(gammaInv, ufl.grad(self.phi_c))
        # V(φ) usando el potencial configurado
        try:
            Vphi = self.potential.evaluate_ufl(self.phi_c)
        except (AttributeError, NotImplementedError):
            Vphi = 0.5 * self.phi_c * self.phi_c
        energy_density = (0.5 * ufl.inner(gradphi, ufl.grad(self.phi_c)) + 0.5 * self.Pi_c * self.Pi_c + Vphi) * sqrtg
        local_e = float(fem.assemble_scalar(fem.form(energy_density * self._dx())))
        return float(self.mesh.comm.allreduce(local_e))

    def boundary_flux(self) -> float:
        """
        Flujo de energía saliente por el borde exterior (positivo = sale).

        Se reporta el drenaje EXACTO del término débil de la BC radiativa:
        en el sistema semi-discreto dE/dt = b(Π), de modo que
            F_out = -b(Π) = ∮ α √γ (√(γ^{nn}) Π² [+ γ^{nn} Π φ/r]) ds
        cierra el balance E(t) + ∫F dt = E(0) hasta el error temporal de RK
        (verificado: el residuo converge ~h² con la resolución). Para una
        onda exactamente saliente coincide con el flujo físico de T_{μν},
        F = -∮ α√γ Π D_nφ ds = ∮ Π² ds.
        """
        if not getattr(self, 'has_sommerfeld', False):
            return 0.0
        n = ufl.FacetNormal(self.mesh)
        alpha = self._ALPHA()
        sqrtg = self._SQRTG()
        gammaInv = self._GAMMAINV()
        gnn = ufl.dot(n, ufl.dot(gammaInv, n))
        drain = ufl.sqrt(gnn) * self.Pi_c * self.Pi_c
        if self.bc_type == "sommerfeld_spherical":
            x = ufl.SpatialCoordinate(self.mesh)
            r = ufl.sqrt(ufl.dot(x, x) + 1.0e-15)
            drain = drain + gnn * self.Pi_c * self.phi_c / r
        formF = fem.form(alpha * sqrtg * drain * self.ds_outer)
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
