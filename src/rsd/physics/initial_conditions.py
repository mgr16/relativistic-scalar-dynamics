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

from rsd.utils.logger import get_logger

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
    φ(x) = v0 + A * exp(-((r - r0)²)/w²) · F(θ, ϕ)

    Factor angular F:
        l = 0: F ≡ 1 — convención esférica histórica (SIN factor Y_00);
            el canal extraído lleva c_00 = √(4π)·A·g(r), como documenta
            fit_log_profile_multipole.
        l ≥ 1: F = Y_lm real ortonormal — la MISMA real_ylm del extractor
            multipolar (una sola fuente para la convención de signos), de
            modo que en el continuo c_lm(r, t=0) = A·g(r) exacto en el canal
            (l, m) y cero en el resto. Con dato idéntico, el modo del
            oráculo 1D es u_l(r, 0) = A·g(r) y la relación de momento
            radial es la misma para todo l (set_initial_gaussian del
            oráculo): el junk del ansatz advectivo es idéntico en ambos
            mundos. Ojo física: para l > 0 la reducción 1D solo es exacta
            con potencial lineal (cuadrático) — un potencial no lineal
            acopla multipolos y eso es física exclusiva del 3D.

    Nota: la forma histórica v0*(1 + A·exp(...)) producía φ ≡ 0 cuando
    v0 = 0 (la amplitud quedaba multiplicada por el vacío). La forma
    aditiva coincide con la anterior para v0 = 1 y es correcta para v0 = 0.
    """

    VALID_DIRECTIONS = ("static", "ingoing", "outgoing", "ingoing_curved")

    def __init__(
        self,
        mesh: dmesh.Mesh,
        V: Optional[fem.FunctionSpace] = None,
        A: float = 1e-3,
        r0: float = 8.0,
        w: float = 2.0,
        v0: float = 1.0,
        direction: str = "static",
        background=None,
        l: int = 0,
        m: int = 0,
    ):
        """
        Parámetros:
            mesh: Malla del dominio
            V: Espacio de funciones
            A: Amplitud de la perturbación (para l ≥ 1 la amplitud física
                del campo es A·max|Y_lm| < A — el presupuesto de
                no-linealidad con A es conservador)
            r0: Centro radial de la perturbación
            w: Ancho de la perturbación
            v0: Valor de vacío del campo (siempre esférico: solo la
                perturbación lleva el factor angular)
            direction: "static" (Π=0, el pulso se divide en mitades entrante
                y saliente), "ingoing" o "outgoing" (momento de onda esférica
                en espacio PLANO: Π = ±(∂_r φ_pert + φ_pert/r)), o
                "ingoing_curved" (momento consistente con el fondo curvo:
                reduce la radiación espuria inicial; requiere `background`)
            background: BackgroundCoeffs con radial_factors_np(r) → (α, β·r̂,
                √γ^rr); solo necesario para "ingoing_curved". La relación es
                    Π = (√γ^rr) ∂_r φ_pert + (c_in/α) φ_pert/r ,
                con c_in = β·r̂ + α√γ^rr la velocidad coordenada entrante
                (en Kerr-Schild c_in = 1 exactamente y se reduce a
                Π = ((1−β_r)∂_rφ + φ/r)/α, validada contra el oráculo 1D).
            l, m: multipolo de la perturbación (enteros, l ≥ 0, |m| ≤ l);
                l=0 es el bump esférico histórico. El momento (cualquier
                direction) lleva el mismo factor angular que φ: el ansatz
                radial factoriza.
        """
        if direction not in self.VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {self.VALID_DIRECTIONS}, got {direction!r}"
            )
        if direction == "ingoing_curved" and background is None:
            raise ValueError(
                "direction='ingoing_curved' requires the `background` argument "
                "(a BackgroundCoeffs with radial_factors_np)"
            )
        if int(l) != l or int(m) != m:
            raise ValueError(f"l and m must be integers, got l={l!r}, m={m!r}")
        if int(l) < 0:
            raise ValueError(f"l must be >= 0, got {l}")
        if abs(int(m)) > int(l):
            raise ValueError(f"|m| must be <= l, got l={l}, m={m}")
        self.mesh = mesh
        self.V = V if V is not None else _scalar_functionspace(mesh)
        self.A = float(A)
        self.r0 = float(r0)
        self.w = float(w)
        self.v0 = float(v0)
        self.direction = direction
        self.background = background
        self.l = int(l)
        self.m = int(m)

        logger.debug(
            f"Creando GaussianBump: A={A}, r0={r0}, w={w}, v0={v0}, "
            f"dir={direction}, (l, m)=({self.l}, {self.m})"
        )
        self.phi = fem.Function(self.V, name="phi_initial")
        # interpolate() evalúa en los puntos correctos de cada elemento
        # (válido en paralelo y para cualquier grado, a diferencia de copiar
        # arrays evaluados en coordenadas de dofs)
        self.phi.interpolate(self._profile)

        self.Pi = None
        if direction != "static":
            self.Pi = fem.Function(self.V, name="Pi_initial")
            self.Pi.interpolate(self._momentum_profile)
        logger.info("GaussianBump inicializado")

    def _angular_factor(self, x: np.ndarray) -> np.ndarray:
        """F(θ, ϕ) en puntos cartesianos (gdim, npuntos); 1 para l = 0.

        Usa la MISMA real_ylm que la extracción multipolar: si esta
        convención y la del extractor divergieran, el canal (l, m) se
        rompería en silencio. El piso de r solo fija ángulos arbitrarios
        donde la perturbación ya es despreciable (r ≈ 0, sin excisión);
        el clip evita que z/r caiga fuera de [−1, 1] por redondeo.
        """
        if self.l == 0:
            return np.ones(x.shape[1])
        from rsd.analysis.extraction import real_ylm

        r = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        r_safe = np.maximum(r, 1e-12)
        theta = np.arccos(np.clip(x[2] / r_safe, -1.0, 1.0))
        phi_az = np.arctan2(x[1], x[0])
        return real_ylm(self.l, self.m, theta, phi_az)

    def _profile(self, x: np.ndarray) -> np.ndarray:
        """Perfil gaussiano; x tiene forma (gdim, npuntos) según DOLFINx."""
        r = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        pert = self.A * np.exp(-((r - self.r0) ** 2) / (self.w ** 2))
        return self.v0 + pert * self._angular_factor(x)

    def _momentum_profile(self, x: np.ndarray) -> np.ndarray:
        """
        Π consistente con una onda esférica pura.

        Plano: para φ = g(r ± t)/r (entrante/saliente) vale
        ∂_tφ = ±(∂_rφ + φ/r), aplicado solo a la parte radiativa (el vacío
        v0 no es una onda):
            Π = s · (∂_r φ_pert + φ_pert / r),  s = +1 entrante, -1 saliente.

        Curvo ("ingoing_curved"): el perfil advecta con la velocidad
        característica local c_in = β·r̂ + α√γ^rr, de modo que
            Π = √γ^rr ∂_r φ_pert + (c_in/α) φ_pert/r ,
        que elimina el transitorio espurio del dato plano sobre fondos
        curvos (en Kerr-Schild c_in = 1 exactamente).

        El piso de r es una fracción del ancho del pulso: con un piso de
        ~1e-12 la cola gaussiana (minúscula pero no nula) dividida por r→0
        produce un pico enorme de Π en el origen (e.g. 1e-7/1e-12 = 1e5).
        Dentro de r < w/10 la cola del cascarón es físicamente irrelevante.
        """
        r = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        r_safe = np.maximum(r, 0.1 * self.w)
        pert = self.A * np.exp(-((r - self.r0) ** 2) / (self.w ** 2))
        dpert_dr = -2.0 * (r - self.r0) / (self.w ** 2) * pert
        ang = self._angular_factor(x)
        if self.direction == "ingoing_curved":
            alpha, beta_r, sqrt_grr = self.background.radial_factors_np(r_safe)
            c_in = beta_r + alpha * sqrt_grr
            return (sqrt_grr * dpert_dr + (c_in / alpha) * pert / r_safe) * ang
        sign = 1.0 if self.direction == "ingoing" else -1.0
        return sign * (dpert_dr + pert / r_safe) * ang
    
    def get_function(self) -> fem.Function:
        """Retorna la función inicializada."""
        return self.phi

    def get_momentum(self) -> Optional[fem.Function]:
        """Retorna Π inicial (None si direction='static')."""
        return self.Pi

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
