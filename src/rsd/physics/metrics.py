from dataclasses import dataclass
from typing import Tuple
import numpy as np
import ufl
from rsd.backends.fem import Constant, is_dolfinx


@dataclass(frozen=True)
class Background:
    alpha: object
    beta: object
    gamma_inv: object
    sqrt_gamma: object
    K: object


class BackgroundCoeffs:
    def build(self, mesh) -> Tuple:
        raise NotImplementedError

    def build_background(self, mesh) -> Background:
        alpha, beta, gamma_inv, sqrt_gamma, K = self.build(mesh)
        return Background(alpha=alpha, beta=beta, gamma_inv=gamma_inv, sqrt_gamma=sqrt_gamma, K=K)

    def max_characteristic_speed(self, mesh) -> float:
        return 1.0

    def radial_factors_np(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(α, β·r̂, √γ^rr) como arrays NumPy en radios r (para datos
        iniciales consistentes con el fondo). Por defecto: espacio plano."""
        r = np.asarray(r, dtype=float)
        one = np.ones_like(r)
        return one, np.zeros_like(r), one


class FlatBackgroundCoeffs(BackgroundCoeffs):
    def build(self, mesh):
        dim = mesh.topology.dim if is_dolfinx() else mesh.geometric_dimension()
        alpha_f = Constant(mesh, 1.0)
        beta_f = None  # vector shift = 0
        gammaInv_f = ufl.Identity(dim)
        sqrtg_f = Constant(mesh, 1.0)
        K_f = Constant(mesh, 0.0)
        return alpha_f, beta_f, gammaInv_f, sqrtg_f, K_f

    def max_characteristic_speed(self, mesh) -> float:
        return 1.0


class SchwarzschildIsotropicCoeffs(BackgroundCoeffs):
    def __init__(self, M: float = 1.0):
        self.M = float(M)

    def build(self, mesh):
        dim = mesh.topology.dim if is_dolfinx() else mesh.geometric_dimension()
        x = ufl.SpatialCoordinate(mesh)
        eps = Constant(mesh, 1.0e-12)
        r = ufl.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + eps)
        m_over_2r = Constant(mesh, self.M) / (2.0 * r)
        psi = 1.0 + m_over_2r
        alpha_f = (1.0 - m_over_2r) / (1.0 + m_over_2r)
        beta_f = None  # sin shift en coordenadas isotrópicas
        gammaInv_f = (psi ** -4) * ufl.Identity(dim)
        sqrtg_f = psi ** 6
        K_f = Constant(mesh, 0.0)
        return alpha_f, beta_f, gammaInv_f, sqrtg_f, K_f

    def max_characteristic_speed(self, mesh) -> float:
        if not is_dolfinx():
            return 1.0
        x = mesh.geometry.x
        if x.shape[1] < 3 or x.shape[0] == 0:
            return 1.0
        r = np.sqrt(np.sum(x[:, :3] ** 2, axis=1))
        r = np.maximum(r, 1e-12)
        m_over_2r = self.M / (2.0 * r)
        alpha = (1.0 - m_over_2r) / (1.0 + m_over_2r)
        return float(np.max(np.abs(alpha)))

    def radial_factors_np(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Isotrópico: α=(1−M/2r)/(1+M/2r), β=0, √γ^rr = ψ⁻² = (1+M/2r)⁻²."""
        r = np.asarray(r, dtype=float)
        m_over_2r = self.M / (2.0 * np.maximum(r, 1e-12))
        psi2 = (1.0 + m_over_2r) ** 2
        alpha = (1.0 - m_over_2r) / (1.0 + m_over_2r)
        return alpha, np.zeros_like(r), 1.0 / psi2


class KerrSchildCoeffs(BackgroundCoeffs):
    """
    Métrica de Kerr en coordenadas Kerr-Schild (cartesianas).
    Basado en g = η + 2 H l ⊗ l, con l_mu nulo respecto a η.
    """

    def __init__(self, M: float = 1.0, a: float = 0.0):
        self.M = float(M)
        self.a = float(a)

    def build(self, mesh):
        dim = mesh.topology.dim if is_dolfinx() else mesh.geometric_dimension()
        if dim != 3:
            raise ValueError("Kerr-Schild requiere dominio 3D.")

        x = ufl.SpatialCoordinate(mesh)
        a = Constant(mesh, self.a)
        M = Constant(mesh, self.M)

        x0, y0, z0 = x[0], x[1], x[2]
        rho2 = x0**2 + y0**2 + z0**2
        a2 = a * a

        r2 = 0.5 * (rho2 - a2 + ufl.sqrt((rho2 - a2) ** 2 + 4.0 * a2 * z0**2))
        r = ufl.sqrt(r2 + Constant(mesh, 1.0e-15))

        denom = r2 + a2
        l = ufl.as_vector(((r * x0 + a * y0) / denom,
                           (r * y0 - a * x0) / denom,
                           z0 / r))
        H = M * r**3 / (r**4 + a2 * z0**2 + Constant(mesh, 1.0e-15))
        l2 = ufl.dot(l, l)
        l = l / ufl.sqrt(l2 + Constant(mesh, 1.0e-15))
        l2 = ufl.dot(l, l)

        factor = 2.0 * H

        gammaInv_f = ufl.Identity(3) - (factor / (1.0 + factor * l2)) * ufl.outer(l, l)
        sqrtg_f = ufl.sqrt(1.0 + factor * l2)
        alpha_f = 1.0 / sqrtg_f
        beta_f = (factor / (1.0 + factor * l2)) * l

        # La foliación Kerr-Schild NO es time-symmetric: K ≠ 0. Para un fondo
        # estacionario, con la convención K_ij = -(1/2)£_n γ_ij se tiene
        #   K = (1/α) D_i β^i = (1/(α√γ)) ∂_i(√γ β^i),
        # que UFL puede evaluar simbólicamente con div().
        # Para a=0 reproduce el valor conocido K = 2M(r+3M)/(r(r+2M))^{3/2}
        # (= 2Mα³(1+3M/r)/r², Baumgarte & Shapiro).
        K_f = ufl.div(sqrtg_f * beta_f) / (alpha_f * sqrtg_f)
        return alpha_f, beta_f, gammaInv_f, sqrtg_f, K_f

    def max_characteristic_speed(self, mesh) -> float:
        if not is_dolfinx():
            return 1.0
        x = mesh.geometry.x
        if x.shape[1] < 3 or x.shape[0] == 0:
            return 1.0
        x0, y0, z0 = x[:, 0], x[:, 1], x[:, 2]
        rho2 = x0**2 + y0**2 + z0**2
        a2 = self.a * self.a
        r2 = 0.5 * (rho2 - a2 + np.sqrt((rho2 - a2) ** 2 + 4.0 * a2 * z0**2))
        r = np.sqrt(r2 + 1.0e-15)
        H = self.M * r**3 / (r**4 + a2 * z0**2 + 1.0e-15)
        factor = 2.0 * H
        alpha = 1.0 / np.sqrt(1.0 + factor)
        beta_norm = np.abs(factor / (1.0 + factor))
        return float(np.max(alpha + beta_norm))

    def radial_factors_np(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Kerr-Schild radial: α=(1+2H)^{-1/2}, β·r̂=2H/(1+2H), √γ^rr=α.

        Exacto para a=0 (H=M/r, el rayo nulo entrante cumple dr/dt=−1).
        Para a≠0 se evalúa con H(r, θ=π/2) ≈ M/r como aproximación radial
        de orden dominante (el dato inicial mejora igualmente frente a la
        relación plana; el residuo se radia como junk reducido).
        """
        r = np.asarray(r, dtype=float)
        H = self.M / np.maximum(r, 1e-12)
        factor = 2.0 * H
        alpha = 1.0 / np.sqrt(1.0 + factor)
        beta_r = factor / (1.0 + factor)
        return alpha, beta_r, alpha


def kerr_excision_window(M: float, a: float) -> Tuple[float, float]:
    """Ventana admisible (lo, hi) del radio de excisión esférico cartesiano.

    Una esfera cartesiana ρ = R_exc barre radios de Boyer-Lindquist
    r ∈ [√(R_exc²−a²), R_exc]; el "do-nothing" del borde interior exige
    contenerla en la región atrapada r₋ < r < r₊, lo que da

        √(r₋² + a²) < R_exc < r₊         (derivación: docs/math/excision_window.md)

    Para |a| ≳ 0.9718 M la ventana se cierra (lo ≥ hi): ninguna esfera
    cartesiana cabe en la región atrapada y se necesita una superficie
    esferoidal r = const.

    Returns:
        (lo, hi): cotas de la ventana. Admisible solo si lo < hi.
    """
    M = float(M)
    a = abs(float(a))
    if a > M:
        raise ValueError(f"spin |a|={a} exceeds M={M} (naked singularity)")
    root = np.sqrt(M * M - a * a)
    r_plus, r_minus = M + root, M - root
    lo = float(np.sqrt(r_minus**2 + a**2))
    if a == 0.0:
        lo = 0.0
    return lo, float(r_plus)


def make_background(metric_cfg: dict) -> BackgroundCoeffs:
    mtype = metric_cfg.get("type", "flat").lower()
    if mtype == "flat":
        return FlatBackgroundCoeffs()
    if mtype == "schwarzschild":
        return SchwarzschildIsotropicCoeffs(M=metric_cfg.get("M", 1.0))
    if mtype == "kerr":
        return KerrSchildCoeffs(M=metric_cfg.get("M", 1.0), a=metric_cfg.get("a", 0.0))
    raise ValueError(f"Métrica no soportada: {mtype}")
