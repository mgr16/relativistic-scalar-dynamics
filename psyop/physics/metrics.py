from typing import Tuple
import ufl
from psyop.backends.fem import Constant, is_dolfinx


class BackgroundCoeffs:
    def build(self, mesh) -> Tuple:
        raise NotImplementedError

    def max_characteristic_speed(self, mesh) -> float:
        return 1.0


class FlatBackgroundCoeffs(BackgroundCoeffs):
    def build(self, mesh):
        dim = mesh.topology.dim if is_dolfinx() else mesh.geometric_dimension()
        alpha_f = Constant(mesh, 1.0)
        beta_f = None  # vector shift = 0
        gammaInv_f = ufl.Identity(dim)
        sqrtg_f = Constant(mesh, 1.0)
        K_f = Constant(mesh, 0.0)
        return alpha_f, beta_f, gammaInv_f, sqrtg_f, K_f


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
        # En práctica ≤ 1; con placeholder dejamos 1.0
        return 1.0


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
        if abs(self.a) > self.M:
            raise ValueError("El spin |a| debe ser menor o igual a M.")

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
        l2 = ufl.dot(l, l)
        l = l / ufl.sqrt(l2 + Constant(mesh, 1.0e-15))
        l2 = ufl.dot(l, l)

        H = M * r**3 / (r**4 + a2 * z0**2 + Constant(mesh, 1.0e-15))

        factor = 2.0 * H

        gammaInv_f = ufl.Identity(3) - (factor / (1.0 + factor * l2)) * ufl.outer(l, l)
        sqrtg_f = ufl.sqrt(1.0 + factor * l2)
        alpha_f = 1.0 / sqrtg_f
        beta_f = (factor / (1.0 + factor * l2)) * l

        K_f = Constant(mesh, 0.0)
        return alpha_f, beta_f, gammaInv_f, sqrtg_f, K_f

    def max_characteristic_speed(self, mesh) -> float:
        return 1.0


def make_background(metric_cfg: dict) -> BackgroundCoeffs:
    mtype = metric_cfg.get("type", "flat").lower()
    if mtype == "flat":
        return FlatBackgroundCoeffs()
    if mtype == "schwarzschild":
        return SchwarzschildIsotropicCoeffs(M=metric_cfg.get("M", 1.0))
    if mtype == "kerr":
        return KerrSchildCoeffs(M=metric_cfg.get("M", 1.0), a=metric_cfg.get("a", 0.0))
    raise ValueError(f"Métrica no soportada: {mtype}")
