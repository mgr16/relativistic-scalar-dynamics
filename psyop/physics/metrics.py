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


def make_background(metric_cfg: dict) -> BackgroundCoeffs:
    mtype = metric_cfg.get("type", "flat").lower()
    if mtype == "flat":
        return FlatBackgroundCoeffs()
    if mtype == "schwarzschild":
        return SchwarzschildIsotropicCoeffs(M=metric_cfg.get("M", 1.0))
    raise ValueError(f"Métrica no soportada: {mtype}")
