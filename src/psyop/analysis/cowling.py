#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor de validez de la aproximación de Cowling (campo de prueba).

PSYOP evoluciona φ sobre un fondo FIJO: la consistencia exige que la
curvatura que el campo generaría sea despreciable frente a la del fondo.
Este monitor cuantifica esa hipótesis con dos números adimensionales:

- ζ_local = max_x [ 8π ρ(x) / √K(x) ], con ρ = T_μν n^μ n^ν la densidad de
  energía medida por observadores normales y √K ≈ 4√3·M/r³ la escala local
  de curvatura del fondo (raíz del escalar de Kretschmann de Schwarzschild;
  para Kerr es la misma escala salvo factores O(a²/r²)).
- ε_global = E_campo / M: masa-energía total del campo frente a la del
  agujero negro.

En fondo plano no hay curvatura de referencia: se reporta
ζ_local = 8π ρ R² (curvatura que el propio campo induciría comparada con
el tamaño del dominio) y ε_global = E/R.

Ambos ≪ 1 ⇒ la aproximación de campo de prueba es consistente. Valores
~0.01 ya ameritan precaución; el CLI loggea un warning sobre ese umbral.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import ufl

from psyop.utils.logger import get_logger

# 8π ρ ~ √K marca backreaction de orden 1; warning dos órdenes antes
DEFAULT_THRESHOLD = 1.0e-2


class CowlingMonitor:
    """Evalúa la validez del campo de prueba durante la evolución.

    Args:
        solver: FirstOrderKGSolver ya configurado (con fondo y potencial)
        metric_cfg: dict de configuración de la métrica ({"type", "M", ...})
    """

    def __init__(self, solver, metric_cfg: Dict):
        from dolfinx import fem

        self._fem = fem
        self.solver = solver
        self.mesh = solver.mesh
        self.metric_type = str(metric_cfg.get("type", "flat")).lower()
        self.M = float(metric_cfg.get("M", 1.0))
        self.R = float(solver.domain_radius)

        # ρ = ½Π² + ½ D_iφ D^iφ + V(φ) (igual que solver.energy, sin √γ)
        gammaInv = solver._GAMMAINV()
        gradphi = ufl.dot(gammaInv, ufl.grad(solver.phi_c))
        try:
            Vphi = solver.potential.evaluate_ufl(solver.phi_c)
        except (AttributeError, NotImplementedError):
            Vphi = 0.5 * solver.phi_c * solver.phi_c
        rho = (
            0.5 * solver.Pi_c * solver.Pi_c
            + 0.5 * ufl.inner(gradphi, ufl.grad(solver.phi_c))
            + Vphi
        )

        x = ufl.SpatialCoordinate(self.mesh)
        if self.metric_type in {"schwarzschild", "kerr"} and self.M > 0:
            r = ufl.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + 1.0e-12)
            sqrt_kretschmann = 4.0 * ufl.sqrt(3.0) * self.M / r**3
            zeta = 8.0 * ufl.pi * rho / sqrt_kretschmann
        else:
            zeta = 8.0 * ufl.pi * rho * self.R**2

        self._V0 = fem.functionspace(self.mesh, ("DG", 0))
        pts = self._V0.element.interpolation_points
        self._expr = fem.Expression(zeta, pts() if callable(pts) else pts)
        self._zeta_fn = fem.Function(self._V0)

    def evaluate(self, energy: float = None) -> Dict[str, float]:
        """Devuelve {"zeta_max", "energy_ratio"} (colectivo sobre el comm).

        Args:
            energy: E_campo ya calculada (evita re-ensamblar solver.energy())
        """
        from mpi4py import MPI

        self._zeta_fn.interpolate(self._expr)
        n_local = self._V0.dofmap.index_map.size_local
        arr = self._zeta_fn.x.array[:n_local]
        local_max = float(np.max(arr)) if arr.size else 0.0
        zeta_max = float(self.mesh.comm.allreduce(local_max, op=MPI.MAX))

        if energy is None:
            energy = self.solver.energy()
        scale = self.M if (self.metric_type != "flat" and self.M > 0) else self.R
        return {"zeta_max": zeta_max, "energy_ratio": float(energy / scale)}

    def check(
        self, t: float, threshold: float = DEFAULT_THRESHOLD, energy: float = None
    ) -> Dict[str, float]:
        """Como evaluate(), con warning (una vez) si se supera el umbral."""
        result = self.evaluate(energy=energy)
        if result["zeta_max"] > threshold and not getattr(self, "_warned", False):
            self._warned = True
            get_logger(__name__).warning(
                f"Cowling validity at t={t:.3f}: zeta_max={result['zeta_max']:.3e} "
                f"> {threshold:.0e}; the test-field approximation is becoming "
                "marginal (the field would backreact on the metric)"
            )
        return result
