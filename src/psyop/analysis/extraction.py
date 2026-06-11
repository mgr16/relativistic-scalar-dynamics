#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extraction.py

Extracción multipolar: proyecta el campo escalar sobre armónicos esféricos
reales en una esfera de extracción de radio dado,
    c_lm(t) = ∮ φ(R_ext, θ, ϕ) Y_lm(θ, ϕ) dΩ ,
con cuadratura Gauss-Legendre en cos θ × trapecio uniforme en ϕ (exacta para
productos de armónicos hasta l = lmax).

Mucho menos ruidosa que muestrear un punto y permite identificar (l, m).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from psyop.utils.logger import get_logger

logger = get_logger(__name__)

try:  # scipy >= 1.15
    from scipy.special import sph_harm_y as _sph_harm_y

    def _complex_ylm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return _sph_harm_y(l, m, theta, phi)

except ImportError:  # pragma: no cover - scipy viejo
    from scipy.special import sph_harm as _sph_harm

    def _complex_ylm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        # sph_harm(m, l, azimut, polar)
        return _sph_harm(m, l, phi, theta)


def real_ylm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Armónicos esféricos reales ortonormales:
        m = 0:  Y_l0
        m > 0:  √2 (−1)^m Re(Y_lm)
        m < 0:  √2 (−1)^m Im(Y_l|m|)
    """
    if m == 0:
        return np.real(_complex_ylm(l, 0, theta, phi))
    if m > 0:
        return np.sqrt(2.0) * (-1.0) ** m * np.real(_complex_ylm(l, m, theta, phi))
    return np.sqrt(2.0) * (-1.0) ** m * np.imag(_complex_ylm(l, -m, theta, phi))


class MultipoleExtractor:
    """
    Extractor multipolar sobre una esfera de radio `radius`.

    La localización de los puntos de cuadratura en la malla (ownership por
    rank) y los Y_lm se precomputan una sola vez; cada llamada a extract()
    solo evalúa el campo y hace la cuadratura + una reducción MPI global.
    """

    def __init__(self, mesh, radius: float, lmax: int = 2,
                 n_theta: Optional[int] = None, n_phi: Optional[int] = None):
        if radius <= 0:
            raise ValueError(f"extraction radius must be > 0, got {radius}")
        if lmax < 0:
            raise ValueError(f"lmax must be >= 0, got {lmax}")
        self.mesh = mesh
        self.radius = float(radius)
        self.lmax = int(lmax)
        # Cuadratura exacta para productos Y_lm·Y_l'm' hasta l = lmax
        self.n_theta = int(n_theta) if n_theta else self.lmax + 2
        self.n_phi = int(n_phi) if n_phi else max(2 * self.lmax + 4, 8)

        # Nodos: Gauss-Legendre en cosθ, uniforme en ϕ
        nodes, gl_weights = np.polynomial.legendre.leggauss(self.n_theta)
        theta = np.arccos(nodes)
        phi = 2.0 * np.pi * np.arange(self.n_phi) / self.n_phi

        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
        w_grid = np.repeat(gl_weights[:, None], self.n_phi, axis=1) * (2.0 * np.pi / self.n_phi)
        self._theta = theta_grid.ravel()
        self._phi = phi_grid.ravel()
        self._weights = w_grid.ravel()

        sin_t = np.sin(self._theta)
        self._points = np.column_stack([
            self.radius * sin_t * np.cos(self._phi),
            self.radius * sin_t * np.sin(self._phi),
            self.radius * np.cos(self._theta),
        ])

        self.modes: List[Tuple[int, int]] = [
            (l, m) for l in range(self.lmax + 1) for m in range(-l, l + 1)
        ]
        self._ylm = {
            (l, m): real_ylm(l, m, self._theta, self._phi) for (l, m) in self.modes
        }

        self._locate_points()
        logger.info(
            f"MultipoleExtractor: R={radius}, lmax={lmax}, "
            f"{self.n_theta}x{self.n_phi} puntos de cuadratura"
        )

    def _locate_points(self) -> None:
        """Precalcula qué puntos de cuadratura posee este rank y en qué celda."""
        import dolfinx.geometry

        tree = dolfinx.geometry.bb_tree(self.mesh, self.mesh.topology.dim)
        candidates = dolfinx.geometry.compute_collisions_points(tree, self._points)
        colliding = dolfinx.geometry.compute_colliding_cells(
            self.mesh, candidates, self._points
        )
        owned_idx = []
        owned_cell = []
        for i in range(self._points.shape[0]):
            links = colliding.links(i)
            if len(links) > 0:
                owned_idx.append(i)
                owned_cell.append(links[0])
        self._owned_idx = np.asarray(owned_idx, dtype=np.int64)
        self._owned_cells = np.asarray(owned_cell, dtype=np.int32)

    def _evaluate(self, phi_fn) -> np.ndarray:
        """Evalúa el campo en todos los puntos de cuadratura (global MPI)."""
        n = self._points.shape[0]
        vals = np.zeros(n)
        mask = np.zeros(n)
        if self._owned_idx.size > 0:
            local = phi_fn.eval(self._points[self._owned_idx], self._owned_cells)
            vals[self._owned_idx] = np.asarray(local).reshape(-1)
            mask[self._owned_idx] = 1.0
        comm = self.mesh.comm
        vals = comm.allreduce(vals)
        mask = comm.allreduce(mask)
        missing = int(np.sum(mask == 0))
        if missing:
            logger.warning(
                f"{missing}/{n} puntos de extracción fuera de la malla "
                f"(¿radio {self.radius} demasiado cerca del borde?)"
            )
        # Puntos compartidos entre ranks: promediar las evaluaciones
        return vals / np.maximum(mask, 1.0)

    def extract(self, phi_fn) -> Dict[Tuple[int, int], float]:
        """Devuelve {(l, m): c_lm} con c_lm = ∮ φ Y_lm dΩ."""
        field = self._evaluate(phi_fn)
        wf = self._weights * field
        return {(l, m): float(np.dot(wf, self._ylm[(l, m)])) for (l, m) in self.modes}

    def header(self) -> List[str]:
        """Nombres de columna para series de tiempo (l{l}_m{m})."""
        return [f"l{l}_m{m}" for (l, m) in self.modes]
