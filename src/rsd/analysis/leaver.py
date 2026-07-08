#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QNM de referencia para escalar (s=0) en Kerr vía fracciones continuas de
Leaver (1985), en la formulación del paquete `qnm` (Stein 2019, JOSS).

Convenciones: G = c = M = 1, spin a ∈ [0, 1), modos estables con
Im(ω) < 0. La ecuación angular (spheroidal, s=0) se resuelve como problema
de autovalores del operador L² − c²cos²θ en la base Y_lm (acoplamiento
pentadiagonal estándar), y la radial con la fracción continua de Leaver
invertida n veces para el overtone n.

Valores de control (Berti, Cardoso & Starinets 2009, escalar n=0, a=0):
    l=0: Mω = 0.110455 − 0.104896 i
    l=1: Mω = 0.292936 − 0.097660 i
    l=2: Mω = 0.483644 − 0.096759 i

Estos son los valores de referencia contra los que se valida la evolución
FEM (tests `slow`); este módulo los provee para cualquier spin sin tablas.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_EIKONAL = 1.0 / (3.0 * np.sqrt(3.0))  # 0.19245: ω ≈ (l+½−i(n+½))/(3√3)


# ----------------------------------------------------------------------
# Ecuación angular: A_lm(c) para s=0, c = a·ω (complejo)
# ----------------------------------------------------------------------

def _cos_theta_coupling(l: int, m: int) -> float:
    """⟨l+1,m|cosθ|l,m⟩ en la base de armónicos esféricos."""
    num = (l + 1) ** 2 - m**2
    den = (2 * l + 1) * (2 * l + 3)
    return np.sqrt(num / den)


def angular_eigenvalue(c: complex, l: int, m: int, nmax: int = 40) -> complex:
    """Autovalor de separación A_lm(c) de la ecuación spheroidal con s=0.

    Resuelve (L² − c²cos²θ) S = A S en la base {Y_l'm}, l' = |m| .. |m|+nmax,
    y devuelve el autovalor que continúa A = l(l+1) en c → 0.
    """
    if l < abs(m):
        raise ValueError(f"l={l} must be >= |m|={abs(m)}")
    l_min = abs(m)
    size = nmax + 1
    ls = np.arange(l_min, l_min + size)

    # cosθ tridiagonal en l; cos²θ = (cosθ)² acopla l, l±2
    cvec = np.array([_cos_theta_coupling(int(li), m) for li in ls])
    cos2_diag = np.zeros(size)
    cos2_diag[0] = cvec[0] ** 2
    cos2_diag[1:] = cvec[:-1] ** 2 + cvec[1:] ** 2
    # corrección: para l = l_min, ⟨cos²⟩ = c_{l-1}² + c_l², con c_{l-1}=0 si l=|m|...
    # (c_{l-1} solo existe dentro del bloque; fuera del bloque l-1 < |m| ⇒ 0)
    cos2_off2 = cvec[:-1] * cvec[1:]

    M = np.zeros((size, size), dtype=complex)
    np.fill_diagonal(M, ls * (ls + 1) - c * c * cos2_diag)
    for i in range(size - 2):
        M[i, i + 2] = -c * c * cos2_off2[i]
        M[i + 2, i] = -c * c * cos2_off2[i]

    eigvals = np.linalg.eigvals(M)
    target = l * (l + 1)
    return complex(eigvals[np.argmin(np.abs(eigvals - target))])


# ----------------------------------------------------------------------
# Ecuación radial: fracción continua de Leaver (s=0, M=1)
# ----------------------------------------------------------------------

def _radial_recurrence_coeffs(omega: complex, a: float, m: int, A: complex):
    """Coeficientes D0..D4 de la recurrencia de Leaver (qnm/radial.py, s=0)."""
    s = 0
    root = np.sqrt(1.0 - a * a)
    r_p, r_m = 1.0 + root, 1.0 - root
    sigma_p = (2.0 * omega * r_p - m * a) / (2.0 * root)
    sigma_m = (2.0 * omega * r_m - m * a) / (2.0 * root)
    zeta = 1.0j * omega
    xi = -s - 1.0j * sigma_p
    eta = -1.0j * sigma_m

    p = root * zeta
    alpha = 1.0 + s + xi + eta - 2.0 * zeta + s
    gamma = 1.0 + s + 2.0 * eta
    delta = 1.0 + s + 2.0 * xi
    sigma = (
        A
        + a * a * omega * omega
        - 8.0 * omega * omega
        + p * (2.0 * alpha + gamma - delta)
        + (1.0 + s - 0.5 * (gamma + delta)) * (s + 0.5 * (gamma + delta))
    )
    D = np.empty(5, dtype=complex)
    D[0] = delta
    D[1] = 4.0 * p - 2.0 * alpha + gamma - delta - 2.0
    D[2] = 2.0 * alpha - gamma + 2.0
    D[3] = alpha * (4.0 * p - delta) - sigma
    D[4] = alpha * (alpha - gamma + 1.0)
    return D


def _abg(n: int, D) -> tuple:
    alpha_n = n * n + (D[0] + 1.0) * n + D[0]
    beta_n = -2.0 * n * n + (D[1] + 2.0) * n + D[3]
    gamma_n = n * n + (D[2] - 3.0) * n + D[4] - D[2] + 2.0
    return alpha_n, beta_n, gamma_n


def _continued_fraction(omega: complex, a: float, m: int, A: complex,
                        n_inv: int = 0, depth: int = 2000) -> complex:
    """Residuo de la condición QNM con la fracción continua invertida n_inv veces.

    Cf. Leaver (1985): la raíz del residuo en ω (con A = A_lm(aω)) es el
    overtone n = n_inv. La cola infinita se evalúa hacia atrás desde `depth`.
    """
    D = _radial_recurrence_coeffs(omega, a, m, A)

    # Cola infinita: evaluación backward (estable) desde depth hasta n_inv+1
    tail = 0.0 + 0.0j
    for n in range(depth, n_inv, -1):
        alpha_n, beta_n, gamma_n = _abg(n, D)
        denom = beta_n - tail
        if denom == 0:
            denom = 1e-300
        alpha_prev = _abg(n - 1, D)[0]
        tail = alpha_prev * gamma_n / denom

    alpha_inv, beta_inv, gamma_inv_ = _abg(n_inv, D)
    residual = beta_inv - tail

    # Parte finita (inversión): fracción hacia abajo desde n_inv
    if n_inv > 0:
        finite = 0.0 + 0.0j
        for n in range(0, n_inv):
            alpha_n, beta_n, gamma_n = _abg(n, D)
            denom = beta_n - finite
            if denom == 0:
                denom = 1e-300
            gamma_next = _abg(n + 1, D)[2]
            finite = alpha_n * gamma_next / denom
        residual -= finite
    return residual


# ----------------------------------------------------------------------
# Root finding y API pública
# ----------------------------------------------------------------------

def _qnm_residual(omega: complex, a: float, l: int, m: int, n: int,
                  depth: int) -> complex:
    A = angular_eigenvalue(a * omega, l, m)
    return _continued_fraction(omega, a, m, A, n_inv=n, depth=depth)


def _complex_secant(f, z0: complex, z1: complex, tol: float = 1e-12,
                    maxiter: int = 100) -> complex:
    f0, f1 = f(z0), f(z1)
    for _ in range(maxiter):
        if f1 == f0:
            break
        z2 = z1 - f1 * (z1 - z0) / (f1 - f0)
        if abs(z2 - z1) < tol * max(1.0, abs(z2)):
            return z2
        z0, f0 = z1, f1
        z1, f1 = z2, f(z2)
    return z1


def kerr_qnm(l: int, m: int = 0, n: int = 0, a: float = 0.0,
             omega_guess: Optional[complex] = None, depth: int = 2000,
             da_max: float = 0.05) -> complex:
    """Frecuencia QNM Mω del escalar (s=0) en Kerr.

    Args:
        l, m: números angulares (l >= |m|)
        n: overtone (0 = fundamental)
        a: spin a/M en [0, 1)
        omega_guess: semilla opcional; si no, eikonal + continuación en spin
        depth: profundidad de la fracción continua
        da_max: paso máximo de la continuación en spin

    Returns:
        Mω complejo con Im < 0 (modo estable, convención e^{-iωt}).
    """
    if not (0.0 <= a < 1.0):
        raise ValueError(f"spin a must be in [0,1), got {a}")
    if l < abs(m):
        raise ValueError(f"l={l} must be >= |m|={abs(m)}")
    if n < 0:
        raise ValueError(f"overtone n must be >= 0, got {n}")

    if omega_guess is None:
        omega_guess = _EIKONAL * ((l + 0.5) - 1.0j * (n + 0.5))

    # Continuación en spin: resolver primero a=0 y avanzar en pasos cortos
    # (la semilla eikonal solo es confiable en Schwarzschild)
    spins = [0.0] if a == 0.0 else list(np.arange(0.0, a, da_max)) + [a]
    omega = omega_guess
    for a_k in spins:
        def f(w, _a=a_k):
            return _qnm_residual(w, _a, l, m, n, depth)

        omega = _complex_secant(f, omega, omega * (1.0 + 1e-5) + 1e-7j)
    if omega.imag > 0:
        omega = omega.conjugate()  # rama espejo: reportar el modo estable
    return complex(omega)


def schwarzschild_qnm(l: int, n: int = 0, depth: int = 2000) -> complex:
    """Mω del escalar en Schwarzschild (caso a=0, m irrelevante)."""
    return kerr_qnm(l=l, m=0, n=n, a=0.0, depth=depth)
