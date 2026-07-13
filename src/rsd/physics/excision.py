"""Dependency-free geometry helpers for black-hole excision."""

from math import sqrt
from typing import Tuple


def kerr_excision_window(M: float, a: float) -> Tuple[float, float]:
    """Return the admissible Cartesian-sphere excision radii ``(lo, hi)``.

    A Cartesian sphere with radius ``R_exc`` spans Boyer--Lindquist radii
    ``[sqrt(R_exc**2 - a**2), R_exc]``. Keeping it inside the trapped region
    therefore requires ``sqrt(r_minus**2 + a**2) < R_exc < r_plus``.
    """
    M = float(M)
    a = abs(float(a))
    if a > M:
        raise ValueError(f"spin |a|={a} exceeds M={M} (naked singularity)")
    root = sqrt(M * M - a * a)
    r_plus, r_minus = M + root, M - root
    lo = sqrt(r_minus**2 + a**2)
    if a == 0.0:
        lo = 0.0
    return float(lo), float(r_plus)
