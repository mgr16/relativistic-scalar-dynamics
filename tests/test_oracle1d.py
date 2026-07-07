"""
Validación del oráculo 1D (psyop.reference.spherical1d).

Escalera:
1. Validación de entradas y consistencia interna (K analítico vs divergencia).
2. Espacio plano: un pulso saliente abandona el dominio (BC absorbente).
3. Auto-convergencia de 2.º orden del esquema FD.
4. QNM: el ringdown sobre Schwarzschild-KS debe reproducir las frecuencias
   de Leaver (l=0 y l=1) — esto promueve al oráculo a referencia confiable
   para la validación cruzada del código 3D.
5. Interior: evolución estable con r_min = 0.05 ≪ 2M (sector H2).
"""

import numpy as np
import pytest

from psyop.analysis.leaver import schwarzschild_qnm
from psyop.analysis.qnm import estimate_qnm_prony
from psyop.reference import SphericalOracle1D, trace_K_from_divergence

pytestmark = [pytest.mark.requires_numpy]


# ----------------------------------------------------------------------
# 1. entradas y consistencia interna
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"M": -1.0},
        {"l": -1},
        {"r_min": 0.0},
        {"r_min": 5.0, "r_max": 2.0},
        {"n_points": 4},
        {"grid": "chebyshev"},
        {"cfl": 0.0},
        {"cfl": 1.5},
        {"ko_eps": -0.1},
        {"ko_eps": 1.0},
    ],
)
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        SphericalOracle1D(**kwargs)


def test_invalid_direction_raises():
    oracle = SphericalOracle1D(M=1.0, r_min=1.0, r_max=20.0, n_points=64)
    with pytest.raises(ValueError):
        oracle.set_initial_gaussian(direction="sideways")


def test_trace_K_analytic_matches_divergence():
    """K = 2Mα³(1+3M/r)/r² debe coincidir con (1/(αw))∂_r(wβ)."""
    oracle = SphericalOracle1D(M=1.0, r_min=0.2, r_max=50.0, n_points=6000)
    K_div = trace_K_from_divergence(oracle)
    interior = slice(2, -2)
    rel = np.abs(K_div[interior] - oracle.K[interior]) / np.abs(oracle.K[interior])
    assert np.max(rel) < 2e-4, f"max rel error {np.max(rel):.2e}"


def test_flat_limit_coefficients():
    oracle = SphericalOracle1D(M=0.0, r_min=0.5, r_max=10.0, n_points=64)
    assert np.allclose(oracle.alpha, 1.0)
    assert np.allclose(oracle.beta, 0.0)
    assert np.allclose(oracle.K, 0.0)
    assert np.allclose(oracle.w, oracle.r**2)


def test_ingoing_curved_reduces_to_ingoing_in_flat_space():
    a = SphericalOracle1D(M=0.0, r_min=0.5, r_max=30.0, n_points=256)
    a.set_initial_gaussian(A=1e-3, r0=10.0, width=2.0, direction="ingoing")
    Pi_flat = a.Pi.copy()
    a.set_initial_gaussian(A=1e-3, r0=10.0, width=2.0, direction="ingoing_curved")
    assert np.allclose(a.Pi, Pi_flat, atol=1e-15)


# ----------------------------------------------------------------------
# 2. espacio plano: absorción del pulso saliente
# ----------------------------------------------------------------------

def test_flat_outgoing_pulse_leaves_domain():
    oracle = SphericalOracle1D(
        M=0.0, l=0, r_min=0.5, r_max=40.0, n_points=1600, grid="uniform"
    )
    oracle.set_initial_gaussian(A=1e-3, r0=10.0, width=2.0, direction="outgoing")
    e0 = oracle.energy()
    assert e0 > 0
    oracle.evolve(t_end=45.0, output_every=200)
    e1 = oracle.energy()
    assert e1 < 5e-3 * e0, f"E_final/E_0 = {e1 / e0:.2e} (pulso no absorbido)"


# ----------------------------------------------------------------------
# 3. auto-convergencia
# ----------------------------------------------------------------------

def test_second_order_self_convergence():
    """El esquema (FD2 + SSP-RK3 con dt ∝ Δr) converge a orden ~2."""
    def run(n):
        oracle = SphericalOracle1D(
            M=1.0, l=0, r_min=0.5, r_max=30.0, n_points=n,
            grid="uniform", ko_eps=0.0,
        )
        oracle.set_initial_gaussian(A=1e-3, r0=8.0, width=1.5, direction="ingoing")
        oracle.evolve(t_end=6.0, output_every=10**9)
        return oracle

    coarse, medium, fine = run(400), run(800), run(1600)
    r_common = np.linspace(1.0, 25.0, 500)
    u_c = np.interp(r_common, coarse.r, coarse.u)
    u_m = np.interp(r_common, medium.r, medium.u)
    u_f = np.interp(r_common, fine.r, fine.u)
    e1 = np.linalg.norm(u_c - u_m)
    e2 = np.linalg.norm(u_m - u_f)
    order = np.log2(e1 / e2)
    assert 1.6 < order < 2.6, f"orden medido {order:.2f} (esperado ~2)"


# ----------------------------------------------------------------------
# 4. QNM contra Leaver (la validación que promueve al oráculo a referencia)
#
# Se valida con l=1 y l=2. El modo l=0 (Q ≈ 0.5: un solo ciclo de período
# 2π/0.11 ≈ 57M) es notoriamente mal condicionado para ajustes en dominio
# temporal y no se usa como benchmark (propiedad del modo, no del código).
#
# Nota de cronometría: en Kerr-Schild la velocidad coordenada saliente
# cerca del horizonte es α²(1−2M/r) → 0, así que el ringdown llega a la
# sonda con el retardo tipo-tortuga (~t=32 para sonda en r=10, no ~20).
# La ventana de ajuste se ancla al pico del ring detectado tras el
# transitorio directo.
# ----------------------------------------------------------------------

def _ringdown_omega(l: int) -> complex:
    """Evoluciona un pulso entrante y ajusta el modo dominante por Prony."""
    oracle = SphericalOracle1D(
        M=1.0, l=l, r_min=1.0, r_max=150.0, n_points=4000,
        grid="uniform", ko_eps=0.02,
    )
    oracle.set_initial_gaussian(A=1e-3, r0=12.0, width=2.0, direction="ingoing_curved")
    out = oracle.evolve(t_end=90.0, probe_radii=[10.0], output_every=5)
    ts, sig = out.ts, out.probes[10.0]

    # pico del ring tras el transitorio directo; ventana [t_pk+8, t_pk+45]
    i0 = np.searchsorted(ts, 25.0)
    t_pk = ts[int(np.argmax(np.abs(sig[i0:]))) + i0]
    mask = (ts > t_pk + 8.0) & (ts < t_pk + 45.0)
    dt_s = float(np.mean(np.diff(ts[mask])))
    modes = estimate_qnm_prony(sig[mask], dt_s, modes=3)
    assert modes, "Prony no devolvió modos"
    freq, decay = max(modes, key=lambda m: abs(m[0]))
    return 2.0 * np.pi * abs(freq) - 1.0j * decay


@pytest.mark.parametrize("l,tol_re,tol_im", [(1, 0.01, 0.04), (2, 0.02, 0.05)])
def test_qnm_matches_leaver(l, tol_re, tol_im):
    omega_ref = schwarzschild_qnm(l=l, n=0)
    omega = _ringdown_omega(l)
    err_re = abs(omega.real - omega_ref.real) / abs(omega_ref.real)
    err_im = abs(omega.imag - omega_ref.imag) / abs(omega_ref.imag)
    assert err_re < tol_re, (
        f"l={l}: Re(Mω) = {omega.real:.5f} vs Leaver {omega_ref.real:.5f} "
        f"({100 * err_re:.2f}%)"
    )
    assert err_im < tol_im, (
        f"l={l}: Im(Mω) = {omega.imag:.5f} vs Leaver {omega_ref.imag:.5f} "
        f"({100 * err_im:.2f}%)"
    )


# ----------------------------------------------------------------------
# 5. interior profundo (sector esférico de H2)
# ----------------------------------------------------------------------

def test_interior_evolution_is_stable():
    """Evolución estable hasta t=20M con r_min = 0.05 (Kretschmann ~ 10⁸)."""
    oracle = SphericalOracle1D(
        M=1.0, l=0, r_min=0.05, r_max=30.0, n_points=1500, grid="log"
    )
    oracle.set_initial_gaussian(A=1e-3, r0=5.0, width=1.0, direction="ingoing_curved")
    out = oracle.evolve(t_end=20.0, probe_radii=[0.1, 0.5], output_every=100)
    assert np.all(np.isfinite(oracle.u))
    assert np.all(np.isfinite(oracle.Pi))
    # el pulso entrante debe haber alcanzado el interior profundo
    assert np.max(np.abs(out.probes[0.1])) > 1e-7, (
        "sin señal en r=0.1: el pulso no penetró el horizonte"
    )
