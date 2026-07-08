"""
Validación cuantitativa de QNM contra Leaver.

Evoluciona un pulso escalar entrante con estructura angular Y_10 sobre
Schwarzschild en coordenadas Kerr-Schild (M=1, horizonte excisado), extrae
c_10(t) en una esfera de extracción y compara el modo dominante del ringdown
con el valor de referencia (Berti, Cardoso & Starinets 2009):

    M·ω (l=1, n=0) = 0.292936 − 0.097660 i

Es la prueba de integración física más fuerte del pipeline completo:
métrica + excisión + BC + esponja + extracción multipolar + Prony.
"""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

try:
    from mpi4py import MPI

    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False

pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_numpy,
    pytest.mark.requires_dolfinx,
    pytest.mark.requires_mesh,
    pytest.mark.skipif(not HAS_DOLFINX, reason="DOLFINx not available"),
]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Referencia Leaver: Mω = OMEGA_RE - OMEGA_IM·i (escalar l=1, n=0)
OMEGA_RE = 0.292936
OMEGA_IM = 0.097660


def _evolve_and_extract():
    import dolfinx.fem as fem

    from rsd.analysis.extraction import MultipoleExtractor
    from rsd.mesh.gmsh import INNER_BOUNDARY_TAG, build_ball_mesh, get_outer_tag
    from rsd.physics.metrics import KerrSchildCoeffs
    from rsd.solvers.first_order import FirstOrderKGSolver
    from rsd.utils.utils import compute_dt_cfl

    M = 1.0
    R, r_inner = 20.0, 1.0
    lc, lc_inner = 1.5, 0.4
    r0, w = 8.0, 2.0
    r_ext, t_end = 6.0, 45.0

    mesh, _, facet_tags = build_ball_mesh(
        R=R, lc=lc, comm=MPI.COMM_WORLD, r_inner=r_inner, lc_inner=lc_inner
    )
    bg = KerrSchildCoeffs(M=M, a=0.0)
    solver = FirstOrderKGSolver(
        mesh=mesh, domain_radius=R, degree=1, potential_type="zero",
        cfl_factor=0.25, sponge={"enabled": True, "width": 5.0, "strength": 1.0},
    )
    solver.set_background(*bg.build(mesh), rebuild=False)
    solver.enable_sommerfeld(facet_tags, get_outer_tag(facet_tags, default=2), rebuild=False)
    solver.enable_excision(facet_tags, inner_tag=INNER_BOUNDARY_TAG, rebuild=False)
    solver.rebuild_operators()

    # Cascarón gaussiano entrante con estructura Y_10 (z/r)
    A = 1e-3

    def phi_profile(x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        return A * (x[2] / np.maximum(r, 1e-12)) * np.exp(-((r - r0) ** 2) / w ** 2)

    def pi_profile(x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        rs = np.maximum(r, 1e-12)
        pert = A * (x[2] / rs) * np.exp(-((r - r0) ** 2) / w ** 2)
        dpert = pert * (-2.0 * (r - r0) / w ** 2)
        return dpert + pert / rs  # entrante (relación de onda esférica plana)

    phi0 = fem.Function(solver.V_scalar)
    phi0.interpolate(phi_profile)
    Pi0 = fem.Function(solver.V_scalar)
    Pi0.interpolate(pi_profile)
    solver.set_initial_conditions(phi0, Pi0)

    extractor = MultipoleExtractor(mesh, radius=r_ext, lmax=1)
    dt = compute_dt_cfl(mesh, cfl=0.25, c_max=bg.max_characteristic_speed(mesh))

    ts, c10 = [], []
    t, step = 0.0, 0
    while t < t_end - 1e-12:
        step_dt = min(dt, t_end - t)
        solver.ssp_rk3_step(step_dt)
        t += step_dt
        step += 1
        if step % 4 == 0:
            phi_f, _ = solver.get_fields()
            ts.append(t)
            c10.append(extractor.extract(phi_f)[(1, 0)])
    return np.array(ts), np.array(c10)


def test_schwarzschild_l1_qnm_matches_leaver():
    from rsd.analysis.qnm import estimate_qnm_prony

    ts, c10 = _evolve_and_extract()
    assert np.all(np.isfinite(c10)), "extracted signal contains NaN/inf"
    assert np.max(np.abs(c10)) > 0, "extracted signal is identically zero"

    # Ventana de ringdown calibrada: tras el paso del pulso y antes de que
    # la cola lenta domine (experimento: docs/validation/summary.md)
    mask = (ts >= 12.0) & (ts <= 40.0)
    assert mask.sum() >= 32, "ringdown window too short"
    dt_sample = ts[mask][1] - ts[mask][0]

    # estimate_qnm_prony ordena por amplitud; seleccionar el modo
    # OSCILATORIO dominante (el transitorio lento tiene freq ~ 0)
    pairs = estimate_qnm_prony(c10[mask], dt_sample, modes=4, svd_rank=4)
    assert pairs, "Prony extracted no modes from the ringdown"
    oscillatory = [
        (f, d) for (f, d) in pairs if 2.0 * np.pi * abs(f) > 0.1 and d > 0
    ]
    assert oscillatory, f"no oscillatory mode found in {pairs}"

    freq, decay = oscillatory[0]
    omega_re = 2.0 * np.pi * abs(freq)
    omega_im = decay

    # Tolerancias calibradas en malla de CI (lc=1.5/0.4): el experimento dio
    # Mω ≈ 0.2505 − 0.0709i (error sistemático de discretización P1, ~15%/27%).
    #
    # Honestidad sobre el poder discriminante: a esta resolución el error de
    # discretización DOMINA sobre el efecto del término αKΠ (un control con
    # K=0 dio 0.2877 − 0.1194i, también dentro de banda). El test valida la
    # cadena completa métrica+excisión+BC+extracción+Prony y detecta errores
    # groseros (signos, excisión rota, inestabilidad → no hay modo
    # oscilatorio), pero discriminar K requiere malla más fina que la de CI.
    assert omega_re == pytest.approx(OMEGA_RE, rel=0.30), (
        f"Re(Mω) = {omega_re:.4f}, Leaver = {OMEGA_RE:.4f}"
    )
    assert omega_im == pytest.approx(OMEGA_IM, rel=0.50), (
        f"-Im(Mω) = {omega_im:.4f}, Leaver = {OMEGA_IM:.4f}"
    )
