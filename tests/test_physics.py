import json
import os
import pytest
import sys
from pathlib import Path
np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import dolfinx.fem as fem
    from mpi4py import MPI
    HAS_DOLFINX = True
except Exception:
    HAS_DOLFINX = False

pytestmark = [
    pytest.mark.requires_numpy,
    pytest.mark.requires_dolfinx,
    pytest.mark.skipif(not HAS_DOLFINX, reason="DOLFINx not available"),
]

if HAS_DOLFINX:
    from psyop.physics.initial_conditions import GaussianBump
    from psyop.physics.metrics import make_background
    from psyop.solvers.first_order import FirstOrderKGSolver
    from psyop.mesh.gmsh import build_ball_mesh, get_outer_tag


def _eval_at_origin(func):
    coords = func.function_space.tabulate_dof_coordinates()
    idx = int(np.argmin(np.linalg.norm(coords, axis=1)))
    return float(func.x.array[idx])


def run_solver(cfg):
    comm = MPI.COMM_WORLD

    R = float(cfg["mesh"]["R"]) if "mesh" in cfg else 10.0
    lc = float(cfg["mesh"].get("lc", 1.0)) if "mesh" in cfg else 1.0

    mesh, cell_tags, facet_tags = build_ball_mesh(R=R, lc=lc, comm=comm)

    bg = make_background(cfg.get("metric", {"type": "flat"}))
    metric_coeffs = bg.build(mesh)

    solver = FirstOrderKGSolver(
        mesh=mesh,
        domain_radius=R,
        degree=cfg["solver"]["degree"],
        potential_type=cfg["solver"]["potential_type"],
        potential_params=cfg["solver"].get("potential_params", {}),
        cfl_factor=cfg["solver"]["cfl"],
        cfg=cfg,
    )

    solver.set_background(*metric_coeffs)

    # Respetar el flag de configuración (legacy "sommerfeld" o canónico
    # "enable_sommerfeld"): antes se habilitaba siempre y el test A/B
    # comparaba dos corridas idénticas
    sommerfeld_enabled = bool(
        cfg["solver"].get("enable_sommerfeld", cfg["solver"].get("sommerfeld", True))
    )
    if sommerfeld_enabled and facet_tags is not None:
        outer_tag = get_outer_tag(facet_tags, default=2)
        solver.enable_sommerfeld(facet_tags, outer_tag)

    ic_cfg = cfg.get("initial_conditions", {})
    ic = GaussianBump(
        mesh,
        V=solver.V_scalar,
        A=float(ic_cfg.get("A", 1e-3)),
        r0=float(ic_cfg.get("r0", R / 3.0)),
        w=float(ic_cfg.get("w", 1.5)),
        v0=float(ic_cfg.get("v0", 0.0)),
    )
    solver.set_initial_conditions(ic.get_function())

    E0 = solver.energy()

    t, step = 0.0, 0
    dt = solver.compute_adaptive_dt()
    t_end = float(cfg["evolution"]["t_end"]) if "evolution" in cfg else 1.0
    output_every = int(cfg["evolution"].get("output_every", 10)) if "evolution" in cfg else 10

    amp_center = _eval_at_origin(solver.get_fields()[0])

    while t < t_end - 1e-12:
        solver.ssp_rk3_step(dt)
        t += dt
        step += 1
        if step % 10 == 0:
            dt = solver.compute_adaptive_dt()
        if step % output_every == 0:
            pass

    E_last = solver.energy()
    amp_center = abs(_eval_at_origin(solver.get_fields()[0]))

    return {"E0": E0, "E_last": E_last, "amp_center": amp_center}


def test_energy_conservation(tmp_path):
    cfg = {
        "schema_version": 1,
        "mesh": {"type": "gmsh", "R": 10.0, "lc": 1.0},
        "metric": {"type": "flat"},
        "solver": {
            "degree": 1,
            # CFL bajo: el drift de energía está dominado por el error
            # temporal (3.er orden); con 0.15 queda ~0.2% vs tolerancia 1%
            "cfl": 0.15,
            "potential_type": "higgs",
            "potential_params": {"m_squared": 1.0, "lambda_coupling": 0.0},
            # Sin BC absorbente: un test de conservación debe medir solo el
            # error numérico, no la absorción (real) del borde
            "sommerfeld": False,
        },
        "initial_conditions": {
            "type": "gaussian",
            "A": 1e-3,
            "r0": 4.0,
            "w": 1.5,
            "v0": 0.0,
        },
        "evolution": {"t_end": 3.0, "output_every": 2, "checkpoint_every": 100},
        "output": {"dir": str(tmp_path), "qnm_analysis": False, "save_series": False},
    }
    res = run_solver(cfg)
    E0, E1 = res["E0"], res["E_last"]
    assert np.isfinite(E0) and np.isfinite(E1)
    drift = abs(E1 - E0) / max(E0, 1e-12)
    assert drift < 1e-2


def test_reflection_reduction(tmp_path):
    # Campo sin masa (sin dispersión) y t_end suficiente para que el pulso
    # alcance el borde: con BC absorbente la energía debe caer claramente
    # frente al caso reflectante.
    cfg_base = {
        "schema_version": 1,
        "mesh": {"type": "gmsh", "R": 5.0, "lc": 1.0},
        "metric": {"type": "flat"},
        "solver": {
            "degree": 1,
            "cfl": 0.3,
            "potential_type": "zero",
        },
        "initial_conditions": {
            "type": "gaussian",
            "A": 5e-3,
            "r0": 1.5,
            "w": 0.8,
            "v0": 0.0,
        },
        "evolution": {"t_end": 9.0, "output_every": 2, "checkpoint_every": 100},
        "output": {"dir": str(tmp_path), "qnm_analysis": False, "save_series": False},
    }
    import copy
    cfg_no = copy.deepcopy(cfg_base)
    cfg_yes = copy.deepcopy(cfg_base)
    cfg_no["solver"]["sommerfeld"] = False
    cfg_yes["solver"]["sommerfeld"] = True

    res_no = run_solver(cfg_no)
    res_yes = run_solver(cfg_yes)
    # La comparación de energías es robusta (la amplitud puntual depende
    # del instante exacto de muestreo)
    assert res_yes["E_last"] < 0.5 * res_no["E_last"]


def _evolve_direction(direction: str, t_end: float = 8.0):
    from psyop.mesh.gmsh import build_ball_mesh, get_outer_tag
    from psyop.solvers.first_order import FirstOrderKGSolver
    from psyop.utils.utils import compute_dt_cfl

    mesh, _, facet_tags = build_ball_mesh(R=10.0, lc=1.2, comm=MPI.COMM_WORLD)
    solver = FirstOrderKGSolver(
        mesh=mesh, domain_radius=10.0, degree=1, potential_type="zero", cfl_factor=0.25
    )
    solver.enable_sommerfeld(facet_tags, get_outer_tag(facet_tags, default=2))
    ic = GaussianBump(
        mesh, V=solver.V_scalar, A=1e-2, r0=5.0, w=1.0, v0=0.0, direction=direction
    )
    solver.set_initial_conditions(ic.get_function(), ic.get_momentum())
    E0 = solver.energy()
    dt = compute_dt_cfl(mesh, cfl=0.25)
    t = 0.0
    while t < t_end - 1e-12:
        step_dt = min(dt, t_end - t)
        solver.ssp_rk3_step(step_dt)
        t += step_dt
    return E0, solver.energy()


def test_outgoing_momentum_empties_domain_faster():
    """
    Π consistente con onda saliente: el pulso sale entero por el borde,
    mientras que con Π=0 la mitad entrante sigue dentro del dominio a t=8
    (rebota por el origen y tarda ~r0+R en salir).
    """
    E0_static, Ef_static = _evolve_direction("static")
    E0_out, Ef_out = _evolve_direction("outgoing")

    assert E0_static > 0 and E0_out > 0
    ratio_static = Ef_static / E0_static
    ratio_out = Ef_out / E0_out
    # Calibrado en malla lc=1.2: static ~0.46, outgoing ~0.18
    assert ratio_out < 0.30, f"outgoing pulse should leave the domain, kept {ratio_out:.2%}"
    assert ratio_out < 0.6 * ratio_static, (
        f"outgoing ({ratio_out:.2%}) should retain much less energy than "
        f"static ({ratio_static:.2%})"
    )


def test_energy_balance_closes_with_boundary_flux():
    """
    E(t) + ∫F dt ≈ E(0): el flujo reportado debe dar cuenta de la energía
    que sale por la BC absorbente (residuo ~h²; calibrado: ~6% en lc=1.0).
    """
    from psyop.mesh.gmsh import build_ball_mesh, get_outer_tag
    from psyop.solvers.first_order import FirstOrderKGSolver
    from psyop.utils.utils import compute_dt_cfl

    mesh, _, facet_tags = build_ball_mesh(R=10.0, lc=1.0, comm=MPI.COMM_WORLD)
    solver = FirstOrderKGSolver(
        mesh=mesh, domain_radius=10.0, degree=1, potential_type="zero", cfl_factor=0.2
    )
    solver.enable_sommerfeld(facet_tags, get_outer_tag(facet_tags, default=2))
    ic = GaussianBump(
        mesh, V=solver.V_scalar, A=1e-2, r0=5.0, w=1.0, v0=0.0, direction="outgoing"
    )
    solver.set_initial_conditions(ic.get_function(), ic.get_momentum())

    E0 = solver.energy()
    dt = compute_dt_cfl(mesh, cfl=0.2)
    t, cum_flux, flux_prev = 0.0, 0.0, solver.boundary_flux()
    while t < 8.0 - 1e-12:
        step_dt = min(dt, 8.0 - t)
        solver.ssp_rk3_step(step_dt)
        t += step_dt
        flux = solver.boundary_flux()
        cum_flux += 0.5 * (flux + flux_prev) * step_dt
        flux_prev = flux

    residual = abs(solver.energy() + cum_flux - E0) / E0
    assert cum_flux > 0, "integrated outgoing flux must be positive"
    assert residual < 0.10, f"energy balance residual too large: {residual:.2%}"


def test_gaussian_bump_direction_is_validated():
    mesh, _, _ = build_ball_mesh(R=4.0, lc=2.0, comm=MPI.COMM_WORLD)
    with pytest.raises(ValueError, match="direction"):
        GaussianBump(mesh, A=0.01, r0=1.5, w=0.8, v0=0.0, direction="sideways")


def test_fourth_order_filter_damps_smooth_modes_less():
    """
    Con el mismo ε, el filtro biarmónico (ko_order=4) debe disipar un pulso
    suave mucho menos que el laplaciano (ko_order=2), sin inestabilidad.
    Calibrado: ~28% (orden 2) vs ~4% (orden 4) en lc=1.0.
    """
    def energy_loss(ko_order):
        mesh, _, _ = build_ball_mesh(R=6.0, lc=1.0, comm=MPI.COMM_WORLD)
        solver = FirstOrderKGSolver(
            mesh=mesh, domain_radius=6.0, degree=1, potential_type="zero",
            cfl_factor=0.2, ko_eps=0.05, ko_order=ko_order,
        )
        ic = GaussianBump(mesh, V=solver.V_scalar, A=1e-2, r0=2.5, w=1.2, v0=0.0)
        solver.set_initial_conditions(ic.get_function())
        E0 = solver.energy()
        for _ in range(60):
            solver.ssp_rk3_step(0.02)
        return (E0 - solver.energy()) / E0

    loss2 = energy_loss(2)
    loss4 = energy_loss(4)
    assert np.isfinite(loss4), "order-4 filter must be stable"
    assert 0.0 <= loss4 < 1.0, f"order-4 filter unstable or wrong: loss={loss4:.2%}"
    assert loss4 < 0.5 * loss2, (
        f"order-4 should damp smooth modes much less: {loss4:.2%} vs {loss2:.2%}"
    )


def _run_massive_with_sponge(sponge_cfg, t_end=20.0):
    from psyop.mesh.gmsh import build_ball_mesh, get_outer_tag
    from psyop.solvers.first_order import FirstOrderKGSolver
    from psyop.utils.utils import compute_dt_cfl

    mesh, _, facet_tags = build_ball_mesh(R=10.0, lc=1.2, comm=MPI.COMM_WORLD)
    solver = FirstOrderKGSolver(
        mesh=mesh, domain_radius=10.0, degree=1,
        potential_type="quadratic", potential_params={"m_squared": 1.0},
        cfl_factor=0.25, sponge=sponge_cfg,
    )
    solver.enable_sommerfeld(facet_tags, get_outer_tag(facet_tags, default=2))
    ic = GaussianBump(
        mesh, V=solver.V_scalar, A=1e-2, r0=4.0, w=1.2, v0=0.0, direction="outgoing"
    )
    solver.set_initial_conditions(ic.get_function(), ic.get_momentum())
    E0 = solver.energy()
    dt = compute_dt_cfl(mesh, cfl=0.25)
    t = 0.0
    while t < t_end - 1e-12:
        step_dt = min(dt, t_end - t)
        solver.ssp_rk3_step(step_dt)
        t += step_dt
    return solver.energy() / E0


def test_sponge_layer_absorbs_massive_field_tails():
    """
    Campo masivo (dispersivo): la BC característica asume v=c y refleja las
    colas lentas; la esponja debe reducir la energía residual.
    Calibrado: ratio con/sin esponja ~0.78 en lc=1.2, t=20.
    """
    ratio_off = _run_massive_with_sponge({"enabled": False})
    ratio_on = _run_massive_with_sponge({"enabled": True, "width": 3.0, "strength": 2.0})
    assert ratio_on < 0.9 * ratio_off, (
        f"sponge should absorb dispersive tails: with={ratio_on:.4f} "
        f"without={ratio_off:.4f}"
    )


def test_sponge_config_is_validated():
    from psyop.mesh.gmsh import build_ball_mesh
    from psyop.solvers.first_order import FirstOrderKGSolver

    mesh, _, _ = build_ball_mesh(R=4.0, lc=2.0, comm=MPI.COMM_WORLD)
    with pytest.raises(ValueError, match="sponge.width"):
        FirstOrderKGSolver(
            mesh=mesh, domain_radius=4.0, degree=1, potential_type="zero",
            cfl_factor=0.2, sponge={"enabled": True, "width": 10.0},
        )


def test_ko_order_is_validated():
    mesh, _, _ = build_ball_mesh(R=4.0, lc=2.0, comm=MPI.COMM_WORLD)
    with pytest.raises(ValueError, match="ko_order"):
        FirstOrderKGSolver(
            mesh=mesh, domain_radius=4.0, degree=1, potential_type="zero",
            cfl_factor=0.2, ko_eps=0.01, ko_order=3,
        )
