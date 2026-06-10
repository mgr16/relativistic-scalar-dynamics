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
            "cfl": 0.3,
            "potential_type": "higgs",
            "potential_params": {"m_squared": 1.0, "lambda_coupling": 0.0},
            "sponge": {"enabled": False},
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
