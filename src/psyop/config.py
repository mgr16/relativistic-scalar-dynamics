from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


DEFAULT_CONFIG: Dict[str, Any] = {
    "schema_version": 1,
    "mesh": {"type": "gmsh", "R": 30.0, "lc": 1.5, "r_inner": 0.0},
    "solver": {
        "degree": 1,
        "cfl": 0.3,
        "potential_type": "quadratic",
        "potential_params": {"m_squared": 1.0},
        "ko_eps": 0.0,
        "ko_order": 2,
        "sponge": {"enabled": False, "width": 0.0, "strength": 1.0},
        "bc_type": "characteristic",
        "outer_tag": 2,
        "enable_sommerfeld": True,
    },
    "metric": {"type": "flat", "M": 1.0},
    "initial_conditions": {"type": "gaussian", "A": 0.01, "r0": 10.0, "w": 3.0, "v0": 1.0},
    "evolution": {"t_end": 50.0, "output_every": 10, "verbose": True},
    "output": {"dir": "results", "qnm_analysis": True, "diagnostics": True, "save_series": True},
}


def load_config(path: str | None) -> Dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    cfg_path = Path(path)
    suffix = cfg_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:  # pragma: no cover
            raise ValueError("YAML config requires PyYAML installed") from exc
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    return _normalize_config(_deep_merge(DEFAULT_CONFIG, _normalize_config(raw)))


def _normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Accept legacy aliases while keeping one canonical runtime shape."""
    normalized = deepcopy(cfg)

    mesh = normalized.setdefault("mesh", {})
    if "type" not in mesh and "mesh_type" in mesh:
        mesh["type"] = mesh["mesh_type"]

    solver = normalized.setdefault("solver", {})
    if "enable_sommerfeld" not in solver and "sommerfeld" in solver:
        solver["enable_sommerfeld"] = bool(solver["sommerfeld"])
    solver.setdefault("enable_sommerfeld", True)

    output = normalized.setdefault("output", {})
    if "dir" not in output and "results_dir" in output:
        output["dir"] = output["results_dir"]
    output.setdefault("save_series", True)

    return normalized


# R/lc bound above which a 3D ball mesh becomes computationally intractable
MAX_RESOLUTION_RATIO = 1000.0

VALID_METRIC_TYPES = {"flat", "schwarzschild", "kerr"}


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _normalize_config(cfg)
    required = ["mesh", "metric", "solver", "initial_conditions", "evolution", "output"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")
    mesh_type = str(cfg["mesh"].get("type", "gmsh")).lower()
    if mesh_type not in {"gmsh", "ball"}:
        raise ValueError("mesh.type must be gmsh or ball")
    R = float(cfg["mesh"]["R"])
    if R <= 0:
        raise ValueError("mesh.R must be > 0")
    lc = float(cfg["mesh"].get("lc", 0))
    if lc <= 0:
        raise ValueError("mesh.lc must be > 0")
    if R / lc > MAX_RESOLUTION_RATIO:
        raise ValueError(
            f"mesh.R/mesh.lc = {R / lc:.0f} exceeds {MAX_RESOLUTION_RATIO:.0f}; "
            "a 3D ball at this resolution is computationally intractable. "
            "Increase mesh.lc or decrease mesh.R."
        )
    r_inner = float(cfg["mesh"].get("r_inner", 0.0) or 0.0)
    if r_inner < 0:
        raise ValueError("mesh.r_inner must be >= 0 (0 disables excision)")
    if r_inner >= R:
        raise ValueError("mesh.r_inner must be smaller than mesh.R")
    lc_inner = cfg["mesh"].get("lc_inner")
    if lc_inner is not None:
        lc_inner = float(lc_inner)
        if not (0 < lc_inner <= lc):
            raise ValueError("mesh.lc_inner must satisfy 0 < lc_inner <= mesh.lc")
        if R / lc_inner > MAX_RESOLUTION_RATIO:
            raise ValueError(
                f"mesh.R/mesh.lc_inner = {R / lc_inner:.0f} exceeds "
                f"{MAX_RESOLUTION_RATIO:.0f}; increase mesh.lc_inner."
            )
    if int(cfg["mesh"].get("geom_order", 1)) not in (1, 2):
        raise ValueError("mesh.geom_order must be 1 (flat facets) or 2 (curved cells)")

    metric_type = str(cfg["metric"].get("type", "flat")).lower()
    if metric_type not in VALID_METRIC_TYPES:
        raise ValueError(f"metric.type must be one of {sorted(VALID_METRIC_TYPES)}")
    M = float(cfg["metric"].get("M", 1.0))
    if metric_type != "flat":
        if M <= 0:
            raise ValueError("metric.M must be > 0 for black hole backgrounds")
        if r_inner <= 0:
            raise ValueError(
                f"metric.type={metric_type!r} requires an excised inner boundary: "
                "set mesh.r_inner > 0. Suggested values: r_inner ~ M for kerr "
                "(Kerr-Schild, horizon-penetrating) and r_inner ~ M/2 for "
                "schwarzschild (isotropic coordinates, horizon at r = M/2)."
            )
    if metric_type == "kerr":
        a = float(cfg["metric"].get("a", 0.0))
        if abs(a) > M:
            raise ValueError("metric.a must satisfy |a| <= M (no naked singularities)")
        # Ventana de outflow del borde excisado (docs/math/excision_window.md):
        # la esfera cartesiana debe caber en la región atrapada r₋ < r < r₊.
        from psyop.physics.metrics import kerr_excision_window

        lo, hi = kerr_excision_window(M, a)
        if lo >= hi:
            raise ValueError(
                f"spin a={a} too high for a Cartesian-sphere excision: the "
                f"trapped-region window is empty (requires |a| < 0.9718 M). "
                "A spheroidal excision surface (r = const) is needed."
            )
        if not (lo < r_inner < hi):
            raise ValueError(
                f"mesh.r_inner = {r_inner} is outside the admissible excision "
                f"window ({lo:.4f}, {hi:.4f}) for a={a}, M={M}: parts of the "
                "excision sphere would leave the trapped region and the inner "
                "'do-nothing' boundary becomes inconsistent (characteristics "
                "re-enter). See docs/math/excision_window.md."
            )

    degree = int(cfg["solver"].get("degree", 1))
    if not (1 <= degree <= 5):
        raise ValueError("solver.degree must be in [1, 5]")
    ko_eps = float(cfg["solver"].get("ko_eps", 0.0))
    if ko_eps < 0:
        raise ValueError("solver.ko_eps must be >= 0")
    if int(cfg["solver"].get("ko_order", 2)) not in (2, 4):
        raise ValueError("solver.ko_order must be 2 or 4")
    if int(cfg["solver"].get("quadrature_degree", 2 * degree + 2)) < 1:
        raise ValueError("solver.quadrature_degree must be >= 1")
    sponge = cfg["solver"].get("sponge") or {}
    if sponge.get("enabled", False):
        width = float(sponge.get("width", 0.0))
        if not (0 < width < R):
            raise ValueError("solver.sponge.width must be in (0, mesh.R) when enabled")
        if float(sponge.get("strength", 1.0)) <= 0:
            raise ValueError("solver.sponge.strength must be > 0 when enabled")
    cfl = float(cfg["solver"].get("cfl", 0.0))
    if not (0.0 < cfl <= 1.0):
        raise ValueError("solver.cfl must be in (0,1]")
    bc_type = str(cfg["solver"].get("bc_type", "characteristic")).lower()
    if bc_type not in {"characteristic", "sommerfeld_spherical"}:
        raise ValueError("solver.bc_type must be characteristic or sommerfeld_spherical")
    if int(cfg["evolution"].get("output_every", 1)) < 1:
        raise ValueError("evolution.output_every must be >= 1")
    if float(cfg["evolution"].get("t_end", 0.0)) <= 0:
        raise ValueError("evolution.t_end must be > 0")
    ic_type = str(cfg["initial_conditions"].get("type", "gaussian")).lower()
    if ic_type != "gaussian":
        raise ValueError("initial_conditions.type must be gaussian")
    ic_direction = str(cfg["initial_conditions"].get("direction", "static")).lower()
    if ic_direction not in {"static", "ingoing", "outgoing", "ingoing_curved"}:
        raise ValueError(
            "initial_conditions.direction must be static, ingoing, outgoing "
            "or ingoing_curved"
        )
    extraction = (cfg.get("analysis", {}) or {}).get("extraction") or {}
    if extraction.get("enabled", False):
        ext_radius = float(extraction.get("radius", 0.6 * R))
        if not (0 < ext_radius < R):
            raise ValueError("analysis.extraction.radius must be in (0, mesh.R)")
        if int(extraction.get("lmax", 2)) < 0:
            raise ValueError("analysis.extraction.lmax must be >= 0")
    physical_units = cfg["output"].get("physical_units") or {}
    if physical_units and float(physical_units.get("M_solar", 0.0)) <= 0:
        raise ValueError("output.physical_units.M_solar must be > 0")
    return cfg
