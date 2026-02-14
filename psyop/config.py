from __future__ import annotations

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
    "mesh": {"R": 30.0, "lc": 1.5, "mesh_type": "gmsh"},
    "solver": {
        "degree": 1,
        "cfl": 0.3,
        "potential_type": "quadratic",
        "potential_params": {"m_squared": 1.0},
        "ko_eps": 0.0,
        "bc_type": "characteristic",
        "outer_tag": 2,
    },
    "metric": {"type": "flat", "M": 1.0},
    "initial_conditions": {"type": "gaussian", "A": 0.01, "r0": 10.0, "w": 3.0, "v0": 1.0},
    "evolution": {"t_end": 50.0, "output_every": 10, "verbose": True},
    "output": {"dir": "results", "qnm_analysis": True, "diagnostics": True},
}


def load_config(path: str | None) -> Dict[str, Any]:
    if path is None:
        return dict(DEFAULT_CONFIG)
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
    return _deep_merge(DEFAULT_CONFIG, raw)


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    required = ["mesh", "metric", "solver", "initial_conditions", "evolution", "output"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")
    if float(cfg["mesh"]["R"]) <= 0:
        raise ValueError("mesh.R must be > 0")
    if float(cfg["mesh"].get("lc", 0)) <= 0:
        raise ValueError("mesh.lc must be > 0")
    degree = int(cfg["solver"].get("degree", 1))
    if degree < 1:
        raise ValueError("solver.degree must be >= 1")
    cfl = float(cfg["solver"].get("cfl", 0.0))
    if not (0.0 < cfl <= 1.0):
        raise ValueError("solver.cfl must be in (0,1]")
    bc_type = str(cfg["solver"].get("bc_type", "characteristic")).lower()
    if bc_type not in {"characteristic", "sommerfeld_spherical"}:
        raise ValueError("solver.bc_type must be characteristic or sommerfeld_spherical")
    return cfg
