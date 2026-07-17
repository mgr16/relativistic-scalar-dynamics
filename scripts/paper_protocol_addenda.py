#!/usr/bin/env python3
"""Emit the manuscript protocol values that live in canonical Python code.

The frozen F2 JSON files intentionally contain results, not every constructor
default used to create them.  This emitter imports the production modules
without invoking either ``main`` function, reads their configuration builders,
and writes a deterministic Phase-3 addendum.  It also records two explicitly
derived manuscript summaries from frozen JSON inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO / "docs/research/phase3/protocol_addenda.json"
PRODUCTION = REPO / "docs/research/phase2/production/production.json"
EXTERIOR = REPO / "docs/research/phase2/exterior/spectroscopy.json"
CONVERGENCE = REPO / "docs/research/phase1/convergence"
CONVERGENCE_P2 = REPO / "docs/research/phase1/convergence_p2"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _nearest_five(value: float) -> int:
    """Round a percentage to the nearest five percentage points."""
    return int(5 * round(float(value) / 5.0))


def _physical_window_orders(convergence_module) -> dict[str, float]:
    """Recompute the note's transit+ring self-convergence from versioned NPZ."""

    def load_window(path: Path) -> tuple[np.ndarray, np.ndarray]:
        with np.load(path) as data:
            times = data["ts"]
            signal = data["c10"]
        mask = times <= 30.0
        return times[mask], signal[mask]

    p1_scales = (2.0, 1.4, 1.0, 0.7)
    p2_scales = (1.4, 1.0, 0.7)
    p1 = [
        load_window(CONVERGENCE / f"waveform_lc{scale:g}.npz")
        for scale in p1_scales
    ]
    p2 = [
        load_window(CONVERGENCE_P2 / f"waveform_lc{scale:g}.npz")
        for scale in p2_scales
    ]
    p1_orders = convergence_module.self_convergence_order(p1, list(p1_scales))
    p2_orders = convergence_module.self_convergence_order(p2, list(p2_scales))
    return {
        "window_end": 30.0,
        "p1_coarse": p1_orders[0],
        "p1_fine": p1_orders[1],
        "p2_fine": p2_orders[0],
    }


def build_payload() -> dict[str, Any]:
    """Collect code-backed protocol values without starting any evolution."""
    scripts = str(REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)

    import exterior_spectroscopy as exterior  # noqa: WPS433
    import interior_production as interior  # noqa: WPS433
    import convergence_study as convergence  # noqa: WPS433

    production = _load_json(PRODUCTION)
    spectroscopy = _load_json(EXTERIOR)
    matrix = production["protocol"]["matrix"]
    mid_l0 = next(
        cell
        for cell in matrix
        if cell["pot"] == "linear" and cell["l"] == 0 and cell["lc"] == 0.04
    )
    interior_cfg = interior.base_cfg(
        production["protocol"]["t_end"],
        mid_l0["lc"],
        mid_l0["l"],
        mid_l0["lmax"],
        False,
    )

    frozen_r40_lcs = sorted(
        {
            float(record["lc"])
            for name, record in spectroscopy["runs"].items()
            if name.startswith("R40_")
        },
        reverse=True,
    )
    exterior_matrix = exterior.matrix(False, [], frozen_r40_lcs)
    small = [record for record in exterior_matrix if record["role"] == "rung"]
    large = [
        record for record in exterior_matrix if record["role"] == "domain_check"
    ]
    if not small or not large:
        raise RuntimeError("canonical exterior matrix is missing a domain family")

    l0_ratios = [
        float(production["discriminator"][f"l0_lc{lc:.3f}"]["l2_ratio"])
        for lc in (0.056, 0.040, 0.028)
    ]
    ladder_sources = [
        "docs/research/phase2/production/production.json::"
        f"/discriminator/l0_lc{lc:.3f}/l2_ratio"
        for lc in (0.056, 0.040, 0.028)
    ]

    fine_pair_key = {
        "linear": "linear_l0_lc0.040 vs linear_l0_lc0.028",
        "mexhat": "mexhat_l0_lc0.040 vs mexhat_l0_lc0.028",
    }
    mesh_component = 100.0 * max(
        float(
            production["ladder"][f"{potential}_l0"]["diff_rms_over_scale"][key]
        )
        for potential, key in fine_pair_key.items()
    )
    absolute_component = 100.0 * max(
        float(production["runs"][f"{potential}_l0_lc0.028"]
              ["dev_vs_1d_primary"]["median"])
        for potential in ("linear", "mexhat")
    )
    budget = sorted(
        (_nearest_five(mesh_component), _nearest_five(absolute_component))
    )

    return {
        "schema_version": 1,
        "generated_by": "scripts/paper_protocol_addenda.py",
        "interior": {
            "domain_radius": interior_cfg["mesh"]["R"],
            "excision_radius": interior_cfg["mesh"]["r_inner"],
            "outer_mesh_scale": interior_cfg["mesh"]["lc"],
            "degree": interior_cfg["solver"]["degree"],
            "cfl": interior_cfg["solver"]["cfl"],
            "extraction_count": interior_cfg["analysis"]["interior_profile"]
            ["n_radii"],
            "extraction_r_min": interior_cfg["analysis"]["interior_profile"]
            ["r_lo"],
            "extraction_r_max": interior_cfg["analysis"]["interior_profile"]
            ["r_hi"],
        },
        "exterior": {
            "small_domain_radius": small[0]["R"],
            "large_domain_radius": large[0]["R"],
            "small_sponge_width": small[0]["sponge_width"],
            "small_sponge_onset": small[0]["R"] - small[0]["sponge_width"],
            "large_sponge_width": large[0]["sponge_width"],
            "large_sponge_onset": large[0]["R"] - large[0]["sponge_width"],
            "end_time": small[0]["t_end"],
            "mesh_levels": [record["lc"] for record in small],
        },
        "derived": {
            "disc_l0_ladder_span_pts": 100.0 * (max(l0_ratios) - min(l0_ratios)),
            "disc_l0_ladder_sources": ladder_sources,
            "production_budget_low_percent": budget[0],
            "production_budget_high_percent": budget[1],
            "production_budget_inputs_percent": {
                "fine_pair_rms_max": mesh_component,
                "fine_absolute_deviation_max": absolute_component,
            },
            "production_budget_method": (
                "nearest-five envelope of the maximum fine paired-rung RMS "
                "difference and maximum fine absolute 3D-to-1D median deviation"
            ),
            "physical_window_convergence": _physical_window_orders(convergence),
        },
        "provenance": {
            "interior": "scripts/interior_production.py::base_cfg",
            "exterior": "scripts/exterior_spectroscopy.py::matrix",
            "derived": {
                "disc_l0_ladder_span_pts": ladder_sources,
                "production_budget": [
                    "docs/research/phase2/production/production.json::"
                    "/ladder/{linear_l0,mexhat_l0}/diff_rms_over_scale/"
                    "{fine-pair}",
                    "docs/research/phase2/production/production.json::"
                    "/runs/{linear,mexhat}_l0_lc0.028/"
                    "dev_vs_1d_primary/median",
                ],
                "physical_window_convergence": [
                    "docs/research/phase1/convergence/waveform_lc*.npz",
                    "docs/research/phase1/convergence_p2/waveform_lc*.npz",
                    "scripts/convergence_study.py::self_convergence_order",
                ],
            },
        },
    }


def render_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _atomic_write(path: Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return False
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rendered = render_json(build_payload())
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
            print(f"protocol addenda missing/stale: {args.output}", file=sys.stderr)
            return 1
        print("protocol addenda: verified; output up to date")
        return 0
    changed = _atomic_write(args.output, rendered)
    print(f"protocol addenda: {'updated' if changed else 'unchanged'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
