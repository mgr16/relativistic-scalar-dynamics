"""Regressions for the code-to-JSON numerical-protocol emitter."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts/paper_protocol_addenda.py"
SPEC = importlib.util.spec_from_file_location("paper_protocol_addenda", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
addenda = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = addenda
SPEC.loader.exec_module(addenda)


def test_payload_matches_imported_production_builders():
    scripts = str(REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import exterior_spectroscopy as exterior
    import interior_production as interior

    payload = addenda.build_payload()
    production = json.loads(addenda.PRODUCTION.read_text(encoding="utf-8"))
    matrix = production["protocol"]["matrix"]
    mid = [
        cell for cell in matrix if cell["pot"] == "linear" and cell["l"] == 0
    ][1]
    cfg = interior.base_cfg(
        production["protocol"]["t_end"],
        mid["lc"],
        mid["l"],
        mid["lmax"],
        False,
    )

    assert payload["interior"] == {
        "domain_radius": cfg["mesh"]["R"],
        "excision_radius": cfg["mesh"]["r_inner"],
        "outer_mesh_scale": cfg["mesh"]["lc"],
        "degree": cfg["solver"]["degree"],
        "cfl": cfg["solver"]["cfl"],
        "extraction_count": cfg["analysis"]["interior_profile"]["n_radii"],
        "extraction_r_min": cfg["analysis"]["interior_profile"]["r_lo"],
        "extraction_r_max": cfg["analysis"]["interior_profile"]["r_hi"],
    }

    spectroscopy = json.loads(addenda.EXTERIOR.read_text(encoding="utf-8"))
    r40_lcs = sorted(
        {
            float(record["lc"])
            for name, record in spectroscopy["runs"].items()
            if name.startswith("R40_")
        },
        reverse=True,
    )
    ext_matrix = exterior.matrix(False, [], r40_lcs)
    small = [record for record in ext_matrix if record["role"] == "rung"]
    large = [record for record in ext_matrix if record["role"] == "domain_check"]
    assert payload["exterior"]["small_domain_radius"] == small[0]["R"]
    assert payload["exterior"]["large_domain_radius"] == large[0]["R"]
    assert payload["exterior"]["small_sponge_onset"] == (
        small[0]["R"] - small[0]["sponge_width"]
    )
    assert payload["exterior"]["large_sponge_onset"] == (
        large[0]["R"] - large[0]["sponge_width"]
    )
    assert payload["exterior"]["mesh_levels"] == [row["lc"] for row in small]


def test_derived_ladder_span_has_all_three_frozen_sources():
    payload = addenda.build_payload()
    production = json.loads(addenda.PRODUCTION.read_text(encoding="utf-8"))
    cells = [
        cell
        for cell in production["protocol"]["matrix"]
        if cell["pot"] == "linear" and cell["l"] == 0
    ]
    values = [
        production["discriminator"][cell["label"].removeprefix("linear_")]
        ["l2_ratio"]
        for cell in cells
    ]
    derived = payload["derived"]
    assert derived["disc_l0_ladder_span_pts"] == 100.0 * (
        max(values) - min(values)
    )
    assert len(derived["disc_l0_ladder_sources"]) == 3
    assert all(source.endswith("/l2_ratio") for source in derived[
        "disc_l0_ladder_sources"
    ])

    convergence = derived["physical_window_convergence"]
    assert convergence["window_end"] == 30.0
    assert convergence["p1_coarse"] == pytest.approx(1.8883136111088141)
    assert convergence["p1_fine"] == pytest.approx(1.3879107912068713)
    assert convergence["p2_fine"] == pytest.approx(1.3321359575554652)


def test_main_is_idempotent_and_check_detects_stale(tmp_path):
    output = tmp_path / "protocol_addenda.json"
    args = ["--output", str(output)]

    assert addenda.main(args) == 0
    first_mtime = output.stat().st_mtime_ns
    assert addenda.main(args) == 0
    assert output.stat().st_mtime_ns == first_mtime
    assert addenda.main([*args, "--check"]) == 0

    output.write_text(output.read_text(encoding="utf-8") + "stale\n", encoding="utf-8")
    assert addenda.main([*args, "--check"]) == 1
