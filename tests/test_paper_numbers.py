#!/usr/bin/env python3
"""Regresiones stdlib-only para la tabla canónica de números de C2."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "paper_numbers.py"
SPEC = importlib.util.spec_from_file_location("paper_numbers", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
numbers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(numbers)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_json_pointer_handles_dotted_keys_arrays_and_escapes():
    document = {
        "runs": {
            "linear_l0_lc0.028": {
                "ratio_iqr": [0.5, 1.25],
                "a/b": {"til~de": "ok"},
            }
        }
    }

    assert numbers._resolve_pointer(
        document, "/runs/linear_l0_lc0.028/ratio_iqr/1"
    ) == 1.25
    assert numbers._resolve_pointer(
        document, "/runs/linear_l0_lc0.028/a~1b/til~0de"
    ) == "ok"
    with pytest.raises(KeyError, match="missing key"):
        numbers._resolve_pointer(document, "/runs/linear_l0_lc0/missing")
    with pytest.raises(KeyError, match="not numeric"):
        numbers._resolve_pointer(
            document, "/runs/linear_l0_lc0.028/ratio_iqr/last"
        )


def test_provenance_resolves_value_uncertainty_and_transforms(tmp_path):
    source_path = tmp_path / "data/source.json"
    _write_json(
        source_path,
        {
            "runs": {
                "l0_lc0.028": {
                    "value": 0.125,
                    "iqr": [0.1, 0.2],
                }
            }
        },
    )
    table = numbers.NumberTable(tmp_path)
    base = "/runs/l0_lc0.028"
    table.scalar(
        "transformed",
        source=numbers._source("data/source.json", f"{base}/value"),
        transform="fraction-to-percent",
        uncertainty={
            "kind": "iqr",
            "low_source": numbers._source("data/source.json", f"{base}/iqr/0"),
            "high_source": numbers._source("data/source.json", f"{base}/iqr/1"),
            "transform": "fraction-to-percent",
        },
        units="%",
        status="citable",
        paper_section="methods",
    )

    entry = table.entries[0]
    assert entry["source"] == "data/source.json::/runs/l0_lc0.028/value"
    assert entry["value"] == pytest.approx(12.5)
    assert entry["uncertainty"]["low"] == pytest.approx(10.0)
    assert entry["uncertainty"]["high"] == pytest.approx(20.0)
    assert entry["uncertainty"]["sources"]["low"].endswith("/iqr/0")
    table.validate_provenance()

    # El chequeo conserva una expectativa independiente de cada scalar.
    table._documents["data/source.json"]["runs"]["l0_lc0.028"]["iqr"][1] = 0.3
    with pytest.raises(ValueError, match="provenance drift"):
        table.validate_provenance()


def test_note_only_is_null_and_has_exact_note_pointer(tmp_path):
    note = tmp_path / "docs/note.md"
    note.parent.mkdir(parents=True)
    note.write_text("# Nota\n\n## Sección\n", encoding="utf-8")
    table = numbers.NumberTable(tmp_path)
    table.note_only(
        "pending",
        units="1",
        note_pointer="docs/note.md::§Sección",
        paper_section="methods",
        caveat="Falta promoción a JSON.",
    )

    entry = table.entries[0]
    assert entry["source"] == "note-only"
    assert entry["value"] is None
    assert entry["uncertainty"] is None
    assert entry["status"] == "pendiente-de-promoción"
    assert entry["note_pointer"] == "docs/note.md::§Sección"


def test_status_enum_and_non_citable_caveat_are_enforced(tmp_path):
    _write_json(tmp_path / "source.json", {"value": 1.0})
    source = numbers._source("source.json", "/value")

    assert "pendiente-de-promoción" in numbers.STATUSES
    with pytest.raises(ValueError, match="invalid status"):
        numbers.NumberTable(tmp_path).scalar(
            "bad-status",
            source=source,
            units="1",
            status="draft",
            paper_section="methods",
            caveat="No válido.",
        )
    with pytest.raises(ValueError, match="needs a caveat"):
        numbers.NumberTable(tmp_path).scalar(
            "missing-caveat",
            source=source,
            units="1",
            status="no-citable",
            paper_section="methods",
        )


def test_main_is_idempotent_and_check_detects_stale_tmp_outputs(
    tmp_path, monkeypatch, capsys,
):
    # Construir una vez contra las fuentes reales; después aislar sólo la
    # escritura/--check en tmp_path. REPO se mueve para que el mensaje final
    # pueda mostrar rutas relativas sin alterar la selección ya construida.
    table = numbers.build_table(REPO)
    monkeypatch.setattr(numbers, "build_table", lambda _repo: table)
    monkeypatch.setattr(numbers, "REPO", tmp_path)
    json_output = tmp_path / "numbers.json"
    markdown_output = tmp_path / "numbers.md"
    args = [
        "--json-output", str(json_output),
        "--markdown-output", str(markdown_output),
    ]

    assert numbers.main(args) == 0
    first_mtimes = (json_output.stat().st_mtime_ns, markdown_output.stat().st_mtime_ns)
    assert numbers.main(args) == 0
    assert first_mtimes == (
        json_output.stat().st_mtime_ns,
        markdown_output.stat().st_mtime_ns,
    )
    assert numbers.main([*args, "--check"]) == 0

    markdown_output.write_text(
        markdown_output.read_text(encoding="utf-8") + "stale\n",
        encoding="utf-8",
    )
    assert numbers.main([*args, "--check"]) == 1
    assert "outputs desactualizados" in capsys.readouterr().err


def test_five_real_entries_match_their_json_sources():
    table = numbers.build_table(REPO)
    by_id = {entry["id"]: entry for entry in table.entries}
    expected_ids = (
        "disc_l2_lc0.028_l2_ratio_frozen",
        "o1_linear_l0_lc0.028_dev_residual",
        "qnm_R40_late_pooled_minus_im",
        "cavity_lc1.4_w1",
        "cowling_phase0_zeta_max_inside_horizon",
    )

    for entry_id in expected_ids:
        entry = by_id[entry_id]
        path, pointer = entry["source"].split("::", 1)
        document = json.loads((REPO / path).read_text(encoding="utf-8"))
        raw = numbers._resolve_pointer(document, pointer)
        transform = numbers.TRANSFORMS[entry.get("transform", "identity")]
        assert entry["value"] == pytest.approx(transform(raw)), entry_id


def test_required_protocol_and_fine_ladder_inventory_is_json_backed():
    table = numbers.build_table(REPO)
    by_id = {entry["id"]: entry for entry in table.entries}
    required_ids = {
        "production_pulse_A",
        "production_pulse_r0",
        "production_pulse_w",
        "production_mexhat_lambda",
        "production_mexhat_vacuum",
        "production_primary_window_lo",
        "production_primary_window_hi",
        "production_primary_order",
        "production_anchor_order",
        "production_truth_window_lo",
        "production_truth_window_hi",
        "production_truth_order",
        "production_t_end",
        "production_strong_tmin",
        "production_strong_tmax",
        "production_strong_fraction",
        "interior_fine_mesh_scale_l0",
        "interior_fine_mesh_scale_l2",
        "interior_fine_ladder_rms_linear_l0",
        "interior_fine_ladder_rms_mexhat_l0",
        "interior_fine_ladder_rms_linear_l2",
        "interior_fine_ladder_rms_mexhat_l2",
        "exterior_l",
        "exterior_m",
        "exterior_r_ext",
        "exterior_pulse_A",
        "exterior_pulse_r0",
        "exterior_pulse_w",
        "exterior_designed_window_length",
        "exterior_designed_window_t_search",
        "exterior_designed_window_mode_count",
        "exterior_tail_tmin",
        "exterior_tail_floor_tmin",
        "exterior_late_window_length",
        "exterior_late_window_min_offset",
    }

    assert required_ids <= set(by_id)
    for entry_id in required_ids:
        entry = by_id[entry_id]
        assert entry["source"] != "note-only", entry_id
        assert "::/" in entry["source"], entry_id
        assert entry["status"] in {"citable", "citable-con-caveat"}, entry_id


def test_new_protocol_entries_match_exact_rfc6901_sources_and_transforms():
    table = numbers.build_table(REPO)
    by_id = {entry["id"]: entry for entry in table.entries}
    expected_sources = {
        "production_pulse_A": (
            "docs/research/phase2/production/production.json::/protocol/pulse/A",
            0.1,
        ),
        "production_mexhat_lambda": (
            "docs/research/phase2/production/production.json::/protocol/pots/"
            "mexhat/potential_params/lambda_coupling",
            0.1,
        ),
        "production_strong_tmax": (
            "docs/research/phase3/o1_calibration.json::/protocol/"
            "production_strong/t_max",
            10.0,
        ),
        "interior_fine_mesh_scale_l2": (
            "docs/research/phase2/production/production.json::/protocol/"
            "matrix/5/lc",
            0.028,
        ),
        "interior_fine_ladder_rms_mexhat_l2": (
            "docs/research/phase2/production/production.json::/ladder/mexhat_l2/"
            "diff_rms_over_scale/mexhat_l2_lc0.040 vs mexhat_l2_lc0.028",
            21.38146249587637,
        ),
        "exterior_r_ext": (
            "docs/research/phase2/exterior/spectroscopy.json::/protocol/r_ext",
            6.0,
        ),
        "exterior_designed_window_mode_count": (
            "docs/research/phase2/exterior/spectroscopy.json::/protocol/windows/"
            "prony_modes",
            4,
        ),
        "exterior_late_window_min_offset": (
            "docs/research/phase2/exterior/spectroscopy.json::/protocol/"
            "late_windows/pooled_min_offset",
            10.0,
        ),
    }

    for entry_id, (source, expected_value) in expected_sources.items():
        entry = by_id[entry_id]
        assert entry["source"] == source
        assert entry["value"] == pytest.approx(expected_value)
        path, pointer = source.split("::", 1)
        document = json.loads((REPO / path).read_text(encoding="utf-8"))
        transform = numbers.TRANSFORMS[entry.get("transform", "identity")]
        assert entry["value"] == pytest.approx(
            transform(numbers._resolve_pointer(document, pointer))
        )
