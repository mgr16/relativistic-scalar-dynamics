#!/usr/bin/env python3
"""Regression tests for the C4 JSON-to-LaTeX numeric interface."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts/paper_tex_numbers.py"
NUMBERS = REPO / "docs/research/phase3/numbers.json"
SPEC = importlib.util.spec_from_file_location("paper_tex_numbers", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
tex_numbers = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tex_numbers
SPEC.loader.exec_module(tex_numbers)


def _entry(entry_id: str, status: str) -> dict[str, object]:
    return {
        "id": entry_id,
        "value": 1.25,
        "uncertainty": None,
        "source": "docs/source.json::/value",
        "status": status,
        "caveat": "Required publication caveat." if status != "citable" else "",
    }


def test_manifest_uses_only_publishable_entries_with_machine_provenance():
    catalog = tex_numbers._load_catalog(NUMBERS)
    macros = [spec.macro for spec in tex_numbers.SPECS]

    assert len(macros) == len(set(macros))
    assert all(tex_numbers.MACRO_RE.fullmatch(name) for name in macros)
    for spec in tex_numbers.SPECS:
        entry = catalog[spec.entry_id]
        assert entry["status"] in tex_numbers.ALLOWED_STATUSES, spec.entry_id
        assert entry["source"] != "note-only", spec.entry_id
        assert "::/" in entry["source"], spec.entry_id


@pytest.mark.parametrize("status", ["no-citable", "degradado-a-prosa"])
def test_render_rejects_non_publication_statuses(status):
    spec = tex_numbers.MacroSpec("Test", "ForbiddenValue", "forbidden", 2)
    with pytest.raises(ValueError, match="publication status"):
        tex_numbers.render_tex({"forbidden": _entry("forbidden", status)}, (spec,))


def test_render_rejects_note_only_and_missing_caveat_provenance():
    spec = tex_numbers.MacroSpec("Test", "Value", "value", 2)
    note_only = _entry("value", "citable")
    note_only["source"] = "note-only"
    with pytest.raises(ValueError, match="machine provenance"):
        tex_numbers.render_tex({"value": note_only}, (spec,))

    missing_caveat = _entry("value", "citable-con-caveat")
    missing_caveat["caveat"] = ""
    with pytest.raises(ValueError, match="caveat is required"):
        tex_numbers.render_tex({"value": missing_caveat}, (spec,))


def test_real_render_has_adjacent_source_comments_and_publication_rounding():
    rendered = tex_numbers.render_tex(tex_numbers._load_catalog(NUMBERS))
    lines = rendered.splitlines()

    # Every generated definition is preceded by its canonical id, so Fable's
    # manuscript spot-check can map the displayed token without guesswork.
    definitions = [index for index, line in enumerate(lines) if line.startswith("\\newcommand")]
    assert len(definitions) == len(tex_numbers.SPECS)
    assert all(lines[index - 1].startswith("% numbers.json::") for index in definitions)

    # Fixed publication precision deliberately retains trailing zeros.
    expected = (
        r"\newcommand{\DiscLZero}{0.923}",
        r"\newcommand{\FrozenDevLinearFine}{13.1}",
        r"\newcommand{\OOneCLinearMedian}{1.020}",
        r"\newcommand{\LeaverRe}{0.48364}",
        r"\newcommand{\RFortyLateReError}{+0.84}",
        r"\newcommand{\CavityLcOneWTwo}{0.5630}",
        r"\newcommand{\CavityLcOneWOneUnc}{0.0010}",
    )
    for definition in expected:
        assert definition in lines


def test_real_macros_resolve_value_and_uncertainty_from_numbers_json():
    payload = json.loads(NUMBERS.read_text(encoding="utf-8"))
    catalog = {entry["id"]: entry for entry in payload["entries"]}
    rendered = tex_numbers.render_tex(catalog)

    checks = (
        ("DiscLTwo", "disc_l2_lc0.028_l2_ratio_frozen", "value", 3),
        ("OOneCMexhatIQRHigh", "o1_c_mexhat_l0_sampled_k32_median", "uncertainty.high", 3),
        ("RFortyLateMinusImUnc", "qnm_R40_late_pooled_minus_im", "uncertainty.value", 5),
        ("DomainDeltaRe", "domain_R40_lc1match_delta_re", "value", 5),
        ("CavityLcOneWTwoUnc", "cavity_lc1_w2", "uncertainty.value", 4),
    )
    for macro, entry_id, field, digits in checks:
        value = tex_numbers._numeric_field(catalog[entry_id], field, entry_id)
        formatted = tex_numbers._format_number(value, digits)
        assert f"% numbers.json::{entry_id}\n\\newcommand{{\\{macro}}}{{{formatted}}}" in rendered


def test_main_is_idempotent_and_check_detects_stale_without_writing(tmp_path, capsys):
    output = tmp_path / "paper/numbers.tex"
    args = ["--numbers", str(NUMBERS), "--output", str(output)]

    assert tex_numbers.main(args) == 0
    first_content = output.read_text(encoding="utf-8")
    first_mtime = output.stat().st_mtime_ns
    assert tex_numbers.main(args) == 0
    assert output.stat().st_mtime_ns == first_mtime
    assert output.read_text(encoding="utf-8") == first_content
    assert tex_numbers.main([*args, "--check"]) == 0

    stale_content = first_content + "% stale\n"
    output.write_text(stale_content, encoding="utf-8")
    stale_mtime = output.stat().st_mtime_ns
    assert tex_numbers.main([*args, "--check"]) == 1
    assert output.read_text(encoding="utf-8") == stale_content
    assert output.stat().st_mtime_ns == stale_mtime
    assert "missing/stale" in capsys.readouterr().err


def test_atomic_write_leaves_no_temporary_and_preserves_unchanged_mtime(tmp_path):
    output = tmp_path / "numbers.tex"
    assert tex_numbers._atomic_write(output, "first\n") is True
    first_mtime = output.stat().st_mtime_ns
    assert tex_numbers._atomic_write(output, "first\n") is False
    assert output.stat().st_mtime_ns == first_mtime
    assert not output.with_name(".numbers.tex.tmp").exists()
