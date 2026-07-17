"""Static integrity checks for the C4 RevTeX manuscript sources."""

from __future__ import annotations

import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
MAIN = REPO / "paper/main.tex"
REFS = REPO / "paper/refs.bib"
NUMBERS = REPO / "docs/research/phase3/numbers.json"


def _main_text() -> str:
    return MAIN.read_text(encoding="utf-8")


def test_all_citations_exist_and_every_bib_entry_is_used():
    text = _main_text()
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", text)
        for key in group.split(",")
    }
    defined = set(re.findall(r"^@\w+\{([^,]+),", REFS.read_text(encoding="utf-8"), re.M))

    assert cited == defined


def test_numeric_provenance_comments_resolve_to_publishable_entries():
    catalog = {
        entry["id"]: entry
        for entry in json.loads(NUMBERS.read_text(encoding="utf-8"))["entries"]
    }
    ids = re.findall(r"^% numbers\.json::(.+)$", _main_text(), re.M)

    assert ids
    for entry_id in ids:
        assert entry_id in catalog
        assert catalog[entry_id]["status"] in {"citable", "citable-con-caveat"}
        assert catalog[entry_id]["source"] != "note-only"


def test_manuscript_has_no_raw_decimal_results_or_retracted_tokens():
    text_without_comments = re.sub(r"(?m)^%.*$", "", _main_text())

    # Protocol and result decimals must enter through paper/numbers.tex macros.
    assert not re.search(r"(?<![A-Za-z])\d+\.\d+(?![A-Za-z])", text_without_comments)
    for forbidden in ("2.1", "0.209", "0.419", "3.3", "peak_ratio", "o1["):
        assert forbidden not in text_without_comments
    assert "to be provided" not in text_without_comments
    assert r"\today" not in text_without_comments


def test_internal_jargon_is_absent_outside_the_single_definition():
    text_without_comments = re.sub(r"(?m)^%.*$", "", _main_text())
    lowered = text_without_comments.lower()

    for forbidden in (r"\btruth\b", r"\brungs?\b", r"\bsmoke\b"):
        assert not re.search(forbidden, lowered)
    frozen_uses = re.findall(r"\bfrozen\b", lowered)
    assert len(frozen_uses) == 1
    assert "production values (frozen before the calibration" in lowered

    abstract = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        text_without_comments,
        re.S,
    )
    assert abstract is not None
    assert not re.search(r"\b(frozen|truth|rung|floor)\b", abstract.group(1).lower())


def test_round_r1_sections_and_protocol_tables_are_present():
    text = _main_text()
    assert r"\section{Data availability}" in text
    assert "[REPOSITORY-URL-AT-DEPOSIT]" in text
    assert r"\appendix" in text
    assert r"\section{Initial data}" in text
    assert r"\section{Numerical protocol}" in text
    assert text.count(r"\begin{table*}") >= 2
    assert r"\toprule" in text and r"\bottomrule" in text


def test_round_r2_sensitivity_boundary_and_tkp_scope_are_present():
    text = _main_text()
    for entry_id in (
        "sens_grid_cell_count",
        "sens_disc_min",
        "sens_disc_max",
        "sens_disc_max_abs_dev",
        "sens_disc_review_threshold",
    ):
        assert f"% numbers.json::{entry_id}" in text
    assert "not uniform robustness across the full scan" in text
    assert "P_l(r/M-1)" in text
    assert "full-interior profiles" in text


def test_all_figure_inputs_exist_as_versioned_pdfs():
    paths = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", _main_text())

    assert len(paths) == 5
    for relative in paths:
        figure = MAIN.parent / relative
        assert figure.is_file()
        assert figure.suffix == ".pdf"


def test_revtex_structure_and_environments_are_balanced():
    text = _main_text()
    assert "{revtex4-2}" in text
    assert "prd" in text and "reprint" in text
    assert text.count("{") == text.count("}")

    stack: list[str] = []
    for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", text):
        action, environment = match.groups()
        if action == "begin":
            stack.append(environment)
        else:
            assert stack and stack.pop() == environment
    assert not stack
