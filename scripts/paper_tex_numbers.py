#!/usr/bin/env python3
"""Emit publication-formatted LaTeX macros from the canonical number table.

Every numerical token intended for ``paper/main.tex`` is selected by its
``numbers.json`` id.  The status gate deliberately accepts only ``citable``
and ``citable-con-caveat`` entries; diagnostic, retracted, note-only, and
degraded values cannot enter the manuscript through this file.

The output is deterministic, preserves publication trailing zeros, and puts a
``numbers.json::<id>`` provenance comment immediately before every macro.
Writes are atomic and idempotent.  ``--check`` compares without writing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parent.parent
DEFAULT_NUMBERS = REPO / "docs/research/phase3/numbers.json"
DEFAULT_OUTPUT = REPO / "paper/numbers.tex"

ALLOWED_STATUSES = frozenset({"citable", "citable-con-caveat"})
MACRO_RE = re.compile(r"[A-Za-z]+\Z")


@dataclass(frozen=True)
class MacroSpec:
    """One TeX macro selected from a canonical entry or its uncertainty."""

    section: str
    macro: str
    entry_id: str
    digits: int
    field: str = "value"
    show_plus: bool = False


SPECS: tuple[MacroSpec, ...] = (
    # Common production protocol.
    MacroSpec("Production protocol", "PulseAmplitude", "production_pulse_A", 2),
    MacroSpec("Production protocol", "PulseRadius", "production_pulse_r0", 1),
    MacroSpec("Production protocol", "PulseWidth", "production_pulse_w", 1),
    MacroSpec(
        "Production protocol", "MexhatLambda", "production_mexhat_lambda", 2
    ),
    MacroSpec(
        "Production protocol", "MexhatVacuum", "production_mexhat_vacuum", 1
    ),
    MacroSpec(
        "Production protocol", "PrimaryWindowLow", "production_primary_window_lo", 2
    ),
    MacroSpec(
        "Production protocol", "PrimaryWindowHigh", "production_primary_window_hi", 2
    ),
    MacroSpec(
        "Production protocol", "PrimaryWindowOrder", "production_primary_order", 0
    ),
    MacroSpec(
        "Production protocol", "AnchorWindowOrder", "production_anchor_order", 0
    ),
    MacroSpec(
        "Production protocol", "TruthWindowLow", "production_truth_window_lo", 2
    ),
    MacroSpec(
        "Production protocol", "TruthWindowHigh", "production_truth_window_hi", 2
    ),
    MacroSpec(
        "Production protocol", "TruthWindowOrder", "production_truth_order", 0
    ),
    MacroSpec("Production protocol", "ProductionEndTime", "production_t_end", 1),
    MacroSpec(
        "Production protocol", "StrongTimeLow", "production_strong_tmin", 1
    ),
    MacroSpec(
        "Production protocol", "StrongTimeHigh", "production_strong_tmax", 1
    ),
    MacroSpec(
        "Production protocol", "StrongFraction", "production_strong_fraction", 2
    ),
    MacroSpec(
        "Production protocol", "FineMeshScaleLZero", "interior_fine_mesh_scale_l0", 3
    ),
    MacroSpec(
        "Production protocol", "FineMeshScaleLTwo", "interior_fine_mesh_scale_l2", 3
    ),
    MacroSpec(
        "Production protocol",
        "FineLadderRmsLinearLZero",
        "interior_fine_ladder_rms_linear_l0",
        1,
    ),
    MacroSpec(
        "Production protocol",
        "FineLadderRmsMexhatLZero",
        "interior_fine_ladder_rms_mexhat_l0",
        1,
    ),
    MacroSpec(
        "Production protocol",
        "FineLadderRmsLinearLTwo",
        "interior_fine_ladder_rms_linear_l2",
        1,
    ),
    MacroSpec(
        "Production protocol",
        "FineLadderRmsMexhatLTwo",
        "interior_fine_ladder_rms_mexhat_l2",
        1,
    ),
    # R1 self-contained numerical protocol.
    MacroSpec(
        "R1 interior protocol", "InteriorDomainRadius", "interior_domain_radius", 1
    ),
    MacroSpec(
        "R1 interior protocol",
        "InteriorExcisionRadius",
        "interior_excision_radius",
        1,
    ),
    MacroSpec(
        "R1 interior protocol",
        "InteriorOuterMeshScale",
        "interior_outer_mesh_scale",
        1,
    ),
    MacroSpec(
        "R1 interior protocol",
        "InteriorElementDegree",
        "interior_element_degree",
        0,
    ),
    MacroSpec("R1 interior protocol", "InteriorCFL", "interior_cfl", 1),
    MacroSpec(
        "R1 interior protocol",
        "InteriorExtractionCount",
        "interior_extraction_count",
        0,
    ),
    MacroSpec(
        "R1 interior protocol",
        "InteriorExtractionRadiusLow",
        "interior_extraction_r_min",
        1,
    ),
    MacroSpec(
        "R1 interior protocol",
        "InteriorExtractionRadiusHigh",
        "interior_extraction_r_max",
        1,
    ),
    MacroSpec(
        "R1 interior protocol",
        "InteriorMeshScaleCoarse",
        "interior_mesh_scale_coarse",
        3,
    ),
    MacroSpec(
        "R1 interior protocol",
        "InteriorMeshScaleMiddle",
        "interior_mesh_scale_middle",
        3,
    ),
    MacroSpec("R1 interior protocol", "InteriorModeLZero", "interior_mode_l0", 0),
    MacroSpec("R1 interior protocol", "InteriorModeLOne", "interior_mode_l1", 0),
    MacroSpec("R1 interior protocol", "InteriorModeLTwo", "interior_mode_l2", 0),
    MacroSpec("R1 interior protocol", "InteriorLmaxLZero", "interior_lmax_l0", 0),
    MacroSpec("R1 interior protocol", "InteriorLmaxLTwo", "interior_lmax_l2", 0),
    MacroSpec(
        "R1 exterior protocol", "ExteriorDomainSmall", "exterior_domain_small", 0
    ),
    MacroSpec(
        "R1 exterior protocol", "ExteriorDomainLarge", "exterior_domain_large", 0
    ),
    MacroSpec(
        "R1 exterior protocol",
        "ExteriorSpongeSmallWidth",
        "exterior_sponge_small_width",
        0,
    ),
    MacroSpec(
        "R1 exterior protocol",
        "ExteriorSpongeSmallOnset",
        "exterior_sponge_small_onset",
        0,
    ),
    MacroSpec(
        "R1 exterior protocol",
        "ExteriorSpongeLargeWidth",
        "exterior_sponge_large_width",
        0,
    ),
    MacroSpec(
        "R1 exterior protocol",
        "ExteriorSpongeLargeOnset",
        "exterior_sponge_large_onset",
        0,
    ),
    MacroSpec("R1 exterior protocol", "ExteriorEndTime", "exterior_end_time", 0),
    MacroSpec(
        "R1 exterior protocol",
        "ExteriorMeshScaleCoarse",
        "exterior_mesh_scale_coarse",
        1,
    ),
    MacroSpec(
        "R1 exterior protocol",
        "ExteriorMeshScaleMiddle",
        "exterior_mesh_scale_middle",
        1,
    ),
    MacroSpec(
        "R1 exterior protocol",
        "ExteriorMeshScaleFine",
        "exterior_mesh_scale_fine",
        1,
    ),
    MacroSpec(
        "R1 exterior protocol",
        "ExteriorLcInnerRatio",
        "exterior_lc_inner_ratio",
        2,
    ),
    # R1 kinetic mechanism, convergence summary, and conservative budget.
    MacroSpec(
        "R1 mechanism", "MechanismRstar", "mechanism_rstar_mexhat", 3
    ),
    MacroSpec(
        "R1 mechanism", "MechanismRstarIQRLow", "mechanism_rstar_iqr_low", 3
    ),
    MacroSpec(
        "R1 mechanism", "MechanismRstarIQRHigh", "mechanism_rstar_iqr_high", 3
    ),
    MacroSpec(
        "R1 mechanism", "MechanismExponent", "mechanism_ratio_exponent", 2
    ),
    MacroSpec(
        "R1 mechanism",
        "MechanismExponentIQRLow",
        "mechanism_ratio_exponent_iqr_low",
        2,
    ),
    MacroSpec(
        "R1 mechanism",
        "MechanismExponentIQRHigh",
        "mechanism_ratio_exponent_iqr_high",
        2,
    ),
    MacroSpec(
        "R1 mechanism", "MechanismRatioThreshold", "mechanism_ratio_threshold", 2
    ),
    MacroSpec(
        "R1 interior result", "DiscLadderSpan", "disc_l0_ladder_span_pts", 1
    ),
    MacroSpec(
        "R1 interior result",
        "ProductionBudgetLow",
        "production_budget_low_percent",
        0,
    ),
    MacroSpec(
        "R1 interior result",
        "ProductionBudgetHigh",
        "production_budget_high_percent",
        0,
    ),
    MacroSpec(
        "R1 Cowling",
        "CowlingInsideMaximum",
        "cowling_phase0_zeta_max_inside_horizon",
        5,
    ),
    MacroSpec(
        "R1 Cowling", "CowlingDeepValue", "cowling_phase0_zeta_at_rmin", 5
    ),
    # R2 one-dimensional sensitivity boundary.
    MacroSpec("R2 sensitivity", "SensitivityLambdaLow", "sens_lambda_min", 2),
    MacroSpec(
        "R2 sensitivity", "SensitivityLambdaBoundary", "sens_lambda_boundary", 2
    ),
    MacroSpec("R2 sensitivity", "SensitivityLambdaHigh", "sens_lambda_max", 2),
    MacroSpec(
        "R2 sensitivity", "SensitivityAmplitudeLow", "sens_amplitude_min", 2
    ),
    MacroSpec(
        "R2 sensitivity", "SensitivityAmplitudeHigh", "sens_amplitude_max", 2
    ),
    MacroSpec(
        "R2 sensitivity", "SensitivityGridCount", "sens_grid_cell_count", 0
    ),
    MacroSpec("R2 sensitivity", "SensitivityDiscLow", "sens_disc_min", 3),
    MacroSpec("R2 sensitivity", "SensitivityDiscHigh", "sens_disc_max", 3),
    MacroSpec(
        "R2 sensitivity", "SensitivityMaxDeviation", "sens_disc_max_abs_dev", 3
    ),
    MacroSpec(
        "R2 sensitivity",
        "SensitivityReviewThreshold",
        "sens_disc_review_threshold",
        2,
    ),
    # Frozen H2 discriminator at each mode's finest available rung.
    MacroSpec(
        "Interior discriminator", "DiscLZero", "disc_l0_lc0.028_l2_ratio_frozen", 3
    ),
    MacroSpec(
        "Interior discriminator", "DiscLOne", "disc_l1_lc0.040_l2_ratio_frozen", 3
    ),
    MacroSpec(
        "Interior discriminator", "DiscLTwo", "disc_l2_lc0.028_l2_ratio_frozen", 3
    ),
    MacroSpec(
        "Interior discriminator", "DiscOracle", "disc_oracle_1d_l0_l2_ratio", 3
    ),
    MacroSpec(
        "Interior discriminator",
        "DiscMedianLZero",
        "disc_l0_lc0.028_ratio_median_frozen",
        3,
    ),
    MacroSpec(
        "Interior discriminator",
        "DiscIQRLowLZero",
        "disc_l0_lc0.028_ratio_median_frozen",
        3,
        "uncertainty.low",
    ),
    MacroSpec(
        "Interior discriminator",
        "DiscIQRHighLZero",
        "disc_l0_lc0.028_ratio_median_frozen",
        3,
        "uncertainty.high",
    ),
    MacroSpec(
        "Interior discriminator",
        "DiscMedianLOne",
        "disc_l1_lc0.040_ratio_median_frozen",
        3,
    ),
    MacroSpec(
        "Interior discriminator",
        "DiscIQRLowLOne",
        "disc_l1_lc0.040_ratio_median_frozen",
        3,
        "uncertainty.low",
    ),
    MacroSpec(
        "Interior discriminator",
        "DiscIQRHighLOne",
        "disc_l1_lc0.040_ratio_median_frozen",
        3,
        "uncertainty.high",
    ),
    MacroSpec(
        "Interior discriminator",
        "DiscMedianLTwo",
        "disc_l2_lc0.028_ratio_median_frozen",
        3,
    ),
    MacroSpec(
        "Interior discriminator",
        "DiscIQRLowLTwo",
        "disc_l2_lc0.028_ratio_median_frozen",
        3,
        "uncertainty.low",
    ),
    MacroSpec(
        "Interior discriminator",
        "DiscIQRHighLTwo",
        "disc_l2_lc0.028_ratio_median_frozen",
        3,
        "uncertainty.high",
    ),
    MacroSpec(
        "Interior discriminator", "TruthFloorLOne", "o1_truth_floor_linear_l1", 1
    ),
    MacroSpec(
        "Interior discriminator", "TruthFloorLTwo", "o1_truth_floor_linear_l2", 1
    ),
    MacroSpec(
        "Interior discriminator",
        "FrozenDevLinearFine",
        "linear_l0_lc0.028_dev_primary_median",
        1,
    ),
    MacroSpec(
        "Interior discriminator",
        "FrozenDevMexhatFine",
        "mexhat_l0_lc0.028_dev_primary_median",
        1,
    ),
    MacroSpec(
        "Interior discriminator", "LumpingDevMedian", "mass_lumping_ab_dev_median", 1
    ),
    MacroSpec(
        "Interior discriminator", "LumpingDevMax", "mass_lumping_ab_dev_max", 1
    ),
    # C2 o1 calibration diagnostics (all retain their caveats in prose).
    MacroSpec(
        "Window calibration",
        "OOneCLinearMedian",
        "o1_c_linear_l0_sampled_k32_median",
        3,
    ),
    MacroSpec(
        "Window calibration",
        "OOneCLinearIQRLow",
        "o1_c_linear_l0_sampled_k32_median",
        3,
        "uncertainty.low",
    ),
    MacroSpec(
        "Window calibration",
        "OOneCLinearIQRHigh",
        "o1_c_linear_l0_sampled_k32_median",
        3,
        "uncertainty.high",
    ),
    MacroSpec(
        "Window calibration",
        "OOneCMexhatMedian",
        "o1_c_mexhat_l0_sampled_k32_median",
        3,
    ),
    MacroSpec(
        "Window calibration",
        "OOneCMexhatIQRLow",
        "o1_c_mexhat_l0_sampled_k32_median",
        3,
        "uncertainty.low",
    ),
    MacroSpec(
        "Window calibration",
        "OOneCMexhatIQRHigh",
        "o1_c_mexhat_l0_sampled_k32_median",
        3,
        "uncertainty.high",
    ),
    MacroSpec(
        "Window calibration",
        "OOneLinearFineBefore",
        "o1_linear_l0_lc0.028_dev_before_common_support",
        2,
    ),
    MacroSpec(
        "Window calibration",
        "OOneLinearFineResidual",
        "o1_linear_l0_lc0.028_dev_residual",
        2,
    ),
    MacroSpec(
        "Window calibration",
        "OOneLinearFineFraction",
        "o1_linear_l0_lc0.028_dev_fraction_explained",
        1,
    ),
    MacroSpec(
        "Window calibration",
        "OOneMexhatFineBefore",
        "o1_mexhat_l0_lc0.028_dev_before_common_support",
        2,
    ),
    MacroSpec(
        "Window calibration",
        "OOneMexhatFineResidual",
        "o1_mexhat_l0_lc0.028_dev_residual",
        2,
    ),
    MacroSpec(
        "Window calibration",
        "OOneMexhatFineFraction",
        "o1_mexhat_l0_lc0.028_dev_fraction_explained",
        1,
    ),
    MacroSpec(
        "Window calibration",
        "OOneDeficitExplained",
        "o1_deficit_fraction_explained",
        1,
    ),
    MacroSpec(
        "Window calibration", "OOneDeficitResidual", "o1_remaining_l2_deficit", 2
    ),
    # Paired-potential off-channel contamination.
    MacroSpec(
        "Off-channel diagnostics",
        "OffLTwoToZeroLinear",
        "linear_l2_lc0.028_junk_offchannel_l0",
        2,
    ),
    MacroSpec(
        "Off-channel diagnostics",
        "OffLTwoToZeroMexhat",
        "mexhat_l2_lc0.028_junk_offchannel_l0",
        2,
    ),
    MacroSpec(
        "Off-channel diagnostics",
        "OffLTwoToFourLinear",
        "linear_l2_lc0.028_junk_offchannel_l4",
        2,
    ),
    MacroSpec(
        "Off-channel diagnostics",
        "OffLTwoToFourMexhat",
        "mexhat_l2_lc0.028_junk_offchannel_l4",
        2,
    ),
    MacroSpec(
        "Off-channel diagnostics",
        "OffLOneToZeroLinear",
        "linear_l1_lc0.040_junk_offchannel_l0",
        2,
    ),
    MacroSpec(
        "Off-channel diagnostics",
        "OffLOneToZeroMexhat",
        "mexhat_l1_lc0.040_junk_offchannel_l0",
        2,
    ),
    # Exterior extraction protocol.
    MacroSpec("Exterior protocol", "ExteriorMultipoleL", "exterior_l", 0),
    MacroSpec("Exterior protocol", "ExteriorMultipoleM", "exterior_m", 0),
    MacroSpec("Exterior protocol", "ExteriorRadius", "exterior_r_ext", 1),
    MacroSpec("Exterior protocol", "ExteriorPulseAmplitude", "exterior_pulse_A", 3),
    MacroSpec("Exterior protocol", "ExteriorPulseRadius", "exterior_pulse_r0", 1),
    MacroSpec("Exterior protocol", "ExteriorPulseWidth", "exterior_pulse_w", 1),
    MacroSpec(
        "Exterior protocol",
        "ExteriorWindowLength",
        "exterior_designed_window_length",
        1,
    ),
    MacroSpec(
        "Exterior protocol",
        "ExteriorSearchTime",
        "exterior_designed_window_t_search",
        1,
    ),
    MacroSpec(
        "Exterior protocol",
        "ExteriorModeCount",
        "exterior_designed_window_mode_count",
        0,
    ),
    MacroSpec("Exterior protocol", "ExteriorTailTime", "exterior_tail_tmin", 1),
    MacroSpec(
        "Exterior protocol", "ExteriorFloorTime", "exterior_tail_floor_tmin", 1
    ),
    MacroSpec(
        "Exterior protocol", "ExteriorLateWindowLength", "exterior_late_window_length", 1
    ),
    MacroSpec(
        "Exterior protocol",
        "ExteriorLateMinOffset",
        "exterior_late_window_min_offset",
        1,
    ),
    # QNM and domain-systematic results.
    MacroSpec("Exterior results", "LeaverRe", "qnm_leaver_l2_re", 5),
    MacroSpec("Exterior results", "LeaverMinusIm", "qnm_leaver_l2_minus_im", 5),
    MacroSpec("Exterior results", "RTwentyFineRe", "qnm_R20_lc0.7_re", 5),
    MacroSpec(
        "Exterior results", "RTwentyFineReError", "qnm_R20_lc0.7_re_error_signed", 2
    ),
    MacroSpec(
        "Exterior results", "RTwentyFineOrder", "qnm_R20_fine_re_order", 2
    ),
    MacroSpec(
        "Exterior results", "RFortyLateRe", "qnm_R40_late_pooled_re", 5
    ),
    MacroSpec(
        "Exterior results",
        "RFortyLateReUnc",
        "qnm_R40_late_pooled_re",
        5,
        "uncertainty.value",
    ),
    MacroSpec(
        "Exterior results",
        "RFortyLateReError",
        "qnm_R40_late_pooled_re_error_signed",
        2,
        show_plus=True,
    ),
    MacroSpec(
        "Exterior results", "RFortyLateMinusIm", "qnm_R40_late_pooled_minus_im", 5
    ),
    MacroSpec(
        "Exterior results",
        "RFortyLateMinusImUnc",
        "qnm_R40_late_pooled_minus_im",
        5,
        "uncertainty.value",
    ),
    MacroSpec(
        "Exterior results",
        "RFortyLateMinusImError",
        "qnm_R40_late_pooled_minus_im_error_signed",
        2,
        show_plus=True,
    ),
    MacroSpec(
        "Exterior results",
        "EarlyBiasLcOne",
        "qnm_R40_lc1match_early_overtone_bias",
        1,
    ),
    MacroSpec(
        "Exterior results",
        "EarlyBiasLcPointSeven",
        "qnm_R40_lc0.7match_early_overtone_bias",
        1,
    ),
    MacroSpec(
        "Exterior results", "DomainDeltaRe", "domain_R40_lc1match_delta_re", 5
    ),
    MacroSpec(
        "Exterior results",
        "DomainDeltaReUnc",
        "domain_R40_lc1match_delta_re",
        5,
        "uncertainty.value",
    ),
    MacroSpec(
        "Exterior results",
        "DomainFloorReductionLcOne",
        "domain_R40_lc1match_tail_floor_reduction",
        1,
    ),
    MacroSpec(
        "Exterior results",
        "DomainFloorReductionLcPointSeven",
        "domain_R40_lc0.7match_tail_floor_reduction",
        1,
    ),
    MacroSpec(
        "Exterior results",
        "DomainPeakFloorRTwenty",
        "domain_R40_lc1match_R20_peak_over_floor",
        1,
    ),
    MacroSpec(
        "Exterior results",
        "DomainPeakFloorRForty",
        "domain_R40_lc1match_R40_peak_over_floor",
        1,
    ),
    # R=20 cavity doublet (diagnostic, explicitly not a physical QNM).
    MacroSpec("Cavity doublet", "CavityLcOneWOne", "cavity_lc1_w1", 4),
    MacroSpec(
        "Cavity doublet",
        "CavityLcOneWOneUnc",
        "cavity_lc1_w1",
        4,
        "uncertainty.value",
    ),
    MacroSpec("Cavity doublet", "CavityLcOneWTwo", "cavity_lc1_w2", 4),
    MacroSpec(
        "Cavity doublet",
        "CavityLcOneWTwoUnc",
        "cavity_lc1_w2",
        4,
        "uncertainty.value",
    ),
)


def _load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    """Load and structurally validate the canonical table by id."""
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    if payload.get("generated_by") != "scripts/paper_numbers.py":
        raise ValueError(f"{path}: unexpected generator")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise TypeError(f"{path}: entries must be a list")

    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise TypeError(f"{path}: malformed number entry")
        entry_id = entry["id"]
        if entry_id in by_id:
            raise ValueError(f"{path}: duplicate number id {entry_id!r}")
        by_id[entry_id] = entry
    return by_id


def _select_entry(
    catalog: Mapping[str, Mapping[str, Any]], entry_id: str
) -> Mapping[str, Any]:
    if entry_id not in catalog:
        raise KeyError(f"numbers.json has no id {entry_id!r}")
    entry = catalog[entry_id]
    status = entry.get("status")
    if status not in ALLOWED_STATUSES:
        raise ValueError(
            f"refusing numbers.json::{entry_id}: publication status {status!r}"
        )
    source = entry.get("source")
    if not isinstance(source, str) or source == "note-only" or "::/" not in source:
        raise ValueError(f"numbers.json::{entry_id}: missing machine provenance")
    if status == "citable-con-caveat" and not entry.get("caveat"):
        raise ValueError(f"numbers.json::{entry_id}: caveat is required")
    return entry


def _numeric_field(entry: Mapping[str, Any], field: str, entry_id: str) -> float:
    current: Any = entry
    for part in field.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"numbers.json::{entry_id} has no field {field!r}")
        current = current[part]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise TypeError(
            f"numbers.json::{entry_id}::{field} is not a numeric scalar: {current!r}"
        )
    value = float(current)
    if not math.isfinite(value):
        raise ValueError(f"numbers.json::{entry_id}::{field} is non-finite")
    return value


def _format_number(value: float, digits: int, show_plus: bool = False) -> str:
    if digits < 0:
        raise ValueError("digits must be non-negative")
    sign = "+" if show_plus else ""
    return format(value, f"{sign}.{digits}f")


def render_tex(
    catalog: Mapping[str, Mapping[str, Any]],
    specs: Iterable[MacroSpec] = SPECS,
) -> str:
    """Render deterministic TeX; status and provenance are checked per macro."""
    specs = tuple(specs)
    macro_names: set[str] = set()
    lines = [
        "% AUTO-GENERATED by scripts/paper_tex_numbers.py; do not edit.",
        "% Source: docs/research/phase3/numbers.json",
        "% Regenerate: python scripts/paper_tex_numbers.py",
    ]
    previous_section: str | None = None
    for spec in specs:
        if not MACRO_RE.fullmatch(spec.macro):
            raise ValueError(f"invalid TeX control-sequence name {spec.macro!r}")
        if spec.macro in macro_names:
            raise ValueError(f"duplicate TeX macro {spec.macro!r}")
        macro_names.add(spec.macro)
        entry = _select_entry(catalog, spec.entry_id)
        value = _numeric_field(entry, spec.field, spec.entry_id)
        if spec.section != previous_section:
            lines.extend(("", f"% --- {spec.section} ---"))
            previous_section = spec.section
        lines.append(f"% numbers.json::{spec.entry_id}")
        lines.append(
            rf"\newcommand{{\{spec.macro}}}{{{_format_number(value, spec.digits, spec.show_plus)}}}"
        )
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, content: str) -> bool:
    """Atomically publish changed content; preserve mtime when unchanged."""
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
    parser.add_argument("--numbers", type=Path, default=DEFAULT_NUMBERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="render and byte-compare the existing output without writing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    catalog = _load_catalog(args.numbers)
    rendered = render_tex(catalog)

    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
            print(f"TeX number macros missing/stale: {args.output}", file=sys.stderr)
            return 1
        print(f"TeX numbers: {len(SPECS)} macros verified; output up to date")
        return 0

    changed = _atomic_write(args.output, rendered)
    try:
        display = args.output.resolve().relative_to(REPO.resolve())
    except ValueError:
        display = args.output
    print(
        f"TeX numbers: {len(SPECS)} macros -> {display} "
        f"({'updated' if changed else 'unchanged'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
