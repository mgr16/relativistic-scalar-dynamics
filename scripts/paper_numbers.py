#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C2 — tabla canónica y trazable de números del manuscrito.

El script lee exclusivamente artefactos JSON versionados y emite, de forma
determinista, ``docs/research/phase3/numbers.json`` y ``numbers.md``. Cada
valor numérico y cada componente de incertidumbre se resuelve mediante una
fuente inequívoca ``archivo::/JSON/Pointer``. Los números que aún viven sólo
en notas se conservan con valor nulo y estado ``pendiente-de-promoción``.

No lee resultados gitignored ni modifica artefactos congelados de F0--F2.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable


REPO = Path(__file__).resolve().parent.parent
DEFAULT_JSON = REPO / "docs/research/phase3/numbers.json"
DEFAULT_MARKDOWN = REPO / "docs/research/phase3/numbers.md"

PRODUCTION = "docs/research/phase2/production/production.json"
EXTERIOR = "docs/research/phase2/exterior/spectroscopy.json"
CAVITY = "docs/research/phase1/cavity/summary.json"
PHASE0 = "docs/research/phase0/pilot_oracle_summary.json"
CALIBRATION = "docs/research/phase3/o1_calibration.json"

STATUSES = {
    "citable",
    "citable-con-caveat",
    "no-citable",
    "pendiente-de-promoción",
    "degradado-a-prosa",
}
PAPER_SECTIONS = {
    "intro", "methods", "interior", "exterior", "discusión",
}


def _source(path: str, pointer: str) -> str:
    if not pointer.startswith("/"):
        raise ValueError(f"JSON Pointer must start with '/': {pointer!r}")
    return f"{path}::{pointer}"


def _decode_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _resolve_pointer(document: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 JSON Pointer, rejecting ambiguous array access."""
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON Pointer {pointer!r}")
    current = document
    for raw_token in pointer[1:].split("/"):
        token = _decode_pointer_token(raw_token)
        if isinstance(current, dict):
            if token not in current:
                raise KeyError(f"JSON Pointer {pointer!r}: missing key {token!r}")
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit():
                raise KeyError(
                    f"JSON Pointer {pointer!r}: array index is not numeric: "
                    f"{token!r}"
                )
            index = int(token)
            if index >= len(current):
                raise IndexError(
                    f"JSON Pointer {pointer!r}: array index {index} out of range"
                )
            current = current[index]
        else:
            raise TypeError(
                f"JSON Pointer {pointer!r}: cannot descend through "
                f"{type(current).__name__}"
            )
    return current


def _identity(value: Any) -> Any:
    return value


def _fraction_to_percent(value: Any) -> float:
    return 100.0 * float(value)


def _absolute(value: Any) -> float:
    return abs(float(value))


def _reciprocal(value: Any) -> float:
    number = float(value)
    if number == 0.0:
        raise ValueError("cannot invert a zero source value")
    return 1.0 / number


TRANSFORMS: dict[str, Callable[[Any], Any]] = {
    "identity": _identity,
    "fraction-to-percent": _fraction_to_percent,
    "absolute": _absolute,
    "reciprocal": _reciprocal,
}


def _validate_scalar(value: Any, context: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{context}: non-finite numeric value {value!r}")
        return
    raise TypeError(f"{context}: expected JSON scalar, got {type(value).__name__}")


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=0.0)
    return left == right


class NumberTable:
    """Build entries while retaining independent provenance checks."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.entries: list[dict[str, Any]] = []
        self._documents: dict[str, Any] = {}
        self._ids: set[str] = set()
        self._checks: list[tuple[str, str, str, Any]] = []

    def document(self, path: str) -> Any:
        if path not in self._documents:
            full = self.repo / path
            if not full.is_file():
                raise FileNotFoundError(full)
            with full.open(encoding="utf-8") as stream:
                self._documents[path] = json.load(stream)
        return self._documents[path]

    def resolve(self, source: str, transform: str = "identity") -> Any:
        if source == "note-only" or "::" not in source:
            raise ValueError(f"invalid JSON-backed source {source!r}")
        path, pointer = source.split("::", 1)
        raw = _resolve_pointer(self.document(path), pointer)
        if transform not in TRANSFORMS:
            raise ValueError(f"unknown transform {transform!r}")
        value = TRANSFORMS[transform](raw)
        _validate_scalar(value, source)
        return value

    def _claim_id(self, entry_id: str) -> None:
        if not entry_id or entry_id in self._ids:
            raise ValueError(f"empty or duplicate entry id {entry_id!r}")
        self._ids.add(entry_id)

    def scalar(
        self,
        entry_id: str,
        *,
        source: str,
        units: str,
        status: str,
        paper_section: str,
        caveat: str = "",
        transform: str = "identity",
        uncertainty: dict[str, Any] | None = None,
        support_source: str | None = None,
    ) -> None:
        self._claim_id(entry_id)
        value = self.resolve(source, transform)
        resolved_uncertainty = self._resolve_uncertainty(entry_id, uncertainty)
        entry: dict[str, Any] = {
            "id": entry_id,
            "value": value,
            "uncertainty": resolved_uncertainty,
            "units": units,
            "source": source,
            "status": status,
            "caveat": caveat,
            "paper_section": paper_section,
        }
        if transform != "identity":
            entry["transform"] = transform
        if support_source is not None:
            support = self.resolve(support_source)
            if isinstance(support, bool) or not isinstance(support, int):
                raise TypeError(f"{entry_id}: support count must be an integer")
            entry["support"] = {"n": support, "source": support_source}
            self._checks.append(
                (f"{entry_id}.support", support_source, "identity", support)
            )
        self._validate_entry(entry)
        self.entries.append(entry)
        self._checks.append((entry_id, source, transform, value))

    def note_only(
        self,
        entry_id: str,
        *,
        units: str,
        note_pointer: str,
        paper_section: str,
        caveat: str,
        resolution: str | None = None,
    ) -> None:
        self._claim_id(entry_id)
        if "::" not in note_pointer:
            raise ValueError(f"{entry_id}: note pointer must include path::section")
        note_path, section = note_pointer.split("::", 1)
        if not (self.repo / note_path).is_file() or not section.strip():
            raise FileNotFoundError(f"invalid note pointer {note_pointer!r}")
        entry = {
            "id": entry_id,
            "value": None,
            "uncertainty": None,
            "units": units,
            "source": "note-only",
            "note_pointer": note_pointer,
            "status": "degradado-a-prosa" if resolution else "pendiente-de-promoción",
            "caveat": caveat,
            "paper_section": paper_section,
        }
        if resolution:
            entry["resolution"] = resolution
        self._validate_entry(entry)
        self.entries.append(entry)

    def _resolve_uncertainty(
        self, entry_id: str, spec: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if spec is None:
            return None
        kind = spec.get("kind")
        if kind == "symmetric":
            source = str(spec["source"])
            transform = str(spec.get("transform", "identity"))
            value = self.resolve(source, transform)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{entry_id}: symmetric uncertainty is not numeric")
            if value < 0:
                raise ValueError(f"{entry_id}: symmetric uncertainty is negative")
            self._checks.append(
                (f"{entry_id}.uncertainty", source, transform, value)
            )
            result = {"kind": "symmetric", "value": value, "source": source}
            if transform != "identity":
                result["transform"] = transform
            return result
        if kind == "iqr":
            low_source = str(spec["low_source"])
            high_source = str(spec["high_source"])
            transform = str(spec.get("transform", "identity"))
            low = self.resolve(low_source, transform)
            high = self.resolve(high_source, transform)
            if not all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in (low, high)
            ):
                raise TypeError(f"{entry_id}: IQR endpoints must be numeric")
            if low > high:
                raise ValueError(f"{entry_id}: reversed IQR [{low}, {high}]")
            self._checks.extend([
                (f"{entry_id}.uncertainty.low", low_source, transform, low),
                (f"{entry_id}.uncertainty.high", high_source, transform, high),
            ])
            result = {
                "kind": "IQR",
                "low": low,
                "high": high,
                "sources": {"low": low_source, "high": high_source},
            }
            if transform != "identity":
                result["transform"] = transform
            return result
        if kind == "budget":
            components = []
            for component in spec.get("components", []):
                label = str(component["label"])
                source = str(component["source"])
                transform = str(component.get("transform", "identity"))
                value = self.resolve(source, transform)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(
                        f"{entry_id}: budget component {label!r} is not numeric"
                    )
                record = {"label": label, "value": value, "source": source}
                if transform != "identity":
                    record["transform"] = transform
                components.append(record)
                self._checks.append(
                    (f"{entry_id}.uncertainty.{label}", source, transform, value)
                )
            if not components:
                raise ValueError(f"{entry_id}: empty uncertainty budget")
            return {"kind": "budget", "components": components}
        raise ValueError(f"{entry_id}: unknown uncertainty kind {kind!r}")

    @staticmethod
    def _validate_entry(entry: dict[str, Any]) -> None:
        missing = {
            "id", "value", "uncertainty", "units", "source", "status",
            "caveat", "paper_section",
        } - set(entry)
        if missing:
            raise KeyError(f"{entry.get('id', '<unknown>')}: missing fields {missing}")
        if entry["status"] not in STATUSES:
            raise ValueError(f"{entry['id']}: invalid status {entry['status']!r}")
        if entry["paper_section"] not in PAPER_SECTIONS:
            raise ValueError(
                f"{entry['id']}: invalid paper section {entry['paper_section']!r}"
            )
        if entry["status"] != "citable" and not entry["caveat"].strip():
            raise ValueError(f"{entry['id']}: non-citable status needs a caveat")
        if entry["source"] == "note-only":
            if entry["status"] not in ("pendiente-de-promoción", "degradado-a-prosa"):
                raise ValueError(f"{entry['id']}: note-only status is inconsistent")
            if entry["value"] is not None or entry["uncertainty"] is not None:
                raise ValueError(f"{entry['id']}: note-only values must be null")
        elif entry["value"] is None:
            raise ValueError(f"{entry['id']}: JSON-backed value is null")

    def validate_provenance(self) -> None:
        for context, source, transform, expected in self._checks:
            actual = self.resolve(source, transform)
            if not _same_value(actual, expected):
                raise ValueError(
                    f"{context}: provenance drift: emitted {expected!r}, "
                    f"source now resolves to {actual!r}"
                )


def _symmetric(source: str, transform: str = "identity") -> dict[str, Any]:
    return {"kind": "symmetric", "source": source, "transform": transform}


def _iqr(low_source: str, high_source: str) -> dict[str, Any]:
    return {"kind": "iqr", "low_source": low_source, "high_source": high_source}


def _budget(*components: tuple[str, str, str]) -> dict[str, Any]:
    return {
        "kind": "budget",
        "components": [
            {"label": label, "source": source, "transform": transform}
            for label, source, transform in components
        ],
    }


def _has_pointer(table: NumberTable, path: str, pointer: str) -> bool:
    try:
        _resolve_pointer(table.document(path), pointer)
    except (KeyError, IndexError, TypeError):
        return False
    return True


def _add_production_protocol(table: NumberTable) -> None:
    """Expose the frozen interior protocol as citable JSON-backed scalars."""
    scalar_specs = (
        ("production_pulse_A", "/protocol/pulse/A", "1"),
        ("production_pulse_r0", "/protocol/pulse/r0", "M"),
        ("production_pulse_w", "/protocol/pulse/w", "M"),
        (
            "production_mexhat_lambda",
            "/protocol/pots/mexhat/potential_params/lambda_coupling",
            "1",
        ),
        (
            "production_mexhat_vacuum",
            "/protocol/pots/mexhat/potential_params/vacuum_value",
            "1",
        ),
        ("production_primary_window_lo", "/protocol/primary/window/0", "M"),
        ("production_primary_window_hi", "/protocol/primary/window/1", "M"),
        ("production_primary_order", "/protocol/primary/order", "orden"),
        ("production_anchor_order", "/protocol/anchor/order", "orden"),
        ("production_truth_window_lo", "/protocol/truth_1d/window/0", "M"),
        ("production_truth_window_hi", "/protocol/truth_1d/window/1", "M"),
        ("production_truth_order", "/protocol/truth_1d/order", "orden"),
        ("production_t_end", "/protocol/t_end", "M"),
    )
    for entry_id, pointer, units in scalar_specs:
        table.scalar(
            entry_id,
            source=_source(PRODUCTION, pointer),
            units=units,
            status="citable",
            paper_section="methods",
        )

    strong_specs = (
        ("production_strong_tmin", "/protocol/production_strong/t_min", "M"),
        ("production_strong_tmax", "/protocol/production_strong/t_max", "M"),
        (
            "production_strong_fraction",
            "/protocol/production_strong/strong_fraction",
            "fracción del máximo",
        ),
    )
    for entry_id, pointer, units in strong_specs:
        table.scalar(
            entry_id,
            source=_source(CALIBRATION, pointer),
            units=units,
            status="citable-con-caveat",
            caveat=(
                "Protocolo de fase fuerte de producción documentado por la "
                "calibración o1 revisada."
            ),
            paper_section="methods",
        )

    # These two matrix records are the actual fine rungs for the l=0 and l=2
    # production ladders.  Keep the array indices explicit so provenance also
    # verifies the frozen matrix ordering.
    for multipole, matrix_index in (("l0", 2), ("l2", 5)):
        table.scalar(
            f"interior_fine_mesh_scale_{multipole}",
            source=_source(PRODUCTION, f"/protocol/matrix/{matrix_index}/lc"),
            units="M",
            status="citable",
            paper_section="methods",
        )

    fine_pairs = (
        (
            "linear_l0",
            "linear_l0_lc0.040 vs linear_l0_lc0.028",
        ),
        (
            "mexhat_l0",
            "mexhat_l0_lc0.040 vs mexhat_l0_lc0.028",
        ),
        (
            "linear_l2",
            "linear_l2_lc0.040 vs linear_l2_lc0.028",
        ),
        (
            "mexhat_l2",
            "mexhat_l2_lc0.040 vs mexhat_l2_lc0.028",
        ),
    )
    for label, pair_key in fine_pairs:
        table.scalar(
            f"interior_fine_ladder_rms_{label}",
            source=_source(
                PRODUCTION,
                f"/ladder/{label}/diff_rms_over_scale/{pair_key}",
            ),
            transform="fraction-to-percent",
            units="% de escala",
            status="citable-con-caveat",
            caveat=(
                "Diferencia RMS normalizada entre los dos rungs finos; es un "
                "diagnóstico de malla, no una barra de error estadística."
            ),
            paper_section="interior",
        )


def _add_frozen_discriminator(table: NumberTable) -> None:
    pairs = (
        "l0_lc0.056", "l0_lc0.040", "l0_lc0.028",
        "l1_lc0.040", "l2_lc0.040", "l2_lc0.028",
    )
    for pair in pairs:
        base = f"/discriminator/{pair}"
        l_gt0 = pair.startswith(("l1_", "l2_"))
        floor_caveat = (
            "Congelado sin corrección: el piso TRUTH_SCAN l>0 es mayor "
            "que el efecto; véanse o1_truth_floor_linear_l1/l2."
            if l_gt0 else ""
        )
        table.scalar(
            f"disc_{pair}_l2_ratio_frozen",
            source=_source(PRODUCTION, f"{base}/l2_ratio"),
            units="1", status="citable", paper_section="interior",
            caveat=floor_caveat,
        )
        table.scalar(
            f"disc_{pair}_ratio_median_frozen",
            source=_source(PRODUCTION, f"{base}/ratio_median"),
            uncertainty=_iqr(
                _source(PRODUCTION, f"{base}/ratio_iqr/0"),
                _source(PRODUCTION, f"{base}/ratio_iqr/1"),
            ),
            units="1", status="citable", paper_section="interior",
            caveat=floor_caveat,
        )
        table.scalar(
            f"disc_{pair}_peak_ratio_frozen",
            source=_source(PRODUCTION, f"{base}/peak_ratio"),
            units="1", status="no-citable", paper_section="interior",
            caveat=(
                "Compara máximos en tiempos distintos y hereda el sesgo "
                "diferencial de ventana; mecanismo revisado en HANDOFF §7."
            ),
        )

    oracle = "/discriminator/oracle_1d_l0"
    table.scalar(
        "disc_oracle_1d_l0_l2_ratio",
        source=_source(PRODUCTION, f"{oracle}/l2_ratio"),
        units="1", status="citable", paper_section="interior",
    )
    table.scalar(
        "disc_oracle_1d_l0_ratio_median",
        source=_source(PRODUCTION, f"{oracle}/ratio_median"),
        units="1", status="citable", paper_section="interior",
    )
    table.scalar(
        "disc_oracle_1d_l0_peak_ratio",
        source=_source(PRODUCTION, f"{oracle}/peak_ratio"),
        units="1", status="no-citable", paper_section="interior",
        caveat="El cociente de picos no es el discriminador primario.",
    )


def _add_production_runs(table: NumberTable) -> None:
    runs = (
        "linear_l0_lc0.056", "linear_l0_lc0.040", "linear_l0_lc0.028",
        "linear_l1_lc0.040", "linear_l2_lc0.040", "linear_l2_lc0.028",
        "mexhat_l0_lc0.056", "mexhat_l0_lc0.040", "mexhat_l0_lc0.028",
        "mexhat_l1_lc0.040", "mexhat_l2_lc0.040", "mexhat_l2_lc0.028",
    )
    production = table.document(PRODUCTION)
    actual_runs = set(production.get("runs", {}))
    if actual_runs != set(runs):
        raise ValueError(
            "production run inventory changed: "
            f"expected {sorted(runs)}, found {sorted(actual_runs)}"
        )

    for run in runs:
        base = f"/runs/{run}"
        coarse = run.endswith("l0_lc0.056")
        for estimator in ("primary", "anchor"):
            dev_key = f"dev_vs_1d_{estimator}"
            if not _has_pointer(table, PRODUCTION, f"{base}/{dev_key}"):
                continue
            for statistic in ("median", "max"):
                status = "no-citable" if coarse else "citable-con-caveat"
                if coarse:
                    caveat = "Rung fuera del régimen de convergencia."
                elif estimator == "primary":
                    caveat = (
                        "Comparación 3D↔1D con piso de interpolación y "
                        "extracción; no es una incertidumbre puramente de malla."
                    )
                else:
                    caveat = (
                        "El ancla o0 es un diagnóstico de fase y conserva "
                        "sesgo de truncamiento radial."
                    )
                table.scalar(
                    f"{run}_dev_{estimator}_{statistic}",
                    source=_source(PRODUCTION, f"{base}/{dev_key}/{statistic}"),
                    transform="fraction-to-percent", units="% de escala 1D",
                    status=status, caveat=caveat, paper_section="interior",
                )

        table.scalar(
            f"{run}_sigma_a_median_primary",
            source=_source(PRODUCTION, f"{base}/sigma_a_median_primary"),
            units="a", status="citable-con-caveat",
            caveat="Error OLS indicativo; el residuo radial está correlacionado.",
            paper_section="methods",
        )
        table.scalar(
            f"{run}_cond_median_primary",
            source=_source(PRODUCTION, f"{base}/cond_median_primary"),
            units="1", status="citable", paper_section="methods",
        )
        table.scalar(
            f"{run}_killing_residual_rel",
            source=_source(PRODUCTION, f"{base}/killing_residual_rel"),
            transform="fraction-to-percent", units="%",
            status="citable-con-caveat",
            caveat=(
                "Dominado por la cuadratura del flujo en la excisión facetada; "
                "no es métrica de calidad del campo."
            ),
            paper_section="methods",
        )
        table.scalar(
            f"{run}_cowling_zeta_max_global",
            source=_source(PRODUCTION, f"{base}/cowling_zeta_max_global"),
            units="1", status="citable-con-caveat",
            caveat=(
                "Monitor global dominado por el exterior de curvatura débil; "
                "no es comparable con el ζ interior de F0."
            ),
            paper_section="discusión",
        )

        off_channels = _resolve_pointer(production, f"{base}/offchannel_peak_over_signal")
        if not isinstance(off_channels, dict):
            raise TypeError(f"{run}: off-channel record is not an object")
        for channel in sorted(off_channels):
            table.scalar(
                f"{run}_junk_offchannel_{channel}",
                source=_source(
                    PRODUCTION, f"{base}/offchannel_peak_over_signal/{channel}"
                ),
                transform="fraction-to-percent", units="% de señal del modo",
                status="citable-con-caveat",
                caveat=(
                    "Cota en fase fuerte con tope temporal; el exceso sobre "
                    "el canal lineal, no el valor aislado, limita acoplamiento."
                ),
                paper_section="interior",
            )

    table.scalar(
        "mass_lumping_ab_dev_median",
        source=_source(PRODUCTION, "/mass_lumping_ab/dev_median"),
        transform="fraction-to-percent", units="% de escala",
        status="citable", paper_section="methods",
    )
    table.scalar(
        "mass_lumping_ab_dev_max",
        source=_source(PRODUCTION, "/mass_lumping_ab/dev_max"),
        transform="fraction-to-percent", units="% de escala",
        status="citable", paper_section="methods",
    )
    table.scalar(
        "mass_lumping_adopted",
        source=_source(PRODUCTION, "/mass_lumping_ab/adopted"),
        units="boolean", status="citable", paper_section="methods",
    )
    table.scalar(
        "mass_lumping_consistent_smoke_wall",
        source=_source(PRODUCTION, "/mass_lumping_ab/wall_consistent_smoke_s"),
        units="s", status="citable-con-caveat",
        caveat="Costo de la corrida consistente reusada, no de la matriz completa.",
        paper_section="methods",
    )
    table.scalar(
        "production_cached_postprocess_wall",
        source=_source(PRODUCTION, "/total_wall_seconds"),
        units="s", status="no-citable",
        caveat="Mide el postproceso con corridas cacheadas, no el costo de evolución.",
        paper_section="methods",
    )


def _add_calibration(table: NumberTable) -> None:
    status = table.resolve(_source(CALIBRATION, "/status"))
    stop_required = table.resolve(_source(CALIBRATION, "/stop_required"))
    reviewer = table.resolve(_source(CALIBRATION, "/review/by"))
    if status != "reviewed-diagnostic" or stop_required is not False:
        raise ValueError(
            "o1 calibration has not been released by the §7 review: "
            f"status={status!r}, stop_required={stop_required!r}"
        )
    if reviewer != "Fable":
        raise ValueError(f"unexpected o1 calibration reviewer {reviewer!r}")

    for profile in ("linear_l0", "mexhat_l0"):
        headline = f"/headlines/c_{profile}_sampled_k32"
        table.scalar(
            f"o1_c_{profile}_sampled_k32_median",
            source=_source(CALIBRATION, f"{headline}/median"),
            uncertainty=_iqr(
                _source(CALIBRATION, f"{headline}/iqr/0"),
                _source(CALIBRATION, f"{headline}/iqr/1"),
            ),
            units="1", status="citable-con-caveat",
            caveat="Cociente definido sólo en la fase fuerte del oráculo.",
            paper_section="interior",
            support_source=_source(CALIBRATION, f"{headline}/n"),
        )
        table.scalar(
            f"o1_c_{profile}_max_abs_deviation",
            source=_source(
                CALIBRATION, f"{headline}/max_abs_deviation_from_one"
            ),
            transform="fraction-to-percent", units="%",
            status="citable-con-caveat",
            caveat="Máximo condicionado a la máscara fuerte; no se rellena fuera.",
            paper_section="interior",
        )
        table.scalar(
            f"o1_{profile}_truth_scan_floor",
            source=_source(
                CALIBRATION, f"/calibrations/{profile}/truth_scan_floor_max"
            ),
            transform="fraction-to-percent", units="%",
            status="citable-con-caveat",
            caveat="Piso al mover la ventana de verdad profunda.",
            paper_section="methods",
        )
        table.scalar(
            f"o1_{profile}_resolution_stability_max",
            source=_source(
                CALIBRATION,
                f"/calibrations/{profile}/resolution_stability/"
                "sampled_k32/max_relative",
            ),
            transform="fraction-to-percent", units="%",
            status="citable-con-caveat",
            caveat="Check de resolución del cociente, no error de la evolución 3D.",
            paper_section="methods",
        )
        table.scalar(
            f"o1_{profile}_sampling_k32_effect_max",
            source=_source(
                CALIBRATION, f"/calibrations/{profile}/sampling_effect/max_relative"
            ),
            transform="fraction-to-percent", units="%",
            status="citable-con-caveat",
            caveat="Diferencia entre ventana densa y los radios K del banco 3D.",
            paper_section="methods",
        )

    for profile in ("linear_l1", "linear_l2"):
        table.scalar(
            f"o1_truth_floor_{profile}",
            source=_source(
                CALIBRATION, f"/calibrations/{profile}/truth_scan_floor_max"
            ),
            transform="fraction-to-percent", units="%",
            status="citable-con-caveat",
            caveat="Piso mayor que el efecto a corregir; impide promover l>0.",
            paper_section="interior",
        )

    for lc in ("0.056", "0.040", "0.028"):
        pair = f"l0_lc{lc}"
        for potential in ("linear", "mexhat"):
            run = f"{potential}_l0_lc{lc}"
            dev = f"/runs/{run}/variants/sampled_k32/dev"
            common_caveat = (
                "Análisis de sistemática sobre la intersección estricta de "
                "fases fuertes; no reemplaza el dev congelado."
            )
            table.scalar(
                f"o1_{run}_dev_fraction_explained",
                source=_source(
                    CALIBRATION, f"{dev}/median_dev_fraction_explained"
                ),
                transform="fraction-to-percent", units="% del dev mediano",
                status="citable-con-caveat", caveat=common_caveat,
                paper_section="interior",
                support_source=_source(CALIBRATION, f"{dev}/n_calibration_support"),
            )
            table.scalar(
                f"o1_{run}_dev_before_common_support",
                source=_source(
                    CALIBRATION, f"{dev}/before_on_calibration_support/median"
                ),
                transform="fraction-to-percent", units="% de escala 1D",
                status="citable-con-caveat", caveat=common_caveat,
                paper_section="interior",
                support_source=_source(CALIBRATION, f"{dev}/n_calibration_support"),
            )
            table.scalar(
                f"o1_{run}_dev_residual",
                source=_source(
                    CALIBRATION, f"{dev}/after_on_calibration_support/median"
                ),
                uncertainty=_budget((
                    "piso_calibración",
                    _source(CALIBRATION, "/verdict/calibration_floor_max"),
                    "fraction-to-percent",
                )),
                transform="fraction-to-percent", units="% de escala 1D",
                status="citable-con-caveat", caveat=common_caveat,
                paper_section="interior",
                support_source=_source(CALIBRATION, f"{dev}/n_calibration_support"),
            )

        disc = f"/discriminator/{pair}/variants/sampled_k32"
        table.scalar(
            f"o1_disc_{pair}_l2_before_common_support",
            source=_source(CALIBRATION, f"{disc}/before_on_calibration_support/l2_ratio"),
            units="1", status="no-citable",
            caveat=(
                "Diagnóstico sobre soporte reducido; el número canónico sigue "
                "siendo el discriminador congelado."
            ),
            paper_section="interior",
            support_source=_source(
                CALIBRATION, f"{disc}/before_on_calibration_support/n_samples"
            ),
        )
        table.scalar(
            f"o1_disc_{pair}_l2_corrected_diagnostic",
            source=_source(CALIBRATION, f"{disc}/after/l2_ratio"),
            uncertainty=_budget((
                "piso_calibración",
                _source(CALIBRATION, "/verdict/calibration_floor_max"),
                "identity",
            )),
            units="1", status="no-citable",
            caveat=(
                "Diagnóstico de sistemática sobre soporte reducido; no reemplaza "
                "el l2_ratio congelado."
            ),
            paper_section="interior",
            support_source=_source(CALIBRATION, f"{disc}/after/n_samples"),
        )

    table.scalar(
        "o1_deficit_classification",
        source=_source(CALIBRATION, "/verdict/classification"),
        units="categoría", status="citable-con-caveat",
        caveat="Veredicto de calibración, no una corrección sustituta del dato F2.",
        paper_section="discusión",
    )
    table.scalar(
        "o1_deficit_fraction_explained",
        source=_source(CALIBRATION, "/verdict/deficit_fraction_explained"),
        transform="fraction-to-percent", units="% del déficit L2",
        status="citable-con-caveat",
        caveat="Estimado sobre soporte común de calibración.",
        paper_section="discusión",
    )
    table.scalar(
        "o1_remaining_l2_deficit",
        source=_source(CALIBRATION, "/verdict/deficit_after_absolute"),
        uncertainty=_budget((
            "piso_calibración",
            _source(CALIBRATION, "/verdict/calibration_floor_max"),
            "fraction-to-percent",
        )),
        transform="fraction-to-percent", units="puntos absolutos",
        status="citable-con-caveat",
        caveat="Residual frente al oráculo sobre soporte común.",
        paper_section="discusión",
    )


def _add_exterior(table: NumberTable) -> None:
    protocol_specs = (
        ("exterior_l", "/protocol/l", "1"),
        ("exterior_m", "/protocol/m", "1"),
        ("exterior_r_ext", "/protocol/r_ext", "M"),
        ("exterior_pulse_A", "/protocol/pulse/A", "1"),
        ("exterior_pulse_r0", "/protocol/pulse/r0", "M"),
        ("exterior_pulse_w", "/protocol/pulse/w", "M"),
        (
            "exterior_designed_window_length",
            "/protocol/windows/window",
            "M",
        ),
        ("exterior_designed_window_t_search", "/protocol/windows/t_search", "M"),
        (
            "exterior_designed_window_mode_count",
            "/protocol/windows/prony_modes",
            "modos",
        ),
        ("exterior_tail_tmin", "/protocol/tail/t_min", "M"),
        ("exterior_tail_floor_tmin", "/protocol/tail/floor_t_min", "M"),
        (
            "exterior_late_window_length",
            "/protocol/late_windows/window",
            "M",
        ),
        (
            "exterior_late_window_min_offset",
            "/protocol/late_windows/pooled_min_offset",
            "M",
        ),
    )
    for entry_id, pointer, units in protocol_specs:
        table.scalar(
            entry_id,
            source=_source(EXTERIOR, pointer),
            units=units,
            status="citable",
            paper_section="methods",
        )

    table.scalar(
        "qnm_leaver_l2_re",
        source=_source(EXTERIOR, "/leaver_l2/0"),
        units="Mω", status="citable", paper_section="exterior",
    )
    table.scalar(
        "qnm_leaver_l2_minus_im",
        source=_source(EXTERIOR, "/leaver_l2/1"), transform="absolute",
        units="Mω", status="citable", paper_section="exterior",
    )

    for index, label in enumerate(("lc1.4", "lc1.0", "lc0.7")):
        rung = f"/ladder_R20/rungs/{index}"
        table.scalar(
            f"qnm_R20_{label}_re",
            source=_source(EXTERIOR, f"{rung}/omega_re"),
            uncertainty=_symmetric(_source(EXTERIOR, f"{rung}/fan_std_re")),
            units="Mω", status="citable-con-caveat",
            caveat="Abanico de ventanas ancladas; la incertidumbre es scatter de fit.",
            paper_section="exterior",
        )
        table.scalar(
            f"qnm_R20_{label}_re_error_signed",
            source=_source(EXTERIOR, f"{rung}/err_re_signed"),
            transform="fraction-to-percent", units="% vs Leaver",
            status="citable-con-caveat",
            caveat="Error firmado del mismo rung y protocolo de ventanas.",
            paper_section="exterior",
        )
    table.scalar(
        "qnm_R20_fine_re_order",
        source=_source(EXTERIOR, "/ladder_R20/order_vs_leaver_re/1/p"),
        units="orden p", status="citable-con-caveat",
        caveat="Orden de la pareja fina; Richardson de tres puntos no aplica.",
        paper_section="exterior",
    )

    pooled = "/late_pooled_all_R40"
    table.scalar(
        "qnm_R40_late_pooled_re",
        source=_source(EXTERIOR, f"{pooled}/omega_re"),
        uncertainty=_symmetric(_source(EXTERIOR, f"{pooled}/omega_re_std")),
        units="Mω", status="citable-con-caveat",
        caveat="Scatter de ventanas tardías y dos rungs, dominado por protocolo.",
        paper_section="exterior",
        support_source=_source(EXTERIOR, f"{pooled}/n_windows"),
    )
    table.scalar(
        "qnm_R40_late_pooled_re_error_signed",
        source=_source(EXTERIOR, f"{pooled}/err_re_signed"),
        transform="fraction-to-percent", units="% vs Leaver",
        status="citable-con-caveat",
        caveat="Error firmado del pooled tardío.", paper_section="exterior",
    )
    table.scalar(
        "qnm_R40_late_pooled_minus_im",
        source=_source(EXTERIOR, f"{pooled}/omega_im_neg"),
        uncertainty=_symmetric(_source(EXTERIOR, f"{pooled}/omega_im_neg_std")),
        units="Mω", status="citable-con-caveat",
        caveat="Scatter de ventanas tardías y dos rungs, dominado por protocolo.",
        paper_section="exterior",
        support_source=_source(EXTERIOR, f"{pooled}/n_windows"),
    )
    table.scalar(
        "qnm_R40_late_pooled_minus_im_error_signed",
        source=_source(EXTERIOR, f"{pooled}/err_im_signed"),
        transform="fraction-to-percent", units="% vs Leaver",
        status="citable-con-caveat",
        caveat="Error firmado del pooled tardío.", paper_section="exterior",
    )

    for index, label in enumerate(("lc1match", "lc0.7match")):
        table.scalar(
            f"qnm_R40_{label}_early_overtone_bias",
            source=_source(
                EXTERIOR, f"/ladder_R40_clean/rungs/{index}/err_im_signed"
            ),
            transform="fraction-to-percent", units="% en −Im vs Leaver",
            status="citable-con-caveat",
            caveat=(
                "Sesgo temprano del protocolo por overtone no separable; no es "
                "estimación del fundamental tardío."
            ),
            paper_section="exterior",
        )

    for key in ("R40_lc1match", "R40_lc0.7match"):
        check = f"/domain_checks/{key}"
        table.scalar(
            f"domain_{key}_tail_floor_reduction",
            source=_source(EXTERIOR, f"{check}/tail_floor_ratio_R40_over_R20"),
            transform="reciprocal", units="× reducción R20/R40",
            status="citable-con-caveat",
            caveat="Inverso determinista del cociente de suelos almacenado.",
            paper_section="exterior",
        )
        for domain in ("R20", "R40"):
            table.scalar(
                f"domain_{key}_{domain}_peak_over_floor",
                source=_source(EXTERIOR, f"{check}/peak_over_floor/{domain}"),
                units="1", status="citable-con-caveat",
                caveat="Diagnóstico del suelo de cavidad del dominio.",
                paper_section="exterior",
            )
    table.scalar(
        "domain_R40_lc1match_delta_re",
        source=_source(EXTERIOR, "/domain_checks/R40_lc1match/delta_omega_re"),
        uncertainty=_symmetric(_source(
            EXTERIOR, "/domain_checks/R40_lc1match/fan_scatter_re"
        )),
        units="Mω", status="citable-con-caveat",
        caveat="Par con graduación apareada; el scatter del rung es la cota.",
        paper_section="exterior",
    )

    for run in ("R20_lc0.7", "R40_lc1match", "R40_lc0.7match"):
        table.scalar(
            f"exterior_{run}_wall",
            source=_source(EXTERIOR, f"/runs/{run}/wall_seconds"),
            units="s", status="citable", paper_section="methods",
        )


def _add_cavity_and_phase0(table: NumberTable) -> None:
    for run in ("lc1.4", "lc1"):
        for line in ("w1", "w2"):
            dw = "dw1" if line == "w1" else "dw2"
            base = f"/runs/{run}/tail_lines"
            table.scalar(
                f"cavity_{run}_{line}",
                source=_source(CAVITY, f"{base}/{line}"),
                uncertainty=_symmetric(_source(CAVITY, f"{base}/{dw}")),
                units="Mω", status="citable-con-caveat",
                caveat="Modo atrapado del dominio R20, no QNM físico.",
                paper_section="exterior",
            )

    table.scalar(
        "cowling_phase0_zeta_max_inside_horizon",
        source=_source(PHASE0, "/cowling_interior/zeta_max_inside_horizon"),
        units="1", status="citable-con-caveat",
        caveat="Piloto Higgs de máxima amplitud; monitor restringido al interior.",
        paper_section="discusión",
    )
    table.scalar(
        "cowling_phase0_zeta_at_rmin",
        source=_source(PHASE0, "/cowling_interior/zeta_at_rmin"),
        units="1", status="citable-con-caveat",
        caveat="Valor del piloto en el radio mínimo, no máximo global.",
        paper_section="discusión",
    )


def _add_note_only(table: NumberTable) -> None:
    promotion = "El valor no existe como scalar en un JSON versionado."
    demoted = (
        "RESUELTO por el orquestador (2026-07-12, HANDOFF §7): degradado a "
        "prosa — el manuscrito lo cita desde la nota; no requiere fila "
        "canónica."
    )
    table.note_only(
        "mass_lumping_f1_speedup",
        units="×", paper_section="methods", caveat=promotion,
        resolution=demoted + (
            " El número de decisión del rechazo (Δa00 28 %/39 %) sí es "
            "canónico: production.json::mass_lumping_ab."
        ),
        note_pointer=(
            "docs/research/plan.md::§3.1, bullet ‘Mass lumping HECHO’"
        ),
    )
    for entry_id in (
        "convergence_physical_p1_coarse",
        "convergence_physical_p1_fine",
        "convergence_physical_p2_fine",
    ):
        table.note_only(
            entry_id, units="orden p", paper_section="methods",
            caveat=promotion,
            resolution=demoted,
            note_pointer=(
                "docs/research/phase1/convergence/note.md::Resultado principal, "
                "tabla ‘tránsito+ring 0–30M’"
            ),
        )
    table.note_only(
        "dissipation_epsilon_max",
        units="1", paper_section="methods", caveat=promotion,
        resolution=demoted + (
            " El paper cita la REGLA (orden 4, ε ≤ 0.5·ε_max, guard del "
            "solver), no el ε_max de la malla del estudio."
        ),
        note_pointer=(
            "docs/research/phase1/dissipation/note.md::§1 ‘Qué pasó con el "
            "barrido original’"
        ),
    )
    for line in ("w1", "w2"):
        table.note_only(
            f"cavity_doublet_headline_{line}",
            units="Mω", paper_section="exterior",
            caveat=(
                "El agregado central y su incertidumbre no son una clave scalar; "
                "las mediciones por rung sí están trazadas arriba."
            ),
            resolution=demoted + (
                " Camino de promoción si C4 lo tabula: re-ajustar las waveforms "
                "versionadas de phase1/cavity con "
                "rsd.analysis.ringdown.fit_tail_lines y emitir un JSON dedicado."
            ),
            note_pointer=(
                "docs/research/phase1/cavity/note.md::§1 ‘Corrección previa’ + "
                "tests/test_cavity_mode_slow.py::docstring/canary bands"
            ),
        )
    for entry_id, section in (
        ("production_matrix_wall_pool", "apertura"),
        ("production_linear_run_wall_range", "§6 ‘Salud y costos’"),
        ("production_mexhat_run_wall_range", "§6 ‘Salud y costos’"),
    ):
        table.note_only(
            entry_id, units="min", paper_section="methods",
            caveat=(
                "Los campos walls_s del JSON reflejan reuso/cache y no contienen "
                "este costo de evolución."
            ),
            resolution=demoted + (
                " Costo histórico de la sesión original; walls_s quedó en 0.0 "
                "tras el re-análisis y no es regenerable."
            ),
            note_pointer=(
                f"docs/research/phase2/production/note.md::{section}"
            ),
        )


def build_table(repo: Path = REPO) -> NumberTable:
    table = NumberTable(repo)
    _add_production_protocol(table)
    _add_frozen_discriminator(table)
    _add_production_runs(table)
    _add_calibration(table)
    _add_exterior(table)
    _add_cavity_and_phase0(table)
    _add_note_only(table)
    table.validate_provenance()
    return table


def _format_number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    magnitude = abs(float(value))
    if magnitude != 0.0 and (magnitude < 1.0e-3 or magnitude >= 1.0e4):
        return f"{float(value):.6e}"
    return f"{float(value):.6g}"


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "sí" if value else "no"
    if isinstance(value, (int, float)):
        return _format_number(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _format_uncertainty(uncertainty: dict[str, Any] | None) -> str:
    if uncertainty is None:
        return "—"
    kind = uncertainty["kind"]
    if kind == "symmetric":
        return f"± {_format_value(uncertainty['value'])}"
    if kind == "IQR":
        return (
            f"IQR [{_format_value(uncertainty['low'])}, "
            f"{_format_value(uncertainty['high'])}]"
        )
    if kind == "budget":
        return "; ".join(
            f"{part['label']}={_format_value(part['value'])}"
            for part in uncertainty["components"]
        )
    raise ValueError(f"unknown rendered uncertainty kind {kind!r}")


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _entry_lookup(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {entry["id"]: entry for entry in entries}


def _interpretation(entries: list[dict[str, Any]]) -> list[str]:
    by_id = _entry_lookup(entries)
    lin = by_id["o1_c_linear_l0_sampled_k32_median"]
    hat = by_id["o1_c_mexhat_l0_sampled_k32_median"]
    explained = by_id["o1_deficit_fraction_explained"]
    remaining = by_id["o1_remaining_l2_deficit"]
    frozen = by_id["disc_l0_lc0.028_l2_ratio_frozen"]
    corrected = by_id["o1_disc_l0_lc0.028_l2_corrected_diagnostic"]
    floor_l1 = by_id["o1_truth_floor_linear_l1"]
    floor_l2 = by_id["o1_truth_floor_linear_l2"]
    return [
        "## Lectura de la calibración o1",
        "",
        (
            "- En fase fuerte, la mediana de c(t) K=32 es "
            f"{_format_value(lin['value'])} para linear l=0 y "
            f"{_format_value(hat['value'])} para mexhat l=0; los IQR y la "
            "procedencia completa están en la tabla."
        ),
        (
            "- La calibración explica "
            f"{_format_value(explained['value'])} {explained['units']} y deja "
            f"{_format_value(remaining['value'])} {remaining['units']} en el "
            "soporte común: veredicto parcial."
        ),
        (
            "- El l2_ratio fino canónico permanece congelado en "
            f"{_format_value(frozen['value'])}. El valor calibrado "
            f"{_format_value(corrected['value'])} es sólo diagnóstico sobre "
            f"n={corrected['support']['n']} y no lo reemplaza."
        ),
        (
            "- Los pares l>0 quedan sin corregir: sus pisos TRUTH_SCAN son "
            f"{_format_value(floor_l1['value'])} {floor_l1['units']} y "
            f"{_format_value(floor_l2['value'])} {floor_l2['units']}, mayores "
            "que el efecto que se intentaría corregir."
        ),
        "",
    ]


def render_markdown(payload: dict[str, Any]) -> str:
    entries = payload["entries"]
    lines = [
        "# Números canónicos para el paper",
        "",
        "Generado por `scripts/paper_numbers.py`. Cada valor JSON-backed y "
        "cada incertidumbre se resuelven desde la procedencia indicada; las "
        "filas note-only permanecen nulas hasta promoción explícita.",
        "",
    ]
    lines.extend(_interpretation(entries))
    for section in ("intro", "methods", "interior", "exterior", "discusión"):
        selected = [entry for entry in entries if entry["paper_section"] == section]
        if not selected:
            continue
        lines.extend([
            f"## {section.capitalize()}",
            "",
            "| id | valor | incertidumbre | unidades | estado | procedencia | caveat |",
            "|---|---:|---|---|---|---|---|",
        ])
        for entry in selected:
            provenance = entry["source"]
            if entry["source"] == "note-only":
                provenance = f"note-only; {entry['note_pointer']}"
            support = entry.get("support")
            caveat = entry["caveat"]
            if support:
                support_text = f"soporte n={support['n']} ({support['source']})"
                caveat = f"{caveat} {support_text}".strip()
            lines.append(
                "| " + " | ".join(
                    _escape_markdown(item) for item in (
                        entry["id"],
                        _format_value(entry["value"]),
                        _format_uncertainty(entry["uncertainty"]),
                        entry["units"],
                        entry["status"],
                        provenance,
                        caveat or "—",
                    )
                ) + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


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
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--check", action="store_true",
        help="valida que los outputs existentes coincidan, sin escribir",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    table = build_table(REPO)
    status_counts = {
        status: sum(entry["status"] == status for entry in table.entries)
        for status in sorted(STATUSES)
    }
    payload = {
        "schema_version": 1,
        "generated_by": "scripts/paper_numbers.py",
        "source_syntax": "repo-relative-file::/RFC6901/JSON/Pointer",
        "status_counts": status_counts,
        "entries": table.entries,
    }
    json_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    markdown_text = render_markdown(payload)

    if args.check:
        expected = ((args.json_output, json_text), (args.markdown_output, markdown_text))
        stale = [
            path for path, content in expected
            if not path.is_file() or path.read_text(encoding="utf-8") != content
        ]
        if stale:
            print("outputs desactualizados: " + ", ".join(map(str, stale)), file=sys.stderr)
            return 1
        print(f"numbers: {len(table.entries)} entradas verificadas; outputs al día")
        return 0

    changed_json = _atomic_write(args.json_output, json_text)
    changed_md = _atomic_write(args.markdown_output, markdown_text)
    print(
        f"numbers: {len(table.entries)} entradas verificadas → "
        f"{args.json_output.relative_to(REPO)}, "
        f"{args.markdown_output.relative_to(REPO)} "
        f"({'actualizados' if changed_json or changed_md else 'sin cambios'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
