#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C2 — calibración por perfil del estimador o1 interior.

Este postproceso lee, sin regenerar nada, las referencias 1D y las series 3D
congeladas de F2. Mide

    c_X(t) = a_o1[0.1,0.5](t) / a_truth_o1[0.02,0.2](t)

en la fase fuerte del oráculo, tanto sobre la malla 1D densa como sobre los
K=32 radios reales del banco 3D. Luego divide por c(t) a ambos miembros de
cada par antes de recomputar dev y discriminadores.

No invoca ``interior_production.py`` ni escribe bajo phase0/phase1/phase2.
Los únicos outputs son ``docs/research/phase3/o1_calibration.json`` y dos
figuras bajo ``docs/research/phase3/figures``.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from rsd.analysis.interior import (  # noqa: E402
    fit_log_profile_multipole,
    fit_log_profile_series,
)

PRIMARY_WINDOW = (0.1, 0.5)
PRIMARY_ORDER = 1
ANCHOR_WINDOW = (0.1, 0.5)
ANCHOR_ORDER = 0
TRUTH_WINDOW = (0.02, 0.2)
TRUTH_SCAN = ((0.015, 0.15), (0.03, 0.3))
CALIBRATION_T_MIN = 4.0
PRODUCTION_T_MIN = 4.0
PRODUCTION_T_MAX = 10.0
STRONG_FRACTION = 0.3
TRANSFER_THRESHOLD = 0.03
DISCRIMINATOR_REVIEW_THRESHOLD = 0.02
DISCRIMINATOR_STOP_THRESHOLD = 0.03
SQRT4PI = float(np.sqrt(4.0 * np.pi))
REVIEW = {
    "by": "Fable",
    "date": "2026-07-12",
    "log": "§7",
    "resolution": (
        "peak_ratio alarm explained by differential window bias at distinct "
        "argmax times; keep as reviewed diagnostic, do not promote peak_ratio"
    ),
}

VARIANTS = ("continuous", "sampled_k32")
DISCRIMINATOR_FIELDS = ("ratio_median", "peak_ratio", "l2_ratio")

REFERENCE_SPECS = {
    "linear_l0": {
        "potential": "linear",
        "l": 0,
        "resolution": 1600,
        "path": "docs/research/phase2/interior/data/"
        "ab_smoke_ref_linear_A0.1_n1600.npz",
    },
    "mexhat_l0": {
        "potential": "mexhat",
        "l": 0,
        "resolution": 1600,
        "path": "docs/research/phase2/interior/data/"
        "ab_smoke_ref_mexhat_A0.1_n1600.npz",
    },
    "linear_l1": {
        "potential": "linear",
        "l": 1,
        "resolution": 1600,
        "path": "docs/research/phase2/production/data/"
        "prod_ref_linear_l1_A0.1_n1600.npz",
    },
    "linear_l2": {
        "potential": "linear",
        "l": 2,
        "resolution": 2600,
        "path": "docs/research/phase2/production/data/"
        "prod_ref_linear_l2_A0.1_n2600.npz",
    },
}

RESOLUTION_CHECK_SPECS = {
    "linear_l0": {
        "potential": "linear",
        "l": 0,
        "resolution": 800,
        "path": "results/phase2_production_fast/data/"
        "prod_ref_linear_l0_A0.1_n800.npz",
    },
    "mexhat_l0": {
        "potential": "mexhat",
        "l": 0,
        "resolution": 800,
        "path": "results/phase2_production_fast/data/"
        "prod_ref_mexhat_l0_A0.1_n800.npz",
    },
}

FROZEN_PRODUCTION = "docs/research/phase2/production/production.json"
DEFAULT_OUTPUT = "docs/research/phase3/o1_calibration.json"
DEFAULT_FIGURE_DIR = "docs/research/phase3/figures"


def _relative(path: Path, repo: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_npz(path: Path, required: Iterable[str]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(required) - set(data.files))
        if missing:
            raise KeyError(f"{path}: missing NPZ keys {missing}")
        return {key: np.asarray(data[key]) for key in data.files}


def _validate_reference(data: dict[str, np.ndarray], path: Path) -> None:
    r = np.asarray(data["r"], dtype=float)
    ts = np.asarray(data["snapshot_ts"], dtype=float)
    snaps = np.asarray(data["snapshots_u"], dtype=float)
    if r.ndim != 1 or ts.ndim != 1 or snaps.shape != (ts.size, r.size):
        raise ValueError(
            f"{path}: expected r (nr,), snapshot_ts (nt,), snapshots_u "
            f"(nt,nr); got {r.shape}, {ts.shape}, {snaps.shape}"
        )
    if np.any(r <= 0.0) or np.any(np.diff(r) <= 0.0):
        raise ValueError(f"{path}: radii must be positive and strictly increasing")
    if np.any(np.diff(ts) <= 0.0):
        raise ValueError(f"{path}: snapshot_ts must be strictly increasing")
    if not np.all(np.isfinite(snaps)):
        raise ValueError(f"{path}: snapshots_u contains non-finite values")


def _validate_run(data: dict[str, np.ndarray], path: Path, l: int) -> None:
    t = np.asarray(data["t"], dtype=float)
    radii = np.asarray(data["radii"], dtype=float)
    modes = np.asarray(data["modes"])
    u = np.asarray(data["u"], dtype=float)
    if t.ndim != 1 or radii.ndim != 1 or modes.ndim != 2 or modes.shape[1] != 2:
        raise ValueError(f"{path}: malformed t/radii/modes arrays")
    if u.shape != (t.size, radii.size, modes.shape[0]):
        raise ValueError(f"{path}: u has incompatible shape {u.shape}")
    if radii.size != 32:
        raise ValueError(f"{path}: expected K=32 radii, found {radii.size}")
    if np.any(np.diff(t) <= 0.0) or np.any(np.diff(radii) <= 0.0):
        raise ValueError(f"{path}: t and radii must be strictly increasing")
    if not any(int(lv) == l and int(m) == 0 for lv, m in modes):
        raise ValueError(f"{path}: target channel ({l}, 0) is absent")
    if not np.all(np.isfinite(u)):
        raise ValueError(f"{path}: u contains non-finite values")


def calibration_strong_mask(ts: np.ndarray, a_truth: np.ndarray) -> np.ndarray:
    """Fase fuerte de la calibración 1D, idéntica al estudio de ventanas."""
    ts = np.asarray(ts, dtype=float)
    a_truth = np.asarray(a_truth, dtype=float)
    eligible = ts >= CALIBRATION_T_MIN
    if not np.any(eligible):
        raise ValueError("calibration has no snapshots at t >= 4M")
    scale = float(np.max(np.abs(a_truth[eligible])))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("calibration truth has zero/non-finite strong scale")
    strong = eligible & (np.abs(a_truth) >= STRONG_FRACTION * scale)
    if not np.any(strong):
        raise ValueError("calibration strong phase is empty")
    return strong


def production_strong_mask(t: np.ndarray, a_anchor: np.ndarray) -> np.ndarray:
    """Máscara congelada de producción: o0, 4<=t<=10 y fracción 0.3."""
    t = np.asarray(t, dtype=float)
    a_anchor = np.asarray(a_anchor, dtype=float)
    window = (t >= PRODUCTION_T_MIN) & (t <= PRODUCTION_T_MAX)
    if not np.any(window):
        raise ValueError("run has no samples in the production strong window")
    scale = float(np.max(np.abs(a_anchor[window])))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("run anchor has zero/non-finite strong scale")
    strong = window & (np.abs(a_anchor) >= STRONG_FRACTION * scale)
    if not np.any(strong):
        raise ValueError("run strong phase is empty")
    return strong


def sample_snapshots_at_radii(
    r_dense: np.ndarray,
    snapshots: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """Muestrea cada snapshot en los radios provistos; nunca los re-deriva."""
    r_dense = np.asarray(r_dense, dtype=float)
    snapshots = np.asarray(snapshots, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if radii[0] < r_dense[0] or radii[-1] > r_dense[-1]:
        raise ValueError("K-bank radii fall outside the dense oracle grid")
    return np.vstack([np.interp(radii, r_dense, profile) for profile in snapshots])


def _ratio_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("cannot summarize an empty/non-finite ratio")
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "iqr": [float(np.percentile(values, 25)), float(np.percentile(values, 75))],
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "max_abs_deviation_from_one": float(np.max(np.abs(values - 1.0))),
    }


def compute_profile_calibration(
    data: dict[str, np.ndarray],
    bank_radii: np.ndarray,
) -> dict[str, Any]:
    """Calcula deep truth y c(t) continuo/K=32 para una referencia."""
    r = np.asarray(data["r"], dtype=float)
    ts = np.asarray(data["snapshot_ts"], dtype=float)
    snapshots = np.asarray(data["snapshots_u"], dtype=float)

    truth = fit_log_profile_series(r, snapshots, TRUTH_WINDOW, order=1)["a"]
    continuous = fit_log_profile_series(
        r, snapshots, PRIMARY_WINDOW, order=PRIMARY_ORDER
    )["a"]
    sampled_profiles = sample_snapshots_at_radii(r, snapshots, bank_radii)
    sampled = fit_log_profile_series(
        bank_radii, sampled_profiles, PRIMARY_WINDOW, order=PRIMARY_ORDER
    )["a"]
    strong = calibration_strong_mask(ts, truth)

    variants: dict[str, dict[str, Any]] = {}
    for name, a_window in (("continuous", continuous), ("sampled_k32", sampled)):
        c = np.full(ts.shape, np.nan, dtype=float)
        if np.any(np.abs(truth[strong]) <= np.finfo(float).tiny):
            raise ValueError("a_truth is zero inside the strong phase")
        c[strong] = a_window[strong] / truth[strong]
        if not np.all(np.isfinite(c[strong])):
            raise ValueError(f"{name}: non-finite c(t) in strong phase")
        variants[name] = {
            "a_window": a_window,
            "c": c,
            "summary": _ratio_summary(c[strong]),
        }

    truth_scans: dict[str, Any] = {}
    truth_floor = 0.0
    for window in TRUTH_SCAN:
        alt_truth = fit_log_profile_series(r, snapshots, window, order=1)["a"]
        truth_rel = np.abs(alt_truth[strong] - truth[strong]) / np.abs(truth[strong])
        truth_floor = max(truth_floor, float(np.max(truth_rel)))
        scan_variants = {}
        for name, variant in variants.items():
            if np.any(np.abs(alt_truth[strong]) <= np.finfo(float).tiny):
                raise ValueError(f"truth scan {window} crosses zero in baseline strong phase")
            c_alt = np.full(ts.shape, np.nan, dtype=float)
            c_alt[strong] = variant["a_window"][strong] / alt_truth[strong]
            rel = np.abs(c_alt[strong] / variant["c"][strong] - 1.0)
            scan_variants[name] = {
                "c": c_alt,
                "relative_to_baseline": {
                    "median": float(np.median(rel)),
                    "max": float(np.max(rel)),
                },
            }
        truth_scans[f"[{window[0]:g},{window[1]:g}]"] = {
            "window": list(window),
            "a_truth": alt_truth,
            "truth_relative_to_baseline": {
                "median": float(np.median(truth_rel)),
                "max": float(np.max(truth_rel)),
            },
            "variants": scan_variants,
        }

    sampling_rel = np.abs(
        variants["sampled_k32"]["c"][strong]
        / variants["continuous"]["c"][strong]
        - 1.0
    )
    return {
        "t": ts,
        "a_truth": truth,
        "strong_mask": strong,
        "truth_scale_after_tmin": float(np.max(np.abs(truth[ts >= CALIBRATION_T_MIN]))),
        "truth_scan_floor_max": truth_floor,
        "variants": variants,
        "truth_scans": truth_scans,
        "sampling_effect": {
            "median_relative": float(np.median(sampling_rel)),
            "max_relative": float(np.max(sampling_rel)),
        },
    }


def interpolate_valid_curve(
    source_t: np.ndarray,
    source_values: np.ndarray,
    source_valid: np.ndarray,
    target_t: np.ndarray,
) -> np.ndarray:
    """Interpola solo dentro de segmentos contiguos válidos, sin rellenar.

    ``np.interp`` extrapola constantes y cruzaría huecos. Aquí cada bloque
    consecutivo de la máscara se interpola por separado; fuera queda NaN.
    """
    source_t = np.asarray(source_t, dtype=float)
    source_values = np.asarray(source_values, dtype=float)
    source_valid = np.asarray(source_valid, dtype=bool)
    target_t = np.asarray(target_t, dtype=float)
    if not (source_t.ndim == source_values.ndim == source_valid.ndim == 1):
        raise ValueError("source curve arrays must be one-dimensional")
    if not (source_t.size == source_values.size == source_valid.size):
        raise ValueError("source curve arrays must have equal length")
    if np.any(np.diff(source_t) <= 0.0):
        raise ValueError("source_t must be strictly increasing")

    valid = source_valid & np.isfinite(source_values)
    indices = np.flatnonzero(valid)
    out = np.full(target_t.shape, np.nan, dtype=float)
    if indices.size == 0:
        return out
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    for group in groups:
        if group.size == 1:
            exact = np.isclose(target_t, source_t[group[0]], rtol=0.0, atol=1e-12)
            out[exact] = source_values[group[0]]
            continue
        inside = (target_t >= source_t[group[0]]) & (target_t <= source_t[group[-1]])
        out[inside] = np.interp(
            target_t[inside], source_t[group], source_values[group]
        )
    return out


def compare_calibration_curves(
    left: dict[str, Any],
    right: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    """Compara c_right/c_left en el soporte fuerte común."""
    left_t = np.asarray(left["t"], dtype=float)
    left_c = np.asarray(left["variants"][variant]["c"], dtype=float)
    left_valid = np.asarray(left["strong_mask"], dtype=bool)
    right_on_left = interpolate_valid_curve(
        right["t"],
        right["variants"][variant]["c"],
        right["strong_mask"],
        left_t,
    )
    common = left_valid & np.isfinite(left_c) & np.isfinite(right_on_left)
    if not np.any(common):
        return {
            "n_common": 0,
            "median_relative": None,
            "max_relative": None,
            "p95_relative": None,
        }
    rel = np.abs(right_on_left[common] / left_c[common] - 1.0)
    return {
        "n_common": int(np.count_nonzero(common)),
        "median_relative": float(np.median(rel)),
        "max_relative": float(np.max(rel)),
        "p95_relative": float(np.percentile(rel, 95)),
    }


def _fit_run(data: dict[str, np.ndarray], l: int) -> dict[str, Any]:
    modes = [tuple(int(v) for v in mode) for mode in data["modes"]]
    fits_primary = fit_log_profile_multipole(
        data["radii"], data["u"], modes, PRIMARY_WINDOW, order=PRIMARY_ORDER
    )[(l, 0)]
    fits_anchor = fit_log_profile_multipole(
        data["radii"], data["u"], modes, ANCHOR_WINDOW, order=ANCHOR_ORDER
    )[(l, 0)]
    t = np.asarray(data["t"], dtype=float)
    return {
        "t": t,
        "a_primary": np.asarray(fits_primary["a"], dtype=float),
        "a_anchor": np.asarray(fits_anchor["a"], dtype=float),
        "strong_mask": production_strong_mask(t, fits_anchor["a"]),
    }


def _deviation_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("cannot summarize empty/non-finite deviations")
    return {
        "n": int(values.size),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


def compute_dev_metrics(
    run: dict[str, Any],
    calibration: dict[str, Any],
    variant: str,
    normalization: float,
) -> dict[str, Any]:
    """Dev antes/después sobre soporte común, con definición congelada."""
    t = np.asarray(run["t"], dtype=float)
    a_raw = np.asarray(run["a_primary"], dtype=float)
    strong = np.asarray(run["strong_mask"], dtype=bool)
    truth_t = np.asarray(calibration["t"], dtype=float)
    truth = normalization * np.asarray(calibration["a_truth"], dtype=float)
    scale = float(np.max(np.abs(truth)))
    full = strong & (t >= truth_t[0]) & (t <= truth_t[-1])
    truth_on_run = np.interp(t, truth_t, truth)
    before_full = _deviation_summary(np.abs(a_raw[full] - truth_on_run[full]) / scale)

    c_on_run = interpolate_valid_curve(
        truth_t,
        calibration["variants"][variant]["c"],
        calibration["strong_mask"],
        t,
    )
    support = full & np.isfinite(c_on_run) & (np.abs(c_on_run) > np.finfo(float).tiny)
    if not np.any(support):
        raise ValueError(f"{variant}: calibration has no overlap with run strong phase")
    a_corrected = np.full(a_raw.shape, np.nan)
    a_corrected[support] = a_raw[support] / c_on_run[support]
    before_support = _deviation_summary(
        np.abs(a_raw[support] - truth_on_run[support]) / scale
    )
    after_support = _deviation_summary(
        np.abs(a_corrected[support] - truth_on_run[support]) / scale
    )
    explained = (
        (before_support["median"] - after_support["median"])
        / before_support["median"]
        if before_support["median"] > 0.0
        else None
    )
    return {
        "scale_1d": scale,
        "before_full_support": before_full,
        "before_on_calibration_support": before_support,
        "after_on_calibration_support": after_support,
        "median_dev_fraction_explained": float(explained) if explained is not None else None,
        "n_calibration_support": int(np.count_nonzero(support)),
    }


def _metrics_from_common(
    a_linear: np.ndarray,
    a_mexhat: np.ndarray,
    common: np.ndarray,
) -> dict[str, Any]:
    a_linear = np.asarray(a_linear, dtype=float)
    a_mexhat = np.asarray(a_mexhat, dtype=float)
    common = np.asarray(common, dtype=bool)
    if not np.any(common):
        raise ValueError("discriminator common strong phase is empty")
    if np.any(np.abs(a_linear[common]) <= np.finfo(float).tiny):
        raise ValueError("discriminator denominator crosses exact zero")
    ratio = a_mexhat[common] / a_linear[common]
    denom = float(np.sum(a_linear[common] ** 2))
    if denom <= 0.0:
        raise ValueError("discriminator L2 denominator is zero")
    return {
        "n_samples": int(np.count_nonzero(common)),
        "ratio_median": float(np.median(ratio)),
        "ratio_iqr": [
            float(np.percentile(ratio, 25)),
            float(np.percentile(ratio, 75)),
        ],
        "peak_ratio": float(
            np.max(np.abs(a_mexhat[common])) / np.max(np.abs(a_linear[common]))
        ),
        "l2_ratio": float(np.sqrt(np.sum(a_mexhat[common] ** 2) / denom)),
    }


def raw_discriminator(linear: dict[str, Any], mexhat: dict[str, Any]) -> dict[str, Any]:
    """Definición exacta de production.py::discriminator."""
    tl = np.asarray(linear["t"], dtype=float)
    th = np.asarray(mexhat["t"], dtype=float)
    a_l = np.asarray(linear["a_primary"], dtype=float)
    a_h = np.interp(tl, th, mexhat["a_primary"])
    s_h = np.interp(tl, th, np.asarray(mexhat["strong_mask"], dtype=float)) > 0.5
    common = np.asarray(linear["strong_mask"], dtype=bool) & s_h
    return _metrics_from_common(a_l, a_h, common)


def corrected_discriminator(
    linear: dict[str, Any],
    mexhat: dict[str, Any],
    calibration_linear: dict[str, Any],
    calibration_mexhat: dict[str, Any],
    variant: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Corrige ambos miembros en sus tiempos nativos y luego recomputa."""
    tl = np.asarray(linear["t"], dtype=float)
    th = np.asarray(mexhat["t"], dtype=float)
    c_l = interpolate_valid_curve(
        calibration_linear["t"],
        calibration_linear["variants"][variant]["c"],
        calibration_linear["strong_mask"],
        tl,
    )
    c_h = interpolate_valid_curve(
        calibration_mexhat["t"],
        calibration_mexhat["variants"][variant]["c"],
        calibration_mexhat["strong_mask"],
        th,
    )
    valid_l = np.isfinite(c_l) & (np.abs(c_l) > np.finfo(float).tiny)
    valid_h = np.isfinite(c_h) & (np.abs(c_h) > np.finfo(float).tiny)
    a_l_corr = np.full(tl.shape, np.nan)
    a_h_corr = np.full(th.shape, np.nan)
    a_l_corr[valid_l] = np.asarray(linear["a_primary"])[valid_l] / c_l[valid_l]
    a_h_corr[valid_h] = np.asarray(mexhat["a_primary"])[valid_h] / c_h[valid_h]
    a_h_corr_on_l = interpolate_valid_curve(th, a_h_corr, valid_h, tl)

    s_h = np.interp(tl, th, np.asarray(mexhat["strong_mask"], dtype=float)) > 0.5
    common = (
        np.asarray(linear["strong_mask"], dtype=bool)
        & s_h
        & valid_l
        & np.isfinite(a_h_corr_on_l)
    )
    after = _metrics_from_common(a_l_corr, a_h_corr_on_l, common)
    raw_h_on_l = np.interp(tl, th, mexhat["a_primary"])
    before_same_support = _metrics_from_common(linear["a_primary"], raw_h_on_l, common)
    return before_same_support, after


def discriminator_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    absolute = {key: float(after[key] - before[key]) for key in DISCRIMINATOR_FIELDS}
    relative = {
        key: float(absolute[key] / before[key]) if before[key] != 0.0 else None
        for key in DISCRIMINATOR_FIELDS
    }
    max_abs = max(abs(value) for value in absolute.values())
    return {
        "absolute": absolute,
        "relative": relative,
        "max_absolute": float(max_abs),
        "review": bool(max_abs > DISCRIMINATOR_REVIEW_THRESHOLD),
        "alarm": bool(max_abs > DISCRIMINATOR_STOP_THRESHOLD),
    }


def _assert_frozen_metrics(
    computed: dict[str, Any],
    frozen: dict[str, Any],
    context: str,
    atol: float = 5e-12,
) -> None:
    for field in DISCRIMINATOR_FIELDS:
        if not np.isclose(computed[field], frozen[field], rtol=0.0, atol=atol):
            raise RuntimeError(
                f"{context}: recomputed {field}={computed[field]} does not match "
                f"frozen value {frozen[field]}"
            )
    if not np.allclose(computed["ratio_iqr"], frozen["ratio_iqr"], rtol=0.0, atol=atol):
        raise RuntimeError(f"{context}: recomputed ratio_iqr does not match frozen value")


def _assert_frozen_dev(
    computed: dict[str, Any],
    frozen: dict[str, Any],
    context: str,
    atol: float = 5e-12,
) -> None:
    for field in ("max", "median"):
        if not np.isclose(computed[field], frozen[field], rtol=0.0, atol=atol):
            raise RuntimeError(
                f"{context}: recomputed dev {field}={computed[field]} does not "
                f"match frozen value {frozen[field]}"
            )


def _resolve_one(pattern: Path) -> Path:
    hits = [Path(hit) for hit in sorted(glob.glob(str(pattern)))]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one match for {pattern}, found {hits}")
    return hits[0]


def resolve_run_series(repo: Path, matrix: list[dict[str, Any]]) -> dict[str, Path]:
    """Resuelve las 12 series canónicas, incluido el fallback l0@0.040."""
    if len(matrix) != 12:
        raise ValueError(f"frozen production matrix must have 12 runs, found {len(matrix)}")
    paths: dict[str, Path] = {}
    for spec in matrix:
        label = str(spec["label"])
        if int(spec["l"]) == 0 and np.isclose(float(spec["lc"]), 0.04):
            pot = str(spec["pot"])
            pattern = (
                repo
                / "results"
                / "phase2_interior_ab"
                / f"run_{pot}"
                / "run_*"
                / "series"
                / "interior_profiles.npz"
            )
        else:
            pattern = (
                repo
                / "results"
                / "phase2_production"
                / f"run_{label}"
                / "run_*"
                / "series"
                / "interior_profiles.npz"
            )
        paths[label] = _resolve_one(pattern)
    return paths


def _serialize_curve(values: np.ndarray, valid: np.ndarray) -> list[float | None]:
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    return [float(value) if keep else None for value, keep in zip(values, valid)]


def serialize_calibration(calibration: dict[str, Any]) -> dict[str, Any]:
    strong = np.asarray(calibration["strong_mask"], dtype=bool)
    out: dict[str, Any] = {
        "t": np.asarray(calibration["t"], dtype=float).tolist(),
        "a_truth": np.asarray(calibration["a_truth"], dtype=float).tolist(),
        "strong_mask": strong.tolist(),
        "n_strong": int(np.count_nonzero(strong)),
        "truth_scale_after_tmin": calibration["truth_scale_after_tmin"],
        "truth_scan_floor_max": calibration["truth_scan_floor_max"],
        "sampling_effect": calibration["sampling_effect"],
        "variants": {},
        "truth_scans": {},
    }
    for name, variant in calibration["variants"].items():
        out["variants"][name] = {
            "a_window": np.asarray(variant["a_window"], dtype=float).tolist(),
            "c": _serialize_curve(variant["c"], strong),
            "summary": variant["summary"],
        }
    for key, scan in calibration["truth_scans"].items():
        out_scan = {
            "window": scan["window"],
            "a_truth": np.asarray(scan["a_truth"], dtype=float).tolist(),
            "truth_relative_to_baseline": scan["truth_relative_to_baseline"],
            "variants": {},
        }
        for name, variant in scan["variants"].items():
            out_scan["variants"][name] = {
                "c": _serialize_curve(variant["c"], strong),
                "relative_to_baseline": variant["relative_to_baseline"],
            }
        out["truth_scans"][key] = out_scan
    return out


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)


def _make_figures(
    calibrations: dict[str, dict[str, Any]],
    run_results: dict[str, Any],
    figure_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    names = ["linear_l0", "mexhat_l0", "linear_l1", "linear_l2"]
    titles = {
        "linear_l0": "linear l=0",
        "mexhat_l0": "mexhat l=0",
        "linear_l1": "linear l=1",
        "linear_l2": "linear l=2",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharex=True)
    for ax, name in zip(axes.ravel(), names):
        calibration = calibrations[name]
        t = np.asarray(calibration["t"])
        strong = np.asarray(calibration["strong_mask"])
        for variant, style, label in (
            ("continuous", "C0-", "ventana continua"),
            ("sampled_k32", "C3--", "muestreo K=32"),
        ):
            curve = np.where(strong, calibration["variants"][variant]["c"], np.nan)
            ax.plot(t, curve, style, lw=1.5, label=label)
        ax.axhline(1.0, color="0.5", lw=0.8)
        ax.set_title(titles[name])
        ax.set_ylabel(r"$c_X(t)=a_{\rm win}/a_{\rm truth}$")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("t/M")
    fig.suptitle("C2: calibración o1 por perfil (solo fase fuerte)")
    fig.tight_layout()
    fig.savefig(figure_dir / "o1_calibration_c.png", dpi=150)
    plt.close(fig)

    labels = []
    before = []
    after = []
    for label, result in run_results.items():
        dev = result.get("variants", {}).get("sampled_k32", {}).get("dev")
        if not dev:
            continue
        labels.append(label.replace("_lc", "\nlc"))
        before.append(dev["before_on_calibration_support"]["median"])
        after.append(dev["after_on_calibration_support"]["median"])
    x = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(max(10, 1.15 * len(labels)), 5.3))
    ax.bar(x - 0.19, before, width=0.38, color="0.55", label="antes")
    ax.bar(x + 0.19, after, width=0.38, color="C2", label="después (K=32)")
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("dev mediana vs deep truth")
    ax.set_title("C2: dev 3D antes/después sobre soporte común")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "o1_calibration_dev.png", dpi=150)
    plt.close(fig)


def _deficit_verdict(
    frozen: dict[str, Any],
    discriminator_results: dict[str, Any],
    calibrations: dict[str, Any],
    resolution_stability: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    fine_pair = discriminator_results["l0_lc0.028"]
    fine = fine_pair["variants"]["sampled_k32"]
    oracle_l2 = float(frozen["discriminator"]["oracle_1d_l0"]["l2_ratio"])
    before_frozen_l2 = float(fine_pair["before_frozen"]["l2_ratio"])
    before_l2 = float(fine["before_on_calibration_support"]["l2_ratio"])
    after_l2 = float(fine["after"]["l2_ratio"])
    deficit_before = oracle_l2 - before_l2
    deficit_before_frozen = oracle_l2 - before_frozen_l2
    deficit_after = oracle_l2 - after_l2
    fraction = (
        (deficit_before - deficit_after) / deficit_before
        if abs(deficit_before) > np.finfo(float).tiny
        else None
    )

    floors = []
    for name in ("linear_l0", "mexhat_l0"):
        calibration = calibrations[name]
        floors.append(float(calibration["truth_scan_floor_max"]))
        floors.append(
            max(
                float(calibration["truth_scans"][key]["variants"]["sampled_k32"]
                      ["relative_to_baseline"]["max"])
                for key in calibration["truth_scans"]
            )
        )
        stability = resolution_stability[name]["sampled_k32"]["max_relative"]
        if stability is not None:
            floors.append(float(stability))
    floor = max(floors)
    if abs(deficit_after) <= floor or (fraction is not None and fraction >= 0.8):
        classification = "total"
    elif fraction is not None and fraction > 0.0 and abs(deficit_after) < abs(deficit_before):
        classification = "partial"
    else:
        classification = "no"

    verdict = {
        "classification": classification,
        "oracle_l2_ratio": oracle_l2,
        "fine_l0_l2_ratio_before_frozen": before_frozen_l2,
        "fine_l0_l2_ratio_before": before_l2,
        "fine_l0_l2_ratio_after": after_l2,
        "deficit_before_frozen_absolute": float(deficit_before_frozen),
        "deficit_before_absolute": float(deficit_before),
        "deficit_after_absolute": float(deficit_after),
        "deficit_fraction_explained": float(fraction) if fraction is not None else None,
        "calibration_floor_max": float(floor),
    }
    budget = {
        "calibration_floor_max": float(floor),
        "remaining_l2_deficit_absolute": float(abs(deficit_after)),
        "fine_l0_ladder_rms_linear": float(
            frozen["ladder"]["linear_l0"]["diff_rms_over_scale"]
            ["linear_l0_lc0.040 vs linear_l0_lc0.028"]
        ),
        "fine_l0_ladder_rms_mexhat": float(
            frozen["ladder"]["mexhat_l0"]["diff_rms_over_scale"]
            ["mexhat_l0_lc0.040 vs mexhat_l0_lc0.028"]
        ),
    }
    return verdict, budget


def build_calibration(repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construye el payload JSON y devuelve además estado interno para figuras."""
    frozen_path = repo / FROZEN_PRODUCTION
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    matrix = frozen["protocol"]["matrix"]
    run_paths = resolve_run_series(repo, matrix)

    # Carga las 12 series canónicas y valida el banco contra el A/B excluido.
    run_data: dict[str, dict[str, np.ndarray]] = {}
    run_specs = {spec["label"]: spec for spec in matrix}
    for label, path in run_paths.items():
        data = _load_npz(path, ("t", "radii", "modes", "u"))
        _validate_run(data, path, int(run_specs[label]["l"]))
        run_data[label] = data

    radii_source_label = "linear_l0_lc0.028"
    bank_radii = np.asarray(run_data[radii_source_label]["radii"], dtype=float)
    radii_checked = dict(run_paths)
    ml_path = _resolve_one(
        repo
        / "results"
        / "phase2_production"
        / "run_linear_l0_lc0.040_ml"
        / "run_*"
        / "series"
        / "interior_profiles.npz"
    )
    radii_checked["linear_l0_lc0.040_ml_EXCLUDED"] = ml_path
    for label, path in radii_checked.items():
        data = _load_npz(path, ("radii",))
        if not np.allclose(data["radii"], bank_radii, rtol=0.0, atol=1e-14):
            raise ValueError(f"{label}: K-bank radii differ from {radii_source_label}")

    references: dict[str, dict[str, np.ndarray]] = {}
    calibrations: dict[str, dict[str, Any]] = {}
    input_references: dict[str, Any] = {}
    for name, spec in REFERENCE_SPECS.items():
        path = repo / spec["path"]
        data = _load_npz(path, ("r", "snapshot_ts", "snapshots_u"))
        _validate_reference(data, path)
        references[name] = data
        calibrations[name] = compute_profile_calibration(data, bank_radii)
        input_references[name] = {
            **spec,
            "sha256": _sha256(path),
            "keys": ["r", "snapshot_ts", "snapshots_u"],
            "n_snapshots": int(data["snapshot_ts"].size),
        }

    resolution_stability: dict[str, Any] = {}
    resolution_inputs: dict[str, Any] = {}
    for name, spec in RESOLUTION_CHECK_SPECS.items():
        path = repo / spec["path"]
        data = _load_npz(path, ("r", "snapshot_ts", "snapshots_u"))
        _validate_reference(data, path)
        low = compute_profile_calibration(data, bank_radii)
        resolution_stability[name] = {
            variant: compare_calibration_curves(calibrations[name], low, variant)
            for variant in VARIANTS
        }
        resolution_inputs[name] = {
            **spec,
            "sha256": _sha256(path),
            "role": "resolution-check-only; not a production rung",
            "calibration": serialize_calibration(low),
        }

    transfer_gate: dict[str, Any] = {}
    for variant in VARIANTS:
        comparison = compare_calibration_curves(
            calibrations["linear_l0"], calibrations["mexhat_l0"], variant
        )
        transfer_gate[variant] = {
            **comparison,
            "threshold": TRANSFER_THRESHOLD,
            "applied_to_l_gt0": False,
            "moot": True,
            "reason": "own-truth TRUTH_SCAN floor ≥ effect",
        }

    fitted_runs = {
        label: _fit_run(run_data[label], int(run_specs[label]["l"]))
        for label in run_data
    }
    run_results: dict[str, Any] = {}
    for label, spec in run_specs.items():
        pot = str(spec["pot"])
        l = int(spec["l"])
        exact_calibration_name = f"{pot}_l{l}"
        run_result: dict[str, Any] = {
            "potential": pot,
            "l": l,
            "lc_inner": float(spec["lc"]),
            "source": _relative(run_paths[label], repo),
            "variants": {},
        }
        for variant in VARIANTS:
            if l == 0:
                correction_name = exact_calibration_name
                applied = True
                reason = "exact l=0 reference for this potential"
            else:
                correction_name = f"linear_l{l}"
                applied = False
                reason = (
                    "own-truth TRUTH_SCAN floor ≥ effect"
                    if pot == "linear"
                    else (
                        "no exact mexhat oracle; linear own-truth TRUTH_SCAN floor "
                        "≥ effect; frozen discriminator retained"
                    )
                )
            variant_result: dict[str, Any] = {
                "applied": applied,
                "correction_source": correction_name if applied else None,
                "reason": reason,
            }
            if applied and exact_calibration_name in calibrations:
                dev = compute_dev_metrics(
                    fitted_runs[label],
                    calibrations[correction_name],
                    variant,
                    SQRT4PI if l == 0 else 1.0,
                )
                frozen_dev = frozen["runs"][label]["dev_vs_1d_primary"]
                _assert_frozen_dev(dev["before_full_support"], frozen_dev, label)
                variant_result["dev"] = dev
                variant_result["dev"]["before_frozen_source"] = (
                    f"{FROZEN_PRODUCTION}::/runs/{label}/dev_vs_1d_primary"
                )
            run_result["variants"][variant] = variant_result
        run_results[label] = run_result

    discriminator_results: dict[str, Any] = {}
    for key, frozen_disc in frozen["discriminator"].items():
        if key == "oracle_1d_l0":
            continue
        left_label = f"linear_{key.replace('_lc', '_lc')}"
        right_label = f"mexhat_{key.replace('_lc', '_lc')}"
        # key is l0_lc0.028; the prefix construction above is intentional.
        if left_label not in fitted_runs or right_label not in fitted_runs:
            raise KeyError(f"cannot resolve discriminator pair {key}")
        l = int(key.split("_", 1)[0][1:])
        before = raw_discriminator(fitted_runs[left_label], fitted_runs[right_label])
        _assert_frozen_metrics(before, frozen_disc, key)
        pair_result: dict[str, Any] = {
            "linear_run": left_label,
            "mexhat_run": right_label,
            "before_frozen": frozen_disc,
            "before_frozen_source": f"{FROZEN_PRODUCTION}::/discriminator/{key}",
            "variants": {},
        }
        for variant in VARIANTS:
            if l == 0:
                cal_l = calibrations["linear_l0"]
                cal_h = calibrations["mexhat_l0"]
                applied = True
                reason = "exact potential-specific l=0 corrections"
            else:
                applied = False
                cal_l = calibrations[f"linear_l{l}"]
                cal_h = cal_l
                reason = (
                    "no exact mexhat oracle; linear own-truth TRUTH_SCAN floor "
                    "≥ effect; frozen discriminator retained"
                )
            if not applied:
                pair_result["variants"][variant] = {
                    "applied": False,
                    "reason": reason,
                    "after": None,
                    "delta": None,
                }
                continue
            before_support, after = corrected_discriminator(
                fitted_runs[left_label], fitted_runs[right_label], cal_l, cal_h, variant
            )
            delta = discriminator_delta(before_support, after)
            delta_vs_frozen = discriminator_delta(frozen_disc, after)
            pair_result["variants"][variant] = {
                "applied": True,
                "reason": reason,
                "before_on_calibration_support": before_support,
                "after": after,
                "delta": delta,
                "delta_vs_frozen": delta_vs_frozen,
            }
        discriminator_results[key] = pair_result

    verdict, error_budget = _deficit_verdict(
        frozen, discriminator_results, calibrations, resolution_stability
    )
    all_deltas = [
        variant_result["delta"]
        for pair in discriminator_results.values()
        for variant_result in pair["variants"].values()
        if variant_result["applied"]
    ]
    threshold_review_triggered = any(delta["review"] for delta in all_deltas)
    threshold_stop_triggered = any(delta["alarm"] for delta in all_deltas)
    max_movement = max(delta["max_absolute"] for delta in all_deltas)
    error_budget["discriminator_movement_max_absolute"] = float(max_movement)

    fine_linear_dev = run_results["linear_l0_lc0.028"]["variants"]["sampled_k32"]["dev"]
    fine_mexhat_dev = run_results["mexhat_l0_lc0.028"]["variants"]["sampled_k32"]["dev"]
    fine_disc_pair = discriminator_results["l0_lc0.028"]
    fine_disc = fine_disc_pair["variants"]["sampled_k32"]
    headlines = {
        "c_linear_l0_sampled_k32": calibrations["linear_l0"]["variants"]
        ["sampled_k32"]["summary"],
        "c_mexhat_l0_sampled_k32": calibrations["mexhat_l0"]["variants"]
        ["sampled_k32"]["summary"],
        "l0_transfer_gate_sampled_k32": transfer_gate["sampled_k32"],
        "fine_l0_dev_linear": {
            "before_median": fine_linear_dev["before_on_calibration_support"]["median"],
            "after_median": fine_linear_dev["after_on_calibration_support"]["median"],
            "fraction_explained": fine_linear_dev["median_dev_fraction_explained"],
        },
        "fine_l0_dev_mexhat": {
            "before_median": fine_mexhat_dev["before_on_calibration_support"]["median"],
            "after_median": fine_mexhat_dev["after_on_calibration_support"]["median"],
            "fraction_explained": fine_mexhat_dev["median_dev_fraction_explained"],
        },
        "fine_l0_discriminator_sampled_k32": {
            "before_frozen": fine_disc_pair["before_frozen"],
            "before_on_calibration_support": fine_disc["before_on_calibration_support"],
            "after": fine_disc["after"],
            "delta": fine_disc["delta"],
        },
        "deficit_verdict": verdict,
        "error_budget": error_budget,
    }

    inputs_runs = {
        label: {
            "path": _relative(path, repo),
            "sha256": _sha256(path),
            "source_kind": (
                "phase2-smoke-fallback"
                if "phase2_interior_ab" in path.as_posix()
                else "phase2-production"
            ),
        }
        for label, path in run_paths.items()
    }
    payload = {
        "schema_version": 1,
        "status": "reviewed-diagnostic",
        "stop_required": False,
        "review": REVIEW,
        "protocol": {
            "truth": {"window": list(TRUTH_WINDOW), "order": 1},
            "truth_scan": [list(window) for window in TRUTH_SCAN],
            "primary": {"window": list(PRIMARY_WINDOW), "order": PRIMARY_ORDER},
            "anchor": {"window": list(ANCHOR_WINDOW), "order": ANCHOR_ORDER},
            "calibration_strong": {
                "t_min": CALIBRATION_T_MIN,
                "strong_fraction": STRONG_FRACTION,
                "definition": "t>=4 and |a_truth|>=0.3*max_{t>=4}|a_truth|",
            },
            "production_strong": {
                "t_min": PRODUCTION_T_MIN,
                "t_max": PRODUCTION_T_MAX,
                "strong_fraction": STRONG_FRACTION,
                "anchor": "o0 [0.1,0.5]",
            },
            "sampled_variant": "snapshots interpolated only to K=32 radii read from 3D",
            "time_interpolation": (
                "numpy linear interpolation within contiguous strong segments; "
                "no extrapolation or filling outside strong phase"
            ),
            "l_gt0_transfer_gate": {
                "status": "moot for C2 because own-truth TRUTH_SCAN floor ≥ effect",
                "future_rule": (
                    "median ≤3% and p95 ≤5% on common support after excluding "
                    "two points adjacent to every segment edge"
                ),
            },
            "discriminator_review_threshold_absolute": DISCRIMINATOR_REVIEW_THRESHOLD,
            "discriminator_stop_threshold_absolute": DISCRIMINATOR_STOP_THRESHOLD,
            "normalization": "sqrt(4pi) for l=0 truth dev; unity for l>=1; c invariant",
        },
        "inputs": {
            "frozen_production": {
                "path": FROZEN_PRODUCTION,
                "sha256": _sha256(frozen_path),
            },
            "references": input_references,
            "resolution_checks": resolution_inputs,
            "run_series": inputs_runs,
            "excluded_mass_lumped_series": {
                "path": _relative(ml_path, repo),
                "sha256": _sha256(ml_path),
                "reason": "mass lumping rejected in frozen F2 production",
            },
            "radii_source": {
                "run": radii_source_label,
                "path": _relative(run_paths[radii_source_label], repo),
                "k": int(bank_radii.size),
                "values": bank_radii.tolist(),
                "all_13_series_identical": True,
            },
        },
        "calibrations": {
            name: {
                "source": spec["path"],
                "potential": spec["potential"],
                "l": spec["l"],
                "resolution": spec["resolution"],
                **serialize_calibration(calibrations[name]),
                "resolution_stability": resolution_stability.get(name),
            }
            for name, spec in REFERENCE_SPECS.items()
        },
        "l_gt0_transfer_gate": transfer_gate,
        "runs": run_results,
        "discriminator": discriminator_results,
        "headlines": headlines,
        "verdict": {
            **verdict,
            "error_budget": error_budget,
            "review_required": False,
            "stop_required": False,
            "diagnostic_review_triggered": threshold_review_triggered,
            "diagnostic_alarm_triggered": threshold_stop_triggered,
            "review_resolution": REVIEW["resolution"],
        },
    }
    figure_state = {"calibrations": calibrations, "runs": run_results}
    return payload, figure_state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    output = (args.output or (repo / DEFAULT_OUTPUT)).resolve()
    figure_dir = (args.figure_dir or (repo / DEFAULT_FIGURE_DIR)).resolve()
    frozen_root = (repo / "docs" / "research" / "phase2").resolve()
    if output == frozen_root or frozen_root in output.parents:
        raise ValueError("C2 output may not be written under frozen phase2")
    if not args.no_figures and (figure_dir == frozen_root or frozen_root in figure_dir.parents):
        raise ValueError("C2 figures may not be written under frozen phase2")

    print("[1/4] inventario y referencias congeladas...", flush=True)
    payload, state = build_calibration(repo)
    print("[2/4] escritura estricta de o1_calibration.json...", flush=True)
    _write_json_atomic(output, payload)
    if not args.no_figures:
        print("[3/4] figuras c(t) y dev antes/después...", flush=True)
        _make_figures(state["calibrations"], state["runs"], figure_dir)
    else:
        print("[3/4] figuras omitidas por --no-figures", flush=True)

    verdict = payload["verdict"]
    print("[4/4] veredicto", flush=True)
    print(
        f"  déficit L2: {verdict['classification']}; "
        f"antes(mismo soporte)={verdict['fine_l0_l2_ratio_before']:.6f}, "
        f"después={verdict['fine_l0_l2_ratio_after']:.6f}; "
        f"movimiento máx={verdict['error_budget']['discriminator_movement_max_absolute']:.3%}",
        flush=True,
    )
    print(f"  JSON: {output}", flush=True)
    if verdict["stop_required"]:
        print("  ALARMA: movimiento >3%; detener C2 y pedir revisión.", flush=True)
        return 2
    if verdict["review_required"]:
        print("  REVIEW: movimiento entre 2% y 3%; no promover sin revisión.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
