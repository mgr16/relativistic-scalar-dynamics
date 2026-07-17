#!/usr/bin/env python3
"""R2 sensitivity grid for the production one-dimensional discriminator.

The nonlinear member is evolved at every (lambda, A) point.  The linear
member is obtained by exact amplitude scaling of the frozen A=0.1 production
reference: the zero-potential oracle is linear, so this avoids eleven
scientifically redundant evolutions without changing its discretization.

Raw nonlinear profiles are cached under ``results/`` during the long scan and
assembled into one new versioned NPZ.  ``--check`` never evolves or writes; it
recomputes every fit, strong mask, L2 discriminator, and propagated OLS error
from that NPZ before comparing the canonical JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from o1_profile_calibration import (  # noqa: E402
    ANCHOR_ORDER,
    ANCHOR_WINDOW,
    PRIMARY_ORDER,
    PRIMARY_WINDOW,
    production_strong_mask,
    raw_discriminator,
)
from rsd.analysis.interior import fit_log_profile_series  # noqa: E402


LAMBDAS = (0.03, 0.1, 0.3, 1.0)
AMPLITUDES = (0.01, 0.03, 0.1)
VACUUM_VALUE = 1.0
N_POINTS = 1600
R_MIN = 0.01
R_MAX = 60.0
T_END = 15.0
SNAPSHOT_COUNT_TARGET = 80
KO_EPS = 0.02
PULSE_R0 = 5.0
PULSE_WIDTH = 1.0
PULSE_DIRECTION = "ingoing_curved"
DISCOVERY_THRESHOLD = 0.15

LINEAR_REFERENCE = REPO / (
    "docs/research/phase2/interior/data/"
    "ab_smoke_ref_linear_A0.1_n1600.npz"
)
ENERGY_SPLIT_REFERENCE = REPO / "docs/research/phase3/data/energy_split_profiles.npz"
DEFAULT_DATA = REPO / "docs/research/phase3/data/sensitivity_scan_profiles.npz"
DEFAULT_JSON = REPO / "docs/research/phase3/sensitivity_scan.json"
DEFAULT_CACHE = REPO / "results/phase3_sensitivity_scan/cache"


def _token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _cell_id(lambda_coupling: float, amplitude: float) -> str:
    return f"lambda_{_token(lambda_coupling)}_A_{_token(amplitude)}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _validate_profiles(
    radii: np.ndarray,
    times: np.ndarray,
    profiles: np.ndarray,
    context: str,
) -> None:
    if radii.ndim != 1 or times.ndim != 1:
        raise ValueError(f"{context}: r and snapshot times must be one-dimensional")
    if profiles.shape != (times.size, radii.size):
        raise ValueError(
            f"{context}: profiles have shape {profiles.shape}, expected "
            f"({times.size}, {radii.size})"
        )
    if radii.size != N_POINTS or np.any(radii <= 0.0) or np.any(np.diff(radii) <= 0.0):
        raise ValueError(f"{context}: invalid {N_POINTS}-point radial grid")
    if times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{context}: snapshot times must increase strictly")
    if not np.all(np.isfinite(profiles)):
        raise ValueError(f"{context}: profiles contain non-finite values")


def _reference_linear() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = _load_npz(LINEAR_REFERENCE)
    required = {"r", "snapshot_ts", "snapshots_u"}
    missing = required - set(data)
    if missing:
        raise KeyError(f"{LINEAR_REFERENCE}: missing {sorted(missing)}")
    radii = np.asarray(data["r"], dtype=float)
    times = np.asarray(data["snapshot_ts"], dtype=float)
    profiles = np.asarray(data["snapshots_u"], dtype=float)
    _validate_profiles(radii, times, profiles, "linear reference")
    return radii, times, profiles


def _reference_energy_cell() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = _load_npz(ENERGY_SPLIT_REFERENCE)
    radii = np.asarray(data["r"], dtype=float)
    times = np.asarray(data["snapshot_ts"], dtype=float)
    profiles = np.asarray(data["snapshots_u"], dtype=float)
    _validate_profiles(radii, times, profiles, "energy-split reference")
    return radii, times, profiles


def _evolve_mexhat(lambda_coupling: float, amplitude: float) -> tuple[np.ndarray, ...]:
    from rsd.reference import SphericalOracle1D

    oracle = SphericalOracle1D(
        M=1.0,
        l=0,
        r_min=R_MIN,
        r_max=R_MAX,
        n_points=N_POINTS,
        grid="log",
        potential_type="mexican_hat",
        potential_params={
            "lambda_coupling": float(lambda_coupling),
            "vacuum_value": VACUUM_VALUE,
        },
        u_infinity=VACUUM_VALUE,
        ko_eps=KO_EPS,
    )
    oracle.set_initial_gaussian(
        A=float(amplitude),
        r0=PULSE_R0,
        width=PULSE_WIDTH,
        direction=PULSE_DIRECTION,
    )
    dt = oracle.compute_dt()
    snapshot_every = max(1, int(np.ceil(T_END / dt)) // SNAPSHOT_COUNT_TARGET)
    result = oracle.evolve(
        t_end=T_END,
        probe_radii=[PRIMARY_WINDOW[0], PRIMARY_WINDOW[1]],
        output_every=200,
        snapshot_every=snapshot_every,
    )
    radii = oracle.r.copy()
    times = np.asarray(result.snapshot_ts, dtype=float)
    profiles = np.asarray(result.snapshots_u, dtype=float)
    _validate_profiles(radii, times, profiles, _cell_id(lambda_coupling, amplitude))
    return radii, times, profiles


def _cache_path(cache_dir: Path, lambda_coupling: float, amplitude: float) -> Path:
    return cache_dir / f"{_cell_id(lambda_coupling, amplitude)}.npz"


def _load_cache(path: Path, lambda_coupling: float, amplitude: float) -> tuple[np.ndarray, ...]:
    data = _load_npz(path)
    if not np.isclose(float(data["lambda_coupling"]), lambda_coupling):
        raise ValueError(f"{path}: lambda metadata mismatch")
    if not np.isclose(float(data["amplitude"]), amplitude):
        raise ValueError(f"{path}: amplitude metadata mismatch")
    radii = np.asarray(data["r"], dtype=float)
    times = np.asarray(data["snapshot_ts"], dtype=float)
    profiles = np.asarray(data["snapshots_u"], dtype=float)
    _validate_profiles(radii, times, profiles, path.name)
    return radii, times, profiles


def _write_cache(
    path: Path,
    lambda_coupling: float,
    amplitude: float,
    radii: np.ndarray,
    times: np.ndarray,
    profiles: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                lambda_coupling=np.asarray(lambda_coupling),
                amplitude=np.asarray(amplitude),
                r=radii,
                snapshot_ts=times,
                snapshots_u=profiles,
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fit_run(radii: np.ndarray, times: np.ndarray, profiles: np.ndarray) -> dict[str, Any]:
    primary = fit_log_profile_series(
        radii, profiles, PRIMARY_WINDOW, order=PRIMARY_ORDER
    )
    anchor = fit_log_profile_series(
        radii, profiles, ANCHOR_WINDOW, order=ANCHOR_ORDER
    )
    return {
        "t": np.asarray(times, dtype=float),
        "a_primary": np.asarray(primary["a"], dtype=float),
        "a_primary_err": np.asarray(primary["a_err"], dtype=float),
        "a_anchor": np.asarray(anchor["a"], dtype=float),
        "strong_mask": production_strong_mask(times, anchor["a"]),
    }


def _common_support(linear: dict[str, Any], mexhat: dict[str, Any]) -> np.ndarray:
    linear_t = np.asarray(linear["t"], dtype=float)
    mexhat_strong = np.interp(
        linear_t,
        np.asarray(mexhat["t"], dtype=float),
        np.asarray(mexhat["strong_mask"], dtype=float),
    ) > 0.5
    return np.asarray(linear["strong_mask"], dtype=bool) & mexhat_strong


def _support_intervals(times: np.ndarray, mask: np.ndarray) -> list[list[float]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [[float(times[group[0]]), float(times[group[-1]])] for group in groups]


def _propagated_l2_sigma(
    linear: dict[str, Any], mexhat: dict[str, Any], common: np.ndarray, value: float
) -> float:
    """Indicative independent-OLS propagation for the imported L2 metric."""
    times = np.asarray(linear["t"], dtype=float)
    a_linear = np.asarray(linear["a_primary"], dtype=float)[common]
    e_linear = np.asarray(linear["a_primary_err"], dtype=float)[common]
    a_mexhat = np.interp(times, mexhat["t"], mexhat["a_primary"])[common]
    e_mexhat = np.interp(times, mexhat["t"], mexhat["a_primary_err"])[common]
    sum_linear = float(np.sum(a_linear**2))
    sum_mexhat = float(np.sum(a_mexhat**2))
    if sum_linear <= 0.0 or sum_mexhat <= 0.0:
        raise ValueError("cannot propagate uncertainty for a zero L2 norm")
    variance_log = float(
        np.sum((a_mexhat * e_mexhat) ** 2) / sum_mexhat**2
        + np.sum((a_linear * e_linear) ** 2) / sum_linear**2
    )
    return float(value * np.sqrt(max(variance_log, 0.0)))


def _analyse_cell(
    radii: np.ndarray,
    linear_times: np.ndarray,
    linear_profiles_a01: np.ndarray,
    mexhat_times: np.ndarray,
    mexhat_profiles: np.ndarray,
    lambda_coupling: float,
    amplitude: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    linear_profiles = linear_profiles_a01 * (float(amplitude) / 0.1)
    linear = _fit_run(radii, linear_times, linear_profiles)
    mexhat = _fit_run(radii, mexhat_times, mexhat_profiles)
    metrics = raw_discriminator(linear, mexhat)
    common = _common_support(linear, mexhat)
    if int(np.count_nonzero(common)) != int(metrics["n_samples"]):
        raise RuntimeError("support reconstruction disagrees with imported discriminator")
    sigma = _propagated_l2_sigma(linear, mexhat, common, metrics["l2_ratio"])
    support_times = linear_times[common]
    cell = {
        "id": _cell_id(lambda_coupling, amplitude),
        "lambda": float(lambda_coupling),
        "amplitude": float(amplitude),
        "D_oracle": float(metrics["l2_ratio"]),
        "sigma_ols": sigma,
        "support": {
            "n_samples": int(support_times.size),
            "time_min": float(support_times[0]),
            "time_max": float(support_times[-1]),
            "contiguous_intervals": _support_intervals(linear_times, common),
        },
        "distance_from_unity": float(abs(metrics["l2_ratio"] - 1.0)),
        "discovery_threshold_exceeded": bool(
            abs(metrics["l2_ratio"] - 1.0) > DISCOVERY_THRESHOLD
        ),
    }
    return cell, linear, mexhat


def _arrays_from_scan(cache_dir: Path, force: bool) -> dict[str, np.ndarray]:
    linear_r, linear_t, linear_u = _reference_linear()
    arrays: dict[str, np.ndarray] = {
        "r": linear_r,
        "linear_snapshot_ts": linear_t,
        "linear_snapshots_u_A0p1": linear_u,
        "grid_lambdas": np.asarray(LAMBDAS),
        "grid_amplitudes": np.asarray(AMPLITUDES),
    }
    energy_r, energy_t, energy_u = _reference_energy_cell()
    if not np.array_equal(energy_r, linear_r):
        raise ValueError("energy-split and linear references use different radial grids")

    total = len(LAMBDAS) * len(AMPLITUDES)
    ordinal = 0
    for lambda_coupling in LAMBDAS:
        for amplitude in AMPLITUDES:
            ordinal += 1
            cell = _cell_id(lambda_coupling, amplitude)
            started = time.perf_counter()
            if np.isclose(lambda_coupling, 0.1) and np.isclose(amplitude, 0.1):
                radii, times, profiles = energy_r, energy_t, energy_u
                action = "reused energy-split reference"
            else:
                cache = _cache_path(cache_dir, lambda_coupling, amplitude)
                if cache.is_file() and not force:
                    radii, times, profiles = _load_cache(
                        cache, lambda_coupling, amplitude
                    )
                    action = "reused cache"
                else:
                    radii, times, profiles = _evolve_mexhat(
                        lambda_coupling, amplitude
                    )
                    _write_cache(
                        cache,
                        lambda_coupling,
                        amplitude,
                        radii,
                        times,
                        profiles,
                    )
                    action = "evolved"
            if not np.array_equal(radii, linear_r):
                raise ValueError(f"{cell}: radial grid differs from the linear reference")
            arrays[f"{cell}__snapshot_ts"] = times
            arrays[f"{cell}__snapshots_u"] = profiles
            print(
                f"[{ordinal:02d}/{total}] {cell}: {action} "
                f"({time.perf_counter() - started:.1f} s)",
                flush=True,
            )
    return arrays


def _audit_spot_check(
    cell: dict[str, Any], linear: dict[str, Any], mexhat: dict[str, Any]
) -> dict[str, Any]:
    return {
        "cell_id": cell["id"],
        "reported_D_oracle": cell["D_oracle"],
        "linear": {
            "t": np.asarray(linear["t"]).tolist(),
            "a_primary": np.asarray(linear["a_primary"]).tolist(),
            "a_anchor": np.asarray(linear["a_anchor"]).tolist(),
            "strong_mask": np.asarray(linear["strong_mask"], dtype=bool).tolist(),
        },
        "mexhat": {
            "t": np.asarray(mexhat["t"]).tolist(),
            "a_primary": np.asarray(mexhat["a_primary"]).tolist(),
            "a_anchor": np.asarray(mexhat["a_anchor"]).tolist(),
            "strong_mask": np.asarray(mexhat["strong_mask"], dtype=bool).tolist(),
        },
    }


def build_payload(arrays: dict[str, np.ndarray], data_path: Path) -> dict[str, Any]:
    radii = np.asarray(arrays["r"], dtype=float)
    linear_t = np.asarray(arrays["linear_snapshot_ts"], dtype=float)
    linear_u = np.asarray(arrays["linear_snapshots_u_A0p1"], dtype=float)
    _validate_profiles(radii, linear_t, linear_u, "versioned linear reference")
    if not np.array_equal(np.asarray(arrays["grid_lambdas"]), np.asarray(LAMBDAS)):
        raise ValueError("versioned lambda grid differs from the contract")
    if not np.array_equal(np.asarray(arrays["grid_amplitudes"]), np.asarray(AMPLITUDES)):
        raise ValueError("versioned amplitude grid differs from the contract")

    cells: list[dict[str, Any]] = []
    audit: dict[str, Any] | None = None
    for lambda_coupling in LAMBDAS:
        for amplitude in AMPLITUDES:
            cell_id = _cell_id(lambda_coupling, amplitude)
            times = np.asarray(arrays[f"{cell_id}__snapshot_ts"], dtype=float)
            profiles = np.asarray(arrays[f"{cell_id}__snapshots_u"], dtype=float)
            _validate_profiles(radii, times, profiles, cell_id)
            cell, linear, mexhat = _analyse_cell(
                radii,
                linear_t,
                linear_u,
                times,
                profiles,
                lambda_coupling,
                amplitude,
            )
            cells.append(cell)
            if np.isclose(lambda_coupling, 0.3) and np.isclose(amplitude, 0.03):
                audit = _audit_spot_check(cell, linear, mexhat)
    if audit is None:
        raise RuntimeError("audit cell is absent from the sensitivity grid")

    distances = np.asarray([cell["distance_from_unity"] for cell in cells])
    values = np.asarray([cell["D_oracle"] for cell in cells])
    worst = cells[int(np.argmax(distances))]
    relative_data = data_path.resolve().relative_to(REPO.resolve()).as_posix()
    return {
        "schema_version": 1,
        "generated_by": "scripts/oracle_sensitivity_scan.py",
        "protocol": {
            "l": 0,
            "lambdas": list(LAMBDAS),
            "amplitudes": list(AMPLITUDES),
            "vacuum_value": VACUUM_VALUE,
            "n_points": N_POINTS,
            "radial_domain": [R_MIN, R_MAX],
            "grid": "log",
            "t_end": T_END,
            "ko_eps": KO_EPS,
            "pulse": {
                "r0": PULSE_R0,
                "width": PULSE_WIDTH,
                "direction": PULSE_DIRECTION,
            },
            "primary_fit": {"window": list(PRIMARY_WINDOW), "order": PRIMARY_ORDER},
            "anchor_fit": {"window": list(ANCHOR_WINDOW), "order": ANCHOR_ORDER},
            "strong_mask": "o0 anchor; 4<=t/M<=10; abs(a)>=0.3*window max",
            "linear_member": (
                "exact amplitude scaling of the frozen zero-potential A=0.1 "
                "n=1600 production reference"
            ),
            "discovery_threshold_abs_D_minus_one": DISCOVERY_THRESHOLD,
        },
        "uncertainty": {
            "field": "sigma_ols",
            "definition": (
                "first-order independent propagation of the per-snapshot OLS "
                "fit errors through the imported production L2 discriminator"
            ),
            "caveat": (
                "indicative fit uncertainty only; the 3D publication budget "
                "is controlled by mesh/extraction systematics"
            ),
        },
        "sources": {
            "linear_reference": {
                "path": LINEAR_REFERENCE.relative_to(REPO).as_posix(),
                "sha256": _sha256(LINEAR_REFERENCE),
            },
            "reused_lambda_0p1_A_0p1": {
                "path": ENERGY_SPLIT_REFERENCE.relative_to(REPO).as_posix(),
                "sha256": _sha256(ENERGY_SPLIT_REFERENCE),
            },
        },
        "cells": cells,
        "summary": {
            "cell_count": len(cells),
            "D_min": float(np.min(values)),
            "D_max": float(np.max(values)),
            "max_abs_dev_from_unity": float(np.max(distances)),
            "worst_cell": worst["id"],
            "discovery_threshold_exceeded": bool(np.any(distances > DISCOVERY_THRESHOLD)),
            "threshold_exceeding_cells": [
                cell["id"] for cell in cells if cell["discovery_threshold_exceeded"]
            ],
        },
        "audit_spot_check": audit,
        "data": {"path": relative_data, "sha256": _sha256(data_path)},
    }


def _same_arrays(path: Path, arrays: dict[str, np.ndarray]) -> bool:
    if not path.is_file():
        return False
    try:
        current = _load_npz(path)
    except (OSError, ValueError):
        return False
    return set(current) == set(arrays) and all(
        np.array_equal(current[name], value, equal_nan=True)
        for name, value in arrays.items()
    )


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _same_arrays(path, arrays):
        return False
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


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
    parser.add_argument("--data-output", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.check:
        if not args.data_output.is_file():
            print(f"sensitivity data missing: {args.data_output}", file=sys.stderr)
            return 1
        arrays = _load_npz(args.data_output)
        payload = build_payload(arrays, args.data_output)
        rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if (
            not args.json_output.is_file()
            or args.json_output.read_text(encoding="utf-8") != rendered
        ):
            print(f"sensitivity JSON missing/stale: {args.json_output}", file=sys.stderr)
            return 1
        print("sensitivity scan: JSON/NPZ verified; outputs up to date")
        return 0

    started = time.perf_counter()
    if args.data_output.is_file() and not args.force:
        arrays = _load_npz(args.data_output)
        data_changed = False
        action = "reused versioned data"
    else:
        arrays = _arrays_from_scan(args.cache_dir, args.force)
        data_changed = _write_npz(args.data_output, arrays)
        action = "completed grid"
    payload = build_payload(arrays, args.data_output)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    json_changed = _atomic_write(args.json_output, rendered)
    summary = payload["summary"]
    print(
        "sensitivity scan: "
        f"{action}; D=[{summary['D_min']:.6g}, {summary['D_max']:.6g}], "
        f"max|D-1|={summary['max_abs_dev_from_unity']:.6g}; "
        f"JSON {'updated' if json_changed else 'unchanged'}, "
        f"NPZ {'updated' if data_changed else 'unchanged'}; "
        f"{time.perf_counter() - started:.1f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
