#!/usr/bin/env python3
"""Measure kinetic domination for the production Mexican-hat oracle.

Two complementary ratios are retained.  ``rhs_ratio`` exactly follows the
Phase-0 dynamical definition, |alpha V'| divided by the sum of the other
RMS/instantaneous terms in dPi/dt.  ``energy_ratio`` is the requested local
potential-to-kinetic energy-density split, V/(Pi^2 + gamma^rr u_r^2)/2.
The former defines the citable crossover and scaling so the F0 cross-check is
like for like; both complete profile families are versioned in the NPZ.
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
DEFAULT_JSON = REPO / "docs/research/phase3/energy_split.json"
DEFAULT_DATA = REPO / "docs/research/phase3/data/energy_split_profiles.npz"

N_POINTS = 1600
R_MIN = 0.01
R_MAX = 60.0
T_END = 15.0
SNAPSHOT_COUNT_TARGET = 80
RATIO_THRESHOLD = 0.01
EXPONENT_WINDOW = (0.05, 0.5)


def _production_module():
    scripts = str(REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import interior_production as production  # noqa: WPS433

    return production


def _contiguous_crossover(
    radii: np.ndarray, ratio: np.ndarray, threshold: float
) -> float:
    """Largest sampled radius in the sub-threshold component from r_min."""
    radii = np.asarray(radii, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    below = np.isfinite(ratio) & (ratio < float(threshold))
    if radii.ndim != 1 or ratio.shape != radii.shape or not below[0]:
        return float("nan")
    failures = np.flatnonzero(~below)
    if failures.size == 0:
        return float(radii[-1])
    index = int(failures[0])
    return float(radii[max(index - 1, 0)])


def _log_slope(
    radii: np.ndarray, ratio: np.ndarray, window: tuple[float, float]
) -> float:
    radii = np.asarray(radii, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    mask = (
        (radii >= window[0])
        & (radii <= window[1])
        & np.isfinite(ratio)
        & (ratio > 0.0)
    )
    if np.count_nonzero(mask) < 3:
        return float("nan")
    return float(np.polyfit(np.log(radii[mask]), np.log(ratio[mask]), 1)[0])


def _analyse_profiles(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    radii = arrays["r"]
    times = arrays["snapshot_ts"]
    rhs = arrays["rhs_ratio"]
    strong = (times >= arrays["strong_t_min"].item()) & (
        times <= arrays["strong_t_max"].item()
    )
    indices = np.flatnonzero(strong)
    if indices.size == 0:
        raise ValueError("energy split contains no snapshots in the strong phase")

    rstars = np.asarray(
        [_contiguous_crossover(radii, rhs[index], RATIO_THRESHOLD) for index in indices]
    )
    exponents = np.asarray(
        [_log_slope(radii, rhs[index], EXPONENT_WINDOW) for index in indices]
    )
    valid_rstar = rstars[np.isfinite(rstars)]
    valid_exponent = exponents[np.isfinite(exponents)]
    if valid_rstar.size == 0 or valid_exponent.size == 0:
        raise ValueError("kinetic-domination summaries are undefined")

    term_names = ("flux_div", "transport", "extrinsic", "angular")
    potential_rms = np.sqrt(np.mean(arrays["rhs_potential"][indices] ** 2, axis=0))
    kinetic_rms = sum(
        np.sqrt(np.mean(arrays[f"rhs_{name}"][indices] ** 2, axis=0))
        for name in term_names
    )
    ratio_rms = potential_rms / np.maximum(kinetic_rms, np.finfo(float).tiny)

    return {
        "strong_snapshot_count": int(indices.size),
        "rstar_median": float(np.median(valid_rstar)),
        "rstar_iqr": [
            float(np.quantile(valid_rstar, 0.25)),
            float(np.quantile(valid_rstar, 0.75)),
        ],
        "ratio_exponent_median": float(np.median(valid_exponent)),
        "ratio_exponent_iqr": [
            float(np.quantile(valid_exponent, 0.25)),
            float(np.quantile(valid_exponent, 0.75)),
        ],
        "temporal_rms_crosscheck": {
            "rstar": _contiguous_crossover(radii, ratio_rms, RATIO_THRESHOLD),
            "ratio_exponent": _log_slope(radii, ratio_rms, EXPONENT_WINDOW),
        },
    }


def _evolve_profiles() -> dict[str, np.ndarray]:
    production = _production_module()
    from rsd.reference import SphericalOracle1D

    spec = production.POTS["mexhat"]
    oracle = SphericalOracle1D(
        M=1.0,
        l=0,
        r_min=R_MIN,
        r_max=R_MAX,
        n_points=N_POINTS,
        grid="log",
        potential_type=spec["potential_type"],
        potential_params=spec["potential_params"],
        u_infinity=spec["v0"],
        ko_eps=0.02,
    )
    oracle.set_initial_gaussian(
        A=production.PULSE["A"],
        r0=production.PULSE["r0"],
        width=production.PULSE["w"],
        direction=production.PULSE["direction"],
    )
    dt = oracle.compute_dt()
    snapshot_every = max(1, int(np.ceil(T_END / dt)) // SNAPSHOT_COUNT_TARGET)
    result = oracle.evolve(
        t_end=T_END,
        probe_radii=[production.TRUTH_WINDOW[0], production.PRIMARY[0][1]],
        output_every=200,
        snapshot_every=snapshot_every,
    )

    times = np.asarray(result.snapshot_ts)
    snapshots_u = np.asarray(result.snapshots_u)
    snapshots_pi = np.asarray(result.snapshots_Pi)
    n_snapshots = times.size
    term_names = ("flux_div", "transport", "extrinsic", "angular", "potential")
    rhs_terms = {
        name: np.empty((n_snapshots, oracle.r.size), dtype=float)
        for name in term_names
    }
    kinetic_energy = np.empty_like(snapshots_u)
    potential_energy = np.empty_like(snapshots_u)

    for index, (u, pi) in enumerate(zip(snapshots_u, snapshots_pi)):
        terms = oracle.rhs_term_breakdown(u, pi)
        for name in term_names:
            rhs_terms[name][index] = terms[name]
        du = oracle._deriv_r(u)
        kinetic_energy[index] = 0.5 * (
            pi**2 + oracle.gamma_rr_inv * du**2
        )
        potential_energy[index] = oracle.potential.evaluate_np(u)

    kinetic_rhs = sum(np.abs(rhs_terms[name]) for name in term_names[:-1])
    rhs_ratio = np.abs(rhs_terms["potential"]) / np.maximum(
        kinetic_rhs, np.finfo(float).tiny
    )
    energy_ratio = np.abs(potential_energy) / np.maximum(
        kinetic_energy, np.finfo(float).tiny
    )

    arrays: dict[str, np.ndarray] = {
        "r": oracle.r.copy(),
        "snapshot_ts": times,
        "snapshots_u": snapshots_u,
        "snapshots_Pi": snapshots_pi,
        "kinetic_energy_density": kinetic_energy,
        "potential_energy_density": potential_energy,
        "energy_ratio": energy_ratio,
        "rhs_ratio": rhs_ratio,
        "strong_t_min": np.asarray(production.T_MIN),
        "strong_t_max": np.asarray(production.T_MAX),
    }
    arrays.update({f"rhs_{name}": values for name, values in rhs_terms.items()})
    return arrays


def _same_arrays(path: Path, arrays: dict[str, np.ndarray]) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as current:
            if set(current.files) != set(arrays):
                return False
            return all(
                np.array_equal(current[name], value, equal_nan=True)
                for name, value in arrays.items()
            )
    except (OSError, ValueError):
        return False


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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_payload(arrays: dict[str, np.ndarray], data_path: Path) -> dict[str, Any]:
    production = _production_module()
    summary = _analyse_profiles(arrays)
    relative_data = data_path.resolve().relative_to(REPO.resolve()).as_posix()
    return {
        "schema_version": 1,
        "generated_by": "scripts/oracle_energy_split.py",
        "protocol": {
            "model": "mexican_hat",
            "l": 0,
            "n_points": N_POINTS,
            "r_min": R_MIN,
            "r_max": R_MAX,
            "grid": "log",
            "ko_eps": 0.02,
            "t_end": T_END,
            "pulse": production.PULSE,
            "potential": production.POTS["mexhat"],
            "strong_phase": [production.T_MIN, production.T_MAX],
            "ratio_threshold": RATIO_THRESHOLD,
            "exponent_window": list(EXPONENT_WINDOW),
        },
        "definitions": {
            "kinetic_energy_density": (
                "0.5*Pi^2 + 0.5*gamma^rr*(partial_r u)^2"
            ),
            "energy_ratio": "abs(V(u))/kinetic_energy_density",
            "rhs_ratio": (
                "abs(alpha*V'(u)) / (abs(flux_div)+abs(transport)+"
                "abs(extrinsic)+abs(angular))"
            ),
            "rstar": (
                "largest sampled radius in the contiguous component from r_min "
                "where rhs_ratio < ratio_threshold; manuscript value is the "
                "median over strong-phase snapshots"
            ),
            "ratio_exponent": (
                "OLS slope of log(rhs_ratio) versus log(r), evaluated per "
                "strong-phase snapshot and summarized by the median"
            ),
        },
        "summary": summary,
        "data": {
            "path": relative_data,
            "sha256": _sha256(data_path),
        },
    }


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
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.check:
        if not args.data_output.is_file():
            print(f"energy-split data missing: {args.data_output}", file=sys.stderr)
            return 1
        arrays = _load_npz(args.data_output)
        payload = build_payload(arrays, args.data_output)
        rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if (
            not args.json_output.is_file()
            or args.json_output.read_text(encoding="utf-8") != rendered
        ):
            print(f"energy-split JSON missing/stale: {args.json_output}", file=sys.stderr)
            return 1
        print("energy split: JSON/NPZ verified; outputs up to date")
        return 0

    started = time.perf_counter()
    if args.data_output.is_file() and not args.force:
        arrays = _load_npz(args.data_output)
        data_changed = False
        action = "reused"
    else:
        arrays = _evolve_profiles()
        data_changed = _write_npz(args.data_output, arrays)
        action = "evolved"
    payload = build_payload(arrays, args.data_output)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    json_changed = _atomic_write(args.json_output, rendered)
    summary = payload["summary"]
    print(
        "energy split: "
        f"{action}; r*={summary['rstar_median']:.6g} M, "
        f"p={summary['ratio_exponent_median']:.6g}; "
        f"JSON {'updated' if json_changed else 'unchanged'}, "
        f"NPZ {'updated' if data_changed else 'unchanged'}; "
        f"{time.perf_counter() - started:.1f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
