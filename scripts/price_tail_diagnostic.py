#!/usr/bin/env python3
"""Diagnose the chronically failing three-dimensional Price-tail test.

The evolution is exactly the one in ``tests/test_price_tails_slow.py`` and is
run at most once.  The versioned time series supports window, RMS-envelope,
and late-floor diagnostics without repeating the expensive solve.
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

from rsd.analysis.tails import fit_power_law_tail  # noqa: E402
from rsd.analysis.ringdown import fit_tail_lines  # noqa: E402


DEFAULT_DATA = REPO / "docs/research/phase3/data/price_tail_diagnostic.npz"
DEFAULT_JSON = REPO / "docs/research/phase3/price_tail_diagnostic.json"
FIT_WINDOWS = ((45.0, 65.0), (55.0, 75.0), (65.0, 85.0), (75.0, 95.0), (80.0, 100.0))
RMS_BINS = ((55.0, 65.0), (65.0, 75.0), (75.0, 85.0), (85.0, 95.0), (95.0, 100.0))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _validate(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(arrays["t"], dtype=float)
    signal = np.asarray(arrays["c10"], dtype=float)
    if times.ndim != 1 or signal.shape != times.shape or times.size < 32:
        raise ValueError("Price diagnostic requires equal nontrivial 1D series")
    if np.any(np.diff(times) <= 0.0) or not np.all(np.isfinite(signal)):
        raise ValueError("Price diagnostic series is not finite/strictly sampled")
    if times[-1] < 100.0:
        raise ValueError("Price diagnostic series ends before the fit window")
    return times, signal


def _evolve() -> dict[str, np.ndarray]:
    from rsd.analysis.ringdown import evolve_kerr_ringdown

    times, signal = evolve_kerr_ringdown(
        a=0.0,
        l=1,
        m_abs=0,
        R=60.0,
        lc=3.0,
        lc_inner=0.5,
        r0=8.0,
        w=2.0,
        r_ext=6.0,
        t_end=110.0,
    )
    arrays = {"t": np.asarray(times, dtype=float), "c10": np.asarray(signal, dtype=float)}
    _validate(arrays)
    return arrays


def _rms_summary(times: np.ndarray, signal: np.ndarray) -> list[dict[str, float | int]]:
    out: list[dict[str, float | int]] = []
    for lo, hi in RMS_BINS:
        mask = (times >= lo) & (times <= hi)
        values = signal[mask]
        out.append(
            {
                "t_min": lo,
                "t_max": hi,
                "t_mid": 0.5 * (lo + hi),
                "n": int(values.size),
                "rms": float(np.sqrt(np.mean(values**2))),
                "mean": float(np.mean(values)),
                "abs_median": float(np.median(np.abs(values))),
            }
        )
    return out


def build_payload(arrays: dict[str, np.ndarray], data_path: Path) -> dict[str, Any]:
    times, signal = _validate(arrays)
    original_p, original_r2 = fit_power_law_tail(times, signal, 55.0, 100.0)
    sweep = []
    for lo, hi in FIT_WINDOWS:
        exponent, r2 = fit_power_law_tail(times, signal, lo, hi)
        sweep.append({"t_min": lo, "t_max": hi, "exponent": exponent, "r2": r2})
    bins = _rms_summary(times, signal)
    mids = np.asarray([row["t_mid"] for row in bins], dtype=float)
    rms = np.asarray([row["rms"] for row in bins], dtype=float)
    envelope_exponent = float(np.polyfit(np.log(mids), np.log(rms), 1)[0])
    initial_peak = float(np.max(np.abs(signal)))
    late = (times >= 90.0) & (times <= 100.0)
    late_rms = float(np.sqrt(np.mean(signal[late] ** 2)))
    late_lines = fit_tail_lines(times, signal, t_min=75.0)
    return {
        "schema_version": 1,
        "generated_by": "scripts/price_tail_diagnostic.py",
        "protocol": {
            "a": 0.0,
            "l": 1,
            "m_abs": 0,
            "R": 60.0,
            "lc": 3.0,
            "lc_inner": 0.5,
            "r0": 8.0,
            "width": 2.0,
            "r_ext": 6.0,
            "t_end": 110.0,
            "boundary_echo_estimate": 106.0,
            "expected_price_exponent": -5.0,
        },
        "original_test_fit": {
            "window": [55.0, 100.0],
            "exponent": original_p,
            "r2": original_r2,
        },
        "window_sweep": sweep,
        "rms_bins": bins,
        "rms_envelope_exponent": envelope_exponent,
        "late_floor_lines": {"t_min": 75.0, **late_lines},
        "amplitude_floor": {
            "initial_peak": initial_peak,
            "late_rms_90_100": late_rms,
            "late_over_initial": float(late_rms / initial_peak),
            "last_over_first_rms_bin": float(rms[-1] / rms[0]),
            "price_prediction_last_over_first": float((mids[-1] / mids[0]) ** -5),
        },
        "data": {
            "path": data_path.resolve().relative_to(REPO.resolve()).as_posix(),
            "sha256": _sha256(data_path),
        },
    }


def _same(path: Path, arrays: dict[str, np.ndarray]) -> bool:
    if not path.is_file():
        return False
    try:
        current = _load(path)
    except (OSError, ValueError):
        return False
    return set(current) == set(arrays) and all(
        np.array_equal(current[name], value, equal_nan=True) for name, value in arrays.items()
    )


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _same(path, arrays):
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
    parser.add_argument("--data-output", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.check:
        if not args.data_output.is_file():
            print(f"Price diagnostic data missing: {args.data_output}", file=sys.stderr)
            return 1
        arrays = _load(args.data_output)
        payload = build_payload(arrays, args.data_output)
        rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if not args.json_output.is_file() or args.json_output.read_text(encoding="utf-8") != rendered:
            print(f"Price diagnostic JSON missing/stale: {args.json_output}", file=sys.stderr)
            return 1
        print("Price-tail diagnostic: JSON/NPZ verified; outputs up to date")
        return 0

    started = time.perf_counter()
    if args.data_output.is_file() and not args.force:
        arrays = _load(args.data_output)
        action = "reused"
        data_changed = False
    else:
        arrays = _evolve()
        data_changed = _write_npz(args.data_output, arrays)
        action = "evolved"
    payload = build_payload(arrays, args.data_output)
    json_changed = _atomic_write(
        args.json_output, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    fit = payload["original_test_fit"]
    print(
        f"Price-tail diagnostic: {action}; p={fit['exponent']:.6g}, "
        f"R2={fit['r2']:.6g}; JSON {'updated' if json_changed else 'unchanged'}, "
        f"NPZ {'updated' if data_changed else 'unchanged'}; "
        f"{time.perf_counter() - started:.1f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
