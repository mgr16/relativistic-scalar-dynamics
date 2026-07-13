#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C3 — regenerate every manuscript figure from versioned artifacts.

The plotting pipeline reads only canonical JSON files and tracked NPZ inputs
under ``docs/research``.  It never reads ``results/``.  Explicit numerical
annotations are resolved by id from ``phase3/numbers.json`` and reject
``no-citable`` or ``degradado-a-prosa`` entries.

Outputs are deterministic PDF (vector) and PNG (preview) pairs under
``paper/figures``.  Rendering is staged fully in memory before any file is
published; unchanged files retain their mtimes.  ``--check`` renders without
writing and byte-compares every expected output.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEFAULT_OUTPUT_DIR = REPO / "paper/figures"
NUMBERS_PATH = REPO / "docs/research/phase3/numbers.json"
PRODUCTION_PATH = REPO / "docs/research/phase2/production/production.json"
CALIBRATION_PATH = REPO / "docs/research/phase3/o1_calibration.json"
SPECTROSCOPY_PATH = REPO / "docs/research/phase2/exterior/spectroscopy.json"
CAVITY_SUMMARY_PATH = REPO / "docs/research/phase1/cavity/summary.json"

PROFILE_PATHS = {
    "linear": REPO / "docs/research/phase3/data/ab_smoke_3d_linear_l0_lc0.040.npz",
    "mexhat": REPO / "docs/research/phase3/data/ab_smoke_3d_mexhat_l0_lc0.040.npz",
}
PROFILE_REFERENCE_PATHS = {
    "linear": REPO
    / "docs/research/phase2/interior/data/ab_smoke_ref_linear_A0.1_n1600.npz",
    "mexhat": REPO
    / "docs/research/phase2/interior/data/ab_smoke_ref_mexhat_A0.1_n1600.npz",
}
CAVITY_WAVEFORM_PATH = REPO / "docs/research/phase1/cavity/waveform_l2_lc1.npz"
DOMAIN_WAVEFORM_PATH = REPO / "docs/research/phase2/exterior/data/wf_R40_lc1match.npz"

# Exact copies of the canonical smoke profiles promoted from the local cache.
# Validation makes the C3 figure provenance auditable and prevents silent drift.
PROFILE_SHA256 = {
    "linear": "46ceca926ac7235e8a4f8ac2bda2aedb232af43d2ef9054285a1b6dd88b5c160",
    "mexhat": "e555cb31f994aa70308653f65fd886315fdc32ad03d4676b0dcad2363a3ab446",
}

VERSIONED_INPUTS = (
    NUMBERS_PATH,
    PRODUCTION_PATH,
    CALIBRATION_PATH,
    SPECTROSCOPY_PATH,
    CAVITY_SUMMARY_PATH,
    *PROFILE_PATHS.values(),
    *PROFILE_REFERENCE_PATHS.values(),
    CAVITY_WAVEFORM_PATH,
    DOMAIN_WAVEFORM_PATH,
)

ALLOWED_NUMBER_STATUSES = {"citable", "citable-con-caveat"}
FORMATS = ("pdf", "png")
PNG_DPI = 240

# Okabe–Ito colorblind-safe palette.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
SKY = "#56B4E9"
PURPLE = "#CC79A7"
YELLOW = "#E69F00"
INK = "#202124"
GRAY = "#6B6F72"
LIGHT_GRAY = "#E6E8EA"

POTENTIAL_STYLE = {
    "linear": (BLUE, "linear"),
    "mexhat": (ORANGE, "Mexican hat"),
}


def paper_style() -> dict[str, Any]:
    """Common PRD-oriented Matplotlib style (single/wide column safe)."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "axes.linewidth": 0.7,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 7.0,
        "legend.frameon": False,
        "lines.linewidth": 1.35,
        "lines.markersize": 4.2,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return document


def _load_npz(path: Path, required: Iterable[str]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(required) - set(data.files))
        if missing:
            raise KeyError(f"{path}: missing NPZ keys {missing}")
        return {key: np.asarray(data[key]) for key in data.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_versioned_inputs() -> None:
    """Reject missing, out-of-tree, or cache-backed paper inputs."""
    docs_root = (REPO / "docs/research").resolve()
    for path in VERSIONED_INPUTS:
        resolved = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if docs_root not in resolved.parents:
            raise ValueError(f"paper input must live under docs/research: {path}")
        relative = resolved.relative_to(REPO.resolve()).as_posix()
        if relative == "results" or relative.startswith("results/"):
            raise ValueError(f"paper figures may not read gitignored results: {path}")
    for potential, expected in PROFILE_SHA256.items():
        if not expected:
            raise ValueError(
                f"profile hash for {potential} has not been approved and recorded"
            )
        actual = _sha256(PROFILE_PATHS[potential])
        if actual != expected:
            raise ValueError(
                f"profile source drift for {potential}: {actual} != {expected}"
            )


class NumberCatalog:
    """Canonical numeric lookup that enforces the C2 publication status."""

    def __init__(self, path: Path = NUMBERS_PATH) -> None:
        payload = _load_json(path)
        if payload.get("generated_by") != "scripts/paper_numbers.py":
            raise ValueError(f"{path}: unexpected generator")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise TypeError(f"{path}: entries must be a list")
        self._entries: dict[str, dict[str, Any]] = {}
        self.used_ids: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
                raise TypeError(f"{path}: malformed number entry")
            entry_id = entry["id"]
            if entry_id in self._entries:
                raise ValueError(f"{path}: duplicate number id {entry_id}")
            self._entries[entry_id] = entry

    def entry(self, entry_id: str) -> dict[str, Any]:
        if entry_id not in self._entries:
            raise KeyError(f"numbers.json has no id {entry_id!r}")
        entry = self._entries[entry_id]
        status = entry.get("status")
        if status not in ALLOWED_NUMBER_STATUSES:
            raise ValueError(
                f"figure attempted to use {entry_id!r} with status {status!r}"
            )
        self.used_ids.add(entry_id)
        return entry

    def scalar(self, entry_id: str) -> float:
        value = self.entry(entry_id).get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{entry_id}: expected numeric scalar, found {value!r}")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{entry_id}: non-finite value")
        return value

    def iqr(self, entry_id: str) -> tuple[float, float]:
        uncertainty = self.entry(entry_id).get("uncertainty")
        if not isinstance(uncertainty, dict) or uncertainty.get("kind") != "IQR":
            raise TypeError(f"{entry_id}: expected IQR uncertainty")
        return float(uncertainty["low"]), float(uncertainty["high"])

    def symmetric(self, entry_id: str) -> float:
        uncertainty = self.entry(entry_id).get("uncertainty")
        if not isinstance(uncertainty, dict) or uncertainty.get("kind") != "symmetric":
            raise TypeError(f"{entry_id}: expected symmetric uncertainty")
        return float(uncertainty["value"])


@dataclass
class PlotContext:
    numbers: NumberCatalog
    production: dict[str, Any]
    calibration: dict[str, Any]
    spectroscopy: dict[str, Any]

    @classmethod
    def load(cls) -> "PlotContext":
        validate_versioned_inputs()
        return cls(
            numbers=NumberCatalog(),
            production=_load_json(PRODUCTION_PATH),
            calibration=_load_json(CALIBRATION_PATH),
            spectroscopy=_load_json(SPECTROSCOPY_PATH),
        )


def _panel_label(ax: Any, label: str) -> None:
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        color=INK,
        zorder=20,
    )


def _finish_axis(ax: Any, *, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=LIGHT_GRAY, lw=0.55, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


def _o1_fit_curve(r: np.ndarray, fit: Any) -> np.ndarray:
    return fit.a * np.log(r) + fit.b + fit.c * r * np.log(r) + fit.d * r


def _interpolate_snapshots(
    ts: np.ndarray, snapshots: np.ndarray, target: float
) -> np.ndarray:
    if target < ts[0] or target > ts[-1]:
        raise ValueError(f"target t={target:g} lies outside reference support")
    upper = int(np.searchsorted(ts, target, side="right"))
    if upper == 0:
        return np.asarray(snapshots[0], dtype=float)
    if upper == ts.size:
        return np.asarray(snapshots[-1], dtype=float)
    lower = upper - 1
    weight = float((target - ts[lower]) / (ts[upper] - ts[lower]))
    return (1.0 - weight) * snapshots[lower] + weight * snapshots[upper]


def _representative_profile_times(calibration: dict[str, Any]) -> tuple[float, ...]:
    """Choose three reproducible snapshots from the common strong support."""
    linear = calibration["calibrations"]["linear_l0"]
    mexhat = calibration["calibrations"]["mexhat_l0"]
    times = np.asarray(linear["t"], dtype=float)
    other_times = np.asarray(mexhat["t"], dtype=float)
    if times.shape != other_times.shape or not np.allclose(times, other_times):
        raise ValueError("linear/mexhat calibration time grids differ")
    common = np.asarray(linear["strong_mask"], dtype=bool) & np.asarray(
        mexhat["strong_mask"], dtype=bool
    )
    indices = np.flatnonzero(common)
    if indices.size < 3:
        raise ValueError("common strong support has fewer than three snapshots")
    fractions = (0.2, 0.6, 0.9)
    selected = [
        indices[int(round(fraction * (indices.size - 1)))] for fraction in fractions
    ]
    return tuple(float(times[index]) for index in selected)


def build_interior_profiles(ctx: PlotContext) -> Figure:
    """3D smoke profiles against the dense oracle and the primary o1 fit."""
    from rsd.analysis.interior import fit_log_profile

    targets = _representative_profile_times(ctx.calibration)
    phase_labels = ("early", "post-crossing", "late")
    colors = (BLUE, ORANGE, GREEN)
    sqrt4pi = float(np.sqrt(4.0 * np.pi))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.72), layout="constrained")

    for panel, (potential, ax) in enumerate(zip(("linear", "mexhat"), axes)):
        profiles = _load_npz(PROFILE_PATHS[potential], ("t", "radii", "modes", "u"))
        reference = _load_npz(
            PROFILE_REFERENCE_PATHS[potential], ("r", "snapshot_ts", "snapshots_u")
        )
        times = np.asarray(profiles["t"], dtype=float)
        radii = np.asarray(profiles["radii"], dtype=float)
        modes = np.asarray(profiles["modes"], dtype=int)
        values = np.asarray(profiles["u"], dtype=float)
        hits = np.flatnonzero(np.all(modes == np.array([0, 0]), axis=1))
        if hits.size != 1 or values.shape[:2] != (times.size, radii.size):
            raise ValueError(f"{PROFILE_PATHS[potential]}: malformed (0,0) profile bank")
        if not (np.all(np.diff(times) > 0) and np.all(np.diff(radii) > 0)):
            raise ValueError(f"{PROFILE_PATHS[potential]}: grids must increase")

        for target, phase_label, color in zip(targets, phase_labels, colors):
            index = int(np.argmin(np.abs(times - target)))
            time = float(times[index])
            u_3d = values[index, :, int(hits[0])] / sqrt4pi
            u_1d = _interpolate_snapshots(
                np.asarray(reference["snapshot_ts"], dtype=float),
                np.asarray(reference["snapshots_u"], dtype=float),
                time,
            )
            fit = fit_log_profile(radii, u_3d, (0.1, 0.5), order=1)
            ax.plot(
                reference["r"],
                u_1d,
                color=color,
                lw=1.25,
                label=phase_label,
                zorder=2,
            )
            ax.plot(
                radii,
                u_3d,
                linestyle="none",
                marker="o",
                ms=2.5,
                markerfacecolor="white",
                markeredgewidth=0.75,
                color=color,
                zorder=4,
            )
            ax.plot(
                radii,
                _o1_fit_curve(radii, fit),
                color=color,
                ls="--",
                lw=0.9,
                zorder=3,
            )

        ax.set_xscale("log")
        ax.set_xlim(0.095, 0.52)
        ax.set_xticks((0.1, 0.2, 0.5), labels=("0.1", "0.2", "0.5"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"$r/M$")
        if panel == 0:
            ax.set_ylabel(r"$u_{00}/\sqrt{4\pi}$")
        ax.text(
            0.97,
            0.93,
            POTENTIAL_STYLE[potential][1],
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=POTENTIAL_STYLE[potential][0],
        )
        _panel_label(ax, f"({chr(ord('a') + panel)})")
        _finish_axis(ax)

    time_legend = axes[0].legend(loc="lower left", ncol=3, columnspacing=0.8)
    axes[0].add_artist(time_legend)
    style_handles = [
        Line2D([], [], color=GRAY, lw=1.25, label="1D oracle"),
        Line2D(
            [], [], color=GRAY, marker="o", markerfacecolor="white", ls="none", label="3D"
        ),
        Line2D([], [], color=GRAY, lw=0.9, ls="--", label="3D o1 fit"),
    ]
    axes[1].legend(handles=style_handles, loc="lower left", ncol=3, columnspacing=0.8)
    return fig


def _production_lcs(production: dict[str, Any], l_value: int) -> list[float]:
    matrix = production.get("protocol", {}).get("matrix", [])
    lcs = sorted(
        {
            float(spec["lc"])
            for spec in matrix
            if int(spec["l"]) == l_value and str(spec["pot"]) == "linear"
        },
        reverse=True,
    )
    if not lcs:
        raise ValueError(f"production protocol has no l={l_value} rungs")
    return lcs


def _disc_ids(label: str) -> tuple[str, str]:
    # numbers.json ids: disc_<label>_l2_ratio_frozen and
    # disc_<label>_ratio_median_frozen (the latter owns its IQR).
    return (
        f"disc_{label}_l2_ratio_frozen",
        f"disc_{label}_ratio_median_frozen",
    )


def _plot_discriminator_point(
    ax: Any,
    x: float,
    label: str,
    numbers: NumberCatalog,
    *,
    l2_color: str = BLUE,
    median_color: str = ORANGE,
) -> tuple[float, float]:
    l2_id, median_id = _disc_ids(label)
    l2_value = numbers.scalar(l2_id)
    median = numbers.scalar(median_id)
    low, high = numbers.iqr(median_id)
    ax.plot(x, l2_value, marker="s", color=l2_color, zorder=4)
    ax.errorbar(
        x,
        median,
        yerr=np.array([[median - low], [high - median]]),
        fmt="o",
        color=median_color,
        capsize=2.5,
        elinewidth=0.9,
        zorder=3,
    )
    return l2_value, median


def build_discriminator(ctx: PlotContext) -> Figure:
    """Frozen L2 discriminator plus the secondary median/IQR by rung/mode."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.72), layout="constrained")
    numbers = ctx.numbers

    ax = axes[0]
    lcs = _production_lcs(ctx.production, 0)
    l2_values = []
    median_values = []
    for lc in lcs:
        label = f"l0_lc{lc:.3f}"
        l2, median = _plot_discriminator_point(ax, lc, label, numbers)
        l2_values.append(l2)
        median_values.append(median)
    ax.plot(lcs, l2_values, color=BLUE, lw=1.0, zorder=2)
    ax.plot(lcs, median_values, color=ORANGE, lw=0.8, alpha=0.8, zorder=2)
    oracle = numbers.scalar("disc_oracle_1d_l0_l2_ratio")
    ax.axhline(oracle, color=GREEN, ls="--", lw=1.1, label="1D oracle (L2)")
    ax.axhline(1.0, color=GRAY, ls=":", lw=0.8)
    ax.set_xlim(max(lcs) + 0.003, min(lcs) - 0.003)
    ax.set_xlabel(r"inner mesh scale $\ell_c/M$")
    ax.set_ylabel(r"$\widehat a/a_{\rm lin}$")
    _panel_label(ax, "(a)")
    _finish_axis(ax)

    ax = axes[1]
    modes = (0, 1, 2)
    for mode in modes:
        label = f"l{mode}_lc0.040"
        _plot_discriminator_point(ax, float(mode), label, numbers)
    ax.plot(
        0.0,
        oracle,
        marker="*",
        color=GREEN,
        ms=7.0,
        zorder=5,
        label="1D oracle (l=0, L2)",
    )
    ax.axhline(1.0, color=GRAY, ls=":", lw=0.8)
    ax.set_xticks(modes, labels=("0", "1", "2"))
    ax.set_xlim(-0.35, 2.35)
    ax.set_xlabel(r"multipole $l$ ($\ell_c=0.040M$)")
    ax.text(
        0.98,
        0.06,
        r"$l>0$: frozen, uncorrected",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=GRAY,
    )
    _panel_label(ax, "(b)")
    _finish_axis(ax)

    handles = [
        Line2D([], [], color=BLUE, marker="s", label="L2 ratio (primary)"),
        Line2D([], [], color=ORANGE, marker="o", label="median + IQR"),
        Line2D([], [], color=GREEN, ls="--", label="1D oracle (L2)"),
    ]
    axes[0].legend(handles=handles, loc="upper right")
    return fig


def _nullable_float_array(values: Iterable[Any]) -> np.ndarray:
    return np.asarray([np.nan if value is None else float(value) for value in values])


def build_o1_calibration(ctx: PlotContext) -> Figure:
    """Window calibration curves and common-support dev before/after."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.72), layout="constrained")
    numbers = ctx.numbers

    ax = axes[0]
    for potential in ("linear", "mexhat"):
        color, display = POTENTIAL_STYLE[potential]
        record = ctx.calibration["calibrations"][f"{potential}_l0"]
        times = np.asarray(record["t"], dtype=float)
        c_values = _nullable_float_array(record["variants"]["sampled_k32"]["c"])
        median_id = f"o1_c_{potential}_l0_sampled_k32_median"
        median = numbers.scalar(median_id)
        ax.plot(
            times,
            c_values,
            color=color,
            marker="o",
            ms=2.8,
            label=f"{display} (median {median:.3f})",
        )
    ax.axhline(1.0, color=GRAY, ls=":", lw=0.8)
    ax.set_xlim(3.9, 7.35)
    ax.set_xlabel(r"$t/M$")
    ax.set_ylabel(r"$c(t)=a_{o1[0.1,0.5]}/a_{\rm truth}$")
    ax.legend(loc="upper right")
    _panel_label(ax, "(a)")
    _finish_axis(ax)

    ax = axes[1]
    lcs = _production_lcs(ctx.production, 0)
    for potential in ("linear", "mexhat"):
        color, _ = POTENTIAL_STYLE[potential]
        before = []
        residual = []
        for lc in lcs:
            prefix = f"o1_{potential}_l0_lc{lc:.3f}"
            before.append(numbers.scalar(f"{prefix}_dev_before_common_support"))
            residual.append(numbers.scalar(f"{prefix}_dev_residual"))
        ax.plot(lcs, before, color=color, marker="o", label=f"{potential}: before")
        ax.plot(
            lcs,
            residual,
            color=color,
            marker="o",
            markerfacecolor="white",
            ls="--",
            label=f"{potential}: residual",
        )
        fine_prefix = f"o1_{potential}_l0_lc{min(lcs):.3f}"
        explained = numbers.scalar(f"{fine_prefix}_dev_fraction_explained")
        ax.annotate(
            f"{explained:.0f}%",
            xy=(min(lcs), residual[-1]),
            xytext=(3, -10 if potential == "linear" else 5),
            textcoords="offset points",
            color=color,
        )
    ax.set_yscale("log")
    ax.set_xlim(max(lcs) + 0.003, min(lcs) - 0.003)
    ax.set_xlabel(r"inner mesh scale $\ell_c/M$")
    ax.set_ylabel("median dev. from 1D (%)")
    ax.text(
        0.02,
        0.06,
        "strict common strong-phase support",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=GRAY,
    )
    ax.legend(loc="upper right", ncol=2, columnspacing=0.8)
    _panel_label(ax, "(b)")
    _finish_axis(ax)
    return fig


def build_qnm_systematics(ctx: PlotContext) -> Figure:
    """QNM real-part ladder and early/late damping systematic."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.72), layout="constrained")
    numbers = ctx.numbers

    ax = axes[0]
    lcs = []
    values = []
    errors = []
    for label in ("1.4", "1.0", "0.7"):
        entry_id = f"qnm_R20_lc{label}_re"
        lcs.append(float(label))
        values.append(numbers.scalar(entry_id))
        errors.append(numbers.symmetric(entry_id))
    ax.errorbar(
        lcs,
        values,
        yerr=errors,
        color=BLUE,
        marker="o",
        capsize=2.5,
        label=r"$R=20M$ window fan",
    )
    leaver_re = numbers.scalar("qnm_leaver_l2_re")
    ax.axhline(leaver_re, color=INK, ls="--", lw=1.0, label="Leaver")
    pooled_re = numbers.scalar("qnm_R40_late_pooled_re")
    pooled_re_std = numbers.symmetric("qnm_R40_late_pooled_re")
    ax.axhspan(
        pooled_re - pooled_re_std,
        pooled_re + pooled_re_std,
        color=GREEN,
        alpha=0.16,
        lw=0,
        label=r"$R=40M$ late pooled",
    )
    ax.axhline(pooled_re, color=GREEN, lw=1.0)
    ax.set_xlim(1.5, 0.6)
    ax.set_xlabel(r"mesh scale $\ell_c/M$")
    ax.set_ylabel(r"$\mathrm{Re}(M\omega)$")
    ax.legend(loc="upper right")
    _panel_label(ax, "(a)")
    _finish_axis(ax)

    ax = axes[1]
    bias_ids = (
        "qnm_R40_lc1match_early_overtone_bias",
        "qnm_R40_lc0.7match_early_overtone_bias",
        "qnm_R40_late_pooled_minus_im_error_signed",
    )
    biases = [numbers.scalar(entry_id) for entry_id in bias_ids]
    x = np.arange(3, dtype=float)
    ax.plot(x[:2], biases[:2], marker="o", color=ORANGE, ls="none", ms=5)
    ax.plot(x[2], biases[2], marker="s", color=GREEN, ls="none", ms=5)
    ax.plot(x, biases, color=GRAY, ls=":", lw=0.8, zorder=1)
    for xpos, value in zip(x, biases):
        ax.annotate(
            f"{value:+.1f}%",
            xy=(xpos, value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            color=ORANGE if xpos < 2 else GREEN,
        )
    ax.axhline(0.0, color=INK, ls="--", lw=1.0)
    ax.set_xticks(
        x,
        labels=(r"early" "\n" r"$\ell_c=1.0$", r"early" "\n" r"$\ell_c=0.7$", "late\npooled"),
    )
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(-0.7, 17.5)
    ax.set_ylabel(r"signed error in $-\mathrm{Im}(M\omega)$ (%)")
    ax.text(
        0.03,
        0.08,
        "early: unresolved overtone",
        transform=ax.transAxes,
        ha="left",
        color=ORANGE,
    )
    _panel_label(ax, "(b)")
    _finish_axis(ax, grid_axis="y")
    return fig


def _tail_spectrum(
    ts: np.ndarray, signal: np.ndarray, t_min: float
) -> tuple[np.ndarray, np.ndarray]:
    mask = ts >= t_min
    if np.count_nonzero(mask) < 16:
        raise ValueError("tail spectrum needs at least 16 samples")
    tail = np.asarray(signal[mask], dtype=float)
    tail = tail - float(np.mean(tail))
    window = np.hanning(tail.size)
    dt = float(np.mean(np.diff(ts[mask])))
    n_pad = 8 * tail.size
    frequencies = np.fft.rfftfreq(n_pad, dt) * 2.0 * np.pi
    amplitude = np.abs(np.fft.rfft(tail * window, n=n_pad)) / np.sum(window)
    return frequencies, amplitude


def build_cavity_domain(ctx: PlotContext) -> Figure:
    """Paired-domain waveform and tail spectrum exposing the R=20 cavity."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.72), layout="constrained")
    numbers = ctx.numbers
    r20 = _load_npz(CAVITY_WAVEFORM_PATH, ("ts", "c20"))
    r40 = _load_npz(DOMAIN_WAVEFORM_PATH, ("ts", "c20"))
    tail_t_min = float(ctx.spectroscopy["protocol"]["tail"]["t_min"])

    ax = axes[0]
    ax.semilogy(r20["ts"], np.abs(r20["c20"]), color=BLUE, label=r"$R=20M$")
    ax.semilogy(r40["ts"], np.abs(r40["c20"]), color=GREEN, label=r"$R=40M$ matched")
    tail_t_max = max(float(r20["ts"][-1]), float(r40["ts"][-1]))
    ax.axvspan(
        tail_t_min, tail_t_max, color=LIGHT_GRAY, alpha=0.55, lw=0
    )
    reduction = numbers.scalar("domain_R40_lc1match_tail_floor_reduction")
    ax.text(
        0.97,
        0.08,
        rf"tail floor reduced $\times {reduction:.1f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=GREEN,
    )
    ax.set_xlim(0.0, 70.0)
    ax.set_xlabel(r"$t/M$")
    ax.set_ylabel(r"$|c_{20}(t)|$")
    ax.legend(loc="upper right")
    _panel_label(ax, "(a)")
    _finish_axis(ax)

    ax = axes[1]
    for data, color, label in (
        (r20, BLUE, r"$R=20M$ tail"),
        (r40, GREEN, r"$R=40M$ tail"),
    ):
        frequency, amplitude = _tail_spectrum(
            np.asarray(data["ts"], dtype=float),
            np.asarray(data["c20"], dtype=float),
            tail_t_min,
        )
        ax.semilogy(frequency, amplitude, color=color, label=label)
    w1 = numbers.scalar("cavity_lc1_w1")
    w2 = numbers.scalar("cavity_lc1_w2")
    ax.axvline(w1, color=BLUE, ls=":", lw=0.9)
    ax.axvline(w2, color=BLUE, ls=":", lw=0.9)
    leaver_re = numbers.scalar("qnm_leaver_l2_re")
    ax.axvline(leaver_re, color=INK, ls="--", lw=1.0, label="Leaver QNM")
    ax.text(
        0.04,
        0.08,
        rf"trapped lines: ${w1:.3f},\ {w2:.3f}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=BLUE,
    )
    ax.set_xlim(0.18, 0.72)
    ax.set_xlabel(r"$M\omega$")
    ax.set_ylabel("windowed tail amplitude")
    ax.legend(loc="upper right")
    _panel_label(ax, "(b)")
    _finish_axis(ax)
    return fig


@dataclass(frozen=True)
class FigureSpec:
    name: str
    builder: Callable[[PlotContext], Figure]


FIGURE_SPECS = (
    FigureSpec("interior_profiles", build_interior_profiles),
    FigureSpec("interior_discriminator", build_discriminator),
    FigureSpec("o1_calibration", build_o1_calibration),
    FigureSpec("qnm_systematics", build_qnm_systematics),
    FigureSpec("cavity_domain", build_cavity_domain),
)


def expected_output_names() -> tuple[str, ...]:
    return tuple(f"{spec.name}.{extension}" for spec in FIGURE_SPECS for extension in FORMATS)


def _render_bytes(fig: Figure, extension: str) -> bytes:
    buffer = io.BytesIO()
    if extension == "pdf":
        metadata: dict[str, Any] = {
            "Creator": "scripts/paper_figures.py",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        }
        fig.savefig(buffer, format="pdf", metadata=metadata)
    elif extension == "png":
        fig.savefig(
            buffer,
            format="png",
            dpi=PNG_DPI,
            metadata={"Software": "scripts/paper_figures.py"},
        )
    else:
        raise ValueError(f"unsupported figure extension {extension!r}")
    return buffer.getvalue()


def render_all(context: PlotContext) -> dict[str, bytes]:
    """Render every expected output before returning any publishable bytes."""
    rendered: dict[str, bytes] = {}
    with plt.rc_context(paper_style()):
        for spec in FIGURE_SPECS:
            figure = spec.builder(context)
            try:
                for extension in FORMATS:
                    rendered[f"{spec.name}.{extension}"] = _render_bytes(figure, extension)
            finally:
                plt.close(figure)
    if tuple(rendered) != expected_output_names():
        raise RuntimeError("internal figure manifest mismatch")
    return rendered


def _atomic_write_bytes(path: Path, content: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == content:
        return False
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="render and byte-compare all outputs without writing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir.resolve()
    context = PlotContext.load()
    rendered = render_all(context)

    if args.check:
        stale = [
            name
            for name, content in rendered.items()
            if not (output_dir / name).is_file() or (output_dir / name).read_bytes() != content
        ]
        if stale:
            print("figure outputs missing/stale: " + ", ".join(stale), file=sys.stderr)
            return 1
        print(
            f"figures: {len(FIGURE_SPECS)} figures / {len(rendered)} files verified; "
            "outputs up to date"
        )
        return 0

    changed = [
        name
        for name, content in rendered.items()
        if _atomic_write_bytes(output_dir / name, content)
    ]
    state = f"updated {len(changed)}" if changed else "unchanged"
    try:
        display_dir = output_dir.relative_to(REPO)
    except ValueError:
        display_dir = output_dir
    print(
        f"figures: {len(FIGURE_SPECS)} figures / {len(rendered)} files -> "
        f"{display_dir} ({state})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
