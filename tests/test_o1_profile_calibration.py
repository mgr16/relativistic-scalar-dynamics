#!/usr/bin/env python3
"""Regresiones numpy-puras para la calibracion o1 de C2.

Los tests fijan las decisiones metodologicas del HANDOFF sin leer las
corridas gitignored ni ejecutar el oraculo: deep truth congelada, fase fuerte
sin relleno, banco K=32 provisto por los NPZ, correccion de ambos miembros y
comparacion del discriminador sobre soporte comun.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.requires_numpy

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "o1_profile_calibration.py"
SPEC = importlib.util.spec_from_file_location("o1_profile_calibration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cal)


def _profile(r, a, b=0.17, z1=-0.23, e1=0.31, z2=0.0):
    """Jerarquia F-S o1, con un termino o2 opcional para sesgar la ventana."""
    return (
        a * np.log(r)
        + b
        + z1 * r * np.log(r)
        + e1 * r
        + z2 * r**2 * np.log(r)
    )


def _reference_data(z2=0.0):
    r = np.geomspace(0.01, 0.8, 1400)
    ts = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    # El gran valor en t=3 prueba que la escala se toma solo para t>=4.
    amplitudes = np.array([9.0, 0.20, 1.0, 0.29, -0.50])
    snapshots = np.stack(
        [_profile(r, a, z2=z2 * (1.0 + 0.1 * k)) for k, a in enumerate(amplitudes)]
    )
    return {"r": r, "snapshot_ts": ts, "snapshots_u": snapshots}, amplitudes


def _calibration(t, c, valid=None, variant="sampled_k32"):
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    if valid is None:
        valid = np.ones(t.shape, dtype=bool)
    return {
        "t": t,
        "strong_mask": np.asarray(valid, dtype=bool),
        "variants": {variant: {"c": c}},
    }


def _run(t, a, strong=None):
    t = np.asarray(t, dtype=float)
    a = np.asarray(a, dtype=float)
    if strong is None:
        strong = np.ones(t.shape, dtype=bool)
    return {
        "t": t,
        "a_primary": a,
        "strong_mask": np.asarray(strong, dtype=bool),
    }


def test_deep_truth_and_strong_ratio_have_no_fill():
    data, amplitudes = _reference_data()
    bank = np.geomspace(0.1, 0.5, 32)

    result = cal.compute_profile_calibration(data, bank)
    direct_truth = cal.fit_log_profile_series(
        data["r"], data["snapshots_u"], (0.02, 0.2), order=1
    )["a"]

    assert np.allclose(result["a_truth"], direct_truth, rtol=0.0, atol=2e-10)
    assert np.allclose(direct_truth, amplitudes, rtol=0.0, atol=2e-10)
    expected_strong = np.array([False, False, True, False, True])
    assert np.array_equal(result["strong_mask"], expected_strong)

    for variant in cal.VARIANTS:
        c = result["variants"][variant]["c"]
        # La variante K=32 interpola el perfil denso a radios que en general
        # no coinciden con nodos 1D; ese error de interpolacion es O(1e-6)
        # en este sintetico y no cambia la identidad fisica c=1.
        tolerance = 2e-10 if variant == "continuous" else 3e-6
        assert np.allclose(c[expected_strong], 1.0, rtol=0.0, atol=tolerance)
        assert np.all(np.isnan(c[~expected_strong]))

    serialized = cal.serialize_calibration(result)
    for variant in cal.VARIANTS:
        curve = serialized["variants"][variant]["c"]
        assert [value is None for value in curve] == (~expected_strong).tolist()

    # Los dos puntos fuertes estan separados por un hueco: ni el hueco ni
    # los extremos se completan/extrapolan al llevar c(t) a otra grilla.
    target = np.array([4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5])
    interpolated = cal.interpolate_valid_curve(
        result["t"],
        result["variants"]["sampled_k32"]["c"],
        result["strong_mask"],
        target,
    )
    assert np.array_equal(np.isfinite(interpolated), [False, True, False, False, False, True, False])


def test_sampled_k32_uses_the_supplied_bank_radii(monkeypatch):
    data, _ = _reference_data(z2=0.8)
    canonical = np.geomspace(0.1, 0.5, 32)
    supplied = canonical.copy()
    supplied[6:26] *= np.linspace(1.002, 0.998, 20)
    assert np.all(np.diff(supplied) > 0.0)
    assert not np.array_equal(supplied, canonical)

    seen = {}
    sample_original = cal.sample_snapshots_at_radii

    def sample_spy(r_dense, snapshots, radii):
        seen["radii"] = np.asarray(radii).copy()
        return sample_original(r_dense, snapshots, radii)

    monkeypatch.setattr(cal, "sample_snapshots_at_radii", sample_spy)
    result = cal.compute_profile_calibration(data, supplied)

    assert np.array_equal(seen["radii"], supplied)
    expected_profiles = np.vstack(
        [np.interp(supplied, data["r"], profile) for profile in data["snapshots_u"]]
    )
    expected_a = cal.fit_log_profile_series(
        supplied, expected_profiles, (0.1, 0.5), order=1
    )["a"]
    assert np.allclose(
        result["variants"]["sampled_k32"]["a_window"],
        expected_a,
        rtol=0.0,
        atol=1e-12,
    )


def test_corrected_discriminator_divides_both_pair_members_first():
    t = np.array([4.0, 5.0, 6.0])
    linear = _run(t, [2.0, 4.0, 6.0])
    mexhat = _run(t, [8.0, 16.0, 24.0])
    c_linear = _calibration(t, [2.0, 2.0, 2.0])
    c_mexhat = _calibration(t, [4.0, 4.0, 4.0])

    before, after = cal.corrected_discriminator(
        linear, mexhat, c_linear, c_mexhat, "sampled_k32"
    )

    assert before["n_samples"] == after["n_samples"] == 3
    for field in ("ratio_median", "peak_ratio", "l2_ratio"):
        assert before[field] == pytest.approx(4.0)
        assert after[field] == pytest.approx(2.0)
    assert before["ratio_iqr"] == pytest.approx([4.0, 4.0])
    assert after["ratio_iqr"] == pytest.approx([2.0, 2.0])


def test_discriminator_delta_uses_before_on_the_same_calibration_support():
    t = np.array([4.0, 5.0, 6.0, 7.0])
    linear = _run(t, [1.0, 1.0, 1.0, 1.0])
    mexhat = _run(t, [1.0, 100.0, 2.0, 4.0])
    valid = np.array([True, False, True, True])
    identity = _calibration(t, [1.0, np.nan, 1.0, 1.0], valid)

    raw_full = cal.raw_discriminator(linear, mexhat)
    before_support, after = cal.corrected_discriminator(
        linear, mexhat, identity, identity, "sampled_k32"
    )
    delta = cal.discriminator_delta(before_support, after)

    assert raw_full["n_samples"] == 4
    assert before_support["n_samples"] == after["n_samples"] == 3
    assert raw_full["ratio_median"] == pytest.approx(3.0)
    assert before_support["ratio_median"] == pytest.approx(2.0)
    assert delta["max_absolute"] == pytest.approx(0.0)
    assert not delta["review"] and not delta["alarm"]


def test_transfer_comparison_exposes_median_and_p95_gate_statistics():
    t = np.arange(4.0, 13.0)
    left = _calibration(t, np.ones(t.size))
    right_values = np.full(t.size, 1.01)
    right_values[4] = 1.10  # outlier interior, no atribuible a borde de segmento
    right = _calibration(t, right_values)

    comparison = cal.compare_calibration_curves(left, right, "sampled_k32")

    assert comparison["n_common"] == t.size
    assert comparison["median_relative"] == pytest.approx(0.01)
    assert comparison["median_relative"] <= 0.03
    assert comparison["p95_relative"] > 0.05
    assert comparison["max_relative"] == pytest.approx(0.10)


def _matrix_specs():
    specs = []
    for potential in ("linear", "mexhat"):
        for l, lcs in ((0, (0.056, 0.040, 0.028)), (1, (0.040,)), (2, (0.040, 0.028))):
            for lc in lcs:
                specs.append(
                    {
                        "pot": potential,
                        "l": l,
                        "lc": lc,
                        "label": f"{potential}_l{l}_lc{lc:.3f}",
                    }
                )
    return specs


def _disc(value=1.0):
    return {
        "n_samples": 4,
        "ratio_median": value,
        "ratio_iqr": [value, value],
        "peak_ratio": value,
        "l2_ratio": value,
    }


def test_reviewed_lgt0_runs_are_denied_for_own_truth_floor(monkeypatch, tmp_path):
    """El review de Fable reemplaza la razon tentativa del gate l=0.

    Se ejercita ``build_calibration`` con todos sus calculos costosos
    sustituidos por sinteticos; asi el test fija el JSON final sin depender
    de ``results/`` ni de los NPZ congelados.
    """
    matrix = _matrix_specs()
    frozen = {
        "protocol": {"matrix": matrix},
        "runs": {
            spec["label"]: {"dev_vs_1d_primary": {"max": 0.2, "median": 0.1}}
            for spec in matrix
        },
        "discriminator": {
            f"l{l}_lc{lc:.3f}": _disc()
            for l, lcs in ((0, (0.056, 0.040, 0.028)), (1, (0.040,)), (2, (0.040, 0.028)))
            for lc in lcs
        },
        "ladder": {},
    }
    frozen_path = tmp_path / cal.FROZEN_PRODUCTION
    frozen_path.parent.mkdir(parents=True)
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")

    run_paths = {
        spec["label"]: tmp_path / "inputs" / f"{spec['label']}.npz" for spec in matrix
    }
    bank = np.geomspace(0.1, 0.5, 32)
    fake_arrays = {
        "radii": bank,
        "t": np.array([4.0, 5.0]),
        "modes": np.array([[0, 0]]),
        "u": np.zeros((2, 32, 1)),
        "r": np.geomspace(0.01, 0.8, 64),
        "snapshot_ts": np.array([4.0, 5.0]),
        "snapshots_u": np.zeros((2, 64)),
    }

    def fake_calibration():
        variants = {
            name: {"c": np.ones(2), "summary": {"n": 2, "median": 1.0}}
            for name in cal.VARIANTS
        }
        return {
            "t": np.array([4.0, 5.0]),
            "a_truth": np.ones(2),
            "strong_mask": np.ones(2, dtype=bool),
            "truth_scan_floor_max": 0.10,
            "truth_scans": {},
            "variants": variants,
        }

    comparison = {
        "n_common": 5,
        "median_relative": 0.01,
        "max_relative": 0.10,
        "p95_relative": 0.07,
    }
    dev = {
        "before_full_support": {"n": 2, "max": 0.2, "median": 0.1},
        "before_on_calibration_support": {"n": 2, "max": 0.2, "median": 0.1},
        "after_on_calibration_support": {"n": 2, "max": 0.1, "median": 0.05},
        "median_dev_fraction_explained": 0.5,
        "n_calibration_support": 2,
    }

    monkeypatch.setattr(cal, "resolve_run_series", lambda repo, specs: run_paths)
    monkeypatch.setattr(cal, "_resolve_one", lambda pattern: tmp_path / "inputs" / "ml.npz")
    monkeypatch.setattr(cal, "_load_npz", lambda path, required: fake_arrays)
    monkeypatch.setattr(cal, "_validate_run", lambda data, path, l: None)
    monkeypatch.setattr(cal, "_validate_reference", lambda data, path: None)
    monkeypatch.setattr(cal, "_sha256", lambda path: "0" * 64)
    monkeypatch.setattr(cal, "compute_profile_calibration", lambda data, radii: fake_calibration())
    monkeypatch.setattr(cal, "compare_calibration_curves", lambda *args: dict(comparison))
    monkeypatch.setattr(cal, "serialize_calibration", lambda calibration: {})
    monkeypatch.setattr(cal, "_fit_run", lambda data, l: _run([4.0, 5.0], [1.0, 1.0]))
    monkeypatch.setattr(cal, "compute_dev_metrics", lambda *args: dict(dev))
    monkeypatch.setattr(cal, "_assert_frozen_dev", lambda *args, **kwargs: None)
    monkeypatch.setattr(cal, "raw_discriminator", lambda *args: _disc())
    monkeypatch.setattr(cal, "_assert_frozen_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(cal, "corrected_discriminator", lambda *args: (_disc(), _disc(1.01)))
    monkeypatch.setattr(
        cal,
        "_deficit_verdict",
        lambda *args: ({"classification": "partial"}, {"calibration_floor_max": 0.10}),
    )

    payload, _ = cal.build_calibration(tmp_path)

    for gate in payload["l_gt0_transfer_gate"].values():
        assert not gate["applied_to_l_gt0"]
        assert gate["moot"]
    reviewed_reason = "own-truth TRUTH_SCAN floor ≥ effect"
    for label, run in payload["runs"].items():
        if run["l"] == 0:
            continue
        for variant in run["variants"].values():
            assert not variant["applied"]
            assert reviewed_reason in variant["reason"]
    assert payload["status"] == "reviewed-diagnostic"
    assert payload["stop_required"] is False
    assert payload["review"] == cal.REVIEW
