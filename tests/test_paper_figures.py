#!/usr/bin/env python3
"""Regresiones numpy/Matplotlib puras para el pipeline de figuras C3."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest


np = pytest.importorskip("numpy")
pytest.importorskip("matplotlib")

pytestmark = pytest.mark.requires_numpy

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "paper_figures.py"
MODULE_NAME = "paper_figures_under_test"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT)
assert SPEC is not None and SPEC.loader is not None
figures = importlib.util.module_from_spec(SPEC)
# ``dataclass`` resolves postponed annotations through ``sys.modules``.
sys.modules[MODULE_NAME] = figures
SPEC.loader.exec_module(figures)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_synthetic_profile(path: Path, amplitude_scale: float) -> None:
    """Small, well-conditioned (0,0) bank for the render-only fallback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.array([4.45, 5.98, 6.74])
    radii = np.geomspace(0.1, 0.5, 32)
    modes = np.array([[0, 0]], dtype=int)
    profiles = []
    for index, _time in enumerate(times):
        a = amplitude_scale * (0.08 + 0.015 * index)
        profile = (
            a * np.log(radii)
            + 0.12
            - 0.03 * radii * np.log(radii)
            + 0.02 * radii
        )
        profiles.append(np.sqrt(4.0 * np.pi) * profile)
    values = np.asarray(profiles, dtype=float)[:, :, None]
    np.savez(path, t=times, radii=radii, modes=modes, u=values)


@pytest.fixture(scope="module")
def rendered_bundle(tmp_path_factory):
    """Render the complete manifest twice, sharing the cost across the module."""
    patcher = pytest.MonkeyPatch()
    approved_profiles = all(path.is_file() for path in figures.PROFILE_PATHS.values())

    if approved_profiles:
        context = figures.PlotContext.load()
    else:
        synthetic_dir = tmp_path_factory.mktemp("paper-profile-inputs")
        synthetic_paths = {
            "linear": synthetic_dir / "linear.npz",
            "mexhat": synthetic_dir / "mexhat.npz",
        }
        _write_synthetic_profile(synthetic_paths["linear"], 1.0)
        _write_synthetic_profile(synthetic_paths["mexhat"], 0.9)
        patcher.setattr(figures, "PROFILE_PATHS", synthetic_paths)
        # Loading the remaining context directly keeps the smoke test independent
        # of a still-pending data-promotion decision while exercising all builders.
        context = figures.PlotContext(
            numbers=figures.NumberCatalog(),
            production=figures._load_json(figures.PRODUCTION_PATH),
            calibration=figures._load_json(figures.CALIBRATION_PATH),
            spectroscopy=figures._load_json(figures.SPECTROSCOPY_PATH),
        )

    first = figures.render_all(context)
    assert figures.plt.get_fignums() == []
    second = figures.render_all(context)
    assert figures.plt.get_fignums() == []
    try:
        yield context, first, second, approved_profiles
    finally:
        patcher.undo()


def test_versioned_input_manifest_stays_under_docs_and_never_uses_results():
    docs_root = (REPO / "docs/research").resolve()
    assert len(figures.VERSIONED_INPUTS) == len(set(figures.VERSIONED_INPUTS))
    for path in figures.VERSIONED_INPUTS:
        resolved = path.resolve()
        assert docs_root in resolved.parents, path
        relative = resolved.relative_to(REPO.resolve())
        assert "results" not in relative.parts, path


def test_profile_hashes_are_pinned_and_validation_detects_drift(
    tmp_path, monkeypatch
):
    assert set(figures.PROFILE_SHA256) == set(figures.PROFILE_PATHS)
    assert all(
        len(digest) == 64 and set(digest) <= set("0123456789abcdef")
        for digest in figures.PROFILE_SHA256.values()
    )

    # Once Marco's approved copies are present, pin the real bytes as well.
    existing = [path.is_file() for path in figures.PROFILE_PATHS.values()]
    assert len(set(existing)) == 1, "the two approved profile copies form one unit"
    if all(existing):
        for potential, path in figures.PROFILE_PATHS.items():
            assert _sha256(path) == figures.PROFILE_SHA256[potential]

    fake_repo = tmp_path / "repo"
    fake_paths = {
        potential: fake_repo / "docs/research/phase3/data" / f"{potential}.npz"
        for potential in ("linear", "mexhat")
    }
    for potential, path in fake_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"approved-{potential}".encode())
    fake_hashes = {key: _sha256(path) for key, path in fake_paths.items()}

    monkeypatch.setattr(figures, "REPO", fake_repo)
    monkeypatch.setattr(figures, "PROFILE_PATHS", fake_paths)
    monkeypatch.setattr(figures, "PROFILE_SHA256", fake_hashes)
    monkeypatch.setattr(figures, "VERSIONED_INPUTS", tuple(fake_paths.values()))
    figures.validate_versioned_inputs()

    fake_paths["linear"].write_bytes(b"drift")
    with pytest.raises(ValueError, match="profile source drift for linear"):
        figures.validate_versioned_inputs()


def test_number_catalog_rejects_non_citable_statuses_and_spot_checks_key_ids():
    catalog = figures.NumberCatalog()
    expected = {
        "disc_l0_lc0.028_l2_ratio_frozen": 0.9228305188279665,
        "o1_c_linear_l0_sampled_k32_median": 1.0201503900413962,
        "qnm_leaver_l2_re": 0.48364387221071303,
        "cavity_lc1_w1": 0.35350000000000026,
        "domain_R40_lc1match_tail_floor_reduction": 47.08859332684689,
    }
    for entry_id, value in expected.items():
        assert catalog.scalar(entry_id) == pytest.approx(value)

    with pytest.raises(ValueError, match="no-citable"):
        catalog.entry("disc_l0_lc0.028_peak_ratio_frozen")
    with pytest.raises(ValueError, match="degradado-a-prosa"):
        catalog.entry("cavity_doublet_headline_w1")


def test_expected_manifest_has_exactly_five_pdf_png_pairs():
    expected_stems = {
        "interior_profiles",
        "interior_discriminator",
        "o1_calibration",
        "qnm_systematics",
        "cavity_domain",
    }
    assert {spec.name for spec in figures.FIGURE_SPECS} == expected_stems
    assert len(figures.FIGURE_SPECS) == 5
    assert set(figures.expected_output_names()) == {
        f"{stem}.{extension}"
        for stem in expected_stems
        for extension in figures.FORMATS
    }


def test_interior_profile_uses_publication_multipole_label(rendered_bundle):
    context, _rendered, _second, _approved = rendered_bundle
    figure = figures.build_interior_profiles(context)
    try:
        assert figure.axes[0].get_ylabel() == r"$\phi_{00}$"
    finally:
        figures.plt.close(figure)


def test_all_builders_render_and_close_with_binary_signatures(rendered_bundle):
    context, rendered, _second, _approved = rendered_bundle
    assert tuple(rendered) == figures.expected_output_names()
    assert len(rendered) == 10
    assert all(callable(spec.builder) for spec in figures.FIGURE_SPECS)
    assert context.numbers.used_ids

    for name, content in rendered.items():
        assert len(content) > 1_000, name
        if name.endswith(".pdf"):
            assert content.startswith(b"%PDF-"), name
        else:
            assert content.startswith(b"\x89PNG\r\n\x1a\n"), name


def test_rendering_is_byte_deterministic(rendered_bundle):
    _context, first, second, _approved = rendered_bundle
    assert first.keys() == second.keys()
    assert {name: hashlib.sha256(data).digest() for name, data in first.items()} == {
        name: hashlib.sha256(data).digest() for name, data in second.items()
    }
    assert first == second


def test_atomic_write_is_idempotent_and_preserves_unchanged_mtime(
    tmp_path, monkeypatch
):
    output = tmp_path / "figures" / "figure.pdf"
    replacements = []
    real_replace = figures.os.replace

    def replace_spy(source, destination):
        replacements.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr(figures.os, "replace", replace_spy)
    assert figures._atomic_write_bytes(output, b"%PDF-new")
    assert output.read_bytes() == b"%PDF-new"
    assert len(replacements) == 1
    assert replacements[0][1] == output
    assert not output.with_name(f".{output.name}.tmp").exists()

    mtime = output.stat().st_mtime_ns
    assert not figures._atomic_write_bytes(output, b"%PDF-new")
    assert output.stat().st_mtime_ns == mtime
    assert len(replacements) == 1


def _snapshot(directory: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def test_main_check_handles_fresh_stale_and_missing_without_mutation(
    tmp_path, monkeypatch, rendered_bundle, capsys
):
    context, rendered, _second, _approved = rendered_bundle
    output_dir = tmp_path / "figures"
    monkeypatch.setattr(figures.PlotContext, "load", classmethod(lambda cls: context))
    monkeypatch.setattr(figures, "render_all", lambda _context: dict(rendered))
    args = ["--output-dir", str(output_dir)]

    # Missing outputs: --check reports failure and must not create the directory.
    assert figures.main([*args, "--check"]) == 1
    assert not output_dir.exists()
    assert "missing/stale" in capsys.readouterr().err

    assert figures.main(args) == 0
    fresh = _snapshot(output_dir)
    assert set(fresh) == set(figures.expected_output_names())
    assert figures.main(args) == 0
    assert _snapshot(output_dir) == fresh
    assert figures.main([*args, "--check"]) == 0
    assert _snapshot(output_dir) == fresh

    stale_path = output_dir / figures.expected_output_names()[0]
    stale_path.write_bytes(stale_path.read_bytes() + b"stale")
    stale = _snapshot(output_dir)
    assert figures.main([*args, "--check"]) == 1
    assert _snapshot(output_dir) == stale
    assert stale_path.name in capsys.readouterr().err

    stale_path.write_bytes(rendered[stale_path.name])
    missing_path = output_dir / figures.expected_output_names()[1]
    missing_path.unlink()
    missing = _snapshot(output_dir)
    assert figures.main([*args, "--check"]) == 1
    assert _snapshot(output_dir) == missing
    assert not missing_path.exists()
    assert missing_path.name in capsys.readouterr().err
