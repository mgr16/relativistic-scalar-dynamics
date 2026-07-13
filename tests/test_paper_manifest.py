"""Tests for the deterministic C5 paper artifact manifest."""

from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "paper_manifest.py"
SPEC = importlib.util.spec_from_file_location("paper_manifest_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest)


def _populate(repo: Path) -> None:
    for index, relative in enumerate(manifest.ARTIFACT_PATHS):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"artifact-{index}: {relative}\n".encode())


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_inventory_is_exact_sorted_reproducibility_closure():
    assert len(manifest.VERSIONED_INPUTS) == 11
    assert len(manifest.GENERATORS) == 4
    assert len(manifest.GENERATED_TABLES) == 3
    assert len(manifest.FIGURES) == 10
    assert len(manifest.MANUSCRIPT) == 4
    assert len(manifest.ARTIFACT_PATHS) == 32
    assert list(manifest.ARTIFACT_PATHS) == sorted(manifest.ARTIFACT_PATHS)
    assert len(set(manifest.ARTIFACT_PATHS)) == len(manifest.ARTIFACT_PATHS)
    assert set(manifest.GENERATORS) == {
        "scripts/paper_numbers.py",
        "scripts/paper_tex_numbers.py",
        "scripts/paper_figures.py",
        "scripts/paper_manifest.py",
    }
    assert set(manifest.GENERATED_TABLES) == {
        "docs/research/phase3/numbers.json",
        "docs/research/phase3/numbers.md",
        "paper/numbers.tex",
    }
    assert set(manifest.MANUSCRIPT) == {
        "paper/README.md", "paper/main.tex", "paper/refs.bib", "paper/main.pdf"
    }
    assert manifest.DEFAULT_OUTPUT.as_posix() not in manifest.ARTIFACT_PATHS
    assert not any(
        path.endswith((".aux", ".bbl", ".blg", ".log", ".out"))
        for path in manifest.ARTIFACT_PATHS
    )


def test_render_is_sorted_repo_relative_timestamp_free_and_deterministic(tmp_path):
    _populate(tmp_path)
    first = manifest.render_manifest(tmp_path)
    second = manifest.render_manifest(tmp_path)
    assert first == second
    lines = first.splitlines()
    paths = [line.split("  ", 1)[1] for line in lines]
    assert paths == list(manifest.ARTIFACT_PATHS)
    assert str(tmp_path) not in first
    assert not re.search(r"\b20\d\d[-:]", first)
    for relative, line in zip(paths, lines, strict=True):
        assert line == f"{_sha256((tmp_path / relative).read_bytes())}  {relative}"


def test_missing_escape_duplicate_and_self_are_rejected(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(FileNotFoundError):
        manifest.render_manifest(tmp_path, ("missing.dat",))
    with pytest.raises(ValueError, match="repo-relative"):
        manifest.render_manifest(tmp_path, ("../outside.dat",))
    with pytest.raises(ValueError, match="duplicate"):
        manifest.render_manifest(tmp_path, ("a", "a"))
    with pytest.raises(ValueError, match="must not contain itself"):
        manifest.render_manifest(tmp_path, (manifest.DEFAULT_OUTPUT.as_posix(),))


def test_artifact_symlink_and_symlinked_parent_are_rejected(tmp_path):
    target = tmp_path / "target.dat"
    target.write_bytes(b"target")
    link = tmp_path / "link.dat"
    try:
        link.symlink_to(target)
    except OSError as exc:  # pragma: no cover - platform without symlink support
        pytest.skip(f"symlinks unavailable: {exc}")
    with pytest.raises(ValueError, match="symlinks"):
        manifest.render_manifest(tmp_path, ("link.dat",))

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "nested.dat").write_bytes(b"nested")
    (tmp_path / "linked-dir").symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        manifest.render_manifest(tmp_path, ("linked-dir/nested.dat",))


def test_atomic_write_is_idempotent_and_cleans_temporary(tmp_path, monkeypatch):
    output = tmp_path / "paper" / "SOURCE_MANIFEST.sha256"
    replacements: list[tuple[Path, Path]] = []
    real_replace = manifest.os.replace

    def replace_spy(source, destination):
        replacements.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr(manifest.os, "replace", replace_spy)
    assert manifest._atomic_write(output, b"first\n")
    assert output.read_bytes() == b"first\n"
    assert replacements == [(output.with_name(f".{output.name}.tmp"), output)]
    assert not output.with_name(f".{output.name}.tmp").exists()

    mtime = output.stat().st_mtime_ns
    assert not manifest._atomic_write(output, b"first\n")
    assert output.stat().st_mtime_ns == mtime
    assert len(replacements) == 1


def test_check_never_writes_for_missing_stale_or_current_manifest(tmp_path):
    _populate(tmp_path)
    output = tmp_path / manifest.DEFAULT_OUTPUT

    assert manifest.main(["--repo", str(tmp_path), "--check"]) == 1
    assert not output.exists()

    output.write_bytes(b"stale\n")
    before = (output.read_bytes(), output.stat().st_mtime_ns)
    assert manifest.main(["--repo", str(tmp_path), "--check"]) == 1
    assert (output.read_bytes(), output.stat().st_mtime_ns) == before

    assert manifest.main(["--repo", str(tmp_path)]) == 0
    current = (output.read_bytes(), output.stat().st_mtime_ns)
    assert manifest.main(["--repo", str(tmp_path), "--check"]) == 0
    assert (output.read_bytes(), output.stat().st_mtime_ns) == current


def test_checked_in_manifest_matches_all_real_artifacts():
    output = REPO / manifest.DEFAULT_OUTPUT
    assert output.is_file()
    expected = manifest.render_manifest(REPO)
    assert output.read_text(encoding="utf-8") == expected
    assert manifest.main(["--check"]) == 0
