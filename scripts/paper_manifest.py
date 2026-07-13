#!/usr/bin/env python3
"""Build the deterministic SHA-256 closure of the reproducible paper.

The manifest uses the conventional ``<sha256>  <repo-relative-path>`` form.
Its inventory is deliberately explicit: generators, the eleven versioned
scientific inputs, and every publication output.  It contains neither build
auxiliaries nor itself, and therefore has no timestamp or circular hash.

Writes are atomic and idempotent.  ``--check`` only compares the expected
bytes with the existing manifest and never creates or modifies a file.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable


REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PurePosixPath("paper/SOURCE_MANIFEST.sha256")

# Inputs consumed by paper_numbers.py plus the six NPZ inputs consumed by
# paper_figures.py.  numbers.json is an output, not an independent input.
VERSIONED_INPUTS: tuple[str, ...] = (
    "docs/research/phase0/pilot_oracle_summary.json",
    "docs/research/phase1/cavity/summary.json",
    "docs/research/phase1/cavity/waveform_l2_lc1.npz",
    "docs/research/phase2/exterior/data/wf_R40_lc1match.npz",
    "docs/research/phase2/exterior/spectroscopy.json",
    "docs/research/phase2/interior/data/ab_smoke_ref_linear_A0.1_n1600.npz",
    "docs/research/phase2/interior/data/ab_smoke_ref_mexhat_A0.1_n1600.npz",
    "docs/research/phase2/production/production.json",
    "docs/research/phase3/data/ab_smoke_3d_linear_l0_lc0.040.npz",
    "docs/research/phase3/data/ab_smoke_3d_mexhat_l0_lc0.040.npz",
    "docs/research/phase3/o1_calibration.json",
)

GENERATORS: tuple[str, ...] = (
    "scripts/paper_figures.py",
    "scripts/paper_manifest.py",
    "scripts/paper_numbers.py",
    "scripts/paper_tex_numbers.py",
)

GENERATED_TABLES: tuple[str, ...] = (
    "docs/research/phase3/numbers.json",
    "docs/research/phase3/numbers.md",
    "paper/numbers.tex",
)

FIGURES: tuple[str, ...] = tuple(
    f"paper/figures/{stem}.{extension}"
    for stem in (
        "cavity_domain",
        "interior_discriminator",
        "interior_profiles",
        "o1_calibration",
        "qnm_systematics",
    )
    for extension in ("pdf", "png")
)

MANUSCRIPT: tuple[str, ...] = (
    "paper/README.md",
    "paper/main.pdf",
    "paper/main.tex",
    "paper/refs.bib",
)

ARTIFACT_PATHS: tuple[str, ...] = tuple(
    sorted(GENERATORS + VERSIONED_INPUTS + GENERATED_TABLES + FIGURES + MANUSCRIPT)
)


def _validate_inventory(paths: Iterable[str]) -> tuple[str, ...]:
    """Return a sorted, unique inventory of safe repository-relative paths."""
    validated: list[str] = []
    for raw_path in paths:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"invalid empty/non-string artifact path: {raw_path!r}")
        if "\\" in raw_path:
            raise ValueError(f"artifact path must use POSIX separators: {raw_path!r}")
        path = PurePosixPath(raw_path)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError(f"artifact path must be normalized and repo-relative: {raw_path!r}")
        normalized = path.as_posix()
        if normalized != raw_path:
            raise ValueError(
                f"artifact path must be normalized and repo-relative: {raw_path!r}"
            )
        if normalized == DEFAULT_OUTPUT.as_posix():
            raise ValueError("the SHA-256 manifest must not contain itself")
        validated.append(normalized)
    if len(validated) != len(set(validated)):
        raise ValueError("artifact inventory contains duplicate paths")
    if validated != sorted(validated):
        raise ValueError("artifact inventory must be sorted")
    return tuple(validated)


def _reject_repo_symlink(path: Path, repo: Path) -> None:
    """Reject a symlink at the artifact or in its in-repository ancestry."""
    current = path
    while current != repo:
        if current.is_symlink():
            raise ValueError(f"manifest artifacts may not be symlinks: {path}")
        if repo not in current.parents:
            raise ValueError(f"artifact escaped repository: {path}")
        current = current.parent


def _artifact_file(repo: Path, relative: str) -> Path:
    repo = repo.resolve(strict=True)
    candidate = repo.joinpath(*PurePosixPath(relative).parts)
    _reject_repo_symlink(candidate, repo)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(repo)
    except ValueError as exc:
        raise ValueError(f"artifact escaped repository: {relative}") from exc
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_manifest(
    repo: Path = REPO, paths: Iterable[str] = ARTIFACT_PATHS
) -> str:
    """Hash every declared artifact and return deterministic manifest text."""
    inventory = _validate_inventory(tuple(paths))
    repo = repo.resolve(strict=True)
    lines = [f"{_sha256(_artifact_file(repo, path))}  {path}" for path in inventory]
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, content: bytes) -> bool:
    """Publish changed bytes atomically, preserving mtime when unchanged."""
    if path.is_symlink():
        raise ValueError(f"manifest output may not be a symlink: {path}")
    if path.is_file() and path.read_bytes() == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def _output_path(repo: Path, raw_output: Path) -> Path:
    repo = repo.resolve(strict=True)
    output = raw_output if raw_output.is_absolute() else repo / raw_output
    if output.is_symlink():
        raise ValueError(f"manifest output may not be a symlink: {output}")
    resolved_parent = output.parent.resolve(strict=False)
    try:
        resolved_parent.relative_to(repo)
    except ValueError as exc:
        raise ValueError(f"manifest output must stay inside repository: {output}") from exc
    return output


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=REPO,
        help="repository root (primarily useful for isolated verification)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(DEFAULT_OUTPUT.as_posix()),
        help="manifest path, relative to --repo unless absolute",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="compare the existing manifest without writing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo = args.repo.resolve(strict=True)
    output = _output_path(repo, args.output)
    expected = render_manifest(repo).encode("utf-8")

    if args.check:
        if not output.is_file() or output.read_bytes() != expected:
            print(f"paper manifest missing/stale: {output}", file=sys.stderr)
            return 1
        print(f"paper manifest: {len(ARTIFACT_PATHS)} artifacts verified; up to date")
        return 0

    changed = _atomic_write(output, expected)
    try:
        display = output.relative_to(repo).as_posix()
    except ValueError:
        display = str(output)
    print(
        f"paper manifest: {len(ARTIFACT_PATHS)} artifacts -> {display} "
        f"({'updated' if changed else 'unchanged'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
