"""Regression checks for the R2 reproducible-environment artifacts."""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
YAML = REPO / "envs/rsd-dolfinx.yml"
LOCK = REPO / "envs/rsd-dolfinx-lock-osx-arm64.txt"


def test_environment_exports_are_present_scrubbed_and_platform_labeled():
    yaml_text = YAML.read_text(encoding="utf-8")
    lock_text = LOCK.read_text(encoding="utf-8")

    for text in (yaml_text, lock_text):
        assert "/Users/" not in text
        assert "marco" not in text.lower()
        assert "prefix:" not in text
    assert "# platform: osx-arm64" in lock_text
    assert "@EXPLICIT" in lock_text
    assert "name: rsd-dolfinx" in yaml_text


def test_portable_export_and_exact_lock_pin_the_audited_core():
    yaml_text = YAML.read_text(encoding="utf-8")
    lock_text = LOCK.read_text(encoding="utf-8")

    for package in (
        "cpython=3.10.19",
        "fenics-dolfinx=0.10.0",
        "gmsh=4.15.0",
        "petsc=3.24.4",
        "petsc4py=3.24.4",
    ):
        assert package in yaml_text
    for token in (
        "fenics-dolfinx-0.10.0-",
        "gmsh-4.15.0-",
        "petsc-3.24.4-",
        "petsc4py-3.24.4-",
        "python-3.10.19-",
    ):
        assert token in lock_text


def test_container_pins_and_documented_version_tables_match():
    workflow = (REPO / ".github/workflows/hpc.yml").read_text(encoding="utf-8")
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert "dolfinx/dolfinx:v0.10.0" in workflow
    assert "dolfinx/dolfinx:stable" not in workflow
    assert "fenics-dolfinx=0.10" in dockerfile

    required_rows = (
        "| Python | 3.10.19 |",
        "| DOLFINx | 0.10.0 |",
        "| PETSc / petsc4py | 3.24.4 / 3.24.4 |",
        "| Gmsh | 4.15.0 |",
    )
    for relative in ("README.md", "paper/README.md"):
        text = (REPO / relative).read_text(encoding="utf-8")
        assert "envs/rsd-dolfinx-lock-osx-arm64.txt" in text
        for row in required_rows:
            assert row in text
