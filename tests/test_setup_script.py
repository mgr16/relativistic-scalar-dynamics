"""
Test the setup_environment.sh script structure and content
"""
import os
from pathlib import Path


def test_setup_script_exists():
    """Verify setup_environment.sh exists"""
    repo_root = Path(__file__).resolve().parent.parent
    setup_script = repo_root / "setup_environment.sh"
    assert setup_script.exists(), "setup_environment.sh should exist"


def test_setup_script_is_executable():
    """Verify setup_environment.sh is executable"""
    repo_root = Path(__file__).resolve().parent.parent
    setup_script = repo_root / "setup_environment.sh"
    assert os.access(setup_script, os.X_OK), "setup_environment.sh should be executable"


def test_setup_script_has_shebang():
    """Verify setup_environment.sh has proper shebang"""
    repo_root = Path(__file__).resolve().parent.parent
    setup_script = repo_root / "setup_environment.sh"
    with open(setup_script) as f:
        first_line = f.readline().strip()
    assert first_line == "#!/bin/bash", "setup_environment.sh should have bash shebang"


def test_setup_script_creates_correct_env():
    """Verify the script creates psyop-dolfinx environment"""
    repo_root = Path(__file__).resolve().parent.parent
    setup_script = repo_root / "setup_environment.sh"
    content = setup_script.read_text()
    assert 'ENV_NAME="psyop-dolfinx"' in content, "Script should define psyop-dolfinx environment"
    assert "conda create -n ${ENV_NAME} python=3.10" in content, "Script should create conda environment with Python 3.10"


def test_setup_script_installs_dolfinx():
    """Verify the script installs DOLFINx"""
    repo_root = Path(__file__).resolve().parent.parent
    setup_script = repo_root / "setup_environment.sh"
    content = setup_script.read_text()
    assert "conda install -c conda-forge dolfinx" in content, "Script should install DOLFINx"


def test_setup_script_installs_dependencies():
    """Verify the script installs all required dependencies"""
    repo_root = Path(__file__).resolve().parent.parent
    setup_script = repo_root / "setup_environment.sh"
    content = setup_script.read_text()
    
    required_deps = ["gmsh", "numpy", "scipy", "matplotlib", "pytest", "pytest-cov", "pyyaml"]
    for dep in required_deps:
        assert dep in content, f"Script should install {dep}"


def test_setup_script_installs_package():
    """Verify the script installs the psyop package in development mode"""
    repo_root = Path(__file__).resolve().parent.parent
    setup_script = repo_root / "setup_environment.sh"
    content = setup_script.read_text()
    assert "pip install -e ." in content, "Script should install psyop package in development mode"


def test_setup_script_verifies_installation():
    """Verify the script includes verification steps"""
    repo_root = Path(__file__).resolve().parent.parent
    setup_script = repo_root / "setup_environment.sh"
    content = setup_script.read_text()
    assert "import dolfinx" in content, "Script should verify DOLFINx import"
    assert "import psyop" in content, "Script should verify psyop import"
