import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_create_example_config(tmp_path: Path):
    cli = pytest.importorskip("psyop.cli")

    config_path = tmp_path / "example.json"
    cli.create_example_config(str(config_path))

    assert config_path.exists()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert "mesh" in data and "solver" in data and "evolution" in data
