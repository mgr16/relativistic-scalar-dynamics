from pathlib import Path
import sys


def test_src_layout_exists():
    repo_root = Path(__file__).resolve().parent.parent
    assert (repo_root / "src" / "psyop").is_dir()


def test_import_from_src_layout():
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))
    import psyop  # noqa: F401
