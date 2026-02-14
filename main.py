from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "src"))

if __name__ == "__main__":
    from psyop.cli import main

    raise SystemExit(main())
