import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _write_series(run_dir: Path):
    series = run_dir / "series"
    series.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 5.0, 128)
    y = np.cos(2 * np.pi * 0.8 * t) * np.exp(-0.1 * t)
    data = np.column_stack([t, y])
    np.savetxt(series / "time_series.csv", data, delimiter=",", header="t,phi", comments="")


def test_postprocess_fft_outputs(tmp_path, monkeypatch):
    from psyop.cli import postprocess_main

    run_dir = tmp_path / "run"
    _write_series(run_dir)
    monkeypatch.setattr(sys, "argv", ["psyop-postprocess", "--run", str(run_dir), "--qnm", "--method", "fft"])
    assert postprocess_main() == 0
    assert (run_dir / "series" / "qnm_spectrum.csv").exists()
    assert (run_dir / "series" / "qnm_peak.json").exists()


def test_postprocess_prony_outputs(tmp_path, monkeypatch):
    from psyop.cli import postprocess_main

    run_dir = tmp_path / "run"
    _write_series(run_dir)
    monkeypatch.setattr(
        sys, "argv", ["psyop-postprocess", "--run", str(run_dir), "--qnm", "--method", "prony", "--modes", "2"]
    )
    assert postprocess_main() == 0
    assert (run_dir / "series" / "qnm_modes.json").exists()
