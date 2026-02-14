import argparse
import os
import numpy as np

from psyop.analysis.qnm import compute_qnm, estimate_peak


def postprocess_main() -> int:
    parser = argparse.ArgumentParser(description="Postprocess PSYOP run directory")
    parser.add_argument("--run", required=True, help="Run directory path")
    parser.add_argument("--qnm", action="store_true", help="Compute QNM spectrum")
    parser.add_argument("--plots", action="store_true", help="Reserved flag for plotting")
    args = parser.parse_args()

    if not args.qnm:
        return 0

    ts_path = os.path.join(args.run, "time_series.txt")
    if not os.path.exists(ts_path):
        raise FileNotFoundError(f"time_series not found: {ts_path}")

    data = np.loadtxt(ts_path)
    if data.ndim != 2 or data.shape[0] < 8:
        raise ValueError("time_series requires at least 8 samples")
    dt = float(np.mean(np.diff(data[:, 0])))
    signal = data[:, 1]
    freqs, spec = compute_qnm(signal, dt)
    f_peak, s_peak = estimate_peak(freqs, spec)
    np.savetxt(os.path.join(args.run, "qnm_spectrum.txt"), np.column_stack([freqs, spec]))
    with open(os.path.join(args.run, "qnm_peak.txt"), "w", encoding="utf-8") as f:
        f.write(f"{f_peak:.12e} {s_peak:.12e}\n")
    return 0
