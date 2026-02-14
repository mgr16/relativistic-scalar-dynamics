import argparse
import json
import os
import numpy as np

from psyop.analysis.qnm import compute_qnm, estimate_peak, estimate_qnm_prony_modes


def postprocess_main() -> int:
    parser = argparse.ArgumentParser(description="Postprocess PSYOP run directory")
    parser.add_argument("--run", required=True, help="Run directory path")
    parser.add_argument("--qnm", action="store_true", help="Compute QNM spectrum")
    parser.add_argument("--plots", action="store_true", help="Reserved flag for plotting")
    parser.add_argument("--window", default="hann", choices=["hann", "tukey"], help="Window for FFT")
    parser.add_argument("--method", default="fft", choices=["fft", "prony"], help="QNM method")
    parser.add_argument("--modes", type=int, default=1, help="Number of modes for Prony (>=1)")
    args = parser.parse_args()

    if not args.qnm:
        return 0

    series_dir = os.path.join(args.run, "series")
    os.makedirs(series_dir, exist_ok=True)
    plots_dir = os.path.join(args.run, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    ts_csv = os.path.join(series_dir, "time_series.csv")
    ts_txt = os.path.join(args.run, "time_series.txt")
    if os.path.exists(ts_csv):
        data = np.loadtxt(ts_csv, delimiter=",", skiprows=1)
    elif os.path.exists(ts_txt):
        data = np.loadtxt(ts_txt)
    else:
        raise FileNotFoundError(f"time_series not found in {ts_csv} or {ts_txt}")

    if data.ndim != 2 or data.shape[0] < 8 or data.shape[1] < 2:
        raise ValueError("time_series requires at least 8 samples with 2 columns")
    if args.modes < 1:
        raise ValueError("--modes must be >= 1")
    dt = float(np.mean(np.diff(data[:, 0])))
    signal = data[:, 1]
    if args.method == "prony":
        prony_modes = estimate_qnm_prony_modes(signal, dt, modes=args.modes)
        with open(os.path.join(series_dir, "qnm_modes.json"), "w", encoding="utf-8") as f:
            json.dump(prony_modes, f, indent=2)
        if prony_modes:
            rows = [[m["frequency"], m["decay"], m["amplitude"], m["phase"], m["score"]] for m in prony_modes]
            np.savetxt(
                os.path.join(series_dir, "qnm_modes.csv"),
                np.asarray(rows, dtype=float),
                delimiter=",",
                header="frequency,decay,amplitude,phase,score",
                comments="",
            )
    else:
        freqs, spec = compute_qnm(signal, dt, window=args.window, detrend=True)
        f_peak, s_peak = estimate_peak(freqs, spec)
        np.savetxt(
            os.path.join(series_dir, "qnm_spectrum.csv"),
            np.column_stack([freqs, spec]),
            delimiter=",",
            header="freq,spectrum",
            comments="",
        )
        with open(os.path.join(series_dir, "qnm_peak.json"), "w", encoding="utf-8") as f:
            json.dump({"frequency": f_peak, "spectrum": s_peak}, f, indent=2)
        if args.plots:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.plot(freqs, spec)
                ax.set_xlabel("frequency")
                ax.set_ylabel("spectrum")
                ax.set_title("QNM spectrum")
                fig.savefig(os.path.join(plots_dir, "qnm_spectrum.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception:
                pass
    return 0
