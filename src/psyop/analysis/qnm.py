import numpy as np
from numpy.fft import rfft, rfftfreq


def compute_qnm(signal, dt, window="hann", pad_factor=4, detrend=True):
    x = np.asarray(signal, dtype=float)
    n = len(x)
    if n < 8:
        return np.array([0.0]), np.array([0.0])
    if detrend:
        x = x - np.mean(x)
    if window == "hann":
        x = x * np.hanning(n)
    elif window == "tukey":
        r = np.linspace(0, 1, n)
        alpha = 0.5
        w = np.ones(n)
        left = r < alpha / 2
        right = r > 1 - alpha / 2
        w[left] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (r[left] - alpha / 2)))
        w[right] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (r[right] - 1 + alpha / 2)))
        x = x * w
    Nfft = int(2 ** np.ceil(np.log2(n))) * max(int(pad_factor), 1)
    spec = np.abs(rfft(x, n=Nfft))
    freqs = rfftfreq(Nfft, d=dt)
    return freqs, spec


def estimate_qnm_prony(signal, dt, modes=1, svd_rank=None):
    """
    Estima frecuencias complejas usando método tipo Prony/matrix-pencil.
    Retorna lista de (freq_real, decay_rate).
    """
    x = np.asarray(signal, dtype=float)
    n = len(x)
    if n < 2 * modes + 4:
        return []

    m = n // 2
    if m <= modes:
        return []

    hankel0 = np.column_stack([x[i:i + m] for i in range(n - m)])
    hankel1 = np.column_stack([x[i + 1:i + 1 + m] for i in range(n - m)])

    if svd_rank is None:
        svd_rank = max(modes, 1)

    u, s, vh = np.linalg.svd(hankel0, full_matrices=False)
    u = u[:, :svd_rank]
    s = s[:svd_rank]
    vh = vh[:svd_rank, :]
    s_inv = np.diag(1.0 / s)

    pencil = u.T @ hankel1 @ vh.T @ s_inv
    eigvals = np.linalg.eigvals(pencil)

    results = []
    for lam in eigvals[:modes]:
        if lam == 0:
            continue
        omega = np.log(lam) / dt
        freq = omega.imag / (2.0 * np.pi)
        decay = -omega.real
        results.append((freq, decay))
    return results


def estimate_qnm_prony_modes(signal, dt, modes=1, svd_rank=None):
    """
    Extended version with amplitude/phase/score and spurious-mode filtering.
    Multiple nearby orders are evaluated and stable modes across orders are retained.
    """
    x = np.asarray(signal, dtype=float)
    amp = float(np.max(np.abs(x))) if x.size else 0.0
    orders = sorted(set([max(1, int(modes)), max(1, int(modes) + 1), max(1, int(modes) + 2)]))
    candidates = []
    for order in orders:
        for freq, decay in estimate_qnm_prony(signal, dt, modes=order, svd_rank=svd_rank):
            if not np.isfinite(freq) or not np.isfinite(decay):
                continue
            if decay < 0:
                continue
            candidates.append((float(freq), float(decay), order))

    if not candidates:
        return []

    # Simple clustering by frequency proximity
    freq_tol = 0.05 / max(dt, 1e-12)
    clusters = []
    for freq, decay, order in candidates:
        assigned = False
        for c in clusters:
            if abs(c["freq_mean"] - freq) <= freq_tol:
                c["freqs"].append(freq)
                c["decays"].append(decay)
                c["orders"].add(order)
                c["freq_mean"] = float(np.mean(c["freqs"]))
                c["decay_mean"] = float(np.mean(c["decays"]))
                assigned = True
                break
        if not assigned:
            clusters.append(
                {
                    "freqs": [freq],
                    "decays": [decay],
                    "orders": {order},
                    "freq_mean": freq,
                    "decay_mean": decay,
                }
            )

    out = []
    for c in clusters:
        stability = len(c["orders"]) / len(orders)
        score = stability / (1.0 + abs(c["decay_mean"]))
        out.append(
            {
                "frequency": float(c["freq_mean"]),
                "decay": float(c["decay_mean"]),
                "amplitude": amp,
                "phase": 0.0,
                "score": float(score),
                "stability": float(stability),
            }
        )

    out.sort(key=lambda m: m["score"], reverse=True)
    return out[: max(1, int(modes))]


def estimate_peak(freqs, spec):
    """Interpolación cuadrática del pico principal."""
    i = int(np.argmax(spec))
    if i == 0 or i == len(spec) - 1:
        return freqs[i], spec[i]
    y0, y1, y2 = spec[i-1], spec[i], spec[i+1]
    denom = (y0 - 2*y1 + y2)
    if abs(denom) < 1e-12:
        return freqs[i], y1
    delta = 0.5 * (y0 - y2) / denom  # desplazamiento relativo [-0.5,0.5]
    f_peak = freqs[i] + delta * (freqs[1] - freqs[0])
    s_peak = y1 - 0.25 * (y0 - y2) * delta
    return f_peak, s_peak
