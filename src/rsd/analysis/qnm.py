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


def _prony_modes_raw(signal, dt, modes=1, svd_rank=None):
    """
    Núcleo Prony/matrix-pencil. Devuelve lista de dicts con
    frequency, decay, amplitude y phase, ordenada por amplitud descendente.
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

    # Filtrar autovalores no informativos o explosivos (|λ| >> 1 desborda
    # la Vandermonde y no corresponde a un modo QNM decayente)
    eigvals = np.array([
        lam for lam in eigvals
        if np.isfinite(lam) and abs(lam) > 1e-12 and abs(lam) < 1.5
    ])
    if eigvals.size == 0:
        return []

    # Amplitudes complejas por mínimos cuadrados sobre la Vandermonde
    steps = np.arange(n)[:, None]
    vander = eigvals[None, :] ** steps
    amps, *_ = np.linalg.lstsq(vander, x, rcond=None)

    results = []
    for lam, a in zip(eigvals, amps):
        omega = np.log(lam) / dt
        results.append(
            {
                "frequency": float(omega.imag / (2.0 * np.pi)),
                "decay": float(-omega.real),
                "amplitude": float(abs(a)),
                "phase": float(np.angle(a)),
            }
        )
    # Modos dominantes primero (antes el orden era el arbitrario de eigvals)
    results.sort(key=lambda mode: mode["amplitude"], reverse=True)
    return results


def estimate_qnm_prony(signal, dt, modes=1, svd_rank=None):
    """
    Estima frecuencias complejas usando método tipo Prony/matrix-pencil.
    Retorna lista de (freq_real, decay_rate) de los modos dominantes.
    """
    raw = _prony_modes_raw(signal, dt, modes=modes, svd_rank=svd_rank)
    return [(m["frequency"], m["decay"]) for m in raw[:modes]]


def estimate_qnm_prony_modes(signal, dt, modes=1, svd_rank=None):
    """
    Extended version with amplitude/phase/score and spurious-mode filtering.
    Multiple nearby orders are evaluated and stable modes across orders are retained.
    Amplitude and phase are fitted by least squares (no longer placeholders).
    """
    orders = sorted(set([max(1, int(modes)), max(1, int(modes) + 1), max(1, int(modes) + 2)]))
    candidates = []
    for order in orders:
        for mode in _prony_modes_raw(signal, dt, modes=order, svd_rank=order)[:order]:
            if not np.isfinite(mode["frequency"]) or not np.isfinite(mode["decay"]):
                continue
            if mode["decay"] < 0:
                continue
            candidates.append({**mode, "order": order})

    if not candidates:
        return []

    # Clustering por |frecuencia|: un modo real aparece como par conjugado
    # ±f y ambos representan el mismo modo físico. La tolerancia es el 2%
    # de Nyquist (la antigua 0.05/dt era tan ancha que fusionaba ±f en un
    # cluster cuya media se cancelaba a 0).
    freq_tol = 0.02 * 0.5 / max(dt, 1e-12)
    clusters = []
    for cand in candidates:
        abs_freq = abs(cand["frequency"])
        assigned = False
        for c in clusters:
            if abs(c["freq_mean"] - abs_freq) <= freq_tol:
                c["members"].append(cand)
                c["orders"].add(cand["order"])
                c["freq_mean"] = float(np.mean([abs(m["frequency"]) for m in c["members"]]))
                assigned = True
                break
        if not assigned:
            clusters.append(
                {
                    "members": [cand],
                    "orders": {cand["order"]},
                    "freq_mean": abs_freq,
                }
            )

    out = []
    for c in clusters:
        stability = len(c["orders"]) / len(orders)
        decay_mean = float(np.mean([m["decay"] for m in c["members"]]))
        # Amplitud/fase del miembro dominante del cluster
        best = max(c["members"], key=lambda m: m["amplitude"])
        score = stability / (1.0 + abs(decay_mean))
        out.append(
            {
                "frequency": float(c["freq_mean"]),
                "decay": decay_mean,
                "amplitude": float(best["amplitude"]),
                "phase": float(best["phase"]),
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
