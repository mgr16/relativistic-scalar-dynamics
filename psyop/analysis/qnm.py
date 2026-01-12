import numpy as np
from numpy.fft import rfft, rfftfreq


def compute_qnm(signal, dt, window="hann", pad_factor=4):
    x = np.asarray(signal, dtype=float)
    n = len(x)
    if n < 8:
        return np.array([0.0]), np.array([0.0])
    if window == "hann":
        w = np.hanning(n)
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
