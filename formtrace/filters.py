"""Small numpy-only signal helpers (no SciPy dependency)."""

from __future__ import annotations

import numpy as np


def fill_gaps(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linearly interpolate invalid samples along axis 0; edges are held constant.

    ``x`` is (T, ...) and ``valid`` is (T,) or broadcastable to x's leading axis.
    Returns NaNs everywhere if fewer than 2 valid samples exist.
    """
    x = np.asarray(x, dtype=float)
    out = x.copy()
    t = np.arange(len(x))
    flat = out.reshape(len(x), -1)
    valid = np.asarray(valid, dtype=bool).reshape(len(x), -1)
    if valid.shape[1] == 1 and flat.shape[1] > 1:
        valid = np.repeat(valid, flat.shape[1], axis=1)
    for j in range(flat.shape[1]):
        ok = valid[:, j] & np.isfinite(flat[:, j])
        if ok.sum() < 2:
            flat[:, j] = np.nan
            continue
        flat[:, j] = np.interp(t, t[ok], flat[ok, j])
    return flat.reshape(x.shape)


def savgol_coeffs(window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay smoothing coefficients (0th derivative, centred)."""
    if window % 2 == 0 or window <= polyorder:
        raise ValueError("window must be odd and > polyorder")
    half = window // 2
    k = np.arange(-half, half + 1, dtype=float)
    A = np.vander(k, polyorder + 1, increasing=True)
    return np.linalg.pinv(A)[0]


def smooth(x: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay along axis 0 with edge-replicate padding. NaNs propagate."""
    x = np.asarray(x, dtype=float)
    if window < 3 or len(x) < 3:
        return x.copy()
    window = min(window, len(x) if len(x) % 2 == 1 else len(x) - 1)
    if window <= polyorder:
        return x.copy()
    c = savgol_coeffs(window, polyorder)
    half = window // 2
    flat = x.reshape(len(x), -1)
    pad = np.concatenate([np.repeat(flat[:1], half, 0), flat, np.repeat(flat[-1:], half, 0)])
    out = np.empty_like(flat)
    for j in range(flat.shape[1]):
        out[:, j] = np.convolve(pad[:, j], c[::-1], mode="valid")
    return out.reshape(x.shape)


def window_for(fps: float, seconds: float, minimum: int = 3) -> int:
    """Odd window length covering roughly ``seconds`` at ``fps``."""
    w = max(minimum, int(round(fps * seconds)))
    return w if w % 2 == 1 else w + 1


def derivative(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """d x / d t along axis 0, using actual (possibly uneven) timestamps."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.zeros_like(x)
    return np.gradient(x, np.asarray(t, dtype=float), axis=0)
