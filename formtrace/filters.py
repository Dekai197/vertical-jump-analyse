"""Small numpy-only signal helpers (no SciPy dependency)."""

from __future__ import annotations

import numpy as np


def fill_gaps(x: np.ndarray, valid: np.ndarray, max_gap: int | None = None) -> np.ndarray:
    """Linearly interpolate invalid samples along axis 0.

    ``x`` is (T, ...); ``valid`` is boolean and broadcastable to ``x`` after
    adding trailing axes (e.g. (T,) or (T, 33) for a (T, 33, 2) array).
    Gaps longer than ``max_gap`` samples, and samples before the first / after
    the last valid one, are left as NaN: inventing a straight line across a
    long occlusion would create fake motion.
    """
    x = np.asarray(x, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    while valid.ndim < x.ndim:
        valid = valid[..., None]
    valid = np.broadcast_to(valid, x.shape)
    T = len(x)
    flat = x.reshape(T, -1).copy()
    vflat = valid.reshape(T, -1)
    t = np.arange(T)
    for j in range(flat.shape[1]):
        ok = vflat[:, j] & np.isfinite(flat[:, j])
        col = np.full(T, np.nan)
        if ok.sum() >= 1:
            col[ok] = flat[ok, j]
        if ok.sum() >= 2:
            idx = t[ok]
            interp = np.interp(t, idx, flat[ok, j])
            inside = (t > idx[0]) & (t < idx[-1]) & ~ok
            if max_gap is not None:
                # length of the gap each sample sits in = next valid - previous valid - 1
                nxt = idx[np.searchsorted(idx, t, side="left").clip(max=len(idx) - 1)]
                prv = idx[(np.searchsorted(idx, t, side="right") - 1).clip(min=0)]
                inside &= (nxt - prv - 1) <= max_gap
            col[inside] = interp[inside]
        flat[:, j] = col
    return flat.reshape(x.shape)


def despike(x: np.ndarray, valid: np.ndarray, window: int, thresh: float) -> np.ndarray:
    """Hampel-style outlier mask for landmark tracks.

    ``x`` is (T, J, 2), ``valid`` (T, J). A sample is dropped (returned mask
    False) when its distance to the rolling median of the valid samples in a
    centred window exceeds ``thresh`` (pixels). Catches single-frame landmark
    swaps / jumps that smoothing would otherwise smear into fake motion.
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    if T < 3:
        return valid.copy()
    half = max(1, window // 2)
    xv = np.where(valid[..., None], x, np.nan)
    pad = np.concatenate([np.full((half,) + x.shape[1:], np.nan), xv,
                          np.full((half,) + x.shape[1:], np.nan)])
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1, axis=0)  # (T, J, 2, W)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(win, axis=-1)
    dev = np.linalg.norm(x - med, axis=-1)
    out = valid.copy()
    out &= ~(np.isfinite(dev) & (dev > thresh))
    return out


def medfilt(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Centred running median of a 1-D signal, NaN-aware (NaN stays NaN)."""
    x = np.asarray(x, dtype=float)
    half = window // 2
    pad = np.concatenate([np.full(half, np.nan), x, np.full(half, np.nan)])
    win = np.lib.stride_tricks.sliding_window_view(pad, window)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmedian(win, axis=1)
    out[~np.isfinite(x)] = np.nan
    return out


def savgol_coeffs(window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay smoothing coefficients (0th derivative, centred)."""
    if window % 2 == 0 or window <= polyorder:
        raise ValueError("window must be odd and > polyorder")
    half = window // 2
    k = np.arange(-half, half + 1, dtype=float)
    A = np.vander(k, polyorder + 1, increasing=True)
    return np.linalg.pinv(A)[0]


def _runs(ok: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index pairs of consecutive True values."""
    d = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def smooth(x: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay along axis 0, applied separately to each run of finite samples
    (edge-replicate padding inside a run), so NaN gaps neither spread nor get bridged."""
    x = np.asarray(x, dtype=float)
    if window < 3 or len(x) < 3:
        return x.copy()
    flat = x.reshape(len(x), -1)
    out = np.full_like(flat, np.nan)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        for a, b in _runs(np.isfinite(col)):
            seg = col[a:b]
            w = min(window, len(seg) if len(seg) % 2 == 1 else len(seg) - 1)
            if w <= polyorder or w < 3:
                out[a:b, j] = seg
                continue
            c = savgol_coeffs(w, polyorder)
            half = w // 2
            pad = np.concatenate([np.repeat(seg[:1], half), seg, np.repeat(seg[-1:], half)])
            out[a:b, j] = np.convolve(pad, c[::-1], mode="valid")
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


def find_peaks(x: np.ndarray, prominence: float) -> list[int]:
    """Indices of local maxima whose topographic prominence >= ``prominence``.

    Plateaus are reported at their centre. NaNs are treated as -inf.
    """
    x = np.where(np.isfinite(x), x, -np.inf)
    n = len(x)
    cands, i = [], 1
    while i < n - 1:
        if x[i] > x[i - 1]:
            j = i
            while j + 1 < n and x[j + 1] == x[i]:
                j += 1
            if j + 1 < n and x[j + 1] < x[i]:
                cands.append((i + j) // 2)
            i = j + 1
        else:
            i += 1
    peaks = []
    for p in cands:
        v = x[p]
        l = p
        lmin = v
        while l > 0 and x[l - 1] <= v:
            l -= 1
            lmin = min(lmin, x[l])
        r = p
        rmin = v
        while r < n - 1 and x[r + 1] <= v:
            r += 1
            rmin = min(rmin, x[r])
        # A side that runs into the array edge without meeting a higher point is still a valid base.
        if v - max(lmin, rmin) >= prominence:
            peaks.append(p)
    return peaks
