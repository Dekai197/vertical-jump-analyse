"""2-D joint geometry. All angles are in degrees, image coordinates (y down)."""

from __future__ import annotations

import numpy as np


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle ABC at vertex B. 180 = fully straight, smaller = more flexed.

    Accepts (..., 2) arrays. Returns NaN where a segment has zero length.
    """
    a, b, c = (np.asarray(v, dtype=float) for v in (a, b, c))
    ba, bc = a - b, c - b
    nba = np.linalg.norm(ba, axis=-1)
    nbc = np.linalg.norm(bc, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(ba * bc, axis=-1) / (nba * nbc)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def lean_from_vertical(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Angle of segment lower->upper from vertical (0 = upright, 90 = horizontal)."""
    d = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    return np.degrees(np.arctan2(np.abs(d[..., 0]), -d[..., 1]))


def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), axis=-1)
