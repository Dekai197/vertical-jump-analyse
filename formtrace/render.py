"""Overlay drawing: neon glow trails, skeleton, angle arcs, HUD panels.

OpenCV's Hershey fonts are ASCII-only, so on-video labels are short English /
numeric tags; the full Chinese explanation lives in the text report.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .landmarks import BONES

# BGR
CYAN = (255, 230, 0)
MAGENTA = (255, 60, 255)
LIME = (80, 255, 120)
ORANGE = (0, 170, 255)
YELLOW = (0, 240, 255)
RED = (60, 60, 255)
WHITE = (255, 255, 255)
GREY = (170, 170, 170)


def ui_scale(frame: np.ndarray) -> float:
    return max(frame.shape[:2]) / 1000.0


def _ipts(pts) -> np.ndarray:
    return np.round(np.asarray(pts, dtype=float)).astype(np.int32).reshape(-1, 1, 2)


def glow_polylines(frame: np.ndarray, segments: list[tuple[np.ndarray, tuple]],
                   thickness: float | None = None) -> None:
    """Draw neon polylines in place. ``segments`` = [(Nx2 points, BGR colour), ...].

    The glow is a blurred thick stroke added onto the frame, then a coloured
    stroke and a thin white core on top. Blur runs only on the bounding box.
    """
    segs = [(np.asarray(p, dtype=float), c) for p, c in segments]
    segs = [(p[np.all(np.isfinite(p), axis=1)], c) for p, c in segs]
    segs = [(p, c) for p, c in segs if len(p) >= 2]
    if not segs:
        return
    s = ui_scale(frame)
    th = max(2, int(round((thickness or 3.0) * s)))
    glow = max(6, int(round(14 * s)))
    allp = np.concatenate([p for p, _ in segs])
    h, w = frame.shape[:2]
    pad = glow * 3
    x0, y0 = np.maximum(np.floor(allp.min(0)).astype(int) - pad, 0)
    x1, y1 = np.minimum(np.ceil(allp.max(0)).astype(int) + pad, [w - 1, h - 1])
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1 + 1, x0:x1 + 1]
    layer = np.zeros_like(roi)
    off = np.array([x0, y0])
    for p, c in segs:
        cv2.polylines(layer, [_ipts(p - off)], False, c, glow, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), glow * 0.6)
    cv2.add(roi, layer, dst=roi)
    for p, c in segs:
        cv2.polylines(roi, [_ipts(p - off)], False, c, th, cv2.LINE_AA)
        cv2.polylines(roi, [_ipts(p - off)], False, WHITE, max(1, th // 3), cv2.LINE_AA)


def draw_skeleton(frame: np.ndarray, xy: np.ndarray, vis: np.ndarray,
                  near: set[int] | None = None, min_vis: float = 0.3) -> None:
    """Thin skeleton; ``near`` joints (camera side) are drawn brighter."""
    s = ui_scale(frame)
    near = near or set()
    for a, b in BONES:
        if not (np.all(np.isfinite(xy[[a, b]])) and vis[a] > min_vis and vis[b] > min_vis):
            continue
        hi = a in near and b in near
        cv2.line(frame, tuple(_ipts(xy[a])[0, 0]), tuple(_ipts(xy[b])[0, 0]),
                 WHITE if hi else GREY, max(1, int(round((2.5 if hi else 1.5) * s))), cv2.LINE_AA)
    for j in range(11, 33):
        if np.all(np.isfinite(xy[j])) and vis[j] > min_vis:
            r = max(2, int(round((4 if j in near else 2.5) * s)))
            cv2.circle(frame, tuple(_ipts(xy[j])[0, 0]), r, LIME if j in near else GREY, -1, cv2.LINE_AA)


def put_text(frame, text, org, scale=1.0, color=WHITE, thick=None, anchor="lt"):
    """Outlined text; ``scale`` is relative to frame size. anchor: lt|ct|lb."""
    s = ui_scale(frame) * 0.75 * scale
    t = max(1, int(round((thick or 2) * ui_scale(frame) * scale)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, t)
    x, y = int(org[0]), int(org[1])
    if anchor == "lt":
        y += th
    elif anchor == "ct":
        x -= tw // 2
        y += th
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), t + max(2, t), cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, color, t, cv2.LINE_AA)
    return tw, th + base


def hud(frame: np.ndarray, lines: list[tuple[str, tuple]], corner: str = "tl",
        title: str | None = None) -> None:
    """Semi-transparent info panel. ``lines`` = [(text, colour), ...]."""
    if not lines and not title:
        return
    s = ui_scale(frame)
    pad, gap = int(14 * s), int(10 * s)
    rows = ([(title, YELLOW)] if title else []) + lines
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, s * 0.75, max(1, int(2 * s)))[0] for t, _ in rows]
    w = max(tw for tw, _ in sizes) + 2 * pad
    h = sum(th for _, th in sizes) + gap * (len(rows) - 1) + 2 * pad
    H, W = frame.shape[:2]
    m = int(20 * s)
    x0 = m if corner in ("tl", "bl") else W - w - m
    y0 = m if corner in ("tl", "tr") else H - h - m
    x1, y1 = min(W, x0 + w), min(H, y0 + h)
    roi = frame[y0:y1, x0:x1]
    roi[:] = (roi * 0.35).astype(np.uint8)
    y = y0 + pad
    for (t, c), (_, th) in zip(rows, sizes):
        put_text(frame, t, (x0 + pad, y), 1.0, c)
        y += th + gap


def angle_arc(frame, a, b, c, color=YELLOW, label: str | None = None) -> None:
    """Draw the angle ABC as an arc at B with a label."""
    a, b, c = (np.asarray(v, dtype=float) for v in (a, b, c))
    if not np.all(np.isfinite([a, b, c])):
        return
    s = ui_scale(frame)
    r = int(min(np.linalg.norm(a - b), np.linalg.norm(c - b), 60 * s) * 0.6)
    if r < 4:
        return
    t1 = math.degrees(math.atan2(a[1] - b[1], a[0] - b[0]))
    t2 = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]))
    d = (t2 - t1 + 540) % 360 - 180  # signed smallest sweep
    center = tuple(_ipts(b)[0, 0])
    cv2.ellipse(frame, center, (r, r), 0, t1, t1 + d, color, max(2, int(3 * s)), cv2.LINE_AA)
    for p in (a, c):
        cv2.line(frame, center, tuple(_ipts(p)[0, 0]), color, max(2, int(3 * s)), cv2.LINE_AA)
    if label:
        mid = math.radians(t1 + d / 2)
        lx = b[0] - math.cos(mid) * r * 1.6
        ly = b[1] - math.sin(mid) * r * 1.6
        put_text(frame, label, (lx, ly), 1.2, color, anchor="ct")


def dashed_vline(frame, x, y0, y1, color=GREY, dash=None) -> None:
    s = ui_scale(frame)
    dash = dash or int(14 * s)
    y = int(min(y0, y1))
    while y < max(y0, y1):
        cv2.line(frame, (int(x), y), (int(x), int(min(y + dash, max(y0, y1)))), color,
                 max(1, int(2 * s)), cv2.LINE_AA)
        y += dash * 2


def banner(frame, text, color=YELLOW) -> None:
    """Large centred label below the HUD area (used on freeze frames); shrinks to fit."""
    scale = 1.8
    s = ui_scale(frame) * 0.75 * scale
    tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, 2)[0][0]
    if tw > frame.shape[1] * 0.92:
        scale *= frame.shape[1] * 0.92 / tw
    put_text(frame, text, (frame.shape[1] / 2, frame.shape[0] * 0.13), scale, color, anchor="ct")
