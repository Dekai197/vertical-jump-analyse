"""Barbell end tracking from a side view.

MediaPipe only models the human body, so the bar is tracked separately:

1. Initialise on the weight plate (its centre is the bar axis):
   - ``auto``: Hough circle search near the lifter's shoulders/wrists, or
   - ``manual``: the user gives a pixel point (x, y) or box (x, y, w, h).
2. Track with OpenCV CSRT, refining each frame with a local Hough circle fit
   so the centre does not slowly drift off the plate.
3. If no plate can be found, fall back to the near-side wrist as a *proxy*
   (labelled as such in the report; wrists sit near, not on, a back-squat bar).

From a side view only the near end of the bar is visible; the far end is
occluded by the lifter, so "both ends" cannot be measured with one camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .landmarks import SIDES

TRACK_MAX_SIDE = 960  # tracker works on a downscaled copy for speed


def _make_csrt():
    if hasattr(cv2, "TrackerCSRT") and hasattr(cv2.TrackerCSRT, "create"):
        return cv2.TrackerCSRT.create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    raise RuntimeError("OpenCV CSRT tracker missing: install opencv-contrib-python")


def find_circles(gray: np.ndarray, r_min: float, r_max: float, max_n: int = 8) -> np.ndarray:
    """Return up to ``max_n`` circles (x, y, r) found in a grayscale image."""
    r_min, r_max = max(4, int(r_min)), max(6, int(r_max))
    if r_max <= r_min:
        return np.empty((0, 3))
    g = cv2.GaussianBlur(gray, (0, 0), 1.5)
    found = []
    for method, p1, p2 in ((getattr(cv2, "HOUGH_GRADIENT_ALT", None), 300, 0.80),
                           (cv2.HOUGH_GRADIENT, 120, 40)):
        if method is None:
            continue
        c = cv2.HoughCircles(g, method, dp=1.5, minDist=r_min, param1=p1, param2=p2,
                             minRadius=r_min, maxRadius=r_max)
        if c is not None:
            found.extend(c.reshape(-1, 3).tolist())
        if found:
            break
    return np.asarray(found[:max_n], dtype=float).reshape(-1, 3)


@dataclass
class BarTrack:
    xy: np.ndarray            # (T, 2) bar-end centre in original pixels, NaN if unknown
    radius_px: float | None   # plate radius at initialisation (None for proxy)
    source: str               # "plate_auto" | "plate_manual" | "box_manual" | "wrist_proxy" | "none"
    refined: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # Hough-refined frames

    @property
    def coverage(self) -> float:
        return float(np.isfinite(self.xy[:, 0]).mean()) if len(self.xy) else 0.0


class BarTracker:
    """Frame consumer for :func:`formtrace.pipeline.extract`."""

    def __init__(self, init: tuple | None = None, init_frame: int = 0, side_hint: str | None = None):
        self.init = init
        self.init_frame = init_frame
        self.side_hint = side_hint
        self.tracker = None
        self.scale = 1.0
        self.r0 = None
        self.source = "none"
        self.xy: list = []
        self.refined: list = []
        self._lost = 0
        self._attempts = 0

    # -- initialisation -------------------------------------------------
    def _body_scale_and_anchors(self, pose_xy: np.ndarray) -> tuple[float, list[np.ndarray]]:
        sides = [self.side_hint] if self.side_hint else ["left", "right"]
        leg, anchors = [], []
        for sd in sides:
            j = SIDES[sd]
            L = (np.linalg.norm(pose_xy[j["hip"]] - pose_xy[j["knee"]])
                 + np.linalg.norm(pose_xy[j["knee"]] - pose_xy[j["ankle"]]))
            leg.append(L)
            anchors += [pose_xy[j["shoulder"]], pose_xy[j["wrist"]]]
        return float(np.nanmax(leg)), [a for a in anchors if np.all(np.isfinite(a))]

    def _auto_init(self, gray: np.ndarray, pose_xy: np.ndarray | None) -> tuple | None:
        if pose_xy is None:
            return None
        leg, anchors = self._body_scale_and_anchors(pose_xy * self.scale)
        if not np.isfinite(leg) or leg <= 0 or not anchors:
            return None
        # 45 cm plate vs ~0.8-1.0 m leg -> radius ~0.22-0.28 leg; allow a wide band.
        circles = find_circles(gray, 0.12 * leg, 0.40 * leg, max_n=12)
        if not len(circles):
            return None
        # Circles come back strongest-first. Take the strongest one near a
        # shoulder/wrist: "nearest" alone picks spurious arcs on the torso.
        d = np.min([np.linalg.norm(circles[:, :2] - a, axis=1) for a in anchors], axis=0)
        ok = np.nonzero(d < 0.6 * leg)[0]
        if not len(ok):
            return None
        x, y, r = circles[ok[0]]
        return float(x), float(y), float(r)

    def _manual_init(self, gray: np.ndarray) -> tuple:
        sx = self.scale
        if len(self.init) == 4:
            x, y, w, h = (v * sx for v in self.init)
            return x + w / 2, y + h / 2, max(w, h) / 2, "box_manual"
        x, y = self.init[0] * sx, self.init[1] * sx
        H, W = gray.shape
        guess = 0.06 * max(H, W)
        circles = find_circles(gray, guess * 0.4, guess * 2.5, max_n=12)
        if len(circles):
            d = np.linalg.norm(circles[:, :2] - [x, y], axis=1)
            k = np.argmin(d)
            if d[k] < circles[k, 2] * 0.6:
                return circles[k, 0], circles[k, 1], circles[k, 2], "plate_manual"
        return x, y, guess, "box_manual"

    def _start(self, small: np.ndarray, cx: float, cy: float, r: float) -> None:
        self.r0 = r
        box = (int(cx - r), int(cy - r), int(2 * r), int(2 * r))
        self.tracker = _make_csrt()
        self.tracker.init(small, box)

    # -- per frame ------------------------------------------------------
    def process(self, idx: int, frame: np.ndarray, pose_xy: np.ndarray | None) -> None:
        H, W = frame.shape[:2]
        self.scale = min(1.0, TRACK_MAX_SIDE / max(H, W))
        small = cv2.resize(frame, (round(W * self.scale), round(H * self.scale)),
                           interpolation=cv2.INTER_AREA) if self.scale < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        pt, refined = None, False
        if self.tracker is None and idx >= self.init_frame:
            if self.init is not None:
                cx, cy, r, self.source = self._manual_init(gray)
                self._start(small, cx, cy, r)
                pt = (cx, cy)
            elif self._attempts < 40 and (idx - self.init_frame) % 3 == 0:
                self._attempts += 1
                found = self._auto_init(gray, pose_xy)
                if found:
                    self.source = "plate_auto"
                    self._start(small, *found)
                    pt = found[:2]
        elif self.tracker is not None:
            ok, box = self.tracker.update(small)
            if ok:
                bx, by, bw, bh = box
                pt = (bx + bw / 2, by + bh / 2)
                if self.source != "box_manual":
                    pt, refined = self._refine(gray, pt)
                self._lost = 0
            else:
                self._lost += 1
        self.xy.append((pt[0] / self.scale, pt[1] / self.scale) if pt is not None else (np.nan, np.nan))
        self.refined.append(refined)

    def _refine(self, gray: np.ndarray, pt: tuple) -> tuple[tuple, bool]:
        r0 = self.r0
        x0, y0 = int(max(0, pt[0] - 1.6 * r0)), int(max(0, pt[1] - 1.6 * r0))
        x1, y1 = int(min(gray.shape[1], pt[0] + 1.6 * r0)), int(min(gray.shape[0], pt[1] + 1.6 * r0))
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            return pt, False
        c = find_circles(crop, 0.8 * r0, 1.2 * r0, max_n=4)
        if not len(c):
            return pt, False
        c[:, 0] += x0
        c[:, 1] += y0
        d = np.linalg.norm(c[:, :2] - pt, axis=1)
        k = int(np.argmin(d))
        if d[k] > 0.3 * r0:
            return pt, False
        return (float(c[k, 0]), float(c[k, 1])), True

    def result(self) -> BarTrack:
        xy = np.asarray(self.xy, dtype=float).reshape(-1, 2)
        r = self.r0 / self.scale if self.r0 else None
        return BarTrack(xy, r, self.source, np.asarray(self.refined, dtype=bool))


def wrist_proxy(wrist_xy: np.ndarray) -> BarTrack:
    return BarTrack(np.asarray(wrist_xy, dtype=float).copy(), None, "wrist_proxy",
                    np.zeros(len(wrist_xy), bool))


def save_cache(path: str, key: str, bar: BarTrack) -> None:
    np.savez_compressed(path, key=key, xy=bar.xy, refined=bar.refined, source=bar.source,
                        radius=np.nan if bar.radius_px is None else bar.radius_px)


def load_cache(path: str | None, key: str) -> BarTrack | None:
    import os
    if not path or not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if str(z["key"]) != key:
            return None
        r = float(z["radius"])
        return BarTrack(z["xy"], None if not np.isfinite(r) else r, str(z["source"]), z["refined"])
    except Exception:
        return None
