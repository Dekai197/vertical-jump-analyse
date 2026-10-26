"""MediaPipe Pose Landmarker wrapper (Tasks API, VIDEO running mode)."""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")
VARIANTS = ("lite", "full", "heavy")
# MediaPipe resizes its input to 256x256 internally, so feeding full 4K frames
# only costs decode/convert time. Long side is capped at this many pixels.
INFER_MAX_SIDE = 960


def model_dir() -> str:
    return os.environ.get("FORMTRACE_MODEL_DIR",
                          os.path.join(os.path.expanduser("~"), ".cache", "formtrace"))


def ensure_model(variant: str = "full", path: str | None = None) -> str:
    """Return a local .task model path, downloading the official file if needed."""
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path
    if variant not in VARIANTS:
        raise ValueError(f"model variant must be one of {VARIANTS}")
    dst = os.path.join(model_dir(), f"pose_landmarker_{variant}.task")
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        urllib.request.urlretrieve(MODEL_URL.format(v=variant), tmp)
        os.replace(tmp, dst)
    return dst


@dataclass
class PoseTrack:
    """Per-frame landmarks in *original-frame pixel* coordinates."""
    xy: np.ndarray        # (T, 33, 2) float, NaN where not detected
    vis: np.ndarray       # (T, 33) visibility in [0, 1], 0 where not detected
    detected: np.ndarray  # (T,) bool
    times: np.ndarray = field(repr=False)  # (T,) seconds

    @property
    def detection_rate(self) -> float:
        return float(self.detected.mean()) if len(self.detected) else 0.0


def _center_scale(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Torso centre and torso length (shoulder-mid to hip-mid) of a (33, 2) pose."""
    sh = (pts[11] + pts[12]) / 2
    hp = (pts[23] + pts[24]) / 2
    return (sh + hp) / 2, float(np.linalg.norm(sh - hp))


class PoseEstimator:
    """Runs the landmarker and keeps following *one* athlete.

    Gyms are full of people in the background. A detection whose torso size
    is far from the subject's running median (e.g. a small background person
    after the athlete walks out of frame) is rejected instead of silently
    swapping identity.

    ``num_poses`` defaults to 1 on purpose: with 1 pose the VIDEO mode keeps
    tracking from the previous frame's landmarks, while with more poses the
    person detector runs much more often, and that detector keys on the face,
    which a weight plate hides in a side-view back squat. On the test clip,
    num_poses=2 dropped the share of frames with a pose from 86% to ~30%.
    """

    def __init__(self, model_path: str, min_conf: float = 0.5, track_conf: float = 0.3,
                 num_poses: int = 1):
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_conf,
            # Lower presence/tracking thresholds keep the tracker alive through
            # partial occlusion (plate in front of the head): on the squat test
            # clip 0.5 -> 0.3 raised frames with the lifter's pose from 47% to 88%.
            # Per-landmark visibility is still filtered downstream.
            min_pose_presence_confidence=track_conf,
            min_tracking_confidence=track_conf,
        )
        self._lm = vision.PoseLandmarker.create_from_options(opts)
        self._last_ts = -1
        self._xy, self._vis, self._det = [], [], []
        self._prev: tuple[np.ndarray, float] | None = None
        self._scales: list[float] = []
        self.rejected = 0

    def _choose(self, cands: list[np.ndarray]) -> int | None:
        cs = [_center_scale(c) for c in cands]
        if self._prev is None:
            return int(np.argmax([sc for _, sc in cs]))
        pc, ps = self._prev
        ref = float(np.median(self._scales[-90:]))
        costs = []
        for c, sc in cs:
            if not (0.6 * ref <= sc <= 1.67 * ref):
                costs.append(np.inf)
                continue
            costs.append(np.linalg.norm(c - pc) / max(ps, 1e-6) + 2 * abs(np.log(sc / ps)))
        k = int(np.argmin(costs))
        return None if not np.isfinite(costs[k]) else k

    def process(self, frame_bgr: np.ndarray, t_ms: float) -> np.ndarray | None:
        """Run on one frame; returns this frame's (33, 2) pixel landmarks or None."""
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        scale = min(1.0, INFER_MAX_SIDE / max(h, w))
        small = cv2.resize(frame_bgr, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA) if scale < 1 else frame_bgr
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        ts = max(int(round(t_ms)), self._last_ts + 1)  # must be strictly increasing
        self._last_ts = ts
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        cands = [np.array([[p.x * w, p.y * h] for p in lm]) for lm in res.pose_landmarks or []]
        k = self._choose(cands) if cands else None
        if cands and k is None:
            self.rejected += 1
        if k is not None:
            pts = res.pose_landmarks[k]
            self._xy.append(cands[k])
            self._vis.append([p.visibility if p.visibility is not None else 1.0 for p in pts])
            self._det.append(True)
            self._prev = _center_scale(cands[k])
            self._scales.append(self._prev[1])
            return cands[k].copy()
        else:
            self._xy.append(np.full((33, 2), np.nan))
            self._vis.append(np.zeros(33))
            self._det.append(False)
            return None

    def result(self, times: np.ndarray) -> PoseTrack:
        return PoseTrack(np.asarray(self._xy, dtype=float), np.asarray(self._vis, dtype=float),
                         np.asarray(self._det, dtype=bool), np.asarray(times, dtype=float))

    def close(self) -> None:
        self._lm.close()
