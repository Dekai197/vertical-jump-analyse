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


class PoseEstimator:
    def __init__(self, model_path: str, min_conf: float = 0.5):
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_conf,
            min_pose_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self._lm = vision.PoseLandmarker.create_from_options(opts)
        self._last_ts = -1
        self._xy, self._vis, self._det = [], [], []

    def process(self, frame_bgr: np.ndarray, t_ms: float) -> None:
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        scale = min(1.0, INFER_MAX_SIDE / max(h, w))
        small = cv2.resize(frame_bgr, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA) if scale < 1 else frame_bgr
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        ts = max(int(round(t_ms)), self._last_ts + 1)  # must be strictly increasing
        self._last_ts = ts
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        if res.pose_landmarks:
            pts = res.pose_landmarks[0]
            self._xy.append([[p.x * w, p.y * h] for p in pts])
            self._vis.append([p.visibility if p.visibility is not None else 1.0 for p in pts])
            self._det.append(True)
        else:
            self._xy.append(np.full((33, 2), np.nan))
            self._vis.append(np.zeros(33))
            self._det.append(False)

    def result(self, times: np.ndarray) -> PoseTrack:
        return PoseTrack(np.asarray(self._xy, dtype=float), np.asarray(self._vis, dtype=float),
                         np.asarray(self._det, dtype=bool), np.asarray(times, dtype=float))

    def close(self) -> None:
        self._lm.close()
