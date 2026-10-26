"""Video reading and writing.

Phone videos are frequently variable-frame-rate (VFR), and the container's
nominal FPS (``CAP_PROP_FPS``) can be wrong by >10% for such files. Every time
value in formtrace is therefore derived from per-frame presentation timestamps
whenever the decoder provides sane ones, and only falls back to ``idx / fps``
otherwise.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    nominal_fps: float
    n_frames: int
    times: np.ndarray = field(repr=False)  # seconds, one per decoded frame
    timestamps_source: str = "pts"  # "pts" or "nominal_fps"

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0]) if self.n_frames > 1 else 0.0

    @property
    def effective_fps(self) -> float:
        """Median frame rate from timestamps (robust to occasional dropped frames)."""
        if self.n_frames < 2:
            return self.nominal_fps
        dt = np.diff(self.times)
        dt = dt[dt > 0]
        return float(1.0 / np.median(dt)) if len(dt) else self.nominal_fps

    @property
    def max_frame_gap(self) -> float:
        return float(np.max(np.diff(self.times))) if self.n_frames > 1 else 0.0


def open_capture(path: str) -> cv2.VideoCapture:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    return cap


def iter_frames(path: str) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (index, pts_ms_or_nan, bgr_frame). Rotation metadata is applied by OpenCV."""
    cap = open_capture(path)
    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pts = cap.get(cv2.CAP_PROP_POS_MSEC)
            yield i, float(pts) if pts is not None else float("nan"), frame
            i += 1
    finally:
        cap.release()


def resolve_times(pts_ms: list[float], nominal_fps: float) -> tuple[np.ndarray, str]:
    """Return per-frame times in seconds plus which source was used."""
    n = len(pts_ms)
    pts = np.asarray(pts_ms, dtype=float) / 1000.0
    if n >= 2 and np.all(np.isfinite(pts)) and np.all(np.diff(pts) > 0):
        return pts - pts[0], "pts"
    fps = nominal_fps if nominal_fps and nominal_fps > 0 else 30.0
    return np.arange(n) / fps, "nominal_fps"


def probe(path: str) -> VideoInfo:
    """Decode the whole file once to get the true frame count and timestamps."""
    cap = open_capture(path)
    nominal = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    pts, w, h = [], 0, 0
    for _, t, frame in iter_frames(path):
        pts.append(t)
        h, w = frame.shape[:2]
    if not pts:
        raise IOError(f"no decodable frames in {path}")
    times, src = resolve_times(pts, nominal)
    return VideoInfo(path, w, h, nominal, len(pts), times, src)


class VideoWriter:
    """Writes frames with OpenCV, then (if ffmpeg exists) re-encodes to H.264.

    OpenCV wheels usually only ship the ``mp4v`` encoder, which browsers and
    many phones refuse to play; H.264 via ffmpeg fixes that.
    """

    def __init__(self, out_path: str, fps: float, size: tuple[int, int]):
        self.out_path = out_path
        self.fps = fps
        self._tmpdir = tempfile.mkdtemp(prefix="formtrace_")
        self._tmp = os.path.join(self._tmpdir, "raw.mp4")
        self._w = cv2.VideoWriter(self._tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self._w.isOpened():
            raise IOError("cv2.VideoWriter failed to open (mp4v)")
        self.encoder = "mp4v"

    def write(self, frame: np.ndarray) -> None:
        self._w.write(frame)

    def close(self) -> str:
        self._w.release()
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            cmd = [ffmpeg, "-y", "-v", "error", "-i", self._tmp, "-c:v", "libx264",
                   "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
                   "-movflags", "+faststart", self.out_path]
            if subprocess.run(cmd).returncode == 0:
                self.encoder = "h264"
        if self.encoder != "h264":
            shutil.move(self._tmp, self.out_path)
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        return self.out_path
