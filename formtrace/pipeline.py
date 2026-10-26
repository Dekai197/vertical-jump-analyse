"""Two-pass pipeline: (1) decode + pose (+ optional trackers), (2) render overlay."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from . import filters, geometry
from .landmarks import SIDES, camera_side
from .pose import PoseEstimator, PoseTrack, ensure_model
from .video import VideoInfo, VideoWriter, iter_frames, open_capture, resolve_times

import cv2


class FrameConsumer(Protocol):
    """Anything that needs to see every decoded frame during pass 1 (e.g. bar tracker)."""
    def process(self, idx: int, frame: np.ndarray) -> None: ...


@dataclass
class Session:
    info: VideoInfo
    raw: PoseTrack
    xy: np.ndarray          # (T, 33, 2) gap-filled + smoothed pixel coords
    vis: np.ndarray         # (T, 33)
    side: str               # camera-facing body side
    timings: dict = field(default_factory=dict)

    @property
    def t(self) -> np.ndarray:
        return self.info.times

    def joint(self, name: str) -> np.ndarray:
        """(T, 2) trajectory of a near-side joint, e.g. ``joint("knee")``."""
        return self.xy[:, SIDES[self.side][name]]

    def knee_angle(self) -> np.ndarray:
        return geometry.joint_angle(self.joint("hip"), self.joint("knee"), self.joint("ankle"))

    def hip_angle(self) -> np.ndarray:
        return geometry.joint_angle(self.joint("shoulder"), self.joint("hip"), self.joint("knee"))

    def leg_length(self) -> float:
        """Median near-side hip->knee->ankle length in px (a body-size scale)."""
        L = (geometry.dist(self.joint("hip"), self.joint("knee"))
             + geometry.dist(self.joint("knee"), self.joint("ankle")))
        return float(np.nanmedian(L))


def extract(video_path: str, model_variant: str = "full", model_path: str | None = None,
            consumers: list[FrameConsumer] | None = None, smooth_s: float = 0.12,
            progress: Callable[[int], None] | None = None) -> Session:
    consumers = consumers or []
    t0 = time.perf_counter()
    cap = open_capture(video_path)
    nominal = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    est = PoseEstimator(ensure_model(model_variant, model_path))
    pts, w, h = [], 0, 0
    fallback_dt = 1000.0 / (nominal if nominal > 0 else 30.0)
    try:
        for i, t_ms, frame in iter_frames(video_path):
            h, w = frame.shape[:2]
            pts.append(t_ms)
            est.process(frame, t_ms if np.isfinite(t_ms) else i * fallback_dt)
            for c in consumers:
                c.process(i, frame)
            if progress:
                progress(i)
        if not pts:
            raise IOError(f"no decodable frames in {video_path}")
        times, src = resolve_times(pts, nominal)
        raw = est.result(times)
    finally:
        est.close()
    info = VideoInfo(video_path, w, h, nominal, len(pts), times, src)
    t1 = time.perf_counter()

    if raw.detected.sum() < 2:
        raise RuntimeError("人体姿态几乎没有被检测到（<2 帧）。请确认画面中有完整的人体、光线充足。")
    filled = filters.fill_gaps(raw.xy, raw.detected)
    xy = filters.smooth(filled, filters.window_for(info.effective_fps, smooth_s, 3))
    side = camera_side(raw.vis)
    return Session(info, raw, xy, raw.vis, side, {"extract_s": round(t1 - t0, 2)})


@dataclass
class Freeze:
    """Hold frame ``idx`` for ``seconds`` with an extra drawing callback."""
    idx: int
    seconds: float
    draw: Callable[[np.ndarray], None]


def render(session: Session, out_path: str,
           draw: Callable[[np.ndarray, int], None],
           freezes: list[Freeze] | None = None) -> str:
    t0 = time.perf_counter()
    fps = session.info.effective_fps
    writer = VideoWriter(out_path, fps, (session.info.width, session.info.height))
    by_idx = {f.idx: f for f in (freezes or [])}
    for i, _, frame in iter_frames(session.info.path):
        if i >= session.info.n_frames:
            break
        draw(frame, i)
        writer.write(frame)
        if i in by_idx:
            fz = by_idx[i]
            still = frame.copy()
            fz.draw(still)
            for _ in range(max(1, int(round(fz.seconds * fps)))):
                writer.write(still)
    writer.close()
    session.timings["render_s"] = round(time.perf_counter() - t0, 2)
    session.timings["encoder"] = writer.encoder
    return out_path


def meta(session: Session) -> dict:
    i = session.info
    return {
        "video": i.path,
        "resolution": [i.width, i.height],
        "frames": i.n_frames,
        "duration_s": round(i.duration, 3),
        "nominal_fps": round(i.nominal_fps, 2),
        "effective_fps": round(i.effective_fps, 2),
        "timestamps_source": i.timestamps_source,
        "max_frame_gap_s": round(i.max_frame_gap, 4),
        "pose_detection_rate": round(session.raw.detection_rate, 3),
        "camera_side": session.side,
        "leg_length_px": round(session.leg_length(), 1),
        "timings": session.timings,
    }


def write_series_csv(session: Session, path: str, extra: dict[str, np.ndarray] | None = None) -> None:
    cols = {"t_s": session.t, "knee_angle_deg": session.knee_angle(),
            "hip_angle_deg": session.hip_angle()}
    for name in ("shoulder", "hip", "knee", "ankle", "heel", "toe"):
        j = session.joint(name)
        cols[f"{name}_x"], cols[f"{name}_y"] = j[:, 0], j[:, 1]
    cols.update(extra or {})
    keys = list(cols)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(keys)
        for r in range(session.info.n_frames):
            wr.writerow([f"{cols[k][r]:.4f}" if np.isfinite(cols[k][r]) else "" for k in keys])
