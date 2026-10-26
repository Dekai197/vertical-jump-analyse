"""Two-pass pipeline: (1) decode + pose (+ optional trackers), (2) render overlay."""

from __future__ import annotations

import csv
import hashlib
import os
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
    """Anything that needs to see every decoded frame during pass 1 (e.g. bar tracker).

    ``pose_xy`` is the raw (33, 2) pixel landmarks of this frame, or None when
    the pose was not detected / comes from cache.
    """
    def process(self, idx: int, frame: np.ndarray, pose_xy: np.ndarray | None) -> None: ...


def video_key(video_path: str) -> str:
    """Identity of a video file for caching (path + size + mtime)."""
    st = os.stat(video_path)
    return f"{os.path.abspath(video_path)}|{st.st_size}|{st.st_mtime_ns}"


def _cache_key(video_path: str, model_variant: str, model_path: str | None) -> str:
    raw = f"pose-v3|{video_key(video_path)}|{model_variant}|{model_path}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _load_cache(path: str | None, key: str):
    if not path or not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if str(z["key"]) != key:
            return None
        return z
    except Exception:
        return None


@dataclass
class Session:
    info: VideoInfo
    raw: PoseTrack
    xy: np.ndarray          # (T, 33, 2) gap-filled + smoothed pixel coords; NaN = unknown
    vis: np.ndarray         # (T, 33); 0 where the landmark was judged invalid
    side: str               # camera-facing body side
    timings: dict = field(default_factory=dict)
    valid: np.ndarray | None = None  # (T, 33) raw per-landmark validity before gap filling

    @property
    def t(self) -> np.ndarray:
        return self.info.times

    def joint(self, name: str) -> np.ndarray:
        """(T, 2) trajectory of a near-side joint, e.g. ``joint("knee")``."""
        return self.xy[:, SIDES[self.side][name]]

    def joint_of(self, side: str, name: str) -> np.ndarray:
        return self.xy[:, SIDES[side][name]]

    def knee_angle(self) -> np.ndarray:
        return self.knee_angle_of(self.side)

    def knee_angle_of(self, side: str) -> np.ndarray:
        return geometry.joint_angle(self.joint_of(side, "hip"), self.joint_of(side, "knee"),
                                    self.joint_of(side, "ankle"))

    def hip_angle(self) -> np.ndarray:
        return geometry.joint_angle(self.joint("shoulder"), self.joint("hip"), self.joint("knee"))

    def leg_length(self) -> float:
        """Body-size scale in px: median near-side hip->knee->ankle length.

        If the feet are never in frame, falls back to 1.70 x trunk length
        (Winter's anthropometric segment proportions: hip-to-ankle ~0.491 H,
        shoulder-to-hip ~0.288 H) and records that in ``leg_length_source``.
        """
        L = (geometry.dist(self.joint("hip"), self.joint("knee"))
             + geometry.dist(self.joint("knee"), self.joint("ankle")))
        if np.isfinite(L).sum() >= 5:
            self.leg_length_source = "measured"
            return float(np.nanmedian(L))
        trunk = geometry.dist(self.joint("shoulder"), self.joint("hip"))
        self.leg_length_source = "trunk_x1.70"
        return float(1.70 * np.nanmedian(trunk))

    def frac_valid(self, name: str) -> float:
        """Share of frames where a near-side joint was actually seen (not interpolated)."""
        if self.valid is None:
            return 1.0
        return float(self.valid[:, SIDES[self.side][name]].mean())


def extract(video_path: str, model_variant: str = "full", model_path: str | None = None,
            consumers: list[FrameConsumer] | None = None, smooth_s: float = 0.12,
            progress: Callable[[int], None] | None = None,
            cache_path: str | None = None) -> Session:
    """Pass 1. With ``cache_path`` the raw pose is stored/reused (keyed on file + model)."""
    consumers = consumers or []
    t0 = time.perf_counter()
    cap = open_capture(video_path)
    nominal = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    key = _cache_key(video_path, model_variant, model_path)
    cached = _load_cache(cache_path, key)
    est = None if cached is not None else PoseEstimator(ensure_model(model_variant, model_path))
    pts, w, h = [], 0, 0
    fallback_dt = 1000.0 / (nominal if nominal > 0 else 30.0)
    try:
        if cached is None or consumers:
            for i, t_ms, frame in iter_frames(video_path):
                h, w = frame.shape[:2]
                pts.append(t_ms)
                pose_xy = None
                if est is not None:
                    pose_xy = est.process(frame, t_ms if np.isfinite(t_ms) else i * fallback_dt)
                elif i < len(cached["det"]) and cached["det"][i]:
                    pose_xy = cached["xy"][i]
                for c in consumers:
                    c.process(i, frame, pose_xy)
                if progress:
                    progress(i)
        if cached is not None:
            times, src = cached["times"], str(cached["src"])
            w, h = (int(v) for v in cached["size"])
            raw = PoseTrack(cached["xy"], cached["vis"], cached["det"].astype(bool), times)
        else:
            if not pts:
                raise IOError(f"no decodable frames in {video_path}")
            times, src = resolve_times(pts, nominal)
            raw = est.result(times)
            if cache_path:
                np.savez_compressed(cache_path, key=key, xy=raw.xy, vis=raw.vis, det=raw.detected,
                                    times=times, src=src, size=np.array([w, h]))
    finally:
        if est is not None:
            est.close()
    info = VideoInfo(video_path, w, h, nominal, len(times), times, src)
    t1 = time.perf_counter()

    if raw.detected.sum() < 2:
        raise RuntimeError("人体姿态几乎没有被检测到（<2 帧）。请确认画面中有完整的人体、光线充足。")
    valid = landmark_validity(raw, w, h)
    fps = info.effective_fps
    trunk = np.nanmedian(np.linalg.norm((raw.xy[:, 11] + raw.xy[:, 12]) / 2
                                        - (raw.xy[:, 23] + raw.xy[:, 24]) / 2, axis=-1))
    if np.isfinite(trunk):
        valid = filters.despike(raw.xy, valid, filters.window_for(fps, 0.2, 5), DESPIKE_TRUNK * trunk)
    filled = filters.fill_gaps(raw.xy, valid, max_gap=max(1, int(round(MAX_GAP_S * fps))))
    xy = filters.smooth(filled, filters.window_for(fps, smooth_s, 3))
    vis = np.where(np.isfinite(xy[..., 0]), np.maximum(raw.vis, MIN_VIS), 0.0)
    side = camera_side(np.where(valid, raw.vis, 0.0))
    return Session(info, raw, xy, vis, side,
                   {"extract_s": round(t1 - t0, 2), "pose_cache_hit": cached is not None},
                   valid)


MIN_VIS = 0.3      # below this MediaPipe is mostly guessing an occluded joint
MAX_GAP_S = 0.25   # longest occlusion that is bridged by interpolation
EDGE_MARGIN = 0.02
DESPIKE_TRUNK = 0.25  # a joint jumping > 1/4 trunk length off its 0.2 s median is a glitch


def landmark_validity(raw: PoseTrack, w: int, h: int) -> np.ndarray:
    """(T, 33) mask: detected, visible enough, and inside the frame.

    MediaPipe extrapolates joints that are outside the image (e.g. feet cut
    off by a tight framing) and can still report visibility ~0.4 for them,
    so the in-frame test is needed in addition to the visibility threshold.
    """
    x, y = raw.xy[..., 0], raw.xy[..., 1]
    with np.errstate(invalid="ignore"):
        inside = ((x >= -EDGE_MARGIN * w) & (x <= (1 + EDGE_MARGIN) * w)
                  & (y >= -EDGE_MARGIN * h) & (y <= (1 + EDGE_MARGIN) * h))
    return raw.detected[:, None] & (raw.vis >= MIN_VIS) & inside


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
    by_idx: dict[int, list[Freeze]] = {}
    for f in freezes or []:
        by_idx.setdefault(f.idx, []).append(f)
    for i, _, frame in iter_frames(session.info.path):
        if i >= session.info.n_frames:
            break
        draw(frame, i)
        writer.write(frame)
        for fz in by_idx.get(i, []):
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
        "leg_length_source": getattr(session, "leg_length_source", "measured"),
        "near_side_seen": {n: round(session.frac_valid(n), 2)
                           for n in ("shoulder", "hip", "knee", "ankle", "heel", "toe", "wrist")},
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
