"""Library entry point shared by the CLI and the web server."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from . import __version__, barbell, jump, pipeline, posemode, report, squat

MODES = ("pose", "squat", "jump")
STAGES = {"extract": "解析帧（姿态/杠铃）", "render": "渲染叠加视频", "done": "完成"}


@dataclass
class Options:
    mode: str = "pose"
    model: str = "full"
    model_path: str | None = None
    video: bool = True
    csv: bool = True
    bar: tuple | None = None          # (x, y) or (x, y, w, h) in pixels of frame ``bar_frame``
    bar_frame: int = 0
    no_bar: bool = False
    plate_cm: float | None = None
    calib: tuple | None = None        # (x1, y1, x2, y2, cm)
    height_cm: float | None = None
    cache: bool = True


def public(obj):
    """JSON-safe copy: drop internal keys (leading underscore), NaN/inf -> None."""
    if isinstance(obj, dict):
        return {k: public(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [public(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def run(video: str, out: str, opts: Options,
        progress: Callable[[str, float], None] | None = None) -> dict:
    """Analyse ``video`` and write overlay.mp4 / report.json / report.md (/ series.csv) into ``out``."""
    if opts.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(video)
    n_hint = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
    cap.release()

    def prog(stage):
        def cb(i: int) -> None:
            if progress and (i % 5 == 0):
                progress(stage, min(1.0, (i + 1) / n_hint))
        return cb

    cache = os.path.join(out, "pose_cache.npz") if opts.cache else None
    consumers, tracker, bar, bar_key = [], None, None, None
    bar_cache = os.path.join(out, "bar_cache.npz") if opts.cache else None
    if opts.mode == "squat" and not opts.no_bar:
        bar_key = f"{pipeline.video_key(video)}|{opts.bar}|{opts.bar_frame}|bar-v1"
        bar = barbell.load_cache(bar_cache, bar_key)
        if bar is None:
            tracker = barbell.BarTracker(opts.bar, opts.bar_frame)
            consumers.append(tracker)
    s = pipeline.extract(video, opts.model, opts.model_path, consumers=consumers,
                         progress=prog("extract"), cache_path=cache)
    if tracker is not None:
        bar = tracker.result()
        if bar_cache:
            barbell.save_cache(bar_cache, bar_key, bar)

    result = {"version": __version__, "mode": opts.mode, "meta": None}
    extra_csv = {}
    if opts.mode == "squat":
        if bar is None:
            bar = barbell.BarTrack(np.full((s.info.n_frames, 2), np.nan), None, "none")
        if bar.coverage < 0.5:
            bar = barbell.wrist_proxy(s.joint("wrist"))
        calib = squat.make_calib(bar, opts.plate_cm, opts.calib)
        res = squat.analyze(s, bar, calib)
        result.update(summary=res["summary"], reps=res["reps"], flags=res["flags"],
                      caveats=res["caveats"], sections=squat.sections(res))
        drawer, freezes = squat.make_drawer(s, bar, res)
        extra_csv = {"bar_x": res["_bar_xy"][:, 0], "bar_y": res["_bar_xy"][:, 1]}
    elif opts.mode == "jump":
        if opts.calib:
            x1, y1, x2, y2, cm = opts.calib
            px_per_m, src = float(np.hypot(x2 - x1, y2 - y1)) / (cm / 100.0), "two_points"
        elif opts.height_cm:
            px_per_m, src = jump.scale_from_height(s, opts.height_cm), "height"
        else:
            px_per_m, src = None, "none"
        res = jump.analyze(s, px_per_m, src)
        result.update(summary=res["summary"], events=res["events"], flags=res["flags"],
                      caveats=res["caveats"], sections=jump.sections(res))
        drawer, freezes = jump.make_drawer(s, res)
        if "_series" in res:
            extra_csv = {"hip_vx_px_s": res["_series"]["vx"],
                         "contact_left": res["_series"]["contact_l"].astype(float),
                         "contact_right": res["_series"]["contact_r"].astype(float)}
    else:
        result["summary"] = posemode.summarize(s)
        drawer, freezes = posemode.make_drawer(s), []

    files = {}
    if opts.video:
        files["video"] = pipeline.render(s, os.path.join(out, "overlay.mp4"), drawer, freezes,
                                         progress=prog("render"))
    if opts.csv:
        files["csv"] = os.path.join(out, "series.csv")
        pipeline.write_series_csv(s, files["csv"], extra_csv)
    result["meta"] = pipeline.meta(s)
    result["quality"] = report.quality_notes(result["meta"])
    result["files"] = files
    files["json"] = os.path.join(out, "report.json")
    files["markdown"] = os.path.join(out, "report.md")
    with open(files["json"], "w", encoding="utf-8") as f:
        json.dump(public(result), f, ensure_ascii=False, indent=2)
    with open(files["markdown"], "w", encoding="utf-8") as f:
        f.write(report.markdown(result))
    if progress:
        progress("done", 1.0)
    return result
