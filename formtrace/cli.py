"""Command-line interface: ``formtrace analyze VIDEO --mode ...``."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Quiet MediaPipe / absl C++ logging before it is imported.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from . import __version__, barbell, jump, pipeline, posemode, report, squat


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="formtrace", description="手机视频运动轨迹与发力瞬间分析")
    p.add_argument("--version", action="version", version=f"formtrace {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="分析一段视频，输出叠加视频 + 报告")
    a.add_argument("video")
    a.add_argument("--mode", choices=["pose", "squat", "jump"], default="pose",
                   help="pose=通用骨架+关节角；squat=深蹲（次数/深度/先抬臀/杠铃轨迹）；"
                        "jump=助跑起跳（触地时间/制动膝角/水平速度损失/腾空）")
    a.add_argument("-o", "--out", default=None, help="输出目录（默认 ./out/<视频名>）")
    a.add_argument("--model", choices=["lite", "full", "heavy"], default="full",
                   help="MediaPipe 模型档位：lite 最快，heavy 最准（默认 full）")
    a.add_argument("--model-path", default=None, help="本地 .task 模型文件（离线使用）")
    a.add_argument("--no-video", action="store_true", help="只出报告，不渲染视频")
    a.add_argument("--csv", action="store_true", help="同时导出逐帧数据 CSV")
    g = a.add_argument_group("杠铃 / 比例尺（squat 模式）")
    g.add_argument("--bar", type=_floats(2, 4), default=None, metavar="X,Y[,W,H]",
                   help="手动指定杠铃片中心点（或框）的像素坐标；不填则自动在肩/手附近找杠铃片")
    g.add_argument("--bar-frame", type=int, default=0, help="--bar 坐标所在的帧号（默认 0）")
    g.add_argument("--no-bar", action="store_true", help="不追踪杠铃，只用手腕近似")
    g.add_argument("--plate-cm", type=float, default=None,
                   help="杠铃片直径（厘米），用于把像素换算成厘米和 m/s，例如 45")
    g.add_argument("--calib", type=_floats(5, 5), default=None, metavar="X1,Y1,X2,Y2,CM",
                   help="画面里两点的像素坐标及其真实距离（厘米），用于标定比例尺")
    j = a.add_argument_group("起跳（jump 模式）")
    j.add_argument("--height-cm", type=float, default=None,
                   help="运动员身高（厘米），用于把像素速度近似换算成 m/s")
    a.add_argument("--no-cache", action="store_true",
                   help="不复用输出目录里的姿态缓存 pose_cache.npz（默认复用，改参数重跑很快）")
    f = sub.add_parser("frame", help="导出某一帧并画上像素坐标网格（用于手动指定 --bar / --calib）")
    f.add_argument("video")
    f.add_argument("--at", type=int, default=0, help="帧号（默认 0）")
    f.add_argument("-o", "--out", default="frame_grid.jpg")
    return p


def run_frame(args) -> str:
    import cv2
    from .video import iter_frames
    for i, _, frame in iter_frames(args.video):
        if i == args.at:
            break
    else:
        raise SystemExit(f"视频只有 {i + 1} 帧")
    h, w = frame.shape[:2]
    step = 100 if max(h, w) <= 2000 else 200
    for x in range(0, w, step):
        cv2.line(frame, (x, 0), (x, h), (0, 255, 255) if x % 500 == 0 else (90, 90, 90), 1)
        cv2.putText(frame, str(x), (x + 2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    for y in range(0, h, step):
        cv2.line(frame, (0, y), (w, y), (0, 255, 255) if y % 500 == 0 else (90, 90, 90), 1)
        cv2.putText(frame, str(y), (2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(args.out, frame)
    return args.out


def _floats(*counts):
    def parse(text: str):
        vals = tuple(float(v) for v in text.split(","))
        if len(vals) not in counts:
            raise argparse.ArgumentTypeError(f"需要 {' 或 '.join(map(str, counts))} 个逗号分隔的数字")
        return vals
    return parse


def _public(obj):
    """Drop internal keys (leading underscore) before writing JSON."""
    if isinstance(obj, dict):
        return {k: _public(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_public(v) for v in obj]
    return obj


def _progress(n_hint: int | None = None):
    def cb(i: int) -> None:
        if i % 30 == 0:
            print(f"\r  解析帧 {i}…", end="", file=sys.stderr, flush=True)
    return cb


def run_analyze(args) -> dict:
    out = args.out or os.path.join("out", os.path.splitext(os.path.basename(args.video))[0])
    os.makedirs(out, exist_ok=True)
    print(f"[1/2] 姿态提取：{args.video}", file=sys.stderr)
    cache = None if args.no_cache else os.path.join(out, "pose_cache.npz")
    consumers = []
    tracker = None
    bar = None
    bar_cache = None if args.no_cache else os.path.join(out, "bar_cache.npz")
    if args.mode == "squat" and not args.no_bar:
        bar_key = f"{pipeline.video_key(args.video)}|{args.bar}|{args.bar_frame}|bar-v1"
        bar = barbell.load_cache(bar_cache, bar_key)
        if bar is None:
            tracker = barbell.BarTracker(args.bar, args.bar_frame)
            consumers.append(tracker)
    s = pipeline.extract(args.video, args.model, args.model_path, consumers=consumers,
                         progress=_progress(), cache_path=cache)
    print("", file=sys.stderr)
    if tracker is not None:
        bar = tracker.result()
        if bar_cache:
            barbell.save_cache(bar_cache, bar_key, bar)

    result = {"version": __version__, "mode": args.mode, "meta": None}
    extra_csv = {}
    if args.mode == "squat":
        if bar is None:
            bar = barbell.BarTrack(np.full((s.info.n_frames, 2), np.nan), None, "none")
        if bar.coverage < 0.5:
            bar = barbell.wrist_proxy(s.joint("wrist"))
        calib = squat.make_calib(bar, args.plate_cm, args.calib)
        res = squat.analyze(s, bar, calib)
        result.update(summary=res["summary"], reps=res["reps"], flags=res["flags"],
                      caveats=res["caveats"], sections=squat.sections(res))
        drawer, freezes = squat.make_drawer(s, bar, res)
        extra_csv = {"bar_x": res["_bar_xy"][:, 0], "bar_y": res["_bar_xy"][:, 1]}
    elif args.mode == "jump":
        if args.calib:
            x1, y1, x2, y2, cm = args.calib
            px_per_m, src = float(np.hypot(x2 - x1, y2 - y1)) / (cm / 100.0), "two_points"
        elif args.height_cm:
            px_per_m, src = jump.scale_from_height(s, args.height_cm), "height"
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
    if not args.no_video:
        print("[2/2] 渲染叠加视频…", file=sys.stderr)
        files["video"] = pipeline.render(s, os.path.join(out, "overlay.mp4"), drawer, freezes)
    if args.csv:
        files["csv"] = os.path.join(out, "series.csv")
        pipeline.write_series_csv(s, files["csv"], extra_csv)
    result["meta"] = pipeline.meta(s)
    result["files"] = files
    files["json"] = os.path.join(out, "report.json")
    files["markdown"] = os.path.join(out, "report.md")
    with open(files["json"], "w", encoding="utf-8") as f:
        json.dump(_public(result), f, ensure_ascii=False, indent=2)
    with open(files["markdown"], "w", encoding="utf-8") as f:
        f.write(report.markdown(result))
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "analyze":
        res = run_analyze(args)
        print(report.markdown(res))
        return 0
    if args.cmd == "frame":
        print(run_frame(args))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
