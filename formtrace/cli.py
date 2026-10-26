"""Command-line interface: ``formtrace analyze VIDEO --mode ...``."""

from __future__ import annotations

import argparse
import os
import sys

# Quiet MediaPipe / absl C++ logging before it is imported.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from . import __version__, analysis, report


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
    w = sub.add_parser("serve", help="启动本地网页界面（拖拽视频即可分析）")
    w.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机 127.0.0.1）")
    w.add_argument("--port", type=int, default=8765)
    w.add_argument("--workdir", default=None, help="上传与结果存放目录（默认 ~/.cache/formtrace/web）")
    w.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
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


def _options(args) -> analysis.Options:
    return analysis.Options(
        mode=args.mode, model=args.model, model_path=args.model_path, video=not args.no_video,
        csv=args.csv, bar=args.bar, bar_frame=args.bar_frame, no_bar=args.no_bar,
        plate_cm=args.plate_cm, calib=args.calib, height_cm=args.height_cm, cache=not args.no_cache)


def _cli_progress():
    last = {"stage": None}

    def cb(stage: str, frac: float) -> None:
        if stage != last["stage"]:
            if last["stage"]:
                print("", file=sys.stderr)
            last["stage"] = stage
        print(f"\r  {analysis.STAGES.get(stage, stage)} {frac * 100:5.1f}%", end="", file=sys.stderr, flush=True)
    return cb


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "analyze":
        out = args.out or os.path.join("out", os.path.splitext(os.path.basename(args.video))[0])
        res = analysis.run(args.video, out, _options(args), _cli_progress())
        print("", file=sys.stderr)
        print(report.markdown(res))
        return 0
    if args.cmd == "serve":
        from . import server
        server.serve(args.host, args.port, args.workdir, open_browser=not args.no_browser)
        return 0
    if args.cmd == "frame":
        print(run_frame(args))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
