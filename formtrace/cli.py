"""Command-line interface: ``formtrace analyze VIDEO --mode ...``."""

from __future__ import annotations

import argparse
import json
import os
import sys

# Quiet MediaPipe / absl C++ logging before it is imported.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from . import __version__, pipeline, posemode, report


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="formtrace", description="手机视频运动轨迹与发力瞬间分析")
    p.add_argument("--version", action="version", version=f"formtrace {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="分析一段视频，输出叠加视频 + 报告")
    a.add_argument("video")
    a.add_argument("--mode", choices=["pose"], default="pose",
                   help="pose=通用骨架+关节角")
    a.add_argument("-o", "--out", default=None, help="输出目录（默认 ./out/<视频名>）")
    a.add_argument("--model", choices=["lite", "full", "heavy"], default="full",
                   help="MediaPipe 模型档位：lite 最快，heavy 最准（默认 full）")
    a.add_argument("--model-path", default=None, help="本地 .task 模型文件（离线使用）")
    a.add_argument("--no-video", action="store_true", help="只出报告，不渲染视频")
    a.add_argument("--csv", action="store_true", help="同时导出逐帧数据 CSV")
    return p


def _progress(n_hint: int | None = None):
    def cb(i: int) -> None:
        if i % 30 == 0:
            print(f"\r  解析帧 {i}…", end="", file=sys.stderr, flush=True)
    return cb


def run_analyze(args) -> dict:
    out = args.out or os.path.join("out", os.path.splitext(os.path.basename(args.video))[0])
    os.makedirs(out, exist_ok=True)
    print(f"[1/2] 姿态提取：{args.video}", file=sys.stderr)
    s = pipeline.extract(args.video, args.model, args.model_path, progress=_progress())
    print("", file=sys.stderr)

    result = {"version": __version__, "mode": args.mode, "meta": None}
    result["summary"] = posemode.summarize(s)
    drawer, freezes = posemode.make_drawer(s), []

    files = {}
    if not args.no_video:
        print("[2/2] 渲染叠加视频…", file=sys.stderr)
        files["video"] = pipeline.render(s, os.path.join(out, "overlay.mp4"), drawer, freezes)
    if args.csv:
        files["csv"] = os.path.join(out, "series.csv")
        pipeline.write_series_csv(s, files["csv"])
    result["meta"] = pipeline.meta(s)
    result["files"] = files
    files["json"] = os.path.join(out, "report.json")
    files["markdown"] = os.path.join(out, "report.md")
    with open(files["json"], "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(files["markdown"], "w", encoding="utf-8") as f:
        f.write(report.markdown(result))
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "analyze":
        res = run_analyze(args)
        print(report.markdown(res))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
