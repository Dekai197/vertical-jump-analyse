"""Local web UI: drag a video in, pick a mode, get the overlay + report.

Standard library only (http.server). Binds to 127.0.0.1 by default; jobs run
one at a time in a background thread because pose estimation already uses
all CPU cores.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import threading
import traceback
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

import cv2

from . import __version__, analysis

MAX_UPLOAD = 1024 ** 3  # 1 GiB
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".3gp"}
SERVED = {"overlay.mp4": "video/mp4", "report.json": "application/json",
          "report.md": "text/markdown; charset=utf-8", "series.csv": "text/csv; charset=utf-8",
          "frame.jpg": "image/jpeg"}
ID_RE = re.compile(r"^[0-9a-f]{32}$")
PREVIEW_MAX_SIDE = 1280


class Jobs:
    def __init__(self, workdir: str):
        self.workdir = workdir
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def dir(self, jid: str) -> str:
        return os.path.join(self.workdir, jid)

    def get(self, jid: str) -> dict | None:
        with self.lock:
            j = self.jobs.get(jid)
            return dict(j) if j else None

    def update(self, jid: str, **kw) -> None:
        with self.lock:
            self.jobs[jid].update(kw)

    def create(self, video_path: str, name: str, info: dict) -> str:
        jid = os.path.basename(os.path.dirname(video_path))
        with self.lock:
            self.jobs[jid] = {"id": jid, "name": name, "video": video_path, "status": "uploaded",
                              "stage": None, "progress": 0.0, "error": None, "info": info}
        return jid

    def submit(self, jid: str, opts: analysis.Options) -> None:
        self.update(jid, status="queued", stage=None, progress=0.0, error=None, opts=opts.__dict__)
        self.q.put((jid, opts))

    def _worker(self) -> None:
        while True:
            jid, opts = self.q.get()
            j = self.get(jid)
            if not j:
                continue
            self.update(jid, status="running")

            def prog(stage: str, frac: float, jid=jid) -> None:
                self.update(jid, stage=stage, progress=round(frac, 3))
            try:
                analysis.run(j["video"], os.path.join(self.dir(jid), "out"), opts, prog)
                self.update(jid, status="done", progress=1.0)
            except Exception as e:  # report to the page instead of killing the worker
                traceback.print_exc()
                self.update(jid, status="error", error=f"{type(e).__name__}: {e}")


def _probe_and_preview(video: str, out_jpg: str) -> dict:
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    info = {"fps": float(cap.get(cv2.CAP_PROP_FPS) or 0),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)}
    cap.release()
    if not ok:
        raise ValueError("无法解码这个视频文件")
    h, w = frame.shape[:2]
    sc = min(1.0, PREVIEW_MAX_SIDE / max(h, w))
    if sc < 1:
        frame = cv2.resize(frame, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_jpg, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    info.update(width=w, height=h)
    return info


def make_handler(jobs: Jobs):
    index_html = resources.files("formtrace").joinpath("web/index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = f"formtrace/{__version__}"

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        # -- helpers ----------------------------------------------------
        def _json(self, obj, status=HTTPStatus.OK):
            body = json.dumps(analysis.public(obj), ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status, msg):
            self._json({"error": msg}, status)

        def _send_file(self, path: str, ctype: str):
            size = os.path.getsize(path)
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            m = re.match(r"bytes=(\d*)-(\d*)$", rng or "")
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:  # suffix range: last N bytes
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    left -= len(chunk)

        # -- routes -----------------------------------------------------
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(index_html)))
                self.end_headers()
                self.wfile.write(index_html)
                return
            if path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            if path == "/api/version":
                return self._json({"version": __version__})
            m = re.match(r"^/api/job/([0-9a-f]{32})$", path)
            if m:
                j = jobs.get(m.group(1))
                if not j:
                    return self._error(HTTPStatus.NOT_FOUND, "任务不存在")
                j.pop("video", None)
                if j["status"] == "done":
                    rp = os.path.join(jobs.dir(j["id"]), "out", "report.json")
                    with open(rp, encoding="utf-8") as f:
                        j["report"] = json.load(f)
                    j["files"] = [n for n in SERVED if n != "frame.jpg" and
                                  os.path.exists(os.path.join(jobs.dir(j["id"]), "out", n))]
                return self._json(j)
            m = re.match(r"^/api/job/([0-9a-f]{32})/([a-z0-9_.]+)$", path)
            if m and m.group(2) in SERVED:
                jid, name = m.groups()
                base = jobs.dir(jid)
                p = os.path.join(base, name) if name == "frame.jpg" else os.path.join(base, "out", name)
                if not jobs.get(jid) or not os.path.exists(p):
                    return self._error(HTTPStatus.NOT_FOUND, "文件不存在")
                return self._send_file(p, SERVED[name])
            self._error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/upload":
                return self._upload()
            if path == "/api/analyze":
                return self._analyze()
            self._error(HTTPStatus.NOT_FOUND, "not found")

        def _upload(self):
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = 0
            if n <= 0:
                return self._error(HTTPStatus.LENGTH_REQUIRED, "空文件")
            if n > MAX_UPLOAD:
                return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "文件超过 1 GB")
            from urllib.parse import unquote
            name = os.path.basename(unquote(self.headers.get("X-Filename", "video.mp4")))[:200]
            ext = os.path.splitext(name)[1].lower()
            if ext not in VIDEO_EXT:
                return self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, f"不支持的格式 {ext or '(无扩展名)'}")
            jid = uuid.uuid4().hex
            d = jobs.dir(jid)
            os.makedirs(d)
            dst = os.path.join(d, "input" + ext)
            left = n
            with open(dst, "wb") as f:
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            if left:
                shutil.rmtree(d, ignore_errors=True)
                return self._error(HTTPStatus.BAD_REQUEST, "上传中断")
            try:
                info = _probe_and_preview(dst, os.path.join(d, "frame.jpg"))
            except Exception as e:
                shutil.rmtree(d, ignore_errors=True)
                return self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(e))
            jobs.create(dst, name, info)
            self._json({"id": jid, "name": name, **info, "frame_url": f"/api/job/{jid}/frame.jpg"})

        def _analyze(self):
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(min(n, 1 << 16)) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._error(HTTPStatus.BAD_REQUEST, "请求格式错误")
            jid = str(body.get("id", ""))
            if not ID_RE.match(jid) or not jobs.get(jid):
                return self._error(HTTPStatus.NOT_FOUND, "任务不存在")
            if jobs.get(jid)["status"] in ("queued", "running"):
                return self._error(HTTPStatus.CONFLICT, "这个视频正在分析中")
            mode = body.get("mode", "pose")
            if mode not in analysis.MODES:
                return self._error(HTTPStatus.BAD_REQUEST, "未知模式")

            def num(key, lo, hi):
                v = body.get(key)
                if v in (None, ""):
                    return None
                v = float(v)
                if not lo <= v <= hi:
                    raise ValueError(key)
                return v
            try:
                bar = body.get("bar")
                bar = tuple(float(v) for v in bar) if bar else None
                if bar is not None and len(bar) not in (2, 4):
                    raise ValueError("bar")
                opts = analysis.Options(
                    mode=mode, model=body.get("model", "full") if body.get("model") in ("lite", "full", "heavy") else "full",
                    bar=bar, plate_cm=num("plate_cm", 5, 100), height_cm=num("height_cm", 80, 250))
            except (TypeError, ValueError) as e:
                return self._error(HTTPStatus.BAD_REQUEST, f"参数无效：{e}")
            jobs.submit(jid, opts)
            self._json({"ok": True})

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765, workdir: str | None = None,
          open_browser: bool = True) -> None:
    workdir = workdir or os.path.join(os.path.expanduser("~"), ".cache", "formtrace", "web")
    os.makedirs(workdir, exist_ok=True)
    jobs = Jobs(workdir)
    httpd = ThreadingHTTPServer((host, port), make_handler(jobs))
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}/"
    print(f"formtrace {__version__} 网页界面：{url}  （Ctrl+C 退出）")
    if host not in ("127.0.0.1", "localhost"):
        print("注意：服务对局域网开放，任何能访问该地址的人都能上传视频。")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
