"""Web server routes that do not need a real video or the pose model."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from formtrace import server


@pytest.fixture()
def base(tmp_path):
    jobs = server.Jobs(str(tmp_path))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(jobs))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _req(url, data=None, headers=None, method=None):
    r = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_index_and_version(base):
    st, body = _req(base + "/")
    assert st == 200 and b"formtrace" in body
    st, body = _req(base + "/api/version")
    assert st == 200 and "version" in json.loads(body)


def test_upload_rejects_bad_extension_and_garbage(base):
    st, _ = _req(base + "/api/upload", b"abc", {"X-Filename": "x.exe"}, "POST")
    assert st == 415
    st, body = _req(base + "/api/upload", b"not a video", {"X-Filename": "x.mp4"}, "POST")
    assert st == 422


def test_unknown_job_and_path_traversal(base):
    assert _req(base + "/api/job/" + "0" * 32)[0] == 404
    assert _req(base + "/api/job/" + "0" * 32 + "/report.json")[0] == 404
    assert _req(base + "/api/job/" + "0" * 32 + "/..%2f..%2fetc%2fpasswd")[0] == 404
    st, _ = _req(base + "/api/analyze", json.dumps({"id": "../../x"}).encode(),
                 {"Content-Type": "application/json"}, "POST")
    assert st == 404
