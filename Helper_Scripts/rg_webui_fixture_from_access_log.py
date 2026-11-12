"""Convert a timestamped uvicorn access log into the WebUI replay fixture.

Usage: rg_webui_fixture_from_access_log.py ACCESS_LOG OUT_JSON
Expects lines like: 2026-09-30 10:00:01,234 127.0.0.1:5000 "GET /api/v1/notes/?x=1 HTTP/1.1" 200

tldw_server's own startup wraps logging.config.dictConfig to route stdlib
loggers (including uvicorn.access) through a Loguru InterceptHandler, so a
plain uvicorn --log-config file handler never receives the access records in
this app. Loguru re-emits the same "%s - "%s %s HTTP/%s" %d" access message
(uvicorn's own AccessFormatter text), just prefixed with Loguru's own
"YYYY-MM-DD HH:MM:SS.mmm | LEVEL | ..." context instead of the plain
"YYYY-MM-DD HH:MM:SS,mmm" uvicorn prefix. The timestamp and request-line are
therefore matched independently (dot- or comma-separated milliseconds, any
prefix/suffix around the quoted request line) so this converter works
against either a raw uvicorn access log or this app's stdout/Loguru log.
"""

import json
import re
import sys
from collections.abc import Iterable
from datetime import datetime

_TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[.,](\d{3})")
_REQ = re.compile(r'"(\w+) (\S+) HTTP/[\d.]+"\s+(\d{3})')


def convert(lines: Iterable[str]) -> list[list[float | str]]:
    """Return ``[seconds_since_first, method, path]`` rows for each /api/ request line."""
    rows: list[list[float | str]] = []
    t0: float | None = None
    for line in lines:
        ts_m = _TS.match(line)
        req_m = _REQ.search(line)
        if not ts_m or not req_m:
            continue
        ts = datetime.strptime(f"{ts_m.group(1)}.{ts_m.group(2)}", "%Y-%m-%d %H:%M:%S.%f").timestamp()
        method, raw_path, _status = req_m.groups()
        path = raw_path.split("?", 1)[0]
        if not path.startswith("/api/"):
            continue
        t0 = ts if t0 is None else t0
        rows.append([round(ts - t0, 3), method, path])
    return rows


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as fh:
        data = convert(fh)
    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))
        fh.write("\n")  # pre-commit's end-of-file-fixer requires a final newline
    print(f"{len(data)} requests over {data[-1][0] if data else 0:.1f}s")
