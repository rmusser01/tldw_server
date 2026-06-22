"""Small LSP-like stdio server used by JSON-RPC client tests."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


def main() -> int:
    trace_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    methods: list[str] = []
    while True:
        message = _read_message()
        if message is None:
            return 0
        header_text, payload = message
        method = payload.get("method")
        if isinstance(method, str):
            methods.append(method)
            _write_trace(trace_path, methods)
        if "id" not in payload:
            if method == "exit":
                _write_trace(trace_path, methods)
                return 0
            continue

        request_id = payload["id"]
        params = payload.get("params") or {}
        if method == "initialize":
            _write_response(
                request_id,
                {
                    "capabilities": {},
                    "received_header": header_text,
                },
            )
        elif method == "shutdown":
            _write_response(request_id, None)
        elif method == "test/echo":
            _write_response(request_id, {"value": params.get("value"), "request_id": request_id})
        elif method == "test/sleep":
            time.sleep(float(params.get("seconds", 1)))
            _write_response(request_id, {"slept": True})
        elif method == "test/stderr":
            sys.stderr.write(str(params.get("message", "")))
            sys.stderr.flush()
            _write_response(request_id, {"ok": True})
        elif method == "test/crash":
            return 42
        elif method == "textDocument/definition":
            _write_response(
                request_id,
                [{"uri": "file:///workspace/pkg/app.py", "range": _sample_range()}],
            )
        elif method == "textDocument/documentSymbol":
            _write_response(
                request_id,
                [{"name": "handler", "kind": 12, "range": _sample_range(), "selectionRange": _sample_range()}],
            )
        elif method == "textDocument/hover":
            _write_response(request_id, {"contents": "hover text", "range": _sample_range()})
        elif method == "textDocument/references":
            _write_response(
                request_id,
                [{"uri": "file:///workspace/pkg/app.py", "range": _sample_range()}],
            )
        elif method == "textDocument/signatureHelp":
            _write_response(request_id, {"signatures": [{"label": "handler(value: str) -> None"}]})
        else:
            _write_error(request_id, code=-32601, message=f"unknown method: {method}")
    return 0


def _read_message() -> tuple[str, dict[str, Any]] | None:
    header_bytes = bytearray()
    while not header_bytes.endswith(b"\r\n\r\n"):
        byte = sys.stdin.buffer.read(1)
        if not byte:
            return None
        header_bytes.extend(byte)
    header_text = header_bytes.decode("ascii", errors="replace")
    content_length = _content_length(header_text)
    body = sys.stdin.buffer.read(content_length)
    if not body:
        return None
    return header_text, json.loads(body.decode("utf-8"))


def _content_length(header_text: str) -> int:
    for line in header_text.splitlines():
        name, _, value = line.partition(":")
        if name.lower() == "content-length":
            return int(value.strip())
    raise RuntimeError("missing Content-Length header")


def _write_response(request_id: object, result: object) -> None:
    _write_payload({"jsonrpc": "2.0", "id": request_id, "result": result})


def _write_error(request_id: object, *, code: int, message: str) -> None:
    _write_payload({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


def _write_payload(payload: dict[str, object]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _write_trace(trace_path: Path | None, methods: list[str]) -> None:
    if trace_path is None:
        return
    trace_path.write_text(json.dumps({"methods": methods}), encoding="utf-8")


def _sample_range() -> dict[str, dict[str, int]]:
    return {"start": {"line": 0, "character": 1}, "end": {"line": 0, "character": 4}}


if __name__ == "__main__":
    raise SystemExit(main())
