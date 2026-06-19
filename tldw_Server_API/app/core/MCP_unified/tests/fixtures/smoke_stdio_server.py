"""Newline-delimited JSON-RPC fixture server for smoke stdio transport tests."""

from __future__ import annotations

import json
import sys
import time
from typing import Any


def main() -> int:
    """Read JSON-RPC lines from stdin and write responses to stdout."""

    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
            response = handle_payload(payload)
        except Exception as exc:  # noqa: BLE001 - fixture should report diagnostics.
            print(f"fixture diagnostic: {exc}", file=sys.stderr, flush=True)
            response = _error_response(None, -32700, "Parse error")

        if response is not None:
            print(json.dumps(response, separators=(",", ":")), flush=True)

    return 0


def handle_payload(payload: object) -> object | None:
    """Handle one JSON-RPC object or batch array."""

    if isinstance(payload, list):
        responses = [
            response for item in payload if (response := handle_message(item)) is not None
        ]
        return responses or None
    return handle_message(payload)


def handle_message(message: object) -> object | None:
    """Handle one JSON-RPC message, suppressing notification responses."""

    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _error_response(None, -32600, "Invalid Request")

    method = message.get("method")
    request_id = message.get("id")
    is_notification = "id" not in message

    if not isinstance(method, str):
        if is_notification:
            return None
        return _error_response(request_id, -32600, "Invalid Request")

    if method == "smoke/secret-stderr":
        print("fixture diagnostic: credential exchange failed", file=sys.stderr, flush=True)
        raise SystemExit(3)
    if method == "smoke/hang":
        time.sleep(5)
        return None
    if method == "smoke/server-notification-before-response":
        _write_stdout(
            {
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {"progress": 1},
            }
        )
        return {"jsonrpc": "2.0", "id": request_id, "result": {"after": "notification"}}
    if method == "smoke/wrong-id-response":
        return {"jsonrpc": "2.0", "id": "wrong-id", "result": {"wrong": True}}
    if method == "smoke/large-response":
        return {"blob": "x" * 4096}

    if is_notification:
        print(f"fixture diagnostic: notification {method}", file=sys.stderr, flush=True)
        return None

    result = _method_result(method, message.get("params"))
    if result is _UNKNOWN_METHOD:
        return _error_response(request_id, -32601, "Method not found")
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _method_result(method: str, params: object) -> object:
    if method == "initialize":
        return {
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "tools": {"available": True},
                "resources": {"available": True},
                "prompts": {"available": True},
            },
            "serverInfo": {"name": "smoke-stdio-fixture", "version": "0.0-test"},
        }
    if method == "ping":
        return {"pong": True}
    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": "echo.search",
                    "description": "Echo a search query.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                }
            ]
        }
    if method == "tools/call":
        params_dict = _params_dict(params)
        if params_dict.get("name") != "echo.search":
            return _UNKNOWN_METHOD
        arguments = params_dict.get("arguments", {})
        query = arguments.get("query", "") if isinstance(arguments, dict) else ""
        return {"content": [{"type": "text", "text": f"echo.search:{query}"}]}
    if method == "resources/list":
        return {
            "resources": [
                {
                    "uri": "resource://smoke/doc",
                    "name": "Smoke document",
                    "mimeType": "text/plain",
                }
            ]
        }
    if method == "resources/read":
        uri = _params_dict(params).get("uri")
        if uri != "resource://smoke/doc":
            return _UNKNOWN_METHOD
        return {
            "contents": [
                {
                    "uri": "resource://smoke/doc",
                    "mimeType": "text/plain",
                    "text": "Smoke fixture resource.",
                }
            ]
        }
    if method == "prompts/list":
        return {"prompts": [{"name": "smoke.review", "description": "Review prompt"}]}
    if method == "prompts/get":
        params_dict = _params_dict(params)
        if params_dict.get("name") != "smoke.review":
            return _UNKNOWN_METHOD
        arguments = params_dict.get("arguments", {})
        topic = arguments.get("topic", "smoke") if isinstance(arguments, dict) else "smoke"
        return {
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": f"Review {topic}"},
                }
            ]
        }
    return _UNKNOWN_METHOD


def _params_dict(params: object) -> dict[str, Any]:
    if isinstance(params, dict):
        return params
    return {}


def _error_response(request_id: object, code: int, message: str) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _write_stdout(payload: object) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


_UNKNOWN_METHOD = object()


if __name__ == "__main__":
    raise SystemExit(main())
