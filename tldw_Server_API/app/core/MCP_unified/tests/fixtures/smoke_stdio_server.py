"""Newline-delimited JSON-RPC fixture server for smoke stdio transport tests."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
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
        tools = [
            {
                "name": "echo.search",
                "description": "Echo a search query.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ]
        if _artifact_root() is not None:
            tools.extend(_artifact_tool_descriptors())
        return {"tools": tools}
    if method == "tools/call":
        params_dict = _params_dict(params)
        tool_name = params_dict.get("name")
        arguments = params_dict.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        if tool_name == "echo.search":
            query = arguments.get("query", "")
            return {"content": [{"type": "text", "text": f"echo.search:{query}"}]}
        if isinstance(tool_name, str) and tool_name.startswith("artifact."):
            return _artifact_tool_result(tool_name, arguments)
        return _UNKNOWN_METHOD
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


def _artifact_root() -> Path | None:
    value = os.environ.get("MCP_SMOKE_ARTIFACT_ROOT")
    if not value:
        return None
    root = Path(value).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _artifact_tool_descriptors() -> list[dict[str, Any]]:
    path_schema = {"type": "string", "minLength": 1}
    return [
        {
            "name": "artifact.read",
            "description": "Read a smoke artifact.",
            "inputSchema": {
                "type": "object",
                "properties": {"path": path_schema},
                "required": ["path"],
            },
        },
        {
            "name": "artifact.summarize",
            "description": "Create a deterministic summary from a smoke artifact.",
            "inputSchema": {
                "type": "object",
                "properties": {"source_path": path_schema},
                "required": ["source_path"],
            },
        },
        {
            "name": "artifact.write",
            "description": "Write a derived smoke artifact.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": path_schema,
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "artifact.stat",
            "description": "Return metadata for a smoke artifact.",
            "inputSchema": {
                "type": "object",
                "properties": {"path": path_schema},
                "required": ["path"],
            },
        },
    ]


def _artifact_tool_result(tool_name: str, arguments: dict[str, Any]) -> object:
    if tool_name == "artifact.read":
        relative_path, target = _artifact_target(arguments.get("path"))
        body = target.read_text(encoding="utf-8")
        return {
            "content": [{"type": "text", "text": body}],
            "structuredContent": {
                "path": relative_path,
                "bytes": len(body.encode("utf-8")),
                "sha256": _sha256_text(body),
                "text": body,
            },
        }
    if tool_name == "artifact.summarize":
        source_path = arguments.get("source_path", arguments.get("path"))
        relative_path, target = _artifact_target(source_path)
        body = target.read_text(encoding="utf-8")
        summary = _summarize_artifact(body, relative_path)
        return {
            "content": [{"type": "text", "text": summary}],
            "structuredContent": {
                "source_path": relative_path,
                "summary_markdown": summary,
                "bytes": len(summary.encode("utf-8")),
                "sha256": _sha256_text(summary),
            },
        }
    if tool_name == "artifact.write":
        relative_path, target = _artifact_target(arguments.get("path"))
        content = arguments.get("content", arguments.get("text", ""))
        if not isinstance(content, str):
            raise ValueError("artifact_content_must_be_text")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {
            "content": [{"type": "text", "text": f"wrote {relative_path}"}],
            "structuredContent": {
                "path": relative_path,
                "bytes": len(content.encode("utf-8")),
                "sha256": _sha256_text(content),
            },
        }
    if tool_name == "artifact.stat":
        relative_path, target = _artifact_target(arguments.get("path"))
        exists = target.exists()
        size = target.stat().st_size if exists else 0
        digest = _sha256_bytes(target.read_bytes()) if exists and target.is_file() else None
        return {
            "content": [
                {"type": "text", "text": f"{relative_path}: {'exists' if exists else 'missing'}"}
            ],
            "structuredContent": {
                "path": relative_path,
                "exists": exists,
                "bytes": size,
                "sha256": digest,
            },
        }
    return _UNKNOWN_METHOD


def _artifact_target(path_value: object) -> tuple[str, Path]:
    root = _artifact_root()
    if root is None:
        raise ValueError("artifact_root_not_configured")
    relative_path = _normalize_relative_artifact_path(path_value)
    target = (root / relative_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("artifact_path_denied") from exc
    return relative_path, target


def _normalize_relative_artifact_path(path_value: object) -> str:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("artifact_path_required")
    candidate = Path(path_value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError("artifact_path_denied")
    return candidate.as_posix()


def _summarize_artifact(body: str, relative_path: str) -> str:
    words = [word.strip(".,:;!?()[]{}\"'").lower() for word in body.split()]
    unique_words = len({word for word in words if word})
    first_line = next((line.strip() for line in body.splitlines() if line.strip()), "")
    if first_line.startswith("#"):
        first_line = first_line.lstrip("#").strip()
    title = first_line or "Untitled artifact"
    return (
        "# Smoke UAT Derived Artifact\n\n"
        f"- Source: {relative_path}\n"
        f"- Title: {title}\n"
        f"- Word count: {len(words)}\n"
        f"- Unique words: {unique_words}\n"
    )


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
