#!/usr/bin/env python3
import json
import sys
import uuid

JSONRPC_VERSION = "2.0"


def _write(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _error_response(request_id, code: int, message: str) -> dict:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _result_response(request_id, result) -> dict:
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}


def _handle_request(payload: dict) -> dict | None:
    method = payload.get("method")
    request_id = payload.get("id")
    params = payload.get("params") or {}

    if method == "initialize":
        result = {
            "protocolVersion": 1,
            "agentCapabilities": {
                "loadSession": False,
                "promptCapabilities": {
                    "image": False,
                    "audio": False,
                    "embeddedContext": False,
                },
                "mcpCapabilities": {"http": False, "sse": False},
                "sessionCapabilities": {},
            },
            "agentInfo": {
                "name": "tldw-acp-stub",
                "title": "TLDW ACP Stub Agent",
                "version": "0.1.0",
            },
            "authMethods": [],
        }
        return _result_response(request_id, result)

    if method == "session/new":
        session_id = f"stub-{uuid.uuid4().hex[:8]}"
        return _result_response(request_id, {"sessionId": session_id})

    if method == "session/prompt":
        session_id = params.get("sessionId")
        if session_id:
            _write(
                {
                    "jsonrpc": JSONRPC_VERSION,
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "event": "message",
                        "content": "stub-response",
                    },
                }
            )
        return _result_response(
            request_id,
            {
                "stopReason": "end",
                "taskCompletion": {
                    "status": "completed",
                    "summary": "Deterministic ACP stub completed the requested task.",
                    "artifacts": [],
                },
            },
        )

    if method in {"session/cancel", "_tldw/session/close"}:
        return _result_response(request_id, None)

    return _error_response(request_id, -32601, "method not found")


def main() -> int:
    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if "method" not in payload:
            continue

        request_id = payload.get("id")
        if request_id is None:
            continue

        response = _handle_request(payload)
        if response is not None:
            _write(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
