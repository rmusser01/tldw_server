from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from mcp_unified.gateway import create_gateway_app

REPO_ROOT = Path(__file__).resolve().parents[5]
GATEWAY_ROOT = REPO_ROOT / "mcp_unified" / "gateway"


def _import_sources(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


class _FakeGatewayRuntime:
    name = "unit-gateway"
    version = "0.0-test"

    def __init__(self) -> None:
        self.list_contexts: list[Any] = []
        self.call_requests: list[tuple[str, dict[str, Any], Any]] = []

    async def list_tools(self, context: Any) -> list[dict[str, Any]]:
        self.list_contexts.append(context)
        return [
            {
                "name": "echo.search",
                "description": "Echo a query.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "metadata": {"category": "test"},
            }
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        self.call_requests.append((name, arguments, context))
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"{name}:{arguments['query']}",
                }
            ]
        }


class _CustomExplodingGatewayRuntime(_FakeGatewayRuntime):
    class RuntimeBackendError(Exception):
        pass

    async def list_tools(self, context: Any) -> list[dict[str, Any]]:
        raise self.RuntimeBackendError("backend unavailable")


def _assert_jsonrpc_error(
    body: dict[str, Any],
    *,
    code: int,
    request_id: Any,
) -> None:
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == request_id
    assert body["error"]["code"] == code
    assert "message" in body["error"]


def test_gateway_package_does_not_import_tldw_server_api() -> None:
    assert GATEWAY_ROOT.exists()
    offenders: dict[str, list[str]] = {}
    for path in GATEWAY_ROOT.rglob("*.py"):
        blocked = sorted(
            source
            for source in _import_sources(path)
            if source == "tldw_Server_API" or source.startswith("tldw_Server_API.")
        )
        if blocked:
            offenders[str(path.relative_to(REPO_ROOT))] = blocked

    assert offenders == {}


def test_gateway_fastapi_app_handles_basic_jsonrpc_flow() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        status = client.get("/mcp/status")
        assert status.status_code == 200
        assert status.json() == {
            "status": "ok",
            "name": "unit-gateway",
            "version": "0.0-test",
        }

        initialized = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {"clientInfo": {"name": "pytest"}},
                "id": "init-1",
            },
        )
        assert initialized.status_code == 200
        assert initialized.json()["result"]["serverInfo"] == {
            "name": "unit-gateway",
            "version": "0.0-test",
        }

        listed = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "tools-1"},
        )
        assert listed.status_code == 200
        listed_body = listed.json()
        assert listed_body["id"] == "tools-1"
        assert listed_body["result"]["tools"][0]["name"] == "echo.search"
        assert runtime.list_contexts[-1].request_id == "tools-1"

        called = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "echo.search",
                    "arguments": {"query": "hello"},
                },
                "id": "call-1",
            },
        )
        assert called.status_code == 200
        called_body = called.json()
        assert called_body["id"] == "call-1"
        assert called_body["result"]["content"][0]["text"] == "echo.search:hello"
        assert runtime.call_requests[-1][0] == "echo.search"


def test_gateway_request_rejects_malformed_json_with_jsonrpc_parse_error() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        response = client.post(
            "/mcp/request",
            content=b"{not valid json",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 200
    _assert_jsonrpc_error(response.json(), code=-32700, request_id=None)


def test_gateway_request_rejects_missing_jsonrpc_member() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        response = client.post(
            "/mcp/request",
            json={"method": "ping", "id": "missing-version"},
        )

    assert response.status_code == 200
    _assert_jsonrpc_error(response.json(), code=-32600, request_id="missing-version")


def test_gateway_request_rejects_invalid_jsonrpc_id_type() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        response = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "ping", "id": {"bad": 1}},
        )

    assert response.status_code == 200
    _assert_jsonrpc_error(response.json(), code=-32600, request_id=None)


def test_gateway_request_rejects_non_object_params_without_coercion() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        response = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": [], "id": "bad-params"},
        )

    assert response.status_code == 200
    _assert_jsonrpc_error(response.json(), code=-32602, request_id="bad-params")
    assert runtime.list_contexts == []


def test_gateway_request_rejects_non_object_tool_arguments_without_coercion() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        response = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "echo.search",
                    "arguments": [],
                },
                "id": "bad-args",
            },
        )

    assert response.status_code == 200
    _assert_jsonrpc_error(response.json(), code=-32602, request_id="bad-args")
    assert runtime.call_requests == []


def test_gateway_request_maps_custom_runtime_exceptions_to_jsonrpc_internal_error() -> None:
    runtime = _CustomExplodingGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "explode"},
        )

    assert response.status_code == 200
    _assert_jsonrpc_error(response.json(), code=-32603, request_id="explode")
