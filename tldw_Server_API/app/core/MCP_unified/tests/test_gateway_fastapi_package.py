from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import mcp_unified.gateway.fastapi as gateway_fastapi
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
        self.resource_list_contexts: list[Any] = []
        self.resource_read_requests: list[tuple[str, Any]] = []
        self.prompt_list_contexts: list[Any] = []
        self.prompt_get_requests: list[tuple[str, dict[str, Any], Any]] = []
        self.module_list_contexts: list[Any] = []
        self.module_health_contexts: list[Any] = []

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

    async def list_resources(self, context: Any) -> list[dict[str, Any]]:
        self.resource_list_contexts.append(context)
        return [
            {
                "uri": "resource://unit/doc",
                "name": "Unit Doc",
                "mimeType": "text/plain",
            }
        ]

    async def read_resource(self, uri: str, context: Any) -> dict[str, Any]:
        self.resource_read_requests.append((uri, context))
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "text/plain",
                    "text": "hello resource",
                }
            ]
        }

    async def list_prompts(self, context: Any) -> list[dict[str, Any]]:
        self.prompt_list_contexts.append(context)
        return [
            {
                "name": "review.prompt",
                "description": "Review a focused topic.",
            }
        ]

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        self.prompt_get_requests.append((name, arguments, context))
        topic = arguments.get("topic", "")
        return {
            "description": "Review a focused topic.",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"{name}:{topic}",
                    },
                }
            ],
        }

    async def list_modules(self, context: Any) -> list[dict[str, Any]]:
        self.module_list_contexts.append(context)
        return [{"module_id": "unit", "name": "Unit Module"}]

    async def get_modules_health(self, context: Any) -> dict[str, Any]:
        self.module_health_contexts.append(context)
        return {
            "unit": {
                "status": "healthy",
                "message": "ok",
                "checks": {},
                "last_check": None,
            }
        }


class _CustomExplodingGatewayRuntime(_FakeGatewayRuntime):
    class RuntimeBackendError(Exception):
        pass

    async def list_tools(self, context: Any) -> list[dict[str, Any]]:
        raise self.RuntimeBackendError("backend unavailable")


class _FakeLogger:
    def __init__(self) -> None:
        self.opt_calls: list[dict[str, Any]] = []
        self.error_calls: list[tuple[str, tuple[Any, ...]]] = []

    def opt(self, **kwargs: Any) -> _FakeLogger:
        self.opt_calls.append(kwargs)
        return self

    def error(self, message: str, *args: Any) -> None:
        self.error_calls.append((message, args))


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
        capabilities = initialized.json()["result"]["capabilities"]
        assert capabilities["resources"]["available"] is True
        assert capabilities["prompts"]["available"] is True

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
        assert runtime.call_requests[-1][1] == {"query": "hello"}
        assert runtime.call_requests[-1][2].request_id == "call-1"


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


def test_gateway_fastapi_app_handles_resource_prompt_and_module_methods() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        resources = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "resources/list", "params": {}, "id": "resources-1"},
        )
        assert resources.status_code == 200
        resources_body = resources.json()
        assert resources_body["id"] == "resources-1"
        assert resources_body["result"]["resources"][0]["uri"] == "resource://unit/doc"
        assert runtime.resource_list_contexts[-1].request_id == "resources-1"

        resource = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "resources/read",
                "params": {"uri": "resource://unit/doc"},
                "id": "read-1",
            },
        )
        assert resource.status_code == 200
        resource_body = resource.json()
        assert resource_body["id"] == "read-1"
        assert resource_body["result"]["contents"][0]["text"] == "hello resource"
        assert runtime.resource_read_requests[-1][0] == "resource://unit/doc"
        assert runtime.resource_read_requests[-1][1].request_id == "read-1"

        prompts = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "prompts/list", "params": {}, "id": "prompts-1"},
        )
        assert prompts.status_code == 200
        prompts_body = prompts.json()
        assert prompts_body["id"] == "prompts-1"
        assert prompts_body["result"]["prompts"][0]["name"] == "review.prompt"
        assert runtime.prompt_list_contexts[-1].request_id == "prompts-1"

        prompt = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "prompts/get",
                "params": {"name": "review.prompt", "arguments": {"topic": "gateway"}},
                "id": "prompt-1",
            },
        )
        assert prompt.status_code == 200
        prompt_body = prompt.json()
        assert prompt_body["id"] == "prompt-1"
        assert prompt_body["result"]["messages"][0]["content"]["text"] == "review.prompt:gateway"
        assert runtime.prompt_get_requests[-1][0] == "review.prompt"
        assert runtime.prompt_get_requests[-1][1] == {"topic": "gateway"}
        assert runtime.prompt_get_requests[-1][2].request_id == "prompt-1"

        modules = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "modules/list", "params": {}, "id": "modules-1"},
        )
        assert modules.status_code == 200
        modules_body = modules.json()
        assert modules_body["id"] == "modules-1"
        assert modules_body["result"]["modules"][0]["module_id"] == "unit"
        assert runtime.module_list_contexts[-1].request_id == "modules-1"

        health = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "modules/health", "params": {}, "id": "health-1"},
        )
        assert health.status_code == 200
        health_body = health.json()
        assert health_body["id"] == "health-1"
        assert health_body["result"]["health"]["unit"]["status"] == "healthy"
        assert runtime.module_health_contexts[-1].request_id == "health-1"


def test_gateway_websocket_handles_basic_jsonrpc_flow() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        with client.websocket_connect("/mcp/ws") as websocket:
            websocket.send_json(
                {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {"clientInfo": {"name": "pytest-ws"}},
                    "id": "ws-init",
                }
            )
            initialized = websocket.receive_json()
            assert initialized["jsonrpc"] == "2.0"
            assert initialized["id"] == "ws-init"
            assert initialized["result"]["protocolVersion"] == "2024-11-05"

            websocket.send_json({"jsonrpc": "2.0", "method": "ping", "id": "ws-ping"})
            ping = websocket.receive_json()
            assert ping == {"jsonrpc": "2.0", "result": {"pong": True}, "id": "ws-ping"}

            websocket.send_json(
                {
                    "jsonrpc": "2.0",
                    "method": "resources/list",
                    "params": {},
                    "id": "ws-resources",
                }
            )
            resources = websocket.receive_json()
            assert resources["jsonrpc"] == "2.0"
            assert resources["id"] == "ws-resources"
            assert resources["result"]["resources"][0]["uri"] == "resource://unit/doc"
            assert runtime.resource_list_contexts[-1].request_id == "ws-resources"
            assert runtime.resource_list_contexts[-1].metadata["path"] == "/mcp/ws"


def test_gateway_websocket_maps_invalid_json_to_parse_error() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        with client.websocket_connect("/mcp/ws") as websocket:
            websocket.send_text("not-json")
            body = websocket.receive_json()

    assert body["jsonrpc"] == "2.0"
    assert body["id"] is None
    assert body["error"]["code"] == -32700
    assert "Parse error" in body["error"]["message"]


def test_gateway_websocket_suppresses_notification_response() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        with client.websocket_connect("/mcp/ws") as websocket:
            websocket.send_json({"jsonrpc": "2.0", "method": "ping"})
            websocket.send_json({"jsonrpc": "2.0", "method": "ping", "id": "after-notification"})
            body = websocket.receive_json()

    assert body == {"jsonrpc": "2.0", "result": {"pong": True}, "id": "after-notification"}


def test_gateway_websocket_batch_omits_notification_responses() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        with client.websocket_connect("/mcp/ws") as websocket:
            websocket.send_json(
                [
                    {"jsonrpc": "2.0", "method": "ping"},
                    {"jsonrpc": "2.0", "method": "ping", "id": "batch-ping"},
                ]
            )
            body = websocket.receive_json()

    assert body == [{"jsonrpc": "2.0", "result": {"pong": True}, "id": "batch-ping"}]


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


def test_gateway_request_rejects_missing_resource_uri() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        response = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "resources/read", "params": {}, "id": "bad-resource"},
        )

    assert response.status_code == 200
    _assert_jsonrpc_error(response.json(), code=-32602, request_id="bad-resource")
    assert runtime.resource_read_requests == []


def test_gateway_request_rejects_missing_prompt_name() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        response = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "prompts/get", "params": {}, "id": "bad-prompt"},
        )

    assert response.status_code == 200
    _assert_jsonrpc_error(response.json(), code=-32602, request_id="bad-prompt")
    assert runtime.prompt_get_requests == []


def test_gateway_prompt_get_accepts_missing_arguments() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        response = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "prompts/get",
                "params": {"name": "review.prompt"},
                "id": "prompt-no-args",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "prompt-no-args"
    assert body["result"]["messages"][0]["content"]["text"] == "review.prompt:"
    assert runtime.prompt_get_requests[-1][1] == {}


def test_gateway_request_rejects_non_object_prompt_arguments_without_coercion() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        response = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "prompts/get",
                "params": {
                    "name": "review.prompt",
                    "arguments": [],
                },
                "id": "bad-prompt-args",
            },
        )

    assert response.status_code == 200
    _assert_jsonrpc_error(response.json(), code=-32602, request_id="bad-prompt-args")
    assert runtime.prompt_get_requests == []


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


def test_gateway_request_logs_custom_runtime_exceptions(monkeypatch: Any) -> None:
    runtime = _CustomExplodingGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")
    fake_logger = _FakeLogger()
    monkeypatch.setattr(gateway_fastapi, "logger", fake_logger, raising=False)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "explode"},
        )

    assert response.status_code == 200
    assert fake_logger.opt_calls == [{"exception": True}]
    assert fake_logger.error_calls == [
        ("Gateway runtime error while handling method={!r} request_id={!r}", ("tools/list", "explode"))
    ]


def test_gateway_notification_runtime_errors_do_not_return_jsonrpc_response() -> None:
    runtime = _CustomExplodingGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}},
        )

    assert response.status_code == 204
    assert response.content == b""


def test_gateway_batch_omits_notification_runtime_errors() -> None:
    runtime = _CustomExplodingGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/mcp/request",
            json=[
                {"jsonrpc": "2.0", "method": "tools/list", "params": {}},
                {"jsonrpc": "2.0", "method": "ping", "id": "ok"},
            ],
        )

    assert response.status_code == 200
    assert response.json() == [{"jsonrpc": "2.0", "result": {"pong": True}, "id": "ok"}]
