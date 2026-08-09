"""Lifecycle and dispatch tests for the strict MCP protocol connection."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

import pytest
from mcp_unified.gateway import (
    GatewayLimits,
    GatewayResourceNotFound,
    GatewayToolExecutionError,
)

pytestmark = pytest.mark.unit


class _MemoryWriter:
    """Collect complete JSON values written by one protocol connection."""

    def __init__(self) -> None:
        self.values: list[Any] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def __call__(self, value: Any) -> None:
        self.entered.set()
        await self.release.wait()
        self.values.append(value)

    def block(self) -> None:
        """Hold the serialized writer lock until ``unblock`` is called."""

        self.entered.clear()
        self.release.clear()

    def unblock(self) -> None:
        self.release.set()


class _CoreRuntime:
    """Small in-memory core runtime with no package-specific module aliases."""

    name = "connection-runtime"
    version = "1.2.3"

    def __init__(self, *, templates: bool = True) -> None:
        self.tools: list[dict[str, Any]] = [
            {
                "name": "echo",
                "description": "Echo an integer",
                "inputSchema": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {"echoed": {"type": "integer"}},
                    "required": ["echoed"],
                    "additionalProperties": False,
                },
                "_meta": {
                    "com.example/runtime": "tool",
                    "io.modelcontextprotocol/serverInfo": {
                        "name": "forged",
                        "version": "0",
                    },
                },
            }
        ]
        self.resources: list[dict[str, Any]] = [
            {
                "name": "guide",
                "uri": "file:///guide.txt",
                "mimeType": "text/plain",
            }
        ]
        self.templates: list[dict[str, Any]] = [
            {
                "name": "user-file",
                "uriTemplate": "file:///users/{user}/files/{name}",
            }
        ]
        self.prompts: list[dict[str, Any]] = [
            {
                "name": "welcome",
                "arguments": [{"name": "name", "required": True}],
            }
        ]
        self.tool_result: Any = {"echoed": 7}
        self.resource_result: Any = {
            "contents": [
                {
                    "uri": "file:///guide.txt",
                    "mimeType": "text/plain",
                    "text": "guide",
                }
            ]
        }
        self.prompt_result: Any = {
            "description": "Welcome",
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": "Hello"},
                }
            ],
        }
        self.contexts: list[Any] = []
        self.call_names: list[str] = []
        self.arguments: list[dict[str, Any]] = []
        self.runtime_entries = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self.call_started = asyncio.Event()
        self.call_release = asyncio.Event()
        self.call_release.set()
        self.swallow_cancellation = False
        if not templates:
            self.list_resource_templates = None  # type: ignore[assignment]

    async def list_tools(self, context: Any) -> list[dict[str, Any]]:
        self._record(context)
        return self.tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Any,
    ) -> Any:
        self._record(context)
        self.call_names.append(name)
        self.arguments.append(arguments)
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        self.call_started.set()
        try:
            try:
                await self.call_release.wait()
            except asyncio.CancelledError:
                if not self.swallow_cancellation:
                    raise
            if isinstance(self.tool_result, BaseException):
                raise self.tool_result
            return self.tool_result
        finally:
            self.active_calls -= 1

    async def list_resources(self, context: Any) -> list[dict[str, Any]]:
        self._record(context)
        return self.resources

    async def list_resource_templates(self, context: Any) -> list[dict[str, Any]]:
        self._record(context)
        return self.templates

    async def read_resource(self, uri: str, context: Any) -> dict[str, Any]:
        self._record(context)
        if isinstance(self.resource_result, BaseException):
            raise self.resource_result
        return self.resource_result

    async def list_prompts(self, context: Any) -> list[dict[str, Any]]:
        self._record(context)
        return self.prompts

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        self._record(context)
        if isinstance(self.prompt_result, BaseException):
            raise self.prompt_result
        return self.prompt_result

    def _record(self, context: Any) -> None:
        self.runtime_entries += 1
        self.contexts.append(context)


def _modern_meta(
    *,
    version: str = "2026-07-28",
    capabilities: dict[str, Any] | None = None,
    client_info: dict[str, Any] | None = None,
    vendor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": capabilities or {},
    }
    meta["io.modelcontextprotocol/protocolVersion"] = version
    if client_info is not None:
        meta["io.modelcontextprotocol/clientInfo"] = client_info
    meta.update(vendor or {})
    return meta


def _modern_request(
    request_id: str | int | None,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    version: str = "2026-07-28",
    include_id: bool = True,
    capabilities: dict[str, Any] | None = None,
    client_info: dict[str, Any] | None = None,
    vendor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
        "params": {
            **(params or {}),
            "_meta": _modern_meta(
                version=version,
                capabilities=capabilities,
                client_info=client_info,
                vendor=vendor,
            ),
        },
    }
    if include_id:
        request["id"] = request_id
    return request


def _legacy_request(
    request_id: str | int | None,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    include_id: bool = True,
) -> dict[str, Any]:
    request: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        request["params"] = params
    if include_id:
        request["id"] = request_id
    return request


def _new_connection(
    runtime: _CoreRuntime,
    writer: _MemoryWriter,
    **kwargs: Any,
) -> Any:
    from mcp_unified.gateway import GatewayProtocolConnection

    return GatewayProtocolConnection(runtime, writer, **kwargs)


async def _initialize_legacy(
    connection: Any,
    writer: _MemoryWriter,
    *,
    version: str = "2025-11-25",
    capabilities: dict[str, Any] | None = None,
    client_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    await connection.receive(
        _legacy_request(
            "init",
            "initialize",
            {
                "protocolVersion": version,
                "capabilities": capabilities or {},
                "clientInfo": client_info or {"name": "test-client", "version": "1.0"},
            },
        )
    )
    await connection.wait_for_idle()
    response = writer.values[-1]
    writer.values.clear()
    return response


def _error(
    request_id: str | int | None,
    code: int,
    message: str,
    *,
    data: Any = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


@pytest.mark.asyncio
async def test_modern_discovery_can_be_the_first_request() -> None:
    """Requiring a prior session operation must break stateless modern discovery."""

    runtime = _CoreRuntime()
    writer = _MemoryWriter()

    connection = _new_connection(runtime, writer)
    await connection.receive(_modern_request(1, "server/discover"))
    await connection.wait_for_idle()

    assert writer.values == [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "supportedVersions": [
                    "2026-07-28",
                    "2025-11-25",
                    "2025-06-18",
                    "2025-03-26",
                    "2024-11-05",
                ],
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {},
                },
                "resultType": "complete",
                "ttlMs": 0,
                "cacheScope": "private",
                "_meta": {
                    "io.modelcontextprotocol/serverInfo": {
                        "name": "connection-runtime",
                        "version": "1.2.3",
                    }
                },
            },
        }
    ]


@pytest.mark.asyncio
async def test_modern_core_methods_project_catalogs_and_results() -> None:
    """Wrong method routing or omitted modern result fields must fail visibly."""

    runtime = _CoreRuntime()
    writer = _MemoryWriter()
    connection = _new_connection(runtime, writer)

    requests = [
        _modern_request(1, "tools/list"),
        _modern_request(2, "tools/call", {"name": "echo", "arguments": {"value": 7}}),
        _modern_request(3, "resources/list"),
        _modern_request(4, "resources/templates/list"),
        _modern_request(5, "resources/read", {"uri": "file:///guide.txt"}),
        _modern_request(6, "prompts/list"),
        _modern_request(7, "prompts/get", {"name": "welcome", "arguments": {"name": "Ada"}}),
    ]
    for request in requests:
        await connection.receive(request)
        await connection.wait_for_idle()

    by_id = {response["id"]: response["result"] for response in writer.values}
    assert by_id[1]["tools"][0]["name"] == "echo"
    assert by_id[1]["tools"][0]["inputSchema"] == runtime.tools[0]["inputSchema"]
    assert by_id[1]["tools"][0]["outputSchema"] == runtime.tools[0]["outputSchema"]
    assert by_id[1]["tools"][0]["_meta"]["com.example/runtime"] == "tool"
    assert by_id[2]["structuredContent"] == {"echoed": 7}
    assert by_id[2]["content"] == [{"type": "text", "text": '{"echoed":7}'}]
    assert {key: by_id[3]["resources"][0][key] for key in ("name", "uri", "mimeType")} == runtime.resources[0]
    assert {key: by_id[4]["resourceTemplates"][0][key] for key in ("name", "uriTemplate")} == runtime.templates[0]
    assert by_id[5]["contents"] == runtime.resource_result["contents"]
    assert {key: by_id[6]["prompts"][0][key] for key in ("name", "arguments")} == runtime.prompts[0]
    assert by_id[7]["messages"] == runtime.prompt_result["messages"]
    for result in by_id.values():
        assert result["resultType"] == "complete"
        assert result["_meta"]["io.modelcontextprotocol/serverInfo"] == {
            "name": "connection-runtime",
            "version": "1.2.3",
        }
    for request_id in (1, 3, 4, 6):
        assert by_id[request_id]["ttlMs"] == 0
        assert by_id[request_id]["cacheScope"] == "private"


@pytest.mark.asyncio
async def test_empty_catalogs_remain_available_with_standard_capabilities() -> None:
    """Treating an empty catalog as an unavailable capability must break discovery."""

    runtime = _CoreRuntime()
    runtime.tools = []
    runtime.resources = []
    runtime.templates = []
    runtime.prompts = []
    writer = _MemoryWriter()
    connection = _new_connection(runtime, writer)

    for request_id, method in enumerate(
        (
            "server/discover",
            "tools/list",
            "resources/list",
            "resources/templates/list",
            "prompts/list",
        ),
        start=1,
    ):
        await connection.receive(_modern_request(request_id, method))
        await connection.wait_for_idle()

    assert writer.values[0]["result"]["capabilities"] == {
        "tools": {},
        "resources": {},
        "prompts": {},
    }
    assert [
        writer.values[index]["result"][field]
        for index, field in enumerate(
            ("tools", "resources", "resourceTemplates", "prompts"),
            start=1,
        )
    ] == [[], [], [], []]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "meta",
    [
        {},
        {"io.modelcontextprotocol/protocolVersion": "2026-07-28"},
        {"io.modelcontextprotocol/clientCapabilities": {}},
        {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientCapabilities": [],
        },
    ],
)
async def test_modern_requests_require_well_formed_reserved_metadata(
    meta: dict[str, Any],
) -> None:
    """Missing or malformed stateless metadata must not reach the runtime."""

    runtime = _CoreRuntime()
    writer = _MemoryWriter()
    connection = _new_connection(runtime, writer)
    request = _modern_request(1, "tools/list")
    request["params"]["_meta"] = meta

    await connection.receive(request)
    await connection.wait_for_idle()

    assert writer.values == [_error(1, -32602, "Invalid params")]
    assert runtime.runtime_entries == 0


@pytest.mark.asyncio
async def test_unsupported_modern_version_advertises_all_five_without_payload() -> None:
    """An incomplete support vector or reflected request data must break negotiation."""

    runtime = _CoreRuntime()
    writer = _MemoryWriter()
    connection = _new_connection(runtime, writer)
    request = _modern_request(1, "server/discover", version="2099-01-01")
    request["params"]["private"] = "do-not-reflect"

    await connection.receive(request)
    await connection.wait_for_idle()

    assert writer.values == [
        _error(
            1,
            -32022,
            "Unsupported protocol version",
            data={
                "requested": "2099-01-01",
                "supported": [
                    "2026-07-28",
                    "2025-11-25",
                    "2025-06-18",
                    "2025-03-26",
                    "2024-11-05",
                ],
            },
        )
    ]
    assert "do-not-reflect" not in json.dumps(writer.values)

    writer.values.clear()
    await connection.receive(
        _legacy_request(
            2,
            "initialize",
            {"protocolVersion": "2025-11-25", "capabilities": {}},
        )
    )
    await connection.wait_for_idle()
    assert writer.values == [_error(2, -32000, "Request rejected")]


@pytest.mark.asyncio
async def test_legacy_initialization_fallback_and_initialized_notification() -> None:
    """Wrong legacy negotiation, capability shape, or notification output must fail."""

    runtime = _CoreRuntime()
    writer = _MemoryWriter()
    connection = _new_connection(runtime, writer)

    await connection.receive(_legacy_request(0, "ping"))
    await connection.wait_for_idle()
    assert writer.values.pop() == {"jsonrpc": "2.0", "id": 0, "result": {}}

    initialized = await _initialize_legacy(
        connection,
        writer,
        version="1900-01-01",
        capabilities={"roots": {}},
        client_info={"name": "legacy-client", "version": "2.0"},
    )
    assert initialized == {
        "jsonrpc": "2.0",
        "id": "init",
        "result": {
            "protocolVersion": "2025-11-25",
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {},
            },
            "serverInfo": {"name": "connection-runtime", "version": "1.2.3"},
        },
    }

    await connection.receive(_legacy_request(None, "notifications/initialized", include_id=False))
    await connection.wait_for_idle()
    assert writer.values == []

    await connection.receive(_legacy_request(2, "tools/list"))
    await connection.wait_for_idle()
    result = writer.values.pop()["result"]
    assert result["tools"][0]["name"] == "echo"
    assert "resultType" not in result
    assert "ttlMs" not in result
    assert result["tools"][0]["inputSchema"] == runtime.tools[0]["inputSchema"]
    assert runtime.contexts[-1].client_capabilities == {"roots": {}}
    assert runtime.contexts[-1].client_info == {
        "name": "legacy-client",
        "version": "2.0",
    }


@pytest.mark.asyncio
async def test_preinitialize_second_initialize_and_era_mixing_are_admission_errors() -> None:
    """Lifecycle and era admission failures must use one payload-free code."""

    runtime = _CoreRuntime()
    writer = _MemoryWriter()
    connection = _new_connection(runtime, writer)

    await connection.receive(_legacy_request(1, "tools/list"))
    await connection.wait_for_idle()
    assert writer.values.pop() == _error(1, -32000, "Request rejected")

    await _initialize_legacy(connection, writer, version="2025-06-18")
    await connection.receive(
        _legacy_request(
            2,
            "initialize",
            {"protocolVersion": "2025-06-18", "capabilities": {}},
        )
    )
    await connection.wait_for_idle()
    assert writer.values.pop() == _error(2, -32000, "Request rejected")

    await connection.receive(_modern_request(3, "ping"))
    await connection.wait_for_idle()
    assert writer.values.pop() == _error(3, -32000, "Request rejected")

    modern_writer = _MemoryWriter()
    modern = _new_connection(_CoreRuntime(), modern_writer)
    await modern.receive(_modern_request(4, "ping"))
    await modern.wait_for_idle()
    modern_writer.values.clear()
    await modern.receive(
        _legacy_request(
            5,
            "initialize",
            {"protocolVersion": "2025-11-25", "capabilities": {}},
        )
    )
    await modern.wait_for_idle()
    assert modern_writer.values == [_error(5, -32000, "Request rejected")]


@pytest.mark.asyncio
async def test_authoritative_context_preserves_typed_ids_and_per_request_capabilities() -> None:
    """Stringifying IDs or reusing modern metadata must corrupt runtime context."""

    runtime = _CoreRuntime()
    writer = _MemoryWriter()
    connection = _new_connection(
        runtime,
        writer,
        metadata={
            "transport": "forged",
            "method": "forged",
            "protocol_version": "forged",
            "safe": "connection",
        },
    )

    for request_id, capability in ((1, "roots"), ("1", "sampling")):
        await connection.receive(
            _modern_request(
                request_id,
                "tools/list",
                capabilities={capability: {}},
                client_info={"name": "client", "version": str(request_id)},
                vendor={"com.example/request": request_id, "transport": "forged"},
            )
        )
        await connection.wait_for_idle()

    contexts = runtime.contexts
    assert [(type(context.request_id), context.request_id) for context in contexts] == [
        (int, 1),
        (str, "1"),
    ]
    assert contexts[0].client_capabilities == {"roots": {}}
    assert contexts[1].client_capabilities == {"sampling": {}}
    assert contexts[0].client_info == {"name": "client", "version": "1"}
    assert all(context.protocol_version == "2026-07-28" for context in contexts)
    assert all(context.protocol_era == "modern" for context in contexts)
    assert all(context.metadata["transport"] == "stdio" for context in contexts)
    assert all(context.metadata["method"] == "tools/list" for context in contexts)
    assert all(context.metadata["safe"] == "connection" for context in contexts)
    assert [context.metadata["com.example/request"] for context in contexts] == [1, "1"]
    assert all("protocol_version" not in context.metadata for context in contexts)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_id", [None, True, False])
async def test_null_and_boolean_ids_are_invalid_requests(request_id: object) -> None:
    """Accepting null or boolean IDs must break typed request correlation."""

    runtime = _CoreRuntime()
    writer = _MemoryWriter()
    connection = _new_connection(runtime, writer)

    await connection.receive(_modern_request(request_id, "ping"))  # type: ignore[arg-type]
    await connection.wait_for_idle()

    assert writer.values == [_error(None, -32600, "Invalid request")]
    assert runtime.runtime_entries == 0


@pytest.mark.asyncio
async def test_duplicate_active_id_is_rejected_before_second_runtime_dispatch() -> None:
    """Dispatching a duplicate typed ID must create ambiguous cancellation state."""

    runtime = _CoreRuntime()
    runtime.call_release.clear()
    writer = _MemoryWriter()
    connection = _new_connection(runtime, writer)
    request = _modern_request(1, "tools/call", {"name": "echo", "arguments": {"value": 7}})

    await connection.receive(request)
    await runtime.call_started.wait()
    await connection.receive(request)
    await asyncio.sleep(0)
    assert writer.values == [_error(1, -32000, "Request rejected")]
    assert runtime.call_names == ["echo"]

    await connection.receive(
        {
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": 1},
        }
    )
    runtime.call_release.set()
    await connection.wait_for_idle()
    assert writer.values == [_error(1, -32000, "Request rejected")]


@pytest.mark.asyncio
async def test_notifications_are_silent_and_module_aliases_are_not_served() -> None:
    """Strict core dispatch must emit neither alias results nor notification errors."""

    runtime = _CoreRuntime()
    writer = _MemoryWriter()
    connection = _new_connection(runtime, writer)

    await connection.receive(_modern_request(1, "modules/list"))
    await connection.receive(_modern_request(2, "modules/health"))
    await connection.receive(_modern_request(None, "unknown/notification", include_id=False))
    await connection.wait_for_idle()

    assert writer.values == [
        _error(1, -32601, "Method not found"),
        _error(2, -32601, "Method not found"),
    ]
    assert runtime.runtime_entries == 0
    assert all("method" not in response for response in writer.values)


@pytest.mark.asyncio
async def test_unknown_names_invalid_results_and_safe_errors_never_reflect_payloads() -> None:
    """Raw names, arguments, results, and exception strings must stay off the wire."""

    cases = [
        (
            _modern_request(
                1,
                "tools/call",
                {"name": "missing", "arguments": {"secret": "payload-secret"}},
            ),
            None,
            _error(1, -32602, "Invalid params"),
        ),
        (
            _modern_request(2, "resources/read", {"uri": "file:///private-secret"}),
            GatewayResourceNotFound(),
            _error(
                2,
                -32602,
                "Resource not found",
                data={"reasonCode": "resource_not_found", "kind": "resource"},
            ),
        ),
        (
            _modern_request(3, "prompts/get", {"name": "missing-secret"}),
            None,
            _error(3, -32602, "Invalid params"),
        ),
    ]
    for request, resource_error, expected in cases:
        runtime = _CoreRuntime()
        if resource_error is not None:
            runtime.resource_result = resource_error
        writer = _MemoryWriter()
        connection = _new_connection(runtime, writer)
        await connection.receive(request)
        await connection.wait_for_idle()
        assert writer.values == [expected]
        assert "payload-secret" not in json.dumps(writer.values)
        assert "private-secret" not in json.dumps(writer.values)
        assert "missing-secret" not in json.dumps(writer.values)

    runtime = _CoreRuntime()
    runtime.tool_result = GatewayToolExecutionError(
        "Tool is temporarily unavailable",
        reason_code="temporarily_unavailable",
    )
    writer = _MemoryWriter()
    connection = _new_connection(runtime, writer)
    await connection.receive(_modern_request(4, "tools/call", {"name": "echo", "arguments": {"value": 7}}))
    await connection.wait_for_idle()
    assert writer.values[0]["result"]["isError"] is True
    assert writer.values[0]["result"]["_meta"]["io.github.rmusser01.mcp-unified/error"] == {
        "reasonCode": "temporarily_unavailable",
        "kind": "tool",
    }


@pytest.mark.asyncio
async def test_tool_schemas_compile_and_input_output_roles_are_enforced() -> None:
    """Skipping descriptor compilation or either instance role must cross bad data."""

    runtime = _CoreRuntime()
    writer = _MemoryWriter()
    connection = _new_connection(runtime, writer)

    await connection.receive(_modern_request(1, "tools/list"))
    await connection.wait_for_idle()
    assert "result" in writer.values.pop()

    await connection.receive(_modern_request(2, "tools/call", {"name": "echo", "arguments": {"value": "bad"}}))
    await connection.wait_for_idle()
    assert writer.values.pop() == _error(2, -32602, "Invalid params")
    assert runtime.call_names == []

    runtime.tool_result = {"echoed": "bad"}
    await connection.receive(_modern_request(3, "tools/call", {"name": "echo", "arguments": {"value": 7}}))
    await connection.wait_for_idle()
    assert writer.values.pop() == _error(3, -32603, "Internal error")

    runtime.tools[0]["inputSchema"] = {"type": 7}
    await connection.receive(_modern_request(4, "tools/list"))
    await connection.wait_for_idle()
    assert writer.values.pop() == _error(4, -32603, "Internal error")


@pytest.mark.asyncio
async def test_current_arbitrary_output_and_legacy_object_output_use_correct_roots() -> None:
    """Forcing current outputs to objects or loosening legacy objects must fail."""

    runtime = _CoreRuntime()
    runtime.tools[0]["outputSchema"] = {
        "type": "array",
        "items": {"type": "integer"},
    }
    runtime.tool_result = [1, 2]
    writer = _MemoryWriter()
    modern = _new_connection(runtime, writer)
    await modern.receive(_modern_request(1, "tools/list"))
    await modern.wait_for_idle()
    assert writer.values.pop()["result"]["tools"][0]["outputSchema"]["type"] == "array"
    await modern.receive(_modern_request(2, "tools/call", {"name": "echo", "arguments": {"value": 7}}))
    await modern.wait_for_idle()
    assert writer.values.pop()["result"]["structuredContent"] == [1, 2]

    legacy_runtime = _CoreRuntime()
    legacy_writer = _MemoryWriter()
    legacy = _new_connection(legacy_runtime, legacy_writer)
    await _initialize_legacy(legacy, legacy_writer, version="2025-11-25")
    await legacy.receive(_legacy_request(3, "tools/list"))
    await legacy.wait_for_idle()
    assert legacy_writer.values.pop()["result"]["tools"][0]["outputSchema"]["type"] == "object"
    legacy_runtime.tool_result = {"echoed": "bad"}
    await legacy.receive(_legacy_request(4, "tools/call", {"name": "echo", "arguments": {"value": 7}}))
    await legacy.wait_for_idle()
    assert legacy_writer.values.pop() == _error(4, -32603, "Internal error")

    fallback_runtime = _CoreRuntime()
    fallback_runtime.tools[0]["outputSchema"] = {
        "type": "array",
        "items": {"type": "integer"},
    }
    fallback_runtime.tool_result = [1, 2]
    fallback_writer = _MemoryWriter()
    fallback = _new_connection(fallback_runtime, fallback_writer)
    await _initialize_legacy(fallback, fallback_writer, version="2025-11-25")
    await fallback.receive(_legacy_request(5, "tools/list"))
    await fallback.wait_for_idle()
    assert "outputSchema" not in fallback_writer.values.pop()["result"]["tools"][0]
    await fallback.receive(
        _legacy_request(
            6,
            "tools/call",
            {"name": "echo", "arguments": {"value": 7}},
        )
    )
    await fallback.wait_for_idle()
    fallback_result = fallback_writer.values.pop()["result"]
    assert fallback_result["content"] == [{"type": "text", "text": "[1,2]"}]
    assert "structuredContent" not in fallback_result

    fallback_runtime.tool_result = [1, "bad"]
    await fallback.receive(
        _legacy_request(
            7,
            "tools/call",
            {"name": "echo", "arguments": {"value": 7}},
        )
    )
    await fallback.wait_for_idle()
    assert fallback_writer.values.pop() == _error(7, -32603, "Internal error")


@pytest.mark.asyncio
async def test_result_and_envelope_byte_limits_return_safe_bounded_errors() -> None:
    """Unchecked aggregate/application bytes must allow oversized runtime output."""

    limits = replace(
        GatewayLimits(),
        max_output_line_bytes=512,
        max_result_bytes=128,
    )
    runtime = _CoreRuntime()
    runtime.tool_result = {"echoed": 7, "secret": "x" * 200}
    writer = _MemoryWriter()
    connection = _new_connection(runtime, writer, limits=limits)

    await connection.receive(_modern_request(1, "tools/call", {"name": "echo", "arguments": {"value": 7}}))
    await connection.wait_for_idle()

    assert writer.values == [
        _error(
            1,
            -33001,
            "Application result exceeds the configured limit",
            data={
                "reasonCode": "result_too_large",
                "kind": "application",
                "limitBytes": 128,
            },
        )
    ]
    assert "secret" not in json.dumps(writer.values)


class _Clock:
    """Deterministic injected monotonic clock for token-bucket assertions."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.asyncio
async def test_empty_and_initialize_batches_are_rejected_before_element_dispatch() -> None:
    """Creating element tasks for invalid lifecycle batches must cross admission."""

    for version in (
        "2026-07-28",
        "2025-11-25",
        "2025-06-18",
        "2025-03-26",
        "2024-11-05",
    ):
        runtime = _CoreRuntime()
        writer = _MemoryWriter()
        connection = _new_connection(runtime, writer)
        initialize = _legacy_request(
            1,
            "initialize",
            {"protocolVersion": version, "capabilities": {}},
        )
        if version == "2026-07-28":
            initialize["params"]["_meta"] = _modern_meta()

        await connection.receive([initialize])
        await connection.wait_for_idle()

        assert writer.values == [_error(None, -32000, "Request rejected")]
        assert runtime.runtime_entries == 0

    runtime = _CoreRuntime()
    writer = _MemoryWriter()
    connection = _new_connection(runtime, writer)
    await connection.receive([])
    await connection.wait_for_idle()
    assert writer.values == [_error(None, -32600, "Invalid request")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "version",
    ["2026-07-28", "2025-11-25", "2025-06-18", "2024-11-05"],
)
async def test_batches_are_rejected_in_the_four_nonbatch_profiles(version: str) -> None:
    """Allowing a batch outside initialized 2025-03-26 must violate its profile."""

    runtime = _CoreRuntime()
    writer = _MemoryWriter()
    connection = _new_connection(runtime, writer)
    if version == "2026-07-28":
        await connection.receive(_modern_request(0, "ping"))
    else:
        await _initialize_legacy(connection, writer, version=version)
    await connection.wait_for_idle()
    writer.values.clear()

    request = _modern_request(1, "ping") if version == "2026-07-28" else _legacy_request(1, "ping")
    await connection.receive([request])
    await connection.wait_for_idle()

    assert writer.values == [_error(None, -32000, "Request rejected")]


@pytest.mark.asyncio
async def test_initialized_2025_03_26_batches_preserve_order_and_notification_silence() -> None:
    """Sequential output or notification placeholders must corrupt batch semantics."""

    runtime = _CoreRuntime()
    writer = _MemoryWriter()
    connection = _new_connection(runtime, writer)
    await _initialize_legacy(connection, writer, version="2025-03-26")

    await connection.receive(
        [
            _legacy_request(1, "ping"),
            _legacy_request(None, "notifications/initialized", include_id=False),
            42,
            _legacy_request(2, "ping"),
        ]
    )
    await connection.wait_for_idle()
    assert writer.values == [
        [
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            _error(None, -32600, "Invalid request"),
            {"jsonrpc": "2.0", "id": 2, "result": {}},
        ]
    ]

    writer.values.clear()
    await connection.receive(
        [
            _legacy_request(None, "notifications/initialized", include_id=False),
            _legacy_request(
                None,
                "notifications/cancelled",
                {"requestId": "already-finished"},
                include_id=False,
            ),
        ]
    )
    await connection.wait_for_idle()
    assert writer.values == []


@pytest.mark.asyncio
async def test_max_batch_items_is_checked_before_any_element_runtime_work() -> None:
    """Late batch limiting must let oversized arrays create request tasks."""

    runtime = _CoreRuntime()
    writer = _MemoryWriter()
    limits = replace(GatewayLimits(), max_batch_items=2)
    connection = _new_connection(runtime, writer, limits=limits)
    await _initialize_legacy(connection, writer, version="2025-03-26")

    await connection.receive(
        [
            _legacy_request(1, "tools/list"),
            _legacy_request(2, "tools/list"),
            _legacy_request(3, "tools/list"),
        ]
    )
    await connection.wait_for_idle()

    assert writer.values == [_error(None, -32000, "Request rejected")]
    assert runtime.runtime_entries == 0


@pytest.mark.asyncio
async def test_max_in_flight_rejects_before_second_runtime_dispatch() -> None:
    """Waiting instead of rejecting above the in-flight limit must over-admit work."""

    runtime = _CoreRuntime()
    runtime.call_release.clear()
    writer = _MemoryWriter()
    connection = _new_connection(
        runtime,
        writer,
        limits=replace(GatewayLimits(), max_in_flight=1),
    )

    await connection.receive(_modern_request(1, "tools/call", {"name": "echo", "arguments": {"value": 7}}))
    await runtime.call_started.wait()
    await connection.receive(_modern_request(2, "tools/call", {"name": "echo", "arguments": {"value": 8}}))
    await asyncio.sleep(0)

    assert writer.values == [_error(2, -32000, "Request rejected")]
    assert runtime.call_names == ["echo"]
    await connection.receive(
        {
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": 1},
        }
    )
    runtime.call_release.set()
    await connection.wait_for_idle()


@pytest.mark.asyncio
async def test_token_bucket_enforces_burst_and_refills_from_injected_clock() -> None:
    """A fixed-window or wall-clock limiter must fail deterministic refill behavior."""

    clock = _Clock()
    runtime = _CoreRuntime()
    writer = _MemoryWriter()
    limits = replace(
        GatewayLimits(),
        max_requests_per_minute=60,
        request_burst=2,
    )
    connection = _new_connection(runtime, writer, limits=limits, clock=clock)

    for request_id in (1, 2, 3):
        await connection.receive(_modern_request(request_id, "ping"))
        await connection.wait_for_idle()
    clock.now += 1.0
    await connection.receive(_modern_request(4, "ping"))
    await connection.wait_for_idle()

    assert writer.values == [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "resultType": "complete",
                "_meta": {
                    "io.modelcontextprotocol/serverInfo": {
                        "name": "connection-runtime",
                        "version": "1.2.3",
                    }
                },
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "resultType": "complete",
                "_meta": {
                    "io.modelcontextprotocol/serverInfo": {
                        "name": "connection-runtime",
                        "version": "1.2.3",
                    }
                },
            },
        },
        _error(3, -32000, "Request rejected"),
        {
            "jsonrpc": "2.0",
            "id": 4,
            "result": {
                "resultType": "complete",
                "_meta": {
                    "io.modelcontextprotocol/serverInfo": {
                        "name": "connection-runtime",
                        "version": "1.2.3",
                    }
                },
            },
        },
    ]


@pytest.mark.asyncio
async def test_cancellation_before_dispatch_suppresses_runtime_and_output() -> None:
    """Scheduling cancellation too late must let immediately-cancelled work start."""

    runtime = _CoreRuntime()
    writer = _MemoryWriter()
    connection = _new_connection(runtime, writer)

    await connection.receive(_modern_request(1, "tools/call", {"name": "echo", "arguments": {"value": 7}}))
    await connection.receive(
        {
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": 1, "reason": "private-reason"},
        }
    )
    await connection.wait_for_idle()

    assert runtime.runtime_entries == 0
    assert writer.values == []


@pytest.mark.asyncio
async def test_cancellation_during_runtime_uses_the_context_token_and_suppresses_output() -> None:
    """A different token or result-after-cancel must leak a late tool response."""

    runtime = _CoreRuntime()
    runtime.call_release.clear()
    writer = _MemoryWriter()
    connection = _new_connection(runtime, writer)
    await connection.receive(_modern_request(1, "tools/call", {"name": "echo", "arguments": {"value": 7}}))
    await runtime.call_started.wait()
    tool_context = runtime.contexts[-1]

    await connection.receive(
        {
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": 1, "reason": "client_cancelled"},
        }
    )
    runtime.call_release.set()
    await connection.wait_for_idle()

    assert tool_context.cancellation is not None
    assert tool_context.cancellation.cancelled is True
    assert tool_context.cancellation.reason == "client_cancelled"
    assert writer.values == []


@pytest.mark.asyncio
async def test_typed_cancellation_only_suppresses_the_matching_id() -> None:
    """Collapsing integer 1 and string '1' must cancel the wrong request too."""

    runtime = _CoreRuntime()
    runtime.call_release.clear()
    writer = _MemoryWriter()
    connection = _new_connection(runtime, writer)
    for request_id in (1, "1"):
        await connection.receive(
            _modern_request(
                request_id,
                "tools/call",
                {"name": "echo", "arguments": {"value": 7}},
            )
        )
    while len(runtime.call_names) < 2:
        await asyncio.sleep(0.01)

    await connection.receive(
        {
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": 1},
        }
    )
    runtime.call_release.set()
    await connection.wait_for_idle()

    assert [response["id"] for response in writer.values] == ["1"]


@pytest.mark.asyncio
async def test_cancellation_at_writer_lock_suppresses_the_late_result() -> None:
    """Checking cancellation before the writer lock must leave a race window."""

    runtime = _CoreRuntime()
    writer = _MemoryWriter()
    writer.block()
    connection = _new_connection(runtime, writer)

    await connection.receive(_modern_request(1, "ping"))
    await writer.entered.wait()
    await connection.receive(_modern_request(2, "tools/call", {"name": "echo", "arguments": {"value": 7}}))
    await runtime.call_started.wait()
    while runtime.active_calls:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    await connection.receive(
        {
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": 2},
        }
    )
    writer.unblock()
    await connection.wait_for_idle()

    assert [response["id"] for response in writer.values] == [1]


@pytest.mark.asyncio
async def test_runtime_that_swallows_task_cancel_still_cannot_emit_a_late_result() -> None:
    """Task cancellation alone must not trust a runtime that returns afterward."""

    runtime = _CoreRuntime()
    runtime.call_release.clear()
    runtime.swallow_cancellation = True
    writer = _MemoryWriter()
    connection = _new_connection(runtime, writer)
    await connection.receive(_modern_request(1, "tools/call", {"name": "echo", "arguments": {"value": 7}}))
    await runtime.call_started.wait()

    await connection.receive(
        {
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": 1},
        }
    )
    runtime.call_release.set()
    await connection.wait_for_idle()

    assert writer.values == []


@pytest.mark.asyncio
async def test_shutdown_is_idempotent_cancels_work_and_closes_receive() -> None:
    """EOF-style shutdown must neither leak work nor accept post-close requests."""

    runtime = _CoreRuntime()
    runtime.call_release.clear()
    writer = _MemoryWriter()
    connection = _new_connection(runtime, writer)
    await connection.receive(_modern_request(1, "tools/call", {"name": "echo", "arguments": {"value": 7}}))
    await runtime.call_started.wait()

    await connection.shutdown()
    await connection.shutdown()

    assert writer.values == []
    assert runtime.active_calls == 0
    with pytest.raises(RuntimeError, match=r"^connection is closed$"):
        await connection.receive(_modern_request(2, "ping"))
