from __future__ import annotations

import hashlib
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified import protocol as protocol_mod
from tldw_Server_API.app.core.MCP_unified.external_servers import manager as manager_mod
from tldw_Server_API.app.core.MCP_unified.external_servers.config_schema import (
    ExternalMCPServerConfig,
    ExternalTransportType,
    ExternalWebSocketConfig,
)
from tldw_Server_API.app.core.MCP_unified.external_servers.transports.base import (
    ExternalMCPTransportAdapter,
    ExternalToolCallResult,
    ExternalToolDefinition,
)
from tldw_Server_API.app.core.MCP_unified.protocol import (
    MCPProtocol,
    PreparedToolCall,
    RequestContext,
)
from tldw_Server_API.app.core.MCP_unified.tool_execution.canonical import canonical_json_bytes
from tldw_Server_API.app.core.MCP_unified.tool_execution.models import (
    CanonicalJsonSnapshot,
    IdempotencyExecutionPolicy,
    PreparedExecutionPolicy,
)
from tldw_Server_API.app.core.MCP_unified.tool_observability import (
    ensure_tool_definition_eval_metadata,
)

LEAKED_DETAIL = "backend exploded /tmp/mcp-secret-token token=sk-mcp-secret"


def test_protocol_hash_arguments_supports_class_level_compatibility_call() -> None:
    actual = MCPProtocol._hash_arguments({"query": "safe"})
    expected = "ae60598c68a349fda472f70368fb68f638ef19dcf2976fb3e42b00de44305901"
    if actual != expected:
        pytest.fail(f"Expected legacy class-level hash {expected}, got {actual}")


def _assert_safe_text(value: object) -> None:
    text = repr(value)
    assert LEAKED_DETAIL not in text
    assert "/tmp/mcp-secret-token" not in text
    assert "sk-mcp-secret" not in text


def _capture_protocol_logs(level: str = "DEBUG") -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = protocol_mod.logger.add(
        lambda message: messages.append(str(message)),
        level=level,
        format="{message} {extra}",
    )
    return messages, sink_id


def _capture_manager_logs(level: str = "WARNING") -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = manager_mod.logger.add(
        lambda message: messages.append(str(message)),
        level=level,
        format="{message} {extra}",
    )
    return messages, sink_id


def _server_config(server_id: str = "docs") -> ExternalMCPServerConfig:
    return ExternalMCPServerConfig(
        id=server_id,
        name="Docs",
        transport=ExternalTransportType.WEBSOCKET,
        websocket=ExternalWebSocketConfig(url="wss://docs.example/ws"),
    )


class _NoopRateLimiter:
    async def check_rate_limit(self, *args: Any, **kwargs: Any) -> None:
        return None


class _LeakyMetadata:
    def __bool__(self) -> bool:
        return True

    def get(self, key: str, default: Any = None) -> Any:
        del key, default
        raise RuntimeError(LEAKED_DETAIL)


class _FakeSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}
        self.recorded_exception_message: str | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


class _FakeTraceContext:
    def __init__(self, span: _FakeSpan) -> None:
        self._span = span

    def __enter__(self) -> _FakeSpan:
        return self._span

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_type, traceback
        if exc_value is not None:
            self._span.recorded_exception_message = str(exc_value)
        return False


class _FakeTelemetry:
    def __init__(self) -> None:
        self.spans: list[_FakeSpan] = []

    def trace_context(self, name: str, attributes: dict[str, Any]) -> _FakeTraceContext:
        del name, attributes
        span = _FakeSpan()
        self.spans.append(span)
        return _FakeTraceContext(span)


class _FailingToolModule:
    name = "demo_module"

    def __init__(self) -> None:
        self.tool_definition = ensure_tool_definition_eval_metadata(
            {
                "name": "demo.read",
                "inputSchema": {"type": "object"},
                "metadata": {"category": "read"},
            }
        )

    async def get_tools(self) -> list[dict[str, Any]]:
        return [self.tool_definition]

    async def get_tool_def(self, tool_name: str) -> dict[str, Any] | None:
        if tool_name == self.tool_definition["name"]:
            return self.tool_definition
        return None

    def is_write_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_def: dict[str, Any] | None = None,
    ) -> bool:
        del tool_name, arguments, tool_def
        return False

    async def execute_with_circuit_breaker(self, operation, *args: Any, **kwargs: Any) -> Any:
        return await operation(*args, **kwargs)

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: RequestContext | None = None,
    ) -> Any:
        del tool_name, arguments, context
        raise RuntimeError(LEAKED_DETAIL)


class _FailingToolRegistry:
    def __init__(self, module: _FailingToolModule) -> None:
        self.module = module

    async def find_module_for_tool(self, tool_name: str) -> _FailingToolModule | None:
        if tool_name == self.module.tool_definition["name"]:
            return self.module
        return None

    def get_module_id_for_tool(self, tool_name: str) -> str | None:
        if tool_name == self.module.tool_definition["name"]:
            return self.module.name
        return None


class _ExternalAdapter(ExternalMCPTransportAdapter):
    def __init__(
        self,
        *,
        connect_error: Exception | None = None,
        discovery_error: Exception | None = None,
        health_error: Exception | None = None,
        call_error: Exception | None = None,
    ) -> None:
        super().__init__(server_id="docs")
        self._connect_error = connect_error
        self._discovery_error = discovery_error
        self._health_error = health_error
        self._call_error = call_error

    @property
    def transport_name(self) -> str:
        return "websocket"

    async def connect(self) -> None:
        if self._connect_error is not None:
            raise self._connect_error

    async def close(self) -> None:
        return None

    async def health_check(self) -> dict[str, bool]:
        if self._health_error is not None:
            raise self._health_error
        return {"configured": True, "connected": True}

    async def list_tools(self) -> list[ExternalToolDefinition]:
        if self._discovery_error is not None:
            raise self._discovery_error
        return [
            ExternalToolDefinition(
                name="repo.search",
                description="Search repos",
                input_schema={"type": "object"},
                metadata={"category": "read"},
            )
        ]

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any = None,
        runtime_auth: Any = None,
    ) -> ExternalToolCallResult:
        del tool_name, arguments, context, runtime_auth
        if self._call_error is not None:
            raise self._call_error
        return ExternalToolCallResult(content=[{"type": "text", "text": "ok"}])


def _prepared_tool_call(
    protocol: MCPProtocol,
    context: RequestContext,
    module: _FailingToolModule,
) -> PreparedToolCall:
    tool_def = module.tool_definition
    tool_name = tool_def["name"]
    tool_args = {"query": "safe"}
    module_id = module.name
    tool_definition_encoded = canonical_json_bytes(tool_def, max_bytes=1_000_000)
    scope_reporting_encoded = canonical_json_bytes(None, max_bytes=256_000)
    tool_definition_snapshot = CanonicalJsonSnapshot(
        encoded=tool_definition_encoded,
        sha256=hashlib.sha256(tool_definition_encoded).hexdigest(),
    )
    scope_reporting_snapshot = CanonicalJsonSnapshot(
        encoded=scope_reporting_encoded,
        sha256=hashlib.sha256(scope_reporting_encoded).hexdigest(),
    )
    policy = PreparedExecutionPolicy(
        version=1,
        effect="read",
        rate_limit_category="read",
        rate_limit_fail_closed=False,
        idempotency=IdempotencyExecutionPolicy(
            inject_argument=False,
            ttl_seconds=300,
            contention_wait_seconds=5,
            finalize_seconds=5,
            lock_ttl_seconds=300,
            max_entries=512,
            max_result_bytes=256_000,
        ),
    )
    arguments_hash = protocol._hash_arguments(tool_args)
    context_fingerprint = protocol._fingerprint_request_context(context)
    integrity_tag = protocol._build_prepared_tool_call_integrity_tag(
        tool_name=tool_name,
        module_id=module_id,
        policy=policy,
        idempotency_cache_key=None,
        normalized_idempotency_key_digest="",
        arguments_hash=arguments_hash,
        context_fingerprint=context_fingerprint,
        idempotency_scope_fingerprint="",
        tool_definition_sha256=tool_definition_snapshot.sha256,
        scope_reporting_sha256=scope_reporting_snapshot.sha256,
    )
    return PreparedToolCall(
        tool_name=tool_name,
        tool_args=tool_args,
        module=module,
        module_id=module_id,
        policy=policy,
        tool_definition_snapshot=tool_definition_snapshot,
        scope_reporting_snapshot=scope_reporting_snapshot,
        normalized_idempotency_key=None,
        normalized_idempotency_key_digest="",
        idempotency_cache_key=None,
        arguments_hash=arguments_hash,
        context_fingerprint=context_fingerprint,
        idempotency_scope_fingerprint="",
        integrity_tag=integrity_tag,
        context=context,
    )


def test_protocol_audit_tool_failure_log_omits_raw_exception_message() -> None:
    protocol = MCPProtocol()
    context = RequestContext(request_id="req-audit", client_id="client-audit")
    messages, sink_id = _capture_protocol_logs(level="ERROR")
    try:
        protocol._audit_tool_event(
            context,
            "demo.read",
            "demo_module",
            status="failure",
            duration_ms=1.5,
            arguments_hash="abc123",
            error=RuntimeError(LEAKED_DETAIL),
        )
    finally:
        protocol_mod.logger.remove(sink_id)

    rendered_logs = "\n".join(messages)
    assert "MCP tool execution failed" in rendered_logs
    assert "RuntimeError" in rendered_logs
    _assert_safe_text(rendered_logs)


@pytest.mark.asyncio
async def test_protocol_metadata_probe_log_omits_raw_exception_message() -> None:
    protocol = MCPProtocol()
    protocol.rate_limiter = _NoopRateLimiter()
    context = RequestContext(request_id="req-meta", client_id="client-meta")
    context.metadata = _LeakyMetadata()
    messages, sink_id = _capture_protocol_logs(level="DEBUG")
    try:
        response = await protocol.process_request(
            {"jsonrpc": "2.0", "method": "ping", "id": "ping-1"},
            context,
        )
    finally:
        protocol_mod.logger.remove(sink_id)

    assert response is not None
    assert response.error is None
    rendered_logs = "\n".join(messages)
    assert "Failed to read rg_ingress_enforced" in rendered_logs
    _assert_safe_text(rendered_logs)
    assert "RuntimeError" in rendered_logs


@pytest.mark.asyncio
async def test_protocol_tool_execution_failure_log_omits_raw_exception_and_traceback() -> None:
    from tldw_Server_API.app.core.MCP_unified.adapters.tldw_runtime import (
        build_default_runtime_dependencies,
    )

    fake_telemetry = _FakeTelemetry()
    module = _FailingToolModule()
    deps = build_default_runtime_dependencies()
    deps.module_registry = _FailingToolRegistry(module)
    deps.telemetry_provider = fake_telemetry
    protocol = MCPProtocol(dependencies=deps)
    protocol.rate_limiter = _NoopRateLimiter()
    context = RequestContext(request_id="req-tool", client_id="client-tool")
    prepared = _prepared_tool_call(protocol, context, module)
    messages, sink_id = _capture_protocol_logs(level="ERROR")
    try:
        with pytest.raises(RuntimeError):
            await protocol.execute_prepared_tool_call(prepared)
    finally:
        protocol_mod.logger.remove(sink_id)

    rendered_logs = "\n".join(messages)
    assert "Tool execution failed" in rendered_logs
    assert "Traceback" not in rendered_logs
    assert "RuntimeError" in rendered_logs
    _assert_safe_text(rendered_logs)
    assert fake_telemetry.spans
    _assert_safe_text([span.attributes for span in fake_telemetry.spans])
    _assert_safe_text([span.recorded_exception_message for span in fake_telemetry.spans])


@pytest.mark.asyncio
async def test_external_server_initialize_sanitizes_discovery_status_and_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = manager_mod.ExternalServerManager().with_server_loader(lambda: [_server_config()])
    adapter = _ExternalAdapter(discovery_error=RuntimeError(LEAKED_DETAIL))
    monkeypatch.setattr(manager_mod, "build_transport_adapter", lambda _server: adapter)
    messages, sink_id = _capture_manager_logs(level="WARNING")
    try:
        await manager.initialize()
    finally:
        manager_mod.logger.remove(sink_id)

    server_rows = await manager.list_servers()
    rendered_logs = "\n".join(messages)
    assert manager.initialized is True
    assert manager._discovery_errors["docs"] == "external_server_initialization_failed"
    assert server_rows[0]["last_error"] == "external_server_initialization_failed"
    assert manager._snapshot_telemetry("docs")["last_error"] == "external_server_discovery_failed"
    assert "External MCP server initialization/discovery failed" in rendered_logs
    assert "RuntimeError" in rendered_logs
    _assert_safe_text(manager._discovery_errors)
    _assert_safe_text(server_rows)
    _assert_safe_text(manager._snapshot_telemetry("docs"))
    _assert_safe_text(rendered_logs)


@pytest.mark.asyncio
async def test_external_server_refresh_discovery_sanitizes_errors_and_telemetry() -> None:
    manager = manager_mod.ExternalServerManager()
    manager._servers = {"docs": _server_config()}
    manager._adapters = {"docs": _ExternalAdapter(discovery_error=RuntimeError(LEAKED_DETAIL))}
    manager._telemetry = {"docs": manager_mod.ExternalServerTelemetry()}

    result = await manager.refresh_discovery("docs")

    assert result["errors"]["docs"] == "external_server_discovery_failed"
    assert manager._discovery_errors["docs"] == "external_server_discovery_failed"
    assert manager._snapshot_telemetry("docs")["last_error"] == "external_server_discovery_failed"
    _assert_safe_text(result)
    _assert_safe_text(manager._discovery_errors)
    _assert_safe_text(manager._snapshot_telemetry("docs"))


@pytest.mark.asyncio
async def test_external_server_list_servers_sanitizes_health_check_error() -> None:
    manager = manager_mod.ExternalServerManager()
    manager._servers = {"docs": _server_config()}
    manager._adapters = {"docs": _ExternalAdapter(health_error=RuntimeError(LEAKED_DETAIL))}
    manager._telemetry = {"docs": manager_mod.ExternalServerTelemetry()}

    rows = await manager.list_servers()

    assert rows[0]["checks"] == {"configured": True, "connected": False, "error": True}
    assert rows[0]["last_error"] == "external_server_health_check_failed"
    _assert_safe_text(rows)


@pytest.mark.asyncio
async def test_external_server_connect_failure_sanitizes_telemetry_and_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = manager_mod.ExternalServerManager().with_server_loader(lambda: [_server_config()])
    adapter = _ExternalAdapter(connect_error=RuntimeError(LEAKED_DETAIL))
    monkeypatch.setattr(manager_mod, "build_transport_adapter", lambda _server: adapter)

    await manager.initialize()

    assert manager._discovery_errors["docs"] == "external_server_initialization_failed"
    assert manager._snapshot_telemetry("docs")["last_error"] == "external_server_connect_failed"
    _assert_safe_text(manager._discovery_errors)
    _assert_safe_text(manager._snapshot_telemetry("docs"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call_error", "expected_last_error"),
    [
        (TimeoutError(LEAKED_DETAIL), "external_server_call_timeout"),
        (RuntimeError(LEAKED_DETAIL), "external_server_call_failed"),
    ],
)
async def test_external_server_call_failure_sanitizes_telemetry(
    call_error: Exception,
    expected_last_error: str,
) -> None:
    manager = manager_mod.ExternalServerManager()
    adapter = _ExternalAdapter(call_error=call_error)

    with pytest.raises(type(call_error)):
        await manager._call_external_tool(
            server_id="docs",
            adapter=adapter,
            upstream_tool_name="repo.search",
            call_args={"q": "safe"},
            context=None,
            runtime_auth=None,
        )

    telemetry = manager._snapshot_telemetry("docs")
    assert telemetry["last_error"] == expected_last_error
    _assert_safe_text(telemetry)
