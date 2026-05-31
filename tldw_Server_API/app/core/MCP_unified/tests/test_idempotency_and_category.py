from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.command_runtime.adapters import (
    derive_step_idempotency_key,
)
from tldw_Server_API.app.core.MCP_unified.config import get_config
from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig, create_tool_definition
from tldw_Server_API.app.core.MCP_unified.modules.implementations.run_command_module import (
    RunCommandModule,
)
from tldw_Server_API.app.core.MCP_unified.modules.registry import get_module_registry
from tldw_Server_API.app.core.MCP_unified.monitoring.metrics import get_metrics_collector
from tldw_Server_API.app.core.MCP_unified.protocol import IdempotencyManager, MCPProtocol, MCPRequest, RequestContext


class AllowAllRBAC:
    async def check_permission(self, *args, **kwargs):
        return True


class _CountingWriteModule(BaseModule):
    def __init__(self, config: ModuleConfig):
        super().__init__(config)
        self.counter = 0

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            create_tool_definition(
                name="write_count",
                description="Increment and return count (write tool)",
                parameters={"properties": {"x": {"type": "string"}}, "required": ["x"]},
                metadata={"category": "ingestion"},
            )
        ]

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]):
        if tool_name == "write_count" and (not isinstance(arguments, dict) or "x" not in arguments):
            raise ValueError("x required")

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context=None):
        # Side effect: increment counter when actually executed
        self.counter += 1
        return f"count:{self.counter}:{arguments.get('x')}"


@pytest.mark.asyncio
async def test_idempotency_dedupes_write_calls():
    registry = get_module_registry()
    await registry.register_module("counting_write", _CountingWriteModule, ModuleConfig(name="counting_write"))

    proto = MCPProtocol()
    proto.rbac_policy = AllowAllRBAC()

    ctx = RequestContext(request_id="id1", user_id="u1", client_id="c1")

    # First call executes the tool
    req1 = MCPRequest(method="tools/call", params={
        "name": "write_count",
        "arguments": {"x": "A"},
        "idempotencyKey": "k-123",
    }, id="r1")
    resp1 = await proto.process_request(req1, ctx)
    assert resp1.error is None
    payload1 = resp1.result
    assert any(part.get("text", "").startswith("count:1:A") for part in payload1.get("content", []))

    # Second call with same idempotencyKey should not increment counter
    req2 = MCPRequest(method="tools/call", params={
        "name": "write_count",
        "arguments": {"x": "A"},
        "idempotencyKey": "k-123",
    }, id="r2")
    resp2 = await proto.process_request(req2, ctx)
    assert resp2.error is None
    payload2 = resp2.result
    assert any(part.get("text", "").startswith("count:1:A") for part in payload2.get("content", []))

    # Metrics should record one miss then one hit
    metrics = get_metrics_collector()
    hits = metrics._metrics.get("idempotency_hit")
    misses = metrics._metrics.get("idempotency_miss")
    assert hits and any(getattr(e, "labels", {}).get("tool") == "write_count" for e in hits)
    assert misses and any(getattr(e, "labels", {}).get("tool") == "write_count" for e in misses)


@pytest.mark.asyncio
async def test_idempotency_key_reuse_with_different_arguments_rejected():
    registry = get_module_registry()
    await registry.register_module("counting_write_diff_args", _CountingWriteModule, ModuleConfig(name="counting_write_diff_args"))

    proto = MCPProtocol()
    proto.rbac_policy = AllowAllRBAC()

    ctx = RequestContext(request_id="id2", user_id="u1", client_id="c1")

    req1 = MCPRequest(method="tools/call", params={
        "name": "write_count",
        "arguments": {"x": "A"},
        "idempotencyKey": "k-diff-1",
    }, id="r1")
    resp1 = await proto.process_request(req1, ctx)
    assert resp1.error is None
    payload1 = resp1.result
    assert any(part.get("text", "").startswith("count:1:A") for part in payload1.get("content", []))

    # Reusing the same key with different args must fail explicitly.
    req2 = MCPRequest(method="tools/call", params={
        "name": "write_count",
        "arguments": {"x": "B"},
        "idempotencyKey": "k-diff-1",
    }, id="r2")
    resp2 = await proto.process_request(req2, ctx)
    assert resp2.error is not None
    assert resp2.error.code == -32602
    assert "idempotency key" in resp2.error.message.lower()


@pytest.mark.asyncio
async def test_idempotency_key_isolated_across_users():
    registry = get_module_registry()
    await registry.register_module("counting_write_user_isolation", _CountingWriteModule, ModuleConfig(name="counting_write_user_isolation"))

    proto = MCPProtocol()
    proto.rbac_policy = AllowAllRBAC()

    req = MCPRequest(method="tools/call", params={
        "name": "write_count",
        "arguments": {"x": "A"},
        "idempotencyKey": "k-shared",
    }, id="r1")

    resp_user_1 = await proto.process_request(req, RequestContext(request_id="u1", user_id="u1", client_id="c1"))
    assert resp_user_1.error is None
    payload_user_1 = resp_user_1.result
    assert any(part.get("text", "").startswith("count:1:A") for part in payload_user_1.get("content", []))

    resp_user_2 = await proto.process_request(
        MCPRequest(
            method="tools/call",
            params={"name": "write_count", "arguments": {"x": "A"}, "idempotencyKey": "k-shared"},
            id="r2",
        ),
        RequestContext(request_id="u2", user_id="u2", client_id="c1"),
    )
    assert resp_user_2.error is None
    payload_user_2 = resp_user_2.result
    assert any(part.get("text", "").startswith("count:2:A") for part in payload_user_2.get("content", []))


class _CategoryProbeLimiter:
    def __init__(self):
        self.last_category = None

    async def check_rate_limit(self, *args, **kwargs):
        self.last_category = kwargs.get("category")
        return True


class _CategoryModule(BaseModule):
    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        # Explicit metadata category must be preferred by protocol
        return [create_tool_definition(
            name="echo_meta_ingestion",
            description="echo",
            parameters={"properties": {"m": {"type": "string"}}, "required": ["m"]},
            metadata={"category": "ingestion"},
        )]

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]):
        if "m" not in arguments:
            raise ValueError("m required")

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context=None):
        return arguments.get("m")


@pytest.mark.asyncio
async def test_category_prefers_metadata_over_config_mapping(monkeypatch):
    registry = get_module_registry()
    await registry.register_module("category_probe", _CategoryModule, ModuleConfig(name="category_probe"))

    proto = MCPProtocol()
    proto.rbac_policy = AllowAllRBAC()

    # Replace rate limiter with a probe
    probe = _CategoryProbeLimiter()
    proto.rate_limiter = probe

    ctx = RequestContext(request_id="cx1", user_id="u1", client_id="c1")
    req = MCPRequest(method="tools/call", params={"name": "echo_meta_ingestion", "arguments": {"m": "ok"}}, id="cx1")
    resp = await proto.process_request(req, ctx)
    assert resp.error is None
    # The category should be 'ingestion' from metadata
    assert probe.last_category == "ingestion"


@pytest.mark.asyncio
async def test_idempotency_local_lock_map_prunes_with_cache_bounds():
    manager = IdempotencyManager()

    async def _execute(value: str):
        return {"content": [{"type": "text", "text": value}]}

    for idx in range(10):
        key = f"k-{idx}"
        await manager.run(
            key,
            lambda i=idx: _execute(str(i)),
            ttl=1,
            max_size=3,
            lock_ttl=2,
        )

    # Local cache is size-bounded, and lock bookkeeping should track cache lifetime.
    assert len(manager._local_cache) <= 3
    assert len(manager._local_locks) <= 3


@pytest.mark.asyncio
async def test_idempotency_warns_when_redis_factory_returns_none(monkeypatch):
    from tldw_Server_API.app.core.MCP_unified import protocol as protocol_mod

    class _Config:
        """Config double that advertises Redis connection settings."""

        def get_redis_connection_params(self) -> dict[str, str]:
            """Return non-empty Redis params so the factory path is exercised."""
            return {"url": "redis://localhost:6379/0"}

    async def _none_factory(**_kwargs: Any) -> None:
        """Return None to simulate a misconfigured Redis factory."""
        return None

    warnings: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record_warning(message: str, *args: Any, **kwargs: Any) -> None:
        """Record warnings emitted by the protocol logger."""
        warnings.append((message, args, kwargs))

    monkeypatch.setattr(protocol_mod, "get_config", lambda: _Config())
    monkeypatch.setattr(protocol_mod.logger, "warning", _record_warning)

    manager = IdempotencyManager(redis_client_factory=_none_factory)

    assert await manager._ensure_redis() is False
    assert manager._redis_client is None
    assert any("returned None" in message for message, _args, _kwargs in warnings)


def test_nested_step_idempotency_is_stable_and_content_derived():
    first = derive_step_idempotency_key("parent-1", ["write", "notes.txt", "hello"], 0)
    second = derive_step_idempotency_key("parent-1", ["write", "notes.txt", "hello"], 0)
    different = derive_step_idempotency_key("parent-1", ["write", "notes.txt", "hello"], 1)
    changed_args = derive_step_idempotency_key("parent-1", ["write", "notes.txt", "goodbye"], 0)

    assert first == second
    assert first is not None
    assert different is not None
    assert first != different
    assert first != changed_args


@pytest.mark.asyncio
async def test_run_prepare_tool_call_classifies_write_chain_dynamically(monkeypatch):
    monkeypatch.setenv("MCP_DISABLE_WRITE_TOOLS", "true")
    try:
        get_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None

    registry = get_module_registry()
    await registry.register_module(
        "run_dynamic_write_prepare",
        RunCommandModule,
        ModuleConfig(name="run_dynamic_write_prepare"),
    )

    proto = MCPProtocol()
    proto.rbac_policy = AllowAllRBAC()
    ctx = RequestContext(request_id="run-write", user_id="u1", client_id="c1")

    with pytest.raises(PermissionError, match="Write tools are disabled"):
        await proto.prepare_tool_call(
            params={"name": "run", "arguments": {"command": "write notes.txt hi"}},
            context=ctx,
        )

    prepared = await proto.prepare_tool_call(
        params={"name": "run", "arguments": {"command": "ls"}},
        context=ctx,
    )
    assert prepared.is_write is False


@pytest.mark.asyncio
async def test_protocol_idempotency_key_is_forwarded_to_run_nested_steps(monkeypatch):
    monkeypatch.setenv("MCP_DISABLE_WRITE_TOOLS", "false")
    try:
        get_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None

    class _PreparedCall:
        def __init__(self, params: dict[str, Any], idempotency_key: str | None = None):
            self.params = dict(params)
            self.idempotency_key = idempotency_key

    class _ProtocolStub:
        def __init__(self) -> None:
            self.prepare_calls: list[_PreparedCall] = []

        async def _handle_tools_list(self, params: dict[str, Any], context: RequestContext) -> dict[str, Any]:
            return {"tools": [{"name": "fs.write_text", "module": "filesystem", "canExecute": True}]}

        async def prepare_tool_call(
            self,
            *,
            params: dict[str, Any],
            context: RequestContext,
            idempotency_key: str | None = None,
        ) -> _PreparedCall:
            prepared = _PreparedCall(params=params, idempotency_key=idempotency_key)
            self.prepare_calls.append(prepared)
            return prepared

        async def execute_prepared_tool_call(self, prepared: _PreparedCall) -> dict[str, Any]:
            return {
                "content": [{"type": "json", "json": {"path": "notes.txt", "bytes_written": 2}}],
                "tool": prepared.params.get("name"),
            }

    registry = get_module_registry()
    nested_protocol = _ProtocolStub()
    await registry.register_module(
        "run_nested_idempotency",
        RunCommandModule,
        ModuleConfig(name="run_nested_idempotency", settings={"protocol": nested_protocol}),
    )

    proto = MCPProtocol()
    proto.rbac_policy = AllowAllRBAC()
    ctx = RequestContext(request_id="run-idem-proto", user_id="u1", client_id="c1")

    response = await proto.process_request(
        MCPRequest(
            method="tools/call",
            params={
                "name": "run",
                "arguments": {"command": "write notes.txt hi"},
                "idempotencyKey": "demo-1",
            },
            id="run-1",
        ),
        ctx,
    )

    assert response.error is None
    assert nested_protocol.prepare_calls[0].idempotency_key == derive_step_idempotency_key(
        "demo-1",
        ["write", "notes.txt", "hi"],
        0,
    )


@pytest.mark.asyncio
async def test_prepare_tool_call_accepts_run_idempotency_key_inside_arguments(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_DISABLE_WRITE_TOOLS", "false")
    monkeypatch.setenv("MCP_AUDIT_LOG_FILE", str(tmp_path / "audit.log"))
    try:
        get_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass

    registry = get_module_registry()
    await registry.register_module(
        "run_nested_argument_idempotency",
        RunCommandModule,
        ModuleConfig(name="run_nested_argument_idempotency"),
    )

    proto = MCPProtocol()
    proto.rbac_policy = AllowAllRBAC()
    ctx = RequestContext(request_id="run-idem-args", user_id="u1", client_id="c1")

    prepared = await proto.prepare_tool_call(
        params={
            "name": "run",
            "arguments": {
                "command": "write notes.txt hi",
                "idempotencyKey": "demo-args-1",
            },
        },
        context=ctx,
    )

    assert prepared.normalized_idempotency_key == "demo-args-1"
