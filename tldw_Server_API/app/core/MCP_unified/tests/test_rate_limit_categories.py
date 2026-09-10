"""Category-specific rate limit tests for MCP Unified."""

import ast
import asyncio
import inspect
import json
import textwrap
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from fastapi import HTTPException
from loguru import logger

from tldw_Server_API.app.core.MCP_unified import protocol as protocol_module
from tldw_Server_API.app.core.MCP_unified.auth.rate_limiter import RateLimitExceeded
from tldw_Server_API.app.core.MCP_unified.config import get_config
from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.registry import get_module_registry
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, MCPRequest
from tldw_Server_API.app.core.MCP_unified.protocol_types import RequestContext
from tldw_Server_API.app.core.MCP_unified.server import MCPServer
from tldw_Server_API.app.core.MCP_unified.tool_execution.models import (
    IdempotencyExecutionPolicy,
    IdempotencyRunResult,
)
from tldw_Server_API.app.core.MCP_unified.tool_execution.runtime import ToolExecutionRuntime


class StubCategoryModule(BaseModule):
    async def on_initialize(self) -> None: ...
    async def on_shutdown(self) -> None: ...
    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}
    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "mock_ingest",
                "description": "",
                "inputSchema": {"type": "object"},
                "metadata": {"category": "ingestion"},
            },
            {
                "name": "mock_read",
                "description": "",
                "inputSchema": {"type": "object"},
                "metadata": {"category": "read"},
            },
        ]
    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        return f"ok:{tool_name}"
    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        # Tests only need the validator hook to exist; no real validation required.
        return None


class _AllowAllRBAC:
    async def check_permission(self, *args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        return True


class _AdmissionModule(BaseModule):
    def __init__(self, config: ModuleConfig) -> None:
        super().__init__(config)
        self.tool_name = str(config.settings["tool_name"])
        self.breaker_entry_count = 0
        self.execute_count = 0
        self.last_arguments: dict[str, Any] | None = None
        self.source_tool_def = {
            "name": self.tool_name,
            "description": "Rate admission policy test tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "idempotencyKey": {"type": "string"},
                },
                "required": ["value"],
                "additionalProperties": True,
            },
            "metadata": {
                "category": str(config.settings.get("category") or "management"),
                "rate_limit_fail_closed": config.settings.get("fail_closed", False),
            },
        }

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ready": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [self.source_tool_def]

    def is_write_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_def: dict[str, Any] | None = None,
    ) -> bool:
        del arguments, tool_def
        return tool_name == self.tool_name

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if tool_name == self.tool_name and not arguments.get("value"):
            raise ValueError("value is required")

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: RequestContext | None = None,
    ) -> dict[str, Any]:
        del tool_name, context
        self.execute_count += 1
        self.last_arguments = dict(arguments)
        return {"value": arguments["value"]}

    async def execute_with_circuit_breaker(
        self,
        operation: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.breaker_entry_count += 1
        return await super().execute_with_circuit_breaker(operation, *args, **kwargs)


class _IdempotencyProbe:
    def __init__(self) -> None:
        self.execute_keys: list[str] = []
        self.policies: list[IdempotencyExecutionPolicy] = []

    async def execute(
        self,
        key: str,
        arguments_hash: str,
        execute: Callable[[], Awaitable[dict[str, Any]]],
        *,
        policy: IdempotencyExecutionPolicy,
    ) -> IdempotencyRunResult:
        del arguments_hash
        self.execute_keys.append(key)
        self.policies.append(policy)
        return IdempotencyRunResult(
            payload=await execute(),
            from_cache=False,
            persistence="none",
        )

    async def shutdown(self) -> None:
        return None


class _AdmissionLimiter:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.failure = failure
        self.categories: list[str] = []

    async def check_rate_limit(self, _key: str, *, category: str) -> None:
        self.categories.append(category)
        if self.failure is not None:
            raise self.failure


def _runtime_config() -> SimpleNamespace:
    return SimpleNamespace(
        validate_input_schema=True,
        disable_write_tools=False,
        idempotency_ttl_seconds=300,
        idempotency_cache_size=512,
        idempotency_wait_seconds=5,
        idempotency_finalize_seconds=5,
        idempotency_result_max_bytes=256_000,
        module_timeout=30,
        tool_category_map={},
    )


async def _prepare_admission_call(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_closed: bool,
    idempotency_key: str | None = "admission-idempotency-key",
) -> tuple[MCPProtocol, _AdmissionModule, Any]:
    module_id = f"admission_{uuid4().hex}"
    tool_name = f"{module_id}.write"
    registry = get_module_registry()
    await registry.register_module(
        module_id,
        _AdmissionModule,
        ModuleConfig(
            name=module_id,
            settings={
                "tool_name": tool_name,
                "category": "management",
                "fail_closed": fail_closed,
            },
        ),
    )
    module = await registry.get_module(module_id)
    assert isinstance(module, _AdmissionModule)
    monkeypatch.setattr(protocol_module, "get_config", _runtime_config)
    protocol = MCPProtocol()
    protocol.rbac_policy = _AllowAllRBAC()
    prepared = await protocol.prepare_tool_call(
        params={"name": tool_name, "arguments": {"value": "alpha"}},
        context=RequestContext(request_id=f"request-{module_id}", user_id="user-1"),
        idempotency_key=idempotency_key,
    )
    return protocol, module, prepared


def _assert_expected_failure(payload: dict[str, Any], *, reason: str, message: str) -> None:
    execution_eval = payload["eval"]
    assert payload == {
        "content": [
            {
                "type": "json",
                "json": {
                    "status": "failed",
                    "reason_code": reason,
                    "message": message,
                },
            }
        ],
        "isError": True,
        "module": payload["module"],
        "tool": payload["tool"],
        "eval": execution_eval,
    }


@pytest.mark.asyncio
async def test_category_limits_ingestion_vs_read(monkeypatch):
    # Configure mapping and strict ingestion RPM
    monkeypatch.setenv("MCP_TOOL_CATEGORY_MAP", '{"mock_ingest":"ingestion","mock_read":"read"}')
    monkeypatch.setenv("MCP_RATE_LIMIT_RPM_INGESTION", "1")
    monkeypatch.setenv("MCP_RATE_LIMIT_RPM_READ", "999")
    monkeypatch.setenv("MCP_RATE_LIMIT_BURST_INGESTION", "1")
    # Reset config cache to pick up env
    try:
        get_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None

    server = MCPServer()
    await server.initialize()
    # Disable RBAC by allowing all for test
    class _AllowAll:
        async def check_permission(self, *args, **kwargs):
            return True
    server.protocol.rbac_policy = _AllowAll()

    class _StubLimiter:
        def __init__(self):
            self.ingestion_hits = 0

        async def check_rate_limit(self, _key: str, *, category: str = "default") -> None:
            if category != "ingestion":
                return
            self.ingestion_hits += 1
            if self.ingestion_hits > 2:
                raise RateLimitExceeded(1)

    server.protocol.rate_limiter = _StubLimiter()

    # Register stub module
    reg = server.module_registry
    await reg.register_module("stub", StubCategoryModule, ModuleConfig(name="stub"))

    # Helper to call tools via HTTP path (returns MCPResponse or raises HTTPException)
    async def call_tool(name: str):
        req = MCPRequest(method="tools/call", params={"name": name, "arguments": {"x": 1}}, id="t1")
        return await server.handle_http_request(req, user_id="u1")

    # First ingest call should pass
    r1 = await call_tool("mock_ingest")
    assert r1.error is None
    # Token-bucket priming allows one additional burst request; third should rate limit
    r2 = await call_tool("mock_ingest")
    assert r2.error is None
    with pytest.raises(HTTPException) as ei:
        await call_tool("mock_ingest")
    assert ei.value.status_code == 429

    # Read calls should be allowed liberally
    r2 = await call_tool("mock_read")
    assert r2.error is None
    r3 = await call_tool("mock_read")
    assert r3.error is None

    await server.shutdown()


@pytest.mark.asyncio
async def test_observer_snapshot_mutation_cannot_change_runtime_admission_or_write_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, module, prepared = await _prepare_admission_call(
        monkeypatch,
        fail_closed=True,
    )
    observer = prepared.tool_def
    assert observer is not None
    observer["metadata"]["category"] = "read"
    observer["metadata"]["rate_limit_fail_closed"] = False
    observer["metadata"]["effect"] = "read"
    observer["inputSchema"]["properties"].pop("idempotencyKey")

    limiter = _AdmissionLimiter()
    idempotency = _IdempotencyProbe()
    protocol.rate_limiter = limiter
    protocol._idempotency = idempotency

    payload = await protocol.execute_prepared_tool_call(prepared)

    assert payload.get("isError") is not True
    assert limiter.categories == ["management"]
    assert len(idempotency.execute_keys) == 1
    assert module.breaker_entry_count == 1
    assert module.execute_count == 1
    assert module.last_arguments == {
        "value": "alpha",
        "idempotencyKey": "admission-idempotency-key",
    }


@pytest.mark.asyncio
async def test_fail_closed_admission_backend_error_returns_exact_failure_before_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, module, prepared = await _prepare_admission_call(
        monkeypatch,
        fail_closed=True,
    )
    observer = prepared.tool_def
    assert observer is not None
    observer["metadata"]["rate_limit_fail_closed"] = False
    secret = "private-admission-backend-detail"
    limiter = _AdmissionLimiter(RuntimeError(secret))
    idempotency = _IdempotencyProbe()
    protocol.rate_limiter = limiter
    protocol._idempotency = idempotency
    captured_logs: list[str] = []
    sink_id = logger.add(captured_logs.append, format="{message}")

    try:
        payload = await protocol.execute_prepared_tool_call(prepared)
    finally:
        logger.remove(sink_id)

    _assert_expected_failure(
        payload,
        reason="rate_limit_unavailable",
        message="Rate-limit admission is temporarily unavailable.",
    )
    assert limiter.categories == ["management"]
    assert idempotency.execute_keys == []
    assert module.breaker_entry_count == 0
    assert module.execute_count == 0
    assert secret not in json.dumps(payload)
    admission_logs = [
        message for message in captured_logs if "rate admission unavailable" in message
    ]
    assert admission_logs
    assert all(secret not in message for message in admission_logs)


@pytest.mark.asyncio
async def test_unflagged_admission_backend_error_preserves_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, module, prepared = await _prepare_admission_call(
        monkeypatch,
        fail_closed=False,
        idempotency_key=None,
    )
    limiter = _AdmissionLimiter(RuntimeError("backend unavailable"))
    protocol.rate_limiter = limiter

    payload = await protocol.execute_prepared_tool_call(prepared)

    assert payload.get("isError") is not True
    assert limiter.categories == ["management"]
    assert module.breaker_entry_count == 1
    assert module.execute_count == 1


@pytest.mark.asyncio
async def test_rate_admission_cancellation_propagates_without_execution_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, module, prepared = await _prepare_admission_call(
        monkeypatch,
        fail_closed=True,
    )
    limiter = _AdmissionLimiter(asyncio.CancelledError())
    idempotency = _IdempotencyProbe()
    protocol.rate_limiter = limiter
    protocol._idempotency = idempotency
    captured_logs: list[str] = []
    sink_id = logger.add(captured_logs.append, format="{message}")

    try:
        with pytest.raises(asyncio.CancelledError):
            await protocol.execute_prepared_tool_call(prepared)
    finally:
        logger.remove(sink_id)

    assert limiter.categories == ["management"]
    assert idempotency.execute_keys == []
    assert module.breaker_entry_count == 0
    assert module.execute_count == 0
    assert getattr(idempotency, "_local_cache", {}) == {}
    assert all("rate admission unavailable" not in message for message in captured_logs)


def test_runtime_admission_has_no_domain_or_concrete_tool_name_branch() -> None:
    source = textwrap.dedent(inspect.getsource(ToolExecutionRuntime.execute_prepared_tool_call))
    lowered = source.lower()

    assert "skills" not in lowered
    assert "model" not in lowered
    assert "provider" not in lowered

    tree = ast.parse(source)
    concrete_tool_branches = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.If, ast.IfExp)):
            continue
        references_tool_name = any(
            isinstance(value, ast.Name) and value.id == "tool_name"
            for value in ast.walk(node.test)
        )
        contains_literal = any(
            isinstance(value, ast.Constant) and isinstance(value.value, str)
            for value in ast.walk(node.test)
        )
        if references_tool_name and contains_literal:
            concrete_tool_branches.append(node)

    assert concrete_tool_branches == []
