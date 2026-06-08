"""Tests for protocol-side MCP tool-use reporting capture."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest

from mcp_unified.tool_use_reporting.models import ToolUseEvent

from tldw_Server_API.app.core.MCP_unified.auth.rate_limiter import RateLimitExceeded
from tldw_Server_API.app.core.MCP_unified.protocol import (
    ErrorCode,
    InvalidParamsException,
    MCPProtocol,
    RequestContext,
)


class _RecordingToolUseRecorder:
    def __init__(self) -> None:
        self.events: list[ToolUseEvent] = []

    async def record_tool_use(self, event: ToolUseEvent) -> None:
        self.events.append(event)


class _FailingToolUseRecorder:
    def __init__(self) -> None:
        self.called = False

    async def record_tool_use(self, event: ToolUseEvent) -> None:
        self.called = True
        raise RuntimeError(f"do not leak {event.requested_tool_name}")


class _AllowAllRbac:
    async def check_permission(self, *_args: Any, **_kwargs: Any) -> bool:
        return True


class _NoopRateLimiter:
    async def check_rate_limit(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _ToolOnlyRateLimiter:
    async def check_rate_limit(self, key: str, *_args: Any, **_kwargs: Any) -> None:
        if ":tool:" in key:
            raise RateLimitExceeded(1)


class _NoopMetrics:
    def __getattr__(self, _name: str):
        return lambda *args, **kwargs: None


class _Span:
    def set_attribute(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _Telemetry:
    def trace_context(self, *_args: Any, **_kwargs: Any):
        return contextlib.nullcontext(_Span())


class _ToolModule:
    name = "test_module"

    def __init__(self, *, write: bool = False, fail: BaseException | None = None) -> None:
        self.write = write
        self.fail = fail
        self.calls = 0
        self.config = SimpleNamespace(timeout_seconds=5)

    async def get_tools(self) -> list[dict[str, Any]]:
        return [self._tool_def("test.write" if self.write else "test.read")]

    async def get_tool_def(self, tool_name: str) -> dict[str, Any]:
        return self._tool_def(tool_name)

    def _tool_def(self, tool_name: str) -> dict[str, Any]:
        properties: dict[str, Any] = {"value": {"type": "string"}}
        if self.write:
            properties["idempotencyKey"] = {"type": "string"}
        return {
            "name": tool_name,
            "description": "",
            "inputSchema": {"type": "object", "properties": properties},
            "metadata": {
                "category": "management" if self.write else "read",
                "eval": {
                    "tool_prompt_id": f"mcp.{tool_name}.v1",
                    "tool_prompt_version": "2026.06.06",
                    "action_family": "write" if self.write else "read",
                    "result_kind": "json",
                    "prompt_variant": "builtin",
                },
            },
        }

    def sanitize_input(self, args: Any) -> Any:
        return args

    def validate_tool_arguments(self, _tool_name: str, _tool_args: dict[str, Any]) -> None:
        return None

    def is_write_tool_call(
        self,
        _tool_name: str,
        _tool_args: dict[str, Any],
        *,
        tool_def: dict[str, Any] | None = None,
    ) -> bool:
        return self.write

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: RequestContext | None = None,
    ) -> dict[str, Any]:
        del context
        if self.fail is not None:
            raise self.fail
        self.calls += 1
        return {"ok": True, "tool": tool_name, "call": self.calls, "arguments": arguments}

    async def execute_with_circuit_breaker(self, func, *args: Any, **kwargs: Any) -> Any:
        return await func(*args, **kwargs)


class _FilesystemPayloadModule(_ToolModule):
    name = "filesystem"

    def __init__(self) -> None:
        super().__init__(write=True)

    async def get_tools(self) -> list[dict[str, Any]]:
        return [self._tool_def("fs.patch")]

    def _tool_def(self, tool_name: str) -> dict[str, Any]:
        return {
            "name": tool_name,
            "description": "",
            "inputSchema": {"type": "object", "properties": {"diff": {"type": "string"}}},
            "metadata": {
                "category": "management",
                "eval": {
                    "tool_prompt_id": "mcp.fs.patch.v1",
                    "tool_prompt_version": "2026.06.04",
                    "task_families": ["filesystem_edit"],
                    "expected_result_kind": "structured_filesystem_edit",
                    "prompt_variant": "builtin",
                },
            },
        }

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: RequestContext | None = None,
    ) -> dict[str, Any]:
        del tool_name, arguments, context
        return {
            "path": "/Users/me/private/story.txt",
            "content": "SECRET FILE CONTENT",
            "read_receipt": "receipt-token-secret",
            "diff": "--- a/private/story.txt\n+++ b/private/story.txt\n",
            "eval": {
                "tool_name": "fs.patch",
                "tool_prompt_id": "mcp.fs.patch.v1",
                "tool_prompt_version": "2026.06.04",
                "action_family": "filesystem_edit",
                "result_kind": "structured_filesystem_edit",
                "path_filter_used": True,
                "truncated": False,
            },
        }


class _Registry:
    def __init__(self, module: _ToolModule) -> None:
        self.module = module

    async def find_module_for_tool(self, _tool_name: str) -> _ToolModule:
        return self.module

    def get_module_id_for_tool(self, _tool_name: str) -> str:
        return self.module.name


def _protocol(
    *,
    module: _ToolModule | None = None,
    recorder: Any | None = None,
    rate_limiter: Any | None = None,
) -> tuple[MCPProtocol, Any]:
    recorder = recorder or _RecordingToolUseRecorder()
    module = module or _ToolModule()
    deps = SimpleNamespace(
        module_registry=_Registry(module),
        rbac_policy=_AllowAllRbac(),
        rate_limiter=rate_limiter or _NoopRateLimiter(),
        metrics_collector=_NoopMetrics(),
        telemetry_provider=_Telemetry(),
        tool_catalog_provider=object(),
        redis_client_factory=lambda **kwargs: None,
        tool_use_recorder=recorder,
    )
    protocol = MCPProtocol(dependencies=deps)
    return protocol, recorder


def _request_context(metadata: dict[str, Any] | None = None) -> RequestContext:
    return RequestContext(
        request_id="tool-use-reporting",
        user_id="user-1",
        client_id="client-1",
        metadata=metadata or {},
    )


@pytest.mark.asyncio
async def test_protocol_records_successful_tool_use_event() -> None:
    protocol, recorder = _protocol()

    response = await protocol._handle_tools_call(
        {"name": "test.read", "arguments": {"value": "ok"}},
        _request_context(metadata={"profile_id": "architect", "model_id": "gpt-4.1"}),
    )

    assert response["tool"] == "test.read"
    event = recorder.events[-1]
    assert event.runtime_surface == "protocol"
    assert event.requested_tool_name == "test.read"
    assert event.effective_tool_name == "test.read"
    assert event.profile_id == "architect"
    assert event.model_id == "gpt-4.1"
    assert event.tool_prompt_id == "mcp.test.read.v1"
    assert event.status == "success"


@pytest.mark.asyncio
async def test_protocol_records_filesystem_eval_metadata_without_sensitive_payload() -> None:
    protocol, recorder = _protocol(module=_FilesystemPayloadModule())

    response = await protocol._handle_tools_call(
        {
            "name": "fs.patch",
            "arguments": {
                "diff": "--- a/private/story.txt\n+++ b/private/story.txt\n",
                "read_receipt_by_path": {"private/story.txt": "receipt-token-secret"},
            },
        },
        _request_context(metadata={"profile_id": "backend-engineer"}),
    )

    assert response["content"][0]["json"]["content"] == "SECRET FILE CONTENT"
    event = recorder.events[-1]
    dumped = event.model_dump_json()
    assert event.requested_tool_name == "fs.patch"
    assert event.action_family == "filesystem_edit"
    assert event.result_kind == "structured_filesystem_edit"
    assert event.path_filter_used is True
    assert event.truncated is False
    assert event.profile_id == "backend-engineer"
    assert "SECRET FILE CONTENT" not in dumped
    assert "receipt-token-secret" not in dumped
    assert "/Users/me" not in dumped
    assert "--- a/private" not in dumped


@pytest.mark.asyncio
async def test_protocol_records_prepare_denial_without_raw_error() -> None:
    protocol, recorder = _protocol()

    with pytest.raises(PermissionError):
        await protocol._handle_tools_call(
            {"name": "test.read", "arguments": {"path": "/Users/me/secret.txt"}},
            _request_context(metadata={"allowed_tools": ["other.tool"]}),
        )

    event = recorder.events[-1]
    assert event.execution_origin == "failed_before_execution"
    assert event.status == "denied"
    assert event.reason_code == "permission_denied"
    assert "/Users/me" not in event.model_dump_json()


@pytest.mark.asyncio
async def test_protocol_records_early_process_request_tool_name_error() -> None:
    protocol, recorder = _protocol()

    response = await protocol.process_request(
        {
            "jsonrpc": "2.0",
            "id": "req-1",
            "method": "tools/call",
            "params": {"name": "../secret", "arguments": {}},
        },
        _request_context(),
    )

    assert response.error.code == ErrorCode.INTERNAL_ERROR
    event = recorder.events[-1]
    assert event.runtime_surface == "protocol"
    assert event.requested_tool_name == "unknown"
    assert event.status == "invalid_params"
    assert event.execution_origin == "failed_before_execution"


@pytest.mark.asyncio
async def test_protocol_recorder_failure_does_not_change_tool_response() -> None:
    recorder = _FailingToolUseRecorder()
    protocol, _ = _protocol(recorder=recorder)

    response = await protocol._handle_tools_call(
        {"name": "test.read", "arguments": {}},
        _request_context(),
    )

    assert response["tool"] == "test.read"
    assert recorder.called is True


@pytest.mark.asyncio
async def test_protocol_event_build_failure_preserves_process_request_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, recorder = _protocol()

    def fail_event(**_kwargs: Any) -> ToolUseEvent:
        raise ValueError("telemetry build failed")

    monkeypatch.setattr(protocol, "_build_tool_use_event", fail_event)

    response = await protocol.process_request(
        {
            "jsonrpc": "2.0",
            "id": "req-1",
            "method": "tools/call",
            "params": {"name": "../secret", "arguments": {}},
        },
        _request_context(),
    )

    assert response.error.code == ErrorCode.INTERNAL_ERROR
    assert "Invalid tool name" in response.error.message
    assert recorder.events == []


@pytest.mark.asyncio
async def test_protocol_event_build_failure_preserves_prepare_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, recorder = _protocol()

    def fail_event(**_kwargs: Any) -> ToolUseEvent:
        raise ValueError("telemetry build failed")

    monkeypatch.setattr(protocol, "_build_tool_use_event", fail_event)

    with pytest.raises(PermissionError):
        await protocol._handle_tools_call(
            {"name": "test.read", "arguments": {"path": "/Users/me/secret.txt"}},
            _request_context(metadata={"allowed_tools": ["other.tool"]}),
        )

    assert recorder.events == []


@pytest.mark.asyncio
async def test_protocol_event_build_failure_preserves_tool_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, recorder = _protocol()

    def fail_event(**_kwargs: Any) -> ToolUseEvent:
        raise ValueError("telemetry build failed")

    monkeypatch.setattr(protocol, "_build_tool_use_event", fail_event)

    response = await protocol._handle_tools_call(
        {"name": "test.read", "arguments": {}},
        _request_context(),
    )

    assert response["tool"] == "test.read"
    assert recorder.events == []


@pytest.mark.asyncio
async def test_protocol_event_build_failure_preserves_tool_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, recorder = _protocol(module=_ToolModule(fail=ValueError("original failure")))

    def fail_event(**_kwargs: Any) -> ToolUseEvent:
        raise RuntimeError("telemetry build failed")

    monkeypatch.setattr(protocol, "_build_tool_use_event", fail_event)

    with pytest.raises(InvalidParamsException, match="original failure"):
        await protocol._handle_tools_call(
            {"name": "test.read", "arguments": {}},
            _request_context(),
        )

    assert recorder.events == []


@pytest.mark.asyncio
async def test_protocol_skips_when_tool_use_already_observed() -> None:
    protocol, recorder = _protocol()

    response = await protocol._handle_tools_call(
        {"name": "test.read", "arguments": {}},
        _request_context(metadata={"mcp_tool_use_observed": True}),
    )

    assert response["tool"] == "test.read"
    assert recorder.events == []


@pytest.mark.asyncio
async def test_protocol_records_execution_failure_without_raw_arguments() -> None:
    protocol, recorder = _protocol(module=_ToolModule(fail=ValueError("bad /Users/me/secret.txt")))

    with pytest.raises(Exception):
        await protocol._handle_tools_call(
            {"name": "test.read", "arguments": {"path": "/Users/me/secret.txt"}},
            _request_context(),
        )

    event = recorder.events[-1]
    assert event.status == "invalid_params"
    assert event.reason_code == "invalid_params"
    assert "/Users/me" not in event.model_dump_json()


@pytest.mark.asyncio
async def test_protocol_records_unavailable_failure_origin_as_unavailable() -> None:
    protocol, recorder = _protocol(module=_ToolModule(fail=LookupError("missing tool")))

    with pytest.raises(Exception):
        await protocol._handle_tools_call(
            {"name": "test.read", "arguments": {}},
            _request_context(),
        )

    event = recorder.events[-1]
    assert event.status == "unavailable"
    assert event.reason_code == "tool_unavailable"
    assert event.execution_origin == "unavailable"


@pytest.mark.asyncio
async def test_protocol_records_idempotency_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_DISABLE_WRITE_TOOLS", "false")
    from tldw_Server_API.app.core.MCP_unified.config import get_config

    get_config.cache_clear()  # type: ignore[attr-defined]
    module = _ToolModule(write=True)
    protocol, recorder = _protocol(module=module)
    context = _request_context()
    params = {
        "name": "test.write",
        "arguments": {"value": "A"},
        "idempotencyKey": "idem-1",
    }

    first = await protocol._handle_tools_call(params, context)
    second = await protocol._handle_tools_call(params, context)

    assert first["content"] == second["content"]
    assert module.calls == 1
    assert recorder.events[-1].execution_origin == "cached"
    assert recorder.events[-1].idempotency_replay is True


@pytest.mark.asyncio
async def test_protocol_process_request_does_not_double_record_handler_rate_limit() -> None:
    protocol, recorder = _protocol(rate_limiter=_ToolOnlyRateLimiter())

    with pytest.raises(RateLimitExceeded):
        await protocol.process_request(
            {
                "jsonrpc": "2.0",
                "id": "req-1",
                "method": "tools/call",
                "params": {"name": "test.read", "arguments": {}},
            },
            _request_context(),
        )

    rate_limited_events = [
        event for event in recorder.events if event.status == "rate_limited"
    ]
    assert len(rate_limited_events) == 1
    assert rate_limited_events[0].execution_origin == "failed_before_execution"
