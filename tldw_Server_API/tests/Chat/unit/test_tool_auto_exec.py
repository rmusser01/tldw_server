from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Chat.tool_auto_exec import (
    build_tool_idempotency_key,
    execute_assistant_tool_calls,
)
from tldw_Server_API.app.core.Tools import tool_executor as tool_executor_module
from tldw_Server_API.app.core.Tools.tool_executor import (
    ToolApprovalRequiredError,
    ToolExecutionError,
    ToolExecutor,
)


class _StubExecutor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        tool_name = kwargs.get("tool_name")
        if tool_name == "notes.forbidden":
            raise ToolExecutionError("Permission denied for tool: notes.forbidden")
        if tool_name == "notes.slow":
            await asyncio.sleep(0.05)
            return {"result": {"value": "late"}, "module": "notes"}
        return {"result": {"echo": kwargs.get("arguments")}, "module": "notes"}


@pytest.mark.asyncio
@pytest.mark.unit
async def test_execute_assistant_tool_calls_success_with_idempotency() -> None:
    stub = _StubExecutor()
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "notes.search",
                "arguments": "{\"q\":\"hello\"}",
            },
        }
    ]

    batch = await execute_assistant_tool_calls(
        tool_calls=tool_calls,
        user_id="1",
        client_id="client-a",
        max_tool_calls=3,
        timeout_ms=5000,
        allow_catalog=["notes.*"],
        attach_idempotency=True,
        idempotency_seed="conv-123",
        tool_executor=stub,
    )

    assert batch.requested_calls == 1
    assert batch.processed_calls == 1
    assert batch.execution_attempts == 1
    assert batch.executed_calls == 1
    assert batch.truncated is False
    assert len(batch.results) == 1
    assert batch.results[0].ok is True
    assert batch.results[0].tool_name == "notes.search"
    assert batch.results[0].idempotency_key is not None
    assert stub.calls[0]["allowed_tools"] == ["notes.*"]
    assert batch.tool_messages()[0]["role"] == "tool"
    assert batch.event_payload()["tool_results"][0]["ok"] is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_execute_assistant_tool_calls_invalid_json_args() -> None:
    stub = _StubExecutor()
    tool_calls = [
        {
            "id": "bad_json",
            "type": "function",
            "function": {
                "name": "notes.search",
                "arguments": "{not-json",
            },
        }
    ]

    batch = await execute_assistant_tool_calls(
        tool_calls=tool_calls,
        user_id="1",
        client_id="client-a",
        max_tool_calls=3,
        timeout_ms=5000,
        allow_catalog=None,
        attach_idempotency=False,
        tool_executor=stub,
    )

    assert batch.execution_attempts == 0
    assert batch.executed_calls == 0
    assert len(batch.results) == 1
    assert batch.results[0].ok is False
    assert batch.results[0].skipped is True
    assert "valid JSON object" in (batch.results[0].error or "")
    assert stub.calls == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_execute_assistant_tool_calls_missing_name() -> None:
    stub = _StubExecutor()
    tool_calls = [
        {
            "id": "missing_name",
            "type": "function",
            "function": {
                "arguments": "{\"q\":\"hello\"}",
            },
        }
    ]

    batch = await execute_assistant_tool_calls(
        tool_calls=tool_calls,
        user_id="1",
        client_id="client-a",
        max_tool_calls=3,
        timeout_ms=5000,
        allow_catalog=None,
        attach_idempotency=False,
        tool_executor=stub,
    )

    assert batch.execution_attempts == 0
    assert len(batch.results) == 1
    assert batch.results[0].ok is False
    assert batch.results[0].skipped is True
    assert "missing function.name" in (batch.results[0].error or "")
    assert stub.calls == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_execute_assistant_tool_calls_permission_denied() -> None:
    stub = _StubExecutor()
    tool_calls = [
        {
            "id": "denied_call",
            "type": "function",
            "function": {
                "name": "notes.forbidden",
                "arguments": "{}",
            },
        }
    ]

    batch = await execute_assistant_tool_calls(
        tool_calls=tool_calls,
        user_id="1",
        client_id="client-a",
        max_tool_calls=3,
        timeout_ms=5000,
        allow_catalog=["notes.*"],
        attach_idempotency=False,
        tool_executor=stub,
    )

    assert batch.execution_attempts == 1
    assert batch.executed_calls == 0
    assert len(batch.results) == 1
    assert batch.results[0].ok is False
    assert "Permission denied" in (batch.results[0].error or "")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_execute_assistant_tool_calls_timeout() -> None:
    stub = _StubExecutor()
    tool_calls = [
        {
            "id": "slow_call",
            "type": "function",
            "function": {
                "name": "notes.slow",
                "arguments": "{}",
            },
        }
    ]

    batch = await execute_assistant_tool_calls(
        tool_calls=tool_calls,
        user_id="1",
        client_id="client-a",
        max_tool_calls=3,
        timeout_ms=1,
        allow_catalog=["notes.*"],
        attach_idempotency=False,
        tool_executor=stub,
    )

    assert batch.execution_attempts == 1
    assert batch.executed_calls == 0
    assert len(batch.results) == 1
    assert batch.results[0].ok is False
    assert batch.results[0].timed_out is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_execute_assistant_tool_calls_cap_and_allow_catalog() -> None:
    stub = _StubExecutor()
    tool_calls = [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "notes.search", "arguments": "{}"},
        },
        {
            "id": "c2",
            "type": "function",
            "function": {"name": "media.search", "arguments": "{}"},
        },
        {
            "id": "c3",
            "type": "function",
            "function": {"name": "notes.search", "arguments": "{}"},
        },
    ]

    batch = await execute_assistant_tool_calls(
        tool_calls=tool_calls,
        user_id="1",
        client_id="client-a",
        max_tool_calls=2,
        timeout_ms=5000,
        allow_catalog=["notes.*"],
        attach_idempotency=False,
        tool_executor=stub,
    )

    assert batch.requested_calls == 3
    assert batch.processed_calls == 2
    assert batch.truncated is True
    assert len(batch.results) == 2
    assert batch.execution_attempts == 1
    assert batch.executed_calls == 1
    assert batch.results[1].ok is False
    assert batch.results[1].skipped is True
    assert "allow-catalog" in (batch.results[1].error or "")


@pytest.mark.unit
def test_build_tool_idempotency_key_is_stable() -> None:
    arguments = {"a": 1, "b": {"c": 2}}
    k1 = build_tool_idempotency_key(
        seed="conv 123",
        tool_call_id="call_1",
        tool_name="notes.search",
        arguments=arguments,
    )
    k2 = build_tool_idempotency_key(
        seed="conv 123",
        tool_call_id="call_1",
        tool_name="notes.search",
        arguments=arguments,
    )
    assert k1 == k2
    assert k1.startswith("chat:")


# ---------------------------------------------------------------------------
# MCP approval-demand signal preservation (final review F1, 2026-10-12)
# ---------------------------------------------------------------------------


class _FakeProtocol:
    def __init__(self, response) -> None:
        self._response = response

    async def process_request(self, req, ctx):
        return self._response


class _FakeMcpServer:
    def __init__(self, response) -> None:
        self.protocol = _FakeProtocol(response)


def _error_response(message: str, data=None, code: int = -32001) -> SimpleNamespace:
    error = SimpleNamespace(message=message, data=data, code=code)
    return SimpleNamespace(error=error, result=None)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_tool_executor_preserves_mcp_approval_demand(monkeypatch) -> None:
    """An AUTHORIZATION_ERROR carrying data.approval raises the typed subclass."""
    response = _error_response(
        "Tool requires approval", data={"approval": {"ticket": "t-1"}}
    )
    monkeypatch.setattr(
        tool_executor_module, "get_mcp_server", lambda: _FakeMcpServer(response)
    )
    executor = ToolExecutor()

    with pytest.raises(ToolApprovalRequiredError) as exc_info:
        await executor.execute(
            user_id="7",
            client_id="scheduled_automation",
            tool_name="fs_write",
            arguments={},
        )

    assert isinstance(exc_info.value, ToolExecutionError)
    assert "requires approval" in str(exc_info.value)
    assert exc_info.value.approval == {"ticket": "t-1"}


@pytest.mark.asyncio
@pytest.mark.unit
async def test_tool_executor_plain_authorization_error_stays_plain(
    monkeypatch,
) -> None:
    """Authorization errors WITHOUT the approval payload stay plain errors."""
    response = _error_response(
        "Permission denied", data={"governance": {"decision": "deny"}}
    )
    monkeypatch.setattr(
        tool_executor_module, "get_mcp_server", lambda: _FakeMcpServer(response)
    )
    executor = ToolExecutor()

    with pytest.raises(ToolExecutionError) as exc_info:
        await executor.execute(
            user_id="7", client_id="api_client", tool_name="fs_write", arguments={}
        )

    assert not isinstance(exc_info.value, ToolApprovalRequiredError)


class _ApprovalDemandExecutor:
    async def execute(self, **kwargs):
        raise ToolApprovalRequiredError(
            "Tool requires approval", approval={"ticket": "t-1"}
        )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_execute_assistant_tool_calls_flags_approval_demand() -> None:
    """An MCP approval demand marks the record via the typed flag."""
    stub = _ApprovalDemandExecutor()
    tool_calls = [
        {
            "id": "approval_call",
            "type": "function",
            "function": {"name": "notes.locked_read", "arguments": "{}"},
        }
    ]

    batch = await execute_assistant_tool_calls(
        tool_calls=tool_calls,
        user_id="1",
        client_id="client-a",
        max_tool_calls=3,
        timeout_ms=5000,
        allow_catalog=["notes.*"],
        attach_idempotency=False,
        tool_executor=stub,
    )

    assert batch.execution_attempts == 1
    assert batch.executed_calls == 0
    assert len(batch.results) == 1
    record = batch.results[0]
    assert record.ok is False
    assert record.approval_required is True
    assert "requires approval" in (record.error or "")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_execute_assistant_tool_calls_plain_error_not_flagged() -> None:
    """A plain ToolExecutionError records approval_required=False."""
    tool_calls = [
        {
            "id": "denied",
            "type": "function",
            "function": {"name": "notes.forbidden", "arguments": "{}"},
        }
    ]

    batch = await execute_assistant_tool_calls(
        tool_calls=tool_calls,
        user_id="1",
        client_id="client-a",
        max_tool_calls=3,
        timeout_ms=5000,
        allow_catalog=["notes.*"],
        attach_idempotency=False,
        tool_executor=_StubExecutor(),
    )

    assert batch.results[0].ok is False
    assert batch.results[0].approval_required is False
