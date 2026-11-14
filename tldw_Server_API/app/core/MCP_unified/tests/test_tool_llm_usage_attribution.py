"""An MCP tool's LLM calls are recorded and quota-checked for the calling user (TASK-13585, ADR-067)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

import pytest

from tldw_Server_API.app.core.LLM_Calls.providers.base import ChatProvider
from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext
from tldw_Server_API.app.core.Usage import llm_accounting
from tldw_Server_API.app.core.Usage.llm_call_context import llm_call_context
from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_calls

pytestmark = pytest.mark.unit

TOOL = "llmtool.ask"


class _Provider(ChatProvider):
    """A provider with a native async chat that reports usage."""

    name = "mcp-fake"
    calls = 0

    def capabilities(self) -> dict[str, Any]:
        """Nothing special."""
        return {}

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Unused."""
        raise NotImplementedError

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        """Unused."""
        return iter(())

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Count the call and answer with usage."""
        type(self).calls += 1
        return {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 6, "completion_tokens": 2}}


class _LLMToolModule(BaseModule):
    """A module whose one tool makes an LLM call, as quizzes.generate does."""

    async def on_initialize(self) -> None:
        """Nothing to set up."""

    async def on_shutdown(self) -> None:
        """Nothing to tear down."""

    async def check_health(self) -> dict[str, bool]:
        """Always healthy."""
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        """One read tool."""
        return [{"name": TOOL, "description": "", "inputSchema": {"type": "object"}, "metadata": {"category": "read"}}]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        """Ask the provider once."""
        response = await _Provider().achat({"model": "m", "messages": [{"role": "user", "content": "q"}]})
        return {"answer": response["choices"][0]["message"]["content"]}


class _AllowAllRBAC:
    """Every permission is granted."""

    async def check_permission(self, *_args: Any, **_kwargs: Any) -> bool:
        """Allow."""
        return True


class _Registry:
    """Routes every tool to one module."""

    def __init__(self, module: BaseModule) -> None:
        """Hold the module."""
        self._module = module

    async def find_module_for_tool(self, _tool_name: str) -> BaseModule:
        """The one module."""
        return self._module

    def get_module_id_for_tool(self, _tool_name: str) -> str:
        """Its name."""
        return self._module.name


@pytest.fixture
async def rows(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture what would be written, and make the test's loop the accounting loop."""
    captured: list[dict[str, Any]] = []

    async def _fake_log(**kwargs: Any) -> None:
        """Record the row instead of writing it."""
        captured.append(kwargs)

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.usage_tracker.log_llm_usage", _fake_log)
    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    monkeypatch.setattr(_Provider, "calls", 0)
    return captured


def _protocol(module: BaseModule) -> MCPProtocol:
    """A protocol that routes tools/call to ``module``."""
    proto = MCPProtocol()
    proto.rbac_policy = _AllowAllRBAC()
    proto.module_registry = _Registry(module)
    return proto


async def _call_tool(proto: MCPProtocol, user_id: str | None = "7") -> dict[str, Any]:
    """One tools/call, as the MCP HTTP and WebSocket transports make it: no content scope, user in the context."""
    context = RequestContext(request_id="usage-1", user_id=user_id, metadata={})
    return await proto._handle_tools_call({"name": TOOL, "arguments": {}}, context)


async def test_a_tool_call_records_one_row_for_the_calling_user(rows) -> None:
    """The MCP user is the user the tool's LLM call is recorded for."""
    result = await _call_tool(_protocol(_LLMToolModule(ModuleConfig(name="llmtool"))))
    assert result["content"][0]["json"]["answer"] == "ok"
    await llm_accounting.drain_accounting()
    (row,) = rows
    assert (row["user_id"], row["provider"], row["prompt_tokens"], row["completion_tokens"]) == (7, "mcp-fake", 6, 2)
    assert (row["operation"], row["endpoint"]) == ("llm_call", "internal")


async def test_concurrent_tool_calls_are_recorded_for_their_own_users(rows) -> None:
    """Two users' tool calls in flight at once each record for their own user."""
    proto = _protocol(_LLMToolModule(ModuleConfig(name="llmtool")))
    await asyncio.gather(_call_tool(proto, "7"), _call_tool(proto, "8"), _call_tool(proto, "7"))
    await llm_accounting.drain_accounting()
    assert sorted(row["user_id"] for row in rows) == [7, 7, 8]


async def test_a_tool_call_with_no_numeric_user_records_nothing(rows) -> None:
    """A service principal is not a user: the call runs and nothing is written."""
    result = await _call_tool(_protocol(_LLMToolModule(ModuleConfig(name="llmtool"))), user_id="svc-bot")
    assert result["content"][0]["json"]["answer"] == "ok"
    await llm_accounting.drain_accounting()
    assert rows == []


async def test_an_over_quota_user_gets_a_tool_error_that_does_not_trip_the_module_breaker(rows, monkeypatch) -> None:
    """A refusal is the user's limit, not a module outage: more refusals than the breaker threshold leave it closed."""
    refuse_llm_calls(monkeypatch)
    module = _LLMToolModule(ModuleConfig(name="llmtool", circuit_breaker_threshold=2))
    proto = _protocol(module)
    for _ in range(3):
        result = await _call_tool(proto)
        assert result["isError"] is True
        assert result["content"][0]["json"]["reason_code"] == "llm_quota_exceeded"
    assert module.is_circuit_breaker_open() is False
    assert _Provider.calls == 0


async def test_a_tool_call_inside_an_admitted_request_is_not_refused(rows, monkeypatch) -> None:
    """A tool run by an admitted /chat/completions request (tool auto-exec) keeps the request's admission (spec §4a)."""
    refuse_llm_calls(monkeypatch)
    with llm_call_context(user_id=7, prechecked=True):
        result = await _call_tool(_protocol(_LLMToolModule(ModuleConfig(name="llmtool"))))
    assert result["content"][0]["json"]["answer"] == "ok"
    assert _Provider.calls == 1


async def test_a_tool_run_by_chat_tool_auto_exec_is_recorded_for_the_user(rows, monkeypatch) -> None:
    """/chat/completions records only its own model call, so a tool it runs records the tool's LLM call."""
    refuse_llm_calls(monkeypatch)
    with llm_call_context(user_id=7, prechecked=True, records_itself=True):  # what chat.py sets around the pipeline
        result = await _call_tool(_protocol(_LLMToolModule(ModuleConfig(name="llmtool"))))
    assert result["content"][0]["json"]["answer"] == "ok"
    await llm_accounting.drain_accounting()
    (row,) = rows
    assert (row["user_id"], row["prompt_tokens"], row["completion_tokens"]) == (7, 6, 2)
