"""Run the actual MCP loop against controlled transport and model boundaries."""

from __future__ import annotations

import asyncio
import copy
import re
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.base import AdapterConfig
from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_llm_caller import LLMResponse, LLMToolCall
from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_result_context import (
    READ_RESULT_TOOL,
    ToolResultPolicy,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.events import AgentEventKind
from tldw_Server_API.app.core.Agent_Client_Protocol.tool_gate import ToolGateResult

pytestmark = pytest.mark.integration
TEXT = "begin secret phrase " + "irrelevant " * 700 + "needle exact evidence at the end"
TOOLS = [{"name": "search", "inputSchema": {"type": "object"}}]


class RecordingCaller:
    """Request a result and optionally follow its real returned source handle."""

    def __init__(self, reread=False, stale_source=None):
        self.calls = []
        self.reread = reread
        self.stale_source = stale_source

    async def call(self, messages, tools):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        if len(self.calls) == 1:
            if self.stale_source:
                return LLMResponse(
                    tool_calls=[LLMToolCall("read1", READ_RESULT_TOOL, {"source_id": self.stale_source})]
                )
            return LLMResponse(tool_calls=[LLMToolCall("search1", "search", {"query": "needle"})])
        if len(self.calls) == 2 and self.reread:
            source = re.search(r"r_[a-f0-9]+", messages[-1]["content"]).group()
            return LLMResponse(
                tool_calls=[
                    LLMToolCall(
                        "read1",
                        READ_RESULT_TOOL,
                        {
                            "source_id": source,
                            "offset": 0,
                            "limit": 19,
                        },
                    )
                ]
            )
        return LLMResponse(text="Done")


def make_runner(caller=None, mode="excerpt", deny_reread=False, tools=None, worker=None, metrics_context=None):
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_runners import LLMDrivenRunner

    events, approvals = [], []
    transport = AsyncMock()
    transport.call_tool.return_value = {"content": [{"type": "text", "text": TEXT}]}
    caller = caller or RecordingCaller()

    async def approve(session_id, tool_name, arguments, **kwargs):
        approvals.append((session_id, tool_name, copy.deepcopy(arguments)))
        return ToolGateResult(approved=not (deny_reread and len(approvals) > 1))

    async def emit(event):
        events.append(event)

    gate = AsyncMock()
    gate.request_approval.side_effect = approve
    runner = LLMDrivenRunner(
        transport,
        emit,
        "session",
        asyncio.Event(),
        caller,
        gate,
        tools or TOOLS,
        result_policy=ToolResultPolicy(mode=mode, min_input_bytes=1024, max_output_bytes=768),
        result_worker=worker,
        run_first_metrics_context=metrics_context,
    )
    return runner, caller, events, approvals, transport


@pytest.mark.asyncio
async def test_policy_bounds_next_model_call_but_keeps_original_event():
    runner, caller, events, _, _ = make_runner()
    await runner.run([{"role": "user", "content": "needle"}])
    assert len(caller.calls[1][0][-1]["content"].encode()) <= 768
    assert "needle exact evidence" in caller.calls[1][0][-1]["content"]
    result = next(event for event in events if event.kind == AgentEventKind.TOOL_RESULT)
    assert result.payload["output"] == TEXT
    assert result.metadata["result_context"]["output_bytes"] <= 768


@pytest.mark.asyncio
async def test_off_preserves_history_and_tool_schema():
    runner, caller, events, _, _ = make_runner(mode="off")
    await runner.run([{"role": "user", "content": "needle"}])
    assert caller.calls[1][0][-1]["content"] == TEXT
    assert [tool["function"]["name"] for tool in caller.calls[0][1]] == ["search"]
    result = next(event for event in events if event.kind == AgentEventKind.TOOL_RESULT)
    assert "result_context" not in result.metadata


@pytest.mark.asyncio
async def test_exact_reread_rechecks_original_call():
    runner, caller, _, approvals, transport = make_runner(RecordingCaller(reread=True))
    await runner.run([{"role": "user", "content": "needle"}])
    assert TEXT[:19] in caller.calls[2][0][-1]["content"]
    assert approvals == [("session", "search", {"query": "needle"})] * 2
    assert transport.call_tool.await_count == 1


@pytest.mark.asyncio
async def test_denied_reread_exposes_no_source_text():
    runner, caller, _, _, _ = make_runner(RecordingCaller(reread=True), deny_reread=True)
    await runner.run([{"role": "user", "content": "needle"}])
    assert "Permission denied" in caller.calls[2][0][-1]["content"]
    assert TEXT[:19] not in caller.calls[2][0][-1]["content"]


@pytest.mark.asyncio
async def test_sources_do_not_survive_into_another_run():
    runner, caller, events, _, transport = make_runner()
    await runner.run([{"role": "user", "content": "needle"}])
    source_id = next(e for e in events if "result_context" in e.metadata).metadata["result_context"]["source_id"]
    caller.calls.clear()
    caller.stale_source = source_id
    await runner.run([{"role": "user", "content": "old source"}])
    assert "Unknown result source" in caller.calls[1][0][-1]["content"]
    assert transport.call_tool.await_count == 1


def test_reserved_read_tool_collision_is_rejected_when_enabled():
    with pytest.raises(ValueError, match="reserved"):
        make_runner(tools=[{"name": READ_RESULT_TOOL}])


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_provider", [None, "other-provider"])
async def test_adapter_rejects_unapproved_worker_provider(monkeypatch, worker_provider):
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter import MCPAdapter

    transport = AsyncMock()
    transport.list_tools.return_value = TOOLS
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.create_transport",
        lambda pc: transport,
    )
    adapter = MCPAdapter()
    await adapter.connect(
        AdapterConfig(
            event_callback=AsyncMock(),
            session_id="test",
            protocol_config={
                "mcp_orchestration": "llm_driven",
                "llm_caller": RecordingCaller(),
                "tool_gate": AsyncMock(),
                "mcp_llm_provider": "local-provider",
                "mcp_tool_result_policy": {"mode": "worker"},
                "mcp_result_worker": AsyncMock(),
                "mcp_result_worker_provider": worker_provider,
            },
        )
    )
    try:
        with pytest.raises(ValueError, match="same provider"):
            await adapter.send_prompt([{"role": "user", "content": "needle"}])
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_adapter_passes_explicit_policy_into_real_runner(monkeypatch):
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter import MCPAdapter

    transport = AsyncMock()
    transport.list_tools.return_value = TOOLS
    transport.call_tool.return_value = {"content": [{"type": "text", "text": TEXT}]}
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.create_transport",
        lambda pc: transport,
    )
    caller = RecordingCaller()
    gate = AsyncMock()
    gate.request_approval.return_value = ToolGateResult(approved=True)
    adapter = MCPAdapter()
    await adapter.connect(
        AdapterConfig(
            event_callback=AsyncMock(),
            session_id="test",
            protocol_config={
                "mcp_orchestration": "llm_driven",
                "llm_caller": caller,
                "tool_gate": gate,
                "mcp_tool_result_policy": {"mode": "excerpt", "min_input_bytes": 1024, "max_output_bytes": 768},
            },
        )
    )
    try:
        await adapter.send_prompt([{"role": "user", "content": "needle"}])
        assert len(caller.calls[1][0][-1]["content"].encode()) <= 768
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_source_store_is_cleared_even_when_traceback_retains_run(monkeypatch, failure):
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters import mcp_runners
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_result_context import ToolResultContext

    contexts, source_ids = [], []

    def capture_context(*args, **kwargs):
        value = ToolResultContext(*args, **kwargs)
        contexts.append(value)
        return value

    monkeypatch.setattr(mcp_runners, "ToolResultContext", capture_context)
    runner, _, _, _, _ = make_runner()

    async def fail_on_result(event):
        if event.kind == AgentEventKind.TOOL_RESULT:
            source_ids.append(event.metadata["result_context"]["source_id"])
            raise failure("stop")

    runner._emit = fail_on_result
    with pytest.raises(failure) as retained_error:
        await runner.run([{"role": "user", "content": "needle"}])
    assert retained_error.value.__traceback__ is not None
    with pytest.raises(ValueError, match="Unknown"):
        contexts[0].read(source_ids[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "excerpt", "worker"])
async def test_completed_result_survives_cancellation_at_transport_return(mode):
    runner, _, events, _, transport = make_runner(mode=mode, worker=AsyncMock())

    async def complete_then_cancel(*args):
        runner._cancel.set()
        return {"content": [{"type": "text", "text": TEXT}]}

    transport.call_tool.side_effect = complete_then_cancel
    if mode == "off":
        await runner.run([{"role": "user", "content": "needle"}])
    else:
        with pytest.raises(asyncio.CancelledError):
            await runner.run([{"role": "user", "content": "needle"}])
    assert [event.kind for event in events] == [AgentEventKind.TOOL_CALL, AgentEventKind.TOOL_RESULT]
    assert events[-1].payload == {"tool_name": "search", "output": TEXT, "is_error": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_method", ["event", "task"])
async def test_completed_result_survives_cancellation_during_worker(cancel_method):
    started, cancelled = asyncio.Event(), asyncio.Event()

    class WaitingWorker:
        async def call(self, messages, tools):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    runner, _, events, _, _ = make_runner(mode="worker", worker=WaitingWorker())
    job = asyncio.create_task(runner.run([{"role": "user", "content": "needle"}]))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        if cancel_method == "event":
            runner._cancel.set()
        else:
            job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(job, timeout=1)
    finally:
        job.cancel()
        await asyncio.gather(job, return_exceptions=True)
    assert cancelled.is_set()
    assert [event.kind for event in events] == [AgentEventKind.TOOL_CALL, AgentEventKind.TOOL_RESULT]
    assert events[-1].payload == {"tool_name": "search", "output": TEXT, "is_error": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("leading_unknown_read", [False, True])
async def test_internal_reads_do_not_replace_real_tool_selection_metrics(monkeypatch, leading_unknown_read):
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters import mcp_runners

    first_tools, fallbacks = [], []
    monkeypatch.setattr(mcp_runners.acp_metrics, "record_run_first_rollout", lambda **kwargs: None)
    monkeypatch.setattr(mcp_runners.acp_metrics, "record_run_first_completion_proxy", lambda **kwargs: None)
    monkeypatch.setattr(
        mcp_runners.acp_metrics,
        "record_run_first_first_tool",
        lambda **kwargs: first_tools.append(kwargs["first_tool"]),
    )
    monkeypatch.setattr(
        mcp_runners.acp_metrics,
        "record_run_first_fallback_after_run",
        lambda **kwargs: fallbacks.append(kwargs["fallback_tool"]),
    )

    class ToolSequenceCaller:
        def __init__(self):
            self.names = ([READ_RESULT_TOOL] if leading_unknown_read else []) + ["run", READ_RESULT_TOOL, "search"]
            self.index = 0

        async def call(self, messages, tools):
            if self.index == len(self.names):
                return LLMResponse(text="Done")
            name = self.names[self.index]
            self.index += 1
            arguments = {"query": "needle"}
            if name == READ_RESULT_TOOL:
                match = re.search(r"r_[a-f0-9]{32}", str(messages[-1]))
                arguments = {"source_id": match.group() if match else "unknown", "offset": 0, "limit": 19}
            return LLMResponse(tool_calls=[LLMToolCall(f"call-{self.index}", name, arguments)])

    runner, _, events, _, _ = make_runner(
        caller=ToolSequenceCaller(),
        tools=[{"name": "run", "inputSchema": {"type": "object"}}, *TOOLS],
        metrics_context={"presentation_variant": "compact"},
    )
    await runner.run([{"role": "user", "content": "needle"}])
    assert first_tools == ["run"]
    assert fallbacks == ["search"]
    assert events[-1].kind == AgentEventKind.COMPLETION
