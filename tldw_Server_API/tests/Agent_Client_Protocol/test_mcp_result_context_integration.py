"""Run the actual MCP loop against controlled transport and model boundaries."""

from __future__ import annotations

import asyncio
import copy
import re
from typing import Any, Literal
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.base import AdapterConfig
from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_llm_caller import LLMCaller, LLMResponse, LLMToolCall
from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_result_context import (
    READ_RESULT_TOOL,
    ToolResultContext,
    ToolResultPolicy,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_runners import LLMDrivenRunner
from tldw_Server_API.app.core.Agent_Client_Protocol.events import AgentEvent, AgentEventKind
from tldw_Server_API.app.core.Agent_Client_Protocol.tool_gate import ToolGateResult
from tldw_Server_API.app.core.exceptions import MCPResultSourceNotFoundError

pytestmark = pytest.mark.integration
TEXT = "begin secret phrase " + "irrelevant " * 700 + "needle exact evidence at the end"
TOOLS = [{"name": "search", "inputSchema": {"type": "object"}}]


class RecordingCaller(LLMCaller):
    """Request a result and optionally follow its real returned source handle."""

    def __init__(self, reread: bool = False, stale_source: str | None = None) -> None:
        """Configure reread behavior and retain isolated model request snapshots."""
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
        self.reread = reread
        self.stale_source = stale_source

    async def call(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        """Issue one search, optionally reread its source, then complete the run."""
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        if len(self.calls) == 1:
            if self.stale_source:
                return LLMResponse(
                    tool_calls=[LLMToolCall("read1", READ_RESULT_TOOL, {"source_id": self.stale_source})]
                )
            return LLMResponse(tool_calls=[LLMToolCall("search1", "search", {"query": "needle"})])
        if len(self.calls) == 2 and self.reread:
            match = re.search(r"r_[a-f0-9]+", messages[-1]["content"])
            assert match is not None
            source = match.group()
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


def make_runner(
    caller: RecordingCaller | None = None,
    mode: Literal["off", "excerpt", "worker"] = "excerpt",
    deny_reread: bool = False,
    tools: list[dict[str, Any]] | None = None,
    worker: LLMCaller | None = None,
    metrics_context: dict[str, Any] | None = None,
) -> tuple[LLMDrivenRunner, RecordingCaller, list[AgentEvent], list[tuple[str, str, dict[str, Any]]], AsyncMock]:
    """Build the real runner with recorded events, approvals, and transport calls."""
    events: list[AgentEvent] = []
    approvals: list[tuple[str, str, dict[str, Any]]] = []
    transport = AsyncMock()
    transport.call_tool.return_value = {"content": [{"type": "text", "text": TEXT}]}
    caller = caller or RecordingCaller()

    async def approve(session_id: str, tool_name: str, arguments: dict[str, Any], **kwargs: Any) -> ToolGateResult:
        """Record original call arguments and optionally deny subsequent reads."""
        approvals.append((session_id, tool_name, copy.deepcopy(arguments)))
        return ToolGateResult(approved=not (deny_reread and len(approvals) > 1))

    async def emit(event: AgentEvent) -> None:
        """Retain every event emitted by the actual runner."""
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
async def test_policy_bounds_next_model_call_but_keeps_original_event() -> None:
    """Evidence compaction must affect model context while preserving the raw event."""
    runner, caller, events, _, _ = make_runner()
    await runner.run([{"role": "user", "content": "needle"}])
    assert len(caller.calls[1][0][-1]["content"].encode()) <= 768
    assert "needle exact evidence" in caller.calls[1][0][-1]["content"]
    result = next(event for event in events if event.kind == AgentEventKind.TOOL_RESULT)
    assert result.payload["output"] == TEXT
    assert result.metadata["result_context"]["output_bytes"] <= 768


@pytest.mark.asyncio
async def test_off_preserves_history_and_tool_schema() -> None:
    """Disabled mode must preserve both original context and server tool definitions."""
    runner, caller, events, _, _ = make_runner(mode="off")
    await runner.run([{"role": "user", "content": "needle"}])
    assert caller.calls[1][0][-1]["content"] == TEXT
    assert [tool["function"]["name"] for tool in caller.calls[0][1]] == ["search"]
    result = next(event for event in events if event.kind == AgentEventKind.TOOL_RESULT)
    assert "result_context" not in result.metadata


@pytest.mark.asyncio
async def test_exact_reread_rechecks_original_call() -> None:
    """A source reread must reauthorize the original tool and arguments."""
    runner, caller, _, approvals, transport = make_runner(RecordingCaller(reread=True))
    await runner.run([{"role": "user", "content": "needle"}])
    assert TEXT[:19] in caller.calls[2][0][-1]["content"]
    assert approvals == [("session", "search", {"query": "needle"})] * 2
    assert transport.call_tool.await_count == 1


@pytest.mark.asyncio
async def test_denied_reread_exposes_no_source_text() -> None:
    """An authorization denial must prevent delivery of retained source text."""
    runner, caller, _, _, _ = make_runner(RecordingCaller(reread=True), deny_reread=True)
    await runner.run([{"role": "user", "content": "needle"}])
    assert "Permission denied" in caller.calls[2][0][-1]["content"]
    assert TEXT[:19] not in caller.calls[2][0][-1]["content"]


@pytest.mark.asyncio
async def test_sources_do_not_survive_into_another_run() -> None:
    """Reusing the runner must not make expired source handles readable."""
    runner, caller, events, _, transport = make_runner()
    await runner.run([{"role": "user", "content": "needle"}])
    source_id = next(e for e in events if "result_context" in e.metadata).metadata["result_context"]["source_id"]
    caller.calls.clear()
    caller.stale_source = source_id
    await runner.run([{"role": "user", "content": "old source"}])
    assert "Unknown result source" in caller.calls[1][0][-1]["content"]
    assert transport.call_tool.await_count == 1


def test_reserved_read_tool_collision_is_rejected_when_enabled() -> None:
    """An enabled policy must reject collisions with its virtual read tool."""
    with pytest.raises(ValueError, match="reserved"):
        make_runner(tools=[{"name": READ_RESULT_TOOL}])


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_provider", [None, "other-provider"])
async def test_adapter_rejects_unapproved_worker_provider(
    monkeypatch: pytest.MonkeyPatch, worker_provider: str | None
) -> None:
    """The adapter must reject absent or mismatched worker provider declarations."""
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
async def test_adapter_passes_explicit_policy_into_real_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit adapter settings must reach the real runner and bound its context."""
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
async def test_source_store_is_cleared_even_when_traceback_retains_run(
    monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    """Retained tracebacks must not keep readable source handles after run failure."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters import mcp_runners
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_result_context import ToolResultContext

    contexts, source_ids = [], []

    def capture_context(*args: Any, **kwargs: Any) -> ToolResultContext:
        """Retain the run store to check cleanup after its traceback survives."""
        value = ToolResultContext(*args, **kwargs)
        contexts.append(value)
        return value

    monkeypatch.setattr(mcp_runners, "ToolResultContext", capture_context)
    runner, _, _, _, _ = make_runner()

    async def fail_on_result(event: AgentEvent) -> None:
        """Capture a real retained handle before interrupting event delivery."""
        if event.kind == AgentEventKind.TOOL_RESULT:
            source_ids.append(event.metadata["result_context"]["source_id"])
            raise failure("stop")

    runner._emit = fail_on_result
    with pytest.raises(failure) as retained_error:
        await runner.run([{"role": "user", "content": "needle"}])
    assert retained_error.value.__traceback__ is not None
    with pytest.raises(MCPResultSourceNotFoundError, match="Unknown"):
        contexts[0].read(source_ids[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "excerpt", "worker"])
async def test_completed_result_survives_cancellation_at_transport_return(
    mode: Literal["off", "excerpt", "worker"],
) -> None:
    """A completed tool must retain its raw event even if cancellation arrives immediately."""
    runner, _, events, _, transport = make_runner(mode=mode, worker=AsyncMock())

    async def complete_then_cancel(*args: Any) -> dict[str, Any]:
        """Finish transport execution and signal cancellation before returning."""
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
async def test_completed_result_survives_cancellation_during_worker(cancel_method: str) -> None:
    """Event and task cancellation must preserve exactly one completed tool result."""
    started, cancelled = asyncio.Event(), asyncio.Event()

    class WaitingWorker(LLMCaller):
        """Keep worker I/O pending so the test can cancel the surrounding run."""

        async def call(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
            """Record worker entry and cleanup around an indefinitely pending call."""
            started.set()
            try:
                return await asyncio.Future[LLMResponse]()
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
async def test_internal_reads_do_not_replace_real_tool_selection_metrics(
    monkeypatch: pytest.MonkeyPatch, leading_unknown_read: bool
) -> None:
    """Virtual reads must not replace first-tool or real fallback measurements."""
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

    class ToolSequenceCaller(RecordingCaller):
        """Interleave local reads with real tools to exercise first-tool metrics."""

        def __init__(self) -> None:
            """Configure the call sequence with an optional leading stale read."""
            super().__init__()
            self.names = ([READ_RESULT_TOOL] if leading_unknown_read else []) + ["run", READ_RESULT_TOOL, "search"]
            self.index = 0

        async def call(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
            """Select the next tool, using an observed source handle for rereads."""
            if self.index == len(self.names):
                return LLMResponse(text="Done")
            name = self.names[self.index]
            self.index += 1
            arguments: dict[str, Any] = {"query": "needle"}
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


@pytest.mark.asyncio
async def test_worker_self_cancellation_does_not_abort_main_model_completion() -> None:
    """A cancelled optional worker still delivers raw events and usable evidence."""
    worker = AsyncMock()
    worker.call.side_effect = asyncio.CancelledError
    runner, caller, events, _, _ = make_runner(mode="worker", worker=worker)
    await runner.run([{"role": "user", "content": "needle"}])
    assert "needle exact evidence" in caller.calls[1][0][-1]["content"]
    results = [event for event in events if event.kind == AgentEventKind.TOOL_RESULT]
    assert len(results) == 1
    assert results[0].payload["output"] == TEXT
    assert results[0].metadata["result_context"]["outcome"] == "worker_error"
    assert events[-1].kind == AgentEventKind.COMPLETION
