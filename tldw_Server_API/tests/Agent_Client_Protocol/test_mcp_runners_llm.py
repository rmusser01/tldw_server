"""Tests for LLMDrivenRunner — ReAct loop: LLM decides tools, ToolGate approves, transport executes."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_llm_caller import (
    LLMResponse,
    LLMToolCall,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.events import AgentEvent, AgentEventKind
from tldw_Server_API.app.core.Agent_Client_Protocol.tool_gate import ToolGateResult

pytestmark = pytest.mark.unit

SESSION_ID = "test-session-llm"

MCP_TOOLS = [
    {
        "name": "search",
        "description": "Web search",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    }
]


def _make_runner(
    llm_responses: list[LLMResponse] | None = None,
    gate_results: list[ToolGateResult] | None = None,
    transport_returns: list[dict] | None = None,
    cancel_after: int | None = None,
    max_iterations: int = 20,
    llm_tools: list[dict] | None = None,
    prompt_fragment: str | None = None,
    run_first_metrics_context: dict[str, object] | None = None,
):
    """Helper to build an LLMDrivenRunner with mocked dependencies."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_runners import (
        LLMDrivenRunner,
    )

    transport = AsyncMock()
    if transport_returns:
        transport.call_tool.side_effect = transport_returns
    else:
        transport.call_tool.return_value = {"content": [{"type": "text", "text": "result"}]}

    llm_caller = AsyncMock()
    if llm_responses:
        llm_caller.call.side_effect = llm_responses
    else:
        llm_caller.call.return_value = LLMResponse(text="Done")

    tool_gate = AsyncMock()
    if gate_results:
        tool_gate.request_approval.side_effect = gate_results
    else:
        tool_gate.request_approval.return_value = ToolGateResult(approved=True)

    callback = AsyncMock()
    cancel_event = asyncio.Event()

    # If cancel_after is set, wrap callback to set cancel after N calls
    if cancel_after is not None:
        original_callback = callback

        call_count = 0

        async def cancelling_callback(event):
            nonlocal call_count
            call_count += 1
            await original_callback(event)
            if call_count >= cancel_after:
                cancel_event.set()

        callback_fn = cancelling_callback
    else:
        callback_fn = callback

    runner = LLMDrivenRunner(
        transport=transport,
        event_callback=callback_fn,
        session_id=SESSION_ID,
        cancel_event=cancel_event,
        llm_caller=llm_caller,
        tool_gate=tool_gate,
        tools=MCP_TOOLS,
        max_iterations=max_iterations,
        llm_tools=llm_tools,
        prompt_fragment=prompt_fragment,
        run_first_metrics_context=run_first_metrics_context,
    )
    return runner, transport, llm_caller, tool_gate, callback, cancel_event


@pytest.mark.asyncio
async def test_llm_driven_single_turn():
    """LLM returns text immediately; verify single COMPLETION event."""
    runner, transport, _llm_caller, _tool_gate, callback, _cancel_event = _make_runner(
        llm_responses=[LLMResponse(text="The answer is 42")]
    )

    await runner.run([{"role": "user", "content": "what is the answer?"}])

    assert callback.await_count == 1
    event: AgentEvent = callback.call_args_list[0][0][0]
    assert event.kind == AgentEventKind.COMPLETION
    assert event.payload["text"] == "The answer is 42"
    assert event.payload["stop_reason"] == "end_turn"
    transport.call_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_llm_driven_multi_turn():
    """LLM returns tool_call first, then text. Verify TOOL_CALL -> TOOL_RESULT -> COMPLETION."""
    tc = LLMToolCall(id="tc1", name="search", arguments={"query": "hello"})
    runner, _transport, _llm_caller, _tool_gate, callback, _cancel_event = _make_runner(
        llm_responses=[
            LLMResponse(tool_calls=[tc]),
            LLMResponse(text="Found the answer"),
        ],
        transport_returns=[
            {"content": [{"type": "text", "text": "search result"}]},
        ],
    )

    await runner.run([{"role": "user", "content": "search hello"}])

    events = [call[0][0] for call in callback.call_args_list]
    assert len(events) == 3

    assert events[0].kind == AgentEventKind.TOOL_CALL
    assert events[0].payload["tool_name"] == "search"
    assert events[0].payload["arguments"] == {"query": "hello"}

    assert events[1].kind == AgentEventKind.TOOL_RESULT
    assert events[1].payload["tool_name"] == "search"
    assert events[1].payload["output"] == "search result"
    assert events[1].payload["is_error"] is False

    assert events[2].kind == AgentEventKind.COMPLETION
    assert events[2].payload["text"] == "Found the answer"


@pytest.mark.asyncio
async def test_llm_driven_max_iterations():
    """LLM always returns tool_calls; verify max_iterations COMPLETION."""
    tc = LLMToolCall(id="tc1", name="search", arguments={"query": "loop"})
    # Return tool_calls every time (more than max_iterations)
    responses = [LLMResponse(tool_calls=[tc]) for _ in range(5)]
    runner, _transport, _llm_caller, _tool_gate, callback, _cancel_event = _make_runner(
        llm_responses=responses,
        max_iterations=3,
    )

    await runner.run([{"role": "user", "content": "loop"}])

    events = [call[0][0] for call in callback.call_args_list]
    # Last event should be COMPLETION with max_iterations
    last = events[-1]
    assert last.kind == AgentEventKind.COMPLETION
    assert last.payload["stop_reason"] == "max_iterations"


@pytest.mark.asyncio
async def test_llm_driven_cancel():
    """Cancel event set after first iteration; verify loop stops."""
    tc = LLMToolCall(id="tc1", name="search", arguments={"query": "test"})
    responses = [
        LLMResponse(tool_calls=[tc]),
        LLMResponse(tool_calls=[tc]),
        LLMResponse(text="done"),
    ]
    # Cancel after 2 events (TOOL_CALL + TOOL_RESULT from first iteration)
    runner, _transport, llm_caller, _tool_gate, _callback, cancel_event = _make_runner(
        llm_responses=responses,
        max_iterations=10,
        cancel_after=2,
    )

    await runner.run([{"role": "user", "content": "test"}])

    # LLM should have been called once (first iteration), then cancel kicks in
    assert llm_caller.call.await_count == 1


@pytest.mark.asyncio
async def test_llm_driven_cancel_stops_remaining_tool_calls_in_same_turn():
    """Cancellation after one tool result should prevent later tool approvals/executions."""
    tool_calls = [
        LLMToolCall(id="tc1", name="search", arguments={"query": "first"}),
        LLMToolCall(id="tc2", name="search", arguments={"query": "second"}),
    ]
    runner, transport, llm_caller, tool_gate, callback, cancel_event = _make_runner(
        llm_responses=[
            LLMResponse(tool_calls=tool_calls),
            LLMResponse(text="done"),
        ],
        transport_returns=[
            {"content": [{"type": "text", "text": "first result"}]},
            {"content": [{"type": "text", "text": "second result"}]},
        ],
        cancel_after=2,
    )

    await runner.run([{"role": "user", "content": "test"}])

    assert cancel_event.is_set() is True
    assert llm_caller.call.await_count == 1
    assert tool_gate.request_approval.await_count == 1
    assert transport.call_tool.await_count == 1

    events = [call[0][0] for call in callback.call_args_list]
    assert [event.kind for event in events] == [
        AgentEventKind.TOOL_CALL,
        AgentEventKind.TOOL_RESULT,
    ]


@pytest.mark.asyncio
async def test_llm_driven_logs_run_first_metric_failures(monkeypatch):
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters import mcp_runners

    runner, _transport, _llm_caller, _tool_gate, _callback, _cancel_event = _make_runner(
        llm_responses=[LLMResponse(text="Done")],
        run_first_metrics_context={
            "agent_type": "mcp",
            "presentation_variant": "acp_phase2a_v1",
            "cohort": "gated",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "eligible": True,
            "ineligible_reason": None,
        },
    )
    warning = Mock()

    def _boom(**_kwargs):
        raise RuntimeError("metrics unavailable")

    monkeypatch.setattr(mcp_runners.acp_metrics, "record_run_first_rollout", _boom)
    monkeypatch.setattr(mcp_runners.logger, "warning", warning)

    await runner.run([{"role": "user", "content": "hi"}])

    assert warning.called
    assert "ACP run-first metric emission failed for rollout" in warning.call_args[0][0]


@pytest.mark.asyncio
async def test_llm_driven_governance_denial():
    """ToolGate returns denied; verify TOOL_RESULT with is_error."""
    tc = LLMToolCall(id="tc1", name="search", arguments={"query": "forbidden"})
    runner, transport, _llm_caller, _tool_gate, callback, _cancel_event = _make_runner(
        llm_responses=[
            LLMResponse(tool_calls=[tc]),
            LLMResponse(text="Okay, denied"),
        ],
        gate_results=[ToolGateResult(approved=False, reason="Not allowed")],
    )

    await runner.run([{"role": "user", "content": "do forbidden thing"}])

    events = [call[0][0] for call in callback.call_args_list]
    # Should have: denied TOOL_RESULT, then COMPLETION
    denied = events[0]
    assert denied.kind == AgentEventKind.TOOL_RESULT
    assert denied.payload["is_error"] is True
    assert "Not allowed" in denied.payload["output"]

    # Transport should NOT have been called (denied)
    transport.call_tool.assert_not_awaited()

    completion = events[-1]
    assert completion.kind == AgentEventKind.COMPLETION


@pytest.mark.asyncio
async def test_llm_driven_transport_error():
    """Transport raises during tool execution; verify ERROR event."""
    tc = LLMToolCall(id="tc1", name="search", arguments={"query": "boom"})
    runner, transport, _llm_caller, _tool_gate, callback, _cancel_event = _make_runner(
        llm_responses=[
            LLMResponse(tool_calls=[tc]),
            LLMResponse(text="recovered"),
        ],
        transport_returns=[RuntimeError("transport failed")],
    )
    # Override: side_effect with exception
    transport.call_tool.side_effect = RuntimeError("transport failed")

    await runner.run([{"role": "user", "content": "boom"}])

    events = [call[0][0] for call in callback.call_args_list]
    # TOOL_CALL, then TOOL_RESULT with error
    tool_call_ev = events[0]
    assert tool_call_ev.kind == AgentEventKind.TOOL_CALL

    tool_result_ev = events[1]
    assert tool_result_ev.kind == AgentEventKind.TOOL_RESULT
    assert tool_result_ev.payload["is_error"] is True
    assert "transport failed" in tool_result_ev.payload["output"]


@pytest.mark.asyncio
async def test_llm_driven_llm_error_emits_error_event():
    """LLM call failures should be surfaced as terminal ERROR events."""
    runner, _transport, llm_caller, _tool_gate, callback, _cancel_event = _make_runner()
    llm_caller.call.side_effect = RuntimeError("llm failed")

    await runner.run([{"role": "user", "content": "boom"}])

    assert callback.await_count == 1
    event: AgentEvent = callback.call_args_list[0][0][0]
    assert event.kind == AgentEventKind.ERROR
    assert "llm failed" in event.payload["error"]


@pytest.mark.asyncio
async def test_llm_driven_gate_error_emits_error_event():
    """Approval-gate failures should be surfaced as terminal ERROR events."""
    tc = LLMToolCall(id="tc1", name="search", arguments={"query": "boom"})
    runner, _transport, _llm_caller, tool_gate, callback, _cancel_event = _make_runner(
        llm_responses=[LLMResponse(tool_calls=[tc])]
    )
    tool_gate.request_approval.side_effect = RuntimeError("gate failed")

    await runner.run([{"role": "user", "content": "boom"}])

    assert callback.await_count == 1
    event: AgentEvent = callback.call_args_list[0][0][0]
    assert event.kind == AgentEventKind.ERROR
    assert "gate failed" in event.payload["error"]


@pytest.mark.asyncio
async def test_llm_driven_with_governance_filter_emits_single_permission_request():
    """Governance-backed runs should not re-enter approval for already-approved tool calls."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.event_bus import SessionEventBus
    from tldw_Server_API.app.core.Agent_Client_Protocol.governance_filter import (
        GovernanceFilter,
        GovernanceToolGate,
    )
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_runners import (
        LLMDrivenRunner,
    )

    tc = LLMToolCall(id="tc1", name="bash", arguments={"cmd": "ls"})
    transport = AsyncMock()
    transport.call_tool.return_value = {"content": [{"type": "text", "text": "ok"}]}
    llm_caller = AsyncMock()
    llm_caller.call.side_effect = [
        LLMResponse(tool_calls=[tc]),
        LLMResponse(text="done"),
    ]

    bus = SessionEventBus(session_id=SESSION_ID)
    gov = GovernanceFilter(bus=bus)
    gate = GovernanceToolGate(governance_filter=gov)
    queue = bus.subscribe("test")

    async def approve_once():
        permission_request = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert permission_request.kind == AgentEventKind.PERMISSION_REQUEST
        await gov.on_permission_response(permission_request.payload["request_id"], decision="approve")

    approve_task = asyncio.create_task(approve_once())

    runner = LLMDrivenRunner(
        transport=transport,
        event_callback=gov.process,
        session_id=SESSION_ID,
        cancel_event=asyncio.Event(),
        llm_caller=llm_caller,
        tool_gate=gate,
        tools=MCP_TOOLS,
    )

    await runner.run([{"role": "user", "content": "run ls"}])
    await approve_task

    events = []
    while not queue.empty():
        events.append(await queue.get())

    assert [event.kind for event in events] == [
        AgentEventKind.TOOL_CALL,
        AgentEventKind.TOOL_RESULT,
        AgentEventKind.COMPLETION,
    ]


@pytest.mark.asyncio
async def test_llm_driven_uses_presented_tools_and_prompt_fragment():
    """Presented ACP tools and prompt fragment should be passed through unchanged."""
    llm_tools = [
        {
            "type": "function",
            "function": {
                "name": "run",
                "description": "Preferred first tool.",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Fallback tool.",
                "parameters": {"type": "object"},
            },
        },
    ]
    runner, _transport, llm_caller, _tool_gate, _callback, _cancel_event = _make_runner(
        llm_responses=[LLMResponse(text="done")],
        llm_tools=llm_tools,
        prompt_fragment="ACP run-first guidance.",
    )

    await runner.run([{"role": "user", "content": "hello"}])

    messages_arg, tools_arg = llm_caller.call.await_args.args
    assert messages_arg[0] == {"role": "system", "content": "ACP run-first guidance."}
    assert messages_arg[1] == {"role": "user", "content": "hello"}
    assert [tool["function"]["name"] for tool in tools_arg] == ["run", "search"]


@pytest.mark.asyncio
async def test_llm_driven_records_run_first_metrics_for_fallback_after_run(monkeypatch):
    """ACP runner should record rollout, first tool, fallback, and end-turn completion."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters import mcp_runners

    rollout_calls: list[dict] = []
    first_tool_calls: list[dict] = []
    fallback_calls: list[dict] = []
    completion_calls: list[dict] = []
    monkeypatch.setattr(
        mcp_runners.acp_metrics,
        "record_run_first_rollout",
        lambda **kwargs: rollout_calls.append(kwargs),
    )
    monkeypatch.setattr(
        mcp_runners.acp_metrics,
        "record_run_first_first_tool",
        lambda **kwargs: first_tool_calls.append(kwargs),
    )
    monkeypatch.setattr(
        mcp_runners.acp_metrics,
        "record_run_first_fallback_after_run",
        lambda **kwargs: fallback_calls.append(kwargs),
    )
    monkeypatch.setattr(
        mcp_runners.acp_metrics,
        "record_run_first_completion_proxy",
        lambda **kwargs: completion_calls.append(kwargs),
    )

    tool_calls = [
        LLMToolCall(id="tc1", name="run", arguments={"command": "ls"}),
        LLMToolCall(id="tc2", name="search", arguments={"query": "docs"}),
    ]
    runner, _transport, _llm_caller, _tool_gate, _callback, _cancel_event = _make_runner(
        llm_responses=[
            LLMResponse(tool_calls=tool_calls),
            LLMResponse(text="done"),
        ],
        transport_returns=[
            {"content": [{"type": "text", "text": "ok"}]},
            {"content": [{"type": "text", "text": "search result"}]},
        ],
        run_first_metrics_context={
            "agent_type": "mcp",
            "presentation_variant": "acp_phase2a_v1",
            "cohort": "gated",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "eligible": True,
            "ineligible_reason": None,
        },
    )

    await runner.run([{"role": "user", "content": "hello"}])

    assert rollout_calls == [
        {
            "agent_type": "mcp",
            "presentation_variant": "acp_phase2a_v1",
            "cohort": "gated",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "eligible": True,
            "ineligible_reason": None,
        }
    ]
    assert first_tool_calls == [
        {
            "agent_type": "mcp",
            "presentation_variant": "acp_phase2a_v1",
            "cohort": "gated",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "eligible": True,
            "ineligible_reason": None,
            "first_tool": "run",
        }
    ]
    assert fallback_calls == [
        {
            "agent_type": "mcp",
            "presentation_variant": "acp_phase2a_v1",
            "cohort": "gated",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "eligible": True,
            "ineligible_reason": None,
            "fallback_tool": "search",
        }
    ]
    assert completion_calls == [
        {
            "agent_type": "mcp",
            "presentation_variant": "acp_phase2a_v1",
            "cohort": "gated",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "eligible": True,
            "ineligible_reason": None,
            "outcome": "end_turn",
        }
    ]


@pytest.mark.asyncio
async def test_llm_driven_records_max_iterations_completion_proxy(monkeypatch):
    """ACP runner should record max-iteration exits as a non-success completion outcome."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters import mcp_runners

    completion_calls: list[dict] = []
    monkeypatch.setattr(mcp_runners.acp_metrics, "record_run_first_rollout", lambda **_kwargs: None)
    monkeypatch.setattr(mcp_runners.acp_metrics, "record_run_first_first_tool", lambda **_kwargs: None)
    monkeypatch.setattr(
        mcp_runners.acp_metrics,
        "record_run_first_fallback_after_run",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        mcp_runners.acp_metrics,
        "record_run_first_completion_proxy",
        lambda **kwargs: completion_calls.append(kwargs),
    )

    tc = LLMToolCall(id="tc1", name="run", arguments={"command": "ls"})
    runner, _transport, _llm_caller, _tool_gate, _callback, _cancel_event = _make_runner(
        llm_responses=[LLMResponse(tool_calls=[tc])],
        max_iterations=1,
        run_first_metrics_context={
            "agent_type": "mcp",
            "presentation_variant": "acp_phase2a_v1",
            "cohort": "gated",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "eligible": True,
            "ineligible_reason": None,
        },
    )

    await runner.run([{"role": "user", "content": "hello"}])

    assert completion_calls == [
        {
            "agent_type": "mcp",
            "presentation_variant": "acp_phase2a_v1",
            "cohort": "gated",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "eligible": True,
            "ineligible_reason": None,
            "outcome": "max_iterations",
        }
    ]


@pytest.mark.asyncio
async def test_llm_driven_logs_expected_run_first_metric_failures(monkeypatch):
    """Expected metric backend failures should be logged without interrupting the runner."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters import mcp_runners

    warning_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        mcp_runners,
        "logger",
        type("LoggerStub", (), {"warning": lambda *args, **kwargs: warning_calls.append((args, kwargs))})(),
    )
    monkeypatch.setattr(
        mcp_runners.acp_metrics,
        "record_run_first_rollout",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("metrics down")),
    )

    runner, _transport, _llm_caller, _tool_gate, callback, _cancel_event = _make_runner(
        llm_responses=[LLMResponse(text="done")],
        run_first_metrics_context={
            "agent_type": "mcp",
            "presentation_variant": "acp_phase2b_v1",
            "cohort": "default_on",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "eligible": True,
            "ineligible_reason": None,
        },
    )

    await runner.run([{"role": "user", "content": "hello"}])

    assert warning_calls
    event: AgentEvent = callback.call_args_list[-1][0][0]
    assert event.kind == AgentEventKind.COMPLETION


@pytest.mark.asyncio
async def test_llm_driven_propagates_unexpected_run_first_metric_failures(monkeypatch):
    """Unexpected programming errors in metrics emission should not be silently swallowed."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters import mcp_runners

    warning_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        mcp_runners,
        "logger",
        type("LoggerStub", (), {"warning": lambda *args, **kwargs: warning_calls.append((args, kwargs))})(),
    )
    monkeypatch.setattr(
        mcp_runners.acp_metrics,
        "record_run_first_rollout",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
    )

    runner, _transport, _llm_caller, _tool_gate, _callback, _cancel_event = _make_runner(
        llm_responses=[LLMResponse(text="done")],
        run_first_metrics_context={
            "agent_type": "mcp",
            "presentation_variant": "acp_phase2b_v1",
            "cohort": "default_on",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "eligible": True,
            "ineligible_reason": None,
        },
    )

    with pytest.raises(AssertionError, match="unexpected"):
        await runner.run([{"role": "user", "content": "hello"}])

    assert warning_calls == []
