"""Tests for MCPAdapter — the main adapter wiring transport + runners + heartbeat + lifecycle."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, patch

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.base import (
    AdapterConfig,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.events import AgentEvent, AgentEventKind
from tldw_Server_API.app.core.exceptions import ValidationError
from tldw_Server_API.tests.run_first_constants import PHASE2C_RUN_FIRST_COHORT

pytestmark = pytest.mark.unit

SESSION_ID = "adapter-session-001"


@pytest.fixture
def mock_transport():
    t = AsyncMock()
    t.is_connected = True
    t.list_tools.return_value = [{"name": "search", "inputSchema": {"type": "object"}}]
    t.call_tool.return_value = {"content": [{"type": "text", "text": "result"}]}
    return t


@pytest.fixture
def collected_events():
    return []


@pytest.fixture
def event_callback(collected_events):
    async def cb(event: AgentEvent):
        collected_events.append(event)
    return cb


def _make_config(event_callback, protocol_config=None):
    return AdapterConfig(
        event_callback=event_callback,
        session_id=SESSION_ID,
        protocol_config=protocol_config or {"mcp_transport": "stdio", "command": "echo"},
    )


def _events_of_kind(events, kind: AgentEventKind):
    return [e for e in events if e.kind == kind]


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------


def test_mcp_adapter_protocol_name():
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter import MCPAdapter

    adapter = MCPAdapter()
    assert adapter.protocol_name == "mcp"


def test_mcp_adapter_not_connected_initially():
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter import MCPAdapter

    adapter = MCPAdapter()
    assert adapter.is_connected is False


def test_mcp_adapter_supports_streaming():
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter import MCPAdapter

    adapter = MCPAdapter()
    assert adapter.supports_streaming is True


# ---------------------------------------------------------------------------
# connect / disconnect lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_adapter_connect_lifecycle(mock_transport, event_callback, collected_events):
    """Connect should: create transport, connect, list_tools, emit agent_started + agent_ready."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter import MCPAdapter

    adapter = MCPAdapter()
    config = _make_config(event_callback)

    with patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.create_transport",
        return_value=mock_transport,
    ):
        await adapter.connect(config)

    # Transport interactions
    mock_transport.connect.assert_awaited_once()
    mock_transport.list_tools.assert_awaited_once()

    # Lifecycle events
    lifecycle_events = _events_of_kind(collected_events, AgentEventKind.LIFECYCLE)
    assert len(lifecycle_events) == 2
    assert lifecycle_events[0].payload["event"] == "agent_started"
    assert lifecycle_events[1].payload["event"] == "agent_ready"

    assert adapter.is_connected is True


@pytest.mark.asyncio
async def test_mcp_adapter_connect_rolls_back_state_when_list_tools_fails(
    mock_transport,
    event_callback,
):
    """Connect failures should close the transport and clear partial adapter state."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter import MCPAdapter

    adapter = MCPAdapter()
    config = _make_config(event_callback)
    mock_transport.list_tools.side_effect = RuntimeError("tools unavailable")

    with patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.create_transport",
        return_value=mock_transport,
    ):
        with pytest.raises(RuntimeError, match="tools unavailable"):
            await adapter.connect(config)

    mock_transport.close.assert_awaited_once()
    assert adapter.is_connected is False
    assert adapter._transport is None
    assert adapter._config is None
    assert adapter._tools == []


@pytest.mark.asyncio
async def test_mcp_adapter_disconnect(mock_transport, event_callback, collected_events):
    """Disconnect should: close transport, emit agent_exited, mark disconnected."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter import MCPAdapter

    adapter = MCPAdapter()
    config = _make_config(event_callback)

    with patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.create_transport",
        return_value=mock_transport,
    ):
        await adapter.connect(config)

    collected_events.clear()
    await adapter.disconnect()

    mock_transport.close.assert_awaited_once()

    lifecycle_events = _events_of_kind(collected_events, AgentEventKind.LIFECYCLE)
    assert len(lifecycle_events) == 1
    assert lifecycle_events[0].payload["event"] == "agent_exited"

    assert adapter.is_connected is False


# ---------------------------------------------------------------------------
# send_prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_adapter_send_prompt_agent_driven(mock_transport, event_callback, collected_events):
    """Agent-driven prompt should emit status_change events and use AgentDrivenRunner."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter import MCPAdapter

    adapter = MCPAdapter()
    config = _make_config(event_callback, protocol_config={
        "mcp_transport": "stdio",
        "command": "echo",
        "mcp_orchestration": "agent_driven",
    })

    with patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.create_transport",
        return_value=mock_transport,
    ):
        await adapter.connect(config)

    collected_events.clear()
    await adapter.send_prompt([{"role": "user", "content": "hello"}])

    # Transport should have been called (agent-driven uses call_tool)
    mock_transport.call_tool.assert_awaited_once()

    # Status changes: idle -> working, then working -> idle
    status_events = _events_of_kind(collected_events, AgentEventKind.STATUS_CHANGE)
    assert len(status_events) == 2
    assert status_events[0].payload == {"from_status": "idle", "to_status": "working"}
    assert status_events[1].payload == {"from_status": "working", "to_status": "idle"}

    # Should have a COMPLETION event from the runner
    completion_events = _events_of_kind(collected_events, AgentEventKind.COMPLETION)
    assert len(completion_events) == 1
    assert completion_events[0].payload["text"] == "result"


@pytest.mark.asyncio
async def test_mcp_adapter_send_prompt_llm_driven(mock_transport, event_callback, collected_events):
    """LLM-driven prompt should use LLMDrivenRunner with llm_caller and tool_gate."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter import MCPAdapter

    adapter = MCPAdapter()

    # Mock LLM caller that returns text completion (no tool calls)
    mock_llm_caller = AsyncMock()
    mock_llm_response = AsyncMock()
    mock_llm_response.text = "LLM final answer"
    mock_llm_response.tool_calls = []
    mock_llm_caller.call = AsyncMock(return_value=mock_llm_response)

    mock_tool_gate = AsyncMock()

    config = _make_config(event_callback, protocol_config={
        "mcp_transport": "stdio",
        "command": "echo",
        "mcp_orchestration": "llm_driven",
        "llm_caller": mock_llm_caller,
        "tool_gate": mock_tool_gate,
    })

    with patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.create_transport",
        return_value=mock_transport,
    ):
        await adapter.connect(config)

    collected_events.clear()
    await adapter.send_prompt([{"role": "user", "content": "hello"}])

    # LLM caller should have been invoked
    mock_llm_caller.call.assert_awaited()

    # Status changes
    status_events = _events_of_kind(collected_events, AgentEventKind.STATUS_CHANGE)
    assert len(status_events) == 2

    # COMPLETION from the LLM
    completion_events = _events_of_kind(collected_events, AgentEventKind.COMPLETION)
    assert len(completion_events) == 1
    assert completion_events[0].payload["text"] == "LLM final answer"


@pytest.mark.asyncio
async def test_mcp_adapter_send_prompt_llm_driven_applies_run_first_presentation(
    mock_transport,
    event_callback,
):
    """Eligible ACP sessions should pass run-first tools and prompt fragment to the LLM runner."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter import MCPAdapter

    adapter = MCPAdapter()

    mock_transport.list_tools.return_value = [
        {
            "name": "search",
            "description": "Search documents",
            "inputSchema": {"type": "object"},
        },
        {
            "name": "run",
            "description": "Execute governed commands",
            "inputSchema": {"type": "object"},
        },
    ]

    mock_llm_caller = AsyncMock()
    mock_llm_response = AsyncMock()
    mock_llm_response.text = "LLM final answer"
    mock_llm_response.tool_calls = []
    mock_llm_caller.call = AsyncMock(return_value=mock_llm_response)

    mock_tool_gate = AsyncMock()

    rollout_calls = []

    config = _make_config(event_callback, protocol_config={
        "mcp_transport": "stdio",
        "command": "echo",
        "mcp_orchestration": "llm_driven",
        "llm_caller": mock_llm_caller,
        "tool_gate": mock_tool_gate,
        "mcp_llm_provider": "openai",
        "mcp_llm_model": "gpt-4o-mini",
    })

    with patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.create_transport",
        return_value=mock_transport,
    ), patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.resolve_acp_run_first_rollout_mode",
        return_value="gated",
    ), patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.resolve_acp_run_first_provider_allowlist",
        return_value=["openai:gpt-4o-mini"],
    ), patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.resolve_acp_run_first_presentation_variant",
        return_value="acp_phase2a_v1",
    ), patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_runners.acp_metrics.record_run_first_rollout",
        side_effect=lambda **kwargs: rollout_calls.append(kwargs),
    ):
        await adapter.connect(config)
        await adapter.send_prompt([{"role": "user", "content": "hello"}])

    messages_arg, tools_arg = mock_llm_caller.call.await_args.args
    assert messages_arg[0]["role"] == "system"
    assert "ACP run-first guidance" in messages_arg[0]["content"]
    assert [tool["function"]["name"] for tool in tools_arg] == ["run", "search"]
    assert "Preferred first tool" in tools_arg[0]["function"]["description"]
    assert tools_arg[0]["function"]["parameters"]["type"] == "object"
    assert tools_arg[1]["function"]["description"].startswith("Fallback tool:")
    assert tools_arg[1]["function"]["parameters"]["type"] == "object"
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


@pytest.mark.asyncio
async def test_mcp_adapter_send_prompt_llm_driven_tracks_default_on_out_of_cohort(
    mock_transport,
    event_callback,
):
    """Default-on ACP sessions should keep fallback tools visible but emit out_of_cohort when filtered by allowlist."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter import MCPAdapter

    adapter = MCPAdapter()

    mock_transport.list_tools.return_value = [
        {
            "name": "search",
            "description": "Search documents",
            "inputSchema": {"type": "object"},
        },
        {
            "name": "run",
            "description": "Execute governed commands",
            "inputSchema": {"type": "object"},
        },
    ]

    mock_llm_caller = AsyncMock()
    mock_llm_response = AsyncMock()
    mock_llm_response.text = "LLM final answer"
    mock_llm_response.tool_calls = []
    mock_llm_caller.call = AsyncMock(return_value=mock_llm_response)

    mock_tool_gate = AsyncMock()
    rollout_calls = []

    config = _make_config(event_callback, protocol_config={
        "mcp_transport": "stdio",
        "command": "echo",
        "mcp_orchestration": "llm_driven",
        "llm_caller": mock_llm_caller,
        "tool_gate": mock_tool_gate,
        "mcp_llm_provider": "openai",
        "mcp_llm_model": "gpt-4o-mini",
    })

    with patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.create_transport",
        return_value=mock_transport,
    ), patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.resolve_acp_run_first_rollout_mode",
        return_value="default_on",
    ), patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.resolve_acp_run_first_provider_allowlist",
        return_value=["anthropic:claude-3-7-sonnet"],
    ), patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.resolve_acp_run_first_presentation_variant",
        return_value="acp_phase2b_v1",
    ), patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_runners.acp_metrics.record_run_first_rollout",
        side_effect=lambda **kwargs: rollout_calls.append(kwargs),
    ):
        await adapter.connect(config)
        await adapter.send_prompt([{"role": "user", "content": "hello"}])

    messages_arg, tools_arg = mock_llm_caller.call.await_args.args
    assert all(
        not (
            message.get("role") == "system"
            and "ACP run-first guidance" in str(message.get("content") or "")
        )
        for message in messages_arg
    )
    assert [tool["function"]["name"] for tool in tools_arg] == ["search", "run"]
    assert rollout_calls == [
        {
            "agent_type": "mcp",
            "presentation_variant": "acp_phase2b_v1",
            "cohort": "out_of_cohort",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "eligible": False,
            "ineligible_reason": "provider_not_in_rollout_allowlist",
        }
    ]
@pytest.mark.asyncio
async def test_mcp_adapter_send_prompt_llm_driven_applies_run_first_presentation_for_google_gemini(
    mock_transport: AsyncMock,
    event_callback: Callable[[AgentEvent], Awaitable[None]],
) -> None:
    """Default-on ACP sessions should present run-first tools for widened Google/Gemini cohort members."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter import MCPAdapter

    adapter = MCPAdapter()

    mock_transport.list_tools.return_value = [
        {
            "name": "search",
            "description": "Search documents",
            "inputSchema": {"type": "object"},
        },
        {
            "name": "run",
            "description": "Execute governed commands",
            "inputSchema": {"type": "object"},
        },
    ]

    mock_llm_caller = AsyncMock()
    mock_llm_response = AsyncMock()
    mock_llm_response.text = "LLM final answer"
    mock_llm_response.tool_calls = []
    mock_llm_caller.call = AsyncMock(return_value=mock_llm_response)

    mock_tool_gate = AsyncMock()
    rollout_calls = []

    config = _make_config(event_callback, protocol_config={
        "mcp_transport": "stdio",
        "command": "echo",
        "mcp_orchestration": "llm_driven",
        "llm_caller": mock_llm_caller,
        "tool_gate": mock_tool_gate,
        "mcp_llm_provider": "google",
        "mcp_llm_model": "gemini-2.5-flash",
    })

    with patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.create_transport",
        return_value=mock_transport,
    ), patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.resolve_acp_run_first_rollout_mode",
        return_value="default_on",
    ), patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.resolve_acp_run_first_provider_allowlist",
        return_value=PHASE2C_RUN_FIRST_COHORT,
    ), patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.resolve_acp_run_first_presentation_variant",
        return_value="acp_phase2b_v1",
    ), patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_runners.acp_metrics.record_run_first_rollout",
        side_effect=lambda **kwargs: rollout_calls.append(kwargs),
    ):
        await adapter.connect(config)
        await adapter.send_prompt([{"role": "user", "content": "hello"}])

    messages_arg, tools_arg = mock_llm_caller.call.await_args.args
    assert messages_arg[0]["role"] == "system"
    assert "ACP run-first guidance" in messages_arg[0]["content"]
    assert [tool["function"]["name"] for tool in tools_arg] == ["run", "search"]
    assert "Preferred first tool" in tools_arg[0]["function"]["description"]
    assert tools_arg[0]["function"]["parameters"]["type"] == "object"
    assert tools_arg[1]["function"]["description"].startswith("Fallback tool:")
    assert tools_arg[1]["function"]["parameters"]["type"] == "object"
    assert rollout_calls == [
        {
            "agent_type": "mcp",
            "presentation_variant": "acp_phase2b_v1",
            "cohort": "default_on",
            "provider": "google",
            "model": "gemini-2.5-flash",
            "eligible": True,
            "ineligible_reason": None,
        }
    ]
async def test_mcp_adapter_send_prompt_llm_driven_requires_llm_caller_and_tool_gate(
    mock_transport,
    event_callback,
):
    """LLM-driven mode should fail with a clear error when required config is missing."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter import MCPAdapter

    adapter = MCPAdapter()
    config = _make_config(event_callback, protocol_config={
        "mcp_transport": "stdio",
        "command": "echo",
        "mcp_orchestration": "llm_driven",
    })

    with patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.create_transport",
        return_value=mock_transport,
    ):
        await adapter.connect(config)

    with pytest.raises(ValueError, match=r"llm_driven.*llm_caller.*tool_gate"):
        await adapter.send_prompt([{"role": "user", "content": "hello"}])


@pytest.mark.asyncio
async def test_mcp_adapter_send_prompt_rejects_unknown_orchestration(
    mock_transport,
    event_callback,
):
    """Unknown orchestration modes should fail fast instead of silently falling back."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter import MCPAdapter

    adapter = MCPAdapter()
    config = _make_config(event_callback, protocol_config={
        "mcp_transport": "stdio",
        "command": "echo",
        "mcp_orchestration": "surprise_mode",
    })

    with patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.create_transport",
        return_value=mock_transport,
    ):
        await adapter.connect(config)

    with pytest.raises(ValidationError, match=r"mcp_orchestration.*surprise_mode"):
        await adapter.send_prompt([{"role": "user", "content": "hello"}])


@pytest.mark.asyncio
async def test_mcp_adapter_send_prompt_not_connected():
    """send_prompt when not connected should raise RuntimeError."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter import MCPAdapter

    adapter = MCPAdapter()
    with pytest.raises(RuntimeError, match="Not connected"):
        await adapter.send_prompt([{"role": "user", "content": "hello"}])


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_adapter_cancel(mock_transport, event_callback):
    """cancel() should set the internal cancel event."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter import MCPAdapter

    adapter = MCPAdapter()
    config = _make_config(event_callback)

    with patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.create_transport",
        return_value=mock_transport,
    ):
        await adapter.connect(config)

    await adapter.cancel()
    # The cancel event should be set (verified indirectly: runner would respect it)
    assert adapter._cancel_event.is_set()


# ---------------------------------------------------------------------------
# is_connected delegation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_adapter_is_connected_delegates(mock_transport, event_callback):
    """is_connected should delegate to transport.is_connected."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter import MCPAdapter

    adapter = MCPAdapter()
    config = _make_config(event_callback)

    with patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.create_transport",
        return_value=mock_transport,
    ):
        await adapter.connect(config)

    assert adapter.is_connected is True

    # Simulate transport disconnect
    mock_transport.is_connected = False
    assert adapter.is_connected is False


# ---------------------------------------------------------------------------
# tool refresh
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_adapter_tool_refresh(mock_transport, event_callback):
    """When mcp_refresh_tools=True, list_tools should be called again on send_prompt."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter import MCPAdapter

    adapter = MCPAdapter()
    config = _make_config(event_callback, protocol_config={
        "mcp_transport": "stdio",
        "command": "echo",
        "mcp_refresh_tools": True,
    })

    with patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_adapter.create_transport",
        return_value=mock_transport,
    ):
        await adapter.connect(config)

    # list_tools called once during connect
    assert mock_transport.list_tools.await_count == 1

    await adapter.send_prompt([{"role": "user", "content": "go"}])

    # list_tools called again during send_prompt
    assert mock_transport.list_tools.await_count == 2
