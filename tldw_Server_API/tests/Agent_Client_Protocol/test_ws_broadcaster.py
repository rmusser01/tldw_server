"""Tests for WSBroadcaster event consumer."""
from __future__ import annotations

import asyncio
import json

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.event_bus import SessionEventBus
from tldw_Server_API.app.core.Agent_Client_Protocol.events import (
    AgentEvent,
    AgentEventKind,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.consumers.ws_broadcaster import (
    WSBroadcaster,
)


def _make_event(kind: AgentEventKind, session_id: str = "sess-1") -> AgentEvent:
    return AgentEvent(session_id=session_id, kind=kind, payload={"data": kind.value})


async def _wait_for(predicate, *, timeout: float = 2.0, interval: float = 0.02):
    """Poll *predicate* until it returns True or *timeout* elapses."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"Timed out waiting for predicate after {timeout}s")


@pytest.mark.asyncio
async def test_ws_broadcaster_delivers_events_full_verbosity():
    """All events should be delivered when verbosity is 'full'."""
    bus = SessionEventBus(session_id="sess-1")
    broadcaster = WSBroadcaster()

    received: list[dict] = []

    async def fake_send(msg: str) -> None:
        received.append(json.loads(msg))

    await broadcaster.start(bus)
    await broadcaster.add_connection("conn-1", fake_send, verbosity="full")

    for kind in (AgentEventKind.THINKING, AgentEventKind.TOOL_CALL, AgentEventKind.COMPLETION):
        await bus.publish(_make_event(kind))

    await _wait_for(lambda: len(received) >= 3)
    await broadcaster.stop()

    assert len(received) == 3
    assert received[0]["kind"] == "thinking"
    assert received[1]["kind"] == "tool_call"
    assert received[2]["kind"] == "completion"


@pytest.mark.asyncio
async def test_ws_broadcaster_summary_filters_thinking():
    """Summary verbosity should drop thinking/tool_call/etc., keep completion."""
    bus = SessionEventBus(session_id="sess-1")
    broadcaster = WSBroadcaster()

    received: list[dict] = []

    async def fake_send(msg: str) -> None:
        received.append(json.loads(msg))

    await broadcaster.start(bus)
    await broadcaster.add_connection("conn-1", fake_send, verbosity="summary")

    await bus.publish(_make_event(AgentEventKind.THINKING))
    await bus.publish(_make_event(AgentEventKind.TOOL_CALL))
    await bus.publish(_make_event(AgentEventKind.COMPLETION))
    await bus.publish(_make_event(AgentEventKind.ERROR))

    # Wait for the 2 events that should pass (completion + error)
    await _wait_for(lambda: len(received) >= 2)
    await broadcaster.stop()

    kinds = [e["kind"] for e in received]
    assert "thinking" not in kinds
    assert "tool_call" not in kinds
    assert "completion" in kinds
    assert "error" in kinds


@pytest.mark.asyncio
async def test_ws_broadcaster_remove_connection():
    """After removing a connection, it should no longer receive events."""
    bus = SessionEventBus(session_id="sess-1")
    broadcaster = WSBroadcaster()

    received: list[dict] = []

    async def fake_send(msg: str) -> None:
        received.append(json.loads(msg))

    await broadcaster.start(bus)
    await broadcaster.add_connection("conn-1", fake_send, verbosity="full")

    await bus.publish(_make_event(AgentEventKind.THINKING))
    await _wait_for(lambda: len(received) >= 1)

    broadcaster.remove_connection("conn-1")

    await bus.publish(_make_event(AgentEventKind.COMPLETION))
    # Give a brief window to confirm no more events arrive
    await asyncio.sleep(0.05)
    await broadcaster.stop()

    assert len(received) == 1
    assert received[0]["kind"] == "thinking"


@pytest.mark.asyncio
async def test_ws_broadcaster_change_verbosity():
    """Changing verbosity mid-stream should affect subsequent filtering."""
    bus = SessionEventBus(session_id="sess-1")
    broadcaster = WSBroadcaster()

    received: list[dict] = []

    async def fake_send(msg: str) -> None:
        received.append(json.loads(msg))

    await broadcaster.start(bus)
    await broadcaster.add_connection("conn-1", fake_send, verbosity="summary")

    # Thinking should be filtered at summary level
    await bus.publish(_make_event(AgentEventKind.THINKING))
    # Publish a summary-visible event to confirm the bus is processing
    await bus.publish(_make_event(AgentEventKind.COMPLETION))
    await _wait_for(lambda: len(received) >= 1)

    # Switch to full verbosity
    broadcaster.set_verbosity("conn-1", "full")

    await bus.publish(_make_event(AgentEventKind.THINKING))
    await _wait_for(lambda: len(received) >= 2)
    await broadcaster.stop()

    # First received should be completion (thinking was filtered at summary)
    # Second should be thinking (after switching to full)
    assert received[0]["kind"] == "completion"
    assert received[1]["kind"] == "thinking"
