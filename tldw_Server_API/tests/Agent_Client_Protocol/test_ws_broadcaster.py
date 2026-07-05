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

pytestmark = pytest.mark.unit


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


async def test_ws_broadcaster_allows_unique_consumer_ids() -> None:
    """Distinct broadcasters should not overwrite each other's bus subscription."""
    bus = SessionEventBus(session_id="sess-unique-consumers")
    first = WSBroadcaster(consumer_id="ws_broadcaster:conn-1")
    second = WSBroadcaster(consumer_id="ws_broadcaster:conn-2")

    try:
        await first.start(bus)
        await second.start(bus)

        assert set(bus._subscribers) == {
            "ws_broadcaster:conn-1",
            "ws_broadcaster:conn-2",
        }
    finally:
        await first.stop()
        await second.stop()
    assert bus._subscribers == {}


async def test_start_reconnect_replay_cleans_up_when_registration_fails() -> None:
    """A failing replay send must not leak the bus subscription or consume task."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.consumers.ws_broadcaster import (
        start_reconnect_replay,
    )

    session_id = "sess-replay-fail"
    bus = SessionEventBus(session_id=session_id)
    await bus.publish(_make_event(AgentEventKind.COMPLETION, session_id=session_id))

    async def failing_send(msg: str) -> None:
        raise RuntimeError("send failed during replay")

    with pytest.raises(RuntimeError, match="send failed during replay"):
        await start_reconnect_replay(
            bus,
            conn_id="conn-replay-fail",
            send_callback=failing_send,
            from_sequence=1,
        )

    # the broadcaster must have been stopped: no leaked subscriber
    assert bus._subscribers == {}


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
