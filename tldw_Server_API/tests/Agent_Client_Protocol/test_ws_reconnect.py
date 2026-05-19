"""Tests for WebSocket reconnect with catch-up replay."""
from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock, patch

from tldw_Server_API.app.core.Agent_Client_Protocol.event_bus import SessionEventBus
from tldw_Server_API.app.core.Agent_Client_Protocol.events import (
    AgentEvent,
    AgentEventKind,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.consumers.ws_broadcaster import (
    WSBroadcaster,
)


def _make_event(
    kind: AgentEventKind,
    session_id: str = "sess-1",
    sequence: int = 0,
) -> AgentEvent:
    return AgentEvent(
        session_id=session_id,
        kind=kind,
        payload={"data": kind.value},
        sequence=sequence,
    )


@pytest.mark.asyncio
async def test_add_connection_with_replay():
    """Connecting with from_sequence > 0 replays buffered events."""
    bus = SessionEventBus(session_id="sess-1")
    broadcaster = WSBroadcaster()
    await broadcaster.start(bus)

    received: list[str] = []

    async def fake_send(msg: str) -> None:
        received.append(msg)

    # Seed a few buffered events in the bus
    ev1 = _make_event(AgentEventKind.THINKING, sequence=1)
    ev2 = _make_event(AgentEventKind.TOOL_CALL, sequence=2)
    ev3 = _make_event(AgentEventKind.COMPLETION, sequence=3)
    for ev in (ev1, ev2, ev3):
        bus._buffer.append(ev)

    with patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.consumers.ws_broadcaster.replay_events",
    ) as mock_replay:
        mock_replay.return_value = 2  # pretend 2 events replayed

        await broadcaster.add_connection(
            "conn-1", fake_send, verbosity="full", from_sequence=5,
        )

        # replay_events should have been called with the bus, sequence, and an emit fn
        mock_replay.assert_awaited_once()
        call_args = mock_replay.call_args
        assert call_args[0][0] is bus
        assert call_args[0][1] == 5
        # The third arg should be a callable (the _emit function)
        assert callable(call_args[0][2])

    # Connection should be registered
    assert "conn-1" in broadcaster._connections
    await broadcaster.stop()


@pytest.mark.asyncio
async def test_add_connection_without_replay():
    """Connecting without from_sequence does no replay."""
    bus = SessionEventBus(session_id="sess-1")
    broadcaster = WSBroadcaster()
    await broadcaster.start(bus)

    async def fake_send(msg: str) -> None:
        pass

    with patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.consumers.ws_broadcaster.replay_events",
    ) as mock_replay:
        await broadcaster.add_connection("conn-2", fake_send, verbosity="full")

        # replay_events should NOT have been called
        mock_replay.assert_not_awaited()

    assert "conn-2" in broadcaster._connections
    await broadcaster.stop()


@pytest.mark.asyncio
async def test_add_connection_replay_without_bus():
    """If the broadcaster has no bus, replay is skipped even with from_sequence."""
    broadcaster = WSBroadcaster()
    # Don't call start() -- _bus remains None

    async def fake_send(msg: str) -> None:
        pass

    with patch(
        "tldw_Server_API.app.core.Agent_Client_Protocol.consumers.ws_broadcaster.replay_events",
    ) as mock_replay:
        await broadcaster.add_connection(
            "conn-3", fake_send, verbosity="full", from_sequence=10,
        )
        mock_replay.assert_not_awaited()

    assert "conn-3" in broadcaster._connections


@pytest.mark.asyncio
async def test_replay_emit_serializes_with_verbosity():
    """The replay emit function should serialize events with verbosity filtering."""
    bus = SessionEventBus(session_id="sess-1")
    broadcaster = WSBroadcaster()
    await broadcaster.start(bus)

    received: list[str] = []

    async def fake_send(msg: str) -> None:
        received.append(msg)

    # Seed events in the bus buffer
    ev1 = _make_event(AgentEventKind.THINKING, sequence=1)
    ev2 = _make_event(AgentEventKind.COMPLETION, sequence=2)
    bus._buffer.append(ev1)
    bus._buffer.append(ev2)

    # Connect with summary verbosity -- thinking should be filtered out
    await broadcaster.add_connection(
        "conn-summary", fake_send, verbosity="summary", from_sequence=1,
    )

    # Only the COMPLETION event should have been sent (THINKING is filtered by summary)
    assert len(received) == 1
    parsed = json.loads(received[0])
    assert parsed["kind"] == "completion"

    await broadcaster.stop()
