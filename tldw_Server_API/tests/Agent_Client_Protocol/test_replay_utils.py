import pytest
import asyncio
from unittest.mock import MagicMock
from tldw_Server_API.app.core.Agent_Client_Protocol.consumers.replay_utils import replay_events


@pytest.mark.asyncio
async def test_replay_sends_buffered_events_in_order():
    """replay_events calls emit_fn for each buffered event from sequence."""
    bus = MagicMock()
    # Create mock events with sequence numbers
    events = []
    for i in range(5):
        ev = MagicMock()
        ev.sequence = i + 1
        events.append(ev)
    bus.snapshot.return_value = events[2:]  # from_sequence=3 returns events 3,4,5

    sent = []
    async def emit(ev):
        sent.append(ev)

    count = await replay_events(bus, from_sequence=3, emit_fn=emit)
    assert count == 3
    assert len(sent) == 3
    assert [e.sequence for e in sent] == [3, 4, 5]
    bus.snapshot.assert_called_once_with(from_sequence=3)


@pytest.mark.asyncio
async def test_replay_zero_sequence_returns_zero():
    """from_sequence=0 means no replay needed."""
    bus = MagicMock()
    sent = []
    async def emit(ev):
        sent.append(ev)

    count = await replay_events(bus, from_sequence=0, emit_fn=emit)
    assert count == 0
    assert len(sent) == 0
    bus.snapshot.assert_not_called()


@pytest.mark.asyncio
async def test_replay_negative_sequence_returns_zero():
    """Negative sequence means no replay."""
    bus = MagicMock()
    count = await replay_events(bus, from_sequence=-1, emit_fn=lambda ev: asyncio.sleep(0))
    assert count == 0


@pytest.mark.asyncio
async def test_replay_empty_buffer():
    """If bus has no events from that sequence, emit nothing."""
    bus = MagicMock()
    bus.snapshot.return_value = []
    sent = []
    async def emit(ev):
        sent.append(ev)

    count = await replay_events(bus, from_sequence=100, emit_fn=emit)
    assert count == 0
    assert len(sent) == 0
