"""Tests for SSEConsumer event consumer."""
from __future__ import annotations

import asyncio
import json

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.event_bus import SessionEventBus
from tldw_Server_API.app.core.Agent_Client_Protocol.events import (
    AgentEvent,
    AgentEventKind,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.consumers.sse_consumer import (
    SSEConsumer,
)


def _make_event(
    kind: AgentEventKind = AgentEventKind.TOOL_CALL,
    session_id: str = "sess-1",
) -> AgentEvent:
    return AgentEvent(session_id=session_id, kind=kind, payload={"data": kind.value})


async def _collect_lines(
    gen,
    *,
    max_lines: int = 10,
    timeout: float = 2.0,
) -> list[str]:
    """Collect up to *max_lines* from an async generator within *timeout*."""
    lines: list[str] = []
    try:
        async with asyncio.timeout(timeout):
            async for line in gen:
                lines.append(line)
                if len(lines) >= max_lines:
                    break
    except (asyncio.TimeoutError, TimeoutError):
        pass
    return lines


# --------------------------------------------------------------------------- #
# SSE format
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sse_consumer_formats_events_as_sse():
    """Events are formatted as 'event: {kind}\\ndata: {json}\\n\\n'."""
    bus = SessionEventBus(session_id="sess-1")
    consumer = SSEConsumer(from_sequence=0, heartbeat_interval=10.0)

    await consumer.start(bus)

    # Publish two events through the bus
    await bus.publish(_make_event(AgentEventKind.TOOL_CALL))
    await bus.publish(_make_event(AgentEventKind.COMPLETION))

    lines = await _collect_lines(consumer.iter_sse_lines(), max_lines=2)
    await consumer.stop()

    assert len(lines) == 2

    # First event
    assert lines[0].startswith("event: tool_call\n")
    assert "data: " in lines[0]
    assert lines[0].endswith("\n\n")
    data_0 = json.loads(lines[0].split("data: ", 1)[1].rstrip("\n"))
    assert data_0["kind"] == "tool_call"
    assert data_0["session_id"] == "sess-1"
    assert data_0["sequence"] == 1

    # Second event
    assert lines[1].startswith("event: completion\n")
    data_1 = json.loads(lines[1].split("data: ", 1)[1].rstrip("\n"))
    assert data_1["kind"] == "completion"
    assert data_1["sequence"] == 2


# --------------------------------------------------------------------------- #
# Heartbeat
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sse_consumer_heartbeat_on_timeout():
    """After heartbeat_interval of no events, emits ': heartbeat\\n\\n'."""
    bus = SessionEventBus(session_id="sess-1")
    consumer = SSEConsumer(heartbeat_interval=0.05)  # 50ms for fast test

    await consumer.start(bus)

    # Don't publish any events -- should get heartbeat(s)
    lines = await _collect_lines(consumer.iter_sse_lines(), max_lines=2, timeout=1.0)
    await consumer.stop()

    assert len(lines) >= 1
    assert lines[0] == ": heartbeat\n\n"


# --------------------------------------------------------------------------- #
# Replay from sequence
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sse_consumer_replays_from_sequence():
    """With from_sequence > 0, replays buffered events on start."""
    bus = SessionEventBus(session_id="sess-1")

    # Pre-populate the bus with 5 events
    for i in range(5):
        await bus.publish(_make_event(AgentEventKind.THINKING))

    # Create consumer that wants events from sequence 3 onwards
    consumer = SSEConsumer(from_sequence=3, heartbeat_interval=10.0)
    await consumer.start(bus)

    # The bus.subscribe(from_sequence=3) replays seq 3, 4, 5 into the queue
    lines = await _collect_lines(consumer.iter_sse_lines(), max_lines=3, timeout=1.0)
    await consumer.stop()

    assert len(lines) == 3
    sequences = []
    for line in lines:
        data = json.loads(line.split("data: ", 1)[1].rstrip("\n"))
        sequences.append(data["sequence"])
    assert sequences == [3, 4, 5]


# --------------------------------------------------------------------------- #
# Stop / unsubscribe
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sse_consumer_stop_unsubscribes():
    """Calling stop() unsubscribes from the bus."""
    bus = SessionEventBus(session_id="sess-1")
    consumer = SSEConsumer(heartbeat_interval=10.0)

    await consumer.start(bus)
    assert consumer.consumer_id in bus._subscribers

    await consumer.stop()
    assert consumer.consumer_id not in bus._subscribers


# --------------------------------------------------------------------------- #
# Unique consumer IDs
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sse_consumer_default_id_is_unique():
    """Two SSEConsumers created without explicit IDs have different consumer_ids."""
    c1 = SSEConsumer()
    c2 = SSEConsumer()
    assert c1.consumer_id != c2.consumer_id
    assert c1.consumer_id.startswith("sse_")
    assert c2.consumer_id.startswith("sse_")


# --------------------------------------------------------------------------- #
# Generator terminates on stop
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sse_consumer_iter_terminates_after_stop():
    """iter_sse_lines() should exit once stop() is called."""
    bus = SessionEventBus(session_id="sess-1")
    consumer = SSEConsumer(heartbeat_interval=0.05)

    await consumer.start(bus)

    collected: list[str] = []

    async def drain():
        async for line in consumer.iter_sse_lines():
            collected.append(line)

    task = asyncio.create_task(drain())

    # Let it run briefly to get at least one heartbeat
    await asyncio.sleep(0.15)
    await consumer.stop()

    # Give the task a moment to finish
    try:
        await asyncio.wait_for(task, timeout=1.0)
    except asyncio.TimeoutError:
        task.cancel()
        pytest.fail("iter_sse_lines did not terminate after stop()")

    # Should have collected at least one heartbeat
    assert len(collected) >= 1
