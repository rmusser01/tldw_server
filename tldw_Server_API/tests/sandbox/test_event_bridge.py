"""Tests for SandboxEventBridge — translates RunStreamHub frames to AgentEvents."""
from __future__ import annotations

import asyncio

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from tldw_Server_API.app.core.Agent_Client_Protocol.events import AgentEvent, AgentEventKind
from tldw_Server_API.app.core.Agent_Client_Protocol.event_bus import SessionEventBus
from tldw_Server_API.app.core.Sandbox.event_bridge import SandboxEventBridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hub_mock(frames: list[dict]) -> MagicMock:
    """Create a mock RunStreamHub that yields *frames* then blocks forever."""
    q: asyncio.Queue[dict] = asyncio.Queue()
    for f in frames:
        q.put_nowait(f)

    hub = MagicMock()
    hub.subscribe.return_value = q
    hub.unsubscribe = MagicMock()
    return hub


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stdout_frame_becomes_terminal_output():
    """A stdout frame from RunStreamHub becomes a TERMINAL_OUTPUT AgentEvent."""
    hub = _make_hub_mock([
        {"type": "stdout", "encoding": "utf8", "data": "hello world", "seq": 1},
    ])
    bus = SessionEventBus(session_id="sess-1")
    bridge = SandboxEventBridge(hub=hub, bus=bus, run_id="run-1", session_id="sess-1")

    await bridge.start()
    # Give the consume loop time to process the single frame
    await asyncio.sleep(0.15)
    await bridge.stop()

    events = bus.snapshot()
    assert len(events) == 1
    ev = events[0]
    assert ev.kind == AgentEventKind.TERMINAL_OUTPUT
    assert ev.payload["stream"] == "stdout"
    assert ev.payload["data"] == "hello world"
    assert ev.session_id == "sess-1"
    assert ev.metadata["run_id"] == "run-1"


@pytest.mark.asyncio
async def test_stderr_frame_becomes_terminal_output():
    """stderr also maps to TERMINAL_OUTPUT with stream='stderr'."""
    hub = _make_hub_mock([
        {"type": "stderr", "encoding": "utf8", "data": "oops", "seq": 2},
    ])
    bus = SessionEventBus(session_id="sess-2")
    bridge = SandboxEventBridge(hub=hub, bus=bus, run_id="run-2", session_id="sess-2")

    await bridge.start()
    await asyncio.sleep(0.15)
    await bridge.stop()

    events = bus.snapshot()
    assert len(events) == 1
    assert events[0].kind == AgentEventKind.TERMINAL_OUTPUT
    assert events[0].payload["stream"] == "stderr"
    assert events[0].payload["data"] == "oops"


@pytest.mark.asyncio
async def test_exit_event_frame_becomes_completion():
    """An event frame with event='end' becomes COMPLETION."""
    hub = _make_hub_mock([
        {"type": "event", "event": "end", "data": {"exit_code": 0}, "seq": 3},
    ])
    bus = SessionEventBus(session_id="sess-3")
    bridge = SandboxEventBridge(hub=hub, bus=bus, run_id="run-3", session_id="sess-3")

    await bridge.start()
    await asyncio.sleep(0.15)
    await bridge.stop()

    events = bus.snapshot()
    assert len(events) == 1
    assert events[0].kind == AgentEventKind.COMPLETION
    assert events[0].payload == {"exit_code": 0}


@pytest.mark.asyncio
async def test_truncated_frame_becomes_error():
    """A truncated frame becomes an ERROR event."""
    hub = _make_hub_mock([
        {"type": "truncated", "reason": "log_cap", "seq": 4},
    ])
    bus = SessionEventBus(session_id="sess-4")
    bridge = SandboxEventBridge(hub=hub, bus=bus, run_id="run-4", session_id="sess-4")

    await bridge.start()
    await asyncio.sleep(0.15)
    await bridge.stop()

    events = bus.snapshot()
    assert len(events) == 1
    assert events[0].kind == AgentEventKind.ERROR
    assert events[0].payload["error"] == "truncated"
    assert events[0].payload["reason"] == "log_cap"


@pytest.mark.asyncio
async def test_heartbeat_frame_becomes_heartbeat():
    """Heartbeat frames are forwarded as HEARTBEAT events."""
    hub = _make_hub_mock([
        {"type": "heartbeat"},
    ])
    bus = SessionEventBus(session_id="sess-5")
    bridge = SandboxEventBridge(hub=hub, bus=bus, run_id="run-5", session_id="sess-5")

    await bridge.start()
    await asyncio.sleep(0.15)
    await bridge.stop()

    events = bus.snapshot()
    assert len(events) == 1
    assert events[0].kind == AgentEventKind.HEARTBEAT


@pytest.mark.asyncio
async def test_generic_event_becomes_lifecycle():
    """An event frame with a non-'end' event name becomes LIFECYCLE."""
    hub = _make_hub_mock([
        {"type": "event", "event": "start", "data": {"pid": 42}, "seq": 5},
    ])
    bus = SessionEventBus(session_id="sess-6")
    bridge = SandboxEventBridge(hub=hub, bus=bus, run_id="run-6", session_id="sess-6")

    await bridge.start()
    await asyncio.sleep(0.15)
    await bridge.stop()

    events = bus.snapshot()
    assert len(events) == 1
    assert events[0].kind == AgentEventKind.LIFECYCLE
    assert events[0].payload["event"] == "start"
    assert events[0].payload["pid"] == 42


@pytest.mark.asyncio
async def test_unknown_frame_type_is_skipped():
    """Unknown frame types are logged and skipped — no event published."""
    hub = _make_hub_mock([
        {"type": "mystery", "data": "???", "seq": 99},
    ])
    bus = SessionEventBus(session_id="sess-7")
    bridge = SandboxEventBridge(hub=hub, bus=bus, run_id="run-7", session_id="sess-7")

    await bridge.start()
    await asyncio.sleep(0.15)
    await bridge.stop()

    events = bus.snapshot()
    assert len(events) == 0


@pytest.mark.asyncio
async def test_bridge_stops_cleanly():
    """Calling stop() unsubscribes from the hub and cancels the task."""
    hub = _make_hub_mock([])
    bus = SessionEventBus(session_id="sess-8")
    bridge = SandboxEventBridge(hub=hub, bus=bus, run_id="run-8", session_id="sess-8")

    await bridge.start()
    # The consume task should be running
    assert bridge._task is not None
    assert not bridge._task.done()

    await bridge.stop()

    # Task is cleaned up
    assert bridge._task is None
    # Hub.unsubscribe was called with the correct run_id and queue
    hub.unsubscribe.assert_called_once()
    call_args = hub.unsubscribe.call_args
    assert call_args[0][0] == "run-8"


@pytest.mark.asyncio
async def test_multiple_frames_in_sequence():
    """Multiple frames are processed in order and produce distinct events."""
    hub = _make_hub_mock([
        {"type": "stdout", "encoding": "utf8", "data": "line1", "seq": 1},
        {"type": "stderr", "encoding": "utf8", "data": "err1", "seq": 2},
        {"type": "event", "event": "end", "data": {}, "seq": 3},
    ])
    bus = SessionEventBus(session_id="sess-9")
    bridge = SandboxEventBridge(hub=hub, bus=bus, run_id="run-9", session_id="sess-9")

    await bridge.start()
    await asyncio.sleep(0.25)
    await bridge.stop()

    events = bus.snapshot()
    assert len(events) == 3
    assert events[0].kind == AgentEventKind.TERMINAL_OUTPUT
    assert events[0].payload["stream"] == "stdout"
    assert events[1].kind == AgentEventKind.TERMINAL_OUTPUT
    assert events[1].payload["stream"] == "stderr"
    assert events[2].kind == AgentEventKind.COMPLETION


@pytest.mark.asyncio
async def test_start_is_idempotent():
    """Calling start() twice does not create a second consume task."""
    hub = _make_hub_mock([])
    bus = SessionEventBus(session_id="sess-10")
    bridge = SandboxEventBridge(hub=hub, bus=bus, run_id="run-10", session_id="sess-10")

    await bridge.start()
    first_task = bridge._task
    await bridge.start()  # Should be a no-op
    assert bridge._task is first_task
    hub.subscribe.assert_called_once()

    await bridge.stop()
