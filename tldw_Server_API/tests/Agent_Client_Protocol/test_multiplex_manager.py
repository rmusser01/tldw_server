"""Tests for MultiplexManager."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.event_bus import SessionEventBus
from tldw_Server_API.app.core.Agent_Client_Protocol.events import AgentEvent, AgentEventKind
from tldw_Server_API.app.core.Agent_Client_Protocol.multiplex.manager import MultiplexManager
from tldw_Server_API.app.core.Agent_Client_Protocol.multiplex.protocol import (
    MultiplexMessage,
    MultiplexMessageType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeSender:
    """Collects messages sent by the manager."""

    def __init__(self) -> None:
        self.messages: list[MultiplexMessage] = []
        self._raw: list[str] = []

    async def __call__(self, raw: str) -> None:
        self._raw.append(raw)
        self.messages.append(MultiplexMessage.from_json(raw))

    def last(self) -> MultiplexMessage:
        return self.messages[-1]

    def clear(self) -> None:
        self.messages.clear()
        self._raw.clear()


class FailingSender:
    """Raises on send to simulate a broken WebSocket transport."""

    async def __call__(self, raw: str) -> None:
        raise RuntimeError("send failed")


def _make_buses(*session_ids: str) -> dict[str, SessionEventBus]:
    return {sid: SessionEventBus(sid) for sid in session_ids}


def _get_bus_fn(buses: dict[str, SessionEventBus]):
    """Return a lookup function matching the ``get_bus_fn`` signature."""
    def fn(session_id: str) -> Optional[SessionEventBus]:
        return buses.get(session_id)
    return fn


def _make_manager(
    sender: FakeSender,
    buses: dict[str, SessionEventBus],
    ping_interval: float = 0,
) -> MultiplexManager:
    return MultiplexManager(
        connection_id="test",
        send_fn=sender,
        get_bus_fn=_get_bus_fn(buses),
        ping_interval=ping_interval,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOpenStream:
    @pytest.mark.asyncio
    async def test_open_known_session(self):
        sender = FakeSender()
        buses = _make_buses("sess-1")
        mgr = _make_manager(sender, buses)

        msg = MultiplexMessage.stream_open("sess-1").to_json()
        await mgr.handle_message(msg)

        assert "sess-1" in mgr.active_streams
        # No error sent
        assert len(sender.messages) == 0
        await mgr.stop()

    @pytest.mark.asyncio
    async def test_open_unknown_session_sends_error(self):
        sender = FakeSender()
        mgr = _make_manager(sender, {})

        msg = MultiplexMessage.stream_open("nonexistent").to_json()
        await mgr.handle_message(msg)

        assert len(sender.messages) == 1
        assert sender.last().type == MultiplexMessageType.ERROR
        assert "Unknown session" in sender.last().payload["error"]
        assert sender.last().stream_id == "nonexistent"
        assert mgr.active_streams == []
        await mgr.stop()

    @pytest.mark.asyncio
    async def test_open_idempotent(self):
        sender = FakeSender()
        buses = _make_buses("sess-1")
        mgr = _make_manager(sender, buses)

        msg = MultiplexMessage.stream_open("sess-1").to_json()
        await mgr.handle_message(msg)
        await mgr.handle_message(msg)

        assert mgr.active_streams == ["sess-1"]
        assert len(sender.messages) == 0
        await mgr.stop()

    @pytest.mark.asyncio
    async def test_invalid_last_sequence_sends_error(self) -> None:
        sender = FakeSender()
        buses = _make_buses("sess-1")
        mgr = _make_manager(sender, buses)

        raw = MultiplexMessage(
            type=MultiplexMessageType.STREAM_OPEN,
            stream_id="sess-1",
            payload={"session_id": "sess-1", "last_sequence": "not-an-int"},
        ).to_json()
        await mgr.handle_message(raw)

        assert mgr.active_streams == []
        assert len(sender.messages) == 1
        assert sender.last().type == MultiplexMessageType.ERROR
        assert "last_sequence" in sender.last().payload["error"]
        await mgr.stop()


class TestCloseStream:
    @pytest.mark.asyncio
    async def test_close_unsubscribes(self):
        sender = FakeSender()
        buses = _make_buses("sess-1")
        mgr = _make_manager(sender, buses)

        await mgr.handle_message(MultiplexMessage.stream_open("sess-1").to_json())
        assert "sess-1" in mgr.active_streams

        # Verify subscription exists on the bus
        bus = buses["sess-1"]
        consumer_id = f"mpx-test-sess-1"
        assert consumer_id in bus._subscribers

        await mgr.handle_message(MultiplexMessage.stream_close("sess-1").to_json())
        assert mgr.active_streams == []
        assert consumer_id not in bus._subscribers

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_close_nonexistent_is_noop(self):
        sender = FakeSender()
        mgr = _make_manager(sender, {})

        await mgr.handle_message(MultiplexMessage.stream_close("nope").to_json())
        # No error -- closing a non-open stream is harmless
        assert len(sender.messages) == 0
        await mgr.stop()


class TestPingPong:
    @pytest.mark.asyncio
    async def test_client_ping_gets_pong(self):
        sender = FakeSender()
        mgr = _make_manager(sender, {})

        await mgr.handle_message(MultiplexMessage.ping().to_json())

        assert len(sender.messages) == 1
        assert sender.last().type == MultiplexMessageType.PONG
        await mgr.stop()

    @pytest.mark.asyncio
    async def test_ping_loop_sends_ping(self):
        sender = FakeSender()
        mgr = _make_manager(sender, {}, ping_interval=0.05)
        mgr.start()

        # Wait long enough for at least one ping
        await asyncio.sleep(0.15)
        await mgr.stop()

        ping_msgs = [m for m in sender.messages if m.type == MultiplexMessageType.PING]
        assert len(ping_msgs) >= 1


class TestInvalidMessages:
    @pytest.mark.asyncio
    async def test_invalid_json_sends_error(self):
        sender = FakeSender()
        mgr = _make_manager(sender, {})

        await mgr.handle_message("not json!!!")

        assert len(sender.messages) == 1
        assert sender.last().type == MultiplexMessageType.ERROR
        assert "Invalid message" in sender.last().payload["error"]
        await mgr.stop()

    @pytest.mark.asyncio
    async def test_unsupported_type_sends_error(self):
        sender = FakeSender()
        mgr = _make_manager(sender, {})

        # STREAM_DATA is server->client only
        raw = MultiplexMessage.stream_data("s", {"x": 1}).to_json()
        await mgr.handle_message(raw)

        assert len(sender.messages) == 1
        assert sender.last().type == MultiplexMessageType.ERROR
        assert "Unsupported" in sender.last().payload["error"]
        await mgr.stop()


class TestStopCleansUp:
    @pytest.mark.asyncio
    async def test_stop_closes_all_streams(self):
        sender = FakeSender()
        buses = _make_buses("a", "b")
        mgr = _make_manager(sender, buses)

        await mgr.handle_message(MultiplexMessage.stream_open("a").to_json())
        await mgr.handle_message(MultiplexMessage.stream_open("b").to_json())
        assert len(mgr.active_streams) == 2

        await mgr.stop()
        assert mgr.active_streams == []

        # Subscribers should be removed from both buses
        assert len(buses["a"]._subscribers) == 0
        assert len(buses["b"]._subscribers) == 0

    @pytest.mark.asyncio
    async def test_stop_cancels_ping(self):
        sender = FakeSender()
        mgr = _make_manager(sender, {}, ping_interval=0.05)
        mgr.start()
        assert mgr._ping_task is not None

        await mgr.stop()
        assert mgr._ping_task is None

    @pytest.mark.asyncio
    async def test_stop_continues_after_forwarder_failure(self) -> None:
        sender = FailingSender()
        buses = _make_buses("a", "b")
        mgr = _make_manager(sender, buses)

        await mgr.handle_message(MultiplexMessage.stream_open("a").to_json())
        await mgr.handle_message(MultiplexMessage.stream_open("b").to_json())

        await buses["a"].publish(
            AgentEvent(session_id="a", kind=AgentEventKind.THINKING, payload={"step": 1}),
        )
        await asyncio.sleep(0.05)

        await mgr.stop()

        assert mgr.active_streams == []
        assert len(buses["a"]._subscribers) == 0
        assert len(buses["b"]._subscribers) == 0


class TestEventForwarding:
    @pytest.mark.asyncio
    async def test_published_event_forwarded(self):
        sender = FakeSender()
        buses = _make_buses("sess-1")
        mgr = _make_manager(sender, buses)

        await mgr.handle_message(MultiplexMessage.stream_open("sess-1").to_json())

        # Publish an event to the bus
        event = AgentEvent(
            session_id="sess-1",
            kind=AgentEventKind.THINKING,
            payload={"text": "reasoning..."},
        )
        await buses["sess-1"].publish(event)

        # Give the forwarding task a moment to pick it up
        await asyncio.sleep(0.05)

        assert len(sender.messages) >= 1
        data_msgs = [m for m in sender.messages if m.type == MultiplexMessageType.STREAM_DATA]
        assert len(data_msgs) == 1
        assert data_msgs[0].stream_id == "sess-1"
        assert data_msgs[0].payload["kind"] == "thinking"
        assert data_msgs[0].payload["session_id"] == "sess-1"

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_multiple_events_forwarded(self):
        sender = FakeSender()
        buses = _make_buses("sess-1")
        mgr = _make_manager(sender, buses)

        await mgr.handle_message(MultiplexMessage.stream_open("sess-1").to_json())

        for i in range(3):
            event = AgentEvent(
                session_id="sess-1",
                kind=AgentEventKind.THINKING,
                payload={"step": i},
            )
            await buses["sess-1"].publish(event)

        await asyncio.sleep(0.1)

        data_msgs = [m for m in sender.messages if m.type == MultiplexMessageType.STREAM_DATA]
        assert len(data_msgs) == 3
        steps = [m.payload["payload"]["step"] for m in data_msgs]
        assert steps == [0, 1, 2]

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_events_only_for_subscribed_streams(self):
        """Events from un-subscribed sessions must not leak."""
        sender = FakeSender()
        buses = _make_buses("a", "b")
        mgr = _make_manager(sender, buses)

        # Only subscribe to "a"
        await mgr.handle_message(MultiplexMessage.stream_open("a").to_json())

        # Publish to both
        await buses["a"].publish(
            AgentEvent(session_id="a", kind=AgentEventKind.THINKING, payload={"from": "a"}),
        )
        await buses["b"].publish(
            AgentEvent(session_id="b", kind=AgentEventKind.THINKING, payload={"from": "b"}),
        )

        await asyncio.sleep(0.05)

        data_msgs = [m for m in sender.messages if m.type == MultiplexMessageType.STREAM_DATA]
        assert len(data_msgs) == 1
        assert data_msgs[0].stream_id == "a"

        await mgr.stop()
