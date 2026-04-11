"""Tests for the MultiplexManager server-side connection manager."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.multiplex.manager import MultiplexManager
from tldw_Server_API.app.core.Agent_Client_Protocol.multiplex.protocol import (
    MultiplexMessage,
    MultiplexMessageType,
)


@pytest.fixture
def mock_bus():
    bus = MagicMock()
    queue: asyncio.Queue = asyncio.Queue()
    bus.subscribe.return_value = queue
    bus.unsubscribe = MagicMock()
    return bus, queue


@pytest.fixture
def manager(mock_bus):
    bus, queue = mock_bus
    sent: list[dict] = []

    async def send(data: str) -> None:
        sent.append(json.loads(data))

    mgr = MultiplexManager(
        send_fn=send,
        get_bus_fn=lambda sid: bus if sid == "session-1" else None,
        ping_interval=999,  # Don't ping during tests
    )
    return mgr, sent, bus, queue


@pytest.mark.asyncio
async def test_open_stream(manager):
    mgr, sent, bus, queue = manager
    await mgr.start()
    try:
        msg = MultiplexMessage.stream_open("session-1").to_json()
        await mgr.handle_message(msg)
        assert "session-1" in mgr.active_streams
        bus.subscribe.assert_called_once()
    finally:
        await mgr.stop()


@pytest.mark.asyncio
async def test_open_stream_unknown_session(manager):
    mgr, sent, bus, queue = manager
    await mgr.start()
    try:
        msg = MultiplexMessage.stream_open("nonexistent").to_json()
        await mgr.handle_message(msg)
        assert len(sent) == 1
        assert sent[0]["type"] == "error"
    finally:
        await mgr.stop()


@pytest.mark.asyncio
async def test_close_stream(manager):
    mgr, sent, bus, queue = manager
    await mgr.start()
    try:
        await mgr.handle_message(MultiplexMessage.stream_open("session-1").to_json())
        await mgr.handle_message(MultiplexMessage.stream_close("session-1").to_json())
        assert "session-1" not in mgr.active_streams
        bus.unsubscribe.assert_called_once()
    finally:
        await mgr.stop()


@pytest.mark.asyncio
async def test_ping_responds_with_pong(manager):
    mgr, sent, bus, queue = manager
    await mgr.start()
    try:
        await mgr.handle_message(MultiplexMessage.ping().to_json())
        assert any(m["type"] == "pong" for m in sent)
    finally:
        await mgr.stop()


@pytest.mark.asyncio
async def test_invalid_message(manager):
    mgr, sent, bus, queue = manager
    await mgr.start()
    try:
        await mgr.handle_message("not valid json{{{")
        assert any(m["type"] == "error" for m in sent)
    finally:
        await mgr.stop()


@pytest.mark.asyncio
async def test_stop_cleans_up_all_streams(manager):
    mgr, sent, bus, queue = manager
    await mgr.start()
    await mgr.handle_message(MultiplexMessage.stream_open("session-1").to_json())
    assert len(mgr.active_streams) == 1
    await mgr.stop()
    assert len(mgr.active_streams) == 0


@pytest.mark.asyncio
async def test_duplicate_stream_open_returns_error(manager):
    mgr, sent, bus, queue = manager
    await mgr.start()
    try:
        await mgr.handle_message(MultiplexMessage.stream_open("session-1").to_json())
        await mgr.handle_message(MultiplexMessage.stream_open("session-1").to_json())
        errors = [m for m in sent if m["type"] == "error"]
        assert len(errors) == 1
        assert "already open" in errors[0]["payload"]["error"].lower()
    finally:
        await mgr.stop()


@pytest.mark.asyncio
async def test_forward_events(manager):
    """Events placed on the bus queue are forwarded as STREAM_DATA."""
    mgr, sent, bus, queue = manager
    await mgr.start()
    try:
        await mgr.handle_message(MultiplexMessage.stream_open("session-1").to_json())

        # Simulate an event arriving on the bus queue
        mock_event = MagicMock()
        mock_event.to_dict.return_value = {"kind": "heartbeat", "sequence": 1}
        await queue.put(mock_event)

        # Give the forwarding task a moment to process
        await asyncio.sleep(0.1)

        data_msgs = [m for m in sent if m["type"] == "stream_data"]
        assert len(data_msgs) == 1
        assert data_msgs[0]["stream_id"] == "session-1"
        assert data_msgs[0]["payload"]["kind"] == "heartbeat"
    finally:
        await mgr.stop()
