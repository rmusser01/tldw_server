"""Tests for the multiplex WebSocket protocol."""

from tldw_Server_API.app.core.Agent_Client_Protocol.multiplex.protocol import (
    MultiplexMessage,
    MultiplexMessageType,
)


def test_stream_open_message():
    msg = MultiplexMessage.stream_open("session-123")
    assert msg.type == MultiplexMessageType.STREAM_OPEN
    assert msg.stream_id == "session-123"


def test_stream_open_with_last_sequence():
    msg = MultiplexMessage.stream_open("s1", last_sequence=5)
    assert msg.payload == {"last_sequence": 5}


def test_stream_data_message():
    msg = MultiplexMessage.stream_data("s1", {"kind": "tool_call"})
    assert msg.stream_id == "s1"
    assert msg.payload["kind"] == "tool_call"


def test_stream_close_message():
    msg = MultiplexMessage.stream_close("s1")
    assert msg.type == MultiplexMessageType.STREAM_CLOSE


def test_ping_pong():
    ping = MultiplexMessage.ping()
    pong = MultiplexMessage.pong()
    assert ping.type == MultiplexMessageType.PING
    assert pong.type == MultiplexMessageType.PONG
    assert ping.stream_id is None


def test_error_message():
    err = MultiplexMessage.error("session not found", stream_id="s1")
    assert err.payload["error"] == "session not found"
    assert err.stream_id == "s1"


def test_to_dict_roundtrip():
    msg = MultiplexMessage.stream_data("s1", {"kind": "text", "data": "hello"})
    d = msg.to_dict()
    restored = MultiplexMessage.from_dict(d)
    assert restored.type == msg.type
    assert restored.stream_id == msg.stream_id
    assert restored.payload == msg.payload


def test_to_json_roundtrip():
    msg = MultiplexMessage.stream_open("s1", last_sequence=10)
    j = msg.to_json()
    restored = MultiplexMessage.from_json(j)
    assert restored.type == MultiplexMessageType.STREAM_OPEN
    assert restored.payload == {"last_sequence": 10}


def test_to_dict_omits_none_fields():
    msg = MultiplexMessage.ping()
    d = msg.to_dict()
    assert "stream_id" not in d
    assert "payload" not in d


def test_all_message_types_exist():
    types = [t.value for t in MultiplexMessageType]
    assert "stream_open" in types
    assert "stream_data" in types
    assert "stream_close" in types
    assert "ping" in types
    assert "pong" in types
    assert "error" in types
