"""Tests for MultiplexMessage and MultiplexMessageType."""
from __future__ import annotations

import json
import time

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.multiplex.protocol import (
    MultiplexMessage,
    MultiplexMessageType,
)


# ---------------------------------------------------------------------------
# Enum values
# ---------------------------------------------------------------------------

class TestMultiplexMessageType:
    def test_all_values(self) -> None:
        expected = {"stream_open", "stream_data", "stream_close", "ping", "pong", "error"}
        assert {e.value for e in MultiplexMessageType} == expected

    def test_str_mixin(self) -> None:
        assert MultiplexMessageType.PING == "ping"
        assert isinstance(MultiplexMessageType.PING, str)


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------

class TestFactoryMethods:
    def test_stream_open_basic(self) -> None:
        msg = MultiplexMessage.stream_open("sess-1")
        assert msg.type == MultiplexMessageType.STREAM_OPEN
        assert msg.stream_id == "sess-1"
        assert msg.payload == {"session_id": "sess-1"}

    def test_stream_open_with_last_sequence(self) -> None:
        msg = MultiplexMessage.stream_open("sess-1", last_sequence=42)
        assert msg.payload == {"session_id": "sess-1", "last_sequence": 42}

    def test_stream_open_last_sequence_zero_omitted(self) -> None:
        msg = MultiplexMessage.stream_open("sess-1", last_sequence=0)
        assert "last_sequence" not in msg.payload

    def test_stream_data(self) -> None:
        data = {"kind": "thinking", "text": "hmm"}
        msg = MultiplexMessage.stream_data("sess-1", data)
        assert msg.type == MultiplexMessageType.STREAM_DATA
        assert msg.stream_id == "sess-1"
        assert msg.payload == data

    def test_stream_close(self) -> None:
        msg = MultiplexMessage.stream_close("sess-1")
        assert msg.type == MultiplexMessageType.STREAM_CLOSE
        assert msg.stream_id == "sess-1"
        assert msg.payload is None

    def test_ping(self) -> None:
        msg = MultiplexMessage.ping()
        assert msg.type == MultiplexMessageType.PING
        assert msg.stream_id is None
        assert msg.payload is None

    def test_pong(self) -> None:
        msg = MultiplexMessage.pong()
        assert msg.type == MultiplexMessageType.PONG
        assert msg.stream_id is None
        assert msg.payload is None

    def test_error_no_stream(self) -> None:
        msg = MultiplexMessage.error("bad request")
        assert msg.type == MultiplexMessageType.ERROR
        assert msg.stream_id is None
        assert msg.payload == {"error": "bad request"}

    def test_error_with_stream(self) -> None:
        msg = MultiplexMessage.error("not found", stream_id="sess-99")
        assert msg.type == MultiplexMessageType.ERROR
        assert msg.stream_id == "sess-99"
        assert msg.payload == {"error": "not found"}


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_omits_none_fields(self) -> None:
        msg = MultiplexMessage.ping()
        d = msg.to_dict()
        assert "stream_id" not in d
        assert "payload" not in d
        assert d["type"] == "ping"
        assert "timestamp" in d

    def test_to_dict_includes_present_fields(self) -> None:
        msg = MultiplexMessage.stream_data("s1", {"key": "val"})
        d = msg.to_dict()
        assert d["stream_id"] == "s1"
        assert d["payload"] == {"key": "val"}

    def test_roundtrip_dict(self) -> None:
        msg = MultiplexMessage.stream_open("sess-abc", last_sequence=7)
        recovered = MultiplexMessage.from_dict(msg.to_dict())
        assert recovered.type == msg.type
        assert recovered.stream_id == msg.stream_id
        assert recovered.payload == msg.payload
        assert abs(recovered.timestamp - msg.timestamp) < 0.01

    def test_roundtrip_json(self) -> None:
        msg = MultiplexMessage.error("oops", stream_id="s2")
        recovered = MultiplexMessage.from_json(msg.to_json())
        assert recovered.type == msg.type
        assert recovered.stream_id == msg.stream_id
        assert recovered.payload == msg.payload

    def test_to_json_compact(self) -> None:
        msg = MultiplexMessage.pong()
        raw = msg.to_json()
        # No spaces after separators
        assert " " not in raw

    def test_from_dict_missing_type_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required key 'type'"):
            MultiplexMessage.from_dict({"stream_id": "x"})

    def test_from_dict_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError):
            MultiplexMessage.from_dict({"type": "nonexistent"})

    def test_from_json_invalid_json_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            MultiplexMessage.from_json("not-json{{{")

    def test_timestamp_defaults_to_now(self) -> None:
        before = time.time()
        msg = MultiplexMessage.ping()
        after = time.time()
        assert before <= msg.timestamp <= after

    def test_from_dict_default_timestamp(self) -> None:
        """When timestamp is absent, from_dict uses current time."""
        before = time.time()
        msg = MultiplexMessage.from_dict({"type": "pong"})
        after = time.time()
        assert before <= msg.timestamp <= after
