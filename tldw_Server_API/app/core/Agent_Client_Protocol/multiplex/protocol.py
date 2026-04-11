"""Multi-session WebSocket multiplexer protocol.

Descoped from Scion's full control channel: only stream multiplexing
and keepalive. No CONNECT/CONNECTED handshake or REQUEST/RESPONSE
tunneling (not needed for local execution).

Message types:
- STREAM_OPEN: subscribe to a session's event stream
- STREAM_DATA: event data frame with stream_id
- STREAM_CLOSE: unsubscribe from a session
- PING/PONG: keepalive
- ERROR: error with optional stream_id
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MultiplexMessageType(str, Enum):
    STREAM_OPEN = "stream_open"
    STREAM_DATA = "stream_data"
    STREAM_CLOSE = "stream_close"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"


@dataclass
class MultiplexMessage:
    """Envelope for all messages on the multiplexed WebSocket."""
    type: MultiplexMessageType
    stream_id: str | None = None
    payload: dict[str, Any] | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.type.value,
            "timestamp": self.timestamp,
        }
        if self.stream_id is not None:
            d["stream_id"] = self.stream_id
        if self.payload is not None:
            d["payload"] = self.payload
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MultiplexMessage:
        return cls(
            type=MultiplexMessageType(d["type"]),
            stream_id=d.get("stream_id"),
            payload=d.get("payload"),
            timestamp=d.get("timestamp", time.time()),
        )

    @classmethod
    def from_json(cls, s: str) -> MultiplexMessage:
        return cls.from_dict(json.loads(s))

    @staticmethod
    def stream_open(session_id: str, last_sequence: int = 0) -> MultiplexMessage:
        """Create a STREAM_OPEN message to subscribe to a session."""
        return MultiplexMessage(
            type=MultiplexMessageType.STREAM_OPEN,
            stream_id=session_id,
            payload={"last_sequence": last_sequence} if last_sequence > 0 else None,
        )

    @staticmethod
    def stream_data(stream_id: str, event_data: dict[str, Any]) -> MultiplexMessage:
        """Create a STREAM_DATA message carrying an event."""
        return MultiplexMessage(
            type=MultiplexMessageType.STREAM_DATA,
            stream_id=stream_id,
            payload=event_data,
        )

    @staticmethod
    def stream_close(session_id: str) -> MultiplexMessage:
        """Create a STREAM_CLOSE message to unsubscribe."""
        return MultiplexMessage(
            type=MultiplexMessageType.STREAM_CLOSE,
            stream_id=session_id,
        )

    @staticmethod
    def ping() -> MultiplexMessage:
        return MultiplexMessage(type=MultiplexMessageType.PING)

    @staticmethod
    def pong() -> MultiplexMessage:
        return MultiplexMessage(type=MultiplexMessageType.PONG)

    @staticmethod
    def error(message: str, stream_id: str | None = None) -> MultiplexMessage:
        return MultiplexMessage(
            type=MultiplexMessageType.ERROR,
            stream_id=stream_id,
            payload={"error": message},
        )
