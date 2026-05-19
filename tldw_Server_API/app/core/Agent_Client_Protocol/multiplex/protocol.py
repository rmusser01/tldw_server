"""Multiplex protocol -- message types for multi-session WebSocket multiplexing."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MultiplexMessageType(str, Enum):
    """Wire-level message types for the multiplex WebSocket."""

    STREAM_OPEN = "stream_open"
    STREAM_DATA = "stream_data"
    STREAM_CLOSE = "stream_close"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"


@dataclass
class MultiplexMessage:
    """Single message on the multiplexed WebSocket.

    Fields *stream_id* and *payload* are optional -- ``to_dict`` omits them
    when ``None`` so that keepalive frames stay tiny.
    """

    type: MultiplexMessageType
    stream_id: Optional[str] = None
    payload: Optional[dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict, omitting ``None`` optional fields."""
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
        """Serialize to a compact JSON string."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiplexMessage:
        """Reconstruct a message from a dict."""
        if "type" not in data:
            raise ValueError("MultiplexMessage.from_dict: missing required key 'type'")
        return cls(
            type=MultiplexMessageType(data["type"]),
            stream_id=data.get("stream_id"),
            payload=data.get("payload"),
            timestamp=data.get("timestamp", time.time()),
        )

    @classmethod
    def from_json(cls, raw: str) -> MultiplexMessage:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(raw))

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def stream_open(cls, session_id: str, last_sequence: int = 0) -> MultiplexMessage:
        """Request to subscribe to *session_id*'s event stream."""
        payload: dict[str, Any] = {"session_id": session_id}
        if last_sequence > 0:
            payload["last_sequence"] = last_sequence
        return cls(
            type=MultiplexMessageType.STREAM_OPEN,
            stream_id=session_id,
            payload=payload,
        )

    @classmethod
    def stream_data(cls, stream_id: str, event_data: dict[str, Any]) -> MultiplexMessage:
        """Wrap an event for delivery over the multiplexed socket."""
        return cls(
            type=MultiplexMessageType.STREAM_DATA,
            stream_id=stream_id,
            payload=event_data,
        )

    @classmethod
    def stream_close(cls, session_id: str) -> MultiplexMessage:
        """Request to unsubscribe from *session_id*'s event stream."""
        return cls(
            type=MultiplexMessageType.STREAM_CLOSE,
            stream_id=session_id,
        )

    @classmethod
    def ping(cls) -> MultiplexMessage:
        """Keepalive ping."""
        return cls(type=MultiplexMessageType.PING)

    @classmethod
    def pong(cls) -> MultiplexMessage:
        """Keepalive pong (response to ping)."""
        return cls(type=MultiplexMessageType.PONG)

    @classmethod
    def error(cls, message: str, stream_id: Optional[str] = None) -> MultiplexMessage:
        """Error message, optionally scoped to a stream."""
        return cls(
            type=MultiplexMessageType.ERROR,
            stream_id=stream_id,
            payload={"error": message},
        )
