"""Helpers for meeting event envelopes and transport framing."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

_SSE_CONTROL_FIELD_PATTERN = re.compile(r"[^A-Za-z0-9_.:-]+")
_SSE_CONTROL_FIELD_MAX_LENGTH = 256


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_meeting_event(
    *,
    event_type: str,
    session_id: str,
    data: dict[str, Any] | None = None,
    event_id: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    return {
        "id": event_id or uuid.uuid4().hex,
        "type": str(event_type),
        "session_id": str(session_id),
        "timestamp": timestamp or utcnow_iso(),
        "data": data or {},
    }


def _sanitize_sse_control_field(value: Any, *, default: str) -> str:
    """Normalize an SSE id/event field so it cannot inject new frame lines."""
    text = str(value or "").strip()
    if not text:
        return default
    text = re.sub(r"[\r\n]+", "_", text)
    text = _SSE_CONTROL_FIELD_PATTERN.sub("_", text)
    text = text[:_SSE_CONTROL_FIELD_MAX_LENGTH].strip("_")
    return text or default


def to_sse_frame(event: dict[str, Any]) -> str:
    payload = json.dumps(event, separators=(",", ":"), default=str)
    event_id = _sanitize_sse_control_field(event.get("id"), default="")
    event_type = _sanitize_sse_control_field(event.get("type"), default="event")
    return f"id: {event_id}\nevent: {event_type}\ndata: {payload}\n\n"
