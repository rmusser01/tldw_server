"""Admin SSE event stream endpoint.

Provides a real-time Server-Sent Events stream aggregating monitoring alerts,
security events, ACP session changes, budget breaches, and job completions
for the admin dashboard.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from loguru import logger

router = APIRouter(tags=["admin-events"])

_NONCRITICAL = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)


# ---------------------------------------------------------------------------
# In-memory event bus
# ---------------------------------------------------------------------------

class AdminEventBus:
    """Simple pub/sub event bus for admin dashboard events.

    Subscribers receive events as dicts via an asyncio.Queue.
    """

    def __init__(self, maxsize: int = 256) -> None:
        self._subscribers: list[asyncio.Queue[dict[str, Any] | None]] = []
        self._lock = asyncio.Lock()
        self._maxsize = maxsize

    async def subscribe(self) -> asyncio.Queue[dict[str, Any] | None]:
        q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=self._maxsize)
        async with self._lock:
            self._subscribers.append(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue[dict[str, Any] | None]) -> None:
        async with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    async def publish(self, event: dict[str, Any]) -> None:
        """Broadcast an event to all subscribers. Drops if queue full."""
        event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        async with self._lock:
            dead: list[asyncio.Queue] = []
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    # Drop oldest event and retry
                    try:
                        q.get_nowait()
                        q.put_nowait(event)
                    except (asyncio.QueueEmpty, asyncio.QueueFull):
                        dead.append(q)
            for q in dead:
                try:
                    self._subscribers.remove(q)
                except ValueError:
                    pass

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


# Module-level singleton
_event_bus: AdminEventBus | None = None
_event_bus_lock = asyncio.Lock()


async def get_admin_event_bus() -> AdminEventBus:
    global _event_bus
    if _event_bus is None:
        async with _event_bus_lock:
            if _event_bus is None:
                _event_bus = AdminEventBus()
    return _event_bus


async def emit_admin_event(
    event_type: str,
    data: dict[str, Any],
    *,
    category: str = "system",
) -> None:
    """Convenience function to publish an admin event from anywhere in the app."""
    try:
        bus = await get_admin_event_bus()
        await bus.publish({
            "event": event_type,
            "category": category,
            "data": data,
        })
    except _NONCRITICAL:
        pass


# ---------------------------------------------------------------------------
# SSE Endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/events/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-sent events stream for admin dashboard events",
            "content": {"text/event-stream": {}},
        },
    },
)
async def admin_events_stream(
    request: Request,
    categories: str | None = Query(
        default=None,
        description="Comma-separated event categories to subscribe to (e.g., 'acp,monitoring,security'). All if omitted.",
    ),
) -> StreamingResponse:
    """Server-Sent Events stream for the admin dashboard.

    Emits events in the following categories:
    - ``acp``: ACP session created/closed/error, token usage updates
    - ``monitoring``: Health alerts, metric threshold breaches
    - ``security``: Failed logins, suspicious activity alerts
    - ``budget``: Budget threshold breaches
    - ``jobs``: Job completions, failures, SLA breaches
    - ``system``: Server start/stop, config changes

    Sends a heartbeat comment every 15 seconds to keep the connection alive.
    """
    bus = await get_admin_event_bus()
    queue = await bus.subscribe()

    category_filter: set[str] | None = None
    if categories:
        category_filter = {c.strip().lower() for c in categories.split(",") if c.strip()}

    async def event_generator():
        try:
            # Initial connection event
            yield _sse_format("connected", {"subscriber_count": bus.subscriber_count})
            last_heartbeat = time.monotonic()

            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield ": heartbeat\n\n"
                    last_heartbeat = time.monotonic()
                    continue

                if event is None:
                    break

                # Apply category filter
                if category_filter:
                    event_category = event.get("category", "system")
                    if event_category not in category_filter:
                        continue

                event_type = event.get("event", "update")
                yield _sse_format(event_type, event)

        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            await bus.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_format(event: str, data: dict[str, Any]) -> str:
    """Format a dict as an SSE frame."""
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"
