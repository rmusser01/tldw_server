"""SSE (Server-Sent Events) consumer for ACP agent events.

Formats AgentEvents as SSE text lines and provides an async generator
for use with FastAPI's StreamingResponse.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncGenerator

from tldw_Server_API.app.core.Agent_Client_Protocol.consumers.base import EventConsumer
from tldw_Server_API.app.core.Agent_Client_Protocol.event_bus import SessionEventBus
from tldw_Server_API.app.core.Agent_Client_Protocol.events import AgentEvent


class SSEConsumer(EventConsumer):
    """Streams AgentEvents as Server-Sent Events.

    Each event is formatted as::

        event: {kind}
        data: {json}

    A heartbeat comment (``": heartbeat\\n\\n"``) is emitted after
    *heartbeat_interval* seconds of silence to keep the connection alive
    through proxies and load balancers.

    Usage with FastAPI::

        consumer = SSEConsumer(from_sequence=last_event_id)
        await consumer.start(bus)
        return StreamingResponse(
            consumer.iter_sse_lines(),
            media_type="text/event-stream",
        )
    """

    consumer_id: str = "sse"

    def __init__(
        self,
        consumer_id: str | None = None,
        from_sequence: int = 0,
        heartbeat_interval: float = 15.0,
    ) -> None:
        if consumer_id is not None:
            self.consumer_id = consumer_id
        else:
            self.consumer_id = f"sse_{uuid.uuid4().hex[:12]}"
        self._from_sequence = from_sequence
        self._heartbeat_interval = heartbeat_interval
        self._queue: asyncio.Queue[AgentEvent] | None = None
        self._bus: SessionEventBus | None = None
        self._stopped = False

    async def on_event(self, event: AgentEvent) -> None:
        """Buffer the event for consumption by :meth:`iter_sse_lines`."""
        if self._queue is not None:
            await self._queue.put(event)

    async def start(self, bus: SessionEventBus) -> None:
        """Subscribe to *bus*, replaying from *from_sequence* if set."""
        self._bus = bus
        # subscribe() returns a queue and replays buffered events when
        # from_sequence > 0, so no separate replay_events() call needed.
        self._queue = bus.subscribe(
            self.consumer_id,
            from_sequence=self._from_sequence,
        )
        self._stopped = False

    async def stop(self) -> None:
        """Unsubscribe from the bus and signal the generator to exit."""
        self._stopped = True
        if self._bus is not None:
            self._bus.unsubscribe(self.consumer_id)
            self._bus = None

    async def iter_sse_lines(self) -> AsyncGenerator[str, None]:
        """Yield SSE-formatted lines. Use with ``StreamingResponse``.

        Blocks on the internal queue, yielding ``event:``/``data:`` frames
        for real events and ``": heartbeat"`` comments on timeout.  The
        generator exits once :meth:`stop` has been called.
        """
        while not self._stopped:
            if self._queue is None:
                break
            try:
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self._heartbeat_interval,
                )
                kind = event.kind.value
                payload = json.dumps(event.to_dict(), default=str)
                yield f"event: {kind}\ndata: {payload}\n\n"
            except (asyncio.TimeoutError, TimeoutError):
                if self._stopped:
                    break
                yield ": heartbeat\n\n"
