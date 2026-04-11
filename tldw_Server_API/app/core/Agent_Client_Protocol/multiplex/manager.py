"""Server-side multiplexer for multi-session WebSocket connections.

Manages per-client stream subscriptions via SessionEventBus.
Each stream_id maps to a session_id. Events from subscribed sessions
are forwarded as STREAM_DATA frames.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Callable, Awaitable

from loguru import logger

from .protocol import MultiplexMessage, MultiplexMessageType


class MultiplexManager:
    """Manages a single client's multiplexed WebSocket connection."""

    def __init__(
        self,
        connection_id: str | None = None,
        send_fn: Callable[[str], Awaitable[None]] | None = None,
        get_bus_fn: Callable[[str], Any] | None = None,  # session_id -> SessionEventBus | None
        ping_interval: float = 30.0,
    ) -> None:
        self._conn_id = connection_id or uuid.uuid4().hex[:12]
        self._send = send_fn
        self._get_bus = get_bus_fn
        self._ping_interval = ping_interval

        # Active streams: stream_id -> {bus, consumer_id, task}
        self._streams: dict[str, dict[str, Any]] = {}
        self._running = False
        self._ping_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the ping keepalive loop."""
        self._running = True
        self._ping_task = asyncio.create_task(self._ping_loop())

    async def stop(self) -> None:
        """Stop all streams and the ping loop."""
        self._running = False
        # Close all streams
        for stream_id in list(self._streams):
            await self._close_stream(stream_id)
        # Cancel ping
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def handle_message(self, raw: str) -> None:
        """Handle an incoming message from the client."""
        try:
            msg = MultiplexMessage.from_json(raw)
        except Exception as exc:
            await self._send_message(MultiplexMessage.error(f"Invalid message: {exc}"))
            return

        if msg.type == MultiplexMessageType.STREAM_OPEN:
            await self._open_stream(msg)
        elif msg.type == MultiplexMessageType.STREAM_CLOSE:
            await self._close_stream(msg.stream_id)
        elif msg.type == MultiplexMessageType.PING:
            await self._send_message(MultiplexMessage.pong())
        elif msg.type == MultiplexMessageType.PONG:
            pass  # Client acknowledging our ping
        else:
            await self._send_message(MultiplexMessage.error(
                f"Unexpected message type: {msg.type}",
                stream_id=msg.stream_id,
            ))

    # ------------------------------------------------------------------
    # Stream management
    # ------------------------------------------------------------------

    async def _open_stream(self, msg: MultiplexMessage) -> None:
        """Subscribe to a session's event stream."""
        session_id = msg.stream_id
        if not session_id:
            await self._send_message(MultiplexMessage.error("stream_id required"))
            return

        if session_id in self._streams:
            await self._send_message(MultiplexMessage.error(
                "Stream already open", stream_id=session_id,
            ))
            return

        # Get the session's event bus
        bus = self._get_bus(session_id) if self._get_bus else None
        if bus is None:
            await self._send_message(MultiplexMessage.error(
                "Session not found or not active", stream_id=session_id,
            ))
            return

        # Subscribe to the bus
        consumer_id = f"mpx_{self._conn_id}_{session_id}"
        last_sequence = 0
        if msg.payload and isinstance(msg.payload, dict):
            last_sequence = msg.payload.get("last_sequence", 0)

        queue = bus.subscribe(consumer_id, from_sequence=last_sequence)

        # Start a forwarding task
        task = asyncio.create_task(self._forward_events(session_id, queue))
        self._streams[session_id] = {
            "bus": bus,
            "consumer_id": consumer_id,
            "task": task,
        }
        logger.debug(
            "Multiplex stream opened: session={} consumer={}",
            session_id,
            consumer_id,
        )

    async def _close_stream(self, stream_id: str | None) -> None:
        """Unsubscribe from a session's event stream."""
        if not stream_id or stream_id not in self._streams:
            return
        info = self._streams.pop(stream_id)
        # Cancel forwarding task
        task = info.get("task")
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        # Unsubscribe from bus
        bus = info.get("bus")
        consumer_id = info.get("consumer_id")
        if bus and consumer_id:
            try:
                bus.unsubscribe(consumer_id)
            except Exception:
                pass
        logger.debug("Multiplex stream closed: session={}", stream_id)

    # ------------------------------------------------------------------
    # Event forwarding
    # ------------------------------------------------------------------

    async def _forward_events(self, stream_id: str, queue: asyncio.Queue) -> None:
        """Read events from bus queue and send as STREAM_DATA frames."""
        try:
            while self._running:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    data_msg = MultiplexMessage.stream_data(
                        stream_id=stream_id,
                        event_data=event.to_dict(),
                    )
                    await self._send_message(data_msg)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Multiplex forward error for stream {}: {}", stream_id, exc)

    # ------------------------------------------------------------------
    # Keepalive
    # ------------------------------------------------------------------

    async def _ping_loop(self) -> None:
        """Send periodic pings to keep the connection alive."""
        try:
            while self._running:
                await asyncio.sleep(self._ping_interval)
                if self._running:
                    await self._send_message(MultiplexMessage.ping())
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    async def _send_message(self, msg: MultiplexMessage) -> None:
        """Send a message to the client."""
        if self._send:
            try:
                await self._send(msg.to_json())
            except Exception as exc:
                logger.debug("Multiplex send failed: {}", exc)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_streams(self) -> list[str]:
        """Return list of currently active stream IDs."""
        return list(self._streams.keys())
