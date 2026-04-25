"""MultiplexManager -- manages one client's multiplexed WebSocket connection.

Allows a single WebSocket to carry events for multiple agent sessions.
Each session is identified by a *stream_id* (== session_id).  The client
sends ``STREAM_OPEN`` / ``STREAM_CLOSE`` to subscribe / unsubscribe.
Events are forwarded as ``STREAM_DATA`` frames.  A periodic ``PING`` /
``PONG`` keepalive keeps the connection alive through proxies.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Awaitable, Callable, Optional

from loguru import logger

from tldw_Server_API.app.core.Agent_Client_Protocol.event_bus import SessionEventBus
from tldw_Server_API.app.core.Agent_Client_Protocol.multiplex.protocol import (
    MultiplexMessage,
    MultiplexMessageType,
)

# Type aliases for the two callables injected at construction time.
SendFn = Callable[[str], Awaitable[None]]
GetBusFn = Callable[[str], Optional[SessionEventBus]]
AuthorizeStreamFn = Callable[[str], Awaitable[tuple[bool, str | None, Any | None]]]
ReleaseStreamFn = Callable[[str, Any | None], None | Awaitable[None]]


class MultiplexManager:
    """Manages stream subscriptions and keepalive for one WebSocket client.

    Parameters
    ----------
    connection_id:
        Human-readable identifier for logging.
    send_fn:
        ``async (str) -> None`` -- sends a text frame over the WebSocket.
    get_bus_fn:
        ``(session_id) -> SessionEventBus | None`` -- looks up a session's
        event bus.  Returns ``None`` when the session is unknown.
    ping_interval:
        Seconds between keepalive ``PING`` frames.  Set to ``0`` to disable.
    """

    def __init__(
        self,
        connection_id: str,
        send_fn: SendFn,
        get_bus_fn: GetBusFn,
        authorize_stream_fn: AuthorizeStreamFn | None = None,
        release_stream_fn: ReleaseStreamFn | None = None,
        ping_interval: float = 30,
    ) -> None:
        self._connection_id = connection_id
        self._send_fn = send_fn
        self._get_bus_fn = get_bus_fn
        self._authorize_stream_fn = authorize_stream_fn
        self._release_stream_fn = release_stream_fn
        self._ping_interval = ping_interval

        # stream_id -> {bus, consumer_id, task, release_state}
        self._streams: dict[str, dict[str, Any]] = {}
        self._ping_task: Optional[asyncio.Task[None]] = None
        self._stopped = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background ping loop (if interval > 0)."""
        if self._ping_interval > 0:
            self._ping_task = asyncio.ensure_future(self._ping_loop())

    async def stop(self) -> None:
        """Close all streams and cancel the ping task."""
        self._stopped = True
        # Cancel ping first
        if self._ping_task is not None:
            await self._cancel_task(self._ping_task, label="ping loop")
            self._ping_task = None

        # Close every open stream
        stream_ids = list(self._streams.keys())
        for sid in stream_ids:
            await self._close_stream(sid)

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def handle_message(self, raw: str) -> None:
        """Parse *raw* JSON and dispatch by message type."""
        try:
            msg = MultiplexMessage.from_json(raw)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning(
                "Multiplex {}: invalid message: {}", self._connection_id, exc,
            )
            await self._send(MultiplexMessage.error(f"Invalid message: {exc}"))
            return

        dispatch = {
            MultiplexMessageType.STREAM_OPEN: self._open_stream,
            MultiplexMessageType.STREAM_CLOSE: self._handle_close,
            MultiplexMessageType.PING: self._handle_ping,
            MultiplexMessageType.PONG: self._handle_pong,
        }
        handler = dispatch.get(msg.type)
        if handler is not None:
            await handler(msg)
        else:
            await self._send(
                MultiplexMessage.error(f"Unsupported message type: {msg.type.value}"),
            )

    # ------------------------------------------------------------------
    # Stream management
    # ------------------------------------------------------------------

    async def _open_stream(self, msg: MultiplexMessage) -> None:
        """Subscribe to a session's event bus and start forwarding."""
        stream_id = msg.stream_id
        if stream_id is None:
            await self._send(MultiplexMessage.error("stream_open requires stream_id"))
            return

        if stream_id in self._streams:
            # Already subscribed -- idempotent
            return

        if msg.payload is not None and not isinstance(msg.payload, dict):
            await self._send(
                MultiplexMessage.error("stream_open payload must be an object", stream_id=stream_id),
            )
            return

        bus = self._get_bus_fn(stream_id)

        last_sequence = 0
        if isinstance(msg.payload, dict) and "last_sequence" in msg.payload:
            try:
                last_sequence = int(msg.payload["last_sequence"])
            except (TypeError, ValueError):
                await self._send(
                    MultiplexMessage.error(
                        "stream_open last_sequence must be a non-negative integer",
                        stream_id=stream_id,
                    ),
                )
                return
            if last_sequence < 0:
                await self._send(
                    MultiplexMessage.error(
                        "stream_open last_sequence must be a non-negative integer",
                        stream_id=stream_id,
                    ),
                )
                return

        release_state: Any | None = None
        if self._authorize_stream_fn is not None:
            allowed, error_message, release_state = await self._authorize_stream_fn(stream_id)
            if not allowed:
                await self._send(
                    MultiplexMessage.error(
                        error_message or "stream_open not authorized",
                        stream_id=stream_id,
                    ),
                )
                return

        if bus is None:
            await self._release_stream(stream_id, release_state)
            await self._send(
                MultiplexMessage.error(f"Unknown session: {stream_id}", stream_id=stream_id),
            )
            return

        consumer_id = f"mpx-{self._connection_id}-{stream_id}"
        queue = bus.subscribe(consumer_id, from_sequence=last_sequence)
        task = asyncio.ensure_future(self._forward_events(stream_id, queue))

        self._streams[stream_id] = {
            "bus": bus,
            "consumer_id": consumer_id,
            "task": task,
            "release_state": release_state,
        }
        logger.debug(
            "Multiplex {}: opened stream {} (last_seq={})",
            self._connection_id,
            stream_id,
            last_sequence,
        )

    async def _handle_close(self, msg: MultiplexMessage) -> None:
        """Handle a STREAM_CLOSE request."""
        stream_id = msg.stream_id
        if stream_id is None:
            await self._send(MultiplexMessage.error("stream_close requires stream_id"))
            return
        await self._close_stream(stream_id)

    async def _close_stream(self, stream_id: str) -> None:
        """Unsubscribe from a session's event bus and cancel the forwarding task."""
        info = self._streams.pop(stream_id, None)
        if info is None:
            return

        # Unsubscribe from the bus
        bus: SessionEventBus = info["bus"]
        bus.unsubscribe(info["consumer_id"])

        # Cancel the forwarding task
        task: asyncio.Task[None] = info["task"]
        await self._cancel_task(task, label=f"stream {stream_id}")
        await self._release_stream(stream_id, info.get("release_state"))

        logger.debug("Multiplex {}: closed stream {}", self._connection_id, stream_id)

    # ------------------------------------------------------------------
    # Event forwarding
    # ------------------------------------------------------------------

    async def _forward_events(self, stream_id: str, queue: asyncio.Queue) -> None:  # type: ignore[type-arg]
        """Read events from *queue* and send as STREAM_DATA frames."""
        try:
            while not self._stopped:
                event = await queue.get()
                data_msg = MultiplexMessage.stream_data(stream_id, event.to_dict())
                await self._send(data_msg)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning(
                "Multiplex {}: stopping forwarder for stream {} after send failure: {}",
                self._connection_id,
                stream_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Keepalive
    # ------------------------------------------------------------------

    async def _ping_loop(self) -> None:
        """Periodically send PING frames."""
        try:
            while not self._stopped:
                await asyncio.sleep(self._ping_interval)
                if self._stopped:
                    break
                await self._send(MultiplexMessage.ping())
        except asyncio.CancelledError:
            pass

    async def _handle_ping(self, msg: MultiplexMessage) -> None:
        """Respond to client PING with PONG."""
        await self._send(MultiplexMessage.pong())

    async def _handle_pong(self, msg: MultiplexMessage) -> None:
        """Client acknowledged our PING -- nothing to do."""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_streams(self) -> list[str]:
        """Return currently subscribed stream IDs."""
        return list(self._streams.keys())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _send(self, msg: MultiplexMessage) -> None:
        """Serialize and send a message via the injected send function."""
        await self._send_fn(msg.to_json())

    async def _cancel_task(self, task: asyncio.Task[None], *, label: str) -> None:
        """Cancel a background task, logging any non-cancellation failure."""
        task.cancel()
        with _suppress_cancelled():
            try:
                await task
            except Exception as exc:
                logger.warning(
                    "Multiplex {}: background {} exited with error during cleanup: {}",
                    self._connection_id,
                    label,
                    exc,
                )

    async def _release_stream(self, stream_id: str, release_state: Any | None) -> None:
        """Release any stream-scoped resource allocations such as quota tokens."""
        if self._release_stream_fn is None or release_state is None:
            return
        try:
            maybe_awaitable = self._release_stream_fn(stream_id, release_state)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        except Exception as exc:
            logger.warning(
                "Multiplex {}: failed to release stream {} resources: {}",
                self._connection_id,
                stream_id,
                exc,
            )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

class _suppress_cancelled:
    """Tiny context manager that swallows ``CancelledError``."""

    def __enter__(self) -> _suppress_cancelled:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, tb: Any) -> bool:
        return exc_type is asyncio.CancelledError
