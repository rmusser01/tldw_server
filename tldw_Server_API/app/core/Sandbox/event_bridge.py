"""Bridge between Sandbox RunStreamHub and ACP SessionEventBus.

Subscribes to RunStreamHub frames for a given run and translates them into
typed AgentEvent objects published on the SessionEventBus.  This lets clients
consume a single unified event stream instead of speaking two protocols.
"""
from __future__ import annotations

import asyncio
from typing import Any, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Sandbox.streams import RunStreamHub
    from tldw_Server_API.app.core.Agent_Client_Protocol.event_bus import SessionEventBus


class SandboxEventBridge:
    """Translates RunStreamHub frames into AgentEvents on a SessionEventBus."""

    def __init__(
        self,
        hub: "RunStreamHub",
        bus: "SessionEventBus",
        run_id: str,
        session_id: str,
    ) -> None:
        self._hub = hub
        self._bus = bus
        self._run_id = run_id
        self._session_id = session_id
        self._task: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[dict[str, Any]] | None = None
        self._stopped = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to the hub and begin translating frames."""
        if self._task is not None:
            return
        self._queue = self._hub.subscribe(self._run_id)
        self._task = asyncio.create_task(self._consume(), name=f"event-bridge-{self._run_id}")
        logger.debug("SandboxEventBridge started for run_id={}", self._run_id)

    async def stop(self) -> None:
        """Stop consuming frames and clean up the subscription."""
        self._stopped = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._queue is not None:
            self._hub.unsubscribe(self._run_id, self._queue)
            self._queue = None
        self._task = None
        logger.debug("SandboxEventBridge stopped for run_id={}", self._run_id)

    # ------------------------------------------------------------------
    # Internal consume loop
    # ------------------------------------------------------------------

    async def _consume(self) -> None:
        """Read frames from the hub queue, translate, and publish to the bus."""
        assert self._queue is not None  # noqa: S101 — guarded by start()
        try:
            while not self._stopped:
                try:
                    frame = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                event = self._translate_frame(frame)
                if event is not None:
                    await self._bus.publish(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.opt(exception=True).error(
                "SandboxEventBridge consume loop failed for run_id={}",
                self._run_id,
            )

    # ------------------------------------------------------------------
    # Frame → AgentEvent translation
    # ------------------------------------------------------------------

    def _translate_frame(self, frame: dict[str, Any]) -> Any:
        """Map a raw sandbox frame dict to an AgentEvent (or *None* to skip)."""
        from tldw_Server_API.app.core.Agent_Client_Protocol.events import (
            AgentEvent,
            AgentEventKind,
        )

        frame_type: str = frame.get("type", "")

        if frame_type in ("stdout", "stderr"):
            return AgentEvent(
                session_id=self._session_id,
                kind=AgentEventKind.TERMINAL_OUTPUT,
                payload={
                    "stream": frame_type,
                    "data": frame.get("data", ""),
                    "encoding": frame.get("encoding", "utf8"),
                },
                metadata={"run_id": self._run_id, "sandbox_seq": frame.get("seq")},
            )

        if frame_type == "event":
            event_name = frame.get("event", "")
            if event_name == "end":
                return AgentEvent(
                    session_id=self._session_id,
                    kind=AgentEventKind.COMPLETION,
                    payload=frame.get("data", {}),
                    metadata={"run_id": self._run_id, "sandbox_seq": frame.get("seq")},
                )
            # Generic lifecycle events (e.g. start, status changes)
            return AgentEvent(
                session_id=self._session_id,
                kind=AgentEventKind.LIFECYCLE,
                payload={"event": event_name, **(frame.get("data") or {})},
                metadata={"run_id": self._run_id, "sandbox_seq": frame.get("seq")},
            )

        if frame_type == "truncated":
            return AgentEvent(
                session_id=self._session_id,
                kind=AgentEventKind.ERROR,
                payload={"error": "truncated", "reason": frame.get("reason", "")},
                metadata={"run_id": self._run_id, "sandbox_seq": frame.get("seq")},
            )

        if frame_type == "heartbeat":
            return AgentEvent(
                session_id=self._session_id,
                kind=AgentEventKind.HEARTBEAT,
                payload={},
                metadata={"run_id": self._run_id},
            )

        # Unknown frame type — log and skip
        logger.debug(
            "SandboxEventBridge: skipping unknown frame type={!r} for run_id={}",
            frame_type,
            self._run_id,
        )
        return None
