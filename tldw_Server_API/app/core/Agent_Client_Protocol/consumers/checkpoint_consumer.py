"""Checkpoint consumer for auto-snapshotting sandbox state.

Subscribes to SessionEventBus and creates snapshots before
FILE_CHANGE events, enabling rollback to any checkpoint.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, TYPE_CHECKING

from loguru import logger

from tldw_Server_API.app.core.Agent_Client_Protocol.consumers.base import EventConsumer
from tldw_Server_API.app.core.Agent_Client_Protocol.event_bus import SessionEventBus
from tldw_Server_API.app.core.Agent_Client_Protocol.events import (
    AgentEvent,
    AgentEventKind,
)

if TYPE_CHECKING:
    pass


# Event kinds that represent file mutations and should trigger a checkpoint.
_FILE_MUTATION_KINDS: frozenset[AgentEventKind] = frozenset({
    AgentEventKind.FILE_CHANGE,
})


class CheckpointConsumer(EventConsumer):
    """Auto-snapshot sandbox state before file mutations.

    Each instance is scoped to a single session. When a ``FILE_CHANGE``
    event arrives, the consumer calls ``SandboxService.create_snapshot``
    and records the mapping ``{sequence: snapshot_id}``.

    If the sandbox service raises (e.g. because a run is active and
    ``_ensure_no_active_session_runs`` fires), the consumer logs a
    warning and continues -- it never crashes the event loop.

    Checkpoints are bounded by *max_checkpoints*; when the limit is
    exceeded the oldest checkpoint is evicted from the local mapping.
    (The underlying snapshot storage may have its own quota enforcement.)
    """

    consumer_id: str = "checkpoint"

    def __init__(
        self,
        sandbox_service: Any,
        session_id: str,
        max_checkpoints: int = 50,
    ) -> None:
        self.consumer_id = f"checkpoint_{uuid.uuid4().hex[:8]}"
        self._service = sandbox_service
        self._session_id = session_id
        self._max_checkpoints = max_checkpoints

        # sequence -> snapshot_id
        self._checkpoints: dict[int, str] = {}

        self._bus: SessionEventBus | None = None
        self._queue: asyncio.Queue[AgentEvent] | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopped: bool = False

    # ------------------------------------------------------------------ #
    # EventConsumer interface
    # ------------------------------------------------------------------ #

    async def on_event(self, event: AgentEvent) -> None:
        """Handle a single event -- snapshot before file mutations."""
        if event.kind not in _FILE_MUTATION_KINDS:
            return

        try:
            result = self._service.create_snapshot(self._session_id)
            if asyncio.iscoroutine(result):
                result = await result

            snapshot_id = result.get("snapshot_id") if isinstance(result, dict) else None
            if snapshot_id:
                self._checkpoints[event.sequence] = snapshot_id
                logger.debug(
                    "Checkpoint created: seq={} snapshot={}",
                    event.sequence,
                    snapshot_id,
                )
                # Evict oldest if over limit
                self._evict_oldest()
            else:
                logger.debug(
                    "Checkpoint skipped for seq {}: create_snapshot returned no snapshot_id",
                    event.sequence,
                )
        except Exception as exc:
            # Active-run guard (SessionActiveRunsConflict) or any other
            # error -- skip gracefully, never crash the consume loop.
            logger.debug(
                "Checkpoint skipped for seq {}: {}",
                event.sequence,
                exc,
            )

    async def start(self, bus: SessionEventBus) -> None:
        """Subscribe to *bus* and begin consuming events."""
        self._bus = bus
        self._queue = bus.subscribe(self.consumer_id)
        self._stopped = False
        self._task = asyncio.create_task(self._consume_loop())

    async def stop(self) -> None:
        """Cancel the consume loop and unsubscribe from the bus."""
        self._stopped = True
        if self._bus is not None:
            self._bus.unsubscribe(self.consumer_id)
            self._bus = None
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # ------------------------------------------------------------------ #
    # Public query helpers
    # ------------------------------------------------------------------ #

    def get_checkpoints(self) -> dict[int, str]:
        """Return a copy of the ``{sequence: snapshot_id}`` mapping."""
        return dict(self._checkpoints)

    def get_nearest_checkpoint(self, target_sequence: int) -> tuple[int, str] | None:
        """Find the nearest checkpoint at or before *target_sequence*.

        Returns ``(sequence, snapshot_id)`` or ``None`` if no checkpoint
        exists at or before the target.
        """
        candidates = {
            seq: sid
            for seq, sid in self._checkpoints.items()
            if seq <= target_sequence
        }
        if not candidates:
            return None
        nearest_seq = max(candidates)
        return nearest_seq, candidates[nearest_seq]

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _evict_oldest(self) -> None:
        """Remove the oldest checkpoint if the limit is exceeded."""
        while len(self._checkpoints) > self._max_checkpoints:
            oldest_seq = min(self._checkpoints)
            del self._checkpoints[oldest_seq]

    async def _consume_loop(self) -> None:
        """Read events from the queue and dispatch to :meth:`on_event`."""
        if self._queue is None:
            return
        while not self._stopped:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=30.0,
                )
                await self.on_event(event)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
