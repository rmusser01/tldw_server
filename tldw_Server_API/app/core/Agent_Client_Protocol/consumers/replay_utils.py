"""Shared replay helper for SSE and WebSocket catch-up.

Both SSE consumers and WebSocket broadcasters need to replay buffered
events when a client connects with a ``from_sequence``.  This module
provides a single implementation so the logic stays consistent.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Agent_Client_Protocol.event_bus import SessionEventBus
    from tldw_Server_API.app.core.Agent_Client_Protocol.events import AgentEvent


async def replay_events(
    bus: SessionEventBus,
    from_sequence: int,
    emit_fn: Callable[[AgentEvent], Awaitable[Any]],
) -> int:
    """Replay buffered events from *from_sequence* via *emit_fn*.

    Parameters
    ----------
    bus:
        The session event bus to read buffered events from.
    from_sequence:
        Sequence number to start replaying from (inclusive).
        If <= 0, no replay is performed.
    emit_fn:
        Async callable invoked once per replayed event.

    Returns
    -------
    int
        Number of events replayed.
    """
    if from_sequence <= 0:
        return 0
    buffered = bus.snapshot(from_sequence=from_sequence)
    for event in buffered:
        await emit_fn(event)
    return len(buffered)
