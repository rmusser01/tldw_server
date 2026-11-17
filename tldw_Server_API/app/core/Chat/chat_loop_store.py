"""In-memory append-only event store for chat loop runs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from threading import RLock
from typing import Any

from tldw_Server_API.app.core.Chat.chat_loop_schemas import ChatLoopEvent, ChatLoopEventType


@dataclass(frozen=True)
class ChatLoopCheckpoint:
    run_id: str
    checkpoint_seq: int
    last_seq: int
    dropped_events: int


@dataclass(frozen=True)
class ChatLoopReplayState:
    run_id: str
    last_seq: int
    checkpoint_seq: int
    tail_events: list[ChatLoopEvent]


class InMemoryChatLoopStore:
    """Stores chat loop events keyed by run ID and supports cursor replay."""

    def __init__(self) -> None:
        self._events: dict[str, list[ChatLoopEvent]] = defaultdict(list)
        self._checkpoints: dict[str, ChatLoopCheckpoint] = {}
        self._lock = RLock()

    def append(self, run_id: str, event: ChatLoopEventType, data: dict[str, Any]) -> ChatLoopEvent:
        """Append one event and return the canonical stored envelope."""
        with self._lock:
            checkpoint = self._checkpoints.get(run_id)
            base_seq = checkpoint.last_seq if checkpoint is not None else 0
            if self._events[run_id]:
                base_seq = self._events[run_id][-1].seq
            next_seq = base_seq + 1
            record = ChatLoopEvent(
                run_id=run_id,
                seq=next_seq,
                event=event,
                data=data,
            )
            self._events[run_id].append(record)
            if checkpoint is not None:
                self._checkpoints[run_id] = ChatLoopCheckpoint(
                    run_id=run_id,
                    checkpoint_seq=checkpoint.checkpoint_seq,
                    last_seq=record.seq,
                    dropped_events=checkpoint.dropped_events,
                )
            return record

    def list_after(self, run_id: str, seq: int) -> list[ChatLoopEvent]:
        """List all events for run_id where event.seq > seq."""
        with self._lock:
            return [event for event in self._events.get(run_id, []) if event.seq > seq]

    def compact(self, run_id: str, *, keep_tail: int = 200) -> ChatLoopCheckpoint | None:
        """Drop old events while preserving a checkpoint and sequence continuity."""
        if keep_tail < 1:
            keep_tail = 1

        with self._lock:
            events = self._events.get(run_id, [])
            if len(events) <= keep_tail:
                return None

            dropped_count = len(events) - keep_tail
            dropped_until_seq = events[dropped_count - 1].seq
            tail = events[-keep_tail:]
            last_seq = tail[-1].seq

            existing = self._checkpoints.get(run_id)
            total_dropped = dropped_count + (existing.dropped_events if existing else 0)
            checkpoint = ChatLoopCheckpoint(
                run_id=run_id,
                checkpoint_seq=dropped_until_seq,
                last_seq=last_seq,
                dropped_events=total_dropped,
            )

            self._events[run_id] = tail
            self._checkpoints[run_id] = checkpoint
            return checkpoint

    def replay(self, run_id: str) -> ChatLoopReplayState | None:
        """Return checkpoint + tail state that clients can use to rebuild run state."""
        with self._lock:
            events = list(self._events.get(run_id, []))
            checkpoint = self._checkpoints.get(run_id)
            if not events and checkpoint is None:
                return None

            tail_last_seq = events[-1].seq if events else 0
            checkpoint_seq = checkpoint.checkpoint_seq if checkpoint else 0
            last_seq = max(tail_last_seq, checkpoint.last_seq if checkpoint else 0)

            return ChatLoopReplayState(
                run_id=run_id,
                last_seq=last_seq,
                checkpoint_seq=checkpoint_seq,
                tail_events=events,
            )
