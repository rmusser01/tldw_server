"""Models for Notes task-backed checklist parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class _ValueTextEnum(str, Enum):
    """Python 3.10-safe string enum with value-based string conversion."""

    def __str__(self) -> str:
        return self.value


class TaskStatus(_ValueTextEnum):
    """Durable task status values."""

    OPEN = "open"
    DONE = "done"


class ProjectionStatus(_ValueTextEnum):
    """Markdown projection status values."""

    LIVE = "live"
    UNLINKED = "unlinked"
    AMBIGUOUS = "ambiguous"
    DELETED = "deleted"


@dataclass(frozen=True)
class TaskLocator:
    """Version-bound location of a checklist line in note markdown."""

    note_id: str
    note_version: int
    line_number: int
    start_offset: int
    end_offset: int
    normalized_text_hash: str
    occurrence_index: int
    block_fingerprint: str


@dataclass(frozen=True)
class ParsedChecklistItem:
    """Structured parser result for one markdown checklist line."""

    note_id: str
    checked: bool
    text: str
    raw_line: str
    metadata: dict[str, Any]
    warnings: list[str]
    locator: TaskLocator
    has_child_content: bool = False


@dataclass(frozen=True)
class ParsedChecklistResult:
    """Checklist parser result for a note body."""

    note_id: str
    note_version: int
    items: list[ParsedChecklistItem] = field(default_factory=list)


@dataclass(frozen=True)
class TaskActor:
    """Actor metadata recorded on task reconciliation events."""

    actor_type: str
    actor_id: str | None = None


@dataclass(frozen=True)
class ReconciliationResult:
    """Summary of one note checklist reconciliation run."""

    note_id: str
    note_version: int
    parsed_count: int
    created_count: int = 0
    updated_count: int = 0
    unlinked_count: int = 0
    ambiguous_count: int = 0
    matched_task_ids: list[str] = field(default_factory=list)
    created_task_ids: list[str] = field(default_factory=list)
    unlinked_task_ids: list[str] = field(default_factory=list)
    warning_count: int = 0
