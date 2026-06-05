"""Models for Notes task-backed checklist parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """Durable task status values."""

    OPEN = "open"
    DONE = "done"


class ProjectionStatus(str, Enum):
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
