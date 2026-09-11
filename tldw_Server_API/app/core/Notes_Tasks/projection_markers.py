"""Canonical hidden markers for managed Notes task checklist lines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from uuid import UUID

_MARKER_RE = re.compile(
    r"<!-- tldw-task:v1:"
    r"(?P<task_id>[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}):"
    r"(?P<revision>[1-9][0-9]*):(?P<object_hash>sha256:[0-9a-f]{64}) -->"
)
_MARKER_PREFIX = "<!-- tldw-task:"


@dataclass(frozen=True)
class TaskMarker:
    """Stable task identity and exact last-projected canonical base."""

    task_id: str
    revision: int
    object_hash: str


@dataclass(frozen=True)
class TaskMarkerParseResult:
    """Visible checklist body plus optional validated projection identity."""

    body: str
    marker: TaskMarker | None
    reason_code: str | None


def render_task_marker(task_id: str, *, revision: int, object_hash: str) -> str:
    """Render one canonical managed checklist marker."""
    marker = TaskMarker(
        task_id=_canonical_task_id(task_id),
        revision=_canonical_revision(revision),
        object_hash=_canonical_hash(object_hash),
    )
    return (
        f"<!-- tldw-task:v1:{marker.task_id}:{marker.revision}:"
        f"{marker.object_hash} -->"
    )


def parse_task_marker(value: str) -> TaskMarker | None:
    """Parse a canonical standalone marker, returning ``None`` when absent."""
    match = _MARKER_RE.fullmatch(value)
    if match is None:
        return None
    return TaskMarker(
        task_id=match.group("task_id"),
        revision=int(match.group("revision")),
        object_hash=match.group("object_hash"),
    )


def task_marker_hash(marker: TaskMarker) -> str:
    """Return the canonical SHA-256 hash of one rendered marker."""
    rendered = render_task_marker(
        marker.task_id,
        revision=marker.revision,
        object_hash=marker.object_hash,
    )
    return "sha256:" + sha256(rendered.encode("ascii")).hexdigest()


def extract_task_marker(value: str) -> TaskMarkerParseResult:
    """Extract one trailing marker without trusting malformed or duplicate IDs."""
    marker_count = value.count(_MARKER_PREFIX)
    if marker_count == 0:
        return TaskMarkerParseResult(body=value, marker=None, reason_code=None)

    marker_start = value.find(_MARKER_PREFIX)
    body = value[:marker_start].rstrip()
    if marker_count > 1:
        return TaskMarkerParseResult(
            body=body,
            marker=None,
            reason_code="duplicate_marker",
        )

    marker = parse_task_marker(value[marker_start:])
    if marker is None:
        return TaskMarkerParseResult(
            body=body,
            marker=None,
            reason_code="malformed_marker",
        )
    return TaskMarkerParseResult(body=body, marker=marker, reason_code=None)


def _canonical_task_id(value: str) -> str:
    """Return a canonical lowercase UUIDv4 task identity."""
    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Task marker task_id must be a canonical UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError("Task marker task_id must be a canonical UUIDv4")
    return value


def _canonical_revision(value: int) -> int:
    """Return a positive canonical task revision."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("Task marker revision must be a positive integer")
    return value


def _canonical_hash(value: str) -> str:
    """Return a lowercase SHA-256 hash."""
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("Task marker object_hash must be a canonical SHA-256 hash")
    return value
