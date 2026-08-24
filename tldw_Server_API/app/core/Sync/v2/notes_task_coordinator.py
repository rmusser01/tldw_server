"""Deterministic Notes task mutation-group planning and projection evidence."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from tldw_Server_API.app.core.Notes_Tasks.models import ParsedChecklistItem
from tldw_Server_API.app.core.Notes_Tasks.projection_markers import task_marker_hash

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.chacha.task_store import TaskStore
    from tldw_Server_API.app.core.Sync.v2.models import SyncEnvelope
    from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


TASK_PROJECTION_ROUTING_KEY = "task_projection"
_PROJECTION_METADATA_FIELDS = {
    "projection_version",
    "task_id",
    "task_envelope_id",
    "task_revision",
    "task_hash",
    "note_envelope_id",
    "note_hash",
    "linked",
    "marker_hash",
}
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class TaskProjectionGroupMetadata:
    """Privacy-safe durable evidence for one managed checklist projection."""

    projection_version: int
    task_id: str
    task_envelope_id: str
    task_revision: int
    task_hash: str
    note_envelope_id: str
    note_hash: str
    linked: bool
    marker_hash: str

    def as_routing_value(self) -> dict[str, object]:
        """Return canonical JSON-safe routing metadata."""
        return {
            "projection_version": self.projection_version,
            "task_id": self.task_id,
            "task_envelope_id": self.task_envelope_id,
            "task_revision": self.task_revision,
            "task_hash": self.task_hash,
            "note_envelope_id": self.note_envelope_id,
            "note_hash": self.note_hash,
            "linked": self.linked,
            "marker_hash": self.marker_hash,
        }


@dataclass(frozen=True, slots=True)
class ProjectionCacheRebuildResult:
    """Result of rebuilding one disposable projection locator cache."""

    projection: dict[str, Any] | None
    reason_code: str | None


def _validate_task_projection_group_metadata(
    value: Mapping[str, object],
) -> TaskProjectionGroupMetadata:
    """Validate the closed, content-free projection anchor contract."""
    if not isinstance(value, Mapping) or set(value) != _PROJECTION_METADATA_FIELDS:
        raise ValueError("Task projection group metadata has unsupported fields")
    projection_version = value["projection_version"]
    if type(projection_version) is not int or projection_version != 1:
        raise ValueError("Task projection version must be 1")
    task_id = _canonical_uuid4(value["task_id"], "task_id")
    task_envelope_id = _canonical_opaque_id(
        value["task_envelope_id"], "task_envelope_id"
    )
    task_revision = value["task_revision"]
    if type(task_revision) is not int or task_revision < 1:
        raise ValueError("Task projection revision must be a positive integer")
    task_hash = _canonical_hash(value["task_hash"], "task_hash")
    note_envelope_id = _canonical_opaque_id(
        value["note_envelope_id"], "note_envelope_id"
    )
    note_hash = _canonical_hash(value["note_hash"], "note_hash")
    linked = value["linked"]
    if type(linked) is not bool:
        raise ValueError("Task projection linked state must be boolean")
    marker_hash = _canonical_hash(value["marker_hash"], "marker_hash")
    return TaskProjectionGroupMetadata(
        projection_version=projection_version,
        task_id=task_id,
        task_envelope_id=task_envelope_id,
        task_revision=task_revision,
        task_hash=task_hash,
        note_envelope_id=note_envelope_id,
        note_hash=note_hash,
        linked=linked,
        marker_hash=marker_hash,
    )


def rebuild_task_projection_cache(
    *,
    task_store: TaskStore,
    sync_store: SyncV2Store,
    owner_user_id: str,
    dataset_id: str,
    note_id: str,
    item: ParsedChecklistItem,
) -> ProjectionCacheRebuildResult:
    """Rebuild one locator cache only from marker and immutable Sync evidence."""
    marker = item.marker
    if item.marker_reason_code is not None:
        return ProjectionCacheRebuildResult(
            projection=None,
            reason_code=item.marker_reason_code,
        )
    if marker is None:
        return ProjectionCacheRebuildResult(
            projection=None,
            reason_code="missing_marker_base",
        )
    task_envelope = sync_store.get_historical_task_envelope(
        owner_user_id=owner_user_id,
        dataset_id=dataset_id,
        task_id=marker.task_id,
        object_revision=marker.revision,
        object_hash=marker.object_hash,
    )
    if task_envelope is None:
        return ProjectionCacheRebuildResult(
            projection=None,
            reason_code="base_unavailable",
        )
    anchor = _projection_anchor_from_envelope(task_envelope)
    if anchor is None:
        return ProjectionCacheRebuildResult(
            projection=None,
            reason_code="base_unavailable",
        )
    if (
        anchor.task_id != marker.task_id
        or anchor.task_revision != marker.revision
        or anchor.task_hash != marker.object_hash
        or anchor.task_envelope_id != task_envelope.client_envelope_id
        or anchor.marker_hash != task_marker_hash(marker)
        or not anchor.linked
        or task_envelope.parent_id != note_id
    ):
        return ProjectionCacheRebuildResult(
            projection=None,
            reason_code="marker_scope_mismatch",
        )
    note_envelope = sync_store.get_projection_note_envelope(
        owner_user_id=owner_user_id,
        dataset_id=dataset_id,
        note_id=note_id,
        envelope_id=anchor.note_envelope_id,
        object_hash=anchor.note_hash,
    )
    if note_envelope is None or not _same_projection_group(
        task_envelope,
        note_envelope,
        anchor,
    ):
        return ProjectionCacheRebuildResult(
            projection=None,
            reason_code="base_unavailable",
        )
    task = task_store.get_task(
        owner_user_id=owner_user_id,
        dataset_id=dataset_id,
        task_id=marker.task_id,
        include_deleted=True,
    )
    if (
        task is None
        or bool(task["deleted"])
        or task["note_id"] != note_id
        or task["projection_status"] != "live"
    ):
        return ProjectionCacheRebuildResult(
            projection=None,
            reason_code="marker_scope_mismatch",
        )
    locator = item.locator
    projection = task_store.set_task_projection(
        owner_user_id=owner_user_id,
        dataset_id=dataset_id,
        task_id=marker.task_id,
        note_id=note_id,
        note_version=locator.note_version,
        line_number=locator.line_number,
        start_offset=locator.start_offset,
        end_offset=locator.end_offset,
        normalized_text_hash=locator.normalized_text_hash,
        occurrence_index=locator.occurrence_index,
        block_fingerprint=locator.block_fingerprint,
        raw_line=item.raw_line,
        has_child_content=item.has_child_content,
        projection_status="live",
    )
    return ProjectionCacheRebuildResult(projection=projection, reason_code=None)


def _projection_anchor_from_envelope(
    envelope: SyncEnvelope,
) -> TaskProjectionGroupMetadata | None:
    """Return one validated projection anchor from immutable routing metadata."""
    raw = envelope.routing_metadata.get(TASK_PROJECTION_ROUTING_KEY)
    if not isinstance(raw, Mapping):
        return None
    try:
        return _validate_task_projection_group_metadata(raw)
    except ValueError:
        return None


def _same_projection_group(
    task_envelope: SyncEnvelope,
    note_envelope: SyncEnvelope,
    anchor: TaskProjectionGroupMetadata,
) -> bool:
    """Return whether both exact envelopes carry the same complete anchor."""
    note_anchor = _projection_anchor_from_envelope(note_envelope)
    return (
        note_anchor == anchor
        and note_envelope.client_envelope_id == anchor.note_envelope_id
        and task_envelope.mutation_group_id is not None
        and task_envelope.mutation_group_id == note_envelope.mutation_group_id
        and task_envelope.mutation_step_count == note_envelope.mutation_step_count
        and task_envelope.mutation_plan_hash == note_envelope.mutation_plan_hash
    )


def _canonical_uuid4(value: object, field_name: str) -> str:
    """Return one exact lowercase UUIDv4 string."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a canonical UUIDv4")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a canonical UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{field_name} must be a canonical UUIDv4")
    return value


def _canonical_opaque_id(value: object, field_name: str) -> str:
    """Return one bounded normalized printable opaque identifier."""
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or unicodedata.normalize("NFKC", value) != value
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
    ):
        raise ValueError(f"{field_name} must be a bounded canonical identifier")
    return value


def _canonical_hash(value: object, field_name: str) -> str:
    """Return one lowercase SHA-256 value."""
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a canonical SHA-256 hash")
    return value
