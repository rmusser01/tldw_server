"""Deterministic Notes task mutation-group planning and projection evidence."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any
from uuid import UUID

from tldw_Server_API.app.core.Notes_Tasks.markdown_parser import parse_note_checklists
from tldw_Server_API.app.core.Notes_Tasks.models import ParsedChecklistItem
from tldw_Server_API.app.core.Notes_Tasks.projection_markers import (
    TaskMarker,
    render_task_marker,
    task_marker_hash,
)

from .errors import SyncStoreError
from .models import SyncDataset
from .mutation_group_validation import SYNC_MUTATION_GROUP_MAX_SIZE
from .server_origin import canonical_payload_hash
from .server_origin_batch import (
    ServerOriginBatchResult,
    ServerOriginMutationStep,
    capture_server_origin_mutation_batch,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.chacha.task_store import TaskStore
    from tldw_Server_API.app.core.Notes_Tasks.service import NotesTaskCaptureMutation
    from tldw_Server_API.app.core.Sync.v2.models import SyncEnvelope
    from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service
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


@dataclass(frozen=True, slots=True)
class NotesTaskMutationPlan:
    """One complete deterministic task mutation group ready for atomic append."""

    steps: tuple[ServerOriginMutationStep, ...]
    idempotency_key: str


@dataclass(slots=True)
class NotesTaskCoordinator:
    """Build and append complete task/activity/note mutation groups."""

    service: SyncV2Service | None = None
    user_id: str | None = None
    dataset_id: str | None = None

    def plan_task_mutation(
        self,
        mutation: NotesTaskCaptureMutation,
        *,
        note_step: ServerOriginMutationStep | None = None,
    ) -> NotesTaskMutationPlan:
        """Plan one task transition with its activity and optional note projection."""

        if note_step is None:
            plan = NotesTaskMutationPlan(
                steps=mutation.steps,
                idempotency_key=mutation.idempotency_key,
            )
            self.validate_plan(plan.steps)
            return plan
        planned_note = _canonical_note_step(
            note_step,
            identity_parts=(mutation.idempotency_key,),
        )
        task_step, activity_step = _bind_projection_anchor(
            mutation,
            note_step=planned_note,
        )
        plan = NotesTaskMutationPlan(
            steps=(task_step, activity_step, planned_note),
            idempotency_key=mutation.idempotency_key,
        )
        self.validate_plan(plan.steps)
        return plan

    def plan_note_reconciliation(
        self,
        mutations: Sequence[NotesTaskCaptureMutation],
        *,
        note_step: ServerOriginMutationStep,
        idempotency_key: str,
    ) -> NotesTaskMutationPlan:
        """Plan all task transitions plus the note as one bounded atomic group."""

        normalized_key = idempotency_key.strip()
        if not normalized_key:
            raise SyncStoreError("notes_task_mutation_group_invalid")
        mutation_tuple = tuple(mutations)
        planned_note = _canonical_note_step(
            note_step,
            identity_parts=(normalized_key, *(item.idempotency_key for item in mutation_tuple)),
        )
        steps: list[ServerOriginMutationStep] = []
        for mutation in mutation_tuple:
            steps.extend(_bind_projection_anchor(mutation, note_step=planned_note))
        steps.append(planned_note)
        plan = NotesTaskMutationPlan(steps=tuple(steps), idempotency_key=normalized_key)
        self.validate_plan(plan.steps)
        return plan

    @staticmethod
    def validate_plan(steps: Sequence[ServerOriginMutationStep]) -> None:
        """Validate the closed task/activity pairs and optional final note step."""

        _validate_task_mutation_plan(steps)

    def capture(
        self,
        plan: NotesTaskMutationPlan,
        *,
        source: str,
    ) -> ServerOriginBatchResult:
        """Append and materialize one already-complete mutation plan."""

        self.validate_plan(plan.steps)
        if self.service is None or not self.user_id:
            raise SyncStoreError("notes_task_coordinator_not_bound")
        return capture_server_origin_mutation_batch(
            service=self.service,
            user_id=self.user_id,
            steps=plan.steps,
            source=source,
            idempotency_key=plan.idempotency_key,
            trusted_notes_task_coordinator=True,
        )

    def capture_note_projection(
        self,
        note_step: ServerOriginMutationStep,
        *,
        idempotency_key: str,
        source: str = "notes.tasks.reconciliation",
    ) -> ServerOriginBatchResult:
        """Append one note-only projection repair without inventing a task mutation."""

        normalized_key = idempotency_key.strip()
        if not normalized_key:
            raise SyncStoreError("notes_task_mutation_group_invalid")
        if self.service is None or not self.user_id:
            raise SyncStoreError("notes_task_coordinator_not_bound")
        planned = _canonical_note_step(
            note_step,
            identity_parts=(normalized_key,),
        )
        return capture_server_origin_mutation_batch(
            service=self.service,
            user_id=self.user_id,
            steps=(planned,),
            source=source,
            idempotency_key=normalized_key,
            trusted_notes_task_coordinator=True,
        )


def resolve_notes_task_coordinator(
    *,
    user_id: str,
    dataset_id: str,
) -> NotesTaskCoordinator | None:
    """Resolve coupled task authority or preserve inactive legacy behavior."""

    from .server_origin import get_active_server_origin_sync_service_for_user

    owner = str(user_id).strip()
    selected_dataset = str(dataset_id).strip()
    if not owner or not selected_dataset:
        raise SyncStoreError("notes_task_sync_scope_invalid")
    service = get_active_server_origin_sync_service_for_user(owner)
    if service is None:
        return None
    matches = [
        dataset
        for dataset in service.store.list_datasets_for_user(owner)
        if dataset.scope_type == "personal"
        and dataset.metadata.get("default_personal") is True
        and dataset.metadata.get("client_family") == "chatbook"
        and dataset.archived_at is None
    ]
    if len(matches) != 1 or matches[0].dataset_id != selected_dataset:
        raise SyncStoreError("notes_task_sync_scope_conflict")
    dataset = matches[0]
    task_domains = {"notes.task", "notes.task_activity"}
    enrolled = task_domains.intersection(dataset.domains)
    if not enrolled:
        return None
    if enrolled != task_domains:
        raise SyncStoreError("notes_task_sync_domains_incomplete")
    _require_task_domains_ready(dataset)
    return NotesTaskCoordinator(
        service=service,
        user_id=owner,
        dataset_id=dataset.dataset_id,
    )


def _require_task_domains_ready(dataset: SyncDataset) -> None:
    """Require the two task domains to share one ready activation state."""

    for readiness_key in ("notes_task_v1", "notes_task_activity_v1"):
        metadata = dataset.metadata.get(readiness_key)
        state = metadata.get("state") if isinstance(metadata, Mapping) else None
        if state != "ready":
            raise SyncStoreError("notes_task_sync_not_ready")


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


def _task_activity_id(identity: Sequence[object]) -> str:
    """Derive one stable UUIDv4 from the complete canonical mutation identity."""

    encoded = json.dumps(
        list(identity),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return str(UUID(bytes=digest[:16], version=4))


def project_task_payload_into_note(
    *,
    content: str,
    note_id: str,
    note_revision: int,
    task_id: str,
    base_revision: int,
    base_hash: str,
    task_revision: int,
    task_hash: str,
    payload: Mapping[str, object],
) -> str:
    """Rewrite the sole exact managed line to one canonical task projection."""

    parsed = parse_note_checklists(
        note_id=note_id,
        note_version=note_revision,
        content=content,
    )
    matches = [
        item
        for item in parsed.items
        if item.marker
        == TaskMarker(
            task_id=task_id,
            revision=base_revision,
            object_hash=base_hash,
        )
    ]
    if len(matches) != 1:
        raise SyncStoreError("notes_task_projection_base_invalid")
    item = matches[0]
    marker_index = item.raw_line.find("[")
    marker_end = item.raw_line.find("]", marker_index + 1)
    if marker_index < 0 or marker_end != marker_index + 2:
        raise SyncStoreError("notes_task_projection_base_invalid")
    status = payload.get("status")
    title = payload.get("title")
    if status not in {"open", "done"} or not isinstance(title, str) or not title:
        raise SyncStoreError("notes_task_projection_payload_invalid")
    body = [title]
    for key, token_name in (
        ("due_date", "due"),
        ("priority", "priority"),
        ("estimate", "estimate"),
    ):
        value = payload.get(key)
        if value is not None:
            body.append(f"@{token_name}({value})")
    body.append(
        render_task_marker(
            task_id,
            revision=task_revision,
            object_hash=task_hash,
        )
    )
    checked = "x" if status == "done" else " "
    new_line = f"{item.raw_line[:marker_index]}[{checked}] {' '.join(body)}"
    return (
        content[: item.locator.start_offset]
        + new_line
        + content[item.locator.end_offset :]
    )


def _canonical_note_step(
    note_step: ServerOriginMutationStep,
    *,
    identity_parts: Sequence[str],
) -> ServerOriginMutationStep:
    """Return a note step with a deterministic immutable envelope identity."""

    if note_step.domain != "notes.note" or note_step.operation != "upsert":
        raise SyncStoreError("notes_task_mutation_group_invalid")
    envelope_id = note_step.client_envelope_id
    if envelope_id is None:
        payload_hash, _ = canonical_payload_hash(dict(note_step.payload))
        digest = hashlib.sha256(
            json.dumps(
                [*identity_parts, note_step.object_id, payload_hash, note_step.object_revision],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        envelope_id = f"notes-task-note-server-{digest[:32]}"
    return replace(note_step, client_envelope_id=envelope_id)


def _bind_projection_anchor(
    mutation: NotesTaskCaptureMutation,
    *,
    note_step: ServerOriginMutationStep,
) -> tuple[ServerOriginMutationStep, ServerOriginMutationStep]:
    """Bind one task/activity pair to its exact task and note envelope evidence."""

    task_step, activity_step = mutation.steps
    task_envelope_id = task_step.client_envelope_id
    note_envelope_id = note_step.client_envelope_id
    if task_envelope_id is None or note_envelope_id is None:
        raise SyncStoreError("notes_task_mutation_group_invalid")
    task_revision = mutation.after.get("canonical_revision")
    task_hash = mutation.after.get("canonical_hash")
    if type(task_revision) is not int or not isinstance(task_hash, str):
        raise SyncStoreError("notes_task_mutation_group_invalid")
    marker = TaskMarker(
        task_id=task_step.object_id,
        revision=task_revision,
        object_hash=task_hash,
    )
    note_hash, _ = canonical_payload_hash(dict(note_step.payload))
    anchor = TaskProjectionGroupMetadata(
        projection_version=1,
        task_id=marker.task_id,
        task_envelope_id=task_envelope_id,
        task_revision=marker.revision,
        task_hash=marker.object_hash,
        note_envelope_id=note_envelope_id,
        note_hash=note_hash,
        linked=(
            (
                mutation.after.get("projection_status") == "live"
                and not bool(mutation.after.get("deleted"))
            )
            or (
                bool(mutation.after.get("deleted"))
                and mutation.before is not None
                and mutation.before.get("projection_status") == "live"
            )
        ),
        marker_hash=task_marker_hash(marker),
    )
    try:
        routing_value = _validate_task_projection_group_metadata(
            anchor.as_routing_value()
        ).as_routing_value()
    except ValueError as exc:
        raise SyncStoreError("notes_task_mutation_group_invalid") from exc
    task_routing = {
        **dict(task_step.routing_metadata),
        TASK_PROJECTION_ROUTING_KEY: routing_value,
    }
    activity_routing = {
        **dict(activity_step.routing_metadata),
        TASK_PROJECTION_ROUTING_KEY: routing_value,
    }
    return (
        replace(task_step, routing_metadata=task_routing),
        replace(activity_step, routing_metadata=activity_routing),
    )


def _validate_task_mutation_plan(
    steps: Sequence[ServerOriginMutationStep],
) -> None:
    """Reject incomplete, ambiguous, oversized, or cross-parent task plans."""

    plan = tuple(steps)
    if len(plan) > SYNC_MUTATION_GROUP_MAX_SIZE:
        raise SyncStoreError("notes_task_mutation_group_limit_exceeded")
    if len(plan) < 2:
        raise SyncStoreError("notes_task_mutation_group_invalid")
    has_note = plan[-1].domain == "notes.note"
    pair_steps = plan[:-1] if has_note else plan
    if not pair_steps or len(pair_steps) % 2:
        raise SyncStoreError("notes_task_mutation_group_invalid")
    if has_note and (
        plan[-1].operation != "upsert"
        or plan[-1].client_envelope_id is None
        or plan[-1].object_revision is None
    ):
        raise SyncStoreError("notes_task_mutation_group_invalid")

    note_ids: set[str] = set()
    object_keys: set[tuple[str, str]] = set()
    envelope_ids: set[str] = set()
    for index in range(0, len(pair_steps), 2):
        task_step = pair_steps[index]
        activity_step = pair_steps[index + 1]
        if (
            task_step.domain != "notes.task"
            or activity_step.domain != "notes.task_activity"
            or task_step.parent_id is None
            or activity_step.parent_id != task_step.parent_id
            or task_step.object_revision is None
            or task_step.client_envelope_id is None
            or activity_step.object_revision != 1
            or activity_step.client_envelope_id is None
            or activity_step.payload.get("task_id") != task_step.object_id
            or activity_step.payload.get("note_id") != task_step.parent_id
        ):
            raise SyncStoreError("notes_task_mutation_group_invalid")
        note_ids.add(task_step.parent_id)
        for step in (task_step, activity_step):
            object_key = (step.domain, step.object_id)
            if object_key in object_keys or step.client_envelope_id in envelope_ids:
                raise SyncStoreError("notes_task_mutation_group_invalid")
            object_keys.add(object_key)
            envelope_ids.add(step.client_envelope_id)
    if len(note_ids) != 1:
        raise SyncStoreError("notes_task_mutation_group_invalid")
    if has_note:
        note_step = plan[-1]
        if note_step.object_id not in note_ids or note_step.client_envelope_id in envelope_ids:
            raise SyncStoreError("notes_task_mutation_group_invalid")


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
    raw_note_anchor = note_envelope.routing_metadata.get(TASK_PROJECTION_ROUTING_KEY)
    note_anchor = _projection_anchor_from_envelope(note_envelope)
    return (
        (raw_note_anchor is None or note_anchor == anchor)
        and note_envelope.client_envelope_id == anchor.note_envelope_id
        and note_envelope.payload_hash == anchor.note_hash
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
