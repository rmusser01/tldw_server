"""Service entry point for note-backed task reconciliation and mutations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol
from uuid import UUID

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError, InputError
from tldw_Server_API.app.core.Notes.organization_capture import (
    active_coordinator,
    capture_note_upsert,
)
from tldw_Server_API.app.core.Notes_Tasks.markdown_parser import parse_note_checklists
from tldw_Server_API.app.core.Notes_Tasks.models import ParsedChecklistItem, ReconciliationResult, TaskActor
from tldw_Server_API.app.core.Notes_Tasks.projection_markers import render_task_marker
from tldw_Server_API.app.core.Notes_Tasks.reconciler import (
    NotesTaskReconciler,
    classify_managed_projection,
)
from tldw_Server_API.app.core.Sync.v2.models import SyncOperation, normalize_sync_timestamp
from tldw_Server_API.app.core.Sync.v2.notes_task_contract import (
    NotesTaskActivityV1,
    TaskActivitySource,
    notes_task_object_hash,
    parse_notes_task_activity_v1,
    parse_notes_task_v1,
)
from tldw_Server_API.app.core.Sync.v2.server_origin_batch import ServerOriginMutationStep

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.chacha.task_store import TaskConnection
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.app.core.Sync.v2.notes_organization_coordinator import (
        NotesOrganizationCoordinator,
    )
    from tldw_Server_API.app.core.Sync.v2.notes_task_coordinator import (
        NotesTaskCoordinator,
    )


_METADATA_TOKEN_ORDER = ("due_date", "priority", "estimate")
_METADATA_TOKEN_NAMES = {"due_date": "due", "priority": "priority", "estimate": "estimate"}
_TASK_STATUSES = {"open", "done"}


def _write_note_content(
    db: CharactersRAGDB,
    *,
    coordinator: NotesOrganizationCoordinator | None,
    note: dict[str, Any],
    content: str,
    expected_version: int,
    conn: TaskConnection | None = None,
) -> None:
    """Persist a projected checklist edit through active Sync when required."""

    if coordinator is not None:
        capture_note_upsert(
            coordinator,
            note_id=str(note["id"]),
            title=str(note.get("title") or ""),
            content=content,
            conversation_id=note.get("conversation_id"),
            message_id=note.get("message_id"),
            expected_version=expected_version,
            source="notes-tasks",
        )
        return
    db.update_note(
        note_id=str(note["id"]),
        update_data={"content": content},
        expected_version=expected_version,
        conn=conn,
    )


class _ChecklistLine(NamedTuple):
    """Parsed projection of one Markdown checklist line."""

    indent: str
    bullet: str
    space: str
    body_part: str


def _parse_checklist_line(raw_line: str) -> _ChecklistLine | None:
    """Parse a projected checklist line without regex backtracking."""
    index = 0
    while index < len(raw_line) and raw_line[index] in " \t":
        index += 1
    indent = raw_line[:index]
    if index >= len(raw_line) or raw_line[index] not in "-*+":
        return None
    bullet = raw_line[index]
    index += 1
    space_start = index
    while index < len(raw_line) and raw_line[index] in " \t":
        index += 1
    if index == space_start:
        return None
    space = raw_line[space_start:index]
    if index + 3 > len(raw_line) or raw_line[index] != "[" or raw_line[index + 2] != "]":
        return None
    marker = raw_line[index + 1]
    if marker not in " xX":
        return None
    body_part = raw_line[index + 3 :]
    if body_part and body_part[0] not in " \t":
        return None
    return _ChecklistLine(indent=indent, bullet=bullet, space=space, body_part=body_part)


def _is_iso_date_token(value: str) -> bool:
    """Return True when a token has the YYYY-MM-DD shape before date parsing."""
    if len(value) != 10 or value[4] != "-" or value[7] != "-":
        return False
    return value[:4].isdigit() and value[5:7].isdigit() and value[8:].isdigit()


def _is_estimate_token(value: str) -> bool:
    """Return True when a task estimate token has an integer plus m/h/d suffix."""
    return len(value) >= 2 and value[:-1].isdigit() and value[-1].casefold() in {"m", "h", "d"}


def _task_text_contains_parseable_metadata_token(text: str) -> bool:
    """Return True when literal task text contains metadata syntax the parser would consume."""

    start = 0
    while True:
        token_start = text.find("@", start)
        if token_start == -1:
            return False
        name_start = token_start + 1
        open_paren = text.find("(", name_start)
        if open_paren == -1:
            return False
        value_start = open_paren + 1
        value_end = text.find(")", value_start)
        if value_end == -1:
            start = value_start
            continue
        token_name = text[name_start:open_paren].casefold()
        if token_name not in {"due", "priority", "estimate"}:
            start = value_end + 1
            continue
        value = text[value_start:value_end]
        if _is_parseable_task_text_metadata_token(name=token_name, value=value):
            return True
        start = value_end + 1


def _is_parseable_task_text_metadata_token(*, name: str, value: str) -> bool:
    """Validate one allowlisted task metadata token using the markdown parser's value rules."""

    normalized_name = name.casefold()
    normalized_value = value.strip()
    if normalized_name == "due":
        if not _is_iso_date_token(normalized_value):
            return False
        try:
            date.fromisoformat(normalized_value)
        except ValueError:
            return False
        return True
    if normalized_name == "priority":
        return normalized_value.casefold() in {"high", "medium", "low"}
    if normalized_name == "estimate":
        return _is_estimate_token(normalized_value)
    return False


@dataclass(frozen=True)
class ReconciliationBatchResult:
    """Summary for opportunistic stale-note reconciliation work."""

    status: str
    processed_notes: int
    remaining_stale_notes: int
    results: list[ReconciliationResult]


@dataclass(frozen=True)
class TaskStoreScope:
    """Trusted product scope used by compatibility REST and MCP callers."""

    owner_user_id: str
    dataset_id: str


@dataclass(frozen=True, slots=True)
class NotesTaskActivityCapture:
    """One deterministic dormant activity derived from a task transition."""

    payload: NotesTaskActivityV1
    step: ServerOriginMutationStep


@dataclass(frozen=True, slots=True)
class NotesTaskCaptureMutation:
    """Canonical dormant capture input for one committed product task mutation."""

    owner_user_id: str
    dataset_id: str
    actor: TaskActor
    operation: SyncOperation
    before: dict[str, Any] | None
    after: dict[str, Any]
    base_revision: int | None
    base_hash: str | None
    restore_intent: bool
    idempotency_key: str
    step: ServerOriginMutationStep
    activity: NotesTaskActivityCapture

    @property
    def steps(self) -> tuple[ServerOriginMutationStep, ServerOriginMutationStep]:
        """Return the future atomic task/activity plan in dependency order."""

        return self.step, self.activity.step


class NotesTaskCaptureCallback(Protocol):
    """Receive one canonical mutation while its product transaction is open."""

    def __call__(
        self,
        mutation: NotesTaskCaptureMutation,
        *,
        conn: TaskConnection | None,
    ) -> None: ...


NotesTaskCoordinatorResolver = Callable[..., "NotesTaskCoordinator | None"]


def _task_activity_metadata(row: dict[str, Any]) -> dict[str, object]:
    """Return the portable metadata subset used by task activity events."""

    payload = row["sync_payload"]
    return {
        key: payload[key]
        for key in (
            "description",
            "priority",
            "due_date",
            "estimate",
            "recurrence",
            "assignee_id",
            "tags",
            "custom",
        )
    }


def _task_activity_values(
    before: dict[str, Any] | None,
    after: dict[str, Any],
) -> tuple[str, dict[str, object] | None, dict[str, object]]:
    """Derive the sole canonical activity shape for one accepted transition."""

    after_payload = after["sync_payload"]
    if before is None:
        return (
            "created",
            None,
            {
                "title": after_payload["title"],
                "status": after_payload["status"],
                "completed_at": after_payload["completed_at"],
                "metadata": _task_activity_metadata(after),
            },
        )
    before_payload = before["sync_payload"]
    if not bool(before.get("deleted")) and bool(after.get("deleted")):
        return (
            "deleted",
            {
                "deleted": False,
                "projection_status": str(before["projection_status"]),
            },
            {"deleted": True, "projection_status": "deleted"},
        )
    if bool(before.get("deleted")) and not bool(after.get("deleted")):
        return (
            "restored",
            {"deleted": True, "projection_status": "deleted"},
            {
                "deleted": False,
                "projection_status": str(after["projection_status"]),
            },
        )
    before_status = str(before_payload["status"])
    after_status = str(after_payload["status"])
    if (before_status, after_status) == ("open", "done"):
        return "completed", {"status": "open"}, {"status": "done"}
    if (before_status, after_status) == ("done", "open"):
        return "reopened", {"status": "done"}, {"status": "open"}
    before_projection = str(before["projection_status"])
    after_projection = str(after["projection_status"])
    if before_projection == "live" and after_projection == "unlinked":
        return (
            "projection_unlinked",
            {"projection_status": "live"},
            {"projection_status": "unlinked"},
        )
    if before_projection in {"unlinked", "ambiguous"} and after_projection == "live":
        return (
            "projection_linked",
            {"projection_status": before_projection},
            {"projection_status": "live"},
        )
    before_metadata = _task_activity_metadata(before)
    after_metadata = _task_activity_metadata(after)
    if before_payload["title"] != after_payload["title"]:
        return (
            "updated",
            {"title": before_payload["title"], "metadata": before_metadata},
            {"title": after_payload["title"], "metadata": after_metadata},
        )
    if before_metadata != after_metadata:
        return "updated", {"metadata": before_metadata}, {"metadata": after_metadata}
    raise ConflictError(
        "Task transition has no portable activity.",
        entity="tasks",
        entity_id=str(after.get("id") or ""),
    )


def build_task_activity_capture(
    *,
    db: CharactersRAGDB,
    owner_user_id: str,
    dataset_id: str,
    actor: TaskActor,
    before: dict[str, Any] | None,
    after: dict[str, Any],
    source_kind: TaskActivitySource = "rest",
) -> NotesTaskActivityCapture:
    """Build one strict, stable activity from canonical product task rows."""

    owner = str(owner_user_id)
    dataset = str(dataset_id)
    after_row = db.task_store._sync_bootstrap_task_row(after, owner)
    before_row = (
        db.task_store._sync_bootstrap_task_row(before, owner)
        if before is not None
        else None
    )
    if (
        after_row.get("owner_user_id") != owner
        or after_row.get("dataset_id") != dataset
        or (
            before_row is not None
            and (
                before_row.get("owner_user_id") != owner
                or before_row.get("dataset_id") != dataset
                or before_row.get("id") != after_row.get("id")
                or before_row.get("note_id") != after_row.get("note_id")
            )
        )
    ):
        raise ConflictError(
            "Task activity capture scope is invalid.",
            entity="tasks",
            entity_id=str(after_row.get("id") or ""),
        )
    event_type, old_value, new_value = _task_activity_values(before_row, after_row)
    occurred_at = normalize_sync_timestamp(after_row.get("updated_at"))
    if occurred_at is None:
        raise ConflictError(
            "Task activity capture timestamp is invalid.",
            entity="tasks",
            entity_id=str(after_row.get("id") or ""),
        )
    identity_hash = hashlib.sha256(
        json.dumps(
            [
                owner,
                dataset,
                after_row["id"],
                int(after_row.get("version") or 0),
                int(after_row["canonical_revision"]),
                after_row["canonical_hash"],
                event_type,
                old_value,
                new_value,
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    activity_id = str(UUID(bytes=bytes.fromhex(identity_hash[:32]), version=4))
    payload = parse_notes_task_activity_v1(
        {
            "activity_id": activity_id,
            "note_id": str(after_row["note_id"]),
            "task_id": str(after_row["id"]),
            "event_type": event_type,
            "actor_type": actor.actor_type,
            "actor_id": actor.actor_id,
            "source_device_id": None,
            "client_occurred_at": occurred_at,
            "source_kind": source_kind,
            "corrects_activity_id": None,
            "old_value": old_value,
            "new_value": new_value,
            "metadata": {},
        },
        owner_user_id=owner,
        bound_actor_type=actor.actor_type,
        bound_actor_id=actor.actor_id,
        authenticated_device_id=None,
        trusted_server_origin=True,
    )
    step = ServerOriginMutationStep(
        domain="notes.task_activity",
        operation="upsert",
        object_id=activity_id,
        parent_id=str(after_row["note_id"]),
        payload=payload.model_dump(mode="json"),
        created_at_client=occurred_at,
        client_envelope_id=f"notes-task-activity-server-{identity_hash[:32]}",
        object_revision=1,
    )
    return NotesTaskActivityCapture(payload=payload, step=step)


def build_task_capture_mutation(
    *,
    db: CharactersRAGDB,
    owner_user_id: str,
    dataset_id: str,
    actor: TaskActor,
    before: dict[str, Any] | None,
    after: dict[str, Any],
    source_kind: TaskActivitySource = "rest",
) -> NotesTaskCaptureMutation:
    """Build one exact, stable task capture input from canonical product rows."""

    owner = str(owner_user_id)
    dataset = str(dataset_id)
    after_row = db.task_store._sync_bootstrap_task_row(after, owner)
    before_row = (
        db.task_store._sync_bootstrap_task_row(before, owner)
        if before is not None
        else None
    )
    if (
        after_row.get("owner_user_id") != owner
        or after_row.get("dataset_id") != dataset
    ):
        raise ConflictError(
            "Task capture scope is invalid.",
            entity="tasks",
            entity_id=str(after_row.get("id") or ""),
        )
    task_id = str(after_row["id"])
    note_id = str(after_row["note_id"])
    revision = int(after_row["canonical_revision"])
    object_hash = str(after_row["canonical_hash"])
    base_revision = None
    base_hash = None
    if before_row is None:
        if revision != 1:
            raise ConflictError(
                "Task capture create revision is invalid.",
                entity="tasks",
                entity_id=task_id,
            )
    else:
        if (
            before_row.get("owner_user_id") != owner
            or before_row.get("dataset_id") != dataset
            or str(before_row.get("id")) != task_id
            or str(before_row.get("note_id")) != note_id
        ):
            raise ConflictError(
                "Task capture identity is invalid.",
                entity="tasks",
                entity_id=task_id,
            )
        base_revision = int(before_row["canonical_revision"])
        base_hash = str(before_row["canonical_hash"])
        if revision != base_revision + 1:
            raise ConflictError(
                "Task capture revision is invalid.",
                entity="tasks",
                entity_id=task_id,
            )
    deleted = bool(after_row.get("deleted"))
    restore_intent = bool(
        before_row is not None
        and before_row.get("deleted")
        and not deleted
    )
    operation: SyncOperation = "tombstone" if deleted else "upsert"
    identity_hash = hashlib.sha256(
        json.dumps(
            [owner, dataset, task_id, revision, object_hash, operation, restore_intent],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    routing_metadata: dict[str, object] = {}
    if restore_intent:
        routing_metadata["restore_intent"] = True
    if base_revision is not None:
        routing_metadata["product_transition_base"] = True
    step = ServerOriginMutationStep(
        domain="notes.task",
        operation=operation,
        object_id=task_id,
        parent_id=note_id,
        payload=dict(after_row["sync_payload"]),
        routing_metadata=routing_metadata,
        client_envelope_id=f"notes-task-server-{identity_hash[:32]}",
        object_revision=revision,
        base_object_revision=base_revision,
        base_object_hash=base_hash,
    )
    activity = build_task_activity_capture(
        db=db,
        owner_user_id=owner,
        dataset_id=dataset,
        actor=actor,
        before=before,
        after=after,
        source_kind=source_kind,
    )
    return NotesTaskCaptureMutation(
        owner_user_id=owner,
        dataset_id=dataset,
        actor=actor,
        operation=operation,
        before=dict(before_row) if before_row is not None else None,
        after=dict(after_row),
        base_revision=base_revision,
        base_hash=base_hash,
        restore_intent=restore_intent,
        idempotency_key=f"notes-task-capture-{identity_hash}",
        step=step,
        activity=activity,
    )


def resolve_task_compatibility_scope(
    db: CharactersRAGDB,
    *,
    authenticated_owner_user_id: str,
) -> TaskStoreScope:
    """Resolve product-owned task scope without accepting a client dataset selector."""
    owner = str(authenticated_owner_user_id).strip()
    if not owner:
        raise InputError("Task owner cannot be empty.")  # noqa: TRY003
    if owner != str(db.client_id):
        raise ConflictError("Task scope is unavailable.", entity="tasks", entity_id=owner)  # noqa: TRY003
    dataset = db.resolve_task_compatibility_dataset_id(owner_user_id=owner)  # type: ignore[attr-defined]
    return TaskStoreScope(owner_user_id=owner, dataset_id=dataset)


class NotesTaskService:
    """Coordinate task-backed checklist reconciliation for saved notes."""

    def __init__(
        self,
        reconciler: NotesTaskReconciler | None = None,
        *,
        task_capture_callback: NotesTaskCaptureCallback | None = None,
        task_coordinator_resolver: NotesTaskCoordinatorResolver | None = None,
    ) -> None:
        self._reconciler = reconciler or NotesTaskReconciler()
        self._task_capture_callback = task_capture_callback
        self._task_coordinator_resolver = task_coordinator_resolver

    def _active_task_coordinator(
        self,
        *,
        scope: TaskStoreScope,
    ) -> NotesTaskCoordinator | None:
        """Resolve owner-bound task authority at the public mutation boundary."""

        resolver = self._task_coordinator_resolver
        if resolver is None:
            from tldw_Server_API.app.core.Sync.v2.notes_task_coordinator import (
                resolve_notes_task_coordinator,
            )

            resolver = resolve_notes_task_coordinator
        return resolver(
            user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
        )

    @staticmethod
    def _task_source(actor: TaskActor) -> tuple[str, TaskActivitySource]:
        """Return the stable batch and activity provenance for one public write."""

        if actor.tool_name:
            return "notes.tasks.mcp", "mcp"
        return "notes.tasks.rest", "rest"

    @staticmethod
    def _planned_metadata_task(
        *,
        db: CharactersRAGDB,
        task: dict[str, Any],
        metadata: dict[str, Any],
        owner_user_id: str,
        occurred_at: str,
    ) -> dict[str, Any]:
        """Build the canonical post-state without mutating product storage."""

        source = db.task_store._sync_bootstrap_task_row(task, owner_user_id)
        raw_payload = dict(source["sync_payload"])
        for key in _METADATA_TOKEN_ORDER:
            raw_payload[key] = metadata.get(key)
        payload = parse_notes_task_v1(raw_payload, owner_user_id=owner_user_id)
        revision = int(source["canonical_revision"]) + 1
        return {
            **task,
            "metadata_json": db.task_store._sync_task_metadata(payload),
            "updated_at": occurred_at,
            "version": int(task["version"]) + 1,
            "canonical_revision": revision,
            "canonical_hash": notes_task_object_hash(
                payload,
                revision=revision,
                deleted=False,
            ),
            "source_diagnostic_code": None,
            "source_diagnostic_hash": None,
        }

    @staticmethod
    def _planned_projected_task(
        *,
        db: CharactersRAGDB,
        task: dict[str, Any],
        owner_user_id: str,
        occurred_at: str,
        text: str | None,
        status: str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build a projected task post-state without touching product storage."""

        source = db.task_store._sync_bootstrap_task_row(task, owner_user_id)
        raw_payload = dict(source["sync_payload"])
        if text is not None:
            raw_payload["title"] = text.strip()
        if status is not None:
            raw_payload["status"] = status
            if status == "done" and source["sync_payload"]["status"] != "done":
                raw_payload["completed_at"] = occurred_at
            elif status == "open":
                raw_payload["completed_at"] = None
        if metadata is not None:
            for key in _METADATA_TOKEN_ORDER:
                raw_payload[key] = metadata.get(key)
        payload = parse_notes_task_v1(raw_payload, owner_user_id=owner_user_id)
        revision = int(source["canonical_revision"]) + 1
        return {
            **task,
            "text": payload.title,
            "status": payload.status,
            "metadata_json": db.task_store._sync_task_metadata(payload),
            "completed_at": payload.completed_at,
            "updated_at": occurred_at,
            "version": int(task["version"]) + 1,
            "canonical_revision": revision,
            "canonical_hash": notes_task_object_hash(
                payload,
                revision=revision,
                deleted=False,
            ),
            "source_diagnostic_code": None,
            "source_diagnostic_hash": None,
        }

    @staticmethod
    def _note_projection_step(
        *,
        coordinator: NotesTaskCoordinator,
        note: dict[str, Any],
        content: str,
    ) -> ServerOriginMutationStep:
        """Build the next note envelope against its exact Sync head."""

        if coordinator.dataset_id is None:
            raise ConflictError(
                "Task coordinator dataset is unavailable.",
                entity="notes",
                entity_id=str(note["id"]),
            )
        head = coordinator.service.store.get_current_head(
            coordinator.dataset_id,
            "notes.note",
            str(note["id"]),
        )
        if head is None or head.object_revision is None:
            raise ConflictError(
                "Task note has no synchronized base.",
                entity="notes",
                entity_id=str(note["id"]),
            )
        return ServerOriginMutationStep(
            domain="notes.note",
            operation="upsert",
            object_id=str(note["id"]),
            payload={
                "title": str(note.get("title") or ""),
                "content": content,
                "conversation_id": note.get("conversation_id"),
                "message_id": note.get("message_id"),
            },
            object_revision=int(head.object_revision) + 1,
        )

    def _create_task_through_sync(
        self,
        *,
        db: CharactersRAGDB,
        scope: TaskStoreScope,
        coordinator: NotesTaskCoordinator,
        note_id: str,
        text: str,
        status: str,
        metadata: dict[str, Any],
        expected_note_version: int,
        actor: TaskActor,
    ) -> dict[str, Any]:
        """Append and materialize one projected task creation group."""

        note = self._require_note_version(
            db,
            note_id=note_id,
            expected_note_version=expected_note_version,
        )
        occurred_at = normalize_sync_timestamp(coordinator.service.clock())
        if occurred_at is None:
            raise ConflictError(
                "Task mutation timestamp is unavailable.",
                entity="tasks",
                entity_id=note_id,
            )
        identity_hash = hashlib.sha256(
            json.dumps(
                [
                    scope.owner_user_id,
                    scope.dataset_id,
                    note_id,
                    expected_note_version,
                    text.strip(),
                    status,
                    metadata,
                    actor.actor_type,
                    actor.actor_id,
                    actor.tool_name,
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        task_id = str(UUID(bytes=bytes.fromhex(identity_hash[:32]), version=4))
        payload = parse_notes_task_v1(
            {
                "task_id": task_id,
                "note_id": note_id,
                "title": text.strip(),
                "description": None,
                "status": status,
                "completed_at": occurred_at if status == "done" else None,
                "priority": metadata.get("priority"),
                "due_date": metadata.get("due_date"),
                "estimate": metadata.get("estimate"),
                "recurrence": None,
                "assignee_id": None,
                "tags": [],
                "custom": {},
            },
            owner_user_id=scope.owner_user_id,
        )
        canonical_hash = notes_task_object_hash(
            payload,
            revision=1,
            deleted=False,
        )
        after = {
            "owner_user_id": scope.owner_user_id,
            "dataset_id": scope.dataset_id,
            "id": task_id,
            "note_id": note_id,
            "text": payload.title,
            "status": payload.status,
            "metadata_json": db.task_store._sync_task_metadata(payload),
            "projection_status": "live",
            "deleted": False,
            "created_at": occurred_at,
            "updated_at": occurred_at,
            "completed_at": payload.completed_at,
            "client_id": db.client_id,
            "version": 1,
            "canonical_revision": 1,
            "canonical_hash": canonical_hash,
            "source_diagnostic_code": None,
            "source_diagnostic_hash": None,
        }
        marker = "x" if status == "done" else " "
        line = f"- [{marker}] {self._render_body(text=payload.title, metadata=metadata)}"
        line += " " + render_task_marker(
            task_id,
            revision=1,
            object_hash=canonical_hash,
        )
        note_step = self._note_projection_step(
            coordinator=coordinator,
            note=note,
            content=self._append_checklist_line(str(note.get("content") or ""), line),
        )
        source, source_kind = self._task_source(actor)
        mutation = build_task_capture_mutation(
            db=db,
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            actor=actor,
            before=None,
            after=after,
            source_kind=source_kind,
        )
        result = coordinator.capture(
            coordinator.plan_task_mutation(mutation, note_step=note_step),
            source=source,
        )
        if not result.fully_applied:
            raise ConflictError(
                "Task projection is incomplete.",
                entity="tasks",
                entity_id=task_id,
            )
        updated_note = self._require_note(db, note_id)
        matches = [
            item
            for item in parse_note_checklists(
                note_id=note_id,
                note_version=int(updated_note["version"]),
                content=str(updated_note.get("content") or ""),
            ).items
            if item.marker is not None and item.marker.task_id == task_id
        ]
        if len(matches) != 1:
            raise ConflictError(
                "Created task projection is unavailable.",
                entity="tasks",
                entity_id=task_id,
            )
        item = matches[0]
        db.set_task_projection(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            task_id=task_id,
            note_id=note_id,
            note_version=int(updated_note["version"]),
            line_number=item.locator.line_number,
            start_offset=item.locator.start_offset,
            end_offset=item.locator.end_offset,
            normalized_text_hash=item.locator.normalized_text_hash,
            occurrence_index=item.locator.occurrence_index,
            block_fingerprint=item.locator.block_fingerprint,
            raw_line=item.raw_line,
            has_child_content=item.has_child_content,
        )
        created = db.get_task(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            task_id=task_id,
        )
        if created is None:
            raise ConflictError(
                "Created task was not found.",
                entity="tasks",
                entity_id=task_id,
            )
        return created

    def _update_projected_task_through_sync(
        self,
        *,
        db: CharactersRAGDB,
        scope: TaskStoreScope,
        coordinator: NotesTaskCoordinator,
        task: dict[str, Any],
        expected_note_version: int | None,
        actor: TaskActor,
        text: str | None,
        status: str | None,
        metadata: dict[str, Any] | None,
        record_only: bool,
    ) -> dict[str, Any]:
        """Append a task/activity/note update before projecting product state."""

        task_id = str(task["id"])
        if record_only:
            raise ConflictError(
                f"Task '{task_id}' is projected into a note and cannot be updated record-only.",
                entity="tasks",
                entity_id=task_id,
            )
        if expected_note_version is None:
            raise InputError("expected_note_version is required for projected task updates.")
        projection = self._require_projection(
            db,
            task_id=task_id,
            scope=scope,
            conn=None,
        )
        note = self._require_note_version(
            db,
            note_id=str(task["note_id"]),
            expected_note_version=expected_note_version,
        )
        self._require_projection_version(projection, expected_note_version, task_id)
        parsed_item = self._find_projected_item(
            note_id=str(note["id"]),
            note_version=int(note["version"]),
            content=str(note.get("content") or ""),
            projection=projection,
            task_id=task_id,
        )
        occurred_at = normalize_sync_timestamp(coordinator.service.clock())
        if occurred_at is None:
            raise ConflictError(
                "Task mutation timestamp is unavailable.",
                entity="tasks",
                entity_id=task_id,
            )
        after = self._planned_projected_task(
            db=db,
            task=task,
            owner_user_id=scope.owner_user_id,
            occurred_at=occurred_at,
            text=text,
            status=status,
            metadata=metadata,
        )
        new_line = self._rewrite_line(
            raw_line=parsed_item.raw_line,
            checked=str(after["status"]) == "done",
            text=str(after["text"]),
            metadata={
                key: after["metadata_json"].get(key)
                for key in _METADATA_TOKEN_ORDER
                if after["metadata_json"].get(key) is not None
            },
            preserve_existing_body=False,
        )
        new_line += " " + render_task_marker(
            task_id,
            revision=int(after["canonical_revision"]),
            object_hash=str(after["canonical_hash"]),
        )
        new_content = self._replace_projection_line(
            content=str(note.get("content") or ""),
            projection=projection,
            new_line=new_line,
        )
        source, source_kind = self._task_source(actor)
        mutation = build_task_capture_mutation(
            db=db,
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            actor=actor,
            before=task,
            after=after,
            source_kind=source_kind,
        )
        note_step = self._note_projection_step(
            coordinator=coordinator,
            note=note,
            content=new_content,
        )
        result = coordinator.capture(
            coordinator.plan_task_mutation(mutation, note_step=note_step),
            source=source,
        )
        if not result.fully_applied:
            raise ConflictError(
                "Task projection is incomplete.",
                entity="tasks",
                entity_id=task_id,
            )
        updated_note = self._require_note(db, str(note["id"]))
        updated_matches = [
            item
            for item in parse_note_checklists(
            note_id=str(note["id"]),
            note_version=int(updated_note["version"]),
            content=str(updated_note.get("content") or ""),
            ).items
            if item.marker is not None and item.marker.task_id == task_id
        ]
        if len(updated_matches) != 1:
            raise ConflictError(
                "Updated task projection is unavailable.",
                entity="tasks",
                entity_id=task_id,
            )
        updated_item = updated_matches[0]
        db.set_task_projection(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            task_id=task_id,
            note_id=str(note["id"]),
            note_version=int(updated_note["version"]),
            line_number=updated_item.locator.line_number,
            start_offset=updated_item.locator.start_offset,
            end_offset=updated_item.locator.end_offset,
            normalized_text_hash=updated_item.locator.normalized_text_hash,
            occurrence_index=updated_item.locator.occurrence_index,
            block_fingerprint=updated_item.locator.block_fingerprint,
            raw_line=updated_item.raw_line,
            has_child_content=updated_item.has_child_content,
        )
        updated = db.get_task(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            task_id=task_id,
        )
        if updated is None:
            raise ConflictError(
                "Updated task was not found.",
                entity="tasks",
                entity_id=task_id,
            )
        return updated

    def _delete_task_through_sync(
        self,
        *,
        db: CharactersRAGDB,
        scope: TaskStoreScope,
        coordinator: NotesTaskCoordinator,
        task_id: str,
        expected_task_version: int,
        expected_note_version: int | None,
        record_only: bool,
        actor: TaskActor,
    ) -> dict[str, Any]:
        """Append a complete task deletion before product materialization."""

        task = self._require_task_version(
            db,
            task_id=task_id,
            expected_task_version=expected_task_version,
            scope=scope,
            conn=None,
        )
        projection_status = str(task["projection_status"])
        if projection_status == "ambiguous":
            raise ConflictError(
                f"Task projection is ambiguous for task '{task_id}'.",
                entity="tasks",
                entity_id=task_id,
            )
        note_step: ServerOriginMutationStep | None = None
        if projection_status == "unlinked":
            if not record_only:
                raise ConflictError(
                    f"Task projection is unlinked for task '{task_id}'. Record-only delete mode is required.",
                    entity="tasks",
                    entity_id=task_id,
                )
        elif projection_status == "live":
            if record_only:
                raise ConflictError(
                    f"Task '{task_id}' is projected into a note and cannot be deleted record-only.",
                    entity="tasks",
                    entity_id=task_id,
                )
            if expected_note_version is None:
                raise InputError(
                    "expected_note_version is required for projected task deletion."
                )
            projection = self._require_projection(
                db,
                task_id=task_id,
                scope=scope,
                conn=None,
            )
            note = self._require_note_version(
                db,
                note_id=str(task["note_id"]),
                expected_note_version=expected_note_version,
            )
            self._require_projection_version(
                projection,
                expected_note_version,
                task_id,
            )
            parsed_item = self._find_projected_item(
                note_id=str(note["id"]),
                note_version=int(note["version"]),
                content=str(note.get("content") or ""),
                projection=projection,
                task_id=task_id,
            )
            if parsed_item.has_child_content:
                raise ConflictError(
                    f"Task '{task_id}' has nested child content and cannot be deleted by default.",
                    entity="tasks",
                    entity_id=task_id,
                )
            new_content = self._delete_projection_line(
                str(note.get("content") or ""),
                projection,
            )
            if not new_content:
                new_content = "\n"
            note_step = self._note_projection_step(
                coordinator=coordinator,
                note=note,
                content=new_content,
            )
        else:
            raise ConflictError(
                f"Task projection is {projection_status} for task '{task_id}'.",
                entity="tasks",
                entity_id=task_id,
            )
        source_row = db.task_store._sync_bootstrap_task_row(
            task,
            scope.owner_user_id,
        )
        occurred_at = normalize_sync_timestamp(coordinator.service.clock())
        if occurred_at is None:
            raise ConflictError(
                "Task mutation timestamp is unavailable.",
                entity="tasks",
                entity_id=task_id,
            )
        revision = int(source_row["canonical_revision"]) + 1
        payload = parse_notes_task_v1(
            source_row["sync_payload"],
            owner_user_id=scope.owner_user_id,
        )
        after = {
            **task,
            "deleted": True,
            "projection_status": "deleted",
            "updated_at": occurred_at,
            "version": int(task["version"]) + 1,
            "canonical_revision": revision,
            "canonical_hash": notes_task_object_hash(
                payload,
                revision=revision,
                deleted=True,
            ),
            "source_diagnostic_code": None,
            "source_diagnostic_hash": None,
        }
        source, source_kind = self._task_source(actor)
        mutation = build_task_capture_mutation(
            db=db,
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            actor=actor,
            before=task,
            after=after,
            source_kind=source_kind,
        )
        result = coordinator.capture(
            coordinator.plan_task_mutation(mutation, note_step=note_step),
            source=source,
        )
        if not result.fully_applied:
            raise ConflictError(
                "Task projection is incomplete.",
                entity="tasks",
                entity_id=task_id,
            )
        deleted = db.task_store.get_task(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            task_id=task_id,
            include_deleted=True,
        )
        if deleted is None or not bool(deleted["deleted"]):
            raise ConflictError(
                "Deleted task was not found.",
                entity="tasks",
                entity_id=task_id,
            )
        return deleted

    def _update_unlinked_metadata_through_sync(
        self,
        *,
        db: CharactersRAGDB,
        scope: TaskStoreScope,
        coordinator: NotesTaskCoordinator,
        task_id: str,
        expected_task_version: int,
        actor: TaskActor,
        metadata: dict[str, Any] | None,
        text: str | None,
        status: str | None,
        record_only: bool,
    ) -> dict[str, Any]:
        """Append an unlinked metadata transition before product materialization."""

        task = self._require_task_version(
            db,
            task_id=task_id,
            expected_task_version=expected_task_version,
            scope=scope,
            conn=None,
        )
        projection_status = str(task["projection_status"])
        if projection_status != "unlinked":
            raise ConflictError(
                f"Task projection is {projection_status} for task '{task_id}'.",
                entity="tasks",
                entity_id=task_id,
            )
        if not record_only or text is not None or status is not None:
            raise ConflictError(
                f"Task projection is unlinked for task '{task_id}'.",
                entity="tasks",
                entity_id=task_id,
            )
        if metadata is None:
            return task
        occurred_at = normalize_sync_timestamp(coordinator.service.clock())
        if occurred_at is None:
            raise ConflictError(
                "Task mutation timestamp is unavailable.",
                entity="tasks",
                entity_id=task_id,
            )
        after = self._planned_metadata_task(
            db=db,
            task=task,
            metadata=metadata,
            owner_user_id=scope.owner_user_id,
            occurred_at=occurred_at,
        )
        source, source_kind = self._task_source(actor)
        mutation = build_task_capture_mutation(
            db=db,
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            actor=actor,
            before=task,
            after=after,
            source_kind=source_kind,
        )
        result = coordinator.capture(
            coordinator.plan_task_mutation(mutation),
            source=source,
        )
        if not result.fully_applied:
            raise ConflictError(
                "Task projection is incomplete.",
                entity="tasks",
                entity_id=task_id,
            )
        updated = db.get_task(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            task_id=task_id,
        )
        if updated is None:
            raise ConflictError(
                "Updated task was not found.",
                entity="tasks",
                entity_id=task_id,
            )
        return updated

    def _capture_task_mutation(
        self,
        *,
        db: CharactersRAGDB,
        scope: TaskStoreScope,
        actor: TaskActor,
        before: dict[str, Any] | None,
        after: dict[str, Any],
        conn: TaskConnection | None,
    ) -> None:
        if self._task_capture_callback is None:
            return
        self._task_capture_callback(
            build_task_capture_mutation(
                db=db,
                owner_user_id=scope.owner_user_id,
                dataset_id=scope.dataset_id,
                actor=actor,
                before=before,
                after=after,
            ),
            conn=conn,
        )

    @staticmethod
    def _internal_reconciliation_actor(actor: TaskActor) -> TaskActor:
        return TaskActor(
            actor_type=actor.actor_type,
            actor_id=actor.actor_id,
            tool_name="notes.tasks.reconciliation",
        )

    @staticmethod
    def _projection_line_for_payload(
        *,
        item: ParsedChecklistItem,
        task_id: str,
        revision: int,
        object_hash: str,
        payload: Any,
    ) -> str:
        """Render one canonical managed line while preserving its list indentation."""

        metadata = {
            key: getattr(payload, key)
            for key in _METADATA_TOKEN_ORDER
            if getattr(payload, key) is not None
        }
        parsed_line = _parse_checklist_line(item.raw_line)
        if parsed_line is None:
            raise ConflictError(
                "Task projection line is no longer a checklist item.",
                entity="tasks",
                entity_id=task_id,
            )
        checked = "x" if payload.status == "done" else " "
        body = NotesTaskService._render_body(
            text=payload.title,
            metadata=metadata,
        )
        return (
            f"{parsed_line.indent}{parsed_line.bullet}{parsed_line.space}"
            f"[{checked}] {body} "
            + render_task_marker(
                task_id,
                revision=revision,
                object_hash=object_hash,
            )
        )

    @staticmethod
    def _apply_line_replacements(
        content: str,
        replacements: list[tuple[int, int, str]],
    ) -> str:
        """Apply already-validated line rewrites from the end of the note."""

        updated = content
        for start, end, line in sorted(replacements, reverse=True):
            updated = f"{updated[:start]}{line}{updated[end:]}"
        return updated

    @staticmethod
    def _record_projection_drift(
        *,
        db: CharactersRAGDB,
        scope: TaskStoreScope,
        coordinator: NotesTaskCoordinator,
        note_id: str,
        task_id: str,
        marker_revision: int,
        marker_hash: str,
        reason_code: str,
    ) -> dict[str, Any]:
        """Persist one deterministic privacy-safe projection drift claim."""

        note_head = coordinator.service.store.get_current_head(
            scope.dataset_id,
            "notes.note",
            note_id,
        )
        task_head = coordinator.service.store.get_current_head(
            scope.dataset_id,
            "notes.task",
            task_id,
        )
        note_claim = (
            (note_head.server_cursor, note_head.payload_hash)
            if note_head is not None
            else (None, None)
        )
        task_claim = (
            (task_head.server_cursor, task_head.payload_hash)
            if task_head is not None
            else (None, None)
        )
        identity = hashlib.sha256(
            json.dumps(
                [
                    scope.owner_user_id,
                    scope.dataset_id,
                    note_id,
                    task_id,
                    marker_revision,
                    marker_hash,
                    *note_claim,
                    *task_claim,
                    reason_code,
                ],
                separators=(",", ":"),
            ).encode("utf-8")
        ).digest()
        drift_id = str(UUID(bytes=identity[:16], version=4))
        return db.task_store.create_task_projection_drift(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            drift_id=drift_id,
            note_id=note_id,
            task_id=task_id,
            marker_base_revision=marker_revision,
            marker_base_hash=marker_hash,
            note_head_cursor=note_claim[0],
            note_head_hash=note_claim[1],
            task_head_cursor=task_claim[0],
            task_head_hash=task_claim[1],
            reason_code=reason_code,
        )

    def _reconcile_managed_note_through_sync(
        self,
        *,
        db: CharactersRAGDB,
        scope: TaskStoreScope,
        coordinator: NotesTaskCoordinator,
        note_id: str,
        note_version: int,
        content: str,
        actor: TaskActor,
    ) -> ReconciliationResult:
        """Converge only marker-authorized checklist projections through Sync."""

        from tldw_Server_API.app.core.Sync.v2.notes_task_coordinator import (
            _projection_anchor_from_envelope,
            _same_projection_group,
        )

        note = self._require_note_version(
            db,
            note_id=note_id,
            expected_note_version=note_version,
        )
        if str(note.get("content") or "") != content:
            raise ConflictError(
                "Note content changed before task reconciliation.",
                entity="notes",
                entity_id=note_id,
            )
        parsed = parse_note_checklists(
            note_id=note_id,
            note_version=note_version,
            content=content,
        )
        marker_counts: dict[str, int] = {}
        for item in parsed.items:
            if item.marker is not None:
                marker_counts[item.marker.task_id] = (
                    marker_counts.get(item.marker.task_id, 0) + 1
                )

        occurred_at = normalize_sync_timestamp(coordinator.service.clock())
        if occurred_at is None:
            raise ConflictError(
                "Task mutation timestamp is unavailable.",
                entity="tasks",
                entity_id=note_id,
            )
        mutations: list[NotesTaskCaptureMutation] = []
        replacements: list[tuple[int, int, str]] = []
        matched_task_ids: set[str] = set()
        updated_task_ids: list[str] = []
        unlinked_task_ids: list[str] = []
        drift_count = 0

        for item in parsed.items:
            marker = item.marker
            if marker is None:
                continue
            task_id = marker.task_id
            matched_task_ids.add(task_id)
            task = db.task_store.get_task(
                owner_user_id=scope.owner_user_id,
                dataset_id=scope.dataset_id,
                task_id=task_id,
                include_deleted=True,
            )
            reason_code: str | None = None
            historical = None
            anchor = None
            if marker_counts[task_id] != 1:
                reason_code = "duplicate_marker"
            elif (
                task is None
                or bool(task["deleted"])
                or str(task["note_id"]) != note_id
                or str(task["projection_status"]) != "live"
            ):
                reason_code = "marker_scope_mismatch"
            else:
                historical = coordinator.service.store.get_historical_task_envelope(
                    owner_user_id=scope.owner_user_id,
                    dataset_id=scope.dataset_id,
                    task_id=task_id,
                    object_revision=marker.revision,
                    object_hash=marker.object_hash,
                )
                anchor = (
                    _projection_anchor_from_envelope(historical)
                    if historical is not None
                    else None
                )
                anchor_note = (
                    coordinator.service.store.get_projection_note_envelope(
                        owner_user_id=scope.owner_user_id,
                        dataset_id=scope.dataset_id,
                        note_id=note_id,
                        envelope_id=anchor.note_envelope_id,
                        object_hash=anchor.note_hash,
                    )
                    if anchor is not None
                    else None
                )
                if historical is None or anchor is None or anchor_note is None:
                    reason_code = "base_unavailable"
                elif (
                    not anchor.linked
                    or historical.parent_id != note_id
                    or not _same_projection_group(historical, anchor_note, anchor)
                ):
                    reason_code = "marker_scope_mismatch"
            if reason_code is not None:
                if task is not None and str(task.get("note_id")) == note_id:
                    self._record_projection_drift(
                        db=db,
                        scope=scope,
                        coordinator=coordinator,
                        note_id=note_id,
                        task_id=task_id,
                        marker_revision=marker.revision,
                        marker_hash=marker.object_hash,
                        reason_code=reason_code,
                    )
                    drift_count += 1
                continue

            assert task is not None and historical is not None
            current = db.task_store._sync_bootstrap_task_row(
                task,
                scope.owner_user_id,
            )
            decision = classify_managed_projection(
                item=item,
                anchor_revision=marker.revision,
                anchor_hash=marker.object_hash,
                anchor_payload=historical.payload,
                current_revision=int(current["canonical_revision"]),
                current_hash=str(current["canonical_hash"]),
                current_payload=current["sync_payload"],
            )
            if decision == "drift":
                self._record_projection_drift(
                    db=db,
                    scope=scope,
                    coordinator=coordinator,
                    note_id=note_id,
                    task_id=task_id,
                    marker_revision=marker.revision,
                    marker_hash=marker.object_hash,
                    reason_code="both_changed",
                )
                drift_count += 1
                continue
            if decision == "no_change":
                continue
            if decision == "note_to_task":
                after = self._planned_projected_task(
                    db=db,
                    task=task,
                    owner_user_id=scope.owner_user_id,
                    occurred_at=occurred_at,
                    text=item.text,
                    status="done" if item.checked else "open",
                    metadata=item.metadata,
                )
                mutation = build_task_capture_mutation(
                    db=db,
                    owner_user_id=scope.owner_user_id,
                    dataset_id=scope.dataset_id,
                    actor=actor,
                    before=task,
                    after=after,
                    source_kind="markdown_reconciliation",
                )
                mutations.append(mutation)
                updated_task_ids.append(task_id)
                payload = parse_notes_task_v1(
                    db.task_store._sync_bootstrap_task_row(
                        after,
                        scope.owner_user_id,
                    )["sync_payload"],
                    owner_user_id=scope.owner_user_id,
                )
                revision = int(after["canonical_revision"])
                object_hash = str(after["canonical_hash"])
            else:
                payload = parse_notes_task_v1(
                    current["sync_payload"],
                    owner_user_id=scope.owner_user_id,
                )
                revision = int(current["canonical_revision"])
                object_hash = str(current["canonical_hash"])
            replacements.append(
                (
                    item.locator.start_offset,
                    item.locator.end_offset,
                    self._projection_line_for_payload(
                        item=item,
                        task_id=task_id,
                        revision=revision,
                        object_hash=object_hash,
                        payload=payload,
                    ),
                )
            )

        live_pairs = db.task_store.list_live_projected_tasks(
            note_id=note_id,
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
        )
        for pair in live_pairs:
            task = pair["task"]
            projection = pair["projection"]
            task_id = str(task["id"])
            if task_id in matched_task_ids:
                continue
            possible_line = next(
                (
                    item
                    for item in parsed.items
                    if item.locator.normalized_text_hash
                    == projection["normalized_text_hash"]
                ),
                None,
            )
            old_item = parse_note_checklists(
                note_id=note_id,
                note_version=int(projection["note_version"]),
                content=str(projection["raw_line"]),
            ).items
            old_marker = old_item[0].marker if old_item else None
            if possible_line is not None and old_marker is not None:
                self._record_projection_drift(
                    db=db,
                    scope=scope,
                    coordinator=coordinator,
                    note_id=note_id,
                    task_id=task_id,
                    marker_revision=old_marker.revision,
                    marker_hash=old_marker.object_hash,
                    reason_code=(
                        possible_line.marker_reason_code or "missing_marker_base"
                    ),
                )
                drift_count += 1
                continue
            source_row = db.task_store._sync_bootstrap_task_row(
                task,
                scope.owner_user_id,
            )
            revision = int(source_row["canonical_revision"]) + 1
            payload = parse_notes_task_v1(
                source_row["sync_payload"],
                owner_user_id=scope.owner_user_id,
            )
            after = {
                **task,
                "projection_status": "unlinked",
                "updated_at": occurred_at,
                "version": int(task["version"]) + 1,
                "canonical_revision": revision,
                "canonical_hash": notes_task_object_hash(
                    payload,
                    revision=revision,
                    deleted=False,
                ),
                "source_diagnostic_code": None,
                "source_diagnostic_hash": None,
            }
            mutations.append(
                build_task_capture_mutation(
                    db=db,
                    owner_user_id=scope.owner_user_id,
                    dataset_id=scope.dataset_id,
                    actor=actor,
                    before=task,
                    after=after,
                    source_kind="markdown_reconciliation",
                )
            )
            unlinked_task_ids.append(task_id)

        new_content = self._apply_line_replacements(content, replacements)
        if mutations:
            note_step = self._note_projection_step(
                coordinator=coordinator,
                note=note,
                content=new_content or "\n",
            )
            reconciliation_key = hashlib.sha256(
                json.dumps(
                    [
                        note_id,
                        note_version,
                        *(mutation.idempotency_key for mutation in mutations),
                        hashlib.sha256(new_content.encode("utf-8")).hexdigest(),
                    ],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            result = coordinator.capture(
                coordinator.plan_note_reconciliation(
                    mutations,
                    note_step=note_step,
                    idempotency_key=f"notes-task-reconcile-{reconciliation_key}",
                ),
                source="notes.tasks.reconciliation",
            )
            if not result.fully_applied:
                raise ConflictError(
                    "Task reconciliation is incomplete.",
                    entity="notes",
                    entity_id=note_id,
                )
        elif new_content != content:
            note_step = self._note_projection_step(
                coordinator=coordinator,
                note=note,
                content=new_content or "\n",
            )
            result = coordinator.capture_note_projection(
                note_step,
                idempotency_key=(
                    "notes-task-note-reconcile-"
                    + hashlib.sha256(
                        json.dumps(
                            [note_id, note_version, new_content],
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                ),
            )
            if not result.fully_applied:
                raise ConflictError(
                    "Task note projection is incomplete.",
                    entity="notes",
                    entity_id=note_id,
                )

        final_note = self._require_note(db, note_id)
        final_parsed = parse_note_checklists(
            note_id=note_id,
            note_version=int(final_note["version"]),
            content=str(final_note.get("content") or ""),
        )
        final_matched: list[str] = []
        for item in final_parsed.items:
            if item.marker is None:
                continue
            task = db.get_task(
                owner_user_id=scope.owner_user_id,
                dataset_id=scope.dataset_id,
                task_id=item.marker.task_id,
            )
            if (
                task is None
                or str(task["note_id"]) != note_id
                or str(task["projection_status"]) != "live"
                or int(task["canonical_revision"]) != item.marker.revision
                or str(task["canonical_hash"]) != item.marker.object_hash
            ):
                continue
            db.set_task_projection(
                owner_user_id=scope.owner_user_id,
                dataset_id=scope.dataset_id,
                task_id=str(task["id"]),
                note_id=note_id,
                note_version=int(final_note["version"]),
                line_number=item.locator.line_number,
                start_offset=item.locator.start_offset,
                end_offset=item.locator.end_offset,
                normalized_text_hash=item.locator.normalized_text_hash,
                occurrence_index=item.locator.occurrence_index,
                block_fingerprint=item.locator.block_fingerprint,
                raw_line=item.raw_line,
                has_child_content=item.has_child_content,
            )
            final_matched.append(str(task["id"]))

        parser_warnings = sum(len(item.warnings) for item in final_parsed.items)
        warning_count = parser_warnings + drift_count
        db.set_reconciliation_state(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            note_id=note_id,
            note_version=int(final_note["version"]),
            status="clean" if warning_count == 0 else "warnings",
            item_count=len(final_parsed.items),
            warning_count=warning_count,
        )
        return ReconciliationResult(
            note_id=note_id,
            note_version=int(final_note["version"]),
            parsed_count=len(final_parsed.items),
            updated_count=len(updated_task_ids),
            unlinked_count=len(unlinked_task_ids),
            matched_task_ids=final_matched,
            unlinked_task_ids=unlinked_task_ids,
            warning_count=warning_count,
        )

    def resolve_projection_drift(
        self,
        *,
        db: CharactersRAGDB,
        note_id: str,
        task_id: str,
        drift_id: str,
        action: str,
        expected_lifecycle_revision: int,
        expected_note_head_cursor: int | None,
        expected_note_head_hash: str | None,
        expected_task_head_cursor: int | None,
        expected_task_head_hash: str | None,
        actor: TaskActor,
        owner_user_id: str | None = None,
    ) -> dict[str, Any]:
        """Resolve one exact open projection drift after validating every claim."""

        from tldw_Server_API.app.core.Sync.v2.notes_task_coordinator import (
            _projection_anchor_from_envelope,
            _same_projection_group,
        )

        if action not in {"keep_task", "accept_markdown", "unlink", "dismiss"}:
            raise InputError("Unsupported projection drift resolution action.")
        if expected_lifecycle_revision != 1:
            raise ConflictError(
                "Projection drift changed concurrently.",
                entity="tasks",
                entity_id=drift_id,
            )
        scope = resolve_task_compatibility_scope(
            db,
            authenticated_owner_user_id=owner_user_id or db.client_id,
        )
        coordinator = self._active_task_coordinator(scope=scope)
        if coordinator is None:
            raise ConflictError(
                "Task Sync is not active for projection drift resolution.",
                entity="tasks",
                entity_id=drift_id,
            )
        drift = db.task_store.get_task_projection_drift(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            note_id=note_id,
            task_id=task_id,
            drift_id=drift_id,
        )
        expected_note = (expected_note_head_cursor, expected_note_head_hash)
        expected_task = (expected_task_head_cursor, expected_task_head_hash)
        if (
            drift is None
            or drift["status"] != "open"
            or (drift["note_head_cursor"], drift["note_head_hash"])
            != expected_note
            or (drift["task_head_cursor"], drift["task_head_hash"])
            != expected_task
        ):
            raise ConflictError(
                "Projection drift changed concurrently.",
                entity="tasks",
                entity_id=drift_id,
            )
        note_head = coordinator.service.store.get_current_head(
            scope.dataset_id,
            "notes.note",
            note_id,
        )
        task_head = coordinator.service.store.get_current_head(
            scope.dataset_id,
            "notes.task",
            task_id,
        )
        current_note_claim = (
            (note_head.server_cursor, note_head.payload_hash)
            if note_head is not None
            else (None, None)
        )
        current_task_claim = (
            (task_head.server_cursor, task_head.payload_hash)
            if task_head is not None
            else (None, None)
        )
        if current_note_claim != expected_note or current_task_claim != expected_task:
            raise ConflictError(
                "Projection drift heads changed concurrently.",
                entity="tasks",
                entity_id=drift_id,
            )
        base_envelope = coordinator.service.store.get_historical_task_envelope(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            task_id=task_id,
            object_revision=int(drift["marker_base_revision"]),
            object_hash=str(drift["marker_base_hash"]),
        )
        anchor = (
            _projection_anchor_from_envelope(base_envelope)
            if base_envelope is not None
            else None
        )
        anchor_note = (
            coordinator.service.store.get_projection_note_envelope(
                owner_user_id=scope.owner_user_id,
                dataset_id=scope.dataset_id,
                note_id=note_id,
                envelope_id=anchor.note_envelope_id,
                object_hash=anchor.note_hash,
            )
            if anchor is not None
            else None
        )
        if (
            base_envelope is None
            or anchor is None
            or anchor_note is None
            or not _same_projection_group(base_envelope, anchor_note, anchor)
        ):
            raise ConflictError(
                "Projection drift anchor is unavailable.",
                entity="tasks",
                entity_id=drift_id,
            )
        if action == "dismiss":
            return db.task_store.compare_and_set_task_projection_drift(
                owner_user_id=scope.owner_user_id,
                dataset_id=scope.dataset_id,
                note_id=note_id,
                task_id=task_id,
                drift_id=drift_id,
                expected_note_head_cursor=expected_note_head_cursor,
                expected_note_head_hash=expected_note_head_hash,
                expected_task_head_cursor=expected_task_head_cursor,
                expected_task_head_hash=expected_task_head_hash,
                status="dismissed",
            )

        note = self._require_note(db, note_id)
        task = db.get_task(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            task_id=task_id,
        )
        if task is None or str(task["note_id"]) != note_id:
            raise ConflictError(
                "Projection drift task is unavailable.",
                entity="tasks",
                entity_id=drift_id,
            )
        parsed = parse_note_checklists(
            note_id=note_id,
            note_version=int(note["version"]),
            content=str(note.get("content") or ""),
        )
        base_matches = [
            item
            for item in parsed.items
            if item.marker is not None
            and item.marker.task_id == task_id
            and item.marker.revision == int(drift["marker_base_revision"])
            and item.marker.object_hash == str(drift["marker_base_hash"])
        ]
        if len(base_matches) != 1:
            raise ConflictError(
                "Projection drift Markdown claim changed concurrently.",
                entity="tasks",
                entity_id=drift_id,
            )
        item = base_matches[0]
        content = str(note.get("content") or "")
        occurred_at = normalize_sync_timestamp(coordinator.service.clock())
        if occurred_at is None:
            raise ConflictError(
                "Task mutation timestamp is unavailable.",
                entity="tasks",
                entity_id=drift_id,
            )
        if action == "keep_task":
            source_row = db.task_store._sync_bootstrap_task_row(
                task,
                scope.owner_user_id,
            )
            payload = parse_notes_task_v1(
                source_row["sync_payload"],
                owner_user_id=scope.owner_user_id,
            )
            new_line = self._projection_line_for_payload(
                item=item,
                task_id=task_id,
                revision=int(source_row["canonical_revision"]),
                object_hash=str(source_row["canonical_hash"]),
                payload=payload,
            )
            note_step = self._note_projection_step(
                coordinator=coordinator,
                note=note,
                content=self._replace_projection_line(
                    content=content,
                    projection={
                        "start_offset": item.locator.start_offset,
                        "end_offset": item.locator.end_offset,
                    },
                    new_line=new_line,
                ),
            )
            capture = coordinator.capture_note_projection(
                note_step,
                idempotency_key=f"notes-task-drift-{drift_id}-keep-task",
            )
        else:
            source_row = db.task_store._sync_bootstrap_task_row(
                task,
                scope.owner_user_id,
            )
            if action == "accept_markdown":
                after = self._planned_projected_task(
                    db=db,
                    task=task,
                    owner_user_id=scope.owner_user_id,
                    occurred_at=occurred_at,
                    text=item.text,
                    status="done" if item.checked else "open",
                    metadata=item.metadata,
                )
                after_payload = parse_notes_task_v1(
                    db.task_store._sync_bootstrap_task_row(
                        after,
                        scope.owner_user_id,
                    )["sync_payload"],
                    owner_user_id=scope.owner_user_id,
                )
                new_line = self._projection_line_for_payload(
                    item=item,
                    task_id=task_id,
                    revision=int(after["canonical_revision"]),
                    object_hash=str(after["canonical_hash"]),
                    payload=after_payload,
                )
            else:
                revision = int(source_row["canonical_revision"]) + 1
                payload = parse_notes_task_v1(
                    source_row["sync_payload"],
                    owner_user_id=scope.owner_user_id,
                )
                after = {
                    **task,
                    "projection_status": "unlinked",
                    "updated_at": occurred_at,
                    "version": int(task["version"]) + 1,
                    "canonical_revision": revision,
                    "canonical_hash": notes_task_object_hash(
                        payload,
                        revision=revision,
                        deleted=False,
                    ),
                    "source_diagnostic_code": None,
                    "source_diagnostic_hash": None,
                }
                marker_text = render_task_marker(
                    task_id,
                    revision=int(drift["marker_base_revision"]),
                    object_hash=str(drift["marker_base_hash"]),
                )
                new_line = item.raw_line.removesuffix(f" {marker_text}")
            mutation = build_task_capture_mutation(
                db=db,
                owner_user_id=scope.owner_user_id,
                dataset_id=scope.dataset_id,
                actor=actor,
                before=task,
                after=after,
                source_kind="repair",
            )
            note_step = self._note_projection_step(
                coordinator=coordinator,
                note=note,
                content=self._replace_projection_line(
                    content=content,
                    projection={
                        "start_offset": item.locator.start_offset,
                        "end_offset": item.locator.end_offset,
                    },
                    new_line=new_line,
                ),
            )
            capture = coordinator.capture(
                coordinator.plan_task_mutation(mutation, note_step=note_step),
                source="notes.tasks.repair",
            )
        if not capture.fully_applied:
            raise ConflictError(
                "Projection drift resolution is incomplete.",
                entity="tasks",
                entity_id=drift_id,
            )
        resolved = db.task_store.compare_and_set_task_projection_drift(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            note_id=note_id,
            task_id=task_id,
            drift_id=drift_id,
            expected_note_head_cursor=expected_note_head_cursor,
            expected_note_head_hash=expected_note_head_hash,
            expected_task_head_cursor=expected_task_head_cursor,
            expected_task_head_hash=expected_task_head_hash,
            status="resolved",
        )
        if action != "unlink":
            final_note = self._require_note(db, note_id)
            final_matches = [
                parsed_item
                for parsed_item in parse_note_checklists(
                    note_id=note_id,
                    note_version=int(final_note["version"]),
                    content=str(final_note.get("content") or ""),
                ).items
                if parsed_item.marker is not None
                and parsed_item.marker.task_id == task_id
            ]
            if len(final_matches) == 1:
                final_item = final_matches[0]
                db.set_task_projection(
                    owner_user_id=scope.owner_user_id,
                    dataset_id=scope.dataset_id,
                    task_id=task_id,
                    note_id=note_id,
                    note_version=int(final_note["version"]),
                    line_number=final_item.locator.line_number,
                    start_offset=final_item.locator.start_offset,
                    end_offset=final_item.locator.end_offset,
                    normalized_text_hash=final_item.locator.normalized_text_hash,
                    occurrence_index=final_item.locator.occurrence_index,
                    block_fingerprint=final_item.locator.block_fingerprint,
                    raw_line=final_item.raw_line,
                    has_child_content=final_item.has_child_content,
                )
        return resolved

    def reconcile_note(
        self,
        *,
        db: CharactersRAGDB,
        note_id: str,
        note_version: int,
        content: str,
        actor: TaskActor,
        owner_user_id: str | None = None,
    ) -> ReconciliationResult:
        scope = resolve_task_compatibility_scope(
            db,
            authenticated_owner_user_id=owner_user_id or db.client_id,
        )
        task_coordinator = self._active_task_coordinator(scope=scope)
        if task_coordinator is not None:
            return self._reconcile_managed_note_through_sync(
                db=db,
                scope=scope,
                coordinator=task_coordinator,
                note_id=note_id,
                note_version=note_version,
                content=content,
                actor=actor,
            )
        return self._reconciler.reconcile_note(
            db=db,
            note_id=note_id,
            note_version=note_version,
            content=content,
            actor=actor,
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
        )

    def reconcile_stale_notes(
        self,
        *,
        db: CharactersRAGDB,
        limit: int,
        actor: TaskActor,
        owner_user_id: str | None = None,
    ) -> ReconciliationBatchResult:
        scope = resolve_task_compatibility_scope(
            db, authenticated_owner_user_id=owner_user_id or db.client_id
        )
        work_limit = max(0, int(limit))
        to_process = db.candidate_notes_for_task_discovery(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            limit=work_limit,
        ) if work_limit else []
        results: list[ReconciliationResult] = []
        for candidate in to_process:
            note = db.get_note_by_id(str(candidate["id"]))
            if not note:
                continue
            results.append(
                self.reconcile_note(
                    db=db,
                    note_id=str(note["id"]),
                    note_version=int(note["version"]),
                    content=str(note.get("content") or ""),
                    actor=actor,
                    owner_user_id=scope.owner_user_id,
                )
            )
        remaining = db.count_candidate_notes_for_task_discovery(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
        )
        return ReconciliationBatchResult(
            status="incomplete" if remaining else "clean",
            processed_notes=len(results),
            remaining_stale_notes=remaining,
            results=results,
        )

    def reconcile_note_current(
        self,
        *,
        db: CharactersRAGDB,
        note_id: str,
        actor: TaskActor,
        owner_user_id: str | None = None,
    ) -> ReconciliationResult:
        note = self._require_note(db, note_id)
        return self.reconcile_note(
            db=db,
            note_id=note_id,
            note_version=int(note["version"]),
            content=str(note.get("content") or ""),
            actor=actor,
            owner_user_id=owner_user_id,
        )

    def ensure_note_reconciled(
        self,
        *,
        db: CharactersRAGDB,
        note_id: str,
        actor: TaskActor,
        owner_user_id: str | None = None,
    ) -> ReconciliationResult | None:
        scope = resolve_task_compatibility_scope(
            db, authenticated_owner_user_id=owner_user_id or db.client_id
        )
        note = self._require_note(db, note_id)
        state = db.get_reconciliation_state(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            note_id=note_id,
        )
        if state is not None and int(state["note_version"]) == int(note["version"]):
            if state["status"] == "clean":
                return None
            return ReconciliationResult(
                note_id=note_id,
                note_version=int(state["note_version"]),
                parsed_count=int(state.get("item_count") or 0),
                warning_count=int(state.get("warning_count") or 0),
            )
        return self.reconcile_note(
            db=db,
            note_id=note_id,
            note_version=int(note["version"]),
            content=str(note.get("content") or ""),
            actor=actor,
            owner_user_id=scope.owner_user_id,
        )

    def create_task_for_note(
        self,
        *,
        db: CharactersRAGDB,
        note_id: str,
        text: str,
        status: str,
        metadata: dict[str, Any],
        expected_note_version: int,
        actor: TaskActor,
        owner_user_id: str | None = None,
    ) -> dict[str, Any]:
        scope = resolve_task_compatibility_scope(
            db, authenticated_owner_user_id=owner_user_id or db.client_id
        )
        self._validate_task_text(text)
        self._validate_task_status(status)
        self._validate_metadata(metadata)
        task_coordinator = self._active_task_coordinator(scope=scope)
        if task_coordinator is not None:
            return self._create_task_through_sync(
                db=db,
                scope=scope,
                coordinator=task_coordinator,
                note_id=note_id,
                text=text,
                status=status,
                metadata=metadata,
                expected_note_version=expected_note_version,
                actor=actor,
            )
        marker = "x" if status == "done" else " "
        line = f"- [{marker}] {self._render_body(text=text.strip(), metadata=metadata)}"

        coordinator = active_coordinator(db, user_id=scope.owner_user_id)
        transaction = db.transaction() if coordinator is None else nullcontext(None)
        with transaction as conn:
            note = self._require_note_version(db, note_id=note_id, expected_note_version=expected_note_version)
            self.reconcile_note(
                db=db,
                note_id=note_id,
                note_version=expected_note_version,
                content=str(note.get("content") or ""),
                actor=self._internal_reconciliation_actor(actor),
                owner_user_id=scope.owner_user_id,
            )
            new_content = self._append_checklist_line(str(note.get("content") or ""), line)
            _write_note_content(
                db,
                coordinator=coordinator,
                note=note,
                content=new_content,
                expected_version=expected_note_version,
                conn=conn,
            )
            updated_note = self._require_note(db, note_id)
            result = self.reconcile_note(
                db=db,
                note_id=note_id,
                note_version=int(updated_note["version"]),
                content=str(updated_note.get("content") or ""),
                actor=actor,
                owner_user_id=scope.owner_user_id,
            )
            if not result.created_task_ids:
                raise ConflictError("Task creation did not create a task record.", entity="tasks", entity_id=note_id)
            task = db.get_task(
                owner_user_id=scope.owner_user_id,
                dataset_id=scope.dataset_id,
                task_id=result.created_task_ids[-1],
            )
            if task is None:
                raise ConflictError("Created task was not found.", entity="tasks", entity_id=note_id)
            self._capture_task_mutation(
                db=db,
                scope=scope,
                actor=actor,
                before=None,
                after=task,
                conn=conn,
            )
            return task

    @staticmethod
    def _task_projection_line(
        *,
        task: dict[str, Any],
    ) -> str:
        """Render one canonical top-level line for an explicit link operation."""

        marker = "x" if str(task["status"]) == "done" else " "
        metadata = dict(task.get("metadata_json") or {})
        body = NotesTaskService._render_body(
            text=str(task["text"]),
            metadata={
                key: metadata[key]
                for key in _METADATA_TOKEN_ORDER
                if metadata.get(key) is not None
            },
        )
        return (
            f"- [{marker}] {body} "
            + render_task_marker(
                str(task["id"]),
                revision=int(task["canonical_revision"]),
                object_hash=str(task["canonical_hash"]),
            )
        )

    @staticmethod
    def _cache_materialized_task_projection(
        *,
        db: CharactersRAGDB,
        scope: TaskStoreScope,
        task_id: str,
    ) -> None:
        """Refresh the disposable locator cache after a linked group applies."""

        task = db.get_task(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            task_id=task_id,
        )
        if task is None or task["projection_status"] != "live":
            raise ConflictError(
                "Linked task product state is unavailable.",
                entity="tasks",
                entity_id=task_id,
            )
        note = db.get_note_by_id(str(task["note_id"]))
        if note is None:
            raise ConflictError(
                "Linked task note is unavailable.",
                entity="tasks",
                entity_id=task_id,
            )
        matches = [
            item
            for item in parse_note_checklists(
                note_id=str(note["id"]),
                note_version=int(note["version"]),
                content=str(note.get("content") or ""),
            ).items
            if item.marker is not None and item.marker.task_id == task_id
        ]
        if len(matches) != 1:
            raise ConflictError(
                "Linked task marker is unavailable.",
                entity="tasks",
                entity_id=task_id,
            )
        item = matches[0]
        db.set_task_projection(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            task_id=task_id,
            note_id=str(note["id"]),
            note_version=int(note["version"]),
            line_number=item.locator.line_number,
            start_offset=item.locator.start_offset,
            end_offset=item.locator.end_offset,
            normalized_text_hash=item.locator.normalized_text_hash,
            occurrence_index=item.locator.occurrence_index,
            block_fingerprint=item.locator.block_fingerprint,
            raw_line=item.raw_line,
            has_child_content=item.has_child_content,
        )

    def restore_task(
        self,
        *,
        db: CharactersRAGDB,
        task_id: str,
        expected_task_version: int,
        expected_note_version: int,
        expected_base_server_cursor: int,
        expected_base_revision: int,
        expected_base_hash: str,
        actor: TaskActor,
        owner_user_id: str | None = None,
    ) -> dict[str, Any]:
        """Restore one exact task tombstone and its verified former projection."""

        scope = resolve_task_compatibility_scope(
            db,
            authenticated_owner_user_id=owner_user_id or db.client_id,
        )
        coordinator = self._active_task_coordinator(scope=scope)
        if coordinator is None or coordinator.service is None:
            raise ConflictError(
                "Task restore requires active synchronized task authority.",
                entity="tasks",
                entity_id=task_id,
            )
        task = db.get_task(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            task_id=task_id,
            include_deleted=True,
        )
        if (
            task is None
            or not bool(task["deleted"])
            or int(task["version"]) != expected_task_version
        ):
            raise ConflictError(
                "Task tombstone changed concurrently.",
                entity="tasks",
                entity_id=task_id,
            )
        task_head = coordinator.service.store.get_current_head(
            scope.dataset_id,
            "notes.task",
            task_id,
        )
        if (
            task_head is None
            or task_head.operation != "tombstone"
            or task_head.server_cursor != expected_base_server_cursor
            or task_head.object_revision != expected_base_revision
            or task_head.payload_hash != expected_base_hash
            or int(task["canonical_revision"]) != expected_base_revision
            or str(task["canonical_hash"]) != expected_base_hash
        ):
            raise ConflictError(
                "Task restore requires the exact current tombstone base.",
                entity="tasks",
                entity_id=task_id,
            )
        note = self._require_note_version(
            db,
            note_id=str(task["note_id"]),
            expected_note_version=expected_note_version,
        )
        occurred_at = normalize_sync_timestamp(coordinator.service.clock())
        if occurred_at is None:
            raise ConflictError(
                "Task mutation timestamp is unavailable.",
                entity="tasks",
                entity_id=task_id,
            )
        source_row = db.task_store._sync_bootstrap_task_row(
            task,
            scope.owner_user_id,
        )
        payload = parse_notes_task_v1(
            source_row["sync_payload"],
            owner_user_id=scope.owner_user_id,
        )
        revision = expected_base_revision + 1
        from tldw_Server_API.app.core.Sync.v2.notes_task_coordinator import (
            _projection_anchor_from_envelope,
        )

        prior_anchor = _projection_anchor_from_envelope(task_head)
        restore_linked = prior_anchor is not None and prior_anchor.linked
        after = {
            **task,
            "deleted": False,
            "projection_status": "live" if restore_linked else "unlinked",
            "updated_at": occurred_at,
            "version": int(task["version"]) + 1,
            "canonical_revision": revision,
            "canonical_hash": notes_task_object_hash(
                payload,
                revision=revision,
                deleted=False,
            ),
            "source_diagnostic_code": None,
            "source_diagnostic_hash": None,
        }
        note_step = None
        if restore_linked:
            note_head = coordinator.service.store.get_current_head(
                scope.dataset_id,
                "notes.note",
                str(note["id"]),
            )
            if (
                prior_anchor is None
                or note_head is None
                or note_head.client_envelope_id != prior_anchor.note_envelope_id
                or note_head.payload_hash != prior_anchor.note_hash
            ):
                raise ConflictError(
                    "Task restore note base is unavailable.",
                    entity="tasks",
                    entity_id=task_id,
                )
            note_content = str(note.get("content") or "")
            if any(
                item.marker is not None and item.marker.task_id == task_id
                for item in parse_note_checklists(
                    note_id=str(note["id"]),
                    note_version=int(note["version"]),
                    content=note_content,
                ).items
            ):
                raise ConflictError(
                    "Task restore marker already exists.",
                    entity="tasks",
                    entity_id=task_id,
                )
            note_step = self._note_projection_step(
                coordinator=coordinator,
                note=note,
                content=self._append_checklist_line(
                    note_content,
                    self._task_projection_line(task=after),
                ),
            )
        source, source_kind = self._task_source(actor)
        mutation = build_task_capture_mutation(
            db=db,
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            actor=actor,
            before=task,
            after=after,
            source_kind=source_kind,
        )
        result = coordinator.capture(
            coordinator.plan_task_mutation(mutation, note_step=note_step),
            source=source,
        )
        if not result.fully_applied:
            raise ConflictError(
                "Task restore projection is incomplete.",
                entity="tasks",
                entity_id=task_id,
            )
        if restore_linked:
            self._cache_materialized_task_projection(
                db=db,
                scope=scope,
                task_id=task_id,
            )
        restored = db.get_task(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            task_id=task_id,
        )
        if restored is None:
            raise ConflictError(
                "Restored task was not found.",
                entity="tasks",
                entity_id=task_id,
            )
        return restored

    def relink_task(
        self,
        *,
        db: CharactersRAGDB,
        task_id: str,
        note_id: str,
        expected_task_version: int,
        expected_note_version: int,
        actor: TaskActor,
        owner_user_id: str | None = None,
    ) -> dict[str, Any]:
        """Relink one unlinked task to its immutable authorized parent note."""

        scope = resolve_task_compatibility_scope(
            db,
            authenticated_owner_user_id=owner_user_id or db.client_id,
        )
        coordinator = self._active_task_coordinator(scope=scope)
        if coordinator is None or coordinator.service is None:
            raise ConflictError(
                "Task relink requires active synchronized task authority.",
                entity="tasks",
                entity_id=task_id,
            )
        task = self._require_task_version(
            db,
            task_id=task_id,
            expected_task_version=expected_task_version,
            scope=scope,
            conn=None,
        )
        if task["projection_status"] != "unlinked" or str(task["note_id"]) != note_id:
            raise ConflictError(
                "Task relink destination is not its authorized parent note.",
                entity="tasks",
                entity_id=task_id,
            )
        note = self._require_note_version(
            db,
            note_id=note_id,
            expected_note_version=expected_note_version,
        )
        content = str(note.get("content") or "")
        if any(
            item.marker is not None and item.marker.task_id == task_id
            for item in parse_note_checklists(
                note_id=note_id,
                note_version=int(note["version"]),
                content=content,
            ).items
        ):
            raise ConflictError(
                "Task relink marker already exists.",
                entity="tasks",
                entity_id=task_id,
            )
        occurred_at = normalize_sync_timestamp(coordinator.service.clock())
        if occurred_at is None:
            raise ConflictError(
                "Task mutation timestamp is unavailable.",
                entity="tasks",
                entity_id=task_id,
            )
        source_row = db.task_store._sync_bootstrap_task_row(
            task,
            scope.owner_user_id,
        )
        payload = parse_notes_task_v1(
            source_row["sync_payload"],
            owner_user_id=scope.owner_user_id,
        )
        revision = int(task["canonical_revision"]) + 1
        after = {
            **task,
            "projection_status": "live",
            "updated_at": occurred_at,
            "version": int(task["version"]) + 1,
            "canonical_revision": revision,
            "canonical_hash": notes_task_object_hash(
                payload,
                revision=revision,
                deleted=False,
            ),
            "source_diagnostic_code": None,
            "source_diagnostic_hash": None,
        }
        note_step = self._note_projection_step(
            coordinator=coordinator,
            note=note,
            content=self._append_checklist_line(
                content,
                self._task_projection_line(task=after),
            ),
        )
        source, source_kind = self._task_source(actor)
        mutation = build_task_capture_mutation(
            db=db,
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            actor=actor,
            before=task,
            after=after,
            source_kind=source_kind,
        )
        result = coordinator.capture(
            coordinator.plan_task_mutation(mutation, note_step=note_step),
            source=source,
        )
        if not result.fully_applied:
            raise ConflictError(
                "Task relink projection is incomplete.",
                entity="tasks",
                entity_id=task_id,
            )
        self._cache_materialized_task_projection(
            db=db,
            scope=scope,
            task_id=task_id,
        )
        relinked = db.get_task(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            task_id=task_id,
        )
        if relinked is None:
            raise ConflictError(
                "Relinked task was not found.",
                entity="tasks",
                entity_id=task_id,
            )
        return relinked

    def update_task(
        self,
        *,
        db: CharactersRAGDB,
        task_id: str,
        expected_task_version: int,
        expected_note_version: int | None,
        actor: TaskActor,
        text: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
        record_only: bool = False,
        owner_user_id: str | None = None,
    ) -> dict[str, Any]:
        scope = resolve_task_compatibility_scope(
            db, authenticated_owner_user_id=owner_user_id or db.client_id
        )
        if text is not None:
            self._validate_task_text(text)
        if status is not None:
            self._validate_task_status(status)
        if metadata is not None:
            self._validate_metadata(metadata)

        task_coordinator = self._active_task_coordinator(scope=scope)
        if task_coordinator is not None:
            task = self._require_task_version(
                db,
                task_id=task_id,
                expected_task_version=expected_task_version,
                scope=scope,
                conn=None,
            )
            if str(task["projection_status"]) == "live":
                return self._update_projected_task_through_sync(
                    db=db,
                    scope=scope,
                    coordinator=task_coordinator,
                    task=task,
                    expected_note_version=expected_note_version,
                    actor=actor,
                    text=text,
                    status=status,
                    metadata=metadata,
                    record_only=record_only,
                )
            return self._update_unlinked_metadata_through_sync(
                db=db,
                scope=scope,
                coordinator=task_coordinator,
                task_id=task_id,
                expected_task_version=expected_task_version,
                actor=actor,
                metadata=metadata,
                text=text,
                status=status,
                record_only=record_only,
            )

        coordinator = active_coordinator(db, user_id=scope.owner_user_id)
        transaction = db.transaction() if coordinator is None else nullcontext(None)
        with transaction as conn:
            if conn is not None:
                db.task_store.lock_authorized_write_scope(
                    owner_user_id=scope.owner_user_id,
                    dataset_id=scope.dataset_id,
                    conn=conn,
                )
            task = self._require_task_version(
                db,
                task_id=task_id,
                expected_task_version=expected_task_version,
                scope=scope,
                conn=conn,
            )
            projection_status = str(task["projection_status"])
            if projection_status == "ambiguous":
                raise ConflictError(f"Task projection is ambiguous for task '{task_id}'.", entity="tasks", entity_id=task_id)
            if projection_status == "unlinked":
                if not record_only or text is not None or status is not None:
                    raise ConflictError(
                        f"Task projection is unlinked for task '{task_id}'.",
                        entity="tasks",
                        entity_id=task_id,
                    )
                if metadata is None:
                    return task
                updated = self._update_record_only_metadata(
                    db=db,
                    conn=conn,
                    task=task,
                    expected_task_version=expected_task_version,
                    metadata=metadata,
                    actor=actor,
                    scope=scope,
                )
                self._capture_task_mutation(
                    db=db,
                    scope=scope,
                    actor=actor,
                    before=task,
                    after=updated,
                    conn=conn,
                )
                return updated
            if projection_status != "live":
                raise ConflictError(
                    f"Task projection is {projection_status} for task '{task_id}'.",
                    entity="tasks",
                    entity_id=task_id,
                )
            if record_only:
                raise ConflictError(
                    f"Task '{task_id}' is projected into a note and cannot be updated record-only.",
                    entity="tasks",
                    entity_id=task_id,
                )
            if expected_note_version is None:
                raise InputError("expected_note_version is required for projected task updates.")

            projection = self._require_projection(db, task_id=task_id, scope=scope, conn=conn)
            note = self._require_note_version(
                db,
                note_id=str(task["note_id"]),
                expected_note_version=expected_note_version,
            )
            self._require_projection_version(projection, expected_note_version, task_id)
            parsed_item = self._find_projected_item(
                note_id=str(note["id"]),
                note_version=int(note["version"]),
                content=str(note.get("content") or ""),
                projection=projection,
                task_id=task_id,
            )
            new_text = text.strip() if text is not None else str(task["text"])
            new_status = status or str(task["status"])
            new_metadata = metadata if metadata is not None else dict(task.get("metadata_json") or {})
            if status is not None and text is None and metadata is None:
                new_line = self._rewrite_marker_only(
                    raw_line=parsed_item.raw_line,
                    checked=new_status == "done",
                )
            else:
                new_line = self._rewrite_line(
                    raw_line=parsed_item.raw_line,
                    checked=new_status == "done",
                    text=new_text,
                    metadata=new_metadata,
                    preserve_existing_body=metadata is not None and text is None,
                )
            new_content = self._replace_projection_line(
                content=str(note.get("content") or ""),
                projection=projection,
                new_line=new_line,
            )
            _write_note_content(
                db,
                coordinator=coordinator,
                note=note,
                content=new_content,
                expected_version=expected_note_version,
                conn=conn,
            )
            updated_note_version = expected_note_version + 1
            updated_task = db.update_task_record(
                owner_user_id=scope.owner_user_id,
                dataset_id=scope.dataset_id,
                task_id=task_id,
                expected_version=expected_task_version,
                text=new_text,
                status=new_status,
                metadata=new_metadata,
                actor_type=actor.actor_type,
                actor_id=actor.actor_id,
                tool_name=actor.tool_name,
                policy_mode=actor.policy_mode,
                approval_id=actor.approval_id,
                idempotency_key=actor.idempotency_key,
                conn=conn,
            )
            updated_parsed_item = self._find_projected_item(
                note_id=str(note["id"]),
                note_version=updated_note_version,
                content=new_content,
                projection={**projection, "raw_line": new_line},
                task_id=task_id,
                line_number=int(projection["line_number"]),
            )
            db.set_task_projection(
                owner_user_id=scope.owner_user_id,
                dataset_id=scope.dataset_id,
                task_id=task_id,
                note_id=str(note["id"]),
                note_version=updated_note_version,
                line_number=updated_parsed_item.locator.line_number,
                start_offset=updated_parsed_item.locator.start_offset,
                end_offset=updated_parsed_item.locator.end_offset,
                normalized_text_hash=updated_parsed_item.locator.normalized_text_hash,
                occurrence_index=updated_parsed_item.locator.occurrence_index,
                block_fingerprint=updated_parsed_item.locator.block_fingerprint,
                raw_line=updated_parsed_item.raw_line,
                has_child_content=updated_parsed_item.has_child_content,
                conn=conn,
            )
            self.reconcile_note(
                db=db,
                note_id=str(note["id"]),
                note_version=updated_note_version,
                content=new_content,
                actor=self._internal_reconciliation_actor(actor),
                owner_user_id=scope.owner_user_id,
            )
            self._capture_task_mutation(
                db=db,
                scope=scope,
                actor=actor,
                before=task,
                after=updated_task,
                conn=conn,
            )
            return updated_task

    def delete_task(
        self,
        *,
        db: CharactersRAGDB,
        task_id: str,
        expected_task_version: int,
        expected_note_version: int | None,
        record_only: bool,
        actor: TaskActor,
        owner_user_id: str | None = None,
    ) -> dict[str, Any]:
        scope = resolve_task_compatibility_scope(
            db, authenticated_owner_user_id=owner_user_id or db.client_id
        )
        task_coordinator = self._active_task_coordinator(scope=scope)
        if task_coordinator is not None:
            return self._delete_task_through_sync(
                db=db,
                scope=scope,
                coordinator=task_coordinator,
                task_id=task_id,
                expected_task_version=expected_task_version,
                expected_note_version=expected_note_version,
                record_only=record_only,
                actor=actor,
            )
        coordinator = active_coordinator(db, user_id=scope.owner_user_id)
        transaction = db.transaction() if coordinator is None else nullcontext(None)
        with transaction as conn:
            if conn is not None:
                db.task_store.lock_authorized_write_scope(
                    owner_user_id=scope.owner_user_id,
                    dataset_id=scope.dataset_id,
                    conn=conn,
                )
            task = self._require_task_version(
                db,
                task_id=task_id,
                expected_task_version=expected_task_version,
                scope=scope,
                conn=conn,
            )
            projection_status = str(task["projection_status"])
            if projection_status == "ambiguous":
                raise ConflictError(f"Task projection is ambiguous for task '{task_id}'.", entity="tasks", entity_id=task_id)
            if projection_status == "unlinked":
                if not record_only:
                    raise ConflictError(
                        f"Task projection is unlinked for task '{task_id}'. Record-only delete mode is required.",
                        entity="tasks",
                        entity_id=task_id,
                    )
                deleted = db.soft_delete_task(
                    owner_user_id=scope.owner_user_id,
                    dataset_id=scope.dataset_id,
                    task_id=task_id,
                    expected_version=expected_task_version,
                    allow_record_only=True,
                    actor_type=actor.actor_type,
                    actor_id=actor.actor_id,
                    tool_name=actor.tool_name,
                    policy_mode=actor.policy_mode,
                    approval_id=actor.approval_id,
                    idempotency_key=actor.idempotency_key,
                    conn=conn,
                )
                self._capture_task_mutation(
                    db=db,
                    scope=scope,
                    actor=actor,
                    before=task,
                    after=deleted,
                    conn=conn,
                )
                return deleted
            if projection_status != "live":
                raise ConflictError(
                    f"Task projection is {projection_status} for task '{task_id}'.",
                    entity="tasks",
                    entity_id=task_id,
                )
            if record_only:
                raise ConflictError(
                    f"Task '{task_id}' is projected into a note and cannot be deleted record-only.",
                    entity="tasks",
                    entity_id=task_id,
                )
            if expected_note_version is None:
                raise InputError("expected_note_version is required for projected task deletion.")

            projection = self._require_projection(db, task_id=task_id, scope=scope, conn=conn)
            note = self._require_note_version(
                db,
                note_id=str(task["note_id"]),
                expected_note_version=expected_note_version,
            )
            self._require_projection_version(projection, expected_note_version, task_id)
            parsed_item = self._find_projected_item(
                note_id=str(note["id"]),
                note_version=int(note["version"]),
                content=str(note.get("content") or ""),
                projection=projection,
                task_id=task_id,
            )
            if parsed_item.has_child_content:
                raise ConflictError(
                    f"Task '{task_id}' has nested child content and cannot be deleted by default.",
                    entity="tasks",
                    entity_id=task_id,
                )
            new_content = self._delete_projection_line(str(note.get("content") or ""), projection)
            _write_note_content(
                db,
                coordinator=coordinator,
                note=note,
                content=new_content,
                expected_version=expected_note_version,
                conn=conn,
            )
            deleted = db.soft_delete_task(
                owner_user_id=scope.owner_user_id,
                dataset_id=scope.dataset_id,
                task_id=task_id,
                expected_version=expected_task_version,
                projection_note_id=str(projection["note_id"]),
                projection_note_version=int(projection["note_version"]),
                projection_line_number=int(projection["line_number"]),
                actor_type=actor.actor_type,
                actor_id=actor.actor_id,
                tool_name=actor.tool_name,
                policy_mode=actor.policy_mode,
                approval_id=actor.approval_id,
                idempotency_key=actor.idempotency_key,
                conn=conn,
            )
            self.reconcile_note(
                db=db,
                note_id=str(note["id"]),
                note_version=expected_note_version + 1,
                content=new_content,
                actor=self._internal_reconciliation_actor(actor),
                owner_user_id=scope.owner_user_id,
            )
            self._capture_task_mutation(
                db=db,
                scope=scope,
                actor=actor,
                before=task,
                after=deleted,
                conn=conn,
            )
            return deleted

    @staticmethod
    def _require_note(db: CharactersRAGDB, note_id: str) -> dict[str, Any]:
        note = db.get_note_by_id(note_id)
        if not note:
            raise ConflictError("Note not found.", entity="notes", entity_id=note_id)
        return note

    def _require_note_version(
        self,
        db: CharactersRAGDB,
        *,
        note_id: str,
        expected_note_version: int,
    ) -> dict[str, Any]:
        note = self._require_note(db, note_id)
        if int(note["version"]) != int(expected_note_version):
            raise ConflictError(
                f"Note version mismatch for ID '{note_id}'. Expected {expected_note_version}, found {note['version']}.",
                entity="notes",
                entity_id=note_id,
            )
        return note

    @staticmethod
    def _require_task_version(
        db: CharactersRAGDB,
        *,
        task_id: str,
        expected_task_version: int,
        scope: TaskStoreScope,
        conn: TaskConnection,
    ) -> dict[str, Any]:
        task = db.task_store._fetch_task(
            task_id,
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            include_deleted=False,
            conn=conn,
        )
        if task is None:
            raise ConflictError(f"Task with ID '{task_id}' not found.", entity="tasks", entity_id=task_id)
        if int(task["version"]) != int(expected_task_version):
            raise ConflictError(
                f"Task version mismatch for ID '{task_id}'. Expected {expected_task_version}, found {task['version']}.",
                entity="tasks",
                entity_id=task_id,
            )
        return task

    @staticmethod
    def _require_projection(
        db: CharactersRAGDB,
        *,
        task_id: str,
        scope: TaskStoreScope,
        conn: TaskConnection,
    ) -> dict[str, Any]:
        projection = db.get_task_projection(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            task_id=task_id,
            conn=conn,
        )
        if projection is None:
            raise ConflictError(f"Task projection is missing for task '{task_id}'.", entity="tasks", entity_id=task_id)
        return projection

    @staticmethod
    def _require_projection_version(projection: dict[str, Any], expected_note_version: int, task_id: str) -> None:
        if int(projection["note_version"]) != int(expected_note_version):
            raise ConflictError(
                f"Task projection is stale for task '{task_id}'.",
                entity="tasks",
                entity_id=task_id,
            )

    @staticmethod
    def _validate_task_text(text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise InputError("Task text cannot be empty.")
        if len(text) > 2000:
            raise InputError("Task text must be 2000 characters or fewer.")
        if "\n" in text or "\r" in text:
            raise InputError("Task text cannot contain newline characters.")
        if _task_text_contains_parseable_metadata_token(text):
            raise InputError(
                "Task text cannot include parseable metadata tokens; "
                "pass due_date, priority, or estimate metadata separately."
            )

    @staticmethod
    def _validate_task_status(status: str) -> None:
        if status not in _TASK_STATUSES:
            raise InputError(f"Unsupported task status '{status}'. Expected 'open' or 'done'.")

    @staticmethod
    def _validate_metadata(metadata: dict[str, Any]) -> None:
        allowed = {"due_date", "priority", "estimate"}
        unknown = sorted(set(metadata) - allowed)
        if unknown:
            raise InputError(f"Unsupported task metadata keys: {', '.join(unknown)}.")
        due_date = metadata.get("due_date")
        if due_date is not None:
            if not isinstance(due_date, str):
                raise InputError("Task due_date metadata must be a string.")
            if not _is_iso_date_token(due_date):
                raise InputError("Task due_date metadata must use YYYY-MM-DD format.")
            try:
                date.fromisoformat(due_date)
            except ValueError as exc:
                raise InputError("Task due_date metadata must be a real ISO date.") from exc
        priority = metadata.get("priority")
        if priority is not None and priority not in {"high", "medium", "low"}:
            raise InputError("Task priority metadata must be high, medium, or low.")
        estimate = metadata.get("estimate")
        if estimate is not None:
            if not isinstance(estimate, str) or not _is_estimate_token(estimate):
                raise InputError("Task estimate metadata must match '<number><m|h|d>'.")

    @staticmethod
    def _append_checklist_line(content: str, line: str) -> str:
        if not content:
            return f"{line}\n"
        if content.endswith("\n\n"):
            return f"{content}{line}\n"
        if content.endswith("\n"):
            return f"{content}\n{line}\n"
        return f"{content}\n\n{line}\n"

    @staticmethod
    def _render_body(*, text: str, metadata: dict[str, Any]) -> str:
        pieces = [text.strip()]
        for key in _METADATA_TOKEN_ORDER:
            value = metadata.get(key)
            if value is not None:
                pieces.append(f"@{_METADATA_TOKEN_NAMES[key]}({value})")
        return " ".join(piece for piece in pieces if piece)

    def _rewrite_line(
        self,
        *,
        raw_line: str,
        checked: bool,
        text: str,
        metadata: dict[str, Any],
        preserve_existing_body: bool,
    ) -> str:
        parsed_line = _parse_checklist_line(raw_line)
        if parsed_line is None:
            raise ConflictError("Task projection line is no longer a checklist item.", entity="tasks")
        marker = "x" if checked else " "
        if preserve_existing_body:
            base_text = text.strip()
            body = self._render_body(text=base_text, metadata=metadata)
        else:
            body = self._render_body(text=text, metadata=metadata)
        return f"{parsed_line.indent}{parsed_line.bullet}{parsed_line.space}[{marker}] {body}"

    @staticmethod
    def _rewrite_marker_only(*, raw_line: str, checked: bool) -> str:
        parsed_line = _parse_checklist_line(raw_line)
        if parsed_line is None:
            raise ConflictError("Task projection line is no longer a checklist item.", entity="tasks")
        marker = "x" if checked else " "
        return (
            f"{parsed_line.indent}{parsed_line.bullet}{parsed_line.space}"
            f"[{marker}]{parsed_line.body_part}"
        )

    @staticmethod
    def _replace_projection_line(*, content: str, projection: dict[str, Any], new_line: str) -> str:
        start = int(projection["start_offset"])
        end = int(projection["end_offset"])
        if start < 0 or end < start or end > len(content):
            raise ConflictError("Task projection offsets are invalid.", entity="tasks")
        return f"{content[:start]}{new_line}{content[end:]}"

    @staticmethod
    def _delete_projection_line(content: str, projection: dict[str, Any]) -> str:
        start = int(projection["start_offset"])
        end = int(projection["end_offset"])
        if start < 0 or end < start or end > len(content):
            raise ConflictError("Task projection offsets are invalid.", entity="tasks")
        if end + 1 < len(content) and content[end : end + 2] == "\r\n":
            end += 2
        elif end < len(content) and content[end] == "\n":
            end += 1
        elif start >= 2 and content[start - 2 : start] == "\r\n":
            start -= 2
        elif start > 0 and content[start - 1] == "\n":
            start -= 1
        return f"{content[:start]}{content[end:]}"

    @staticmethod
    def _find_projected_item(
        *,
        note_id: str,
        note_version: int,
        content: str,
        projection: dict[str, Any],
        task_id: str,
        line_number: int | None = None,
    ) -> ParsedChecklistItem:
        parsed = parse_note_checklists(note_id=note_id, note_version=note_version, content=content)
        expected_line_number = int(line_number if line_number is not None else projection["line_number"])
        matches = [item for item in parsed.items if item.locator.line_number == expected_line_number]
        if len(matches) != 1:
            raise ConflictError(
                f"Task projection line is ambiguous for task '{task_id}'.",
                entity="tasks",
                entity_id=task_id,
            )
        return matches[0]

    @staticmethod
    def _update_record_only_metadata(
        *,
        db: CharactersRAGDB,
        conn: TaskConnection,
        task: dict[str, Any],
        expected_task_version: int,
        metadata: dict[str, Any],
        actor: TaskActor,
        scope: TaskStoreScope,
    ) -> dict[str, Any]:
        return db.update_unlinked_task_metadata_record_only(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            task_id=str(task["id"]),
            expected_version=expected_task_version,
            metadata=metadata,
            actor_type=actor.actor_type,
            actor_id=actor.actor_id,
            tool_name=actor.tool_name,
            policy_mode=actor.policy_mode,
            approval_id=actor.approval_id,
            idempotency_key=actor.idempotency_key,
            conn=conn,
        )
