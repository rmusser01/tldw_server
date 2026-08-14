"""Service entry point for note-backed task reconciliation and mutations."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, NamedTuple

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError, InputError
from tldw_Server_API.app.core.Notes.organization_capture import (
    active_coordinator,
    capture_note_upsert,
)
from tldw_Server_API.app.core.Notes_Tasks.markdown_parser import parse_note_checklists
from tldw_Server_API.app.core.Notes_Tasks.models import ParsedChecklistItem, ReconciliationResult, TaskActor
from tldw_Server_API.app.core.Notes_Tasks.reconciler import NotesTaskReconciler

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.chacha.task_store import TaskConnection
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.app.core.Sync.v2.notes_organization_coordinator import (
        NotesOrganizationCoordinator,
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


def resolve_task_compatibility_scope(
    db: CharactersRAGDB,
    *,
    authenticated_owner_user_id: str,
) -> TaskStoreScope:
    """Resolve product-owned task scope without accepting a client dataset selector."""
    owner, dataset = db.resolve_task_compatibility_scope(
        owner_user_id=str(authenticated_owner_user_id)
    )
    return TaskStoreScope(owner_user_id=owner, dataset_id=dataset)


class NotesTaskService:
    """Coordinate task-backed checklist reconciliation for saved notes."""

    def __init__(self, reconciler: NotesTaskReconciler | None = None) -> None:
        self._reconciler = reconciler or NotesTaskReconciler()

    @staticmethod
    def _internal_reconciliation_actor(actor: TaskActor) -> TaskActor:
        return TaskActor(
            actor_type=actor.actor_type,
            actor_id=actor.actor_id,
            tool_name="notes.tasks.reconciliation",
        )

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
            note_id,
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
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
        marker = "x" if status == "done" else " "
        line = f"- [{marker}] {self._render_body(text=text.strip(), metadata=metadata)}"

        coordinator = active_coordinator(db, user_id=actor.actor_id)
        transaction = db.transaction() if coordinator is None else nullcontext(None)
        with transaction:
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
            task = db.get_task_scoped(
                owner_user_id=scope.owner_user_id,
                dataset_id=scope.dataset_id,
                task_id=result.created_task_ids[-1],
            )
            if task is None:
                raise ConflictError("Created task was not found.", entity="tasks", entity_id=note_id)
            return task

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

        coordinator = active_coordinator(db, user_id=actor.actor_id)
        transaction = db.transaction() if coordinator is None else nullcontext(None)
        with transaction as conn:
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
                return self._update_record_only_metadata(
                    db=db,
                    conn=conn,
                    task=task,
                    expected_task_version=expected_task_version,
                    metadata=metadata,
                    actor=actor,
                    scope=scope,
                )
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
        coordinator = active_coordinator(db, user_id=actor.actor_id)
        transaction = db.transaction() if coordinator is None else nullcontext(None)
        with transaction as conn:
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
                return db.soft_delete_task(
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
            task_id,
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
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
