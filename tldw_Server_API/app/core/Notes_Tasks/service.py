"""Service entry point for note-backed task reconciliation and mutations."""

from __future__ import annotations

from datetime import date
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError, InputError
from tldw_Server_API.app.core.Notes_Tasks.markdown_parser import parse_note_checklists
from tldw_Server_API.app.core.Notes_Tasks.models import ParsedChecklistItem, ReconciliationResult, TaskActor
from tldw_Server_API.app.core.Notes_Tasks.reconciler import NotesTaskReconciler

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.app.core.DB_Management.chacha.task_store import TaskConnection


_CHECKLIST_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<bullet>[-*+])(?P<space>[ \t]+)\[(?P<marker>[ xX])\](?P<body_part>(?:[ \t]+(?P<body>.*)|[ \t]*))$"
)
_METADATA_TOKEN_ORDER = ("due_date", "priority", "estimate")
_METADATA_TOKEN_NAMES = {"due_date": "due", "priority": "priority", "estimate": "estimate"}


@dataclass(frozen=True)
class ReconciliationBatchResult:
    """Summary for opportunistic stale-note reconciliation work."""

    status: str
    processed_notes: int
    remaining_stale_notes: int
    results: list[ReconciliationResult]


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
    ) -> ReconciliationResult:
        return self._reconciler.reconcile_note(
            db=db,
            note_id=note_id,
            note_version=note_version,
            content=content,
            actor=actor,
        )

    def reconcile_stale_notes(
        self,
        *,
        db: CharactersRAGDB,
        limit: int,
        actor: TaskActor,
    ) -> ReconciliationBatchResult:
        work_limit = max(0, int(limit))
        candidates = db.candidate_notes_for_task_discovery(limit=work_limit + 1 if work_limit else 1)
        to_process = candidates[:work_limit]
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
                )
            )
        remaining = max(0, len(candidates) - len(to_process))
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
    ) -> ReconciliationResult:
        note = self._require_note(db, note_id)
        return self.reconcile_note(
            db=db,
            note_id=note_id,
            note_version=int(note["version"]),
            content=str(note.get("content") or ""),
            actor=actor,
        )

    def ensure_note_reconciled(
        self,
        *,
        db: CharactersRAGDB,
        note_id: str,
        actor: TaskActor,
    ) -> ReconciliationResult | None:
        note = self._require_note(db, note_id)
        state = db.get_reconciliation_state(note_id)
        if state is not None and int(state["note_version"]) == int(note["version"]) and state["status"] == "clean":
            return None
        return self.reconcile_note(
            db=db,
            note_id=note_id,
            note_version=int(note["version"]),
            content=str(note.get("content") or ""),
            actor=actor,
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
    ) -> dict[str, Any]:
        self._validate_task_text(text)
        self._validate_metadata(metadata)
        marker = "x" if status == "done" else " "
        line = f"- [{marker}] {self._render_body(text=text.strip(), metadata=metadata)}"

        with db.transaction():
            note = self._require_note_version(db, note_id=note_id, expected_note_version=expected_note_version)
            self.reconcile_note(
                db=db,
                note_id=note_id,
                note_version=expected_note_version,
                content=str(note.get("content") or ""),
                actor=self._internal_reconciliation_actor(actor),
            )
            new_content = self._append_checklist_line(str(note.get("content") or ""), line)
            db.update_note(
                note_id=note_id,
                update_data={"content": new_content},
                expected_version=expected_note_version,
            )
            updated_note = self._require_note(db, note_id)
            result = self.reconcile_note(
                db=db,
                note_id=note_id,
                note_version=int(updated_note["version"]),
                content=str(updated_note.get("content") or ""),
                actor=actor,
            )
            if not result.created_task_ids:
                raise ConflictError("Task creation did not create a task record.", entity="tasks", entity_id=note_id)
            task = db.get_task(result.created_task_ids[-1])
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
    ) -> dict[str, Any]:
        if text is not None:
            self._validate_task_text(text)
        if metadata is not None:
            self._validate_metadata(metadata)

        with db.transaction() as conn:
            task = self._require_task_version(
                db,
                task_id=task_id,
                expected_task_version=expected_task_version,
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

            projection = self._require_projection(db, task_id=task_id, conn=conn)
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
            db.update_note(
                note_id=str(note["id"]),
                update_data={"content": new_content},
                expected_version=expected_note_version,
                conn=conn,
            )
            updated_note_version = expected_note_version + 1
            updated_task = db.update_task_record(
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
    ) -> dict[str, Any]:
        with db.transaction() as conn:
            task = self._require_task_version(
                db,
                task_id=task_id,
                expected_task_version=expected_task_version,
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

            projection = self._require_projection(db, task_id=task_id, conn=conn)
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
            db.update_note(
                note_id=str(note["id"]),
                update_data={"content": new_content},
                expected_version=expected_note_version,
                conn=conn,
            )
            deleted = db.soft_delete_task(
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
        conn: TaskConnection,
    ) -> dict[str, Any]:
        task = db.task_store._fetch_task(task_id, include_deleted=False, conn=conn)
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
    def _require_projection(db: CharactersRAGDB, *, task_id: str, conn: TaskConnection) -> dict[str, Any]:
        projection = db.task_store._fetch_projection(task_id, conn=conn)
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
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", due_date) is None:
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
            if not isinstance(estimate, str) or re.fullmatch(r"\d+[mhd]", estimate) is None:
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
        match = _CHECKLIST_RE.match(raw_line)
        if match is None:
            raise ConflictError("Task projection line is no longer a checklist item.", entity="tasks")
        marker = "x" if checked else " "
        if preserve_existing_body:
            base_text = text.strip()
            body = self._render_body(text=base_text, metadata=metadata)
        else:
            body = self._render_body(text=text, metadata=metadata)
        return f"{match.group('indent')}{match.group('bullet')}{match.group('space')}[{marker}] {body}"

    @staticmethod
    def _rewrite_marker_only(*, raw_line: str, checked: bool) -> str:
        match = _CHECKLIST_RE.match(raw_line)
        if match is None:
            raise ConflictError("Task projection line is no longer a checklist item.", entity="tasks")
        marker = "x" if checked else " "
        return (
            f"{match.group('indent')}{match.group('bullet')}{match.group('space')}"
            f"[{marker}]{match.group('body_part')}"
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
        if end < len(content) and content[end] == "\n":
            end += 1
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
    ) -> dict[str, Any]:
        task_id = str(task["id"])
        metadata_json = db.task_store._json_dumps(metadata, "metadata")
        now = db._get_current_utc_timestamp_iso()
        cursor = db.task_store._execute(
            conn,
            """
            UPDATE note_tasks
               SET metadata_json = ?,
                   updated_at = ?,
                   version = version + 1
             WHERE id = ? AND version = ? AND deleted = ? AND projection_status = ?
            """,
            (
                metadata_json,
                now,
                task_id,
                expected_task_version,
                db.task_store._deleted_value(False),
                "unlinked",
            ),
        )
        if getattr(cursor, "rowcount", None) == 0:
            raise ConflictError(
                f"Task version mismatch for ID '{task_id}'. Expected {expected_task_version}.",
                entity="tasks",
                entity_id=task_id,
            )
        updated = db.task_store._fetch_task(task_id, include_deleted=True, conn=conn)
        if updated is None:
            raise ConflictError(f"Task with ID '{task_id}' not found.", entity="tasks", entity_id=task_id)
        db.record_task_event(
            task_id=task_id,
            note_id=str(updated["note_id"]),
            event_type="updated",
            actor_type=actor.actor_type,
            actor_id=actor.actor_id,
            tool_name=actor.tool_name,
            policy_mode=actor.policy_mode,
            approval_id=actor.approval_id,
            old_value={"metadata": task.get("metadata_json") or {}},
            new_value=(
                {
                    "metadata": updated.get("metadata_json") or {},
                    "idempotency_key": actor.idempotency_key,
                }
                if actor.idempotency_key
                else {"metadata": updated.get("metadata_json") or {}}
            ),
            conn=conn,
        )
        return updated
