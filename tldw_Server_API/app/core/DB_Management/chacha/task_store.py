from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any, Callable

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendConnectionWrapper,
    BackendType,
    CharactersRAGDBError,
    ConflictError,
    InputError,
    logger,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


TaskConnection = sqlite3.Connection | BackendConnectionWrapper


class TaskStore:
    """Persistence helper for task-backed note checklist records."""

    _TASK_JSON_FIELDS = ("metadata_json",)
    _EVENT_JSON_FIELDS = ("old_value_json", "new_value_json")
    _TASK_STATUSES = {"open", "done"}
    _PROJECTION_STATUSES = {"live", "unlinked", "deleted", "ambiguous"}
    _MIN_LIMIT = 1
    _MAX_LIMIT = 500

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    def _deleted_value(self, deleted: bool) -> bool | int:
        """Return the backend-native value for a soft-delete flag."""
        return deleted if self._db.backend_type == BackendType.POSTGRESQL else int(deleted)

    def _execute(
        self,
        conn: TaskConnection,
        query: str,
        params: tuple[Any, ...] | dict[str, Any] | None = None,
    ) -> Any:
        prepared_query, prepared_params = self._db._prepare_backend_statement(query, params)
        return conn.execute(prepared_query, prepared_params or ())

    def _read(
        self,
        query: str,
        params: tuple[Any, ...] | dict[str, Any] | None = None,
        *,
        conn: TaskConnection | None = None,
    ) -> Any:
        if conn is None:
            return self._db.execute_query(query, params)
        return self._execute(conn, query, params)

    def _with_transaction(self, fn: Callable[[TaskConnection], Any], conn: TaskConnection | None) -> Any:
        if conn is None:
            with self._db.transaction() as transaction_conn:
                return fn(transaction_conn)
        return fn(conn)

    @staticmethod
    def _json_dumps(value: dict[str, Any] | None, field_name: str) -> str:
        if value is None:
            return "{}"
        if not isinstance(value, dict):
            raise InputError(f"{field_name} must be a JSON object.")  # noqa: TRY003
        try:
            return json.dumps(value, sort_keys=True)
        except TypeError as exc:
            raise InputError(f"{field_name} must be JSON serializable.") from exc  # noqa: TRY003

    @staticmethod
    def _decode_row(row: Any, json_fields: tuple[str, ...]) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        for field in json_fields:
            value = item.get(field)
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                continue
            try:
                item[field] = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                logger.warning("Failed decoding task JSON field {} for row {}", field, item.get("id", "N/A"))
                item[field] = None
        return item

    def _clamp_limit(self, limit: int) -> int:
        """Normalize caller-provided row limits before passing them into SQL."""
        try:
            requested_limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise InputError("limit must be an integer.") from exc  # noqa: TRY003
        return min(max(requested_limit, self._MIN_LIMIT), self._MAX_LIMIT)

    def _decode_task_row(self, row: Any) -> dict[str, Any] | None:
        return self._decode_row(row, self._TASK_JSON_FIELDS)

    def _decode_event_row(self, row: Any) -> dict[str, Any] | None:
        return self._decode_row(row, self._EVENT_JSON_FIELDS)

    @staticmethod
    def _require_live_projection_row(task_id: str, projection: dict[str, Any] | None) -> dict[str, Any]:
        """Require an existing projection row for mutations of live projected tasks."""
        if projection is None:
            raise ConflictError(
                f"Task projection is missing for live task '{task_id}'.", entity="tasks", entity_id=task_id
            )  # noqa: TRY003
        return projection

    def _require_active_note(
        self,
        note_id: str,
        task_id: str,
        *,
        conn: TaskConnection,
    ) -> None:
        """Require a task's owning note to exist and not be soft-deleted."""
        cursor = self._read(
            "SELECT deleted FROM notes WHERE id = ?",
            (note_id,),
            conn=conn,
        )
        row = cursor.fetchone()
        if row is None:
            raise ConflictError("Task note not found.", entity="tasks", entity_id=task_id)  # noqa: TRY003
        if bool(row["deleted"]):
            raise ConflictError(
                f"Task note '{note_id}' is deleted.", entity="tasks", entity_id=task_id
            )  # noqa: TRY003

    @staticmethod
    def _raise_write_integrity_error(
        exc: sqlite3.IntegrityError,
        *,
        operation: str,
        entity_id: str,
        reference: str = "reference",
    ) -> None:
        msg = str(exc).lower()
        if "foreign key constraint failed" in msg:
            raise ConflictError(
                f"{operation} {reference} reference not found.", entity="tasks", entity_id=entity_id
            ) from exc  # noqa: TRY003
        if "unique constraint failed" in msg:
            raise ConflictError(f"{operation} already exists.", entity="tasks", entity_id=entity_id) from exc  # noqa: TRY003
        raise CharactersRAGDBError(f"Database integrity error during {operation}: {exc}") from exc  # noqa: TRY003

    @staticmethod
    def _raise_write_backend_error(
        exc: BackendDatabaseError,
        *,
        operation: str,
        entity_id: str,
        reference: str = "reference",
    ) -> None:
        msg = str(exc).lower()
        if "foreign key" in msg:
            raise ConflictError(
                f"{operation} {reference} reference not found.", entity="tasks", entity_id=entity_id
            ) from exc  # noqa: TRY003
        if "duplicate key" in msg or "unique constraint" in msg:
            raise ConflictError(f"{operation} already exists.", entity="tasks", entity_id=entity_id) from exc  # noqa: TRY003
        raise CharactersRAGDBError(f"Backend error during {operation}: {exc}") from exc  # noqa: TRY003

    def _fetch_task(
        self,
        task_id: str,
        *,
        include_deleted: bool,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any] | None:
        if include_deleted:
            cursor = self._read("SELECT * FROM note_tasks WHERE id = ?", (task_id,), conn=conn)
            return self._decode_task_row(cursor.fetchone())
        cursor = self._read(
            """
            SELECT t.*
              FROM note_tasks t
              JOIN notes n ON n.id = t.note_id
             WHERE t.id = ? AND t.deleted = ? AND n.deleted = ?
            """,
            (task_id, self._deleted_value(False), self._deleted_value(False)),
            conn=conn,
        )
        return self._decode_task_row(cursor.fetchone())

    def _fetch_projection(
        self,
        task_id: str,
        *,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any] | None:
        cursor = self._read(
            "SELECT * FROM task_note_projections WHERE task_id = ?",
            (task_id,),
            conn=conn,
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    @staticmethod
    def _validate_status(status: str) -> str:
        if status not in TaskStore._TASK_STATUSES:
            raise InputError("status must be one of ['done', 'open'].")  # noqa: TRY003
        return status

    @staticmethod
    def _validate_projection_status(status: str) -> str:
        if status not in TaskStore._PROJECTION_STATUSES:
            raise InputError(
                "projection_status must be one of ['ambiguous', 'deleted', 'live', 'unlinked']."
            )  # noqa: TRY003
        return status

    def _require_expected_version(
        self, task: dict[str, Any] | None, expected_version: int, task_id: str
    ) -> dict[str, Any]:
        if task is None:
            raise ConflictError(
                f"Task with ID '{task_id}' not found.", entity="tasks", entity_id=task_id
            )  # noqa: TRY003
        if int(task["version"]) != int(expected_version):
            raise ConflictError(
                f"Task version mismatch for ID '{task_id}'. Expected {expected_version}, found {task['version']}.",
                entity="tasks",
                entity_id=task_id,
            )  # noqa: TRY003
        return task

    @staticmethod
    def _require_active_task(task: dict[str, Any] | None, task_id: str) -> dict[str, Any]:
        if task is None:
            raise ConflictError(
                f"Task with ID '{task_id}' not found.", entity="tasks", entity_id=task_id
            )  # noqa: TRY003
        if bool(task["deleted"]):
            raise ConflictError(
                f"Task with ID '{task_id}' is deleted.", entity="tasks", entity_id=task_id
            )  # noqa: TRY003
        return task

    @staticmethod
    def _require_record_update_allowed(task: dict[str, Any], task_id: str) -> None:
        projection_status = task["projection_status"]
        if projection_status == "ambiguous":
            raise ConflictError(
                f"Task projection is ambiguous for task '{task_id}'.", entity="tasks", entity_id=task_id
            )  # noqa: TRY003
        if projection_status == "unlinked":
            raise ConflictError(
                f"Task projection is unlinked for task '{task_id}'. Record-only update mode is required.",
                entity="tasks",
                entity_id=task_id,
            )  # noqa: TRY003
        if projection_status == "deleted":
            raise ConflictError(
                f"Task projection is deleted for active task '{task_id}'.", entity="tasks", entity_id=task_id
            )  # noqa: TRY003

    def _resolve_projection_state(
        self,
        task: dict[str, Any],
        task_id: str,
        *,
        conn: TaskConnection,
    ) -> tuple[str, dict[str, Any] | None]:
        projection = self._fetch_projection(task_id, conn=conn)
        task_status = task["projection_status"]
        if projection is None:
            return task_status, None
        projection_status = projection["projection_status"]
        if task_status != projection_status:
            raise ConflictError(
                (
                    f"Task projection status mismatch for task '{task_id}': "
                    f"task row is '{task_status}', projection row is '{projection_status}'."
                ),
                entity="tasks",
                entity_id=task_id,
            )  # noqa: TRY003
        if task["note_id"] != projection["note_id"]:
            raise ConflictError(
                (
                    f"Task projection ownership mismatch for task '{task_id}': "
                    f"task row note is '{task['note_id']}', projection row note is '{projection['note_id']}'."
                ),
                entity="tasks",
                entity_id=task_id,
            )  # noqa: TRY003
        return task_status, projection

    def create_task(
        self,
        *,
        note_id: str,
        text: str,
        status: str = "open",
        metadata: dict[str, Any] | None = None,
        task_id: str | None = None,
        projection_status: str = "live",
        actor_type: str | None = None,
        actor_id: str | None = None,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any]:
        """Create a task record linked to a note."""
        normalized_text = text.strip() if isinstance(text, str) else ""
        if not normalized_text:
            raise InputError("Task text cannot be empty.")  # noqa: TRY003
        final_task_id = task_id or self._db._generate_uuid()
        normalized_status = self._validate_status(status)
        normalized_projection_status = self._validate_projection_status(projection_status)
        if normalized_projection_status == "deleted":
            raise InputError("projection_status 'deleted' is reserved for soft_delete_task.")  # noqa: TRY003
        metadata_json = self._json_dumps(metadata, "metadata")
        now = self._db._get_current_utc_timestamp_iso()
        completed_at = now if normalized_status == "done" else None

        def _execute_create(transaction_conn: TaskConnection) -> dict[str, Any]:
            self._require_active_note(note_id, final_task_id, conn=transaction_conn)
            self._execute(
                transaction_conn,
                """
                INSERT INTO note_tasks (
                    id, note_id, text, status, metadata_json, projection_status, deleted,
                    created_at, updated_at, completed_at, client_id, version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    final_task_id,
                    note_id,
                    normalized_text,
                    normalized_status,
                    metadata_json,
                    normalized_projection_status,
                    self._deleted_value(False),
                    now,
                    now,
                    completed_at,
                    self._db.client_id,
                    1,
                ),
            )
            if actor_type:
                self.record_task_event(
                    task_id=final_task_id,
                    note_id=note_id,
                    event_type="created",
                    actor_type=actor_type,
                    actor_id=actor_id,
                    new_value={
                        "text": normalized_text,
                        "status": normalized_status,
                        "metadata": metadata or {},
                    },
                    conn=transaction_conn,
                )
            task = self._fetch_task(final_task_id, include_deleted=True, conn=transaction_conn)
            if task is None:
                raise CharactersRAGDBError(f"Failed to read created task '{final_task_id}'.")  # noqa: TRY003
            return task

        try:
            return self._with_transaction(_execute_create, conn)
        except sqlite3.IntegrityError as exc:
            self._raise_integrity_error(exc, final_task_id)
        except BackendDatabaseError as exc:
            self._raise_backend_error(exc, final_task_id)
        raise CharactersRAGDBError(f"Failed to create task '{final_task_id}'.")  # noqa: TRY003

    def get_task(self, task_id: str, include_deleted: bool = False) -> dict[str, Any] | None:
        """Return one task by ID."""
        return self._fetch_task(task_id, include_deleted=include_deleted)

    def list_tasks(
        self,
        *,
        note_id: str | None = None,
        status: str | None = None,
        projection_status: str | None = None,
        include_deleted: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List tasks with optional note/status filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if note_id is not None:
            clauses.append("t.note_id = ?")
            params.append(note_id)
        if status is not None:
            clauses.append("t.status = ?")
            params.append(self._validate_status(status))
        if projection_status is not None:
            clauses.append("t.projection_status = ?")
            params.append(self._validate_projection_status(projection_status))
        if not include_deleted:
            clauses.append("t.deleted = ?")
            params.append(self._deleted_value(False))
            clauses.append("n.deleted = ?")
            params.append(self._deleted_value(False))
        query = "SELECT t.* FROM note_tasks t"
        if not include_deleted:
            query += " JOIN notes n ON n.id = t.note_id"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY t.created_at ASC, t.id ASC LIMIT ?"
        params.append(self._clamp_limit(limit))
        cursor = self._read(query, tuple(params))
        return [self._decode_task_row(row) for row in cursor.fetchall()]

    def update_task_record(
        self,
        *,
        task_id: str,
        expected_version: int,
        text: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
        projection_status: str | None = None,
        actor_type: str | None = None,
        actor_id: str | None = None,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any]:
        """Update mutable task fields using optimistic locking."""
        if text is not None and not text.strip():
            raise InputError("Task text cannot be empty.")  # noqa: TRY003
        if projection_status is not None:
            raise InputError(
                "projection_status cannot be updated with update_task_record; use projection helpers."
            )  # noqa: TRY003
        normalized_status = self._validate_status(status) if status is not None else None
        metadata_json = self._json_dumps(metadata, "metadata") if metadata is not None else None

        def _execute_update(transaction_conn: TaskConnection) -> dict[str, Any]:
            old = self._require_active_task(
                self._require_expected_version(
                    self._fetch_task(task_id, include_deleted=True, conn=transaction_conn),
                    expected_version,
                    task_id,
                ),
                task_id,
            )
            self._require_active_note(old["note_id"], task_id, conn=transaction_conn)
            projection_state, _ = self._resolve_projection_state(old, task_id, conn=transaction_conn)
            old["projection_status"] = projection_state
            self._require_record_update_allowed(old, task_id)
            now = self._db._get_current_utc_timestamp_iso()
            new_text = text.strip() if text is not None else old["text"]
            new_status = normalized_status or old["status"]
            new_metadata_json = (
                metadata_json if metadata_json is not None else json.dumps(old["metadata_json"], sort_keys=True)
            )
            completed_at = old.get("completed_at")
            if new_status == "done" and old["status"] != "done":
                completed_at = now
            elif new_status == "open":
                completed_at = None

            update_cursor = self._execute(
                transaction_conn,
                """
                UPDATE note_tasks
                   SET text = ?,
                       status = ?,
                       metadata_json = ?,
                       updated_at = ?,
                       completed_at = ?,
                       version = version + 1
                 WHERE id = ? AND version = ?
                """,
                (
                    new_text,
                    new_status,
                    new_metadata_json,
                    now,
                    completed_at,
                    task_id,
                    expected_version,
                ),
            )
            if getattr(update_cursor, "rowcount", None) == 0:
                raise ConflictError(
                    f"Task version mismatch for ID '{task_id}'. Expected {expected_version}.",
                    entity="tasks",
                    entity_id=task_id,
                )  # noqa: TRY003
            updated = self._fetch_task(task_id, include_deleted=True, conn=transaction_conn)
            if updated is None:
                raise ConflictError(
                    f"Task with ID '{task_id}' not found.", entity="tasks", entity_id=task_id
                )  # noqa: TRY003
            if actor_type and old["status"] != updated["status"]:
                self.record_task_event(
                    task_id=task_id,
                    note_id=updated["note_id"],
                    event_type="status_changed",
                    actor_type=actor_type,
                    actor_id=actor_id,
                    old_value={"status": old["status"]},
                    new_value={"status": updated["status"]},
                    conn=transaction_conn,
                )
            elif actor_type and (old["text"] != updated["text"] or old["metadata_json"] != updated["metadata_json"]):
                self.record_task_event(
                    task_id=task_id,
                    note_id=updated["note_id"],
                    event_type="updated",
                    actor_type=actor_type,
                    actor_id=actor_id,
                    old_value={"text": old["text"], "metadata": old["metadata_json"]},
                    new_value={"text": updated["text"], "metadata": updated["metadata_json"]},
                    conn=transaction_conn,
                )
            return updated

        return self._with_transaction(_execute_update, conn)

    def set_task_projection(
        self,
        *,
        task_id: str,
        note_id: str,
        note_version: int,
        line_number: int,
        start_offset: int,
        end_offset: int,
        normalized_text_hash: str,
        occurrence_index: int,
        block_fingerprint: str,
        raw_line: str,
        has_child_content: bool,
        projection_status: str = "live",
        conn: TaskConnection | None = None,
    ) -> dict[str, Any]:
        """Create or replace a task's markdown projection locator."""
        normalized_projection_status = self._validate_projection_status(projection_status)
        if normalized_projection_status == "deleted":
            raise InputError("projection_status 'deleted' is reserved for soft_delete_task.")  # noqa: TRY003
        now = self._db._get_current_utc_timestamp_iso()

        def _execute_projection(transaction_conn: TaskConnection) -> dict[str, Any]:
            task = self._require_active_task(
                self._fetch_task(task_id, include_deleted=True, conn=transaction_conn),
                task_id,
            )
            self._require_active_note(task["note_id"], task_id, conn=transaction_conn)
            projection_state, _ = self._resolve_projection_state(task, task_id, conn=transaction_conn)
            if task["note_id"] != note_id:
                raise ConflictError(
                    f"Task projection note does not match owning note for task '{task_id}'.",
                    entity="tasks",
                    entity_id=task_id,
                )  # noqa: TRY003
            update_cursor = self._execute(
                transaction_conn,
                """
                UPDATE note_tasks
                   SET projection_status = ?,
                       updated_at = ?,
                       version = CASE
                           WHEN projection_status != ? THEN version + 1
                           ELSE version
                       END
                 WHERE id = ? AND note_id = ? AND deleted = ? AND projection_status = ?
                """,
                (
                    normalized_projection_status,
                    now,
                    normalized_projection_status,
                    task_id,
                    note_id,
                    self._deleted_value(False),
                    projection_state,
                ),
            )
            if getattr(update_cursor, "rowcount", None) == 0:
                raise ConflictError(
                    f"Task projection changed concurrently for task '{task_id}'.",
                    entity="tasks",
                    entity_id=task_id,
                )  # noqa: TRY003
            self._execute(
                transaction_conn,
                """
                INSERT INTO task_note_projections (
                    task_id, note_id, note_version, line_number, start_offset, end_offset,
                    normalized_text_hash, occurrence_index, block_fingerprint, raw_line,
                    has_child_content, projection_status, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    note_id = excluded.note_id,
                    note_version = excluded.note_version,
                    line_number = excluded.line_number,
                    start_offset = excluded.start_offset,
                    end_offset = excluded.end_offset,
                    normalized_text_hash = excluded.normalized_text_hash,
                    occurrence_index = excluded.occurrence_index,
                    block_fingerprint = excluded.block_fingerprint,
                    raw_line = excluded.raw_line,
                    has_child_content = excluded.has_child_content,
                    projection_status = excluded.projection_status,
                    updated_at = excluded.updated_at
                """,
                (
                    task_id,
                    note_id,
                    int(note_version),
                    int(line_number),
                    int(start_offset),
                    int(end_offset),
                    normalized_text_hash,
                    int(occurrence_index),
                    block_fingerprint,
                    raw_line,
                    self._deleted_value(bool(has_child_content)),
                    normalized_projection_status,
                    now,
                ),
            )
            projection = self._fetch_projection(task_id, conn=transaction_conn)
            if projection is None:
                raise CharactersRAGDBError(f"Failed to read projection for task '{task_id}'.")  # noqa: TRY003
            return projection

        return self._with_transaction(_execute_projection, conn)

    def mark_task_unlinked(
        self,
        *,
        task_id: str,
        expected_version: int,
        actor_type: str | None = None,
        actor_id: str | None = None,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any]:
        """Mark a task's projection as unlinked after reconciliation."""

        def _execute_unlink(transaction_conn: TaskConnection) -> dict[str, Any]:
            old = self._require_active_task(
                self._require_expected_version(
                    self._fetch_task(task_id, include_deleted=True, conn=transaction_conn),
                    expected_version,
                    task_id,
                ),
                task_id,
            )
            self._require_active_note(old["note_id"], task_id, conn=transaction_conn)
            projection_state, projection = self._resolve_projection_state(old, task_id, conn=transaction_conn)
            if projection_state != "live":
                raise ConflictError(
                    f"Task projection is {projection_state} for task '{task_id}'.",
                    entity="tasks",
                    entity_id=task_id,
                )  # noqa: TRY003
            self._require_live_projection_row(task_id, projection)
            now = self._db._get_current_utc_timestamp_iso()
            update_cursor = self._execute(
                transaction_conn,
                """
                UPDATE note_tasks
                   SET projection_status = ?,
                       updated_at = ?,
                       version = version + 1
                 WHERE id = ? AND version = ? AND projection_status = ?
                """,
                ("unlinked", now, task_id, expected_version, projection_state),
            )
            if getattr(update_cursor, "rowcount", None) == 0:
                raise ConflictError(
                    f"Task version mismatch for ID '{task_id}'. Expected {expected_version}.",
                    entity="tasks",
                    entity_id=task_id,
                )  # noqa: TRY003
            projection_update_cursor = self._execute(
                transaction_conn,
                """
                UPDATE task_note_projections
                   SET projection_status = ?,
                       updated_at = ?
                 WHERE task_id = ? AND projection_status = ?
                """,
                ("unlinked", now, task_id, projection_state),
            )
            if getattr(projection_update_cursor, "rowcount", None) == 0:
                raise ConflictError(
                    f"Task projection changed concurrently for task '{task_id}'.",
                    entity="tasks",
                    entity_id=task_id,
                )  # noqa: TRY003
            updated = self._fetch_task(task_id, include_deleted=True, conn=transaction_conn)
            if actor_type:
                self.record_task_event(
                    task_id=task_id,
                    note_id=old["note_id"],
                    event_type="unlinked",
                    actor_type=actor_type,
                    actor_id=actor_id,
                    old_value={"projection_status": old["projection_status"]},
                    new_value={"projection_status": "unlinked"},
                    conn=transaction_conn,
                )
            if updated is None:
                raise CharactersRAGDBError(f"Failed to read unlinked task '{task_id}'.")  # noqa: TRY003
            return updated

        return self._with_transaction(_execute_unlink, conn)

    def soft_delete_task(
        self,
        *,
        task_id: str,
        expected_version: int,
        projection_note_id: str | None = None,
        projection_note_version: int | None = None,
        projection_line_number: int | None = None,
        allow_record_only: bool = False,
        actor_type: str | None = None,
        actor_id: str | None = None,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any]:
        """Soft-delete a task, requiring projection context for live projected rows."""

        def _execute_delete(transaction_conn: TaskConnection) -> dict[str, Any]:
            old = self._require_active_task(
                self._require_expected_version(
                    self._fetch_task(task_id, include_deleted=True, conn=transaction_conn),
                    expected_version,
                    task_id,
                ),
                task_id,
            )
            self._require_active_note(old["note_id"], task_id, conn=transaction_conn)
            projection_status, projection = self._resolve_projection_state(old, task_id, conn=transaction_conn)
            if projection_status == "ambiguous":
                raise ConflictError(
                    f"Task projection is ambiguous for task '{task_id}'.", entity="tasks", entity_id=task_id
                )  # noqa: TRY003
            if projection_status == "unlinked" and not allow_record_only:
                raise ConflictError(
                    f"Task projection is unlinked for task '{task_id}'. Record-only delete mode is required.",
                    entity="tasks",
                    entity_id=task_id,
                )  # noqa: TRY003
            if projection_status == "deleted":
                raise ConflictError(
                    f"Task projection is deleted for active task '{task_id}'.", entity="tasks", entity_id=task_id
                )  # noqa: TRY003
            if projection_status == "live" and not allow_record_only:
                projection = self._require_live_projection_row(task_id, projection)
                matches_projection = (
                    projection_note_id == projection["note_id"]
                    and projection_note_version == projection["note_version"]
                    and projection_line_number == projection["line_number"]
                )
                if not matches_projection:
                    raise InputError(
                        "Task projection deletion is ambiguous without a matching projection locator."
                    )  # noqa: TRY003
            now = self._db._get_current_utc_timestamp_iso()
            update_cursor = self._execute(
                transaction_conn,
                """
                UPDATE note_tasks
                   SET deleted = ?,
                       projection_status = ?,
                       updated_at = ?,
                       version = version + 1
                 WHERE id = ? AND version = ? AND deleted = ? AND projection_status = ?
                """,
                (
                    self._deleted_value(True),
                    "deleted",
                    now,
                    task_id,
                    expected_version,
                    self._deleted_value(False),
                    projection_status,
                ),
            )
            if getattr(update_cursor, "rowcount", None) == 0:
                raise ConflictError(
                    f"Task version mismatch for ID '{task_id}'. Expected {expected_version}.",
                    entity="tasks",
                    entity_id=task_id,
                )  # noqa: TRY003
            projection_update_cursor = self._execute(
                transaction_conn,
                """
                UPDATE task_note_projections
                   SET projection_status = ?,
                       updated_at = ?
                 WHERE task_id = ?
                """,
                ("deleted", now, task_id),
            )
            if projection_status == "live" and not allow_record_only:
                if getattr(projection_update_cursor, "rowcount", None) == 0:
                    raise ConflictError(
                        f"Task projection changed concurrently for task '{task_id}'.",
                        entity="tasks",
                        entity_id=task_id,
                    )  # noqa: TRY003
            updated = self._fetch_task(task_id, include_deleted=True, conn=transaction_conn)
            if actor_type:
                self.record_task_event(
                    task_id=task_id,
                    note_id=old["note_id"],
                    event_type="deleted",
                    actor_type=actor_type,
                    actor_id=actor_id,
                    old_value={"deleted": old["deleted"], "projection_status": old["projection_status"]},
                    new_value={"deleted": True, "projection_status": "deleted"},
                    conn=transaction_conn,
                )
            if updated is None:
                raise CharactersRAGDBError(f"Failed to read deleted task '{task_id}'.")  # noqa: TRY003
            return updated

        return self._with_transaction(_execute_delete, conn)

    def record_task_event(
        self,
        *,
        event_type: str,
        actor_type: str,
        task_id: str | None = None,
        note_id: str | None = None,
        actor_id: str | None = None,
        tool_name: str | None = None,
        policy_mode: str | None = None,
        approval_id: str | None = None,
        old_value: dict[str, Any] | None = None,
        new_value: dict[str, Any] | None = None,
        event_id: str | None = None,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any]:
        """Append a task activity/audit event."""
        final_event_id = event_id or self._db._generate_uuid()
        now = self._db._get_current_utc_timestamp_iso()
        old_value_json = self._json_dumps(old_value, "old_value") if old_value is not None else None
        new_value_json = self._json_dumps(new_value, "new_value") if new_value is not None else None

        def _execute_event(transaction_conn: TaskConnection) -> dict[str, Any]:
            self._execute(
                transaction_conn,
                """
                INSERT INTO task_events (
                    id, task_id, note_id, event_type, actor_type, actor_id, tool_name,
                    policy_mode, approval_id, old_value_json, new_value_json, created_at, client_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    final_event_id,
                    task_id,
                    note_id,
                    event_type,
                    actor_type,
                    actor_id,
                    tool_name,
                    policy_mode,
                    approval_id,
                    old_value_json,
                    new_value_json,
                    now,
                    self._db.client_id,
                ),
            )
            cursor = self._read("SELECT * FROM task_events WHERE id = ?", (final_event_id,), conn=transaction_conn)
            event = self._decode_event_row(cursor.fetchone())
            if event is None:
                raise CharactersRAGDBError(f"Failed to read task event '{final_event_id}'.")  # noqa: TRY003
            return event

        try:
            return self._with_transaction(_execute_event, conn)
        except sqlite3.IntegrityError as exc:
            self._raise_write_integrity_error(
                exc,
                operation="Task event",
                entity_id=final_event_id,
                reference="task or note",
            )
        except BackendDatabaseError as exc:
            self._raise_write_backend_error(
                exc,
                operation="Task event",
                entity_id=final_event_id,
                reference="task or note",
            )
        raise CharactersRAGDBError(f"Failed to record task event '{final_event_id}'.")  # noqa: TRY003

    def list_task_activity(
        self,
        *,
        task_id: str | None = None,
        note_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List task events by task or note scope."""
        clauses: list[str] = []
        params: list[Any] = []
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if note_id is not None:
            clauses.append("note_id = ?")
            params.append(note_id)
        query = "SELECT * FROM task_events"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        if self._db.backend_type == BackendType.SQLITE:
            query += " ORDER BY created_at ASC, rowid ASC LIMIT ?"
        else:
            query += " ORDER BY created_at ASC, id ASC LIMIT ?"
        params.append(self._clamp_limit(limit))
        cursor = self._read(query, tuple(params))
        return [self._decode_event_row(row) for row in cursor.fetchall()]

    def mark_task_activity_read(
        self,
        event_id: str,
        *,
        user_id: str,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any]:
        """Mark an activity event read for one user."""
        now = self._db._get_current_utc_timestamp_iso()

        def _execute_read(transaction_conn: TaskConnection) -> dict[str, Any]:
            self._execute(
                transaction_conn,
                """
                INSERT INTO task_event_read_state (event_id, user_id, read_at, dismissed_at)
                VALUES (?, ?, ?, NULL)
                ON CONFLICT(event_id, user_id) DO UPDATE SET
                    read_at = COALESCE(task_event_read_state.read_at, excluded.read_at)
                """,
                (event_id, user_id, now),
            )
            state = self.get_task_activity_read_state(event_id, user_id=user_id, conn=transaction_conn)
            if state is None:
                raise CharactersRAGDBError(f"Failed to read task event read state for '{event_id}'.")  # noqa: TRY003
            return state

        try:
            return self._with_transaction(_execute_read, conn)
        except sqlite3.IntegrityError as exc:
            self._raise_write_integrity_error(
                exc,
                operation="Task event read state",
                entity_id=event_id,
                reference="event",
            )
        except BackendDatabaseError as exc:
            self._raise_write_backend_error(
                exc,
                operation="Task event read state",
                entity_id=event_id,
                reference="event",
            )
        raise CharactersRAGDBError(f"Failed to mark task event '{event_id}' read.")  # noqa: TRY003

    def mark_task_activity_dismissed(
        self,
        event_id: str,
        *,
        user_id: str,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any]:
        """Mark an activity event dismissed for one user."""
        now = self._db._get_current_utc_timestamp_iso()

        def _execute_dismiss(transaction_conn: TaskConnection) -> dict[str, Any]:
            self._execute(
                transaction_conn,
                """
                INSERT INTO task_event_read_state (event_id, user_id, read_at, dismissed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(event_id, user_id) DO UPDATE SET
                    read_at = COALESCE(task_event_read_state.read_at, excluded.read_at),
                    dismissed_at = excluded.dismissed_at
                """,
                (event_id, user_id, now, now),
            )
            state = self.get_task_activity_read_state(event_id, user_id=user_id, conn=transaction_conn)
            if state is None:
                raise CharactersRAGDBError(f"Failed to read task event read state for '{event_id}'.")  # noqa: TRY003
            return state

        try:
            return self._with_transaction(_execute_dismiss, conn)
        except sqlite3.IntegrityError as exc:
            self._raise_write_integrity_error(
                exc,
                operation="Task event read state",
                entity_id=event_id,
                reference="event",
            )
        except BackendDatabaseError as exc:
            self._raise_write_backend_error(
                exc,
                operation="Task event read state",
                entity_id=event_id,
                reference="event",
            )
        raise CharactersRAGDBError(f"Failed to mark task event '{event_id}' dismissed.")  # noqa: TRY003

    def get_task_activity_read_state(
        self,
        event_id: str,
        *,
        user_id: str,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any] | None:
        """Return read/dismiss state for one event and user."""
        cursor = self._read(
            """
            SELECT * FROM task_event_read_state
             WHERE event_id = ? AND user_id = ?
            """,
            (event_id, user_id),
            conn=conn,
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_reconciliation_state(self, note_id: str) -> dict[str, Any] | None:
        """Return the last task reconciliation state for a note."""
        cursor = self._read(
            "SELECT * FROM note_task_reconciliation_state WHERE note_id = ?",
            (note_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def set_reconciliation_state(
        self,
        *,
        note_id: str,
        note_version: int,
        status: str,
        item_count: int,
        warning_count: int,
        cursor: str | None = None,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any]:
        """Upsert task reconciliation progress for a note/version."""
        now = self._db._get_current_utc_timestamp_iso()

        def _execute_state(transaction_conn: TaskConnection) -> dict[str, Any]:
            self._execute(
                transaction_conn,
                """
                INSERT INTO note_task_reconciliation_state (
                    note_id, note_version, status, reconciled_at, item_count, warning_count, cursor
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(note_id) DO UPDATE SET
                    note_version = excluded.note_version,
                    status = excluded.status,
                    reconciled_at = excluded.reconciled_at,
                    item_count = excluded.item_count,
                    warning_count = excluded.warning_count,
                    cursor = excluded.cursor
                """,
                (
                    note_id,
                    int(note_version),
                    status,
                    now,
                    int(item_count),
                    int(warning_count),
                    cursor,
                ),
            )
            state = (
                self.get_reconciliation_state(note_id)
                if conn is None
                else self._get_reconciliation_state_in_conn(note_id, transaction_conn)
            )
            if state is None:
                raise CharactersRAGDBError(f"Failed to read reconciliation state for note '{note_id}'.")  # noqa: TRY003
            return state

        try:
            return self._with_transaction(_execute_state, conn)
        except sqlite3.IntegrityError as exc:
            self._raise_write_integrity_error(
                exc,
                operation="Task reconciliation state",
                entity_id=note_id,
                reference="note",
            )
        except BackendDatabaseError as exc:
            self._raise_write_backend_error(
                exc,
                operation="Task reconciliation state",
                entity_id=note_id,
                reference="note",
            )
        raise CharactersRAGDBError(f"Failed to set reconciliation state for note '{note_id}'.")  # noqa: TRY003

    def _get_reconciliation_state_in_conn(self, note_id: str, conn: TaskConnection) -> dict[str, Any] | None:
        cursor = self._read(
            "SELECT * FROM note_task_reconciliation_state WHERE note_id = ?",
            (note_id,),
            conn=conn,
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def candidate_notes_for_task_discovery(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return checklist-bearing notes whose task reconciliation is stale or missing."""
        cursor = self._read(
            """
            SELECT n.id, n.title, n.version, n.last_modified
              FROM notes n
              LEFT JOIN note_task_reconciliation_state r ON r.note_id = n.id
             WHERE n.deleted = ?
               AND (
                    n.content LIKE ?
                 OR n.content LIKE ?
                 OR n.content LIKE ?
               )
               AND (
                    r.note_id IS NULL
                 OR r.note_version < n.version
                 OR r.status != ?
               )
             ORDER BY n.last_modified DESC, n.id ASC
             LIMIT ?
            """,
            (
                self._deleted_value(False),
                "%- [ ]%",
                "%- [x]%",
                "%- [X]%",
                "clean",
                self._clamp_limit(limit),
            ),
        )
        return [dict(row) for row in cursor.fetchall()]

    def _raise_integrity_error(self, exc: sqlite3.IntegrityError, task_id: str) -> None:
        msg = str(exc).lower()
        if "foreign key constraint failed" in msg:
            raise ConflictError("Task note not found.", entity="tasks", entity_id=task_id) from exc  # noqa: TRY003
        if "unique constraint failed" in msg:
            raise ConflictError(
                f"Task with ID '{task_id}' already exists.", entity="tasks", entity_id=task_id
            ) from exc  # noqa: TRY003
        raise CharactersRAGDBError(f"Database integrity error for task '{task_id}': {exc}") from exc  # noqa: TRY003

    def _raise_backend_error(self, exc: BackendDatabaseError, task_id: str) -> None:
        msg = str(exc).lower()
        if "foreign key" in msg:
            raise ConflictError("Task note not found.", entity="tasks", entity_id=task_id) from exc  # noqa: TRY003
        if "duplicate key" in msg or "unique constraint" in msg:
            raise ConflictError(
                f"Task with ID '{task_id}' already exists.", entity="tasks", entity_id=task_id
            ) from exc  # noqa: TRY003
        raise CharactersRAGDBError(f"Backend error for task '{task_id}': {exc}") from exc  # noqa: TRY003
