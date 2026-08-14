from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendConnectionWrapper,
    BackendType,
    CharactersRAGDBError,
    ConflictError,
    InputError,
    logger,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


TaskConnection = sqlite3.Connection | BackendConnectionWrapper


class TaskStore:
    """Persistence helper for task-backed note checklist records."""

    _TASK_JSON_FIELDS = ("metadata_json",)
    _EVENT_JSON_FIELDS = ("old_value_json", "new_value_json")
    _CHECKLIST_DISCOVERY_MARKER_PATTERNS = ("%[ ]%", "%[x]%", "%[X]%")
    _CHECKLIST_DISCOVERY_BULLET_PATTERNS = ("%-%", "%*%", "%+%")
    _TASK_STATUSES = {"open", "done"}
    _PROJECTION_STATUSES = {"live", "unlinked", "deleted", "ambiguous"}
    _MIN_LIMIT = 1
    _MAX_LIMIT = 500
    _LOCAL_UNBOUND = "local-unbound"

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    def _scope(
        self,
        owner_user_id: str,
        dataset_id: str,
    ) -> tuple[str, str]:
        """Validate one exact canonical owner/dataset scope."""
        owner = str(owner_user_id).strip()
        dataset = str(dataset_id).strip()
        if not owner or not dataset:
            raise InputError("Task owner and dataset scope cannot be empty.")  # noqa: TRY003
        if self._db.backend_type == BackendType.POSTGRESQL and (
            owner != str(self._db.client_id) or dataset != self._LOCAL_UNBOUND
        ):
            raise ConflictError(
                "Task scope is unavailable on PostgreSQL schema v59.",
                entity="tasks",
                entity_id=owner,
            )  # noqa: TRY003
        return owner, dataset

    @property
    def _uses_postgres_v59_schema(self) -> bool:
        return self._db.backend_type == BackendType.POSTGRESQL

    def _canonical_task_values(self, task: Mapping[str, Any]) -> tuple[int, str, str | None, str | None]:
        revision = int(task.get("canonical_revision") or task.get("version") or 1)
        source = dict(task)
        if isinstance(source.get("metadata_json"), dict):
            source["metadata_json"] = self._json_dumps(source["metadata_json"], "metadata")
        object_hash, code, diagnostic_hash = self._db._canonicalize_legacy_task_v60(
            source,
            owner_user_id=str(source["owner_user_id"]),
        )
        return revision, object_hash, code, diagnostic_hash

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
    def _event_value(value: dict[str, Any], *, idempotency_key: str | None = None) -> dict[str, Any]:
        if idempotency_key is None:
            return value
        return {**value, "idempotency_key": idempotency_key}

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

    @staticmethod
    def _normalize_offset(offset: int) -> int:
        """Normalize caller-provided offsets before passing them into SQL."""
        try:
            requested_offset = int(offset)
        except (TypeError, ValueError) as exc:
            raise InputError("offset must be an integer.") from exc  # noqa: TRY003
        if requested_offset < 0:
            raise InputError("offset must be >= 0.")  # noqa: TRY003
        return requested_offset

    @staticmethod
    def _escape_like(value: str) -> str:
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

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
        owner_user_id: str,
        conn: TaskConnection,
    ) -> None:
        """Require a task's owning note to exist and not be soft-deleted."""
        cursor = self._read(
            "SELECT deleted FROM notes WHERE client_id = ? AND id = ?",
            (owner_user_id, note_id),
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
        owner_user_id: str,
        dataset_id: str,
        include_deleted: bool,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any] | None:
        if self._uses_postgres_v59_schema:
            if include_deleted:
                cursor = self._read(
                    "SELECT * FROM note_tasks WHERE client_id = ? AND id = ?",
                    (owner_user_id, task_id),
                    conn=conn,
                )
                return self._decode_task_row(cursor.fetchone())
            cursor = self._read(
                """
                SELECT t.*
                  FROM note_tasks t
                  JOIN notes n ON n.id = t.note_id AND n.client_id = t.client_id
                 WHERE t.client_id = ? AND t.id = ? AND t.deleted = ? AND n.deleted = ?
                """,
                (owner_user_id, task_id, self._deleted_value(False), self._deleted_value(False)),
                conn=conn,
            )
            return self._decode_task_row(cursor.fetchone())
        if include_deleted:
            cursor = self._read(
                "SELECT * FROM note_tasks WHERE owner_user_id = ? AND dataset_id = ? AND id = ?",
                (owner_user_id, dataset_id, task_id),
                conn=conn,
            )
            return self._decode_task_row(cursor.fetchone())
        cursor = self._read(
            """
            SELECT t.*
              FROM note_tasks t
              JOIN notes n ON n.id = t.note_id
             WHERE t.owner_user_id = ? AND t.dataset_id = ? AND t.id = ?
               AND n.client_id = t.owner_user_id AND t.deleted = ? AND n.deleted = ?
            """,
            (owner_user_id, dataset_id, task_id, self._deleted_value(False), self._deleted_value(False)),
            conn=conn,
        )
        return self._decode_task_row(cursor.fetchone())

    def _fetch_projection(
        self,
        task_id: str,
        *,
        owner_user_id: str,
        dataset_id: str,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any] | None:
        if self._uses_postgres_v59_schema:
            cursor = self._read(
                """
                SELECT p.*
                  FROM task_note_projections p
                  JOIN note_tasks t ON t.id = p.task_id
                 WHERE t.client_id = ? AND p.task_id = ?
                """,
                (owner_user_id, task_id),
                conn=conn,
            )
        else:
            cursor = self._read(
                "SELECT * FROM task_note_projections WHERE owner_user_id = ? AND dataset_id = ? AND task_id = ?",
                (owner_user_id, dataset_id, task_id),
                conn=conn,
            )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_task_projection(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        task_id: str,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any] | None:
        """Return the markdown projection row for one task."""
        owner, dataset = self._scope(owner_user_id, dataset_id)
        return self._fetch_projection(task_id, owner_user_id=owner, dataset_id=dataset, conn=conn)

    def get_note_reconciliation_snapshot(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        note_id: str,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any] | None:
        """Return note fields needed to validate task reconciliation input."""
        owner, _dataset = self._scope(owner_user_id, dataset_id)
        cursor = self._read(
            "SELECT id, version, content, deleted FROM notes WHERE client_id = ? AND id = ?",
            (owner, note_id),
            conn=conn,
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_live_projected_tasks(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        note_id: str,
        conn: TaskConnection | None = None,
    ) -> list[dict[str, dict[str, Any]]]:
        """Return live tasks for a note together with their projection rows.

        A task row that claims to be live without a projection row is an
        inconsistent projection state. Reconciliation must fail closed in that
        case rather than creating a replacement task beside an orphaned live row.
        """
        owner, dataset = self._scope(owner_user_id, dataset_id)
        if self._uses_postgres_v59_schema:
            cursor = self._read(
                """
                SELECT *
                  FROM note_tasks
                 WHERE client_id = ? AND note_id = ? AND deleted = ? AND projection_status = ?
                 ORDER BY created_at ASC, id ASC
                """,
                (owner, note_id, self._deleted_value(False), "live"),
                conn=conn,
            )
        else:
            cursor = self._read(
                """
                SELECT *
                  FROM note_tasks
                 WHERE owner_user_id = ? AND dataset_id = ? AND note_id = ?
                   AND deleted = ? AND projection_status = ?
                 ORDER BY created_at ASC, id ASC
                """,
                (owner, dataset, note_id, self._deleted_value(False), "live"),
                conn=conn,
            )
        projected_tasks: list[dict[str, dict[str, Any]]] = []
        for row in cursor.fetchall():
            task = self._decode_task_row(row)
            if task is None:
                continue
            projection_state, projection = self._resolve_projection_state(
                task,
                task["id"],
                owner_user_id=owner,
                dataset_id=dataset,
                conn=conn,
            )
            if projection_state != "live":
                raise ConflictError(
                    f"Task projection is {projection_state} for live task '{task['id']}'.",
                    entity="tasks",
                    entity_id=task["id"],
                )  # noqa: TRY003
            projected_tasks.append(
                {
                    "task": task,
                    "projection": self._require_live_projection_row(task["id"], projection),
                }
            )
        return projected_tasks

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
        owner_user_id: str,
        dataset_id: str,
        conn: TaskConnection,
    ) -> tuple[str, dict[str, Any] | None]:
        projection = self._fetch_projection(
            task_id,
            owner_user_id=owner_user_id,
            dataset_id=dataset_id,
            conn=conn,
        )
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
        owner_user_id: str,
        dataset_id: str,
        note_id: str,
        text: str,
        status: str = "open",
        metadata: dict[str, Any] | None = None,
        task_id: str | None = None,
        projection_status: str = "live",
        actor_type: str | None = None,
        actor_id: str | None = None,
        tool_name: str | None = None,
        policy_mode: str | None = None,
        approval_id: str | None = None,
        idempotency_key: str | None = None,
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
        owner, dataset = self._scope(owner_user_id, dataset_id)
        canonical_values = None
        if not self._uses_postgres_v59_schema:
            canonical_values = self._canonical_task_values(
                {
                    "owner_user_id": owner,
                    "id": final_task_id,
                    "note_id": note_id,
                    "text": normalized_text,
                    "status": normalized_status,
                    "metadata_json": metadata_json,
                    "projection_status": normalized_projection_status,
                    "deleted": 0,
                    "created_at": now,
                    "updated_at": now,
                    "completed_at": completed_at,
                    "client_id": self._db.client_id,
                    "version": 1,
                    "canonical_revision": 1,
                }
            )

        def _execute_create(transaction_conn: TaskConnection) -> dict[str, Any]:
            self._require_active_note(note_id, final_task_id, owner_user_id=owner, conn=transaction_conn)
            if self._uses_postgres_v59_schema:
                self._execute(
                    transaction_conn,
                    """
                    INSERT INTO note_tasks (
                        id, note_id, text, status, metadata_json, projection_status,
                        deleted, created_at, updated_at, completed_at, client_id, version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        owner,
                        1,
                    ),
                )
            else:
                assert canonical_values is not None  # nosec B101
                canonical_revision, canonical_hash, diagnostic_code, diagnostic_hash = canonical_values
                self._execute(
                    transaction_conn,
                    """
                    INSERT INTO note_tasks (
                        owner_user_id, dataset_id, id, note_id, text, status, metadata_json,
                        projection_status, deleted, created_at, updated_at, completed_at, client_id,
                        version, canonical_revision, canonical_hash, source_diagnostic_code,
                        source_diagnostic_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        owner, dataset, final_task_id, note_id, normalized_text, normalized_status,
                        metadata_json, normalized_projection_status, self._deleted_value(False),
                        now, now, completed_at, self._db.client_id, 1, canonical_revision,
                        canonical_hash, diagnostic_code, diagnostic_hash,
                    ),
                )
            if actor_type:
                self.record_task_event(
                    task_id=final_task_id,
                    note_id=note_id,
                    owner_user_id=owner,
                    dataset_id=dataset,
                    event_type="created",
                    actor_type=actor_type,
                    actor_id=actor_id,
                    tool_name=tool_name,
                    policy_mode=policy_mode,
                    approval_id=approval_id,
                    new_value=self._event_value(
                        {
                            "text": normalized_text,
                            "status": normalized_status,
                            "metadata": metadata or {},
                        },
                        idempotency_key=idempotency_key,
                    ),
                    conn=transaction_conn,
                )
            task = self._fetch_task(
                final_task_id,
                owner_user_id=owner,
                dataset_id=dataset,
                include_deleted=True,
                conn=transaction_conn,
            )
            if task is None:
                raise CharactersRAGDBError(f"Failed to read created task '{final_task_id}'.")  # noqa: TRY003
            return task

        try:
            return self._with_transaction(_execute_create, conn)
        except sqlite3.IntegrityError as exc:
            self._raise_integrity_error(exc, final_task_id)
        except BackendDatabaseError as exc:
            self._raise_backend_error(exc, final_task_id)

    def get_task(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        task_id: str,
        include_deleted: bool = False,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any] | None:
        """Return one task by ID."""
        owner, dataset = self._scope(owner_user_id, dataset_id)
        return self._fetch_task(
            task_id,
            owner_user_id=owner,
            dataset_id=dataset,
            include_deleted=include_deleted,
            conn=conn,
        )

    def list_tasks(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        note_id: str | None = None,
        status: str | None = None,
        projection_status: str | None = None,
        include_deleted: bool = False,
        include_unlinked: bool = True,
        query: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List tasks with optional note/status filters."""
        owner, dataset = self._scope(owner_user_id, dataset_id)
        if self._uses_postgres_v59_schema:
            clauses: list[str] = ["t.client_id = ?"]
            params: list[Any] = [owner]
        else:
            clauses = ["t.owner_user_id = ?", "t.dataset_id = ?"]
            params = [owner, dataset]
        if note_id is not None:
            clauses.append("t.note_id = ?")
            params.append(note_id)
        if status is not None:
            clauses.append("t.status = ?")
            params.append(self._validate_status(status))
        if projection_status is not None:
            clauses.append("t.projection_status = ?")
            params.append(self._validate_projection_status(projection_status))
        elif not include_unlinked:
            clauses.append("t.projection_status != ?")
            params.append("unlinked")
        if query is not None and str(query).strip():
            escaped_query = self._escape_like(str(query).strip().lower())
            clauses.append("LOWER(t.text) LIKE ? ESCAPE '\\'")
            params.append(f"%{escaped_query}%")
        if metadata_filters is not None:
            if not isinstance(metadata_filters, dict):
                raise InputError("metadata_filters must be a JSON object.")  # noqa: TRY003
            for key, value in sorted(metadata_filters.items()):
                if key not in {"due_date", "priority", "estimate"}:
                    raise InputError(f"Unsupported metadata filter: {key}.")  # noqa: TRY003
                if self._db.backend_type == BackendType.POSTGRESQL:
                    clauses.append("(t.metadata_json::jsonb ->> ?) = ?")
                    params.extend([key, str(value)])
                else:
                    clauses.append("json_extract(t.metadata_json, ?) = ?")
                    params.extend([f"$.{key}", str(value)])
        if not include_deleted:
            clauses.append("t.deleted = ?")
            params.append(self._deleted_value(False))
            clauses.append("n.deleted = ?")
            params.append(self._deleted_value(False))
        sql_query = "SELECT t.* FROM note_tasks t"
        if not include_deleted:
            if self._uses_postgres_v59_schema:
                sql_query += " JOIN notes n ON n.id = t.note_id AND n.client_id = t.client_id"
            else:
                sql_query += " JOIN notes n ON n.id = t.note_id AND n.client_id = t.owner_user_id"
        if clauses:
            sql_query += " WHERE " + " AND ".join(clauses)
        sql_query += " ORDER BY t.created_at ASC, t.id ASC LIMIT ? OFFSET ?"
        params.append(self._clamp_limit(limit))
        params.append(self._normalize_offset(offset))
        cursor = self._read(sql_query, tuple(params))
        return [self._decode_task_row(row) for row in cursor.fetchall()]

    def update_unlinked_task_metadata_record_only(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        task_id: str,
        expected_version: int,
        metadata: dict[str, Any],
        actor_type: str,
        actor_id: str | None = None,
        tool_name: str | None = None,
        policy_mode: str | None = None,
        approval_id: str | None = None,
        idempotency_key: str | None = None,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any]:
        """Update metadata for an unlinked task without modifying note content."""
        metadata_json = self._json_dumps(metadata, "metadata")
        owner, dataset = self._scope(owner_user_id, dataset_id)

        def _execute_metadata_update(transaction_conn: TaskConnection) -> dict[str, Any]:
            old = self._require_active_task(
                self._require_expected_version(
                    self._fetch_task(
                        task_id, owner_user_id=owner, dataset_id=dataset,
                        include_deleted=True, conn=transaction_conn,
                    ),
                    expected_version,
                    task_id,
                ),
                task_id,
            )
            if old["projection_status"] != "unlinked":
                raise ConflictError(
                    f"Task projection is {old['projection_status']} for task '{task_id}'.",
                    entity="tasks",
                    entity_id=task_id,
                )  # noqa: TRY003
            self._require_active_note(old["note_id"], task_id, owner_user_id=owner, conn=transaction_conn)
            now = self._db._get_current_utc_timestamp_iso()
            if self._uses_postgres_v59_schema:
                cursor = self._execute(
                    transaction_conn,
                    """
                    UPDATE note_tasks
                       SET metadata_json = ?, updated_at = ?, version = version + 1
                     WHERE client_id = ? AND id = ? AND version = ?
                       AND deleted = ? AND projection_status = ?
                    """,
                    (
                        metadata_json, now, owner, task_id, expected_version,
                        self._deleted_value(False), "unlinked",
                    ),
                )
            else:
                canonical_revision, canonical_hash, diagnostic_code, diagnostic_hash = self._canonical_task_values(
                    {
                        **old,
                        "metadata_json": metadata_json,
                        "updated_at": now,
                        "version": int(old["version"]) + 1,
                        "canonical_revision": int(old["canonical_revision"]) + 1,
                    }
                )
                cursor = self._execute(
                    transaction_conn,
                    """
                    UPDATE note_tasks
                       SET metadata_json = ?, updated_at = ?, version = version + 1,
                           canonical_revision = ?, canonical_hash = ?,
                           source_diagnostic_code = ?, source_diagnostic_hash = ?
                     WHERE owner_user_id = ? AND dataset_id = ? AND id = ?
                       AND version = ? AND deleted = ? AND projection_status = ?
                    """,
                    (
                        metadata_json, now, canonical_revision, canonical_hash,
                        diagnostic_code, diagnostic_hash, owner, dataset, task_id,
                        expected_version, self._deleted_value(False), "unlinked",
                    ),
                )
            if getattr(cursor, "rowcount", None) == 0:
                raise ConflictError(
                    f"Task version mismatch for ID '{task_id}'. Expected {expected_version}.",
                    entity="tasks",
                    entity_id=task_id,
                )  # noqa: TRY003
            updated = self._fetch_task(
                task_id, owner_user_id=owner, dataset_id=dataset,
                include_deleted=True, conn=transaction_conn,
            )
            if updated is None:
                raise ConflictError(f"Task with ID '{task_id}' not found.", entity="tasks", entity_id=task_id)  # noqa: TRY003
            self.record_task_event(
                task_id=task_id,
                note_id=str(updated["note_id"]),
                owner_user_id=owner,
                dataset_id=dataset,
                event_type="updated",
                actor_type=actor_type,
                actor_id=actor_id,
                tool_name=tool_name,
                policy_mode=policy_mode,
                approval_id=approval_id,
                old_value={"metadata": old.get("metadata_json") or {}},
                new_value=self._event_value(
                    {"metadata": updated.get("metadata_json") or {}},
                    idempotency_key=idempotency_key,
                ),
                conn=transaction_conn,
            )
            return updated

        return self._with_transaction(_execute_metadata_update, conn)

    def update_task_record(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        task_id: str,
        expected_version: int,
        text: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
        projection_status: str | None = None,
        actor_type: str | None = None,
        actor_id: str | None = None,
        tool_name: str | None = None,
        policy_mode: str | None = None,
        approval_id: str | None = None,
        idempotency_key: str | None = None,
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
        owner, dataset = self._scope(owner_user_id, dataset_id)

        def _execute_update(transaction_conn: TaskConnection) -> dict[str, Any]:
            old = self._require_active_task(
                self._require_expected_version(
                    self._fetch_task(
                        task_id, owner_user_id=owner, dataset_id=dataset,
                        include_deleted=True, conn=transaction_conn,
                    ),
                    expected_version,
                    task_id,
                ),
                task_id,
            )
            self._require_active_note(old["note_id"], task_id, owner_user_id=owner, conn=transaction_conn)
            projection_state, _ = self._resolve_projection_state(
                old, task_id, owner_user_id=owner, dataset_id=dataset, conn=transaction_conn
            )
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
            if self._uses_postgres_v59_schema:
                update_cursor = self._execute(
                    transaction_conn,
                    """
                    UPDATE note_tasks
                       SET text = ?, status = ?, metadata_json = ?, updated_at = ?,
                           completed_at = ?, version = version + 1
                     WHERE client_id = ? AND id = ? AND version = ?
                    """,
                    (
                        new_text,
                        new_status,
                        new_metadata_json,
                        now,
                        completed_at,
                        owner,
                        task_id,
                        expected_version,
                    ),
                )
            else:
                canonical_revision, canonical_hash, diagnostic_code, diagnostic_hash = self._canonical_task_values(
                    {
                        **old,
                        "text": new_text,
                        "status": new_status,
                        "metadata_json": new_metadata_json,
                        "updated_at": now,
                        "completed_at": completed_at,
                        "version": int(old["version"]) + 1,
                        "canonical_revision": int(old["canonical_revision"]) + 1,
                    }
                )
                update_cursor = self._execute(
                    transaction_conn,
                    """
                    UPDATE note_tasks
                       SET text = ?, status = ?, metadata_json = ?, updated_at = ?,
                           completed_at = ?, version = version + 1,
                           canonical_revision = ?, canonical_hash = ?,
                           source_diagnostic_code = ?, source_diagnostic_hash = ?
                     WHERE owner_user_id = ? AND dataset_id = ? AND id = ? AND version = ?
                    """,
                    (
                        new_text, new_status, new_metadata_json, now, completed_at,
                        canonical_revision, canonical_hash, diagnostic_code, diagnostic_hash,
                        owner, dataset, task_id, expected_version,
                    ),
                )
            if getattr(update_cursor, "rowcount", None) == 0:
                raise ConflictError(
                    f"Task version mismatch for ID '{task_id}'. Expected {expected_version}.",
                    entity="tasks",
                    entity_id=task_id,
                )  # noqa: TRY003
            updated = self._fetch_task(
                task_id, owner_user_id=owner, dataset_id=dataset,
                include_deleted=True, conn=transaction_conn,
            )
            if updated is None:
                raise ConflictError(
                    f"Task with ID '{task_id}' not found.", entity="tasks", entity_id=task_id
                )  # noqa: TRY003
            if actor_type and old["status"] != updated["status"]:
                self.record_task_event(
                    task_id=task_id,
                    note_id=updated["note_id"],
                    owner_user_id=owner,
                    dataset_id=dataset,
                    event_type="status_changed",
                    actor_type=actor_type,
                    actor_id=actor_id,
                    tool_name=tool_name,
                    policy_mode=policy_mode,
                    approval_id=approval_id,
                    old_value={"status": old["status"]},
                    new_value=self._event_value({"status": updated["status"]}, idempotency_key=idempotency_key),
                    conn=transaction_conn,
                )
            elif actor_type and (old["text"] != updated["text"] or old["metadata_json"] != updated["metadata_json"]):
                self.record_task_event(
                    task_id=task_id,
                    note_id=updated["note_id"],
                    owner_user_id=owner,
                    dataset_id=dataset,
                    event_type="updated",
                    actor_type=actor_type,
                    actor_id=actor_id,
                    tool_name=tool_name,
                    policy_mode=policy_mode,
                    approval_id=approval_id,
                    old_value={"text": old["text"], "metadata": old["metadata_json"]},
                    new_value=self._event_value(
                        {"text": updated["text"], "metadata": updated["metadata_json"]},
                        idempotency_key=idempotency_key,
                    ),
                    conn=transaction_conn,
                )
            return updated

        return self._with_transaction(_execute_update, conn)

    def set_task_projection(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
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
        owner, dataset = self._scope(owner_user_id, dataset_id)

        def _execute_projection(transaction_conn: TaskConnection) -> dict[str, Any]:
            task = self._require_active_task(
                self._fetch_task(
                    task_id, owner_user_id=owner, dataset_id=dataset,
                    include_deleted=True, conn=transaction_conn,
                ),
                task_id,
            )
            self._require_active_note(task["note_id"], task_id, owner_user_id=owner, conn=transaction_conn)
            projection_state, _ = self._resolve_projection_state(
                task, task_id, owner_user_id=owner, dataset_id=dataset, conn=transaction_conn
            )
            if task["note_id"] != note_id:
                raise ConflictError(
                    f"Task projection note does not match owning note for task '{task_id}'.",
                    entity="tasks",
                    entity_id=task_id,
                )  # noqa: TRY003
            if self._uses_postgres_v59_schema:
                update_cursor = self._execute(
                    transaction_conn,
                    """
                    UPDATE note_tasks
                       SET projection_status = ?, updated_at = ?,
                           version = CASE WHEN projection_status != ? THEN version + 1 ELSE version END
                     WHERE client_id = ? AND id = ? AND note_id = ?
                       AND deleted = ? AND projection_status = ?
                    """,
                    (
                        normalized_projection_status, now, normalized_projection_status,
                        owner, task_id, note_id, self._deleted_value(False), projection_state,
                    ),
                )
            else:
                update_cursor = self._execute(
                    transaction_conn,
                    """
                    UPDATE note_tasks
                       SET projection_status = ?, updated_at = ?,
                           version = CASE WHEN projection_status != ? THEN version + 1 ELSE version END
                     WHERE owner_user_id = ? AND dataset_id = ? AND id = ?
                       AND note_id = ? AND deleted = ? AND projection_status = ?
                    """,
                    (
                        normalized_projection_status, now, normalized_projection_status,
                        owner, dataset, task_id, note_id, self._deleted_value(False), projection_state,
                    ),
                )
            if getattr(update_cursor, "rowcount", None) == 0:
                raise ConflictError(
                    f"Task projection changed concurrently for task '{task_id}'.",
                    entity="tasks",
                    entity_id=task_id,
                )  # noqa: TRY003
            projection_values = (
                task_id, note_id, int(note_version), int(line_number), int(start_offset), int(end_offset),
                normalized_text_hash, int(occurrence_index), block_fingerprint, raw_line,
                self._deleted_value(bool(has_child_content)), normalized_projection_status, now,
            )
            if self._uses_postgres_v59_schema:
                self._execute(
                    transaction_conn,
                    """
                    INSERT INTO task_note_projections (
                        task_id, note_id, note_version, line_number, start_offset, end_offset,
                        normalized_text_hash, occurrence_index, block_fingerprint, raw_line,
                        has_child_content, projection_status, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(task_id) DO UPDATE SET
                        note_id = excluded.note_id, note_version = excluded.note_version,
                        line_number = excluded.line_number, start_offset = excluded.start_offset,
                        end_offset = excluded.end_offset, normalized_text_hash = excluded.normalized_text_hash,
                        occurrence_index = excluded.occurrence_index,
                        block_fingerprint = excluded.block_fingerprint, raw_line = excluded.raw_line,
                        has_child_content = excluded.has_child_content,
                        projection_status = excluded.projection_status, updated_at = excluded.updated_at
                    """,
                    projection_values,
                )
            else:
                self._execute(
                    transaction_conn,
                    """
                    INSERT INTO task_note_projections (
                        owner_user_id, dataset_id, task_id, note_id, note_version, line_number,
                        start_offset, end_offset, normalized_text_hash, occurrence_index,
                        block_fingerprint, raw_line, has_child_content, projection_status, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(owner_user_id, dataset_id, task_id) DO UPDATE SET
                        note_id = excluded.note_id, note_version = excluded.note_version,
                        line_number = excluded.line_number, start_offset = excluded.start_offset,
                        end_offset = excluded.end_offset, normalized_text_hash = excluded.normalized_text_hash,
                        occurrence_index = excluded.occurrence_index,
                        block_fingerprint = excluded.block_fingerprint, raw_line = excluded.raw_line,
                        has_child_content = excluded.has_child_content,
                        projection_status = excluded.projection_status, updated_at = excluded.updated_at
                    """,
                    (owner, dataset, *projection_values),
                )
            projection = self._fetch_projection(
                task_id, owner_user_id=owner, dataset_id=dataset, conn=transaction_conn
            )
            if projection is None:
                raise CharactersRAGDBError(f"Failed to read projection for task '{task_id}'.")  # noqa: TRY003
            return projection

        return self._with_transaction(_execute_projection, conn)

    def mark_task_unlinked(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        task_id: str,
        expected_version: int,
        actor_type: str | None = None,
        actor_id: str | None = None,
        tool_name: str | None = None,
        policy_mode: str | None = None,
        approval_id: str | None = None,
        idempotency_key: str | None = None,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any]:
        """Mark a task's projection as unlinked after reconciliation."""

        owner, dataset = self._scope(owner_user_id, dataset_id)

        def _execute_unlink(transaction_conn: TaskConnection) -> dict[str, Any]:
            old = self._require_active_task(
                self._require_expected_version(
                    self._fetch_task(
                        task_id, owner_user_id=owner, dataset_id=dataset,
                        include_deleted=True, conn=transaction_conn,
                    ),
                    expected_version,
                    task_id,
                ),
                task_id,
            )
            self._require_active_note(old["note_id"], task_id, owner_user_id=owner, conn=transaction_conn)
            projection_state, projection = self._resolve_projection_state(
                old, task_id, owner_user_id=owner, dataset_id=dataset, conn=transaction_conn
            )
            if projection_state != "live":
                raise ConflictError(
                    f"Task projection is {projection_state} for task '{task_id}'.",
                    entity="tasks",
                    entity_id=task_id,
                )  # noqa: TRY003
            self._require_live_projection_row(task_id, projection)
            now = self._db._get_current_utc_timestamp_iso()
            if self._uses_postgres_v59_schema:
                update_cursor = self._execute(
                    transaction_conn,
                    """
                    UPDATE note_tasks
                       SET projection_status = ?, updated_at = ?, version = version + 1
                     WHERE client_id = ? AND id = ? AND version = ? AND projection_status = ?
                    """,
                    ("unlinked", now, owner, task_id, expected_version, projection_state),
                )
            else:
                update_cursor = self._execute(
                    transaction_conn,
                    """
                    UPDATE note_tasks
                       SET projection_status = ?, updated_at = ?, version = version + 1
                     WHERE owner_user_id = ? AND dataset_id = ? AND id = ?
                       AND version = ? AND projection_status = ?
                    """,
                    ("unlinked", now, owner, dataset, task_id, expected_version, projection_state),
                )
            if getattr(update_cursor, "rowcount", None) == 0:
                raise ConflictError(
                    f"Task version mismatch for ID '{task_id}'. Expected {expected_version}.",
                    entity="tasks",
                    entity_id=task_id,
                )  # noqa: TRY003
            projection_locator = (
                projection_state, projection["note_id"], projection["note_version"],
                projection["line_number"], projection["normalized_text_hash"],
                projection["occurrence_index"], projection["block_fingerprint"],
            )
            if self._uses_postgres_v59_schema:
                projection_update_cursor = self._execute(
                    transaction_conn,
                    """
                    UPDATE task_note_projections
                       SET projection_status = ?, updated_at = ?
                     WHERE task_id = ? AND projection_status = ? AND note_id = ?
                       AND note_version = ? AND line_number = ? AND normalized_text_hash = ?
                       AND occurrence_index = ? AND block_fingerprint = ?
                       AND EXISTS (
                           SELECT 1 FROM note_tasks task
                            WHERE task.id = task_note_projections.task_id AND task.client_id = ?
                       )
                    """,
                    ("unlinked", now, task_id, *projection_locator, owner),
                )
            else:
                projection_update_cursor = self._execute(
                    transaction_conn,
                    """
                    UPDATE task_note_projections
                       SET projection_status = ?, updated_at = ?
                     WHERE owner_user_id = ? AND dataset_id = ? AND task_id = ?
                       AND projection_status = ? AND note_id = ? AND note_version = ?
                       AND line_number = ? AND normalized_text_hash = ?
                       AND occurrence_index = ? AND block_fingerprint = ?
                    """,
                    ("unlinked", now, owner, dataset, task_id, *projection_locator),
                )
            if getattr(projection_update_cursor, "rowcount", None) == 0:
                raise ConflictError(
                    f"Task projection changed concurrently for task '{task_id}'.",
                    entity="tasks",
                    entity_id=task_id,
                )  # noqa: TRY003
            updated = self._fetch_task(
                task_id, owner_user_id=owner, dataset_id=dataset,
                include_deleted=True, conn=transaction_conn,
            )
            if actor_type:
                self.record_task_event(
                    task_id=task_id,
                    note_id=old["note_id"],
                    owner_user_id=owner,
                    dataset_id=dataset,
                    event_type="unlinked",
                    actor_type=actor_type,
                    actor_id=actor_id,
                    tool_name=tool_name,
                    policy_mode=policy_mode,
                    approval_id=approval_id,
                    old_value={"projection_status": old["projection_status"]},
                    new_value=self._event_value(
                        {"projection_status": "unlinked"},
                        idempotency_key=idempotency_key,
                    ),
                    conn=transaction_conn,
                )
            if updated is None:
                raise CharactersRAGDBError(f"Failed to read unlinked task '{task_id}'.")  # noqa: TRY003
            return updated

        return self._with_transaction(_execute_unlink, conn)

    def soft_delete_task(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        task_id: str,
        expected_version: int,
        projection_note_id: str | None = None,
        projection_note_version: int | None = None,
        projection_line_number: int | None = None,
        allow_record_only: bool = False,
        actor_type: str | None = None,
        actor_id: str | None = None,
        tool_name: str | None = None,
        policy_mode: str | None = None,
        approval_id: str | None = None,
        idempotency_key: str | None = None,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any]:
        """Soft-delete a task, requiring projection context for live projected rows."""

        owner, dataset = self._scope(owner_user_id, dataset_id)

        def _execute_delete(transaction_conn: TaskConnection) -> dict[str, Any]:
            old = self._require_active_task(
                self._require_expected_version(
                    self._fetch_task(
                        task_id, owner_user_id=owner, dataset_id=dataset,
                        include_deleted=True, conn=transaction_conn,
                    ),
                    expected_version,
                    task_id,
                ),
                task_id,
            )
            self._require_active_note(old["note_id"], task_id, owner_user_id=owner, conn=transaction_conn)
            projection_status, projection = self._resolve_projection_state(
                old, task_id, owner_user_id=owner, dataset_id=dataset, conn=transaction_conn
            )
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
            if self._uses_postgres_v59_schema:
                update_cursor = self._execute(
                    transaction_conn,
                    """
                    UPDATE note_tasks
                       SET deleted = ?, projection_status = ?, updated_at = ?, version = version + 1
                     WHERE client_id = ? AND id = ? AND version = ?
                       AND deleted = ? AND projection_status = ?
                    """,
                    (
                        self._deleted_value(True), "deleted", now, owner, task_id,
                        expected_version, self._deleted_value(False), projection_status,
                    ),
                )
            else:
                canonical_revision, canonical_hash, diagnostic_code, diagnostic_hash = self._canonical_task_values(
                    {
                        **old,
                        "deleted": 1,
                        "projection_status": "deleted",
                        "updated_at": now,
                        "version": int(old["version"]) + 1,
                        "canonical_revision": int(old["canonical_revision"]) + 1,
                    }
                )
                update_cursor = self._execute(
                    transaction_conn,
                    """
                    UPDATE note_tasks
                       SET deleted = ?, projection_status = ?, updated_at = ?, version = version + 1,
                           canonical_revision = ?, canonical_hash = ?,
                           source_diagnostic_code = ?, source_diagnostic_hash = ?
                     WHERE owner_user_id = ? AND dataset_id = ? AND id = ?
                       AND version = ? AND deleted = ? AND projection_status = ?
                    """,
                    (
                        self._deleted_value(True), "deleted", now, canonical_revision,
                        canonical_hash, diagnostic_code, diagnostic_hash, owner, dataset,
                        task_id, expected_version, self._deleted_value(False), projection_status,
                    ),
                )
            if getattr(update_cursor, "rowcount", None) == 0:
                raise ConflictError(
                    f"Task version mismatch for ID '{task_id}'. Expected {expected_version}.",
                    entity="tasks",
                    entity_id=task_id,
                )  # noqa: TRY003
            locator = ()
            if projection_status == "live" and not allow_record_only:
                locator = (
                    projection_status, projection["note_id"], projection["note_version"],
                    projection["line_number"], projection["normalized_text_hash"],
                    projection["occurrence_index"], projection["block_fingerprint"],
                )
            if self._uses_postgres_v59_schema:
                locator_sql = ""
                if locator:
                    locator_sql = (
                        " AND projection_status = ? AND note_id = ? AND note_version = ?"
                        " AND line_number = ? AND normalized_text_hash = ?"
                        " AND occurrence_index = ? AND block_fingerprint = ?"
                    )
                projection_update_cursor = self._execute(
                    transaction_conn,
                    "UPDATE task_note_projections SET projection_status = ?, updated_at = ? "  # nosec B608
                    "WHERE task_id = ?" + locator_sql +
                    " AND EXISTS (SELECT 1 FROM note_tasks task "  # nosec B608
                    "WHERE task.id = task_note_projections.task_id AND task.client_id = ?)",
                    ("deleted", now, task_id, *locator, owner),
                )
            else:
                locator_sql = ""
                if locator:
                    locator_sql = (
                        " AND projection_status = ? AND note_id = ? AND note_version = ?"
                        " AND line_number = ? AND normalized_text_hash = ?"
                        " AND occurrence_index = ? AND block_fingerprint = ?"
                    )
                projection_update_cursor = self._execute(
                    transaction_conn,
                    "UPDATE task_note_projections SET projection_status = ?, updated_at = ? "  # nosec B608
                    "WHERE owner_user_id = ? AND dataset_id = ? AND task_id = ?" + locator_sql,
                    ("deleted", now, owner, dataset, task_id, *locator),
                )
            if projection_status == "live" and not allow_record_only:
                if getattr(projection_update_cursor, "rowcount", None) == 0:
                    raise ConflictError(
                        f"Task projection changed concurrently for task '{task_id}'.",
                        entity="tasks",
                        entity_id=task_id,
                    )  # noqa: TRY003
            updated = self._fetch_task(
                task_id, owner_user_id=owner, dataset_id=dataset,
                include_deleted=True, conn=transaction_conn,
            )
            if actor_type:
                self.record_task_event(
                    task_id=task_id,
                    note_id=old["note_id"],
                    owner_user_id=owner,
                    dataset_id=dataset,
                    event_type="deleted",
                    actor_type=actor_type,
                    actor_id=actor_id,
                    tool_name=tool_name,
                    policy_mode=policy_mode,
                    approval_id=approval_id,
                    old_value={"deleted": old["deleted"], "projection_status": old["projection_status"]},
                    new_value=self._event_value(
                        {"deleted": True, "projection_status": "deleted"},
                        idempotency_key=idempotency_key,
                    ),
                    conn=transaction_conn,
                )
            if updated is None:
                raise CharactersRAGDBError(f"Failed to read deleted task '{task_id}'.")  # noqa: TRY003
            return updated

        return self._with_transaction(_execute_delete, conn)

    def record_task_event(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        task_id: str | None = None,
        note_id: str | None = None,
        event_type: str,
        actor_type: str,
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
        if not isinstance(note_id, str) or not note_id.strip():
            raise InputError("Task event note_id is required.")  # noqa: TRY003
        note_id = note_id.strip()
        final_event_id = event_id or self._db._generate_uuid()
        now = self._db._get_current_utc_timestamp_iso()
        old_value_json = self._json_dumps(old_value, "old_value") if old_value is not None else None
        new_value_json = self._json_dumps(new_value, "new_value") if new_value is not None else None
        owner, dataset = self._scope(owner_user_id, dataset_id)
        event_hash = None if self._uses_postgres_v59_schema else self._db._note_task_v60_hash(
            {
                "owner_user_id": owner,
                "dataset_id": dataset,
                "id": final_event_id,
                "task_id": task_id,
                "note_id": note_id,
                "event_type": event_type,
                "actor_type": actor_type,
                "actor_id": actor_id,
                "tool_name": tool_name,
                "policy_mode": policy_mode,
                "approval_id": approval_id,
                "old_value_json": old_value_json,
                "new_value_json": new_value_json,
                "created_at": now,
                "client_id": self._db.client_id,
            }
        )

        def _execute_event(transaction_conn: TaskConnection) -> dict[str, Any]:
            self._require_active_note(
                note_id, final_event_id, owner_user_id=owner, conn=transaction_conn
            )
            if task_id is not None:
                task = self._fetch_task(
                    task_id,
                    owner_user_id=owner,
                    dataset_id=dataset,
                    include_deleted=True,
                    conn=transaction_conn,
                )
                if task is None:
                    raise ConflictError("Task event task not found.", entity="tasks", entity_id=task_id)  # noqa: TRY003
                if str(task["note_id"]) != note_id:
                    raise ConflictError(
                        "Task event note does not match the task's owning note.",
                        entity="tasks",
                        entity_id=task_id,
                    )  # noqa: TRY003
            if self._uses_postgres_v59_schema:
                self._execute(
                    transaction_conn,
                    """
                    INSERT INTO task_events (
                        id, task_id, note_id, event_type, actor_type, actor_id, tool_name,
                        policy_mode, approval_id, old_value_json, new_value_json, created_at, client_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        final_event_id, task_id, note_id, event_type, actor_type, actor_id,
                        tool_name, policy_mode, approval_id, old_value_json, new_value_json,
                        now, owner,
                    ),
                )
                cursor = self._read(
                    "SELECT * FROM task_events WHERE client_id = ? AND id = ?",
                    (owner, final_event_id),
                    conn=transaction_conn,
                )
                event = self._decode_event_row(cursor.fetchone())
                if event is None:
                    raise CharactersRAGDBError(f"Failed to read task event '{final_event_id}'.")  # noqa: TRY003
                return event
            self._execute(
                transaction_conn,
                """
                INSERT INTO task_events (
                    owner_user_id, dataset_id, id, task_id, note_id, event_type, actor_type,
                    actor_id, tool_name, policy_mode, approval_id, old_value_json, new_value_json,
                    created_at, client_id, sync_revision, sync_object_hash, sync_server_cursor,
                    source_device_id, client_occurred_at, source_kind, corrects_activity_id,
                    deleted, deleted_at, delete_reason, source_diagnostic_code,
                    source_diagnostic_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner,
                    dataset,
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
                    1,
                    event_hash,
                    None,
                    None,
                    now,
                    "rest",
                    None,
                    self._deleted_value(False),
                    None,
                    None,
                    "legacy_task_activity_unverified",
                    event_hash,
                ),
            )
            cursor = self._read(
                "SELECT * FROM task_events WHERE owner_user_id = ? AND dataset_id = ? AND id = ?",
                (owner, dataset, final_event_id),
                conn=transaction_conn,
            )
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

    def list_task_activity(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        task_id: str | None = None,
        note_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List task events by task or note scope."""
        owner, dataset = self._scope(owner_user_id, dataset_id)
        if self._uses_postgres_v59_schema:
            clauses: list[str] = ["client_id = ?"]
            params: list[Any] = [owner]
        else:
            clauses = ["owner_user_id = ?", "dataset_id = ?"]
            params = [owner, dataset]
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

    def list_recent_task_activity(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        task_id: str | None = None,
        note_id: str | None = None,
        actor_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List task events newest-first without changing the ascending default helper."""
        owner, dataset = self._scope(owner_user_id, dataset_id)
        if self._uses_postgres_v59_schema:
            clauses: list[str] = ["client_id = ?"]
            params: list[Any] = [owner]
        else:
            clauses = ["owner_user_id = ?", "dataset_id = ?"]
            params = [owner, dataset]
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if note_id is not None:
            clauses.append("note_id = ?")
            params.append(note_id)
        if actor_type is not None:
            clauses.append("actor_type = ?")
            params.append(actor_type)
        query = "SELECT * FROM task_events"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        if self._db.backend_type == BackendType.SQLITE:
            query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        else:
            query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(self._clamp_limit(limit))
        cursor = self._read(query, tuple(params))
        return [self._decode_event_row(row) for row in cursor.fetchall()]

    def list_recent_unread_task_activity(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        user_id: str,
        task_id: str | None = None,
        note_id: str | None = None,
        actor_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List newest unread task events for a user, applying visibility before limit."""
        owner, dataset = self._scope(owner_user_id, dataset_id)
        if user_id != owner:
            return []
        if self._uses_postgres_v59_schema:
            query = """
                SELECT events.*
                FROM task_events AS events
                LEFT JOIN task_event_read_state AS state
                  ON state.event_id = events.id AND state.user_id = ?
                WHERE events.client_id = ?
                  AND (state.event_id IS NULL OR (state.read_at IS NULL AND state.dismissed_at IS NULL))
                  AND (? IS NULL OR events.task_id = ?)
                  AND (? IS NULL OR events.note_id = ?)
                  AND (? IS NULL OR events.actor_type = ?)
            """
        else:
            query = """
                SELECT events.*
                FROM task_events AS events
                LEFT JOIN task_event_read_state AS state
                  ON state.owner_user_id = events.owner_user_id
                 AND state.dataset_id = events.dataset_id
                 AND state.event_id = events.id AND state.user_id = ?
                WHERE events.owner_user_id = ? AND events.dataset_id = ?
                  AND (state.event_id IS NULL OR (state.read_at IS NULL AND state.dismissed_at IS NULL))
                  AND (? IS NULL OR events.task_id = ?)
                  AND (? IS NULL OR events.note_id = ?)
                  AND (? IS NULL OR events.actor_type = ?)
            """
        if self._db.backend_type == BackendType.SQLITE:
            query += " ORDER BY events.created_at DESC, events.rowid DESC LIMIT ?"
        else:
            query += " ORDER BY events.created_at DESC, events.id DESC LIMIT ?"
        if self._uses_postgres_v59_schema:
            params: list[Any] = [user_id, owner, task_id, task_id, note_id, note_id, actor_type, actor_type]
        else:
            params = [user_id, owner, dataset, task_id, task_id, note_id, note_id, actor_type, actor_type]
        params.append(self._clamp_limit(limit))
        cursor = self._read(query, tuple(params))
        return [self._decode_event_row(row) for row in cursor.fetchall()]

    def mark_task_activity_read(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        event_id: str,
        user_id: str,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any]:
        """Mark an activity event read for one user."""
        now = self._db._get_current_utc_timestamp_iso()
        owner, dataset = self._scope(owner_user_id, dataset_id)
        if user_id != owner:
            raise ConflictError("Task event not found.", entity="tasks", entity_id=event_id)  # noqa: TRY003

        def _execute_read(transaction_conn: TaskConnection) -> dict[str, Any]:
            if self._uses_postgres_v59_schema:
                event = self._read(
                    "SELECT id FROM task_events WHERE client_id = ? AND id = ?",
                    (owner, event_id),
                    conn=transaction_conn,
                ).fetchone()
                if event is None:
                    raise ConflictError("Task event not found.", entity="tasks", entity_id=event_id)  # noqa: TRY003
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
                state = self.get_task_activity_read_state(
                    owner_user_id=owner,
                    dataset_id=dataset,
                    event_id=event_id,
                    user_id=user_id,
                    conn=transaction_conn,
                )
                if state is None:
                    raise CharactersRAGDBError(f"Failed to read task event read state for '{event_id}'.")  # noqa: TRY003
                return state
            self._execute(
                transaction_conn,
                """
                INSERT INTO task_event_read_state (
                    owner_user_id, dataset_id, event_id, user_id, read_at, dismissed_at
                ) VALUES (?, ?, ?, ?, ?, NULL)
                ON CONFLICT(owner_user_id, dataset_id, event_id, user_id) DO UPDATE SET
                    read_at = COALESCE(task_event_read_state.read_at, excluded.read_at)
                """,
                (owner, dataset, event_id, user_id, now),
            )
            state = self.get_task_activity_read_state(
                owner_user_id=owner, dataset_id=dataset, event_id=event_id, user_id=user_id,
                conn=transaction_conn,
            )
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

    def mark_task_activity_dismissed(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        event_id: str,
        user_id: str,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any]:
        """Mark an activity event dismissed for one user."""
        now = self._db._get_current_utc_timestamp_iso()
        owner, dataset = self._scope(owner_user_id, dataset_id)
        if user_id != owner:
            raise ConflictError("Task event not found.", entity="tasks", entity_id=event_id)  # noqa: TRY003

        def _execute_dismiss(transaction_conn: TaskConnection) -> dict[str, Any]:
            if self._uses_postgres_v59_schema:
                event = self._read(
                    "SELECT id FROM task_events WHERE client_id = ? AND id = ?",
                    (owner, event_id),
                    conn=transaction_conn,
                ).fetchone()
                if event is None:
                    raise ConflictError("Task event not found.", entity="tasks", entity_id=event_id)  # noqa: TRY003
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
                state = self.get_task_activity_read_state(
                    owner_user_id=owner,
                    dataset_id=dataset,
                    event_id=event_id,
                    user_id=user_id,
                    conn=transaction_conn,
                )
                if state is None:
                    raise CharactersRAGDBError(f"Failed to read task event read state for '{event_id}'.")  # noqa: TRY003
                return state
            self._execute(
                transaction_conn,
                """
                INSERT INTO task_event_read_state (
                    owner_user_id, dataset_id, event_id, user_id, read_at, dismissed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner_user_id, dataset_id, event_id, user_id) DO UPDATE SET
                    read_at = COALESCE(task_event_read_state.read_at, excluded.read_at),
                    dismissed_at = excluded.dismissed_at
                """,
                (owner, dataset, event_id, user_id, now, now),
            )
            state = self.get_task_activity_read_state(
                owner_user_id=owner, dataset_id=dataset, event_id=event_id, user_id=user_id,
                conn=transaction_conn,
            )
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

    def get_task_activity_read_state(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        event_id: str,
        user_id: str,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any] | None:
        """Return read/dismiss state for one event and user."""
        owner, dataset = self._scope(owner_user_id, dataset_id)
        if user_id != owner:
            return None
        if self._uses_postgres_v59_schema:
            cursor = self._read(
                """
                SELECT state.*
                  FROM task_event_read_state state
                  JOIN task_events event ON event.id = state.event_id
                 WHERE event.client_id = ? AND state.event_id = ? AND state.user_id = ?
                """,
                (owner, event_id, user_id),
                conn=conn,
            )
        else:
            cursor = self._read(
                """
                SELECT * FROM task_event_read_state
                 WHERE owner_user_id = ? AND dataset_id = ? AND event_id = ? AND user_id = ?
                """,
                (owner, dataset, event_id, user_id),
                conn=conn,
            )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_reconciliation_state(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        note_id: str,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any] | None:
        """Return the last task reconciliation state for a note."""
        owner, dataset = self._scope(owner_user_id, dataset_id)
        if self._uses_postgres_v59_schema:
            cursor = self._read(
                """
                SELECT state.*
                  FROM note_task_reconciliation_state state
                  JOIN notes note ON note.id = state.note_id
                 WHERE note.client_id = ? AND state.note_id = ?
                """,
                (owner, note_id),
                conn=conn,
            )
        else:
            cursor = self._read(
                "SELECT * FROM note_task_reconciliation_state "
                "WHERE owner_user_id = ? AND dataset_id = ? AND note_id = ?",
                (owner, dataset, note_id),
                conn=conn,
            )
        row = cursor.fetchone()
        return dict(row) if row else None

    def set_reconciliation_state(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
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
        owner, dataset = self._scope(owner_user_id, dataset_id)

        def _execute_state(transaction_conn: TaskConnection) -> dict[str, Any]:
            if self._uses_postgres_v59_schema:
                note = self._read(
                    "SELECT id FROM notes WHERE client_id = ? AND id = ?",
                    (owner, note_id),
                    conn=transaction_conn,
                ).fetchone()
                if note is None:
                    raise ConflictError("Task note not found.", entity="tasks", entity_id=note_id)  # noqa: TRY003
                self._execute(
                    transaction_conn,
                    """
                    INSERT INTO note_task_reconciliation_state (
                        note_id, note_version, status, reconciled_at,
                        item_count, warning_count, cursor
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
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
            else:
                self._execute(
                    transaction_conn,
                    """
                    INSERT INTO note_task_reconciliation_state (
                        owner_user_id, dataset_id, note_id, note_version, status,
                        reconciled_at, item_count, warning_count, cursor
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(owner_user_id, dataset_id, note_id) DO UPDATE SET
                        note_version = excluded.note_version,
                        status = excluded.status,
                        reconciled_at = excluded.reconciled_at,
                        item_count = excluded.item_count,
                        warning_count = excluded.warning_count,
                        cursor = excluded.cursor
                    """,
                    (
                        owner,
                        dataset,
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
                self.get_reconciliation_state(
                    owner_user_id=owner, dataset_id=dataset, note_id=note_id
                )
                if conn is None
                else self._get_reconciliation_state_in_conn(
                    note_id, owner_user_id=owner, dataset_id=dataset, conn=transaction_conn
                )
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

    def _get_reconciliation_state_in_conn(
        self,
        note_id: str,
        *,
        owner_user_id: str,
        dataset_id: str,
        conn: TaskConnection,
    ) -> dict[str, Any] | None:
        if self._uses_postgres_v59_schema:
            cursor = self._read(
                """
                SELECT state.*
                  FROM note_task_reconciliation_state state
                  JOIN notes note ON note.id = state.note_id
                 WHERE note.client_id = ? AND state.note_id = ?
                """,
                (owner_user_id, note_id),
                conn=conn,
            )
        else:
            cursor = self._read(
                "SELECT * FROM note_task_reconciliation_state "
                "WHERE owner_user_id = ? AND dataset_id = ? AND note_id = ?",
                (owner_user_id, dataset_id, note_id),
                conn=conn,
            )
        row = cursor.fetchone()
        return dict(row) if row else None

    def candidate_notes_for_task_discovery(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return checklist-bearing notes whose task reconciliation is stale or missing."""
        owner, dataset = self._scope(owner_user_id, dataset_id)
        if self._uses_postgres_v59_schema:
            cursor = self._read(
                """
                SELECT n.id, n.title, n.version, n.last_modified
                  FROM notes n
                  LEFT JOIN note_task_reconciliation_state r ON r.note_id = n.id
                 WHERE n.client_id = ? AND n.deleted = ?
                   AND (n.content LIKE ? OR n.content LIKE ? OR n.content LIKE ?)
                   AND (n.content LIKE ? OR n.content LIKE ? OR n.content LIKE ?)
                   AND (r.note_id IS NULL OR r.note_version < n.version)
                 ORDER BY n.last_modified DESC, n.id ASC
                 LIMIT ?
                """,
                (
                    owner,
                    self._deleted_value(False),
                    *self._CHECKLIST_DISCOVERY_MARKER_PATTERNS,
                    *self._CHECKLIST_DISCOVERY_BULLET_PATTERNS,
                    self._clamp_limit(limit),
                ),
            )
            return [dict(row) for row in cursor.fetchall()]
        cursor = self._read(
            """
            SELECT n.id, n.title, n.version, n.last_modified
              FROM notes n
              LEFT JOIN note_task_reconciliation_state r
                ON r.owner_user_id = n.client_id AND r.dataset_id = ? AND r.note_id = n.id
             WHERE n.client_id = ? AND n.deleted = ?
               AND (
                    n.content LIKE ?
                 OR n.content LIKE ?
                 OR n.content LIKE ?
               )
               AND (
                    n.content LIKE ?
                 OR n.content LIKE ?
                 OR n.content LIKE ?
               )
               AND (
                    r.note_id IS NULL
                 OR r.note_version < n.version
               )
             ORDER BY n.last_modified DESC, n.id ASC
             LIMIT ?
            """,
            (
                dataset,
                owner,
                self._deleted_value(False),
                *self._CHECKLIST_DISCOVERY_MARKER_PATTERNS,
                *self._CHECKLIST_DISCOVERY_BULLET_PATTERNS,
                self._clamp_limit(limit),
            ),
        )
        return [dict(row) for row in cursor.fetchall()]

    def count_candidate_notes_for_task_discovery(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        note_id: str | None = None,
    ) -> int:
        """Count checklist-bearing notes whose task reconciliation is stale or missing."""
        owner, dataset = self._scope(owner_user_id, dataset_id)
        if self._uses_postgres_v59_schema:
            params: list[Any] = [
                owner,
                self._deleted_value(False),
                *self._CHECKLIST_DISCOVERY_MARKER_PATTERNS,
                *self._CHECKLIST_DISCOVERY_BULLET_PATTERNS,
            ]
            sql_query = """
                SELECT COUNT(*) AS stale_count
                  FROM notes n
                  LEFT JOIN note_task_reconciliation_state r ON r.note_id = n.id
                 WHERE n.client_id = ? AND n.deleted = ?
                   AND (n.content LIKE ? OR n.content LIKE ? OR n.content LIKE ?)
                   AND (n.content LIKE ? OR n.content LIKE ? OR n.content LIKE ?)
                   AND (r.note_id IS NULL OR r.note_version < n.version)
            """
            if note_id is not None:
                sql_query += " AND n.id = ?"
                params.append(note_id)
            cursor = self._read(sql_query, tuple(params))
            row = cursor.fetchone()
            return int(row["stale_count"] if row else 0)
        params: list[Any] = [
            dataset,
            owner,
            self._deleted_value(False),
            *self._CHECKLIST_DISCOVERY_MARKER_PATTERNS,
            *self._CHECKLIST_DISCOVERY_BULLET_PATTERNS,
        ]
        sql_query = """
            SELECT COUNT(*) AS stale_count
              FROM notes n
              LEFT JOIN note_task_reconciliation_state r
                ON r.owner_user_id = n.client_id AND r.dataset_id = ? AND r.note_id = n.id
             WHERE n.client_id = ? AND n.deleted = ?
               AND (
                    n.content LIKE ?
                 OR n.content LIKE ?
                 OR n.content LIKE ?
               )
               AND (
                    n.content LIKE ?
                 OR n.content LIKE ?
                 OR n.content LIKE ?
               )
               AND (
                    r.note_id IS NULL
                 OR r.note_version < n.version
               )
        """
        if note_id is not None:
            sql_query += " AND n.id = ?"
            params.append(note_id)
        cursor = self._read(sql_query, tuple(params))
        row = cursor.fetchone()
        return int(row["stale_count"] if row else 0)

    def bind_local_task_graph_to_dataset(
        self,
        *,
        owner_user_id: str,
        target_dataset_id: str,
        conn: TaskConnection | None = None,
    ) -> dict[str, int]:
        """Atomically rekey one owner's complete local task graph to an enrolled dataset."""
        owner = str(owner_user_id).strip()
        target = str(target_dataset_id).strip()
        if not owner or not target or target == self._LOCAL_UNBOUND:
            raise InputError("A non-sentinel owner and target dataset are required.")  # noqa: TRY003
        if self._db.backend_type != BackendType.SQLITE:
            raise InputError("Local task graph binding is only supported by SQLite storage.")  # noqa: TRY003

        tables = (
            "note_tasks",
            "task_note_projections",
            "task_events",
            "task_event_read_state",
            "note_task_reconciliation_state",
            "task_projection_drifts",
        )

        def _counts(transaction_conn: TaskConnection, dataset: str) -> dict[str, int]:
            result: dict[str, int] = {}
            for table in tables:
                row = self._read(
                    f"SELECT COUNT(*) AS count FROM {table} "  # nosec B608
                    "WHERE owner_user_id = ? AND dataset_id = ?",
                    (owner, dataset),
                    conn=transaction_conn,
                ).fetchone()
                result[table] = int(row["count"])
            return result

        def _execute_bind(transaction_conn: TaskConnection) -> dict[str, int]:
            source_counts = _counts(transaction_conn, self._LOCAL_UNBOUND)
            target_counts = _counts(transaction_conn, target)
            if any(target_counts.values()):
                raise ConflictError(
                    "Task dataset binding target collision.", entity="tasks", entity_id=target
                )  # noqa: TRY003
            if not any(source_counts.values()):
                return source_counts

            # Rekey roots only. Composite ON UPDATE CASCADE relationships carry
            # projections, task-linked events, read-state, and drift rows.
            self._execute(
                transaction_conn,
                "UPDATE note_tasks SET dataset_id = ? WHERE owner_user_id = ? AND dataset_id = ?",
                (target, owner, self._LOCAL_UNBOUND),
            )
            self._execute(
                transaction_conn,
                "UPDATE task_events SET dataset_id = ? WHERE owner_user_id = ? AND dataset_id = ?",
                (target, owner, self._LOCAL_UNBOUND),
            )
            self._execute(
                transaction_conn,
                "UPDATE note_task_reconciliation_state SET dataset_id = ? "
                "WHERE owner_user_id = ? AND dataset_id = ?",
                (target, owner, self._LOCAL_UNBOUND),
            )

            remaining = _counts(transaction_conn, self._LOCAL_UNBOUND)
            rebound = _counts(transaction_conn, target)
            if any(remaining.values()) or rebound != source_counts:
                raise ConflictError(
                    "Task dataset binding failed complete-set verification.",
                    entity="tasks",
                    entity_id=target,
                )  # noqa: TRY003
            return rebound

        return self._with_transaction(_execute_bind, conn)

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
