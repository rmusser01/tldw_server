from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, TypeAlias, TypeVar

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
from tldw_Server_API.app.core.Sync.v2.models import normalize_sync_timestamp
from tldw_Server_API.app.core.Sync.v2.notes_task_contract import (
    NotesTaskActivityTombstoneV1,
    NotesTaskActivityV1,
    NotesTaskV1Payload,
    convert_legacy_task_event,
    notes_task_activity_object_hash,
    notes_task_object_hash,
    parse_notes_task_v1,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


TaskConnection: TypeAlias = sqlite3.Connection | BackendConnectionWrapper
ScopedReadT = TypeVar("ScopedReadT")


class TaskStore:
    """Persistence helper for task-backed note checklist records."""

    _TASK_JSON_FIELDS = ("metadata_json",)
    _EVENT_JSON_FIELDS = ("old_value_json", "new_value_json")
    _CHECKLIST_DISCOVERY_MARKER_PATTERNS = ("%[ ]%", "%[x]%", "%[X]%")
    _CHECKLIST_DISCOVERY_BULLET_PATTERNS = ("%-%", "%*%", "%+%")
    _TASK_STATUSES = {"open", "done"}
    _PROJECTION_STATUSES = {"live", "unlinked", "deleted", "ambiguous"}
    _PROJECTION_DRIFT_REASONS = {
        "missing_marker_base",
        "malformed_marker",
        "duplicate_marker",
        "marker_scope_mismatch",
        "base_unavailable",
        "both_changed",
        "ambiguous_legacy_match",
        "unsupported_markdown",
    }
    _PROJECTION_DRIFT_STATUSES = {"open", "resolved", "dismissed"}
    _MIN_LIMIT = 1
    _MAX_LIMIT = 500
    _LOCAL_UNBOUND = "local-unbound"
    _BIND_TABLE_ORDER = (
        ("note_tasks", "id"),
        ("task_note_projections", "task_id"),
        ("task_events", "id"),
        ("task_event_read_state", "event_id,user_id"),
        ("note_task_reconciliation_state", "note_id"),
        ("task_projection_drifts", "id"),
    )

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
        if self._db.backend_type == BackendType.POSTGRESQL:
            if owner != str(self._db.client_id):
                raise ConflictError(
                    "Task owner does not match the authenticated PostgreSQL client.",
                    entity="tasks",
                    entity_id=owner,
                )  # noqa: TRY003
        return owner, dataset

    def _set_postgres_dataset_scope(
        self,
        conn: TaskConnection,
        dataset_id: str,
    ) -> None:
        """Set transaction-local PostgreSQL task RLS scope on one connection."""
        if self._db.backend_type == BackendType.POSTGRESQL:
            self._execute(
                conn,
                "SELECT set_config('app.current_dataset_id', ?, true)",
                (dataset_id,),
            )

    def _with_scoped_read(
        self,
        *,
        dataset_id: str,
        conn: TaskConnection | None,
        fn: Callable[[TaskConnection | None], ScopedReadT],
    ) -> ScopedReadT:
        """Run a read with transaction-local PostgreSQL dataset scope."""
        if self._db.backend_type != BackendType.POSTGRESQL:
            return fn(conn)
        if conn is not None:
            self._set_postgres_dataset_scope(conn, dataset_id)
            return fn(conn)
        with self._db.transaction() as transaction_conn:
            self._set_postgres_dataset_scope(transaction_conn, dataset_id)
            return fn(transaction_conn)

    def _require_authorized_write_scope(
        self,
        conn: TaskConnection,
        *,
        owner_user_id: str,
        dataset_id: str,
    ) -> None:
        """Serialize with binding and reject writes outside the private authority scope."""
        if self._db.backend_type == BackendType.POSTGRESQL:
            self._execute(
                conn,
                "LOCK TABLE note_task_scope_authority IN SHARE MODE",
            )
        rows = self._read(
            "SELECT owner_user_id,dataset_id FROM note_task_scope_authority "
            "WHERE owner_user_id = ? LIMIT 2",
            (owner_user_id,),
            conn=conn,
        ).fetchall()
        if not rows:
            if dataset_id == self._LOCAL_UNBOUND:
                self._set_postgres_dataset_scope(conn, dataset_id)
                return
            raise ConflictError(
                "Task write scope is not bound.",
                entity="tasks",
                entity_id=owner_user_id,
            )  # noqa: TRY003
        row_owner = str(rows[0]["owner_user_id"]).strip()
        authority_dataset = str(rows[0]["dataset_id"]).strip()
        if (
            len(rows) != 1
            or row_owner != owner_user_id
            or not authority_dataset
            or authority_dataset == self._LOCAL_UNBOUND
            or dataset_id != authority_dataset
        ):
            raise ConflictError(
                "Task write scope conflicts with bound authority.",
                entity="tasks",
                entity_id=owner_user_id,
            )  # noqa: TRY003
        self._set_postgres_dataset_scope(conn, dataset_id)

    def lock_authorized_write_scope(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        conn: TaskConnection,
    ) -> None:
        """Fence a wider note/task transaction before it reads or mutates graph rows."""
        owner = str(owner_user_id).strip()
        dataset = str(dataset_id).strip()
        if not owner or not dataset:
            raise InputError("Task owner and dataset scope cannot be empty.")  # noqa: TRY003
        if self._db.backend_type == BackendType.POSTGRESQL and owner != str(self._db.client_id):
            raise ConflictError(
                "Task owner does not match the authenticated PostgreSQL client.",
                entity="tasks",
                entity_id=owner,
            )  # noqa: TRY003
        self._require_authorized_write_scope(
            conn,
            owner_user_id=owner,
            dataset_id=dataset,
        )

    def _canonical_task_values(self, task: Mapping[str, Any]) -> tuple[int, str, str | None, str | None]:
        revision = int(task.get("canonical_revision") or task.get("version") or 1)
        source = dict(task)
        if isinstance(source.get("metadata_json"), dict):
            source["metadata_json"] = self._json_dumps(source["metadata_json"], "metadata")
        source["completed_at"] = normalize_sync_timestamp(source.get("completed_at"))
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
    def _require_nonempty_identity(value: str, field_name: str) -> str:
        """Return one non-empty opaque identity without rewriting it."""
        if not isinstance(value, str) or not value or value != value.strip():
            raise InputError(f"{field_name} must be a non-empty canonical string.")
        return value

    @staticmethod
    def _validate_projection_hash(value: str, field_name: str) -> str:
        """Return a lowercase SHA-256 value used in projection claims."""
        if (
            not isinstance(value, str)
            or len(value) != 71
            or not value.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in value[7:])
        ):
            raise InputError(f"{field_name} must be a canonical SHA-256 hash.")
        return value

    @classmethod
    def _validate_projection_head_claim(
        cls,
        cursor: int | None,
        object_hash: str | None,
        field_name: str,
    ) -> tuple[int | None, str | None]:
        """Validate one optional cursor/hash pair used by drift CAS."""
        if cursor is None and object_hash is None:
            return None, None
        if (
            isinstance(cursor, bool)
            or not isinstance(cursor, int)
            or cursor < 1
            or object_hash is None
        ):
            raise InputError(
                f"{field_name} cursor and hash must be supplied together."
            )
        return cursor, cls._validate_projection_hash(object_hash, f"{field_name}_hash")

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
        for_update: bool = False,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any] | None:
        lock_clause = (
            " FOR UPDATE"
            if for_update and self._db.backend_type == BackendType.POSTGRESQL
            else ""
        )
        if include_deleted:
            cursor = self._read(
                "SELECT * FROM note_tasks "
                "WHERE owner_user_id = ? AND dataset_id = ? AND id = ?"
                + lock_clause,  # nosec B608
                (owner_user_id, dataset_id, task_id),
                conn=conn,
            )
            return self._decode_task_row(cursor.fetchone())
        if lock_clause:
            lock_clause = " FOR UPDATE OF t"
        query = (
            """
            SELECT t.*
              FROM note_tasks t
              JOIN notes n ON n.id = t.note_id
             WHERE t.owner_user_id = ? AND t.dataset_id = ? AND t.id = ?
               AND n.client_id = t.owner_user_id AND t.deleted = ? AND n.deleted = ?
            """
            + lock_clause  # nosec B608
        )
        cursor = self._read(
            query,
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
        return self._with_scoped_read(
            dataset_id=dataset,
            conn=conn,
            fn=lambda read_conn: self._fetch_projection(
                task_id,
                owner_user_id=owner,
                dataset_id=dataset,
                conn=read_conn,
            ),
        )

    def get_note_reconciliation_snapshot(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        note_id: str,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any] | None:
        """Return note fields needed to validate task reconciliation input."""
        owner, dataset = self._scope(owner_user_id, dataset_id)

        def _read_snapshot(read_conn: TaskConnection | None) -> dict[str, Any] | None:
            cursor = self._read(
                "SELECT id, version, content, deleted FROM notes WHERE client_id = ? AND id = ?",
                (owner, note_id),
                conn=read_conn,
            )
            row = cursor.fetchone()
            return dict(row) if row else None

        return self._with_scoped_read(
            dataset_id=dataset,
            conn=conn,
            fn=_read_snapshot,
        )

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

        def _read_projected_tasks(
            read_conn: TaskConnection | None,
        ) -> list[dict[str, dict[str, Any]]]:
            cursor = self._read(
                """
                SELECT *
                  FROM note_tasks
                 WHERE owner_user_id = ? AND dataset_id = ? AND note_id = ?
                   AND deleted = ? AND projection_status = ?
                 ORDER BY created_at ASC, id ASC
                """,
                (owner, dataset, note_id, self._deleted_value(False), "live"),
                conn=read_conn,
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
                    conn=read_conn,
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
                        "projection": self._require_live_projection_row(
                            task["id"], projection
                        ),
                    }
                )
            return projected_tasks

        return self._with_scoped_read(
            dataset_id=dataset,
            conn=conn,
            fn=_read_projected_tasks,
        )

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

    @staticmethod
    def _sync_task_materialization_checkpoint(_stage: str) -> None:
        """No-op seam used to prove product transaction rollback."""

    @staticmethod
    def _sync_task_activity_materialization_checkpoint(_stage: str) -> None:
        """No-op seam used to prove activity transaction rollback."""

    @staticmethod
    def _sync_task_metadata(payload: NotesTaskV1Payload) -> dict[str, Any]:
        recurrence = (
            payload.recurrence.model_dump(mode="json")
            if payload.recurrence is not None
            else None
        )
        return {
            "description": payload.description,
            "priority": payload.priority,
            "due_date": payload.due_date,
            "estimate": payload.estimate,
            "recurrence": recurrence,
            "assignee_id": payload.assignee_id,
            "tags": list(payload.tags),
            "custom": dict(payload.custom),
        }

    def verify_sync_task_postcondition(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        payload: NotesTaskV1Payload,
        canonical_revision: int,
        canonical_hash: str,
        deleted: bool,
        expected_projection_status: str | None = None,
        conn: TaskConnection | None = None,
    ) -> bool:
        """Return whether the exact canonical task product state is durable."""

        owner, dataset = self._scope(owner_user_id, dataset_id)

        def _verify(read_conn: TaskConnection | None) -> bool:
            row = self._fetch_task(
                payload.task_id,
                owner_user_id=owner,
                dataset_id=dataset,
                include_deleted=True,
                conn=read_conn,
            )
            if row is None:
                return False
            projection_status = str(row["projection_status"])
            if expected_projection_status is not None:
                projection_matches = projection_status == expected_projection_status
            else:
                projection_matches = projection_status in {
                    "live",
                    "unlinked",
                    "ambiguous",
                }
            return bool(
                row["owner_user_id"] == owner
                and row["dataset_id"] == dataset
                and row["id"] == payload.task_id
                and row["note_id"] == payload.note_id
                and row["text"] == payload.title
                and row["status"] == payload.status
                and row["metadata_json"] == self._sync_task_metadata(payload)
                and normalize_sync_timestamp(row.get("completed_at"))
                == payload.completed_at
                and int(row["canonical_revision"]) == canonical_revision
                and row["canonical_hash"] == canonical_hash
                and bool(row["deleted"]) is deleted
                and projection_matches
                and row.get("source_diagnostic_code") is None
                and row.get("source_diagnostic_hash") is None
            )

        return self._with_scoped_read(
            dataset_id=dataset,
            conn=conn,
            fn=_verify,
        )

    def apply_sync_task_projection_status(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        task_id: str,
        note_id: str,
        projection_status: str,
        conn: TaskConnection,
    ) -> dict[str, Any]:
        """Apply validated Sync projection linkage without advancing task lineage."""

        owner, dataset = self._scope(owner_user_id, dataset_id)
        status = self._validate_projection_status(projection_status)
        if status not in {"live", "unlinked"}:
            raise InputError("Sync task projection status must be live or unlinked.")
        self._require_authorized_write_scope(
            conn,
            owner_user_id=owner,
            dataset_id=dataset,
        )
        current = self._require_active_task(
            self._fetch_task(
                task_id,
                owner_user_id=owner,
                dataset_id=dataset,
                include_deleted=True,
                conn=conn,
            ),
            task_id,
        )
        if str(current["note_id"]) != note_id:
            raise ConflictError(
                "Sync task projection note does not match its task.",
                entity="tasks",
                entity_id=task_id,
            )
        self._execute(
            conn,
            "UPDATE note_tasks SET projection_status = ? "
            "WHERE owner_user_id = ? AND dataset_id = ? AND id = ? AND note_id = ?",
            (status, owner, dataset, task_id, note_id),
        )
        self._execute(
            conn,
            "UPDATE task_note_projections SET projection_status = ? "
            "WHERE owner_user_id = ? AND dataset_id = ? AND task_id = ? AND note_id = ?",
            (status, owner, dataset, task_id, note_id),
        )
        updated = self._fetch_task(
            task_id,
            owner_user_id=owner,
            dataset_id=dataset,
            include_deleted=True,
            conn=conn,
        )
        if updated is None or str(updated["projection_status"]) != status:
            raise CharactersRAGDBError("Sync task projection status did not apply.")
        return updated

    def apply_sync_task_create(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        payload: NotesTaskV1Payload,
        canonical_revision: int,
        canonical_hash: str,
        conn: TaskConnection,
    ) -> dict[str, Any]:
        """Create one canonical unlinked task without recording activity."""

        owner, dataset = self._scope(owner_user_id, dataset_id)
        if canonical_revision != 1:
            raise ConflictError(
                "Sync task create revision is invalid.",
                entity="tasks",
                entity_id=payload.task_id,
            )  # noqa: TRY003
        self._require_authorized_write_scope(
            conn,
            owner_user_id=owner,
            dataset_id=dataset,
        )
        if self._fetch_task(
            payload.task_id,
            owner_user_id=owner,
            dataset_id=dataset,
            include_deleted=True,
            conn=conn,
        ) is not None:
            raise ConflictError(
                "Sync task identity already exists.",
                entity="tasks",
                entity_id=payload.task_id,
            )  # noqa: TRY003
        self._require_active_note(
            payload.note_id,
            payload.task_id,
            owner_user_id=owner,
            conn=conn,
        )
        now = self._db._get_current_utc_timestamp_iso()
        self._execute(
            conn,
            """
            INSERT INTO note_tasks (
                owner_user_id, dataset_id, id, note_id, text, status,
                metadata_json, projection_status, deleted, created_at, updated_at,
                completed_at, client_id, version, canonical_revision, canonical_hash,
                source_diagnostic_code, source_diagnostic_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (
                owner,
                dataset,
                payload.task_id,
                payload.note_id,
                payload.title,
                payload.status,
                self._json_dumps(self._sync_task_metadata(payload), "metadata"),
                "unlinked",
                self._deleted_value(False),
                now,
                now,
                payload.completed_at,
                self._db.client_id,
                1,
                canonical_revision,
                canonical_hash,
            ),
        )
        self._sync_task_materialization_checkpoint("create")
        if not self.verify_sync_task_postcondition(
            owner_user_id=owner,
            dataset_id=dataset,
            payload=payload,
            canonical_revision=canonical_revision,
            canonical_hash=canonical_hash,
            deleted=False,
            expected_projection_status="unlinked",
            conn=conn,
        ):
            raise CharactersRAGDBError("Sync task create postcondition failed.")  # noqa: TRY003
        created = self._fetch_task(
            payload.task_id,
            owner_user_id=owner,
            dataset_id=dataset,
            include_deleted=True,
            conn=conn,
        )
        if created is None:
            raise CharactersRAGDBError("Sync task create readback failed.")  # noqa: TRY003
        return created

    def apply_sync_task_upsert(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        payload: NotesTaskV1Payload,
        base_revision: int,
        base_hash: str,
        canonical_revision: int,
        canonical_hash: str,
        conn: TaskConnection,
    ) -> dict[str, Any]:
        """Update one live canonical task without recording activity."""

        return self._apply_sync_task_transition(
            owner_user_id=owner_user_id,
            dataset_id=dataset_id,
            payload=payload,
            base_revision=base_revision,
            base_hash=base_hash,
            canonical_revision=canonical_revision,
            canonical_hash=canonical_hash,
            deleted=False,
            restore=False,
            conn=conn,
        )

    def apply_sync_task_tombstone(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        payload: NotesTaskV1Payload,
        base_revision: int,
        base_hash: str,
        canonical_revision: int,
        canonical_hash: str,
        conn: TaskConnection,
    ) -> dict[str, Any]:
        """Soft-delete one live canonical task without recording activity."""

        return self._apply_sync_task_transition(
            owner_user_id=owner_user_id,
            dataset_id=dataset_id,
            payload=payload,
            base_revision=base_revision,
            base_hash=base_hash,
            canonical_revision=canonical_revision,
            canonical_hash=canonical_hash,
            deleted=True,
            restore=False,
            conn=conn,
        )

    def apply_sync_task_restore(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        payload: NotesTaskV1Payload,
        base_revision: int,
        base_hash: str,
        canonical_revision: int,
        canonical_hash: str,
        conn: TaskConnection,
    ) -> dict[str, Any]:
        """Restore one exact task tombstone as an unlinked live task."""

        return self._apply_sync_task_transition(
            owner_user_id=owner_user_id,
            dataset_id=dataset_id,
            payload=payload,
            base_revision=base_revision,
            base_hash=base_hash,
            canonical_revision=canonical_revision,
            canonical_hash=canonical_hash,
            deleted=False,
            restore=True,
            conn=conn,
        )

    def _apply_sync_task_transition(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        payload: NotesTaskV1Payload,
        base_revision: int,
        base_hash: str,
        canonical_revision: int,
        canonical_hash: str,
        deleted: bool,
        restore: bool,
        conn: TaskConnection,
    ) -> dict[str, Any]:
        owner, dataset = self._scope(owner_user_id, dataset_id)
        if canonical_revision != base_revision + 1:
            raise ConflictError(
                "Sync task canonical revision is stale.",
                entity="tasks",
                entity_id=payload.task_id,
            )  # noqa: TRY003
        self._require_authorized_write_scope(
            conn,
            owner_user_id=owner,
            dataset_id=dataset,
        )
        current = self._fetch_task(
            payload.task_id,
            owner_user_id=owner,
            dataset_id=dataset,
            include_deleted=True,
            for_update=True,
            conn=conn,
        )
        if (
            current is None
            or current["note_id"] != payload.note_id
            or int(current["canonical_revision"]) != base_revision
            or current["canonical_hash"] != base_hash
            or bool(current["deleted"]) is not restore
        ):
            raise ConflictError(
                "Sync task product state does not match its canonical base.",
                entity="tasks",
                entity_id=payload.task_id,
            )  # noqa: TRY003
        self._require_active_note(
            payload.note_id,
            payload.task_id,
            owner_user_id=owner,
            conn=conn,
        )
        now = self._db._get_current_utc_timestamp_iso()
        projection_status = "deleted" if deleted else "unlinked" if restore else None
        projection_sql = (
            ", projection_status = ?" if projection_status is not None else ""
        )
        params: tuple[Any, ...] = (
            payload.title,
            payload.status,
            self._json_dumps(self._sync_task_metadata(payload), "metadata"),
            now,
            payload.completed_at,
            self._deleted_value(deleted),
            canonical_revision,
            canonical_hash,
        )
        if projection_status is not None:
            params += (projection_status,)
        params += (
            owner,
            dataset,
            payload.task_id,
            base_revision,
            base_hash,
            self._deleted_value(restore),
        )
        cursor = self._execute(
            conn,
            "UPDATE note_tasks SET text = ?, status = ?, metadata_json = ?, "  # nosec B608
            "updated_at = ?, completed_at = ?, deleted = ?, version = version + 1, "
            "canonical_revision = ?, canonical_hash = ?, source_diagnostic_code = NULL, "
            "source_diagnostic_hash = NULL"
            + projection_sql
            + " WHERE owner_user_id = ? AND dataset_id = ? AND id = ? "
            "AND canonical_revision = ? AND canonical_hash = ? AND deleted = ?",  # nosec B608
            params,
        )
        if getattr(cursor, "rowcount", None) == 0:
            raise ConflictError(
                "Sync task product state changed concurrently.",
                entity="tasks",
                entity_id=payload.task_id,
            )  # noqa: TRY003
        self._sync_task_materialization_checkpoint(
            "restore" if restore else "tombstone" if deleted else "upsert"
        )
        expected_projection = (
            "deleted" if deleted else "unlinked" if restore else None
        )
        if not self.verify_sync_task_postcondition(
            owner_user_id=owner,
            dataset_id=dataset,
            payload=payload,
            canonical_revision=canonical_revision,
            canonical_hash=canonical_hash,
            deleted=deleted,
            expected_projection_status=expected_projection,
            conn=conn,
        ):
            raise CharactersRAGDBError("Sync task transition postcondition failed.")  # noqa: TRY003
        updated = self._fetch_task(
            payload.task_id,
            owner_user_id=owner,
            dataset_id=dataset,
            include_deleted=True,
            conn=conn,
        )
        if updated is None:
            raise CharactersRAGDBError("Sync task transition readback failed.")  # noqa: TRY003
        return updated

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
        canonical_revision, canonical_hash, diagnostic_code, diagnostic_hash = self._canonical_task_values(
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
            self._require_authorized_write_scope(
                transaction_conn, owner_user_id=owner, dataset_id=dataset
            )
            self._require_active_note(note_id, final_task_id, owner_user_id=owner, conn=transaction_conn)
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
        return self._with_scoped_read(
            dataset_id=dataset,
            conn=conn,
            fn=lambda read_conn: self._fetch_task(
                task_id,
                owner_user_id=owner,
                dataset_id=dataset,
                include_deleted=include_deleted,
                conn=read_conn,
            ),
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
        clauses: list[str] = ["t.owner_user_id = ?", "t.dataset_id = ?"]
        params: list[Any] = [owner, dataset]
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
            sql_query += " JOIN notes n ON n.id = t.note_id AND n.client_id = t.owner_user_id"
        if clauses:
            sql_query += " WHERE " + " AND ".join(clauses)
        sql_query += " ORDER BY t.created_at ASC, t.id ASC LIMIT ? OFFSET ?"
        params.append(self._clamp_limit(limit))
        params.append(self._normalize_offset(offset))

        def _read_tasks(read_conn: TaskConnection | None) -> list[dict[str, Any]]:
            cursor = self._read(sql_query, tuple(params), conn=read_conn)
            return [self._decode_task_row(row) for row in cursor.fetchall()]

        return self._with_scoped_read(
            dataset_id=dataset,
            conn=None,
            fn=_read_tasks,
        )

    def page_tasks_for_sync_bootstrap(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        after_task_id: str | None = None,
        limit: int = 500,
        conn: TaskConnection | None = None,
    ) -> list[dict[str, Any]]:
        """Return one canonical task keyset page for private Sync bootstrap."""

        if isinstance(limit, bool) or not 1 <= limit <= 500:
            raise ValueError("Notes task bootstrap page limit must be 1..500")
        owner, dataset = self._scope(owner_user_id, dataset_id)
        cursor = str(after_task_id or "")

        def _page(read_conn: TaskConnection | None) -> list[dict[str, Any]]:
            rows = self._read(
                "SELECT * FROM note_tasks WHERE owner_user_id = ? AND dataset_id = ? "
                "AND id > ? ORDER BY id ASC LIMIT ?",
                (owner, dataset, cursor, limit),
                conn=read_conn,
            ).fetchall()
            return [self._sync_bootstrap_task_row(row, owner) for row in rows]

        return self._with_scoped_read(dataset_id=dataset, conn=conn, fn=_page)

    @staticmethod
    def _sync_bootstrap_task_row(row: Mapping[str, Any], owner: str) -> dict[str, Any]:
        decoded = dict(row)
        raw_metadata = decoded.get("metadata_json")
        if isinstance(raw_metadata, str):
            try:
                metadata = json.loads(raw_metadata)
            except (TypeError, ValueError) as exc:
                raise CharactersRAGDBError("notes_task_source_invalid") from exc
        else:
            metadata = raw_metadata
        if not isinstance(metadata, dict) or decoded.get("source_diagnostic_code") is not None:
            raise CharactersRAGDBError("notes_task_source_invalid")
        metadata_keys = set(metadata)
        legacy_keys = {"due_date", "priority", "estimate"}
        canonical_keys = {
            "description",
            "priority",
            "due_date",
            "estimate",
            "recurrence",
            "assignee_id",
            "tags",
            "custom",
        }
        legacy = metadata_keys.issubset(legacy_keys)
        if not legacy and metadata_keys != canonical_keys:
            raise CharactersRAGDBError("notes_task_source_invalid")
        raw_payload: dict[str, Any] = {
            "task_id": decoded.get("id"),
            "note_id": decoded.get("note_id"),
            "title": decoded.get("text"),
            "description": None if legacy else metadata.get("description"),
            "status": decoded.get("status"),
            "completed_at": normalize_sync_timestamp(decoded.get("completed_at")),
            "priority": metadata.get("priority"),
            "due_date": metadata.get("due_date"),
            "estimate": metadata.get("estimate"),
            "recurrence": None if legacy else metadata.get("recurrence"),
            "assignee_id": None if legacy else metadata.get("assignee_id"),
            "tags": [] if legacy else metadata.get("tags"),
            "custom": {} if legacy else metadata.get("custom"),
        }
        try:
            payload = parse_notes_task_v1(raw_payload, owner_user_id=owner)
            revision = int(decoded.get("canonical_revision") or 0)
            expected_hash = notes_task_object_hash(
                payload,
                revision=revision,
                deleted=bool(decoded.get("deleted")),
            )
        except (TypeError, ValueError) as exc:
            raise CharactersRAGDBError("notes_task_source_invalid") from exc
        if decoded.get("canonical_hash") != expected_hash:
            raise CharactersRAGDBError("notes_task_source_invalid")
        decoded["metadata_json"] = metadata
        decoded["sync_payload"] = payload.model_dump(mode="json")
        return decoded

    def page_legacy_events_for_sync_bootstrap(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        after_created_at: str | None = None,
        after_activity_id: str | None = None,
        limit: int = 1_000,
        conn: TaskConnection | None = None,
    ) -> list[dict[str, Any]]:
        """Return one exact legacy/adopted activity source page."""

        if isinstance(limit, bool) or not 1 <= limit <= 1_000:
            raise ValueError("Notes task activity bootstrap page limit must be 1..1000")
        if (after_created_at is None) != (after_activity_id is None):
            raise ValueError("Notes task activity bootstrap cursor fields must be paired")
        normalized_after = normalize_sync_timestamp(after_created_at)
        if after_created_at is not None and (
            normalized_after != after_created_at
            or not isinstance(after_activity_id, str)
            or not after_activity_id
        ):
            raise ValueError("Notes task activity bootstrap cursor is invalid")
        owner, dataset = self._scope(owner_user_id, dataset_id)
        clauses = [
            "e.owner_user_id = ?",
            "e.dataset_id = ?",
            "((e.sync_server_cursor IS NULL "
            "AND e.source_diagnostic_code = 'legacy_task_activity_unverified') "
            "OR (e.sync_server_cursor IS NOT NULL "
            "AND e.source_kind = 'trusted_bootstrap_v1'))",
            "(e.task_id IS NULL OR t.id IS NOT NULL)",
        ]
        params: list[Any] = [owner, dataset]
        if normalized_after is not None:
            clauses.append("(e.created_at > ? OR (e.created_at = ? AND e.id > ?))")
            params.extend((normalized_after, normalized_after, after_activity_id))
        params.append(limit)
        # Predicate fragments are fixed above; every value remains parameterized.
        query = (
            "SELECT e.*, t.note_id AS resolved_task_note_id FROM task_events e "  # nosec B608
            "LEFT JOIN note_tasks t ON t.owner_user_id = e.owner_user_id "
            "AND t.dataset_id = e.dataset_id AND t.id = e.task_id "
            "WHERE "
            + " AND ".join(clauses)
            + " ORDER BY e.created_at ASC, e.id ASC LIMIT ?"
        )

        def _page(read_conn: TaskConnection | None) -> list[dict[str, Any]]:
            rows = self._read(query, tuple(params), conn=read_conn).fetchall()
            result: list[dict[str, Any]] = []
            for raw in rows:
                row = self._decode_event_row(raw)
                if row is None:
                    continue
                row["created_at"] = normalize_sync_timestamp(row.get("created_at"))
                result.append(row)
            return result

        return self._with_scoped_read(dataset_id=dataset, conn=conn, fn=_page)

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
            self._require_authorized_write_scope(
                transaction_conn, owner_user_id=owner, dataset_id=dataset
            )
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
            self._require_authorized_write_scope(
                transaction_conn, owner_user_id=owner, dataset_id=dataset
            )
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
            self._require_authorized_write_scope(
                transaction_conn, owner_user_id=owner, dataset_id=dataset
            )
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
            self._require_authorized_write_scope(
                transaction_conn, owner_user_id=owner, dataset_id=dataset
            )
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
            self._require_authorized_write_scope(
                transaction_conn, owner_user_id=owner, dataset_id=dataset
            )
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

    @staticmethod
    def _sync_activity_legacy_context(
        payload: NotesTaskActivityV1,
    ) -> tuple[str | None, str | None, str | None]:
        """Map verified legacy context into the existing audit columns."""

        context = payload.metadata.get("legacy_context")
        if not isinstance(context, Mapping):
            return None, None, None
        return tuple(
            value if isinstance((value := context.get(field)), str) else None
            for field in ("tool_name", "policy_mode", "approval_id")
        )

    def _require_sync_activity_parents(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        payload: NotesTaskActivityV1,
        conn: TaskConnection,
    ) -> None:
        """Require exact owner/dataset/note/task and correction scope."""

        note = self._read(
            "SELECT id FROM notes WHERE client_id = ? AND id = ?",
            (owner_user_id, payload.note_id),
            conn=conn,
        ).fetchone()
        if note is None:
            raise ConflictError(
                "Sync task activity note not found.",
                entity="task_activity",
                entity_id=payload.activity_id,
            )  # noqa: TRY003
        if payload.task_id is not None:
            task = self._fetch_task(
                payload.task_id,
                owner_user_id=owner_user_id,
                dataset_id=dataset_id,
                include_deleted=True,
                conn=conn,
            )
            if task is None or task["note_id"] != payload.note_id:
                raise ConflictError(
                    "Sync task activity task scope does not match its note.",
                    entity="task_activity",
                    entity_id=payload.activity_id,
                )  # noqa: TRY003
        if payload.corrects_activity_id is not None:
            corrected = self.get_sync_task_activity(
                owner_user_id=owner_user_id,
                dataset_id=dataset_id,
                activity_id=payload.corrects_activity_id,
                conn=conn,
            )
            if (
                corrected is None
                or corrected["note_id"] != payload.note_id
                or corrected["task_id"] != payload.task_id
            ):
                raise ConflictError(
                    "Sync task activity correction scope does not match its target.",
                    entity="task_activity",
                    entity_id=payload.activity_id,
                )  # noqa: TRY003

    @staticmethod
    def _sync_activity_create_row_matches(
        row: Mapping[str, Any],
        payload: NotesTaskActivityV1,
    ) -> bool:
        """Return whether immutable product columns equal a canonical create."""

        tool_name, policy_mode, approval_id = TaskStore._sync_activity_legacy_context(payload)
        return bool(
            row["id"] == payload.activity_id
            and row["note_id"] == payload.note_id
            and row["task_id"] == payload.task_id
            and row["event_type"] == payload.event_type
            and row["actor_type"] == payload.actor_type
            and row["actor_id"] == payload.actor_id
            and row["tool_name"] == tool_name
            and row["policy_mode"] == policy_mode
            and row["approval_id"] == approval_id
            and row["old_value_json"] == payload.old_value
            and row["new_value_json"] == payload.new_value
            and row["source_device_id"] == payload.source_device_id
            and normalize_sync_timestamp(row["client_occurred_at"])
            == payload.client_occurred_at
            and row["source_kind"] == payload.source_kind
            and row["corrects_activity_id"] == payload.corrects_activity_id
            and row.get("source_diagnostic_code") is None
            and row.get("source_diagnostic_hash") is None
        )

    def verify_sync_task_activity_postcondition(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        payload: NotesTaskActivityV1 | NotesTaskActivityTombstoneV1,
        sync_revision: int,
        sync_object_hash: str,
        sync_server_cursor: int,
        original_payload: NotesTaskActivityV1 | None = None,
        conn: TaskConnection | None = None,
    ) -> bool:
        """Return whether the exact immutable activity state is durable."""

        owner, dataset = self._scope(owner_user_id, dataset_id)
        activity_id = (
            payload.activity_id
            if isinstance(payload, NotesTaskActivityV1)
            else (original_payload.activity_id if original_payload is not None else "")
        )

        def _verify(read_conn: TaskConnection | None) -> bool:
            row = self.get_sync_task_activity(
                owner_user_id=owner,
                dataset_id=dataset,
                activity_id=activity_id,
                conn=read_conn,
            )
            if row is None:
                return False
            create_payload = payload if isinstance(payload, NotesTaskActivityV1) else original_payload
            if create_payload is None or not self._sync_activity_create_row_matches(row, create_payload):
                return False
            lifecycle_matches = (
                not bool(row["deleted"])
                and row["deleted_at"] is None
                and row["delete_reason"] is None
                if sync_revision == 1
                else bool(row["deleted"])
                and isinstance(payload, NotesTaskActivityTombstoneV1)
                and normalize_sync_timestamp(row["deleted_at"]) == payload.deleted_at
                and row["delete_reason"] == payload.delete_reason
            )
            return bool(
                row["owner_user_id"] == owner
                and row["dataset_id"] == dataset
                and int(row["sync_revision"]) == sync_revision
                and row["sync_object_hash"] == sync_object_hash
                and int(row["sync_server_cursor"]) == sync_server_cursor
                and lifecycle_matches
            )

        return self._with_scoped_read(dataset_id=dataset, conn=conn, fn=_verify)

    def create_sync_task_activity(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        payload: NotesTaskActivityV1,
        sync_object_hash: str,
        sync_server_cursor: int,
        conn: TaskConnection,
    ) -> dict[str, Any]:
        """Insert one canonical revision-1 activity exactly once."""

        owner, dataset = self._scope(owner_user_id, dataset_id)
        expected_hash = notes_task_activity_object_hash(payload, revision=1, deleted=False)
        if sync_object_hash != expected_hash or isinstance(sync_server_cursor, bool) or sync_server_cursor < 1:
            raise ConflictError(
                "Sync task activity create lineage is invalid.",
                entity="task_activity",
                entity_id=payload.activity_id,
            )  # noqa: TRY003
        self._require_authorized_write_scope(conn, owner_user_id=owner, dataset_id=dataset)
        existing = self.get_sync_task_activity(
            owner_user_id=owner,
            dataset_id=dataset,
            activity_id=payload.activity_id,
            conn=conn,
        )
        self._require_sync_activity_parents(
            owner_user_id=owner,
            dataset_id=dataset,
            payload=payload,
            conn=conn,
        )
        tool_name, policy_mode, approval_id = self._sync_activity_legacy_context(payload)
        if existing is not None:
            if not self._legacy_sync_activity_matches(
                existing,
                payload=payload,
                owner_user_id=owner,
                dataset_id=dataset,
                conn=conn,
            ):
                raise ConflictError(
                    "Sync task activity identity already exists.",
                    entity="task_activity",
                    entity_id=payload.activity_id,
                )  # noqa: TRY003
            cursor = self._execute(
                conn,
                """
                UPDATE task_events
                   SET event_type = ?, actor_type = ?, actor_id = ?, tool_name = ?,
                       policy_mode = ?, approval_id = ?, old_value_json = ?,
                       new_value_json = ?, sync_revision = 1, sync_object_hash = ?,
                       sync_server_cursor = ?, source_device_id = ?, client_occurred_at = ?,
                       source_kind = ?, corrects_activity_id = ?, deleted = ?,
                       deleted_at = NULL, delete_reason = NULL,
                       source_diagnostic_code = NULL, source_diagnostic_hash = NULL
                 WHERE owner_user_id = ? AND dataset_id = ? AND id = ?
                   AND sync_server_cursor IS NULL
                   AND source_diagnostic_code = 'legacy_task_activity_unverified'
                """,
                (
                    payload.event_type,
                    payload.actor_type,
                    payload.actor_id,
                    tool_name,
                    policy_mode,
                    approval_id,
                    json.dumps(payload.old_value, sort_keys=True)
                    if payload.old_value is not None
                    else None,
                    json.dumps(payload.new_value, sort_keys=True)
                    if payload.new_value is not None
                    else None,
                    sync_object_hash,
                    sync_server_cursor,
                    payload.source_device_id,
                    payload.client_occurred_at,
                    payload.source_kind,
                    payload.corrects_activity_id,
                    self._deleted_value(False),
                    owner,
                    dataset,
                    payload.activity_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ConflictError(
                    "Sync task activity legacy adoption compare-and-swap failed.",
                    entity="task_activity",
                    entity_id=payload.activity_id,
                )  # noqa: TRY003
            self._sync_task_activity_materialization_checkpoint("adopt")
            if not self.verify_sync_task_activity_postcondition(
                owner_user_id=owner,
                dataset_id=dataset,
                payload=payload,
                sync_revision=1,
                sync_object_hash=sync_object_hash,
                sync_server_cursor=sync_server_cursor,
                conn=conn,
            ):
                raise CharactersRAGDBError("Sync task activity adoption postcondition failed.")  # noqa: TRY003
            adopted = self.get_sync_task_activity(
                owner_user_id=owner,
                dataset_id=dataset,
                activity_id=payload.activity_id,
                conn=conn,
            )
            if adopted is None:
                raise CharactersRAGDBError("Sync task activity adoption readback failed.")  # noqa: TRY003
            return adopted
        now = self._db._get_current_utc_timestamp_iso()
        self._execute(
            conn,
            """
            INSERT INTO task_events (
                owner_user_id, dataset_id, id, task_id, note_id, event_type, actor_type,
                actor_id, tool_name, policy_mode, approval_id, old_value_json, new_value_json,
                created_at, client_id, sync_revision, sync_object_hash, sync_server_cursor,
                source_device_id, client_occurred_at, source_kind, corrects_activity_id,
                deleted, deleted_at, delete_reason, source_diagnostic_code,
                source_diagnostic_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL)
            """,
            (
                owner,
                dataset,
                payload.activity_id,
                payload.task_id,
                payload.note_id,
                payload.event_type,
                payload.actor_type,
                payload.actor_id,
                tool_name,
                policy_mode,
                approval_id,
                json.dumps(payload.old_value, sort_keys=True) if payload.old_value is not None else None,
                json.dumps(payload.new_value, sort_keys=True) if payload.new_value is not None else None,
                now,
                self._db.client_id,
                sync_object_hash,
                sync_server_cursor,
                payload.source_device_id,
                payload.client_occurred_at,
                payload.source_kind,
                payload.corrects_activity_id,
                self._deleted_value(False),
            ),
        )
        self._sync_task_activity_materialization_checkpoint("create")
        if not self.verify_sync_task_activity_postcondition(
            owner_user_id=owner,
            dataset_id=dataset,
            payload=payload,
            sync_revision=1,
            sync_object_hash=sync_object_hash,
            sync_server_cursor=sync_server_cursor,
            conn=conn,
        ):
            raise CharactersRAGDBError("Sync task activity create postcondition failed.")  # noqa: TRY003
        created = self.get_sync_task_activity(
            owner_user_id=owner,
            dataset_id=dataset,
            activity_id=payload.activity_id,
            conn=conn,
        )
        if created is None:
            raise CharactersRAGDBError("Sync task activity create readback failed.")  # noqa: TRY003
        return created

    def _legacy_sync_activity_matches(
        self,
        row: Mapping[str, Any],
        *,
        payload: NotesTaskActivityV1,
        owner_user_id: str,
        dataset_id: str,
        conn: TaskConnection,
    ) -> bool:
        """Verify that an unmaterialized legacy row converts to the payload."""

        if (
            row.get("sync_server_cursor") is not None
            or row.get("source_diagnostic_code") != "legacy_task_activity_unverified"
            or bool(row.get("deleted"))
        ):
            return False
        resolved_note_id: str | None = None
        if row.get("task_id") is not None:
            task = self._fetch_task(
                str(row["task_id"]),
                owner_user_id=owner_user_id,
                dataset_id=dataset_id,
                include_deleted=True,
                conn=conn,
            )
            if task is None:
                return False
            resolved_note_id = str(task["note_id"])
        source = {
            key: row.get(key)
            for key in (
                "id",
                "task_id",
                "note_id",
                "event_type",
                "actor_type",
                "actor_id",
                "tool_name",
                "policy_mode",
                "approval_id",
                "old_value",
                "new_value",
                "created_at",
                "client_id",
            )
        }
        source["old_value"] = row.get("old_value_json")
        source["new_value"] = row.get("new_value_json")
        try:
            converted = convert_legacy_task_event(
                source,
                owner_user_id=owner_user_id,
                resolved_task_note_id=resolved_note_id,
            )
        except Exception:  # noqa: BLE001 - mismatch is a closed boolean result.
            return False
        return converted.model_dump(mode="json") == payload.model_dump(mode="json")

    def tombstone_sync_task_activity(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        activity_id: str,
        payload: NotesTaskActivityTombstoneV1,
        original_payload: NotesTaskActivityV1,
        base_server_cursor: int,
        base_hash: str,
        sync_object_hash: str,
        sync_server_cursor: int,
        conn: TaskConnection,
    ) -> dict[str, Any]:
        """Apply the exact one-way revision-2 activity tombstone CAS."""

        owner, dataset = self._scope(owner_user_id, dataset_id)
        expected_hash = notes_task_activity_object_hash(
            payload,
            revision=2,
            deleted=True,
            activity_id=activity_id,
            original_create_hash=base_hash,
        )
        if (
            activity_id != original_payload.activity_id
            or sync_object_hash != expected_hash
            or isinstance(sync_server_cursor, bool)
            or sync_server_cursor < 1
        ):
            raise ConflictError(
                "Sync task activity tombstone lineage is invalid.",
                entity="task_activity",
                entity_id=activity_id,
            )  # noqa: TRY003
        self._require_authorized_write_scope(conn, owner_user_id=owner, dataset_id=dataset)
        current = self.get_sync_task_activity(
            owner_user_id=owner,
            dataset_id=dataset,
            activity_id=activity_id,
            conn=conn,
        )
        if (
            current is None
            or bool(current["deleted"])
            or int(current["sync_revision"]) != 1
            or current["sync_server_cursor"] != base_server_cursor
            or current["sync_object_hash"] != base_hash
            or not self._sync_activity_create_row_matches(current, original_payload)
        ):
            raise ConflictError(
                "Sync task activity base state does not match.",
                entity="task_activity",
                entity_id=activity_id,
            )  # noqa: TRY003
        cursor = self._execute(
            conn,
            """
            UPDATE task_events
               SET sync_revision = 2, sync_object_hash = ?, sync_server_cursor = ?,
                   deleted = ?, deleted_at = ?, delete_reason = ?,
                   source_diagnostic_code = NULL, source_diagnostic_hash = NULL
             WHERE owner_user_id = ? AND dataset_id = ? AND id = ?
               AND sync_revision = 1 AND sync_object_hash = ? AND deleted = ?
            """,
            (
                sync_object_hash,
                sync_server_cursor,
                self._deleted_value(True),
                payload.deleted_at,
                payload.delete_reason,
                owner,
                dataset,
                activity_id,
                base_hash,
                self._deleted_value(False),
            ),
        )
        if cursor.rowcount != 1:
            raise ConflictError(
                "Sync task activity tombstone compare-and-swap failed.",
                entity="task_activity",
                entity_id=activity_id,
            )  # noqa: TRY003
        self._sync_task_activity_materialization_checkpoint("tombstone")
        if not self.verify_sync_task_activity_postcondition(
            owner_user_id=owner,
            dataset_id=dataset,
            payload=payload,
            original_payload=original_payload,
            sync_revision=2,
            sync_object_hash=sync_object_hash,
            sync_server_cursor=sync_server_cursor,
            conn=conn,
        ):
            raise CharactersRAGDBError("Sync task activity tombstone postcondition failed.")  # noqa: TRY003
        tombstoned = self.get_sync_task_activity(
            owner_user_id=owner,
            dataset_id=dataset,
            activity_id=activity_id,
            conn=conn,
        )
        if tombstoned is None:
            raise CharactersRAGDBError("Sync task activity tombstone readback failed.")  # noqa: TRY003
        return tombstoned

    def get_sync_task_activity(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        activity_id: str,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any] | None:
        """Read one activity through its exact canonical scope."""

        owner, dataset = self._scope(owner_user_id, dataset_id)

        def _get(read_conn: TaskConnection | None) -> dict[str, Any] | None:
            row = self._read(
                """
                SELECT e.* FROM task_events e
                JOIN notes n ON n.client_id = e.owner_user_id AND n.id = e.note_id
                LEFT JOIN note_tasks t
                  ON t.owner_user_id = e.owner_user_id AND t.dataset_id = e.dataset_id
                 AND t.id = e.task_id AND t.note_id = e.note_id
                WHERE e.owner_user_id = ? AND e.dataset_id = ? AND e.id = ?
                  AND (e.task_id IS NULL OR t.id IS NOT NULL)
                """,
                (owner, dataset, activity_id),
                conn=read_conn,
            ).fetchone()
            return self._decode_event_row(row)

        return self._with_scoped_read(dataset_id=dataset, conn=conn, fn=_get)

    def page_sync_task_activity(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        after_server_cursor: int | None = None,
        after_activity_id: str | None = None,
        limit: int = 100,
        conn: TaskConnection | None = None,
    ) -> list[dict[str, Any]]:
        """Keyset-page canonical activity by server cursor and activity ID."""

        owner, dataset = self._scope(owner_user_id, dataset_id)
        if (after_server_cursor is None) != (after_activity_id is None):
            raise InputError("Activity keyset cursor fields must be provided together.")  # noqa: TRY003
        if after_server_cursor is not None and (
            isinstance(after_server_cursor, bool)
            or not isinstance(after_server_cursor, int)
            or after_server_cursor < 1
            or not isinstance(after_activity_id, str)
            or not after_activity_id
        ):
            raise InputError("Activity keyset cursor is invalid.")  # noqa: TRY003
        try:
            bounded_limit = min(max(int(limit), 1), 1_000)
        except (TypeError, ValueError) as exc:
            raise InputError("limit must be an integer.") from exc  # noqa: TRY003
        clauses = [
            "e.owner_user_id = ?",
            "e.dataset_id = ?",
            "e.sync_server_cursor IS NOT NULL",
            "(e.task_id IS NULL OR t.id IS NOT NULL)",
        ]
        params: list[Any] = [owner, dataset]
        if after_server_cursor is not None:
            clauses.append(
                "(e.sync_server_cursor > ? OR "
                "(e.sync_server_cursor = ? AND e.id > ?))"
            )
            params.extend((after_server_cursor, after_server_cursor, after_activity_id))
        params.append(bounded_limit)
        # Predicate fragments are fixed above; every value remains parameterized.
        query = (
            "SELECT e.* FROM task_events e "  # nosec B608
            "JOIN notes n ON n.client_id = e.owner_user_id AND n.id = e.note_id "
            "LEFT JOIN note_tasks t ON t.owner_user_id = e.owner_user_id "
            "AND t.dataset_id = e.dataset_id AND t.id = e.task_id AND t.note_id = e.note_id "
            "WHERE "
            + " AND ".join(clauses)
            + " ORDER BY e.sync_server_cursor ASC, e.id ASC LIMIT ?"
        )

        def _page(read_conn: TaskConnection | None) -> list[dict[str, Any]]:
            rows = self._read(query, tuple(params), conn=read_conn).fetchall()
            return [self._decode_event_row(row) for row in rows]

        return self._with_scoped_read(dataset_id=dataset, conn=conn, fn=_page)

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
        event_hash = self._db._note_task_v60_hash(
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
            self._require_authorized_write_scope(
                transaction_conn, owner_user_id=owner, dataset_id=dataset
            )
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
        clauses: list[str] = ["owner_user_id = ?", "dataset_id = ?"]
        params: list[Any] = [owner, dataset]
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

        def _read_activity(read_conn: TaskConnection | None) -> list[dict[str, Any]]:
            cursor = self._read(query, tuple(params), conn=read_conn)
            return [self._decode_event_row(row) for row in cursor.fetchall()]

        return self._with_scoped_read(
            dataset_id=dataset,
            conn=None,
            fn=_read_activity,
        )

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
        clauses: list[str] = ["owner_user_id = ?", "dataset_id = ?"]
        params: list[Any] = [owner, dataset]
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

        def _read_activity(read_conn: TaskConnection | None) -> list[dict[str, Any]]:
            cursor = self._read(query, tuple(params), conn=read_conn)
            return [self._decode_event_row(row) for row in cursor.fetchall()]

        return self._with_scoped_read(
            dataset_id=dataset,
            conn=None,
            fn=_read_activity,
        )

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
        query = """
            SELECT events.*
            FROM task_events AS events
            LEFT JOIN task_event_read_state AS state
              ON state.owner_user_id = events.owner_user_id
             AND state.dataset_id = events.dataset_id
             AND state.event_id = events.id AND state.user_id = ?
            WHERE events.owner_user_id = ? AND events.dataset_id = ?
              AND (state.event_id IS NULL OR (state.read_at IS NULL AND state.dismissed_at IS NULL))
        """
        params: list[Any] = [user_id, owner, dataset]
        for column, value in (
            ("task_id", task_id),
            ("note_id", note_id),
            ("actor_type", actor_type),
        ):
            if value is not None:
                # Column names are fixed above; values remain parameterized.
                query += f" AND events.{column} = ?"  # nosec B608
                params.append(value)
        if self._db.backend_type == BackendType.SQLITE:
            query += " ORDER BY events.created_at DESC, events.rowid DESC LIMIT ?"
        else:
            query += " ORDER BY events.created_at DESC, events.id DESC LIMIT ?"
        params.append(self._clamp_limit(limit))

        def _read_activity(read_conn: TaskConnection | None) -> list[dict[str, Any]]:
            cursor = self._read(query, tuple(params), conn=read_conn)
            return [self._decode_event_row(row) for row in cursor.fetchall()]

        return self._with_scoped_read(
            dataset_id=dataset,
            conn=None,
            fn=_read_activity,
        )

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
            self._require_authorized_write_scope(
                transaction_conn, owner_user_id=owner, dataset_id=dataset
            )
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
            self._require_authorized_write_scope(
                transaction_conn, owner_user_id=owner, dataset_id=dataset
            )
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

        def _read_state(read_conn: TaskConnection | None) -> dict[str, Any] | None:
            cursor = self._read(
                """
                SELECT * FROM task_event_read_state
                 WHERE owner_user_id = ? AND dataset_id = ? AND event_id = ? AND user_id = ?
                """,
                (owner, dataset, event_id, user_id),
                conn=read_conn,
            )
            row = cursor.fetchone()
            return dict(row) if row else None

        return self._with_scoped_read(
            dataset_id=dataset,
            conn=conn,
            fn=_read_state,
        )

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

        def _read_state(read_conn: TaskConnection | None) -> dict[str, Any] | None:
            cursor = self._read(
                "SELECT * FROM note_task_reconciliation_state "
                "WHERE owner_user_id = ? AND dataset_id = ? AND note_id = ?",
                (owner, dataset, note_id),
                conn=read_conn,
            )
            row = cursor.fetchone()
            return dict(row) if row else None

        return self._with_scoped_read(
            dataset_id=dataset,
            conn=conn,
            fn=_read_state,
        )

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
            self._require_authorized_write_scope(
                transaction_conn, owner_user_id=owner, dataset_id=dataset
            )
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
        cursor = self._read(
            "SELECT * FROM note_task_reconciliation_state "
            "WHERE owner_user_id = ? AND dataset_id = ? AND note_id = ?",
            (owner_user_id, dataset_id, note_id),
            conn=conn,
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def _fetch_task_projection_drift(
        self,
        drift_id: str,
        *,
        owner_user_id: str,
        dataset_id: str,
        note_id: str | None = None,
        task_id: str | None = None,
        for_update: bool = False,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any] | None:
        """Fetch one authorized drift row, optionally locking it for CAS."""
        lock_clause = (
            " FOR UPDATE"
            if for_update and self._db.backend_type == BackendType.POSTGRESQL
            else ""
        )
        query = (
            "SELECT * FROM task_projection_drifts "  # nosec B608 -- fixed SQL;
            "WHERE owner_user_id = ? AND dataset_id = ? AND id = ? "
            "AND (? IS NULL OR note_id = ?) AND (? IS NULL OR task_id = ?)"
            + lock_clause
        )  # the only concatenated value is a backend-controlled row-lock suffix
        cursor = self._read(
            query,
            (
                owner_user_id,
                dataset_id,
                drift_id,
                note_id,
                note_id,
                task_id,
                task_id,
            ),
            conn=conn,
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    @staticmethod
    def _projection_drift_claims(row: Mapping[str, Any]) -> tuple[Any, ...]:
        """Return the immutable privacy-safe claims of one drift row."""
        return (
            row["note_id"],
            row["task_id"],
            int(row["marker_base_revision"]),
            row["marker_base_hash"],
            row.get("note_head_cursor"),
            row.get("note_head_hash"),
            row.get("task_head_cursor"),
            row.get("task_head_hash"),
            row["reason_code"],
        )

    def create_task_projection_drift(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        drift_id: str,
        note_id: str,
        task_id: str,
        marker_base_revision: int,
        marker_base_hash: str,
        note_head_cursor: int | None,
        note_head_hash: str | None,
        task_head_cursor: int | None,
        task_head_hash: str | None,
        reason_code: str,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any]:
        """Create one privacy-safe drift row or return its exact replay."""
        owner, dataset = self._scope(owner_user_id, dataset_id)
        drift = self._require_nonempty_identity(drift_id, "drift_id")
        note = self._require_nonempty_identity(note_id, "note_id")
        task = self._require_nonempty_identity(task_id, "task_id")
        if (
            isinstance(marker_base_revision, bool)
            or not isinstance(marker_base_revision, int)
            or marker_base_revision < 1
        ):
            raise InputError("marker_base_revision must be a positive integer.")
        base_hash = self._validate_projection_hash(
            marker_base_hash, "marker_base_hash"
        )
        note_cursor, note_hash = self._validate_projection_head_claim(
            note_head_cursor, note_head_hash, "note_head"
        )
        task_cursor, task_hash = self._validate_projection_head_claim(
            task_head_cursor, task_head_hash, "task_head"
        )
        if reason_code not in self._PROJECTION_DRIFT_REASONS:
            raise InputError("reason_code is not a supported projection drift reason.")
        expected_claims = (
            note,
            task,
            marker_base_revision,
            base_hash,
            note_cursor,
            note_hash,
            task_cursor,
            task_hash,
            reason_code,
        )
        now = self._db._get_current_utc_timestamp_iso()

        def _execute_create(transaction_conn: TaskConnection) -> dict[str, Any]:
            self._require_authorized_write_scope(
                transaction_conn, owner_user_id=owner, dataset_id=dataset
            )
            existing = self._fetch_task_projection_drift(
                drift,
                owner_user_id=owner,
                dataset_id=dataset,
                for_update=True,
                conn=transaction_conn,
            )
            if existing is not None:
                if self._projection_drift_claims(existing) != expected_claims:
                    raise ConflictError(
                        "Projection drift ID has changed claims.",
                        entity="tasks",
                        entity_id=drift,
                    )  # noqa: TRY003
                return existing
            task_row = self._fetch_task(
                task,
                owner_user_id=owner,
                dataset_id=dataset,
                include_deleted=True,
                conn=transaction_conn,
            )
            if task_row is None or task_row["note_id"] != note:
                raise ConflictError(
                    "Projection drift task reference not found.",
                    entity="tasks",
                    entity_id=drift,
                )  # noqa: TRY003
            self._execute(
                transaction_conn,
                """
                INSERT INTO task_projection_drifts (
                    owner_user_id, dataset_id, id, note_id, task_id,
                    marker_base_revision, marker_base_hash, note_head_cursor,
                    note_head_hash, task_head_cursor, task_head_hash, reason_code,
                    status, created_at, updated_at, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner,
                    dataset,
                    drift,
                    note,
                    task,
                    marker_base_revision,
                    base_hash,
                    note_cursor,
                    note_hash,
                    task_cursor,
                    task_hash,
                    reason_code,
                    "open",
                    now,
                    now,
                    None,
                ),
            )
            created = self._fetch_task_projection_drift(
                drift,
                owner_user_id=owner,
                dataset_id=dataset,
                note_id=note,
                task_id=task,
                conn=transaction_conn,
            )
            if created is None:
                raise CharactersRAGDBError(
                    "Failed to read created projection drift."
                )
            return created

        try:
            return self._with_transaction(_execute_create, conn)
        except sqlite3.IntegrityError as exc:
            self._raise_write_integrity_error(
                exc,
                operation="Projection drift",
                entity_id=drift,
                reference="task",
            )
        except BackendDatabaseError as exc:
            self._raise_write_backend_error(
                exc,
                operation="Projection drift",
                entity_id=drift,
                reference="task",
            )

    def get_task_projection_drift(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        note_id: str,
        task_id: str,
        drift_id: str,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any] | None:
        """Return one drift only through its complete authorized scope."""
        owner, dataset = self._scope(owner_user_id, dataset_id)

        return self._with_scoped_read(
            dataset_id=dataset,
            conn=conn,
            fn=lambda read_conn: self._fetch_task_projection_drift(
                drift_id,
                owner_user_id=owner,
                dataset_id=dataset,
                note_id=note_id,
                task_id=task_id,
                conn=read_conn,
            ),
        )

    def list_task_projection_drifts(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        note_id: str,
        task_id: str | None = None,
        status: str = "open",
        limit: int = 100,
        offset: int = 0,
        conn: TaskConnection | None = None,
    ) -> list[dict[str, Any]]:
        """Return one bounded owner/dataset/note drift page."""
        owner, dataset = self._scope(owner_user_id, dataset_id)
        if status not in self._PROJECTION_DRIFT_STATUSES:
            raise InputError("status is not a supported projection drift status.")
        page_limit = self._clamp_limit(limit)
        page_offset = self._normalize_offset(offset)

        def _read_page(read_conn: TaskConnection | None) -> list[dict[str, Any]]:
            if task_id is None:
                cursor = self._read(
                    """
                    SELECT * FROM task_projection_drifts
                     WHERE owner_user_id = ? AND dataset_id = ? AND note_id = ?
                       AND status = ?
                     ORDER BY updated_at DESC, id DESC
                     LIMIT ? OFFSET ?
                    """,
                    (owner, dataset, note_id, status, page_limit, page_offset),
                    conn=read_conn,
                )
            else:
                cursor = self._read(
                    """
                    SELECT * FROM task_projection_drifts
                     WHERE owner_user_id = ? AND dataset_id = ? AND note_id = ?
                       AND task_id = ? AND status = ?
                     ORDER BY updated_at DESC, id DESC
                     LIMIT ? OFFSET ?
                    """,
                    (
                        owner,
                        dataset,
                        note_id,
                        task_id,
                        status,
                        page_limit,
                        page_offset,
                    ),
                    conn=read_conn,
                )
            return [dict(row) for row in cursor.fetchall()]

        return self._with_scoped_read(
            dataset_id=dataset,
            conn=conn,
            fn=_read_page,
        )

    def has_open_task_projection_drift_for_task_envelope(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        task_id: str,
        object_revision: int,
        object_hash: str,
        server_cursor: int,
        conn: TaskConnection | None = None,
    ) -> bool:
        """Return whether an open drift names one exact immutable task envelope."""

        owner, dataset = self._scope(owner_user_id, dataset_id)
        task = self._require_nonempty_identity(task_id, "task_id")
        if type(object_revision) is not int or object_revision < 1:
            raise InputError("object_revision must be a positive integer.")
        canonical_hash = self._validate_projection_hash(object_hash, "object_hash")
        if type(server_cursor) is not int or server_cursor < 1:
            raise InputError("server_cursor must be a positive integer.")

        def _read_reference(read_conn: TaskConnection | None) -> bool:
            row = self._read(
                """
                SELECT 1 FROM task_projection_drifts
                 WHERE owner_user_id = ? AND dataset_id = ? AND task_id = ?
                   AND status = 'open'
                   AND (
                        (marker_base_revision = ? AND marker_base_hash = ?)
                        OR (task_head_cursor = ? AND task_head_hash = ?)
                   )
                 LIMIT 1
                """,
                (
                    owner,
                    dataset,
                    task,
                    object_revision,
                    canonical_hash,
                    server_cursor,
                    canonical_hash,
                ),
                conn=read_conn,
            ).fetchone()
            return row is not None

        return self._with_scoped_read(
            dataset_id=dataset,
            conn=conn,
            fn=_read_reference,
        )

    def has_open_task_projection_drift_for_note_envelope(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        note_id: str,
        object_hash: str,
        server_cursor: int,
        conn: TaskConnection | None = None,
    ) -> bool:
        """Return whether an open drift names one exact immutable note envelope."""

        owner, dataset = self._scope(owner_user_id, dataset_id)
        note = self._require_nonempty_identity(note_id, "note_id")
        canonical_hash = self._validate_projection_hash(object_hash, "object_hash")
        if type(server_cursor) is not int or server_cursor < 1:
            raise InputError("server_cursor must be a positive integer.")

        def _read_reference(read_conn: TaskConnection | None) -> bool:
            row = self._read(
                """
                SELECT 1 FROM task_projection_drifts
                 WHERE owner_user_id = ? AND dataset_id = ? AND note_id = ?
                   AND status = 'open'
                   AND note_head_cursor = ? AND note_head_hash = ?
                 LIMIT 1
                """,
                (owner, dataset, note, server_cursor, canonical_hash),
                conn=read_conn,
            ).fetchone()
            return row is not None

        return self._with_scoped_read(
            dataset_id=dataset,
            conn=conn,
            fn=_read_reference,
        )

    def compare_and_set_task_projection_drift(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        note_id: str,
        task_id: str,
        drift_id: str,
        expected_note_head_cursor: int | None,
        expected_note_head_hash: str | None,
        expected_task_head_cursor: int | None,
        expected_task_head_hash: str | None,
        status: str,
        conn: TaskConnection | None = None,
    ) -> dict[str, Any]:
        """Resolve or dismiss an open drift only for exact current claims."""
        owner, dataset = self._scope(owner_user_id, dataset_id)
        if status not in {"resolved", "dismissed"}:
            raise InputError("status must be 'resolved' or 'dismissed'.")
        expected_note = self._validate_projection_head_claim(
            expected_note_head_cursor, expected_note_head_hash, "note_head"
        )
        expected_task = self._validate_projection_head_claim(
            expected_task_head_cursor, expected_task_head_hash, "task_head"
        )
        now = self._db._get_current_utc_timestamp_iso()

        def _execute_cas(transaction_conn: TaskConnection) -> dict[str, Any]:
            self._require_authorized_write_scope(
                transaction_conn, owner_user_id=owner, dataset_id=dataset
            )
            current = self._fetch_task_projection_drift(
                drift_id,
                owner_user_id=owner,
                dataset_id=dataset,
                note_id=note_id,
                task_id=task_id,
                for_update=True,
                conn=transaction_conn,
            )
            if (
                current is None
                or current["status"] != "open"
                or (current["note_head_cursor"], current["note_head_hash"])
                != expected_note
                or (current["task_head_cursor"], current["task_head_hash"])
                != expected_task
            ):
                raise ConflictError(
                    "Projection drift changed concurrently.",
                    entity="tasks",
                    entity_id=drift_id,
                )  # noqa: TRY003
            updated = self._execute(
                transaction_conn,
                """
                UPDATE task_projection_drifts
                   SET status = ?, updated_at = ?, resolved_at = ?
                 WHERE owner_user_id = ? AND dataset_id = ? AND note_id = ?
                   AND task_id = ? AND id = ? AND status = 'open'
                """,
                (
                    status,
                    now,
                    now,
                    owner,
                    dataset,
                    note_id,
                    task_id,
                    drift_id,
                ),
            )
            if getattr(updated, "rowcount", None) != 1:
                raise ConflictError(
                    "Projection drift changed concurrently.",
                    entity="tasks",
                    entity_id=drift_id,
                )  # noqa: TRY003
            result = self._fetch_task_projection_drift(
                drift_id,
                owner_user_id=owner,
                dataset_id=dataset,
                note_id=note_id,
                task_id=task_id,
                conn=transaction_conn,
            )
            if result is None:
                raise CharactersRAGDBError(
                    "Failed to read updated projection drift."
                )
            return result

        return self._with_transaction(_execute_cas, conn)

    def candidate_notes_for_task_discovery(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return checklist-bearing notes whose task reconciliation is stale or missing."""
        owner, dataset = self._scope(owner_user_id, dataset_id)

        def _read_candidates(read_conn: TaskConnection | None) -> list[dict[str, Any]]:
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
                conn=read_conn,
            )
            return [dict(row) for row in cursor.fetchall()]

        return self._with_scoped_read(
            dataset_id=dataset,
            conn=None,
            fn=_read_candidates,
        )

    def count_candidate_notes_for_task_discovery(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        note_id: str | None = None,
    ) -> int:
        """Count checklist-bearing notes whose task reconciliation is stale or missing."""
        owner, dataset = self._scope(owner_user_id, dataset_id)
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
        def _read_count(read_conn: TaskConnection | None) -> int:
            cursor = self._read(sql_query, tuple(params), conn=read_conn)
            row = cursor.fetchone()
            return int(row["stale_count"] if row else 0)

        return self._with_scoped_read(
            dataset_id=dataset,
            conn=None,
            fn=_read_count,
        )

    def resolve_task_compatibility_dataset_id(
        self,
        *,
        owner_user_id: str,
        conn: TaskConnection | None = None,
    ) -> str:
        """Resolve the private immutable owner-to-dataset binding, or the sentinel."""
        owner = str(owner_user_id).strip()
        if not owner:
            raise InputError("Task owner cannot be empty.")  # noqa: TRY003
        postgres = self._db.backend_type == BackendType.POSTGRESQL
        if postgres and owner != str(self._db.client_id):
            raise ConflictError(
                "Task compatibility scope is unavailable.",
                entity="tasks",
                entity_id=owner,
            )  # noqa: TRY003

        rows = self._read(
            "SELECT owner_user_id,dataset_id FROM note_task_scope_authority "
            "WHERE owner_user_id = ? LIMIT 2",
            (owner,),
            conn=conn,
        ).fetchall()
        if not rows:
            return self._LOCAL_UNBOUND
        if len(rows) != 1:
            raise ConflictError(
                "Task compatibility scope is inconsistent.",
                entity="tasks",
                entity_id=owner,
            )  # noqa: TRY003
        row_owner = str(rows[0]["owner_user_id"]).strip()
        dataset = str(rows[0]["dataset_id"]).strip()
        if row_owner != owner or not dataset or dataset == self._LOCAL_UNBOUND:
            raise ConflictError(
                "Task compatibility scope is inconsistent.",
                entity="tasks",
                entity_id=owner,
            )  # noqa: TRY003
        return dataset

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
        postgres = self._db.backend_type == BackendType.POSTGRESQL
        if postgres and owner != str(self._db.client_id):
            raise ConflictError(
                "Task owner does not match the authenticated PostgreSQL client.",
                entity="tasks",
                entity_id=owner,
            )  # noqa: TRY003

        def _snapshot(
            transaction_conn: TaskConnection,
            dataset: str,
        ) -> dict[str, tuple[int, str]]:
            snapshot: dict[str, tuple[int, str]] = {}
            for table, ordering in self._BIND_TABLE_ORDER:
                lock_clause = " FOR UPDATE" if postgres else ""
                rows = self._read(
                    f"SELECT * FROM {table} WHERE owner_user_id = ? AND dataset_id = ? "  # nosec B608
                    f"ORDER BY {ordering}{lock_clause}",  # nosec B608
                    (owner, dataset),
                    conn=transaction_conn,
                ).fetchall()
                canonical_rows = [
                    {
                        key: value
                        for key, value in dict(row).items()
                        if key != "dataset_id"
                    }
                    for row in rows
                ]
                snapshot[table] = (
                    len(canonical_rows),
                    self._db._note_task_v60_hash(
                        self._db._note_task_v60_json_safe(canonical_rows)
                    ),
                )
            return snapshot

        def _counts(snapshot: dict[str, tuple[int, str]]) -> dict[str, int]:
            return {table: count for table, (count, _hash) in snapshot.items()}

        def _graph_datasets(transaction_conn: TaskConnection) -> set[str]:
            datasets: set[str] = set()
            for table, _ordering in self._BIND_TABLE_ORDER:
                rows = self._read(
                    f"SELECT DISTINCT dataset_id FROM {table} WHERE owner_user_id = ?",  # nosec B608
                    (owner,),
                    conn=transaction_conn,
                ).fetchall()
                for row in rows:
                    dataset = str(row["dataset_id"]).strip()
                    if not dataset:
                        raise ConflictError(
                            "Task dataset binding found malformed graph scope.",
                            entity="tasks",
                            entity_id=target,
                        )  # noqa: TRY003
                    datasets.add(dataset)
            return datasets

        def _prove_parents(transaction_conn: TaskConnection) -> None:
            invalid = self._read(
                """
                SELECT 1 FROM (
                  SELECT t.id
                    FROM note_tasks t
                   WHERE t.owner_user_id = ? AND t.dataset_id = ?
                     AND NOT EXISTS(
                       SELECT 1 FROM notes n
                        WHERE n.client_id = t.owner_user_id AND n.id = t.note_id
                     )
                  UNION ALL
                  SELECT p.task_id
                    FROM task_note_projections p
                   WHERE p.owner_user_id = ? AND p.dataset_id = ? AND (
                     NOT EXISTS(
                       SELECT 1 FROM note_tasks t
                        WHERE t.owner_user_id = p.owner_user_id
                          AND t.dataset_id = p.dataset_id AND t.id = p.task_id
                          AND t.note_id = p.note_id
                     ) OR NOT EXISTS(
                       SELECT 1 FROM notes n
                        WHERE n.client_id = p.owner_user_id AND n.id = p.note_id
                     )
                   )
                  UNION ALL
                  SELECT e.id
                    FROM task_events e
                   WHERE e.owner_user_id = ? AND e.dataset_id = ? AND (
                     NOT EXISTS(
                       SELECT 1 FROM notes n
                        WHERE n.client_id = e.owner_user_id AND n.id = e.note_id
                     ) OR (e.task_id IS NOT NULL AND NOT EXISTS(
                       SELECT 1 FROM note_tasks t
                        WHERE t.owner_user_id = e.owner_user_id
                          AND t.dataset_id = e.dataset_id AND t.id = e.task_id
                          AND t.note_id = e.note_id
                     )) OR (e.corrects_activity_id IS NOT NULL AND NOT EXISTS(
                       SELECT 1 FROM task_events corrected
                        WHERE corrected.owner_user_id = e.owner_user_id
                          AND corrected.dataset_id = e.dataset_id
                          AND corrected.id = e.corrects_activity_id
                     ))
                   )
                  UNION ALL
                  SELECT r.event_id
                    FROM task_event_read_state r
                   WHERE r.owner_user_id = ? AND r.dataset_id = ? AND (
                     r.user_id <> r.owner_user_id OR NOT EXISTS(
                       SELECT 1 FROM task_events e
                        WHERE e.owner_user_id = r.owner_user_id
                          AND e.dataset_id = r.dataset_id AND e.id = r.event_id
                     )
                   )
                  UNION ALL
                  SELECT r.note_id
                    FROM note_task_reconciliation_state r
                   WHERE r.owner_user_id = ? AND r.dataset_id = ?
                     AND NOT EXISTS(
                       SELECT 1 FROM notes n
                        WHERE n.client_id = r.owner_user_id AND n.id = r.note_id
                     )
                  UNION ALL
                  SELECT d.id
                    FROM task_projection_drifts d
                   WHERE d.owner_user_id = ? AND d.dataset_id = ? AND (
                     NOT EXISTS(
                       SELECT 1 FROM note_tasks t
                        WHERE t.owner_user_id = d.owner_user_id
                          AND t.dataset_id = d.dataset_id AND t.id = d.task_id
                          AND t.note_id = d.note_id
                     ) OR NOT EXISTS(
                       SELECT 1 FROM notes n
                        WHERE n.client_id = d.owner_user_id AND n.id = d.note_id
                     )
                   )
                ) invalid_parents
                LIMIT 1
                """,
                (
                    owner, self._LOCAL_UNBOUND,
                    owner, self._LOCAL_UNBOUND,
                    owner, self._LOCAL_UNBOUND,
                    owner, self._LOCAL_UNBOUND,
                    owner, self._LOCAL_UNBOUND,
                    owner, self._LOCAL_UNBOUND,
                ),
                conn=transaction_conn,
            ).fetchone()
            if invalid is not None:
                raise ConflictError(
                    "Task dataset binding failed parent proof.",
                    entity="tasks",
                    entity_id=target,
                )  # noqa: TRY003

        def _prepare_postgres(transaction_conn: TaskConnection) -> None:
            if not postgres:
                return
            version = self._db._get_schema_version_postgres(transaction_conn, lock=True)
            if version != self._db._POSTGRES_SCHEMA_VERSION:
                raise ConflictError(
                    "Task dataset binding requires the current PostgreSQL schema.",
                    entity="tasks",
                    entity_id=target,
                )  # noqa: TRY003
            self._execute(
                transaction_conn,
                "LOCK TABLE note_task_scope_authority, notes, note_tasks, "
                "task_note_projections, task_events, "
                "task_event_read_state, note_task_reconciliation_state, task_projection_drifts "
                "IN ACCESS EXCLUSIVE MODE",
            )
            self._db._verify_note_task_schema_postgres(transaction_conn)
            for table, _ordering in self._BIND_TABLE_ORDER:
                self._execute(
                    transaction_conn,
                    f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY",  # nosec B608
                )

        def _finish_postgres(transaction_conn: TaskConnection) -> None:
            if not postgres:
                return
            for table, _ordering in self._BIND_TABLE_ORDER:
                self._execute(
                    transaction_conn,
                    f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY",  # nosec B608
                )
            self._db._verify_note_task_schema_postgres(transaction_conn)

        def _execute_bind_body(transaction_conn: TaskConnection) -> dict[str, int]:
            _prepare_postgres(transaction_conn)
            authority_lock = " FOR UPDATE" if postgres else ""
            authority_rows = self._read(
                # authority_lock is a fixed backend suffix, never caller input.
                "SELECT owner_user_id,dataset_id FROM note_task_scope_authority "
                f"WHERE owner_user_id = ?{authority_lock}",  # nosec B608
                (owner,),
                conn=transaction_conn,
            ).fetchall()
            if len(authority_rows) > 1:
                raise ConflictError(
                    "Task dataset binding authority is inconsistent.",
                    entity="tasks",
                    entity_id=target,
                )  # noqa: TRY003
            authority_dataset = None
            if authority_rows:
                authority_owner = str(authority_rows[0]["owner_user_id"]).strip()
                authority_dataset = str(authority_rows[0]["dataset_id"]).strip()
                if (
                    authority_owner != owner
                    or not authority_dataset
                    or authority_dataset == self._LOCAL_UNBOUND
                ):
                    raise ConflictError(
                        "Task dataset binding authority is inconsistent.",
                        entity="tasks",
                        entity_id=target,
                    )  # noqa: TRY003
                if authority_dataset != target:
                    raise ConflictError(
                        "Task dataset binding is immutable.",
                        entity="tasks",
                        entity_id=target,
                    )  # noqa: TRY003

            graph_datasets = _graph_datasets(transaction_conn)
            source_snapshot = _snapshot(transaction_conn, self._LOCAL_UNBOUND)
            target_snapshot = _snapshot(transaction_conn, target)
            source_counts = _counts(source_snapshot)
            target_counts = _counts(target_snapshot)
            if authority_dataset == target:
                if graph_datasets - {target} or any(source_counts.values()):
                    raise ConflictError(
                        "Task dataset binding authority conflicts with graph scope.",
                        entity="tasks",
                        entity_id=target,
                    )  # noqa: TRY003
                _finish_postgres(transaction_conn)
                return target_counts
            if graph_datasets - {self._LOCAL_UNBOUND, target}:
                raise ConflictError(
                    "Task dataset binding found an unowned graph scope.",
                    entity="tasks",
                    entity_id=target,
                )  # noqa: TRY003
            if any(target_counts.values()):
                raise ConflictError(
                    "Task dataset binding target collision.", entity="tasks", entity_id=target
                )  # noqa: TRY003
            if not any(source_counts.values()):
                self._execute(
                    transaction_conn,
                    "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
                    (owner, target),
                )
                _finish_postgres(transaction_conn)
                return source_counts
            _prove_parents(transaction_conn)

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

            remaining = _snapshot(transaction_conn, self._LOCAL_UNBOUND)
            rebound = _snapshot(transaction_conn, target)
            if any(_counts(remaining).values()) or rebound != source_snapshot:
                raise ConflictError(
                    "Task dataset binding failed complete-set verification.",
                    entity="tasks",
                    entity_id=target,
                )  # noqa: TRY003
            self._execute(
                transaction_conn,
                "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
                (owner, target),
            )
            _finish_postgres(transaction_conn)
            return _counts(rebound)

        def _execute_bind(transaction_conn: TaskConnection) -> dict[str, int]:
            if not postgres:
                return _execute_bind_body(transaction_conn)
            self._execute(transaction_conn, "SAVEPOINT bind_local_task_graph")
            try:
                result = _execute_bind_body(transaction_conn)
            except Exception:  # noqa: BLE001 - rollback must cover every failed bind
                self._execute(transaction_conn, "ROLLBACK TO SAVEPOINT bind_local_task_graph")
                self._execute(transaction_conn, "RELEASE SAVEPOINT bind_local_task_graph")
                self._db._verify_note_task_schema_postgres(transaction_conn)
                raise
            self._execute(transaction_conn, "RELEASE SAVEPOINT bind_local_task_graph")
            return result

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
