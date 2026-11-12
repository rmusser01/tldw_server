"""Prompt Studio evaluations: a prompt scored over a set of test cases."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from typing import Any, Optional

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.prompt_studio_db.repositories._common import DB_ERRORS
from tldw_Server_API.app.core.DB_Management.Prompts_DB import DatabaseError, InputError
from tldw_Server_API.app.core.DB_Management.retry_policy import run_with_contention_retry

_JSON_FIELDS = frozenset({"model_configs", "test_case_ids", "test_run_ids", "aggregate_metrics"})
# Column names are interpolated into UPDATE, so only these may be set.
_UPDATABLE_FIELDS = _JSON_FIELDS | {
    "project_id", "prompt_id", "name", "description", "total_tokens", "total_cost",
    "status", "error_message", "client_id", "started_at", "completed_at",
}


class EvaluationsRepository:
    def __init__(self, session: Any):
        self.session = session

    def create(
        self,
        *,
        prompt_id: int,
        project_id: int,
        model_configs: Optional[dict[str, Any]] = None,
        status: str = "running",
        test_case_ids: Optional[Iterable[int]] = None,
        client_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> dict[str, Any]:
        db = self.session
        payload = (
            str(uuid.uuid4()),
            prompt_id,
            project_id,
            name,
            description,
            json.dumps(model_configs) if model_configs is not None else None,
            status,
            json.dumps(list(test_case_ids) if test_case_ids is not None else []),
            client_id or db.client_id,
        )
        insert_sql = """
            INSERT INTO prompt_studio_evaluations (
                uuid, prompt_id, project_id, name, description, model_configs, status,
                test_case_ids, started_at, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
            RETURNING *
        """

        def _insert() -> dict[str, Any]:
            with db._write_lock, db.transaction() as conn:
                cursor = db._cursor_exec(conn, insert_sql, payload)
                row = cursor.fetchone()
                return db._row_to_dict(cursor, row) if row else {}

        try:
            return run_with_contention_retry(_insert)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to create evaluation: {exc}") from exc  # noqa: TRY003

    def update(self, evaluation_id: int, updates: dict[str, Any]) -> dict[str, Any]:
        if not updates:
            evaluation = self.get(evaluation_id)
            if evaluation is None:
                raise InputError(f"Evaluation {evaluation_id} not found")  # noqa: TRY003
            return evaluation
        unknown = set(updates) - _UPDATABLE_FIELDS
        if unknown:
            raise InputError(f"Cannot update evaluation fields: {sorted(unknown)}")  # noqa: TRY003

        db = self.session
        jsonb_cast = "::jsonb" if db.backend_type == BackendType.POSTGRESQL else ""
        set_clauses: list[str] = []
        params: list[Any] = []
        for field, value in updates.items():
            if field in _JSON_FIELDS and value is not None:
                set_clauses.append(f"{field} = ?{jsonb_cast}")
                params.append(json.dumps(value))
            else:
                set_clauses.append(f"{field} = ?")
                params.append(value)
        params.append(evaluation_id)
        update_sql = (
            "UPDATE prompt_studio_evaluations SET "  # nosec B608 - allowlisted columns
            + ", ".join(set_clauses)
            + " WHERE id = ? RETURNING *"
        )

        def _update() -> dict[str, Any]:
            with db._write_lock, db.transaction() as conn:
                cursor = db._cursor_exec(conn, update_sql, params)
                row = cursor.fetchone()
                if not row:
                    raise InputError(f"Evaluation {evaluation_id} not found")  # noqa: TRY003
                return db._row_to_dict(cursor, row)

        try:
            return run_with_contention_retry(_update)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to update evaluation {evaluation_id}: {exc}") from exc  # noqa: TRY003

    def delete(self, evaluation_id: int) -> bool:
        """Delete an evaluation; False if it did not exist. (The table has no soft-delete columns.)"""
        db = self.session

        def _delete() -> bool:
            with db._write_lock, db.transaction() as conn:
                row = db._cursor_exec(
                    conn, "DELETE FROM prompt_studio_evaluations WHERE id = ? RETURNING id", (evaluation_id,)
                ).fetchone()
                return row is not None

        try:
            return run_with_contention_retry(_delete)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to delete evaluation {evaluation_id}: {exc}") from exc  # noqa: TRY003

    def cancel_if_active(self, evaluation_id: int, message: str) -> bool:
        """Mark a pending or running evaluation cancelled; a finished one is left alone."""
        db = self.session

        def _cancel() -> bool:
            with db._write_lock, db.transaction() as conn:
                row = db._cursor_exec(
                    conn,
                    "UPDATE prompt_studio_evaluations SET status = 'cancelled', error_message = ?,"
                    " completed_at = CURRENT_TIMESTAMP WHERE id = ? AND status IN ('pending', 'running') RETURNING id",
                    (message, evaluation_id),
                ).fetchone()
                return row is not None

        try:
            return run_with_contention_retry(_cancel)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to cancel evaluation {evaluation_id}: {exc}") from exc  # noqa: TRY003

    def get(self, evaluation_id: int) -> Optional[dict[str, Any]]:
        db = self.session
        try:
            cursor = db._execute("SELECT * FROM prompt_studio_evaluations WHERE id = ?", (evaluation_id,))
            row = cursor.fetchone()
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to fetch evaluation {evaluation_id}: {exc}") from exc  # noqa: TRY003
        return db._row_to_dict(cursor, row) if row else None

    def list(
        self,
        *,
        project_id: Optional[int] = None,
        prompt_id: Optional[int] = None,
        status: Optional[str] = None,
        page: int = 1,
        per_page: int = 20,
    ) -> dict[str, Any]:
        if page < 1:
            raise InputError("Page index must be >= 1")  # noqa: TRY003
        if per_page < 1:
            raise InputError("Items per page must be >= 1")  # noqa: TRY003

        db = self.session
        conditions: list[str] = []
        params: list[Any] = []
        for column, value in (("project_id", project_id), ("prompt_id", prompt_id), ("status", status)):
            if value is not None:
                conditions.append(f"{column} = ?")
                params.append(value)
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        # SQLite already sorts NULLs last under DESC, so NULLS LAST means the same on both.
        list_sql = (
            f"SELECT * FROM prompt_studio_evaluations{where_clause}"  # nosec B608
            " ORDER BY started_at DESC NULLS LAST, id DESC LIMIT ? OFFSET ?"
        )

        def _read() -> tuple[int, list[dict[str, Any]]]:
            count_cursor = db._execute(f"SELECT COUNT(*) FROM prompt_studio_evaluations{where_clause}", params)  # nosec B608
            total_row = count_cursor.fetchone()
            cursor = db._execute(list_sql, [*params, per_page, (page - 1) * per_page])
            rows = [db._row_to_dict(cursor, row) for row in cursor.fetchall() if row]
            return (int(total_row[0]) if total_row and total_row[0] is not None else 0), rows

        try:
            total, evaluations = run_with_contention_retry(_read)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to list evaluations: {exc}") from exc  # noqa: TRY003

        return {
            "evaluations": evaluations,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": (total + per_page - 1) // per_page,
            },
        }
