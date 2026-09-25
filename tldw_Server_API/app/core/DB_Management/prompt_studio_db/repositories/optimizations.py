"""Prompt Studio optimizations: an optimizer run over a prompt, and its iterations."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from typing import Any, Optional

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.prompt_studio_db.repositories._common import DB_ERRORS, json_or_none
from tldw_Server_API.app.core.DB_Management.Prompts_DB import DatabaseError, InputError
from tldw_Server_API.app.core.DB_Management.retry_policy import run_with_contention_retry

ACTIVE_STATUSES = ("pending", "running")
# Moving INTO one of these is only allowed from an active status, so a late
# worker cannot overwrite a run that already finished.
GUARDED_STATUS_TARGETS = frozenset({"running", "completed", "failed", "cancelled"})

_JSON_FIELDS = frozenset({"optimization_config", "initial_metrics", "final_metrics", "test_case_ids"})
# Column names are interpolated into UPDATE, so only these may be set.
_UPDATABLE_FIELDS = _JSON_FIELDS | {
    "name", "initial_prompt_id", "optimized_prompt_id", "optimizer_type", "improvement_percentage",
    "iterations_completed", "max_iterations", "bootstrap_samples", "status", "error_message",
    "total_tokens", "total_cost", "client_id", "uuid",
}


def _paged(page: int, per_page: int) -> None:
    if page < 1:
        raise InputError("Page index must be >= 1")  # noqa: TRY003
    if per_page < 1:
        raise InputError("Items per page must be >= 1")  # noqa: TRY003


def _pagination(page: int, per_page: int, total: int) -> dict[str, int]:
    return {"page": page, "per_page": per_page, "total": total, "total_pages": (total + per_page - 1) // per_page}


def _loggable(value: Any) -> Any:
    if not isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(json.dumps(value, default=str))
    except TypeError:
        return str(value)


class OptimizationsRepository:
    def __init__(self, session: Any):
        self.session = session

    def create(
        self,
        *,
        project_id: int,
        name: Optional[str],
        initial_prompt_id: Optional[int],
        optimizer_type: str,
        optimization_config: Optional[dict[str, Any]] = None,
        max_iterations: Optional[int] = None,
        bootstrap_samples: Optional[int] = None,
        status: str = "pending",
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        db = self.session
        optimization_uuid = str(uuid.uuid4())
        # Result columns are written as explicit NULLs, not left to column defaults.
        payload = (
            optimization_uuid, project_id, name, initial_prompt_id, None, optimizer_type,
            json_or_none(optimization_config), None, None, None, 0, max_iterations, bootstrap_samples,
            status, None, None, None, client_id or db.client_id,
        )
        insert_sql = """
            INSERT INTO prompt_studio_optimizations (
                uuid, project_id, name, initial_prompt_id, optimized_prompt_id,
                optimizer_type, optimization_config, initial_metrics, final_metrics,
                improvement_percentage, iterations_completed, max_iterations,
                bootstrap_samples, status, error_message, total_tokens, total_cost,
                client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING *
        """

        def _insert() -> dict[str, Any]:
            with db._write_lock, db.transaction() as conn:
                cursor = db._cursor_exec(conn, insert_sql, payload)
                row = cursor.fetchone()
                return db._row_to_dict(cursor, row) if row else {}

        try:
            optimization = run_with_contention_retry(_insert)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to create optimization: {exc}") from exc  # noqa: TRY003
        db._log_sync_event(
            "prompt_studio_optimization",
            optimization_uuid,
            "create",
            {"project_id": project_id, "optimizer_type": optimizer_type, "status": status},
        )
        return optimization

    def get(self, optimization_id: int, *, include_deleted: bool = False) -> Optional[dict[str, Any]]:
        db = self.session
        query = "SELECT * FROM prompt_studio_optimizations WHERE id = ?"
        if not include_deleted:
            query += " AND deleted = FALSE"

        def _read() -> Optional[dict[str, Any]]:
            cursor = db._execute(query, (optimization_id,))
            row = cursor.fetchone()
            return db._row_to_dict(cursor, row) if row else None

        try:
            return run_with_contention_retry(_read)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to fetch optimization {optimization_id}: {exc}") from exc  # noqa: TRY003

    def list(
        self,
        *,
        project_id: Optional[int] = None,
        status: Optional[str] = None,
        include_deleted: bool = False,
        page: int = 1,
        per_page: int = 20,
    ) -> dict[str, Any]:
        _paged(page, per_page)
        db = self.session
        conditions: list[str] = []
        params: list[Any] = []
        if project_id is not None:
            conditions.append("project_id = ?")
            params.append(project_id)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        if not include_deleted:
            conditions.append("deleted = FALSE")
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        list_sql = (
            f"SELECT * FROM prompt_studio_optimizations{where_clause}"  # nosec B608
            " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
        )

        def _read() -> tuple[int, list[dict[str, Any]]]:
            count_row = db._execute(f"SELECT COUNT(*) FROM prompt_studio_optimizations{where_clause}", params).fetchone()  # nosec B608
            cursor = db._execute(list_sql, [*params, per_page, (page - 1) * per_page])
            rows = [db._row_to_dict(cursor, row) for row in cursor.fetchall() if row]
            return (int(count_row[0]) if count_row and count_row[0] is not None else 0), rows

        try:
            total, optimizations = run_with_contention_retry(_read)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to list optimizations: {exc}") from exc  # noqa: TRY003
        return {"optimizations": optimizations, "pagination": _pagination(page, per_page, total)}

    def update(
        self,
        optimization_id: int,
        updates: dict[str, Any],
        *,
        set_started_at: bool = False,
        set_completed_at: bool = False,
        expected_statuses: Optional[Iterable[str]] = None,
        expected_uuid: Optional[str] = None,
        _return_transition_applied: bool = False,
    ) -> dict[str, Any] | tuple[dict[str, Any], bool]:
        """Apply ``updates``; with ``expected_*`` only if the row still matches (compare-and-set).

        A guarded update that does not match returns the row unchanged; the second element of
        the ``_return_transition_applied`` tuple says whether it was applied.
        """
        expected_status_values = tuple(dict.fromkeys(str(v) for v in expected_statuses)) if expected_statuses is not None else ()
        if expected_statuses is not None and not expected_status_values:
            raise ValueError("expected_statuses cannot be empty")  # noqa: TRY003
        expected_uuid_value = str(expected_uuid).strip() if expected_uuid is not None else None
        if expected_uuid is not None and not expected_uuid_value:
            raise ValueError("expected_uuid cannot be empty")  # noqa: TRY003
        unknown = set(updates) - _UPDATABLE_FIELDS
        if unknown:
            raise InputError(f"Cannot update optimization fields: {sorted(unknown)}")  # noqa: TRY003
        guarded = bool(expected_status_values) or expected_uuid_value is not None

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
        if set_started_at:
            set_clauses.append("started_at = CURRENT_TIMESTAMP")
        if set_completed_at:
            set_clauses.append("completed_at = CURRENT_TIMESTAMP")

        def _result(optimization: dict[str, Any], applied: bool) -> Any:
            return (optimization, applied) if _return_transition_applied else optimization

        if not set_clauses:
            optimization = self.get(optimization_id, include_deleted=True)
            if optimization is None:
                raise InputError(f"Optimization {optimization_id} not found")  # noqa: TRY003
            return _result(optimization, False)

        where_sql = " WHERE id = ?"
        params.append(optimization_id)
        if expected_status_values:
            where_sql += f" AND status IN ({', '.join('?' * len(expected_status_values))})"
            params.extend(expected_status_values)
        if expected_uuid_value is not None:
            where_sql += " AND uuid = ?"
            params.append(expected_uuid_value)
        update_sql = (
            "UPDATE prompt_studio_optimizations SET "  # nosec B608 - allowlisted columns
            + ", ".join(set_clauses)
            + where_sql
            + " RETURNING *"
        )

        def _update() -> Optional[dict[str, Any]]:
            with db._write_lock, db.transaction() as conn:
                cursor = db._cursor_exec(conn, update_sql, params)
                row = cursor.fetchone()
                if not row and not guarded:
                    raise InputError(f"Optimization {optimization_id} not found")  # noqa: TRY003
                return db._row_to_dict(cursor, row) if row else None

        try:
            updated = run_with_contention_retry(_update)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to update optimization {optimization_id}: {exc}") from exc  # noqa: TRY003

        if updated is None:
            current = self.get(optimization_id, include_deleted=True)
            if current is None:
                raise InputError(f"Optimization {optimization_id} not found")  # noqa: TRY003
            return _result(current, False)

        log_payload = {key: _loggable(value) for key, value in updates.items()}
        if set_started_at:
            log_payload["started_at"] = "CURRENT_TIMESTAMP"
        if set_completed_at:
            log_payload["completed_at"] = "CURRENT_TIMESTAMP"
        db._log_sync_event("prompt_studio_optimization", updated.get("uuid", ""), "update", log_payload)
        return _result(updated, True)

    def set_status(
        self,
        optimization_id: int,
        status: str,
        *,
        error_message: Optional[str] = None,
        mark_started: bool = False,
        mark_completed: bool = False,
    ) -> dict[str, Any]:
        updates: dict[str, Any] = {"status": status}
        if error_message is not None:
            updates["error_message"] = error_message
        return self.update(
            optimization_id,
            updates,
            set_started_at=mark_started,
            set_completed_at=mark_completed,
            expected_statuses=ACTIVE_STATUSES if status in GUARDED_STATUS_TARGETS else None,
        )

    def complete(
        self,
        optimization_id: int,
        *,
        optimized_prompt_id: Optional[int] = None,
        iterations_completed: Optional[int] = None,
        initial_metrics: Optional[dict[str, Any]] = None,
        final_metrics: Optional[dict[str, Any]] = None,
        improvement_percentage: Optional[float] = None,
        total_tokens: Optional[int] = None,
        total_cost: Optional[float] = None,
        _return_transition_applied: bool = False,
    ) -> dict[str, Any] | tuple[dict[str, Any], bool]:
        results = {
            "optimized_prompt_id": optimized_prompt_id,
            "iterations_completed": iterations_completed,
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "improvement_percentage": improvement_percentage,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
        }
        # Success clears any diagnostic an earlier attempt left; omitted results stay untouched.
        updates = {"status": "completed", "error_message": None, **{k: v for k, v in results.items() if v is not None}}
        return self.update(
            optimization_id,
            updates,
            set_completed_at=True,
            expected_statuses=ACTIVE_STATUSES,
            _return_transition_applied=_return_transition_applied,
        )

    def record_iteration(
        self,
        optimization_id: int,
        *,
        iteration_number: int,
        prompt_variant: Optional[dict[str, Any]] = None,
        metrics: Optional[dict[str, Any]] = None,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
        note: Optional[str] = None,
    ) -> dict[str, Any]:
        db = self.session
        payload = (
            str(uuid.uuid4()), optimization_id, iteration_number, json_or_none(prompt_variant),
            json_or_none(metrics), tokens_used, cost, note,
        )
        insert_sql = """
            INSERT INTO prompt_studio_optimization_iterations (
                uuid, optimization_id, iteration_number, prompt_variant, metrics,
                tokens_used, cost, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING *
        """

        def _insert() -> dict[str, Any]:
            with db._write_lock, db.transaction() as conn:
                cursor = db._cursor_exec(conn, insert_sql, payload)
                row = cursor.fetchone()
                return db._row_to_dict(cursor, row) if row else {}

        try:
            record = run_with_contention_retry(_insert)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to record optimization iteration: {exc}") from exc  # noqa: TRY003
        db._log_sync_event(
            "prompt_studio_optimization_iteration",
            record.get("uuid", ""),
            "create",
            {"optimization_id": optimization_id, "iteration_number": iteration_number},
        )
        return record

    def list_iterations(self, optimization_id: int, *, page: int = 1, per_page: int = 50) -> dict[str, Any]:
        _paged(page, per_page)
        db = self.session

        def _read() -> tuple[int, list[dict[str, Any]]]:
            count_row = db._execute(
                "SELECT COUNT(*) FROM prompt_studio_optimization_iterations WHERE optimization_id = ?", (optimization_id,)
            ).fetchone()
            cursor = db._execute(
                "SELECT * FROM prompt_studio_optimization_iterations WHERE optimization_id = ?"
                " ORDER BY iteration_number ASC, id ASC LIMIT ? OFFSET ?",
                (optimization_id, per_page, (page - 1) * per_page),
            )
            rows = [db._row_to_dict(cursor, row) for row in cursor.fetchall() if row]
            return (int(count_row[0]) if count_row and count_row[0] is not None else 0), rows

        try:
            total, iterations = run_with_contention_retry(_read)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to list optimization iterations: {exc}") from exc  # noqa: TRY003
        return {"iterations": iterations, "pagination": _pagination(page, per_page, total)}
