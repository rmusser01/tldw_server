"""Prompt Studio signatures: the input/output contract a project's prompts implement."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from typing import Any, Optional, Union

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.prompt_studio_db.repositories._common import (
    DB_ERRORS,
    is_unique_violation,
    json_or_none,
)
from tldw_Server_API.app.core.DB_Management.Prompts_DB import ConflictError, DatabaseError, InputError
from tldw_Server_API.app.core.DB_Management.retry_policy import run_with_contention_retry

_JSON_FIELDS = frozenset({"input_schema", "output_schema", "constraints", "validation_rules"})
# Other keys in `updates` are ignored, as before; column names are interpolated.
_UPDATABLE_FIELDS = _JSON_FIELDS | {"name"}


class SignaturesRepository:
    def __init__(self, session: Any):
        self.session = session

    def create(
        self,
        project_id: int,
        name: str,
        *,
        input_schema: Iterable[Any],
        output_schema: Iterable[Any],
        constraints: Optional[Any] = None,
        validation_rules: Optional[Any] = None,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if not name or not str(name).strip():
            raise InputError("Signature name cannot be empty")  # noqa: TRY003
        db = self.session
        signature_uuid = str(uuid.uuid4())
        payload = (
            signature_uuid,
            project_id,
            str(name).strip(),
            json.dumps(list(input_schema) if input_schema is not None else []),
            json.dumps(list(output_schema) if output_schema is not None else []),
            json_or_none(constraints),
            json_or_none(validation_rules),
            client_id or db.client_id,
        )
        insert_sql = """
            INSERT INTO prompt_studio_signatures (
                uuid, project_id, name, input_schema, output_schema,
                constraints, validation_rules, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING *
        """

        def _insert() -> dict[str, Any]:
            with db._write_lock, db.transaction() as conn:
                cursor = db._cursor_exec(conn, insert_sql, payload)
                row = cursor.fetchone()
                return db._row_to_dict(cursor, row) if row else {}

        try:
            signature = run_with_contention_retry(_insert)
        except (*DB_ERRORS, ConflictError) as exc:
            if is_unique_violation(exc):
                raise ConflictError(  # noqa: TRY003
                    f"Signature with name '{name}' already exists for project {project_id}"
                ) from exc
            raise DatabaseError(f"Failed to create signature: {exc}") from exc  # noqa: TRY003
        db._log_sync_event(
            "prompt_studio_signature", signature_uuid, "create", {"project_id": project_id, "name": name}
        )
        return signature

    def get(self, signature_id: int, *, include_deleted: bool = False) -> Optional[dict[str, Any]]:
        db = self.session
        query = "SELECT * FROM prompt_studio_signatures WHERE id = ?"
        if not include_deleted:
            query += " AND deleted = FALSE"

        def _read() -> Optional[dict[str, Any]]:
            cursor = db._execute(query, (signature_id,))
            row = cursor.fetchone()
            return db._row_to_dict(cursor, row) if row else None

        try:
            return run_with_contention_retry(_read)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to fetch signature {signature_id}: {exc}") from exc  # noqa: TRY003

    def list(
        self,
        project_id: int,
        *,
        include_deleted: bool = False,
        search: Optional[str] = None,
        page: int = 1,
        per_page: int = 20,
        return_pagination: bool = False,
    ) -> Union[dict[str, Any], list[dict[str, Any]]]:
        if page < 1:
            raise InputError("Page index must be >= 1")  # noqa: TRY003
        if per_page < 1:
            raise InputError("Items per page must be >= 1")  # noqa: TRY003
        db = self.session
        conditions = ["project_id = ?"]
        params: list[Any] = [project_id]
        if not include_deleted:
            conditions.append("deleted = FALSE")
        if search:
            # SQLite's LIKE is already case-insensitive for ASCII.
            comparator = "ILIKE" if db.backend_type == BackendType.POSTGRESQL else "LIKE"
            conditions.append(f"name {comparator} ?")
            params.append(f"%{search}%")
        where_clause = " WHERE " + " AND ".join(conditions)
        list_sql = (
            f"SELECT * FROM prompt_studio_signatures{where_clause}"  # nosec B608
            " ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?"
        )

        def _read() -> tuple[int, list[dict[str, Any]]]:
            count_cursor = db._execute(f"SELECT COUNT(*) FROM prompt_studio_signatures{where_clause}", params)  # nosec B608
            total_row = count_cursor.fetchone()
            cursor = db._execute(list_sql, [*params, per_page, (page - 1) * per_page])
            rows = [db._row_to_dict(cursor, row) for row in cursor.fetchall() if row]
            return (int(total_row[0]) if total_row and total_row[0] is not None else 0), rows

        try:
            total, signatures = run_with_contention_retry(_read)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to list signatures: {exc}") from exc  # noqa: TRY003

        if not return_pagination:
            return signatures
        return {
            "signatures": signatures,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": (total + per_page - 1) // per_page,
            },
        }

    def update(self, signature_id: int, updates: dict[str, Any]) -> dict[str, Any]:
        db = self.session
        applied = {key: value for key, value in updates.items() if key in _UPDATABLE_FIELDS}
        if not applied:
            signature = self.get(signature_id, include_deleted=True)
            if signature is None:
                raise InputError(f"Signature {signature_id} not found or already deleted")  # noqa: TRY003
            return signature

        set_clauses = [f"{field} = ?" for field in applied] + ["updated_at = CURRENT_TIMESTAMP"]
        params: list[Any] = [
            json.dumps(value) if field in _JSON_FIELDS and value is not None else value
            for field, value in applied.items()
        ]
        params.append(signature_id)
        update_sql = (
            "UPDATE prompt_studio_signatures SET "  # nosec B608 - allowlisted columns
            + ", ".join(set_clauses)
            + " WHERE id = ? AND deleted = FALSE RETURNING *"
        )

        def _update() -> dict[str, Any]:
            with db._write_lock, db.transaction() as conn:
                cursor = db._cursor_exec(conn, update_sql, params)
                row = cursor.fetchone()
                if not row:
                    raise InputError(f"Signature {signature_id} not found or already deleted")  # noqa: TRY003
                return db._row_to_dict(cursor, row)

        try:
            signature = run_with_contention_retry(_update)
        except (*DB_ERRORS, ConflictError) as exc:
            if is_unique_violation(exc):
                raise ConflictError("Signature update conflicts with an existing record") from exc  # noqa: TRY003
            raise DatabaseError(f"Failed to update signature {signature_id}: {exc}") from exc  # noqa: TRY003
        db._log_sync_event("prompt_studio_signature", signature.get("uuid", ""), "update", applied)
        return signature

    def delete(self, signature_id: int, *, hard_delete: bool = False) -> bool:
        db = self.session
        if hard_delete:
            sql = "DELETE FROM prompt_studio_signatures WHERE id = ? RETURNING uuid"
        else:
            sql = (
                "UPDATE prompt_studio_signatures SET deleted = TRUE, deleted_at = CURRENT_TIMESTAMP"
                " WHERE id = ? AND deleted = FALSE RETURNING uuid"
            )

        def _delete() -> Optional[str]:
            with db._write_lock, db.transaction() as conn:
                row = db._cursor_exec(conn, sql, (signature_id,)).fetchone()
                if not row:
                    return None
                return row["uuid"] if isinstance(row, dict) else row[0]

        try:
            signature_uuid = run_with_contention_retry(_delete)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to delete signature {signature_id}: {exc}") from exc  # noqa: TRY003
        if signature_uuid is None:
            return False
        db._log_sync_event("prompt_studio_signature", signature_uuid, "delete", {"hard": hard_delete})
        return True
