"""Prompt Studio prompts: named, versioned prompt records within a project."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.prompt_studio_db.prompt_fields import _prepare_prompt_record_fields
from tldw_Server_API.app.core.DB_Management.prompt_studio_db.repositories._common import (
    DB_ERRORS,
    is_unique_violation,
    json_or_none,
)
from tldw_Server_API.app.core.DB_Management.Prompts_DB import ConflictError, DatabaseError, InputError
from tldw_Server_API.app.core.DB_Management.retry_policy import run_with_contention_retry


class PromptsRepository:
    def __init__(self, session: Any):
        self.session = session

    def create(
        self,
        project_id: int,
        name: str,
        *,
        signature_id: Optional[int] = None,
        version_number: int = 1,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        prompt_format: str = "legacy",
        prompt_schema_version: Optional[int] = None,
        prompt_definition: Optional[Any] = None,
        few_shot_examples: Optional[Any] = None,
        modules_config: Optional[Any] = None,
        parent_version_id: Optional[int] = None,
        change_description: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        fields = _prepare_prompt_record_fields(
            prompt_format=prompt_format,
            prompt_schema_version=prompt_schema_version,
            prompt_definition=prompt_definition,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        db = self.session
        prompt_uuid = str(uuid.uuid4())
        payload = (
            prompt_uuid,
            project_id,
            signature_id,
            version_number,
            name,
            fields["system_prompt"],
            fields["user_prompt"],
            fields["prompt_format"],
            fields["prompt_schema_version"],
            json_or_none(fields["prompt_definition"]),
            json_or_none(few_shot_examples),
            json_or_none(modules_config),
            parent_version_id,
            change_description,
            client_id or db.client_id,
        )
        insert_sql = """
            INSERT INTO prompt_studio_prompts (
                uuid, project_id, signature_id, version_number, name, system_prompt,
                user_prompt, prompt_format, prompt_schema_version, prompt_definition,
                few_shot_examples, modules_config, parent_version_id,
                change_description, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING *
        """

        def _insert() -> dict[str, Any]:
            with db._write_lock, db.transaction() as conn:
                cursor = db._cursor_exec(conn, insert_sql, payload)
                row = cursor.fetchone()
                return db._row_to_dict(cursor, row) if row else {}

        try:
            prompt = run_with_contention_retry(_insert)
        except (*DB_ERRORS, ConflictError) as exc:
            if is_unique_violation(exc):
                raise ConflictError(f"Prompt with name '{name}' already exists in project {project_id}") from exc  # noqa: TRY003
            raise DatabaseError(f"Failed to create prompt: {exc}") from exc  # noqa: TRY003
        db._log_sync_event(
            "prompt_studio_prompt",
            prompt_uuid,
            "create",
            {"project_id": project_id, "name": name, "version_number": version_number},
        )
        return prompt

    def _get_one(self, query: str, prompt_id: int) -> Optional[dict[str, Any]]:
        db = self.session

        def _read() -> Optional[dict[str, Any]]:
            cursor = db._execute(query, (prompt_id,))
            row = cursor.fetchone()
            return db._row_to_dict(cursor, row) if row else None

        try:
            return run_with_contention_retry(_read)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to fetch prompt {prompt_id}: {exc}") from exc  # noqa: TRY003

    def get(self, prompt_id: int, include_deleted: bool = False) -> Optional[dict[str, Any]]:
        query = "SELECT * FROM prompt_studio_prompts WHERE id = ?"
        if not include_deleted:
            query += " AND deleted = FALSE"
        return self._get_one(query, prompt_id)

    def get_with_project(self, prompt_id: int, *, include_deleted: bool = False) -> Optional[dict[str, Any]]:
        query = """
            SELECT p.*, proj.user_id AS project_user_id
            FROM prompt_studio_prompts p
            JOIN prompt_studio_projects proj ON p.project_id = proj.id
            WHERE p.id = ?
        """
        if not include_deleted:
            query += " AND p.deleted = FALSE"
        return self._get_one(query, prompt_id)

    def list(
        self,
        project_id: int,
        *,
        page: int = 1,
        per_page: int = 20,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        if page < 1:
            raise InputError("Page index must be >= 1")  # noqa: TRY003
        if per_page < 1:
            raise InputError("Items per page must be >= 1")  # noqa: TRY003
        db = self.session
        base_clause = "FROM prompt_studio_prompts WHERE project_id = ?"
        if not include_deleted:
            base_clause += " AND deleted = FALSE"
        # id DESC breaks updated_at ties, which SQLite's one-second timestamps make common.
        list_sql = f"SELECT * {base_clause} ORDER BY updated_at DESC, version_number DESC, id DESC LIMIT ? OFFSET ?"  # nosec B608

        def _read() -> tuple[int, list[dict[str, Any]]]:
            count_cursor = db._execute(f"SELECT COUNT(*) {base_clause}", (project_id,))  # nosec B608
            total_row = count_cursor.fetchone()
            cursor = db._execute(list_sql, (project_id, per_page, (page - 1) * per_page))
            rows = [db._row_to_dict(cursor, row) for row in cursor.fetchall() if row]
            return (int(total_row[0]) if total_row and total_row[0] is not None else 0), rows

        try:
            total, prompts = run_with_contention_retry(_read)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to list prompts for project {project_id}: {exc}") from exc  # noqa: TRY003
        return {
            "prompts": prompts,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": (total + per_page - 1) // per_page,
            },
        }

    def ensure_stub(
        self,
        *,
        prompt_id: int,
        project_id: int,
        name: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> None:
        """Ensure a placeholder prompt exists under an id chosen elsewhere."""
        if not prompt_id or not project_id:
            return
        db = self.session
        params = (prompt_id, project_id, name or f"Auto-Created Prompt {prompt_id}", client_id or db.client_id)

        def _insert() -> None:
            with db._write_lock, db.transaction() as conn:
                if db._cursor_exec(conn, "SELECT 1 FROM prompt_studio_prompts WHERE id = ?", (prompt_id,)).fetchone():
                    return
                db._cursor_exec(
                    conn,
                    "INSERT OR IGNORE INTO prompt_studio_prompts (id, uuid, project_id, version_number, name, client_id)"
                    " VALUES (?, ?, ?, 1, ?, ?)",
                    (params[0], str(uuid.uuid4()), *params[1:]),
                )
                if db.backend_type == BackendType.POSTGRESQL:
                    # An explicit id does not advance the serial sequence; without this the
                    # next create_prompt draws this id and fails as a false name conflict.
                    db._cursor_exec(
                        conn,
                        "SELECT setval(pg_get_serial_sequence('prompt_studio_prompts', 'id'),"
                        " GREATEST((SELECT MAX(id) FROM prompt_studio_prompts), 1))",
                    )

        try:
            run_with_contention_retry(_insert)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to create placeholder prompt {prompt_id}: {exc}") from exc  # noqa: TRY003
