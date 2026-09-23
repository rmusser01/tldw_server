"""Prompt Studio projects: the owner-scoped container for everything else."""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.prompt_studio_db.repositories._common import (
    DB_ERRORS,
    is_unique_violation,
    json_or_none,
)
from tldw_Server_API.app.core.DB_Management.Prompts_DB import ConflictError, DatabaseError, InputError
from tldw_Server_API.app.core.DB_Management.retry_policy import run_with_contention_retry

MIN_PROJECT_NAME_LENGTH = 1
MAX_PROJECT_NAME_LENGTH = 255
# Other keys in `updates` are ignored, as before; column names are interpolated.
_UPDATABLE_FIELDS = frozenset({"name", "description", "status", "metadata"})
_PROJECT_COLUMNS = (
    "id, uuid, name, description, user_id, client_id, status, deleted, deleted_at, "
    "created_at, updated_at, last_modified, version, metadata"
)


def _validated_name(name: Any) -> str:
    """InputError is a ValueError, which is what callers of the backend variant caught."""
    if not name or not isinstance(name, str):
        raise InputError("Project name must be a non-empty string")  # noqa: TRY003
    name = name.strip()
    if len(name) < MIN_PROJECT_NAME_LENGTH:
        raise InputError("Project name cannot be empty")  # noqa: TRY003
    if len(name) > MAX_PROJECT_NAME_LENGTH:
        raise InputError(f"Project name cannot exceed {MAX_PROJECT_NAME_LENGTH} characters")  # noqa: TRY003
    return name


class ProjectsRepository:
    def __init__(self, session: Any):
        self.session = session

    def create(
        self,
        name: str,
        description: Optional[str] = None,
        status: str = "draft",
        metadata: Optional[dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        name = _validated_name(name)
        db = self.session
        project_uuid = str(uuid.uuid4())
        payload = (
            project_uuid,
            name,
            description,
            user_id or db.client_id,
            db.client_id,
            status,
            json_or_none(metadata),
        )
        insert_sql = f"""
            INSERT INTO prompt_studio_projects
            (uuid, name, description, user_id, client_id, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            RETURNING {_PROJECT_COLUMNS}
        """  # nosec B608 - constant column list

        def _insert() -> dict[str, Any]:
            with db._write_lock, db.transaction() as conn:
                cursor = db._cursor_exec(conn, insert_sql, payload)
                row = cursor.fetchone()
                return db._row_to_dict(cursor, row) if row else {}

        try:
            project = run_with_contention_retry(_insert)
        except (*DB_ERRORS, ConflictError) as exc:
            if is_unique_violation(exc):
                raise ConflictError(f"Project with name '{name}' already exists for this user") from exc  # noqa: TRY003
            raise DatabaseError(f"Failed to create project: {exc}") from exc  # noqa: TRY003
        db._log_sync_event(
            "prompt_studio_project",
            project_uuid,
            "create",
            {"name": name, "description": description, "status": status},
        )
        return project

    def get(self, project_id: int, include_deleted: bool = False) -> Optional[dict[str, Any]]:
        db = self.session
        query = f"SELECT {_PROJECT_COLUMNS} FROM prompt_studio_projects WHERE id = ?"  # nosec B608
        if not include_deleted:
            query += " AND deleted = FALSE"

        def _read() -> Optional[dict[str, Any]]:
            cursor = db._execute(query, (project_id,))
            row = cursor.fetchone()
            return db._row_to_dict(cursor, row) if row else None

        try:
            return run_with_contention_retry(_read)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to get project {project_id}: {exc}") from exc  # noqa: TRY003

    def list(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        include_deleted: bool = False,
        page: int = 1,
        per_page: int = 20,
        search: Optional[str] = None,
    ) -> dict[str, Any]:
        db = self.session
        conditions: list[str] = []
        params: list[Any] = []
        if not include_deleted:
            conditions.append("deleted = FALSE")
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if search:
            comparator = "ILIKE" if db.backend_type == BackendType.POSTGRESQL else "LIKE"
            conditions.append(f"(name {comparator} ? OR description {comparator} ?)")
            params.extend([f"%{search}%"] * 2)
        where_sql = " WHERE " + " AND ".join(conditions) if conditions else ""
        list_sql = f"""
            SELECT p.*,
                   (SELECT COUNT(*) FROM prompt_studio_prompts
                    WHERE project_id = p.id AND deleted = FALSE) AS prompt_count,
                   (SELECT COUNT(*) FROM prompt_studio_test_cases
                    WHERE project_id = p.id AND deleted = FALSE) AS test_case_count
            FROM prompt_studio_projects p
            {where_sql}
            ORDER BY p.updated_at DESC
            LIMIT ? OFFSET ?
        """  # nosec B608

        def _read() -> tuple[int, list[dict[str, Any]]]:
            count_cursor = db._execute(f"SELECT COUNT(*) FROM prompt_studio_projects{where_sql}", params)  # nosec B608
            total_row = count_cursor.fetchone()
            cursor = db._execute(list_sql, [*params, per_page, (page - 1) * per_page])
            rows = [db._row_to_dict(cursor, row) for row in cursor.fetchall() if row]
            return (int(total_row[0]) if total_row and total_row[0] is not None else 0), rows

        try:
            total, projects = run_with_contention_retry(_read)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to list projects: {exc}") from exc  # noqa: TRY003
        return {
            "projects": projects,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": (total + per_page - 1) // per_page if per_page else 0,
            },
        }

    def update(self, project_id: int, updates: dict[str, Any]) -> dict[str, Any]:
        applied = {key: value for key, value in updates.items() if key in _UPDATABLE_FIELDS}
        if "name" in applied:
            applied["name"] = _validated_name(applied["name"])
        if not applied:
            project = self.get(project_id, include_deleted=True)
            if project is None:
                raise InputError(f"Project {project_id} not found or already deleted")  # noqa: TRY003
            return project

        db = self.session
        set_clauses = [f"{field} = ?" for field in applied] + ["updated_at = CURRENT_TIMESTAMP"]
        params: list[Any] = [
            json.dumps(value) if field == "metadata" and value is not None else value
            for field, value in applied.items()
        ]
        params.append(project_id)
        update_sql = (
            "UPDATE prompt_studio_projects SET "  # nosec B608 - allowlisted columns
            + ", ".join(set_clauses)
            + f" WHERE id = ? AND deleted = FALSE RETURNING {_PROJECT_COLUMNS}"
        )

        def _update() -> dict[str, Any]:
            with db._write_lock, db.transaction() as conn:
                cursor = db._cursor_exec(conn, update_sql, params)
                row = cursor.fetchone()
                if not row:
                    raise InputError(f"Project {project_id} not found or already deleted")  # noqa: TRY003
                return db._row_to_dict(cursor, row)

        try:
            project = run_with_contention_retry(_update)
        except (*DB_ERRORS, ConflictError) as exc:
            if is_unique_violation(exc):
                raise ConflictError(f"Project with name '{applied.get('name')}' already exists for this user") from exc  # noqa: TRY003
            raise DatabaseError(f"Failed to update project {project_id}: {exc}") from exc  # noqa: TRY003
        db._log_sync_event("prompt_studio_project", project.get("uuid", ""), "update", applied)
        return project

    def delete(self, project_id: int, hard_delete: bool = False) -> bool:
        db = self.session
        if hard_delete:
            # Related rows go with it through ON DELETE CASCADE.
            sql = "DELETE FROM prompt_studio_projects WHERE id = ? RETURNING uuid"
        else:
            sql = (
                "UPDATE prompt_studio_projects SET deleted = TRUE, deleted_at = CURRENT_TIMESTAMP"
                " WHERE id = ? AND deleted = FALSE RETURNING uuid"
            )

        def _delete() -> Optional[str]:
            with db._write_lock, db.transaction() as conn:
                row = db._cursor_exec(conn, sql, (project_id,)).fetchone()
                if not row:
                    return None
                return row["uuid"] if isinstance(row, dict) else row[0]

        try:
            project_uuid = run_with_contention_retry(_delete)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to delete project {project_id}: {exc}") from exc  # noqa: TRY003
        if project_uuid is None:
            return False
        db._log_sync_event("prompt_studio_project", project_uuid, "delete", {"hard": hard_delete})
        return True
