"""Package-owned data-table metadata CRUD helpers for the Media DB runtime."""

from __future__ import annotations

import json
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.errors import InputError

_DATA_TABLES_UNSET = object()


def _serialize_column_hints(column_hints: str | dict[str, Any] | list[Any] | None) -> str | None:
    """Normalize column_hints into the stored JSON representation."""
    if column_hints is None:
        return None
    if isinstance(column_hints, str):
        try:
            json.loads(column_hints)
        except json.JSONDecodeError as exc:
            raise InputError(f"Invalid column_hints JSON: {exc}") from exc  # noqa: TRY003
        return column_hints
    return json.dumps(column_hints)


def _build_data_table_filters(
    self: Any,
    *,
    status: str | None = None,
    search: str | None = None,
    workspace_tag: str | None = None,
    include_deleted: bool = False,
    owner_user_id: int | str | None = None,
) -> tuple[list[str], list[Any]]:
    """Build shared WHERE conditions for metadata table reads."""
    owner_filter = self._resolve_data_tables_owner(owner_user_id)
    conditions: list[str] = []
    params: list[Any] = []
    if not include_deleted:
        conditions.append("deleted = 0")
    if owner_filter is not None:
        conditions.append("client_id = ?")
        params.append(owner_filter)
    if status:
        conditions.append("status = ?")
        params.append(str(status))
    if workspace_tag:
        conditions.append("workspace_tag = ?")
        params.append(str(workspace_tag))
    if search:
        like_op = "ILIKE" if self.backend_type == BackendType.POSTGRESQL else "LIKE"
        conditions.append(f"(name {like_op} ? OR description {like_op} ?)")
        pattern = f"%{search}%"
        params.extend([pattern, pattern])
    return conditions, params


def create_data_table(
    self: Any,
    *,
    name: str,
    prompt: str,
    description: str | None = None,
    workspace_tag: str | None = None,
    column_hints: str | dict[str, Any] | list[Any] | None = None,
    status: str = "queued",
    row_count: int = 0,
    generation_model: str | None = None,
    table_uuid: str | None = None,
    generation_job: dict[str, str] | None = None,
    owner_user_id: int | None = None,
) -> dict[str, Any]:
    """Create a data table metadata record and return the row."""
    if not name:
        raise InputError("name is required")  # noqa: TRY003
    if not prompt:
        raise InputError("prompt is required")  # noqa: TRY003

    now = self._get_current_utc_timestamp_str()
    table_uuid = table_uuid or self._generate_uuid()
    owner_client_id = self._resolve_data_tables_owner(owner_user_id) or str(self.client_id)
    column_hints_json = _serialize_column_hints(column_hints)

    with self.transaction() as conn:
        self._execute_with_connection(
            conn,
            """
            INSERT INTO data_tables (
                uuid, name, description, workspace_tag, prompt, column_hints_json, status,
                row_count, generation_model, last_error, generation_job_json,
                created_at, updated_at, last_modified, version, client_id, deleted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                table_uuid,
                name,
                description,
                workspace_tag,
                prompt,
                column_hints_json,
                status,
                int(row_count),
                generation_model,
                None,
                json.dumps(generation_job, sort_keys=True) if generation_job is not None else None,
                now,
                now,
                now,
                1,
                owner_client_id,
                False,
            ),
        )
        row = self._fetchone_with_connection(
            conn,
            "SELECT * FROM data_tables WHERE uuid = ?",
            (table_uuid,),
        )
    return row or {}


def get_data_table(
    self: Any,
    table_id: int,
    *,
    include_deleted: bool = False,
    owner_user_id: int | None = None,
) -> dict[str, Any] | None:
    """Fetch a data table by id."""
    owner_filter = self._resolve_data_tables_owner(owner_user_id)
    conditions = ["id = ?"]
    params: list[Any] = [int(table_id)]
    if not include_deleted:
        conditions.append("deleted = 0")
    if owner_filter is not None:
        conditions.append("client_id = ?")
        params.append(owner_filter)
    sql = "SELECT * FROM data_tables WHERE " + " AND ".join(conditions) + " LIMIT 1"  # nosec B608
    row = self.execute_query(sql, tuple(params)).fetchone()
    return dict(row) if row else None


def get_data_table_by_uuid(
    self: Any,
    table_uuid: str,
    *,
    include_deleted: bool = False,
    owner_user_id: int | None = None,
) -> dict[str, Any] | None:
    """Fetch a data table by uuid."""
    if not table_uuid:
        return None
    owner_filter = self._resolve_data_tables_owner(owner_user_id)
    conditions = ["uuid = ?"]
    params: list[Any] = [str(table_uuid)]
    if not include_deleted:
        conditions.append("deleted = 0")
    if owner_filter is not None:
        conditions.append("client_id = ?")
        params.append(owner_filter)
    sql = "SELECT * FROM data_tables WHERE " + " AND ".join(conditions) + " LIMIT 1"  # nosec B608
    row = self.execute_query(sql, tuple(params)).fetchone()
    return dict(row) if row else None


def list_data_tables(
    self: Any,
    *,
    status: str | None = None,
    search: str | None = None,
    workspace_tag: str | None = None,
    limit: int = 50,
    offset: int = 0,
    include_deleted: bool = False,
    owner_user_id: int | None = None,
) -> list[dict[str, Any]]:
    """List data tables with optional filters."""
    try:
        limit = int(limit)
        offset = int(offset)
    except (TypeError, ValueError):
        limit, offset = 50, 0
    limit = max(1, min(500, limit))
    offset = max(0, offset)

    conditions, params = _build_data_table_filters(
        self,
        status=status,
        search=search,
        workspace_tag=workspace_tag,
        include_deleted=include_deleted,
        owner_user_id=owner_user_id,
    )
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    sql = (
        "SELECT * FROM data_tables "  # nosec B608
        f"{where_clause} "
        "ORDER BY updated_at DESC, id DESC "
        "LIMIT ? OFFSET ?"
    )
    params.extend([limit, offset])
    rows = self.execute_query(sql, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def count_data_tables(
    self: Any,
    *,
    status: str | None = None,
    search: str | None = None,
    workspace_tag: str | None = None,
    include_deleted: bool = False,
    owner_user_id: int | None = None,
) -> int:
    """Count data tables matching optional filters."""
    conditions, params = _build_data_table_filters(
        self,
        status=status,
        search=search,
        workspace_tag=workspace_tag,
        include_deleted=include_deleted,
        owner_user_id=owner_user_id,
    )
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    sql = f"SELECT COUNT(*) as total FROM data_tables {where_clause}"  # nosec B608
    row = self.execute_query(sql, tuple(params)).fetchone()
    if not row:
        return 0
    total = row.get("total", 0) if isinstance(row, dict) else row[0]
    return int(total or 0)


def update_data_table(
    self: Any,
    table_id: int,
    *,
    owner_user_id: int | str | None = None,
    name: str | None = None,
    description: str | None = None,
    prompt: str | None = None,
    status: str | None = None,
    row_count: int | None = None,
    generation_model: str | None = None,
    last_error: Any = _DATA_TABLES_UNSET,
    column_hints: str | dict[str, Any] | list[Any] | None = None,
    generation_job: Any = _DATA_TABLES_UNSET,
    expected_generation_job: Any = _DATA_TABLES_UNSET,
    expected_version: int | None = None,
) -> dict[str, Any] | None:
    """Update data table metadata and return the updated row."""
    owner_filter = self._resolve_data_tables_owner(owner_user_id)
    update_parts: list[str] = []
    params: list[Any] = []

    if name is not None:
        update_parts.append("name = ?")
        params.append(name)
    if description is not None:
        update_parts.append("description = ?")
        params.append(description)
    if prompt is not None:
        update_parts.append("prompt = ?")
        params.append(prompt)
    if status is not None:
        update_parts.append("status = ?")
        params.append(status)
    if row_count is not None:
        update_parts.append("row_count = ?")
        params.append(int(row_count))
    if generation_model is not None:
        update_parts.append("generation_model = ?")
        params.append(generation_model)
    if last_error is not _DATA_TABLES_UNSET:
        update_parts.append("last_error = ?")
        params.append(last_error)
    if column_hints is not None:
        update_parts.append("column_hints_json = ?")
        params.append(_serialize_column_hints(column_hints))

    if generation_job is not _DATA_TABLES_UNSET:
        update_parts.append("generation_job_json = ?")
        params.append(json.dumps(generation_job, sort_keys=True) if generation_job is not None else None)

    if not update_parts:
        return self.get_data_table(int(table_id), include_deleted=True, owner_user_id=owner_user_id)

    now = self._get_current_utc_timestamp_str()
    update_parts.append("updated_at = ?")
    params.append(now)
    update_parts.append("last_modified = ?")
    params.append(now)
    update_parts.append("version = version + 1")

    params.append(int(table_id))
    sql = "UPDATE data_tables SET " + ", ".join(update_parts) + " WHERE id = ?"  # nosec B608
    if owner_filter is not None:
        sql += " AND client_id = ?"
        params.append(owner_filter)
    # Keep the returned version under the UPDATE's row lock.
    with self.transaction() as conn:
        if expected_generation_job is not _DATA_TABLES_UNSET or expected_version is not None:
            locked_sql = "SELECT * FROM data_tables WHERE id = ? AND client_id = ? AND deleted = 0"
            if self.backend_type == BackendType.POSTGRESQL:
                locked_sql += " FOR UPDATE"
            current = self._fetchone_with_connection(conn, locked_sql, (int(table_id), owner_filter or str(self.client_id)))
            if not current:
                return None
            if expected_generation_job is not _DATA_TABLES_UNSET and current.get("generation_job_json") != expected_generation_job:
                return None
            if expected_version is not None and current["version"] != expected_version:
                return None
        self._execute_with_connection(conn, sql, tuple(params))
        read_sql = "SELECT * FROM data_tables WHERE id = ?"
        read_params: list[Any] = [int(table_id)]
        if owner_filter is not None:
            read_sql += " AND client_id = ?"
            read_params.append(owner_filter)
        return self._fetchone_with_connection(conn, read_sql, tuple(read_params))


def soft_delete_data_table(self: Any, table_id: int, owner_user_id: int | None = None) -> bool:
    """Soft delete a data table and its related rows."""
    now = self._get_current_utc_timestamp_str()
    owner_filter = self._resolve_data_tables_owner(owner_user_id)
    with self.transaction() as conn:
        params: list[Any] = [now, now, int(table_id)]
        where_clause = "WHERE id = ? AND deleted = 0"
        if owner_filter is not None:
            where_clause += " AND client_id = ?"
            params.append(owner_filter)
        cur = self._execute_with_connection(
            conn,
            """
            UPDATE data_tables
            SET deleted = 1,
                updated_at = ?,
                last_modified = ?,
                version = version + 1
            {where_clause}
            """.format_map(locals()),  # nosec B608
            tuple(params),
        )
        updated = int(getattr(cur, "rowcount", 0) or 0)
        if updated:
            self._soft_delete_data_table_children(
                conn,
                int(table_id),
                now,
                owner_user_id=owner_user_id,
            )
    return bool(updated)
