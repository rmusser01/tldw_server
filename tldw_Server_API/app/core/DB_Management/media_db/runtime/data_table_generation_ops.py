"""Package-owned data-table generation persistence helper for the Media DB runtime."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.errors import ConflictError, InputError


def persist_data_table_generation(
    self: Any,
    table_id: int,
    *,
    columns: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    sources: list[dict[str, Any]] | None = None,
    status: str = "ready",
    row_count: int | None = None,
    generation_model: str | None = None,
    last_error: Any = None,
    owner_user_id: int | str | None = None,
    expected_version: int | None = None,
    expected_uuid: str | None = None,
    expected_generation_job: str | None = None,
) -> dict[str, Any] | None:
    """Persist generated table data and update table metadata."""
    owner_value = None
    if owner_user_id is not None:
        owner_value = str(owner_user_id).strip()
        if not owner_value:
            raise InputError("owner_user_id is required")  # noqa: TRY003
    if not columns:
        raise InputError("columns are required")  # noqa: TRY003
    if rows is None:
        raise InputError("rows are required")  # noqa: TRY003

    write_client_id = self._resolve_data_table_write_client_id(
        int(table_id),
        owner_user_id=owner_user_id,
    )
    now = self._get_current_utc_timestamp_str()

    column_rows: list[tuple[Any, ...]] = []
    for idx, column in enumerate(columns):
        name = column.get("name")
        col_type = column.get("type")
        if not name or not col_type:
            raise InputError("column name and type are required")  # noqa: TRY003
        column_id = column.get("column_id") or column.get("id") or self._generate_uuid()
        position = column.get("position", idx)
        column_rows.append(
            (
                int(table_id),
                str(column_id),
                str(name),
                str(col_type),
                column.get("description"),
                column.get("format"),
                int(position),
                now,
                now,
                1,
                write_client_id,
                False,
                column.get("prev_version"),
                column.get("merge_parent_uuid"),
            )
        )
    column_ids = {str(row[1]) for row in column_rows}

    row_rows: list[tuple[Any, ...]] = []
    for idx, row in enumerate(rows):
        row_json = row.get("row_json", row.get("data"))
        row_json = self._normalize_data_table_row_json(
            row_json,
            column_ids=column_ids,
            validate_keys=True,
        )
        row_id = row.get("row_id") or row.get("id") or self._generate_uuid()
        row_index = row.get("row_index", idx)
        row_hash = row.get("row_hash")
        if row_hash is None:
            row_hash = hashlib.sha256(row_json.encode("utf-8")).hexdigest()
        row_rows.append(
            (
                int(table_id),
                str(row_id),
                int(row_index),
                row_json,
                row_hash,
                now,
                now,
                1,
                write_client_id,
                False,
                row.get("prev_version"),
                row.get("merge_parent_uuid"),
            )
        )

    source_rows: list[tuple[Any, ...]] = []
    if sources is not None:
        for src in sources:
            source_type = src.get("source_type")
            source_id = src.get("source_id")
            if not source_type or source_id is None:
                raise InputError("source_type and source_id are required")  # noqa: TRY003
            snapshot = src.get("snapshot_json")
            if snapshot is not None and not isinstance(snapshot, str):
                snapshot = json.dumps(snapshot)
            retrieval = src.get("retrieval_params_json")
            if retrieval is not None and not isinstance(retrieval, str):
                retrieval = json.dumps(retrieval)
            source_rows.append(
                (
                    int(table_id),
                    str(source_type),
                    str(source_id),
                    src.get("title"),
                    snapshot,
                    retrieval,
                    now,
                    now,
                    1,
                    write_client_id,
                    False,
                    src.get("prev_version"),
                    src.get("merge_parent_uuid"),
                )
            )

    update_parts = ["status = ?", "row_count = ?", "last_error = ?"]
    params: list[Any] = [
        status,
        int(row_count if row_count is not None else len(rows)),
        last_error,
    ]
    update_parts.append("updated_at = ?")
    params.append(now)
    update_parts.append("last_modified = ?")
    params.append(now)
    update_parts.append("version = version + 1")
    if generation_model is not None:
        update_parts.append("generation_model = ?")
        params.append(generation_model)
    params.append(int(table_id))

    with self.transaction() as conn:
        if owner_value is not None:
            actual_owner = self._get_data_table_owner_client_id(conn, int(table_id))
            if not actual_owner:
                raise InputError("data_table_not_found")  # noqa: TRY003
            if actual_owner != owner_value:
                raise InputError("data_table_owner_mismatch")  # noqa: TRY003
        if expected_version is not None or self.backend_type == BackendType.POSTGRESQL:
            # Editors and publishers take the parent lock before any child writes.
            query = "SELECT * FROM data_tables WHERE id = ? AND client_id = ? AND deleted = 0"
            if self.backend_type == BackendType.POSTGRESQL:
                query += " FOR UPDATE"
            current = self._fetchone_with_connection(conn, query, (int(table_id), write_client_id))
            if not current:
                raise ConflictError("Data table identity changed")
        if expected_version is not None:
            if current["uuid"] != expected_uuid:
                raise ConflictError("Data table identity changed")
            # Both sentinels were inserted atomically; tombstones survive ordinary edits.
            applied_column = self._fetchone_with_connection(
                conn, "SELECT 1 FROM data_table_columns WHERE table_id = ? AND client_id = ? AND column_id = ?",
                (int(table_id), owner_value, column_rows[0][1]),
            )
            applied_row = self._fetchone_with_connection(
                conn, "SELECT 1 FROM data_table_rows WHERE table_id = ? AND client_id = ? AND row_id = ?",
                (int(table_id), owner_value, row_rows[0][1]),
            ) if row_rows else None
            if applied_column and applied_row:
                return current
            if expected_generation_job is not None and current.get("generation_job_json") != expected_generation_job:
                raise ConflictError("Data table generation was superseded")
            if applied_column or applied_row or int(current["version"]) != expected_version:
                raise ConflictError("Saved data table changed during generation")
        self._execute_with_connection(
            conn,
            """
            UPDATE data_table_columns
            SET deleted = 1,
                last_modified = ?,
                version = version + 1
            WHERE table_id = ? AND deleted = 0
            """,
            (now, int(table_id)),
        )
        self._execute_with_connection(
            conn,
            """
            UPDATE data_table_rows
            SET deleted = 1,
                last_modified = ?,
                version = version + 1
            WHERE table_id = ? AND deleted = 0
            """,
            (now, int(table_id)),
        )
        if sources is not None:
            self._execute_with_connection(
                conn,
                """
                UPDATE data_table_sources
                SET deleted = 1,
                    last_modified = ?,
                    version = version + 1
                WHERE table_id = ? AND deleted = 0
                """,
                (now, int(table_id)),
            )
        self.execute_many(
            """
            INSERT INTO data_table_columns (
                table_id, column_id, name, type, description, format, position,
                created_at, last_modified, version, client_id, deleted, prev_version, merge_parent_uuid
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            column_rows,
            commit=False,
            connection=conn,
        )
        self.execute_many(
            """
            INSERT INTO data_table_rows (
                table_id, row_id, row_index, row_json, row_hash,
                created_at, last_modified, version, client_id, deleted, prev_version, merge_parent_uuid
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row_rows,
            commit=False,
            connection=conn,
        )
        if source_rows:
            self.execute_many(
                """
                INSERT INTO data_table_sources (
                    table_id, source_type, source_id, title, snapshot_json, retrieval_params_json,
                    created_at, last_modified, version, client_id, deleted, prev_version, merge_parent_uuid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                source_rows,
                commit=False,
                connection=conn,
            )
        sql = "UPDATE data_tables SET " + ", ".join(update_parts) + " WHERE id = ?"  # nosec B608
        self._execute_with_connection(conn, sql, tuple(params))

    return self.get_data_table(int(table_id), include_deleted=True)
