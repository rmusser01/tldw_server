"""Package-owned data-table content replacement helper for the Media DB runtime."""

from __future__ import annotations

import hashlib
from typing import Any

from tldw_Server_API.app.core.DB_Management.media_db.errors import InputError


def replace_data_table_contents(
    self: Any,
    table_id: int,
    *,
    owner_user_id: int | str,
    columns: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> tuple[int, int]:
    """Replace table columns and rows, returning counts inserted."""
    owner_value = str(owner_user_id).strip()
    if not owner_value:
        raise InputError("owner_user_id is required")  # noqa: TRY003
    if not columns:
        raise InputError("columns are required")  # noqa: TRY003
    if rows is None:
        raise InputError("rows are required")  # noqa: TRY003

    table_id_int = int(table_id)
    write_client_id = self._resolve_data_table_write_client_id(
        table_id_int,
        owner_user_id=owner_value,
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
                table_id_int,
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
                table_id_int,
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

    with self.transaction() as conn:
        actual_owner = self._get_data_table_owner_client_id(conn, table_id_int)
        if not actual_owner:
            raise InputError("data_table_not_found")  # noqa: TRY003
        if actual_owner != owner_value:
            raise InputError("data_table_owner_mismatch")  # noqa: TRY003

        self._execute_with_connection(
            conn,
            """
            UPDATE data_table_columns
            SET deleted = 1,
                last_modified = ?,
                version = version + 1
            WHERE table_id = ? AND deleted = 0
            """,
            (now, table_id_int),
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
            (now, table_id_int),
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

    return len(column_rows), len(row_rows)


__all__ = ["replace_data_table_contents"]
