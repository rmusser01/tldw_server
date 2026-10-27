"""Package-owned data-table child-content helpers for the Media DB runtime."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from tldw_Server_API.app.core.DB_Management.media_db.errors import InputError


def _owner_has_table_access(self: Any, table_id: int, owner_filter: str | None) -> bool:
    """Return whether the explicit owner filter still owns the table."""
    if owner_filter is None:
        return True
    owned = self.execute_query(
        "SELECT 1 FROM data_tables WHERE id = ? AND client_id = ? LIMIT 1",
        (int(table_id), owner_filter),
    ).fetchone()
    return bool(owned)


def _soft_delete_child_records(
    self: Any,
    table_name: str,
    table_id: int,
    owner_user_id: int | None = None,
) -> int:
    """Soft delete a child table's rows for the target data table."""
    now = self._get_current_utc_timestamp_str()
    owner_filter = self._resolve_data_tables_owner(owner_user_id)
    params: list[Any] = [now, int(table_id)]
    where_clause = "WHERE table_id = ? AND deleted = 0"
    if owner_filter is not None:
        where_clause += " AND client_id = ?"
        params.append(owner_filter)
    cur = self.execute_query(
        """
        UPDATE {table_name}
        SET deleted = 1,
            last_modified = ?,
            version = version + 1
        {where_clause}
        """.format_map(locals()),  # nosec B608
        tuple(params),
        commit=True,
    )
    return int(getattr(cur, "rowcount", 0) or 0)


def get_data_table_counts(
    self: Any,
    table_ids: list[int],
    *,
    owner_user_id: int | None = None,
) -> dict[int, dict[str, int]]:
    """Return column/source counts for the provided table ids."""
    ids = [int(table_id) for table_id in table_ids if table_id is not None]
    if not ids:
        return {}
    owner_filter = self._resolve_data_tables_owner(owner_user_id)
    placeholders = ",".join(["?"] * len(ids))
    owner_clause = ""
    params: list[Any] = list(ids)
    if owner_filter is not None:
        owner_clause = " AND dt.client_id = ?"
        params.append(owner_filter)
    columns_sql = (
        "SELECT c.table_id, COUNT(*) as count FROM data_table_columns c "  # nosec B608
        "INNER JOIN data_tables dt ON dt.id = c.table_id "
        f"WHERE c.deleted = 0 AND dt.deleted = 0 AND c.table_id IN ({placeholders}){owner_clause} "
        "GROUP BY c.table_id"
    )
    sources_sql = (
        "SELECT s.table_id, COUNT(*) as count FROM data_table_sources s "  # nosec B608
        "INNER JOIN data_tables dt ON dt.id = s.table_id "
        f"WHERE s.deleted = 0 AND dt.deleted = 0 AND s.table_id IN ({placeholders}){owner_clause} "
        "GROUP BY s.table_id"
    )
    columns_rows = self.execute_query(columns_sql, tuple(params)).fetchall()
    sources_rows = self.execute_query(sources_sql, tuple(params)).fetchall()

    counts: dict[int, dict[str, int]] = {
        table_id: {"column_count": 0, "source_count": 0} for table_id in ids
    }
    for row in columns_rows:
        table_id = int(row.get("table_id") if isinstance(row, dict) else row[0])
        count = int(row.get("count") if isinstance(row, dict) else row[1])
        counts.setdefault(table_id, {})["column_count"] = count
    for row in sources_rows:
        table_id = int(row.get("table_id") if isinstance(row, dict) else row[0])
        count = int(row.get("count") if isinstance(row, dict) else row[1])
        counts.setdefault(table_id, {})["source_count"] = count
    return counts


def insert_data_table_columns(
    self: Any,
    table_id: int,
    columns: list[dict[str, Any]],
    *,
    owner_user_id: int | str | None = None,
) -> int:
    """Insert data table columns and return count inserted."""
    if not columns:
        return 0
    owner_filter = self._resolve_data_tables_owner(owner_user_id)
    if not _owner_has_table_access(self, int(table_id), owner_filter):
        return 0
    write_client_id = owner_filter or self._resolve_data_table_write_client_id(
        int(table_id),
        owner_user_id=owner_user_id,
    )
    now = self._get_current_utc_timestamp_str()
    rows: list[tuple[Any, ...]] = []
    for idx, column in enumerate(columns):
        name = column.get("name")
        col_type = column.get("type")
        if not name or not col_type:
            raise InputError("column name and type are required")  # noqa: TRY003
        column_id = column.get("column_id") or column.get("id") or self._generate_uuid()
        position = column.get("position", idx)
        rows.append(
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
    with self.transaction() as conn:
        self.execute_many(
            """
            INSERT INTO data_table_columns (
                table_id, column_id, name, type, description, format, position,
                created_at, last_modified, version, client_id, deleted, prev_version, merge_parent_uuid
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
            commit=False,
            connection=conn,
        )
    return len(rows)


def list_data_table_columns(
    self: Any,
    table_id: int,
    *,
    include_deleted: bool = False,
    owner_user_id: int | None = None,
) -> list[dict[str, Any]]:
    """List columns for a data table."""
    owner_filter = self._resolve_data_tables_owner(owner_user_id)
    conditions = ["table_id = ?"]
    params: list[Any] = [int(table_id)]
    if not include_deleted:
        conditions.append("deleted = 0")
    if owner_filter is not None:
        conditions.append("client_id = ?")
        params.append(owner_filter)
    sql = (
        "SELECT * FROM data_table_columns WHERE "  # nosec B608
        + " AND ".join(conditions)
        + " ORDER BY position ASC, id ASC"
    )
    rows = self.execute_query(sql, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def soft_delete_data_table_columns(
    self: Any,
    table_id: int,
    owner_user_id: int | None = None,
) -> int:
    """Soft delete columns for a data table."""
    return _soft_delete_child_records(
        self,
        "data_table_columns",
        table_id,
        owner_user_id=owner_user_id,
    )


def insert_data_table_rows(
    self: Any,
    table_id: int,
    rows: list[dict[str, Any]],
    *,
    validate_keys: bool = True,
    owner_user_id: int | str | None = None,
) -> int:
    """Insert data table rows and return count inserted."""
    if not rows:
        return 0
    owner_filter = self._resolve_data_tables_owner(owner_user_id)
    if not _owner_has_table_access(self, int(table_id), owner_filter):
        return 0
    write_client_id = owner_filter or self._resolve_data_table_write_client_id(
        int(table_id),
        owner_user_id=owner_user_id,
    )
    column_ids: set[str] | None = None
    if validate_keys:
        columns = self.list_data_table_columns(int(table_id), owner_user_id=owner_user_id)
        if not columns:
            raise InputError("data_table_columns_required")
        column_ids = {str(col.get("column_id") or "") for col in columns}
        if "" in column_ids:
            column_ids.discard("")
    now = self._get_current_utc_timestamp_str()
    insert_rows: list[tuple[Any, ...]] = []
    for idx, row in enumerate(rows):
        row_json = row.get("row_json", row.get("data"))
        row_json = self._normalize_data_table_row_json(
            row_json,
            column_ids=column_ids,
            validate_keys=validate_keys,
        )
        row_id = row.get("row_id") or row.get("id") or self._generate_uuid()
        row_index = row.get("row_index", idx)
        row_hash = row.get("row_hash")
        if row_hash is None:
            row_hash = hashlib.sha256(row_json.encode("utf-8")).hexdigest()
        insert_rows.append(
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
    with self.transaction() as conn:
        self.execute_many(
            """
            INSERT INTO data_table_rows (
                table_id, row_id, row_index, row_json, row_hash,
                created_at, last_modified, version, client_id, deleted, prev_version, merge_parent_uuid
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            insert_rows,
            commit=False,
            connection=conn,
        )
    return len(insert_rows)


def list_data_table_rows(
    self: Any,
    table_id: int,
    *,
    limit: int = 200,
    offset: int = 0,
    include_deleted: bool = False,
    owner_user_id: int | None = None,
) -> list[dict[str, Any]]:
    """List rows for a data table."""
    try:
        limit = int(limit)
        offset = int(offset)
    except (TypeError, ValueError):
        limit, offset = 200, 0
    limit = max(1, min(2000, limit))
    offset = max(0, offset)
    owner_filter = self._resolve_data_tables_owner(owner_user_id)
    conditions = ["table_id = ?"]
    params: list[Any] = [int(table_id)]
    if not include_deleted:
        conditions.append("deleted = 0")
    if owner_filter is not None:
        conditions.append("client_id = ?")
        params.append(owner_filter)
    sql = (
        "SELECT * FROM data_table_rows WHERE "  # nosec B608
        + " AND ".join(conditions)
        + " ORDER BY row_index ASC, id ASC LIMIT ? OFFSET ?"
    )
    params.extend([limit, offset])
    rows = self.execute_query(sql, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def soft_delete_data_table_rows(
    self: Any,
    table_id: int,
    owner_user_id: int | None = None,
) -> int:
    """Soft delete rows for a data table."""
    return _soft_delete_child_records(
        self,
        "data_table_rows",
        table_id,
        owner_user_id=owner_user_id,
    )


def insert_data_table_sources(
    self: Any,
    table_id: int,
    sources: list[dict[str, Any]],
    *,
    owner_user_id: int | str | None = None,
) -> int:
    """Insert sources for a data table and return count inserted."""
    if not sources:
        return 0
    owner_filter = self._resolve_data_tables_owner(owner_user_id)
    if not _owner_has_table_access(self, int(table_id), owner_filter):
        return 0
    write_client_id = owner_filter or self._resolve_data_table_write_client_id(
        int(table_id),
        owner_user_id=owner_user_id,
    )
    now = self._get_current_utc_timestamp_str()
    rows: list[tuple[Any, ...]] = []
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
        rows.append(
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
    with self.transaction() as conn:
        self.execute_many(
            """
            INSERT INTO data_table_sources (
                table_id, source_type, source_id, title, snapshot_json, retrieval_params_json,
                created_at, last_modified, version, client_id, deleted, prev_version, merge_parent_uuid
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
            commit=False,
            connection=conn,
        )
    return len(rows)


def list_data_table_sources(
    self: Any,
    table_id: int,
    *,
    include_deleted: bool = False,
    owner_user_id: int | None = None,
) -> list[dict[str, Any]]:
    """List sources for a data table."""
    owner_filter = self._resolve_data_tables_owner(owner_user_id)
    conditions = ["table_id = ?"]
    params: list[Any] = [int(table_id)]
    if not include_deleted:
        conditions.append("deleted = 0")
    if owner_filter is not None:
        conditions.append("client_id = ?")
        params.append(owner_filter)
    sql = (
        "SELECT * FROM data_table_sources WHERE "  # nosec B608
        + " AND ".join(conditions)
        + " ORDER BY id ASC"
    )
    rows = self.execute_query(sql, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def soft_delete_data_table_sources(
    self: Any,
    table_id: int,
    owner_user_id: int | None = None,
) -> int:
    """Soft delete sources for a data table."""
    return _soft_delete_child_records(
        self,
        "data_table_sources",
        table_id,
        owner_user_id=owner_user_id,
    )
