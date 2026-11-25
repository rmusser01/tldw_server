"""Response serialization shared by native and Workspace Data Tables routes."""

from __future__ import annotations

import json
from typing import Any

from tldw_Server_API.app.api.v1.schemas.data_tables_schemas import DataTableSummary


def parse_json_value(raw: str | None) -> Any:
    """Parse a JSON string when possible, otherwise return the original value."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def table_summary_from_row(
    row: dict[str, Any],
    *,
    column_count: int | None = None,
    source_count: int | None = None,
) -> DataTableSummary:
    """Build a DataTableSummary response from a database row."""
    return DataTableSummary(
        uuid=str(row.get("uuid") or ""),
        name=str(row.get("name") or ""),
        description=row.get("description"),
        workspace_tag=row.get("workspace_tag"),
        prompt=str(row.get("prompt") or ""),
        column_hints=parse_json_value(row.get("column_hints_json")),
        status=str(row.get("status") or ""),
        row_count=int(row.get("row_count") or 0),
        column_count=column_count
        if column_count is not None
        else row.get("column_count"),
        generation_model=row.get("generation_model"),
        last_error=row.get("last_error"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        last_modified=row.get("last_modified"),
        version=int(row.get("version")) if row.get("version") is not None else None,
        source_count=source_count
        if source_count is not None
        else row.get("source_count"),
    )
