from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger

_TOOL_CATALOG_LOOKUP_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _coerce_catalog_id(catalog_id: Any) -> int | None:
    """Return a numeric catalog id when the request supplied one."""
    if catalog_id is None:
        return None
    try:
        return int(catalog_id)
    except (TypeError, ValueError):
        return None


def _row_value(row: Any, key: str, index: int) -> Any:
    """Read a value from dict-like, record-like, or tuple-like DB rows."""
    if row is None:
        return None
    if isinstance(row, Mapping):
        return row.get(key)

    keys = getattr(row, "keys", None)
    if callable(keys):
        try:
            return row[key]
        except (IndexError, KeyError, TypeError):
            pass

    if isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
        return row[index] if len(row) > index else None

    try:
        return row[index]
    except (IndexError, KeyError, TypeError):
        return None


async def _resolve_tool_catalog_name(
    db: Any,
    *,
    name: str,
    metadata: Mapping[str, Any],
) -> int | None:
    """Resolve a catalog name using team, org, then global precedence."""
    team_id = metadata.get("team_id")
    org_id = metadata.get("org_id")
    row = None
    try:
        if team_id is not None:
            row = await db.fetchone(
                "SELECT id FROM tool_catalogs WHERE name = ? AND team_id = ?",
                name,
                team_id,
            )
        if row is None and org_id is not None:
            row = await db.fetchone(
                "SELECT id FROM tool_catalogs WHERE name = ? AND org_id = ? AND team_id IS NULL",
                name,
                org_id,
            )
        if row is None:
            row = await db.fetchone(
                "SELECT id FROM tool_catalogs WHERE name = ? AND org_id IS NULL AND team_id IS NULL",
                name,
            )
        value = _row_value(row, "id", 0)
        return int(value) if value is not None else None
    except _TOOL_CATALOG_LOOKUP_EXCEPTIONS as exc:
        logger.debug("MCP catalog lookup failed: {}", exc.__class__.__name__)
        return None


async def _resolve_tool_catalog_entries(
    db: Any,
    *,
    catalog_id: int,
    strict: bool,
) -> set[str] | None:
    """Return tool names for a resolved catalog id."""
    try:
        rows = await db.fetchall(
            "SELECT tool_name FROM tool_catalog_entries WHERE catalog_id = ?",
            catalog_id,
        )
    except _TOOL_CATALOG_LOOKUP_EXCEPTIONS as exc:
        logger.debug("MCP catalog entries lookup failed: {}", exc.__class__.__name__)
        return set() if strict else None

    names: set[str] = set()
    for row in rows or []:
        value = _row_value(row, "tool_name", 0)
        if isinstance(value, str):
            names.add(value)
    if names:
        return names
    return set() if strict else None


async def resolve_tool_catalog_filter_names(
    db: Any,
    *,
    catalog_name: str | None,
    catalog_id: Any,
    metadata: Mapping[str, Any] | None,
    strict: bool,
) -> set[str] | None:
    """Resolve MCP catalog request parameters into a tool-name filter.

    The AuthNZ database pool normalizes SQLite and PostgreSQL placeholder
    styles behind ``fetchone``/``fetchall``. Keeping this SQL in DB_Management
    preserves the repository database boundary while allowing the MCP runtime
    adapter and admin service to remain host-service bridges instead of query
    owners.
    """
    resolved_id = _coerce_catalog_id(catalog_id)
    if resolved_id is None and isinstance(catalog_name, str) and catalog_name.strip():
        resolved_id = await _resolve_tool_catalog_name(
            db,
            name=catalog_name.strip(),
            metadata=metadata or {},
        )

    if resolved_id is None:
        return set() if strict else None

    return await _resolve_tool_catalog_entries(
        db,
        catalog_id=resolved_id,
        strict=strict,
    )
