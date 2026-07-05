"""Small SQLite schema helpers shared by database-backed modules."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any


FetchAll = Callable[[str], Awaitable[list[Any]]]
Execute = Callable[[str], Awaitable[Any]]


def _trusted_sqlite_identifier(identifier: str) -> str:
    """Return a trusted SQLite identifier or raise for unsafe input."""
    value = str(identifier or "").strip()
    if not value or not value.replace("_", "").isalnum():
        raise ValueError(f"Unsafe SQLite identifier: {identifier!r}")
    return value


def _table_info_column_name(row: Any) -> str:
    """Extract the column name from a SQLite PRAGMA table_info row."""
    if isinstance(row, Mapping) and row.get("name") is not None:
        return str(row["name"]).lower()

    keys = getattr(row, "keys", None)
    if callable(keys) and "name" in keys():
        return str(row["name"]).lower()

    if (
        isinstance(row, Sequence)
        and not isinstance(row, str | bytes | bytearray)
        and len(row) > 1
    ):
        return str(row[1]).lower()

    try:
        return str(row[1]).lower()
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"Unable to extract SQLite table_info column name from row: {row!r}") from exc


async def ensure_sqlite_column_exists(
    *,
    fetchall: FetchAll,
    execute: Execute,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    """Add a column to a SQLite table when PRAGMA table_info says it is missing."""
    table_identifier = _trusted_sqlite_identifier(table_name)
    column_identifier = _trusted_sqlite_identifier(column_name)
    definition = str(column_definition or "").strip()
    if not definition or ";" in definition:
        raise ValueError(f"Unsafe SQLite column definition: {column_definition!r}")

    rows = await fetchall(f"PRAGMA table_info({table_identifier})")  # nosec B608
    column_names = {_table_info_column_name(row) for row in rows}
    if column_identifier.lower() not in column_names:
        await execute(
            f"ALTER TABLE {table_identifier} ADD COLUMN {column_identifier} {definition}"  # nosec B608
        )
