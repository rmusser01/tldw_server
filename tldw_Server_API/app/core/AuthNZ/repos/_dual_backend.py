"""Query helpers for AuthNZ repositories that run on a request transaction connection.

``get_db_transaction`` yields an asyncpg connection on PostgreSQL (``fetch``/``execute``
with ``$n`` placeholders and positional args) and an aiosqlite-style connection on
SQLite (``execute(sql, params)`` returning a cursor). Queries are written once with
``?`` placeholders; ``dollar`` rewrites them for asyncpg.
"""

from __future__ import annotations

from typing import Any


def dollar(sql: str) -> str:
    """Rewrite ``?`` placeholders as ``$1..$n``. The SQL must not contain a literal ``?``."""
    parts = sql.split("?")
    return "".join(part + (f"${i}" if i < len(parts) else "") for i, part in enumerate(parts, start=1))


def as_dict(row: Any, columns: tuple[str, ...]) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return row
    if hasattr(row, "keys"):
        return {str(key): row[key] for key in row.keys()}
    return {key: row[idx] if idx < len(row) else None for idx, key in enumerate(columns)}


async def fetch_all(
    conn: Any, is_postgres: bool, sql: str, args: tuple[Any, ...] | list[Any], columns: tuple[str, ...]
) -> list[dict[str, Any]]:
    if is_postgres:
        rows = await conn.fetch(dollar(sql), *args)
    else:
        cursor = await conn.execute(sql, tuple(args))
        rows = await cursor.fetchall()
    return [as_dict(row, columns) for row in rows or []]


async def fetch_one(
    conn: Any, is_postgres: bool, sql: str, args: tuple[Any, ...] | list[Any], columns: tuple[str, ...]
) -> dict[str, Any] | None:
    rows = await fetch_all(conn, is_postgres, sql, args, columns)
    return rows[0] if rows else None


async def execute(conn: Any, is_postgres: bool, sql: str, args: tuple[Any, ...] | list[Any] = ()) -> None:
    if is_postgres:
        await conn.execute(dollar(sql), *args)
    else:
        await conn.execute(sql, tuple(args))


async def fetch_value(conn: Any, is_postgres: bool, sql: str, args: tuple[Any, ...] | list[Any] = ()) -> Any:
    """The first column of the first row, or None."""
    if is_postgres:
        return await conn.fetchval(dollar(sql), *args)
    cursor = await conn.execute(sql, tuple(args))
    row = await cursor.fetchone()
    if not row:
        return None
    return next(iter(row.values())) if isinstance(row, dict) else row[0]
