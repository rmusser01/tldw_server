from __future__ import annotations

import inspect
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote


def _iter_database_list_rows(conn: Any) -> Iterable[Any]:
    rows = conn.execute("PRAGMA database_list")
    if rows is None:
        return ()
    return rows


def _is_in_memory_connection(conn: Any) -> bool:
    try:
        rows = tuple(_iter_database_list_rows(conn))
    except Exception:
        return False

    if not rows:
        return False

    for row in rows:
        if len(row) >= 3 and not row[2]:
            return True
    return False


def configure_sqlite_connection(
    conn: Any,
    *,
    use_wal: bool = True,
    synchronous: str | None = "NORMAL",
    foreign_keys: bool = True,
    busy_timeout_ms: int = 5000,
    temp_store: str = "MEMORY",
    cache_size: int | None = None,
    enable_on_memory: bool = False,
) -> None:
    is_memory = _is_in_memory_connection(conn)

    if use_wal and (enable_on_memory or not is_memory):
        conn.execute("PRAGMA journal_mode=WAL")

    if synchronous:
        conn.execute(f"PRAGMA synchronous={synchronous}")

    conn.execute(f"PRAGMA foreign_keys={'ON' if foreign_keys else 'OFF'}")
    conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")

    if temp_store:
        conn.execute(f"PRAGMA temp_store={temp_store}")

    if cache_size is not None:
        conn.execute(f"PRAGMA cache_size={int(cache_size)}")


def _sqlite_readonly_uri(db_path: Path) -> str:
    return f"file:{quote(str(db_path), safe='/:')}?mode=ro"


def run_sqlite_quick_check(
    db_path: str | Path,
    *,
    timeout_s: float = 1.0,
    busy_timeout_ms: int = 1000,
) -> list[str]:
    path = Path(db_path)
    if not path.exists():
        return []

    with sqlite3.connect(_sqlite_readonly_uri(path), uri=True, timeout=timeout_s) as conn:
        configure_sqlite_connection(
            conn,
            use_wal=False,
            synchronous=None,
            foreign_keys=False,
            busy_timeout_ms=busy_timeout_ms,
            temp_store=None,
        )
        rows = conn.execute("PRAGMA quick_check;").fetchall()

    return [str(row[0]).strip() for row in rows if row and str(row[0]).strip()]


def begin_immediate_if_needed(conn: Any) -> bool:
    if getattr(conn, "in_transaction", False):
        return False

    conn.execute("BEGIN IMMEDIATE")
    return True


async def _iter_database_list_rows_async(conn: Any) -> Iterable[Any]:
    rows = await conn.execute("PRAGMA database_list")
    if rows is None:
        return ()
    fetchall = getattr(rows, "fetchall", None)
    if callable(fetchall):
        result = fetchall()
        if inspect.isawaitable(result):
            result = await result
        return result or ()
    return rows


async def _is_in_memory_connection_async(conn: Any) -> bool:
    try:
        rows = tuple(await _iter_database_list_rows_async(conn))
    except Exception:
        return False

    if not rows:
        return False

    for row in rows:
        if len(row) >= 3 and not row[2]:
            return True
    return False


async def configure_sqlite_connection_async(
    conn: Any,
    *,
    use_wal: bool = True,
    synchronous: str | None = "NORMAL",
    foreign_keys: bool = True,
    busy_timeout_ms: int = 5000,
    temp_store: str = "MEMORY",
    cache_size: int | None = None,
    enable_on_memory: bool = False,
) -> None:
    is_memory = await _is_in_memory_connection_async(conn)

    if use_wal and (enable_on_memory or not is_memory):
        await conn.execute("PRAGMA journal_mode=WAL")

    if synchronous:
        await conn.execute(f"PRAGMA synchronous={synchronous}")

    await conn.execute(f"PRAGMA foreign_keys={'ON' if foreign_keys else 'OFF'}")
    await conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")

    if temp_store:
        await conn.execute(f"PRAGMA temp_store={temp_store}")

    if cache_size is not None:
        await conn.execute(f"PRAGMA cache_size={int(cache_size)}")
