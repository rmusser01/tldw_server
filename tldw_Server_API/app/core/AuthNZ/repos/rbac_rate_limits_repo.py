"""Owner of rbac_role_rate_limits and rbac_user_rate_limits (TASK-13317 AC2).

Callers hold either a request transaction connection (``get_db_transaction``: an asyncpg
connection on PostgreSQL, an aiosqlite-style connection on SQLite) or the pool itself
(``DatabasePool``, for enforcement). Functions say which they take; backend selection is
the caller's, passed as ``is_postgres``, because tests patch how endpoints detect it.
"""

from __future__ import annotations

from typing import Any

_COLUMNS = ("scope", "id", "resource", "limit_per_min", "burst")
_LIST_QUERIES = (
    "SELECT 'role' AS scope, role_id AS id, resource, limit_per_min, burst"
    " FROM rbac_role_rate_limits ORDER BY role_id, resource",
    "SELECT 'user' AS scope, user_id AS id, resource, limit_per_min, burst"
    " FROM rbac_user_rate_limits ORDER BY user_id, resource",
)


def _as_dict(row: Any, columns: tuple[str, ...]) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return row
    if hasattr(row, "keys"):
        return {str(key): row[key] for key in row.keys()}
    return {key: row[idx] if idx < len(row) else None for idx, key in enumerate(columns)}


def _dollar(sql: str) -> str:
    """Rewrite ``?`` placeholders as ``$1..$n`` for asyncpg."""
    parts = sql.split("?")
    return "".join(part + (f"${i}" if i < len(parts) else "") for i, part in enumerate(parts, start=1))


async def _fetch(conn: Any, is_postgres: bool, sql: str, args: tuple[Any, ...], columns: tuple[str, ...]) -> list[dict[str, Any]]:
    if is_postgres:
        rows = await conn.fetch(_dollar(sql), *args)
    else:
        cursor = await conn.execute(sql, args)
        rows = await cursor.fetchall()
    return [_as_dict(row, columns) for row in rows or []]


async def _write(conn: Any, is_postgres: bool, pg_sql: str, sqlite_sql: str, args: tuple[Any, ...]) -> None:
    if is_postgres:
        await conn.execute(pg_sql, *args)
    else:
        await conn.execute(sqlite_sql, args)
        await conn.commit()


# --- Admin management (transaction connection) ---------------------------------------


async def list_all(conn: Any, *, is_postgres: bool) -> list[dict[str, Any]]:
    """Every role limit, then every user limit, as {scope, id, resource, limit_per_min, burst}."""
    rows: list[dict[str, Any]] = []
    for query in _LIST_QUERIES:
        rows.extend(await _fetch(conn, is_postgres, query, (), _COLUMNS))
    return rows


async def upsert_role_limit(
    conn: Any, *, is_postgres: bool, role_id: int, resource: str, limit_per_min: int | None, burst: int | None
) -> None:
    await _write(
        conn,
        is_postgres,
        "INSERT INTO rbac_role_rate_limits (role_id, resource, limit_per_min, burst) VALUES ($1, $2, $3, $4)"
        " ON CONFLICT (role_id, resource) DO UPDATE SET"
        " limit_per_min = EXCLUDED.limit_per_min, burst = EXCLUDED.burst",
        "INSERT OR REPLACE INTO rbac_role_rate_limits (role_id, resource, limit_per_min, burst) VALUES (?, ?, ?, ?)",
        (role_id, resource, limit_per_min, burst),
    )


async def clear_role_limits(conn: Any, *, is_postgres: bool, role_id: int) -> None:
    await _write(
        conn,
        is_postgres,
        "DELETE FROM rbac_role_rate_limits WHERE role_id = $1",
        "DELETE FROM rbac_role_rate_limits WHERE role_id = ?",
        (role_id,),
    )


async def upsert_user_limit(
    conn: Any, *, is_postgres: bool, user_id: int, resource: str, limit_per_min: int | None, burst: int | None
) -> None:
    await _write(
        conn,
        is_postgres,
        "INSERT INTO rbac_user_rate_limits (user_id, resource, limit_per_min, burst) VALUES ($1, $2, $3, $4)"
        " ON CONFLICT (user_id, resource) DO UPDATE SET"
        " limit_per_min = EXCLUDED.limit_per_min, burst = EXCLUDED.burst",
        "INSERT OR REPLACE INTO rbac_user_rate_limits (user_id, resource, limit_per_min, burst) VALUES (?, ?, ?, ?)",
        (user_id, resource, limit_per_min, burst),
    )


# --- Simulator reads (transaction connection) ----------------------------------------


async def user_limits(conn: Any, *, is_postgres: bool, user_id: int) -> list[dict[str, Any]]:
    return await _fetch(
        conn,
        is_postgres,
        "SELECT resource, limit_per_min, burst FROM rbac_user_rate_limits WHERE user_id = ?",
        (int(user_id),),
        ("resource", "limit_per_min", "burst"),
    )


async def role_limits(conn: Any, *, is_postgres: bool, user_id: int) -> list[dict[str, Any]]:
    """Limits on the user's current (unexpired) roles, with the role name."""
    return await _fetch(
        conn,
        is_postgres,
        "SELECT rrl.resource, rrl.limit_per_min, rrl.burst, r.name AS role_name"
        " FROM rbac_role_rate_limits rrl"
        " JOIN roles r ON rrl.role_id = r.id"
        " JOIN user_roles ur ON ur.role_id = r.id"
        " WHERE ur.user_id = ? AND (ur.expires_at IS NULL OR ur.expires_at > CURRENT_TIMESTAMP)",
        (int(user_id),),
        ("resource", "limit_per_min", "burst", "role_name"),
    )


# --- Enforcement (pool) ----------------------------------------------------------------


async def effective_limits(db_pool: Any, user_id: Any, resource: str) -> tuple[Any, Any]:
    """(user row, strictest role row) for ``resource``; each a row of (limit_per_min, burst) or None."""
    if db_pool.pool:  # PostgreSQL
        user_limit = await db_pool.fetchone(
            "SELECT limit_per_min, burst FROM rbac_user_rate_limits WHERE user_id = $1 AND resource = $2",
            user_id,
            resource,
        )
        role_ids = await db_pool.fetchall(
            "SELECT role_id FROM user_roles WHERE user_id = $1 AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)",
            user_id,
        )
        role_limit = None
        if role_ids:
            role_limit = await db_pool.fetchone(
                "SELECT MIN(limit_per_min) as limit_per_min, MIN(burst) as burst"
                " FROM rbac_role_rate_limits WHERE role_id = ANY($1) AND resource = $2",
                [r["role_id"] for r in role_ids],
                resource,
            )
        return user_limit, role_limit
    async with db_pool.acquire() as conn:
        c1 = await conn.execute(
            "SELECT limit_per_min, burst FROM rbac_user_rate_limits WHERE user_id = ? AND resource = ?",
            (user_id, resource),
        )
        user_limit = await c1.fetchone()
        c2 = await conn.execute(
            "SELECT MIN(rl.limit_per_min), MIN(rl.burst) FROM rbac_role_rate_limits rl"
            " JOIN user_roles ur ON ur.role_id = rl.role_id"
            " WHERE ur.user_id = ? AND (ur.expires_at IS NULL OR ur.expires_at > CURRENT_TIMESTAMP)"
            " AND rl.resource = ?",
            (user_id, resource),
        )
        return user_limit, await c2.fetchone()
