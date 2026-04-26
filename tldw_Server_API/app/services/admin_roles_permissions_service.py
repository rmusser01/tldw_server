from __future__ import annotations

from typing import Any

from loguru import logger

def _is_postgres_connection(db: Any) -> bool:
    """Resolve backend mode from connection/adapter shape without global probing."""
    sqlite_hint = getattr(db, "_is_sqlite", None)
    if isinstance(sqlite_hint, bool):
        return not sqlite_hint

    # SQLite shims in AuthNZ DatabasePool expose underlying aiosqlite connection as `_c`.
    if getattr(db, "_c", None) is not None:
        return False

    module_name = getattr(type(db), "__module__", "")
    if isinstance(module_name, str) and module_name.startswith("asyncpg"):
        return True

    # Last-resort interface hint for test doubles and adapters.
    return callable(getattr(db, "fetchrow", None))


async def list_roles(db) -> list[dict[str, Any]]:
    is_pg = _is_postgres_connection(db)
    try:
        if is_pg:
            rows = await db.fetch("SELECT id, name, description, COALESCE(is_system, FALSE) as is_system FROM roles ORDER BY name")
            return [dict(r) for r in rows]
        cur = await db.execute("SELECT id, name, description, COALESCE(is_system, 0) as is_system FROM roles ORDER BY name")
        return [{"id": r[0], "name": r[1], "description": r[2], "is_system": bool(r[3])} for r in await cur.fetchall()]
    except Exception as e:
        logger.error("Failed to list roles")
        raise


async def create_role(db, name: str, description: str | None = None, is_system: bool = False) -> dict[str, Any]:
    from tldw_Server_API.app.core.AuthNZ.exceptions import DuplicateRoleError
    is_pg = _is_postgres_connection(db)
    try:
        if is_pg:
            # Pre-check case-insensitive
            exists = await db.fetchrow("SELECT 1 FROM roles WHERE LOWER(name) = LOWER($1)", name)
            if exists:
                raise DuplicateRoleError(name)
            row = await db.fetchrow(
                "INSERT INTO roles (name, description, is_system) VALUES ($1, $2, $3) RETURNING id, name, description, is_system",
                name, description, is_system,
            )
            return dict(row)
        # SQLite path
        # Pre-check duplicate (case-insensitive)
        curx = await db.execute("SELECT 1 FROM roles WHERE LOWER(name) = LOWER(?)", (name,))
        if await curx.fetchone():
            raise DuplicateRoleError(name)
        cur = await db.execute("INSERT INTO roles (name, description, is_system) VALUES (?, ?, ?)", (name, description, 1 if is_system else 0))
        commit = getattr(db, "commit", None)
        if callable(commit):
            await commit()
        rid = cur.lastrowid
        cur2 = await db.execute("SELECT id, name, description, COALESCE(is_system,0) FROM roles WHERE id = ?", (rid,))
        r = await cur2.fetchone()
        return {"id": r[0], "name": r[1], "description": r[2], "is_system": bool(r[3])}
    except Exception as e:
        logger.error("Failed to create role")
        raise


async def delete_role(db, role_id: int) -> bool:
    is_pg = _is_postgres_connection(db)
    try:
        if is_pg:
            await db.execute("DELETE FROM roles WHERE id = $1 AND COALESCE(is_system, FALSE) = FALSE", role_id)
        else:
            await db.execute("DELETE FROM roles WHERE id = ? AND COALESCE(is_system, 0) = 0", (role_id,))
            commit = getattr(db, "commit", None)
            if callable(commit):
                await commit()
        return True
    except Exception as e:
        logger.error("Failed to delete role")
        raise


async def list_role_permissions(db, role_id: int) -> list[dict[str, Any]]:
    is_pg = _is_postgres_connection(db)
    try:
        if is_pg:
            rows = await db.fetch(
                """
                SELECT p.id, p.name, p.description, p.category
                FROM permissions p
                JOIN role_permissions rp ON p.id = rp.permission_id
                WHERE rp.role_id = $1
                ORDER BY p.name
                """,
                role_id,
            )
            return [dict(r) for r in rows]
        cur = await db.execute(
            """
            SELECT p.id, p.name, p.description, p.category
            FROM permissions p
            JOIN role_permissions rp ON p.id = rp.permission_id
            WHERE rp.role_id = ?
            ORDER BY p.name
            """,
            (role_id,),
        )
        return [{"id": r[0], "name": r[1], "description": r[2], "category": r[3]} for r in await cur.fetchall()]
    except Exception as e:
        logger.error("Failed to list role permissions")
        raise


async def list_tool_permissions(db) -> list[dict[str, Any]]:
    is_pg = _is_postgres_connection(db)
    try:
        if is_pg:
            rows = await db.fetch("SELECT name, description, category FROM permissions WHERE name LIKE 'tools.execute:%' ORDER BY name")
            return [dict(r) for r in rows]
        cur = await db.execute("SELECT name, description, category FROM permissions WHERE name LIKE 'tools.execute:%' ORDER BY name")
        return [{"name": r[0], "description": r[1], "category": r[2]} for r in await cur.fetchall()]
    except Exception as e:
        logger.error("Failed to list tool permissions")
        raise


async def delete_tool_permission(db, full_name: str) -> bool:
    is_pg = _is_postgres_connection(db)
    try:
        if is_pg:
            await db.execute("DELETE FROM permissions WHERE name = $1", full_name)
        else:
            await db.execute("DELETE FROM permissions WHERE name = ?", (full_name,))
            commit = getattr(db, "commit", None)
            if callable(commit):
                await commit()
        return True
    except Exception as e:
        logger.error("Failed to delete tool permission")
        raise


async def ensure_permission(db, name: str, description: str, category: str = 'tools') -> dict[str, Any]:
    """Idempotently create a permission and return its row."""
    is_pg = _is_postgres_connection(db)
    if is_pg:
        await db.execute(
            "INSERT INTO permissions (name, description, category) VALUES ($1, $2, $3) ON CONFLICT (name) DO NOTHING",
            name, description, category,
        )
        row = await db.fetchrow("SELECT id, name, description, category FROM permissions WHERE name = $1", name)
        if not row:
            raise RuntimeError("Failed to create or fetch permission")
        return dict(row)
    cur = await db.execute("SELECT id, name, description, category FROM permissions WHERE name = ?", (name,))
    r = await cur.fetchone()
    if not r:
        await db.execute("INSERT INTO permissions (name, description, category) VALUES (?, ?, ?)", (name, description, category))
        commit = getattr(db, "commit", None)
        if callable(commit):
            await commit()
        cur = await db.execute("SELECT id, name, description, category FROM permissions WHERE name = ?", (name,))
        r = await cur.fetchone()
    return {"id": r[0], "name": r[1], "description": r[2], "category": r[3]}


async def grant_tool_permission_to_role(db, role_id: int, permission_name: str, description: str) -> dict[str, Any]:
    perm = await ensure_permission(db, permission_name, description, category='tools')
    is_pg = _is_postgres_connection(db)
    try:
        if is_pg:
            await db.execute("INSERT INTO role_permissions (role_id, permission_id) VALUES ($1, $2) ON CONFLICT DO NOTHING", role_id, perm['id'])
            return perm
        await db.execute("INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES (?, ?)", (role_id, perm['id']))
        commit = getattr(db, "commit", None)
        if callable(commit):
            await commit()
        # Verify mapping
        cur2 = await db.execute("SELECT 1 FROM role_permissions WHERE role_id = ? AND permission_id = ?", (role_id, perm['id']))
        exists = await cur2.fetchone()
        if not exists:
            await db.execute("INSERT INTO role_permissions (role_id, permission_id) VALUES (?, ?)", (role_id, perm['id']))
            if callable(commit):
                await commit()
        return perm
    except Exception as e:
        logger.error("Failed to grant tool permission to role")
        raise


async def revoke_tool_permission_from_role(db, role_id: int, permission_name: str) -> bool:
    is_pg = _is_postgres_connection(db)
    try:
        if is_pg:
            row = await db.fetchrow("SELECT id FROM permissions WHERE name = $1", permission_name)
            if not row:
                return False
            await db.execute("DELETE FROM role_permissions WHERE role_id = $1 AND permission_id = $2", role_id, row['id'])
            return True
        cur = await db.execute("SELECT id FROM permissions WHERE name = ?", (permission_name,))
        r = await cur.fetchone()
        if not r:
            return False
        await db.execute("DELETE FROM role_permissions WHERE role_id = ? AND permission_id = ?", (role_id, r[0]))
        commit = getattr(db, "commit", None)
        if callable(commit):
            await commit()
        return True
    except Exception as e:
        logger.error("Failed to revoke tool permission from role")
        raise
