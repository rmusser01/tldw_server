"""RBAC administration queries: roles, permissions, grants, user roles and overrides.

Moved out of the admin RBAC endpoints (TASK-13362). Every function runs on the caller's
request-transaction connection (see ``_dual_backend``); the caller passes
``is_postgres`` because the endpoints' backend detection is what tests patch.
"""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.AuthNZ.repos._dual_backend import execute, fetch_all, fetch_one, fetch_value

_PERMISSION_COLUMNS = ("id", "name", "description", "category")


# --- permissions ------------------------------------------------------------------------


async def get_permission(conn: Any, *, is_postgres: bool, name: str) -> dict[str, Any] | None:
    return await fetch_one(
        conn, is_postgres, "SELECT id, name, description, category FROM permissions WHERE name = ?", (name,),
        _PERMISSION_COLUMNS,
    )


async def ensure_permission(
    conn: Any, *, is_postgres: bool, name: str, description: str, category: str
) -> dict[str, Any] | None:
    """The permission row, created first if missing."""
    if is_postgres:
        await execute(
            conn, True,
            "INSERT INTO permissions (name, description, category) VALUES (?, ?, ?) ON CONFLICT (name) DO NOTHING",
            (name, description, category),
        )
    elif await get_permission(conn, is_postgres=False, name=name) is None:
        await execute(
            conn, False, "INSERT INTO permissions (name, description, category) VALUES (?, ?, ?)",
            (name, description, category),
        )
    return await get_permission(conn, is_postgres=is_postgres, name=name)


def _permission_filter(is_postgres: bool, category: str | None, search: str | None) -> tuple[str, list[Any]]:
    """WHERE clause over ``permissions`` (usable unqualified or joined as ``p``) and its args."""
    clauses: list[str] = []
    args: list[Any] = []
    if category:
        clauses.append("category = ?")
        args.append(category)
    if search:
        like = "ILIKE" if is_postgres else "LIKE"
        clauses.append(f"(name {like} ? OR description {like} ?)")
        args.extend([f"%{search}%"] * 2)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", args


async def list_permissions(
    conn: Any, *, is_postgres: bool, category: str | None = None, search: str | None = None
) -> list[dict[str, Any]]:
    where, args = _permission_filter(is_postgres, category, search)
    return await fetch_all(
        conn, is_postgres,
        f"SELECT id, name, description, category FROM permissions{where} ORDER BY name",  # nosec B608
        args, _PERMISSION_COLUMNS,
    )


async def permission_categories(conn: Any, *, is_postgres: bool) -> list[str]:
    rows = await fetch_all(
        conn, is_postgres,
        "SELECT DISTINCT category FROM permissions WHERE category IS NOT NULL ORDER BY category", (), ("category",),
    )
    return [row["category"] for row in rows]


async def create_permission(
    conn: Any, *, is_postgres: bool, name: str, description: str | None, category: str | None
) -> dict[str, Any] | None:
    """The new permission, or None if one with that name exists (case-insensitively)."""
    exists = await fetch_one(conn, is_postgres, "SELECT 1 AS one FROM permissions WHERE LOWER(name) = LOWER(?)", (name,), ("one",))
    if exists:
        return None
    await execute(
        conn, is_postgres, "INSERT INTO permissions (name, description, category) VALUES (?, ?, ?)",
        (name, description, category),
    )
    return await get_permission(conn, is_postgres=is_postgres, name=name)


async def permission_id_by_name(conn: Any, *, is_postgres: bool, name: str) -> int | None:
    row = await get_permission(conn, is_postgres=is_postgres, name=name)
    return int(row["id"]) if row else None


# --- role grants ------------------------------------------------------------------------


async def grant_permission(conn: Any, *, is_postgres: bool, role_id: int, permission_id: int) -> None:
    await execute(
        conn, is_postgres,
        "INSERT INTO role_permissions (role_id, permission_id) VALUES (?, ?) ON CONFLICT DO NOTHING"
        if is_postgres
        else "INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES (?, ?)",
        (role_id, permission_id),
    )


async def revoke_permission(conn: Any, *, is_postgres: bool, role_id: int, permission_id: int) -> None:
    await execute(
        conn, is_postgres, "DELETE FROM role_permissions WHERE role_id = ? AND permission_id = ?", (role_id, permission_id)
    )


async def list_role_tool_permissions(conn: Any, *, is_postgres: bool, role_id: int) -> list[dict[str, Any]]:
    return await fetch_all(
        conn, is_postgres,
        "SELECT p.name, p.description, p.category FROM permissions p"
        " JOIN role_permissions rp ON rp.permission_id = p.id"
        " WHERE rp.role_id = ? AND p.name LIKE 'tools.execute:%' ORDER BY p.name",
        (role_id,), ("name", "description", "category"),
    )


async def grant_tool_permissions(
    conn: Any, *, is_postgres: bool, role_id: int, permissions: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    """Grant each (name, description), creating missing catalog entries; returns what was granted."""
    granted: list[dict[str, Any]] = []
    for name, description in permissions:
        row = await ensure_permission(conn, is_postgres=is_postgres, name=name, description=description, category="tools")
        if not row:
            continue
        await grant_permission(conn, is_postgres=is_postgres, role_id=role_id, permission_id=int(row["id"]))
        granted.append(row)
    return granted


async def revoke_tool_permissions(conn: Any, *, is_postgres: bool, role_id: int, names: list[str]) -> list[str]:
    revoked: list[str] = []
    for name in names:
        permission_id = await permission_id_by_name(conn, is_postgres=is_postgres, name=name)
        if permission_id is not None:
            await revoke_permission(conn, is_postgres=is_postgres, role_id=role_id, permission_id=permission_id)
            revoked.append(name)
    return revoked


async def permissions_with_prefix(conn: Any, *, is_postgres: bool, prefix: str) -> list[dict[str, Any]]:
    return await fetch_all(
        conn, is_postgres, "SELECT id, name, description, category FROM permissions WHERE name LIKE ?",
        (prefix + "%",), _PERMISSION_COLUMNS,
    )


# --- roles / matrix -----------------------------------------------------------------------


async def get_role(conn: Any, *, is_postgres: bool, role_id: int) -> dict[str, Any] | None:
    return await fetch_one(conn, is_postgres, "SELECT id, name FROM roles WHERE id = ?", (int(role_id),), ("id", "name"))


async def roles_page(
    conn: Any,
    *,
    is_postgres: bool,
    role_search: str | None,
    role_names: list[str] | None,
    limit: int | None,
    offset: int | None,
) -> tuple[int, list[dict[str, Any]]]:
    """(total matching roles, the requested page of them) ordered by name."""
    clauses: list[str] = []
    args: list[Any] = []
    if role_search:
        clauses.append("name ILIKE ?" if is_postgres else "name LIKE ?")
        args.append(f"%{role_search}%")
    if role_names:
        if is_postgres:
            clauses.append("name = ANY(?)")
            args.append(list(role_names))
        else:
            clauses.append(f"name IN ({','.join('?' * len(role_names))})")
            args.extend(role_names)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    total = await fetch_value(conn, is_postgres, f"SELECT COUNT(*) FROM roles{where}", args)  # nosec B608
    rows = await fetch_all(
        conn, is_postgres,
        f"SELECT id, name, description, COALESCE(is_system, {'FALSE' if is_postgres else '0'}) AS is_system"  # nosec B608
        f" FROM roles{where} ORDER BY name LIMIT ? OFFSET ?",
        [*args, limit, offset], ("id", "name", "description", "is_system"),
    )
    return int(total or 0), rows


async def role_permission_grants(
    conn: Any,
    *,
    is_postgres: bool,
    category: str | None = None,
    search: str | None = None,
    role_ids: list[int] | None = None,
) -> list[tuple[int, int]]:
    """(role_id, permission_id) grants over the filtered permissions, optionally for some roles."""
    where, args = _permission_filter(is_postgres, category, search)
    sql = (
        "SELECT rp.role_id, rp.permission_id FROM role_permissions rp"  # nosec B608
        f" JOIN permissions p ON p.id = rp.permission_id{where}"
    )
    if role_ids:
        if is_postgres:
            sql += " AND rp.role_id = ANY(?)"
            args.append(list(role_ids))
        else:
            sql += f" AND rp.role_id IN ({','.join('?' * len(role_ids))})"
            args.extend(role_ids)
    rows = await fetch_all(conn, is_postgres, sql, args, ("role_id", "permission_id"))
    return [(int(row["role_id"]), int(row["permission_id"])) for row in rows]


# --- users ----------------------------------------------------------------------------------


async def add_user_role(conn: Any, *, is_postgres: bool, user_id: int, role_id: int) -> None:
    await execute(
        conn, is_postgres,
        "INSERT INTO user_roles (user_id, role_id) VALUES (?, ?) ON CONFLICT (user_id, role_id) DO NOTHING"
        if is_postgres
        else "INSERT OR IGNORE INTO user_roles (user_id, role_id) VALUES (?, ?)",
        (user_id, role_id),
    )


async def remove_user_role(conn: Any, *, is_postgres: bool, user_id: int, role_id: int) -> None:
    await execute(conn, is_postgres, "DELETE FROM user_roles WHERE user_id = ? AND role_id = ?", (user_id, role_id))


async def upsert_user_override(
    conn: Any, *, is_postgres: bool, user_id: int, permission_id: int, granted: bool, expires_at: Any
) -> None:
    await execute(
        conn, is_postgres,
        "INSERT INTO user_permissions (user_id, permission_id, granted, expires_at) VALUES (?, ?, ?, ?)"
        " ON CONFLICT (user_id, permission_id) DO UPDATE SET granted = EXCLUDED.granted, expires_at = EXCLUDED.expires_at"
        if is_postgres
        else "INSERT OR REPLACE INTO user_permissions (user_id, permission_id, granted, expires_at) VALUES (?, ?, ?, ?)",
        (user_id, permission_id, granted, expires_at),
    )


async def delete_user_override(conn: Any, *, is_postgres: bool, user_id: int, permission_id: int) -> None:
    await execute(
        conn, is_postgres, "DELETE FROM user_permissions WHERE user_id = ? AND permission_id = ?", (user_id, permission_id)
    )
