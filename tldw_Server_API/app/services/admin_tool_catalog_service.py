from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import (
    build_postgres_in_clause,
    build_sqlite_in_clause,
)
from tldw_Server_API.app.core.DB_Management import Tool_Catalog_DB as tool_catalog_db
from tldw_Server_API.app.core.exceptions import ToolCatalogConflictError


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


async def resolve_tool_catalog_filter_names(
    db: Any,
    *,
    catalog_name: str | None,
    catalog_id: Any,
    metadata: Mapping[str, Any] | None,
    strict: bool,
) -> set[str] | None:
    """Resolve MCP catalog request parameters through the DB_Management layer."""
    return await tool_catalog_db.resolve_tool_catalog_filter_names(
        db,
        catalog_name=catalog_name,
        catalog_id=catalog_id,
        metadata=metadata,
        strict=strict,
    )


async def list_tool_catalogs(db, *, org_id: int | None, team_id: int | None, limit: int, offset: int) -> list[dict[str, Any]]:
    pg = _is_postgres_connection(db)
    where: list[str] = []
    params: list[Any] = []
    if org_id is not None:
        where.append("org_id = $1" if pg else "org_id = ?")
        params.append(org_id)
    if team_id is not None:
        if pg:
            where.append(f"team_id = ${len(params)+1}")
        else:
            where.append("team_id = ?")
        params.append(team_id)
    where_clause = (" WHERE " + " AND ".join(where)) if where else ""
    try:
        if pg:
            limit_param = len(params) + 1
            offset_param = len(params) + 2
            list_catalogs_sql_template = (
                "SELECT id, name, description, org_id, team_id, COALESCE(is_active,TRUE) as is_active, created_at, updated_at "
                "FROM tool_catalogs{where_clause} ORDER BY created_at DESC LIMIT ${limit_param} OFFSET ${offset_param}"
            )
            q = list_catalogs_sql_template.format_map(locals())  # nosec B608
            rows = await db.fetch(q, *params, limit, offset)
            return [dict(r) for r in rows]
        list_catalogs_sql_template = (
            "SELECT id, name, description, org_id, team_id, COALESCE(is_active,1), created_at, updated_at "
            "FROM tool_catalogs{where_clause} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        list_catalogs_sql = list_catalogs_sql_template.format_map(locals())  # nosec B608
        cur = await db.execute(
            list_catalogs_sql,
            [*params, limit, offset],
        )
        rows = await cur.fetchall()
        return [
            {"id": r[0], "name": r[1], "description": r[2], "org_id": r[3], "team_id": r[4], "is_active": bool(r[5]), "created_at": r[6], "updated_at": r[7]}
            for r in rows
        ]
    except Exception:
        logger.error("Failed to list tool catalogs")
        raise


async def list_visible_tool_catalogs(
    db,
    *,
    scope_norm: str,
    admin_all: bool,
    org_ids: set[int] | None = None,
    team_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """
    List catalogs visible to the caller scope for MCP unified discovery.

    This encapsulates backend-specific SQL for global/org/team scope listing
    so API endpoints do not branch on backend details.
    """
    pg = _is_postgres_connection(db)
    visible_rows: list[dict[str, Any]] = []
    org_ids = org_ids or set()
    team_ids = team_ids or set()

    async def _extend_rows(rows: list[Any]) -> None:
        for row in rows or []:
            if isinstance(row, dict) or hasattr(row, "keys"):
                data = dict(row)
                data["is_active"] = bool(data.get("is_active", True))
            else:
                data = {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "org_id": row[3],
                    "team_id": row[4],
                    "is_active": bool(row[5]),
                    "created_at": row[6],
                    "updated_at": row[7],
                }
            visible_rows.append(data)

    try:
        if admin_all:
            if scope_norm in {"all", "global"}:
                if pg:
                    rows = await db.fetch(
                        "SELECT id, name, description, org_id, team_id, COALESCE(is_active, TRUE) as is_active, created_at, updated_at "
                        "FROM tool_catalogs WHERE org_id IS NULL AND team_id IS NULL ORDER BY created_at DESC"
                    )
                else:
                    cur = await db.execute(
                        "SELECT id, name, description, org_id, team_id, COALESCE(is_active,1), created_at, updated_at "
                        "FROM tool_catalogs WHERE org_id IS NULL AND team_id IS NULL ORDER BY created_at DESC"
                    )
                    rows = await cur.fetchall()
                await _extend_rows(rows)

            if scope_norm in {"all", "org"}:
                if pg:
                    rows = await db.fetch(
                        "SELECT id, name, description, org_id, team_id, COALESCE(is_active, TRUE) as is_active, created_at, updated_at "
                        "FROM tool_catalogs WHERE org_id IS NOT NULL AND team_id IS NULL ORDER BY created_at DESC"
                    )
                else:
                    cur = await db.execute(
                        "SELECT id, name, description, org_id, team_id, COALESCE(is_active,1), created_at, updated_at "
                        "FROM tool_catalogs WHERE org_id IS NOT NULL AND team_id IS NULL ORDER BY created_at DESC"
                    )
                    rows = await cur.fetchall()
                await _extend_rows(rows)

            if scope_norm in {"all", "team"}:
                if pg:
                    rows = await db.fetch(
                        "SELECT id, name, description, org_id, team_id, COALESCE(is_active, TRUE) as is_active, created_at, updated_at "
                        "FROM tool_catalogs WHERE team_id IS NOT NULL ORDER BY created_at DESC"
                    )
                else:
                    cur = await db.execute(
                        "SELECT id, name, description, org_id, team_id, COALESCE(is_active,1), created_at, updated_at "
                        "FROM tool_catalogs WHERE team_id IS NOT NULL ORDER BY created_at DESC"
                    )
                    rows = await cur.fetchall()
                await _extend_rows(rows)
            return visible_rows

        if scope_norm in {"all", "global"}:
            if pg:
                rows = await db.fetch(
                    "SELECT id, name, description, org_id, team_id, COALESCE(is_active, TRUE) as is_active, created_at, updated_at "
                    "FROM tool_catalogs WHERE org_id IS NULL AND team_id IS NULL ORDER BY created_at DESC"
                )
            else:
                cur = await db.execute(
                    "SELECT id, name, description, org_id, team_id, COALESCE(is_active,1), created_at, updated_at "
                    "FROM tool_catalogs WHERE org_id IS NULL AND team_id IS NULL ORDER BY created_at DESC"
                )
                rows = await cur.fetchall()
            await _extend_rows(rows)

        if scope_norm in {"all", "org"} and org_ids:
            if pg:
                placeholders, params = build_postgres_in_clause(sorted(org_ids))
                rows = await db.fetch(
                    "SELECT id, name, description, org_id, team_id, COALESCE(is_active, TRUE) as is_active, created_at, updated_at "
                    f"FROM tool_catalogs WHERE org_id IN ({placeholders}) AND team_id IS NULL ORDER BY created_at DESC",  # nosec B608
                    *params,
                )
            else:
                placeholders, params = build_sqlite_in_clause(sorted(org_ids))
                org_catalogs_sql_template = (
                    "SELECT id, name, description, org_id, team_id, COALESCE(is_active,1), created_at, updated_at "
                    "FROM tool_catalogs WHERE org_id IN ({placeholders}) AND team_id IS NULL ORDER BY created_at DESC"
                )
                org_catalogs_sql = org_catalogs_sql_template.format_map(locals())  # nosec B608
                cur = await db.execute(
                    org_catalogs_sql,
                    params,
                )
                rows = await cur.fetchall()
            await _extend_rows(rows)

        if scope_norm in {"all", "team"} and team_ids:
            if pg:
                placeholders, params = build_postgres_in_clause(sorted(team_ids))
                rows = await db.fetch(
                    "SELECT id, name, description, org_id, team_id, COALESCE(is_active, TRUE) as is_active, created_at, updated_at "
                    f"FROM tool_catalogs WHERE team_id IN ({placeholders}) ORDER BY created_at DESC",  # nosec B608
                    *params,
                )
            else:
                placeholders, params = build_sqlite_in_clause(sorted(team_ids))
                team_catalogs_sql_template = (
                    "SELECT id, name, description, org_id, team_id, COALESCE(is_active,1), created_at, updated_at "
                    "FROM tool_catalogs WHERE team_id IN ({placeholders}) ORDER BY created_at DESC"
                )
                team_catalogs_sql = team_catalogs_sql_template.format_map(locals())  # nosec B608
                cur = await db.execute(
                    team_catalogs_sql,
                    params,
                )
                rows = await cur.fetchall()
            await _extend_rows(rows)

        return visible_rows
    except Exception:
        logger.error("Failed to list visible tool catalogs")
        raise


async def create_tool_catalog(db, *, name: str, description: str | None, org_id: int | None, team_id: int | None, is_active: bool) -> dict[str, Any]:
    pg = _is_postgres_connection(db)
    try:
        if pg:
            # Pre-check case-insensitive existence within scope
            exists = await db.fetchrow(
                "SELECT 1 FROM tool_catalogs WHERE LOWER(name) = LOWER($1) AND ((org_id IS NOT DISTINCT FROM $2) AND (team_id IS NOT DISTINCT FROM $3))",
                name, org_id, team_id,
            )
            if exists:
                raise ToolCatalogConflictError("Catalog already exists")
            await db.execute(
                "INSERT INTO tool_catalogs (name, description, org_id, team_id, is_active) VALUES ($1,$2,$3,$4,$5)",
                name, description, org_id, team_id, is_active,
            )
            row = await db.fetchrow(
                "SELECT id, name, description, org_id, team_id, COALESCE(is_active, TRUE) as is_active, created_at, updated_at FROM tool_catalogs WHERE name = $1 AND ((org_id IS NOT DISTINCT FROM $2) AND (team_id IS NOT DISTINCT FROM $3))",
                name, org_id, team_id,
            )
            return dict(row)
        # SQLite path
        cur = await db.execute(
            "SELECT id FROM tool_catalogs WHERE LOWER(name) = LOWER(?) AND ( (org_id IS ? OR org_id = ?) AND (team_id IS ? OR team_id = ?) )",
            (name, None, org_id, None, team_id),
        )
        if await cur.fetchone():
            raise ToolCatalogConflictError("Catalog already exists")
        await db.execute(
            "INSERT INTO tool_catalogs (name, description, org_id, team_id, is_active) VALUES (?, ?, ?, ?, ?)",
            (name, description, org_id, team_id, 1 if is_active else 0),
        )
        cur2 = await db.execute(
            "SELECT id, name, description, org_id, team_id, is_active, created_at, updated_at FROM tool_catalogs WHERE name = ? AND ( (org_id IS ? OR org_id = ?) AND (team_id IS ? OR team_id = ?) )",
            (name, None, org_id, None, team_id),
        )
        r = await cur2.fetchone()
        return {"id": r[0], "name": r[1], "description": r[2], "org_id": r[3], "team_id": r[4], "is_active": bool(r[5]), "created_at": r[6], "updated_at": r[7]}
    except Exception:
        logger.error("Failed to create tool catalog")
        raise


async def get_tool_catalog(db, catalog_id: int) -> dict[str, Any] | None:
    pg = _is_postgres_connection(db)
    try:
        if pg:
            row = await db.fetchrow(
                "SELECT id, name, description, org_id, team_id, COALESCE(is_active, TRUE) as is_active, created_at, updated_at FROM tool_catalogs WHERE id = $1",
                catalog_id,
            )
            return dict(row) if row else None
        cur = await db.execute(
            "SELECT id, name, description, org_id, team_id, COALESCE(is_active,1), created_at, updated_at FROM tool_catalogs WHERE id = ?",
            (catalog_id,),
        )
        r = await cur.fetchone()
        if not r:
            return None
        return {
            "id": r[0],
            "name": r[1],
            "description": r[2],
            "org_id": r[3],
            "team_id": r[4],
            "is_active": bool(r[5]),
            "created_at": r[6],
            "updated_at": r[7],
        }
    except Exception:
        logger.error("Failed to get tool catalog")
        raise


async def delete_tool_catalog(db, catalog_id: int) -> None:
    pg = _is_postgres_connection(db)
    try:
        if pg:
            await db.execute("DELETE FROM tool_catalogs WHERE id = $1", catalog_id)
            return
        await db.execute("DELETE FROM tool_catalogs WHERE id = ?", (catalog_id,))
        commit = getattr(db, "commit", None)
        if callable(commit):
            await commit()
    except Exception:
        logger.error("Failed to delete tool catalog")
        raise


async def list_tool_catalog_entries(
    db,
    catalog_id: int,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    pg = _is_postgres_connection(db)
    try:
        if pg:
            rows = await db.fetch(
                "SELECT catalog_id, tool_name, module_id FROM tool_catalog_entries WHERE catalog_id = $1 ORDER BY tool_name LIMIT $2 OFFSET $3",
                catalog_id,
                limit,
                offset,
            )
            return [dict(r) for r in rows]
        cur = await db.execute(
            "SELECT catalog_id, tool_name, module_id FROM tool_catalog_entries WHERE catalog_id = ? ORDER BY tool_name LIMIT ? OFFSET ?",
            (catalog_id, limit, offset),
        )
        rows = await cur.fetchall()
        return [{"catalog_id": r[0], "tool_name": r[1], "module_id": r[2]} for r in rows]
    except Exception:
        logger.error("Failed to list tool catalog entries")
        raise


async def add_tool_catalog_entry(db, catalog_id: int, tool_name: str, module_id: str | None) -> dict[str, Any]:
    pg = _is_postgres_connection(db)
    try:
        if pg:
            await db.execute("INSERT INTO tool_catalog_entries (catalog_id, tool_name, module_id) VALUES ($1,$2,$3) ON CONFLICT (catalog_id, tool_name) DO NOTHING", catalog_id, tool_name, module_id)
            row = await db.fetchrow("SELECT catalog_id, tool_name, module_id FROM tool_catalog_entries WHERE catalog_id = $1 AND tool_name = $2", catalog_id, tool_name)
            return dict(row) if row else {"catalog_id": catalog_id, "tool_name": tool_name, "module_id": module_id}
        cur = await db.execute("SELECT catalog_id, tool_name, module_id FROM tool_catalog_entries WHERE catalog_id = ? AND tool_name = ?", (catalog_id, tool_name))
        r = await cur.fetchone()
        if not r:
            await db.execute("INSERT OR IGNORE INTO tool_catalog_entries (catalog_id, tool_name, module_id) VALUES (?, ?, ?)", (catalog_id, tool_name, module_id))
            commit = getattr(db, "commit", None)
            if callable(commit):
                await commit()
            cur2 = await db.execute("SELECT catalog_id, tool_name, module_id FROM tool_catalog_entries WHERE catalog_id = ? AND tool_name = ?", (catalog_id, tool_name))
            r = await cur2.fetchone()
        return {"catalog_id": r[0], "tool_name": r[1], "module_id": r[2]} if r else {"catalog_id": catalog_id, "tool_name": tool_name, "module_id": module_id}
    except Exception:
        logger.error("Failed to add tool catalog entry")
        raise


async def delete_tool_catalog_entry(db, catalog_id: int, tool_name: str) -> None:
    pg = _is_postgres_connection(db)
    try:
        if pg:
            await db.execute("DELETE FROM tool_catalog_entries WHERE catalog_id = $1 AND tool_name = $2", catalog_id, tool_name)
            return
        await db.execute("DELETE FROM tool_catalog_entries WHERE catalog_id = ? AND tool_name = ?", (catalog_id, tool_name))
        commit = getattr(db, "commit", None)
        if callable(commit):
            await commit()
    except Exception:
        logger.error("Failed to delete tool catalog entry")
        raise
