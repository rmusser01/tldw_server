"""
MCP Discovery Module for Unified MCP.

Provides read-only discovery tools that allow an LLM to list catalogs, modules,
and tools with optional filtering for progressive disclosure.
"""

from __future__ import annotations

import contextlib
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import build_sqlite_in_clause, get_db_pool
from tldw_Server_API.app.core.AuthNZ.orgs_teams import list_org_memberships_for_user
from tldw_Server_API.app.core.MCP_unified.environment import is_truthy
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext

from ..base import BaseModule, create_tool_definition


class MCPDiscoveryModule(BaseModule):
    """Discovery module for listing catalogs, modules, and tools."""

    async def on_initialize(self) -> None:
        logger.info(f"Initializing MCP discovery module: {self.name}")

    async def on_shutdown(self) -> None:
        logger.info(f"Shutting down MCP discovery module: {self.name}")

    async def check_health(self) -> dict[str, bool]:
        return {"initialized": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            create_tool_definition(
                name="mcp.catalogs.list",
                description="List MCP tool catalogs visible to the current user.",
                parameters={
                    "properties": {
                        "scope": {
                            "type": "string",
                            "enum": ["all", "global", "org", "team"],
                            "description": "Filter by scope: all, global, org, or team.",
                        },
                    },
                    "required": [],
                },
                metadata={"category": "discovery", "catalog_exempt": True},
            ),
            create_tool_definition(
                name="mcp.modules.list",
                description="List MCP modules visible to the current user.",
                parameters={"properties": {}, "required": []},
                metadata={"category": "discovery", "catalog_exempt": True},
            ),
            create_tool_definition(
                name="mcp.tools.list",
                description="List MCP tools with optional catalog/module filters.",
                parameters={
                    "properties": {
                        "catalog": {"type": "string", "description": "Catalog name."},
                        "catalog_id": {"type": "integer", "description": "Catalog id."},
                        "catalog_strict": {
                            "type": "boolean",
                            "description": "Return empty tool list when catalog resolution fails.",
                        },
                        "module": {"type": "string", "description": "Single module id filter."},
                        "modules": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Multiple module ids filter.",
                        },
                    },
                    "required": [],
                },
                metadata={"category": "discovery", "catalog_exempt": True},
            ),
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        args = self.sanitize_input(arguments or {})
        if context is None:
            raise ValueError("Request context is required for MCP discovery tools")
        if tool_name == "mcp.catalogs.list":
            return await self._list_catalogs(args, context)
        if tool_name == "mcp.modules.list":
            return await self._list_modules(context)
        if tool_name == "mcp.tools.list":
            return await self._list_tools(args, context)
        raise ValueError(f"Unknown tool: {tool_name}")

    async def _list_modules(self, context: RequestContext) -> dict[str, Any]:
        protocol = MCPProtocol()
        return await protocol._handle_modules_list({}, context)

    async def _list_tools(self, args: dict[str, Any], context: RequestContext) -> dict[str, Any]:
        params: dict[str, Any] = {}

        catalog = args.get("catalog")
        if isinstance(catalog, str) and catalog.strip():
            params["catalog"] = catalog.strip()

        catalog_id = args.get("catalog_id")
        if catalog_id is not None:
            with contextlib.suppress(TypeError, ValueError):
                params["catalog_id"] = int(catalog_id)

        catalog_strict = args.get("catalog_strict")
        if isinstance(catalog_strict, bool):
            params["catalog_strict"] = catalog_strict
        elif isinstance(catalog_strict, (int, float)):
            params["catalog_strict"] = bool(catalog_strict)
        elif isinstance(catalog_strict, str):
            params["catalog_strict"] = is_truthy(catalog_strict)

        modules: list[str] = []
        module_single = args.get("module")
        if isinstance(module_single, str) and module_single.strip():
            modules.append(module_single.strip())
        elif isinstance(module_single, list):
            modules.extend([str(m).strip() for m in module_single if str(m).strip()])

        module_list = args.get("modules")
        if isinstance(module_list, list):
            modules.extend([str(m).strip() for m in module_list if str(m).strip()])
        elif isinstance(module_list, str) and module_list.strip():
            modules.append(module_list.strip())

        if modules:
            unique = list(dict.fromkeys(modules))
            params["module"] = unique if len(unique) > 1 else unique[0]

        protocol = MCPProtocol()
        return await protocol._handle_tools_list(params, context)

    async def _list_catalogs(self, args: dict[str, Any], context: RequestContext) -> dict[str, Any]:
        scope = str(args.get("scope") or "all").strip().lower()
        if scope not in {"all", "global", "org", "team"}:
            raise ValueError("Invalid scope. Use one of: all, global, org, team.")

        pool = await get_db_pool()

        admin_all = await self._is_admin(context, pool)
        org_ids, team_ids = await self._resolve_memberships(context, admin_all)

        catalogs: dict[str, list[dict[str, Any]]] = {"global": [], "org": [], "team": []}

        async def _add_rows(rows: list[Any], scope_name: str) -> None:
            for row in rows or []:
                entry = self._catalog_from_row(row, scope_name)
                if entry:
                    catalogs[scope_name].append(entry)

        if scope in {"all", "global"}:
            rows = await pool.fetchall(
                "SELECT id, name, description, org_id, team_id, COALESCE(is_active,1), created_at, updated_at "
                "FROM tool_catalogs WHERE org_id IS NULL AND team_id IS NULL ORDER BY created_at DESC"
            )
            await _add_rows(rows, "global")

        if scope in {"all", "org"}:
            if admin_all:
                rows = await pool.fetchall(
                    "SELECT id, name, description, org_id, team_id, COALESCE(is_active,1), created_at, updated_at "
                    "FROM tool_catalogs WHERE org_id IS NOT NULL AND team_id IS NULL ORDER BY created_at DESC"
                )
                await _add_rows(rows, "org")
            elif org_ids:
                placeholders, params = build_sqlite_in_clause(sorted(org_ids))
                org_ids_clause = f"({placeholders})"
                org_catalogs_sql_template = (
                    "SELECT id, name, description, org_id, team_id, COALESCE(is_active,1), created_at, updated_at "
                    "FROM tool_catalogs WHERE org_id IN {org_ids_clause} AND team_id IS NULL ORDER BY created_at DESC"
                )
                org_catalogs_sql = org_catalogs_sql_template.format_map(locals())  # nosec B608
                rows = await pool.fetchall(
                    org_catalogs_sql,
                    params,
                )
                await _add_rows(rows, "org")

        if scope in {"all", "team"}:
            if admin_all:
                rows = await pool.fetchall(
                    "SELECT id, name, description, org_id, team_id, COALESCE(is_active,1), created_at, updated_at "
                    "FROM tool_catalogs WHERE team_id IS NOT NULL ORDER BY created_at DESC"
                )
                await _add_rows(rows, "team")
            elif team_ids:
                placeholders, params = build_sqlite_in_clause(sorted(team_ids))
                team_ids_clause = f"({placeholders})"
                team_catalogs_sql_template = (
                    "SELECT id, name, description, org_id, team_id, COALESCE(is_active,1), created_at, updated_at "
                    "FROM tool_catalogs WHERE team_id IN {team_ids_clause} ORDER BY created_at DESC"
                )
                team_catalogs_sql = team_catalogs_sql_template.format_map(locals())  # nosec B608
                rows = await pool.fetchall(
                    team_catalogs_sql,
                    params,
                )
                await _add_rows(rows, "team")

        # Deduplicate across scopes by id
        seen: set[int] = set()
        for scope_name in ("global", "org", "team"):
            deduped: list[dict[str, Any]] = []
            for entry in catalogs[scope_name]:
                entry_id = entry.get("id")
                if isinstance(entry_id, int) and entry_id in seen:
                    continue
                if isinstance(entry_id, int):
                    seen.add(entry_id)
                deduped.append(entry)
            catalogs[scope_name] = deduped

        total = sum(len(v) for v in catalogs.values())
        return {"catalogs": catalogs, "count": total}

    async def _resolve_memberships(self, context: RequestContext, admin_all: bool) -> tuple[set[int], set[int]]:
        org_ids: set[int] = set()
        team_ids: set[int] = set()
        metadata = getattr(context, "metadata", {}) or {}

        org_raw = metadata.get("org_id")
        if org_raw is not None:
            with contextlib.suppress(TypeError, ValueError):
                org_ids.add(int(org_raw))

        team_raw = metadata.get("team_id")
        if team_raw is not None:
            with contextlib.suppress(TypeError, ValueError):
                team_ids.add(int(team_raw))

        if admin_all:
            return org_ids, team_ids

        uid = self._coerce_user_id(context.user_id)
        if uid is None:
            return org_ids, team_ids

        try:
            memberships = await list_org_memberships_for_user(uid)
            for m in memberships or []:
                try:
                    org_id = int(m.get("org_id"))
                except (TypeError, ValueError):
                    continue
                org_ids.add(org_id)
        except (OSError, RuntimeError, TypeError, ValueError):
            logger.debug("MCP discovery: org membership lookup failed; details redacted")

        try:
            pool = await get_db_pool()
            rows = await pool.fetchall(
                "SELECT team_id FROM team_members WHERE user_id = ? AND status = 'active'",
                uid,
            )
            for row in rows or []:
                try:
                    val = row["team_id"] if isinstance(row, dict) else row[0]
                    team_ids.add(int(val))
                except (TypeError, ValueError):
                    continue
        except (OSError, RuntimeError, TypeError, ValueError):
            logger.debug("MCP discovery: team membership lookup failed; details redacted")

        return org_ids, team_ids

    async def _is_admin(self, context: RequestContext, pool: Any) -> bool:
        metadata = getattr(context, "metadata", {}) or {}
        roles = metadata.get("roles") or []
        if isinstance(roles, str):
            roles = [roles]
        if any(str(r).lower() == "admin" for r in roles):
            return True

        uid = self._coerce_user_id(context.user_id)
        if uid is None:
            return False
        try:
            row = await pool.fetchone(
                "SELECT 1 FROM user_roles ur JOIN roles r ON r.id = ur.role_id "
                "WHERE ur.user_id = ? AND r.name = 'admin' LIMIT 1",
                uid,
            )
            return row is not None
        except (OSError, RuntimeError, TypeError, ValueError):
            return False

    @staticmethod
    def _coerce_user_id(user_id: Any) -> int | None:
        try:
            return int(user_id)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _serialize_value(val: Any) -> Any:
        if hasattr(val, "isoformat"):
            try:
                return val.isoformat()
            except (OSError, RuntimeError, TypeError, ValueError):
                return str(val)
        return val

    def _catalog_from_row(self, row: Any, scope: str) -> dict[str, Any] | None:
        def _get(key: str, idx: int) -> Any:
            try:
                if isinstance(row, dict):
                    return row.get(key)
                return row[idx]
            except (AttributeError, IndexError, KeyError, TypeError):
                return None

        entry_id = _get("id", 0)
        name = _get("name", 1)
        if entry_id is None or name is None:
            return None
        return {
            "id": int(entry_id),
            "name": str(name),
            "description": _get("description", 2),
            "org_id": _get("org_id", 3),
            "team_id": _get("team_id", 4),
            "is_active": bool(_get("is_active", 5)),
            "created_at": self._serialize_value(_get("created_at", 6)),
            "updated_at": self._serialize_value(_get("updated_at", 7)),
            "scope": scope,
        }
