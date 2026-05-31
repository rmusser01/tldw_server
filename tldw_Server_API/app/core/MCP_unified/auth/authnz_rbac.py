"""
AuthNZ-backed RBAC adapter for MCP Unified.

Uses the project's AuthNZ roles/permissions tables instead of in-memory roles.
"""

from __future__ import annotations

import contextlib
from functools import lru_cache

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool

from .rbac import Action, Resource


@lru_cache(maxsize=256)
def _map_to_permission(resource: Resource, action: Action, resource_id: str | None = None) -> str | None:
    """Map MCP Resource/Action to AuthNZ permission code.

    Returns a dotted permission string matching AuthNZ.permissions.name or None if unmapped.
    """
    base = None
    if resource == Resource.MEDIA:
        base = "media"
    elif resource == Resource.USER:
        base = "users"
    elif resource == Resource.SETTINGS:
        base = "system"
    elif resource == Resource.TOOL:
        # Treat executing tools as general read capability for now (extend as needed)
        base = "media"
    elif resource == Resource.PROMPT:
        base = "prompts"
    elif resource == Resource.NOTE:
        base = "notes"
    elif resource == Resource.CONVERSATION:
        base = "conversations"
    elif resource == Resource.TRANSCRIPT:
        base = "transcripts"
    elif resource == Resource.RESOURCE:
        base = "resources"
    elif resource == Resource.MODULE:
        base = "modules"
    else:
        base = None

    if not base:
        return None

    if action == Action.CREATE:
        return f"{base}.create"
    if action == Action.READ:
        return f"{base}.read"
    if action == Action.UPDATE:
        return f"{base}.update"
    if action == Action.DELETE:
        return f"{base}.delete"
    if action == Action.EXECUTE:
        # For tools, use specific permission name tools.execute:<tool_name>
        if resource == Resource.TOOL and resource_id:
            return f"tools.execute:{resource_id}"
        # Otherwise treat execute as read baseline
        return f"{base}.read"
    if action == Action.ADMIN:
        return "system.configure"
    return None


class AuthNZRBAC:
    """RBAC checker that consults AuthNZ DB (roles, permissions, overrides)."""

    def __init__(self, db_pool: DatabasePool | None = None):
        self._pool = db_pool

    async def _get_pool(self) -> DatabasePool:
        if self._pool:
            return self._pool
        self._pool = await get_db_pool()
        return self._pool

    async def check_permission(
        self,
        user_id: str | None,
        resource: Resource,
        action: Action,
        resource_id: str | None = None,
    ) -> bool:
        if not user_id:
            return False
        perm = _map_to_permission(resource, action, resource_id)
        if not perm:
            logger.warning(
                "Denied MCP permission check due to missing mapping",
                resource=resource.value,
                action=action.value,
                resource_id=resource_id,
            )
            return False

        pool = await self._get_pool()
        uid = int(user_id) if isinstance(user_id, str) and user_id.isdigit() else user_id
        try:
            # Admin bypass: if user has admin role, allow
            is_admin = await pool.fetchone(
                """
                SELECT 1 FROM user_roles ur
                JOIN roles r ON r.id = ur.role_id
                WHERE ur.user_id = ? AND r.name = 'admin'
                LIMIT 1
                """,
                uid,
            )
            if is_admin is not None:
                return True

            # Explicit user permission overrides
            row = await pool.fetchone(
                """
                SELECT up.granted FROM user_permissions up
                JOIN permissions p ON p.id = up.permission_id
                WHERE up.user_id = ? AND p.name = ?
                """,
                uid, perm,
            )
            if row is not None:
                return bool(row.get("granted", 0))

            # Explicit user wildcard for tools.execute:*
            if resource == Resource.TOOL and action == Action.EXECUTE:
                row_wildcard = await pool.fetchone(
                    """
                    SELECT up.granted FROM user_permissions up
                    JOIN permissions p ON p.id = up.permission_id
                    WHERE up.user_id = ? AND p.name = ?
                    """,
                    uid, "tools.execute:*",
                )
                if row_wildcard is not None:
                    return bool(row_wildcard.get("granted", 0))

            # Role-based permissions
            row2 = await pool.fetchone(
                """
                SELECT 1 FROM user_roles ur
                JOIN role_permissions rp ON rp.role_id = ur.role_id
                JOIN permissions p ON p.id = rp.permission_id
                WHERE ur.user_id = ? AND p.name = ?
                LIMIT 1
                """,
                uid, perm,
            )
            if row2 is not None:
                return True

            # Fallback: wildcard permission for tools.execute:*
            if resource == Resource.TOOL and action == Action.EXECUTE:
                wildcard = "tools.execute:*"
                row3 = await pool.fetchone(
                    """
                    SELECT 1 FROM user_roles ur
                    JOIN role_permissions rp ON rp.role_id = ur.role_id
                    JOIN permissions p ON p.id = rp.permission_id
                    WHERE ur.user_id = ? AND p.name = ?
                    LIMIT 1
                    """,
                    uid, wildcard,
                )
                if row3 is not None:
                    return True
            return False
        except Exception as e:
            logger.debug("AuthNZ RBAC check failed ({})", type(e).__name__)
            # Fail safe: deny if DB unavailable
            return False

    async def _ensure_permission_exists(self, name: str, description: str = "", category: str = "tools") -> None:
        """Insert permission into catalog if missing (idempotent)."""
        try:
            pool = await self._get_pool()
            row = await pool.fetchone("SELECT id FROM permissions WHERE name = ?", name)
            if row is None:
                await pool.execute(
                    "INSERT INTO permissions (name, description, category) VALUES (?, ?, ?)",
                    name, description, category,
                )
        except Exception as e:
            logger.debug("Ensure permission exists failed ({})", type(e).__name__)


_authnz_rbac: AuthNZRBAC | None = None


def get_rbac_policy() -> AuthNZRBAC:
    global _authnz_rbac
    if _authnz_rbac is None:
        _authnz_rbac = AuthNZRBAC()
    return _authnz_rbac


def reset_rbac_policy() -> None:
    """Reset cached RBAC policy (used in tests when DB/config changes)."""
    global _authnz_rbac
    _authnz_rbac = None
    with contextlib.suppress(Exception):
        _map_to_permission.cache_clear()
