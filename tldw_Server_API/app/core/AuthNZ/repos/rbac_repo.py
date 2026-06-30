from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TypedDict

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.db_config import get_configured_user_database
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.UserDatabase_v2 import UserDatabase
from tldw_Server_API.app.core.exceptions import ResourceNotFoundError


class RolePermissionsResult(TypedDict):
    role_name: str
    permissions: list[str]
    tool_permissions: list[str]
    all_permissions: list[str]


@dataclass
class AuthnzRbacRepo:
    """
    Repository facade for RBAC permission lookups.

    This wrapper centralizes calls into ``UserDatabase_v2`` so that higher-level
    helpers depend on a small, testable surface instead of constructing their
    own database handles.
    """

    client_id: str = "rbac_service"

    @cached_property
    def _db(self) -> UserDatabase:
        """
        Resolve and cache the UserDatabase via the central configuration helper.

        RBAC lookups are frequent, so caching the database handle per-repo
        instance avoids repeated construction overhead. Tests that need to
        exercise different AUTH_MODE or DATABASE_URL configurations should
        construct a fresh AuthnzRbacRepo instance to obtain a new backend.
        """
        return get_configured_user_database(client_id=self.client_id)

    def get_effective_permissions(self, user_id: int) -> list[str]:
        """
        Return the effective permission codes for the given user.

        This delegates to the configured RBAC backend (SQLite/Postgres) via
        ``UserDatabase_v2``.
        """
        db = self._db
        try:
            return db.get_user_permissions(user_id)
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                'AuthnzRbacRepo.get_effective_permissions failed for user_id={}: {}',
                user_id,
                exc,
            )
            raise

    def has_permission(self, user_id: int, permission: str) -> bool:
        """Return True when the RBAC backend reports the permission as allowed."""
        db = self._db
        try:
            return db.has_permission(user_id, permission)
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                'AuthnzRbacRepo.has_permission failed for user_id={} perm={}: {}',
                user_id,
                permission,
                exc,
            )
            raise

    def get_user_roles(self, user_id: int) -> list[dict]:
        """
        Return active roles for a user.

        This wraps the common join between ``roles`` and ``user_roles`` and
        normalizes backend differences so callers do not need to issue their
        own SQL.
        """
        db = self._db
        try:
            if getattr(db.backend, "backend_type", None) == BackendType.POSTGRESQL:
                query = """
                SELECT
                    r.id,
                    r.name,
                    r.description,
                    COALESCE(r.is_system, FALSE) AS is_system
                FROM roles r
                JOIN user_roles ur ON r.id = ur.role_id
                WHERE ur.user_id = ?
                  AND (ur.expires_at IS NULL OR ur.expires_at > CURRENT_TIMESTAMP)
                ORDER BY r.name
                """
            else:
                query = """
                SELECT
                    r.id,
                    r.name,
                    r.description,
                    COALESCE(r.is_system, 0) AS is_system
                FROM roles r
                JOIN user_roles ur ON r.id = ur.role_id
                WHERE ur.user_id = ?
                  AND (ur.expires_at IS NULL OR ur.expires_at > CURRENT_TIMESTAMP)
                ORDER BY r.name
                """
            result = db.backend.execute(
                query,
                (int(user_id),),
            )
            return [dict(row) for row in result.rows]
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                'AuthnzRbacRepo.get_user_roles failed for user_id={}: {}',
                user_id,
                exc,
            )
            raise

    def get_user_overrides(self, user_id: int) -> list[dict]:
        """
        Return user-specific permission overrides.

        Each row includes:
        - permission_id
        - permission_name
        - granted (0/1 or bool)
        - expires_at (backend-native representation)
        """
        db = self._db
        try:
            result = db.backend.execute(
                """
                SELECT
                    p.id AS permission_id,
                    p.name AS permission_name,
                    up.granted,
                    up.expires_at
                FROM user_permissions up
                JOIN permissions p ON up.permission_id = p.id
                WHERE up.user_id = ?
                ORDER BY p.name
                """,
                (int(user_id),),
            )
            return [dict(row) for row in result.rows]
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                'AuthnzRbacRepo.get_user_overrides failed for user_id={}: {}',
                user_id,
                exc,
            )
            raise

    def get_role_effective_permissions(self, role_id: int) -> RolePermissionsResult:
        """
        Return effective permissions for a role, split into regular and tool permissions.

        The response shape mirrors the admin API:
        - role_name
        - permissions
        - tool_permissions
        - all_permissions
        """
        db = self._db
        try:
            # Fetch role information
            role_rows = db.backend.execute(
                "SELECT id, name FROM roles WHERE id = ?",
                (int(role_id),),
            )
            if not role_rows.rows:
                raise ResourceNotFoundError("role", identifier=str(role_id), detail="role_not_found")
            role_name = str(role_rows.rows[0]["name"])

            # Fetch permission names for this role
            perm_rows = db.backend.execute(
                """
                SELECT p.name
                FROM permissions p
                JOIN role_permissions rp ON p.id = rp.permission_id
                WHERE rp.role_id = ?
                ORDER BY p.name
                """,
                (int(role_id),),
            )
            names = [str(r["name"]) for r in perm_rows.rows]

            tool_prefix = "tools.execute:"
            tool_permissions = [n for n in names if n.startswith(tool_prefix)]
            permissions = [n for n in names if not n.startswith(tool_prefix)]
            all_permissions = sorted(tool_permissions + permissions)

            return {
                "role_name": role_name,
                "permissions": permissions,
                "tool_permissions": tool_permissions,
                "all_permissions": all_permissions,
            }
        except ResourceNotFoundError:
            # Preserve not-found contract for callers that distinguish role-not-found
            raise
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                'AuthnzRbacRepo.get_role_effective_permissions failed for role_id={}: {}',
                role_id,
                exc,
            )
            raise

    def get_role_id_by_name(self, role_name: str) -> int | None:
        """
        Look up a role id by its name.

        This helper centralizes the roles table access so callers do not need to
        embed backend-specific SQL.
        """
        db = self._db
        try:
            result = db.backend.execute(
                "SELECT id FROM roles WHERE name = ?",
                (str(role_name),),
            )
            if not result.rows:
                return None
            return int(result.rows[0]["id"])
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                'AuthnzRbacRepo.get_role_id_by_name failed for role_name={}: {}',
                role_name,
                exc,
            )
            raise
