from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.settings import get_profile, get_settings
from tldw_Server_API.app.core.DB_Management.Users_DB import DatabaseError, UserNotFoundError, UsersDB


@dataclass
class AuthnzUsersRepo:
    """
    Repository for AuthNZ user records.

    This thin wrapper centralizes read access to the ``users`` table so that
    AuthNZ code paths can depend on a small, well-defined surface rather than
    calling ad-hoc helpers. It builds on the existing ``UsersDB`` abstraction
    and shares the same ``DatabasePool`` instance.
    """

    db_pool: DatabasePool

    def _is_postgres_backend(self) -> bool:
        """
        Return True when the underlying DatabasePool is using PostgreSQL.

        Backend routing should rely on DatabasePool state rather than probing
        connection method availability.
        """
        return bool(getattr(self.db_pool, "pool", None))

    @classmethod
    async def from_pool(cls) -> AuthnzUsersRepo:
        """Construct a repository bound to the global AuthNZ DatabasePool."""
        pool = await get_db_pool()
        return cls(db_pool=pool)

    async def _users_db(self) -> UsersDB:
        db = UsersDB(self.db_pool)
        await db.initialize()
        return db

    @staticmethod
    def _normalize_user_record(row: Any) -> dict[str, Any]:
        """
        Normalize backend-specific row types to a consistent dict with
        JSON-friendly UUID strings and primitive types.
        """
        try:
            base = dict(row) if hasattr(row, "keys") or isinstance(row, dict) else dict(row or {})
        except (TypeError, ValueError):
            base = {}
        user_dict: dict[str, Any] = dict(base)
        if "id" in user_dict:
            with contextlib.suppress(TypeError, ValueError):
                user_dict["id"] = int(user_dict["id"])
        if "uuid" in user_dict and user_dict["uuid"] is not None:
            with contextlib.suppress(TypeError, ValueError):
                user_dict["uuid"] = str(user_dict["uuid"])
        for field in ("is_active", "is_verified", "is_superuser"):
            if field in user_dict:
                with contextlib.suppress(TypeError, ValueError):
                    user_dict[field] = bool(user_dict[field])
        if "storage_quota_mb" in user_dict and user_dict["storage_quota_mb"] is not None:
            with contextlib.suppress(TypeError, ValueError):
                user_dict["storage_quota_mb"] = int(user_dict["storage_quota_mb"])
        if "storage_used_mb" in user_dict and user_dict["storage_used_mb"] is not None:
            with contextlib.suppress(TypeError, ValueError):
                user_dict["storage_used_mb"] = float(user_dict["storage_used_mb"])
        return user_dict

    async def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
        """
        Fetch a user row by integer id.

        Returns ``None`` when the user does not exist. All other errors are
        surfaced to callers so they can apply appropriate HTTP semantics.
        """
        db = await self._users_db()
        try:
            row = await db.get_user_by_id(user_id)
            if row is None:
                return None
            return self._normalize_user_record(row)
        except UserNotFoundError:
            return None
        except DatabaseError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"AuthnzUsersRepo.get_user_by_id failed: {exc}")
            raise

    async def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        """Fetch a user row by username."""
        db = await self._users_db()
        try:
            row = await db.get_user_by_username(username)
            if row is None:
                return None
            return self._normalize_user_record(row)
        except DatabaseError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"AuthnzUsersRepo.get_user_by_username failed: {exc}")
            raise

    async def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        """Fetch a user row by email address."""
        db = await self._users_db()
        try:
            row = await db.get_user_by_email(email)
            if row is None:
                return None
            return self._normalize_user_record(row)
        except DatabaseError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"AuthnzUsersRepo.get_user_by_email failed: {exc}")
            raise

    async def create_user(
        self,
        *,
        username: str,
        email: str,
        password_hash: str,
        role: str | None = None,
        is_active: bool = True,
        is_verified: bool = False,
        user_uuid: str | None = None,
    ) -> int:
        """
        Create a new user row, enforcing profile-based invariants.

        In the local-single-user profile, creating additional users beyond the
        bootstrapped admin (SINGLE_USER_FIXED_ID) is forbidden as a hard
        constraint. This helper raises DatabaseError in that profile for any
        attempt to create users.
        """
        settings = get_settings()
        profile = get_profile()
        if isinstance(profile, str) and profile.strip().lower() in {"local-single-user", "single_user"}:
            msg = "User creation is forbidden in local-single-user profile"
            logger.warning(msg)
            raise DatabaseError(msg)

        db = await self._users_db()
        try:
            user_id = await db.create_user(
                username=username,
                email=email,
                password_hash=password_hash,
                role=role or settings.DEFAULT_USER_ROLE,
                is_active=is_active,
                is_verified=is_verified,
                user_uuid=user_uuid,
            )
            return int(user_id)
        except Exception as exc:
            logger.error(f"AuthnzUsersRepo.create_user failed: {exc}")
            raise

    async def get_user_by_uuid(self, user_uuid: str) -> dict[str, Any] | None:
        """Fetch a user row by UUID string."""
        db = await self._users_db()
        try:
            row = await db.get_user_by_uuid(user_uuid)
            if row is None:
                return None
            return self._normalize_user_record(row)
        except DatabaseError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"AuthnzUsersRepo.get_user_by_uuid failed: {exc}")
            raise

    async def list_users(
        self,
        *,
        offset: int,
        limit: int,
        role: str | None = None,
        admin_capable: bool = False,
        is_active: bool | None = None,
        mfa_enabled: bool | None = None,
        search: str | None = None,
        org_ids: list[int] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Return a window of users and the total count matching the filters.

        This mirrors the admin list endpoint semantics (role, is_active,
        username/email search) while encapsulating backend-specific SQL.
        """
        db = await self._users_db()
        # Reuse the underlying DatabasePool directly for count + projection
        is_pg = db._using_postgres_backend()

        conditions: list[str] = []
        params: list[Any] = []
        param_count = 0
        join_clause = ""

        if org_ids is not None:
            if len(org_ids) == 0:
                return [], 0
            join_clause = " JOIN org_members om ON om.user_id = users.id"
            if is_pg:
                param_count += 1
                conditions.append(f"om.org_id = ANY(${param_count})")
                params.append(org_ids)
                conditions.append("om.status = 'active'")
            else:
                placeholders = ", ".join(["?"] * len(org_ids))
                conditions.append(f"om.org_id IN ({placeholders})")
                params.extend(org_ids)
                conditions.append("om.status = 'active'")

        if role:
            param_count += 1
            conditions.append(f"role = ${param_count}" if is_pg else "role = ?")
            params.append(role)

        if admin_capable:
            if is_pg:
                role_pos = param_count + 1
                rbac_pos = param_count + 2
                admin_capable_condition = """(
                        users.role = ${role_pos}
                        OR COALESCE(users.is_superuser, FALSE) = TRUE
                        OR EXISTS (
                            SELECT 1
                            FROM user_roles ur
                            JOIN roles r ON r.id = ur.role_id
                            WHERE ur.user_id = users.id
                              AND r.name = ANY(${rbac_pos})
                        )
                    )"""
                conditions.append(admin_capable_condition.format_map(locals()))  # nosec B608
                params.append("admin")
                params.append(["admin", "owner", "super_admin"])
                param_count += 2
            else:
                placeholders = ", ".join(["?"] * 3)
                admin_capable_condition = """(
                        users.role = ?
                        OR COALESCE(users.is_superuser, 0) = 1
                        OR EXISTS (
                            SELECT 1
                            FROM user_roles ur
                            JOIN roles r ON r.id = ur.role_id
                            WHERE ur.user_id = users.id
                              AND r.name IN ({placeholders})
                        )
                    )"""
                conditions.append(admin_capable_condition.format_map(locals()))  # nosec B608
                params.extend(["admin", "admin", "owner", "super_admin"])

        if is_active is not None:
            param_count += 1
            conditions.append(f"is_active = ${param_count}" if is_pg else "is_active = ?")
            params.append(is_active)

        if search:
            param_count += 1
            search_pattern = f"%{search}%"
            if is_pg:
                conditions.append(f"(username ILIKE ${param_count} OR email ILIKE ${param_count})")
                params.append(search_pattern)
            else:
                conditions.append("(username LIKE ? OR email LIKE ?)")
                params.append(search_pattern)
                params.append(search_pattern)

        if mfa_enabled is not None:
            if is_pg:
                if mfa_enabled:
                    conditions.append("COALESCE(users.two_factor_enabled, FALSE) = TRUE")
                else:
                    conditions.append("COALESCE(users.two_factor_enabled, FALSE) = FALSE")
            else:
                if mfa_enabled:
                    conditions.append("COALESCE(users.two_factor_enabled, 0) = 1")
                else:
                    conditions.append("COALESCE(users.two_factor_enabled, 0) = 0")

        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""

        try:
            # Total count
            count_query_template = "SELECT COUNT(DISTINCT users.id) FROM users{join_clause}{where_clause}"
            count_query = count_query_template.format_map(locals())  # nosec B608
            total = await db.db_pool.fetchval(count_query, *params)

            # Page of users
            if is_pg:
                limit_pos = param_count + 1
                offset_pos = param_count + 2
                query_template = """
                    SELECT DISTINCT users.id, users.uuid, users.username, users.email, users.role, users.is_active, users.is_verified,
                           users.two_factor_enabled, created_at, last_login, storage_quota_mb, storage_used_mb
                    FROM users{join_clause}{where_clause}
                    ORDER BY users.created_at DESC
                    LIMIT ${limit_pos} OFFSET ${offset_pos}
                """
                query = query_template.format_map(locals())  # nosec B608
                q_params = [*params, limit, offset]
                rows = await db.db_pool.fetch(query, *q_params)
            else:
                query_template = """
                    SELECT DISTINCT users.id, users.uuid, users.username, users.email, users.role, users.is_active, users.is_verified,
                           users.two_factor_enabled, created_at, last_login, storage_quota_mb, storage_used_mb
                    FROM users{join_clause}{where_clause}
                    ORDER BY users.created_at DESC
                    LIMIT ? OFFSET ?
                """
                query = query_template.format_map(locals())  # nosec B608
                q_params = [*params, limit, offset]
                rows = await db.db_pool.fetchall(query, *q_params)

            users: list[dict[str, Any]] = []
            for row in rows:
                if hasattr(row, "keys") or isinstance(row, dict):
                    r = dict(row)
                    user_dict = {
                        "id": int(r.get("id")),
                        "uuid": str(r.get("uuid")) if r.get("uuid") is not None else None,
                        "username": r.get("username"),
                        "email": r.get("email"),
                        "role": r.get("role"),
                        "is_active": bool(r.get("is_active")),
                        "is_verified": bool(r.get("is_verified")),
                        "mfa_enabled": bool(r.get("two_factor_enabled")),
                        "created_at": r.get("created_at"),
                        "last_login": r.get("last_login"),
                        "storage_quota_mb": int(r.get("storage_quota_mb") or 0),
                        "storage_used_mb": float(r.get("storage_used_mb") or 0.0),
                    }
                else:
                    user_dict = {
                        "id": int(row[0]),
                        "uuid": str(row[1]) if row[1] is not None else None,
                        "username": row[2],
                        "email": row[3],
                        "role": row[4],
                        "is_active": bool(row[5]),
                        "is_verified": bool(row[6]),
                        "mfa_enabled": bool(row[7]),
                        "created_at": row[8],
                        "last_login": row[9],
                        "storage_quota_mb": int(row[10] or 0),
                        "storage_used_mb": float(row[11] or 0.0),
                    }
                users.append(user_dict)

            return users, int(total or 0)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"AuthnzUsersRepo.list_users failed: {exc}")
            raise

    async def ensure_single_user_admin_user(
        self,
        *,
        user_id: int,
        username: str = "single_user",
        email: str = "single_user@example.local",
        password_hash: str = "",
    ) -> None:
        """
        Ensure the bootstrapped single-user admin row exists and is active.

        This helper centralizes backend-specific upsert/update SQL used by
        initialization and single-user seed paths.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    await conn.execute(
                        """
                        INSERT INTO users (id, username, email, password_hash, is_active, is_verified, role)
                        VALUES ($1, $2, $3, $4, TRUE, TRUE, 'admin')
                        ON CONFLICT (id) DO NOTHING
                        """,
                        int(user_id),
                        str(username),
                        str(email),
                        str(password_hash),
                    )
                    await conn.execute(
                        "UPDATE users SET role = 'admin', is_active = TRUE, is_verified = TRUE WHERE id = $1",
                        int(user_id),
                    )
                else:
                    await conn.execute(
                        """
                        INSERT OR IGNORE INTO users (id, username, email, password_hash, is_active, is_verified, role)
                        VALUES (?, ?, ?, ?, 1, 1, 'admin')
                        """,
                        (int(user_id), str(username), str(email), str(password_hash)),
                    )
                    await conn.execute(
                        "UPDATE users SET role = 'admin', is_active = 1, is_verified = 1 WHERE id = ?",
                        (int(user_id),),
                    )
                    # sqlite transaction shims may require explicit commit
                    await conn.commit()
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzUsersRepo.ensure_single_user_admin_user failed: {exc}")
            raise

    async def assign_role_if_missing(
        self,
        *,
        user_id: int,
        role_name: str,
    ) -> None:
        """
        Ensure the user has the specified role assignment in ``user_roles``.

        No-op when the role does not exist.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    role_id = await conn.fetchval(
                        "SELECT id FROM roles WHERE name = $1",
                        str(role_name),
                    )
                    if role_id is None:
                        return
                    await conn.execute(
                        "INSERT INTO user_roles (user_id, role_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                        int(user_id),
                        int(role_id),
                    )
                    return

                cursor = await conn.execute(
                    "SELECT id FROM roles WHERE name = ?",
                    (str(role_name),),
                )
                row = await cursor.fetchone()
                if not row:
                    return
                role_id = int(row[0])
                await conn.execute(
                    "INSERT OR IGNORE INTO user_roles (user_id, role_id) VALUES (?, ?)",
                    (int(user_id), role_id),
                )
                await conn.commit()
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzUsersRepo.assign_role_if_missing failed: {exc}")
            raise

    async def has_role_assignment(
        self,
        *,
        user_id: int,
        role_name: str,
    ) -> bool:
        """Return True when the user currently has the named role assignment."""
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    exists = await conn.fetchval(
                        """
                        SELECT 1
                        FROM user_roles ur
                        JOIN roles r ON r.id = ur.role_id
                        WHERE ur.user_id = $1
                          AND r.name = $2
                          AND (ur.expires_at IS NULL OR ur.expires_at > CURRENT_TIMESTAMP)
                        LIMIT 1
                        """,
                        int(user_id),
                        str(role_name),
                    )
                    return bool(exists)

                cursor = await conn.execute(
                    """
                    SELECT 1
                    FROM user_roles ur
                    JOIN roles r ON r.id = ur.role_id
                    WHERE ur.user_id = ?
                      AND r.name = ?
                      AND (ur.expires_at IS NULL OR ur.expires_at > CURRENT_TIMESTAMP)
                    LIMIT 1
                    """,
                    (int(user_id), str(role_name)),
                )
                return await cursor.fetchone() is not None
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzUsersRepo.has_role_assignment failed: {exc}")
            raise

    async def remove_role_if_present(
        self,
        *,
        user_id: int,
        role_name: str,
    ) -> bool:
        """Remove the named user role assignment when present."""
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    row = await conn.fetchrow(
                        """
                        DELETE FROM user_roles ur
                        USING roles r
                        WHERE ur.role_id = r.id
                          AND ur.user_id = $1
                          AND r.name = $2
                        RETURNING ur.user_id
                        """,
                        int(user_id),
                        str(role_name),
                    )
                    return row is not None

                cursor = await conn.execute(
                    "SELECT id FROM roles WHERE name = ?",
                    (str(role_name),),
                )
                row = await cursor.fetchone()
                if not row:
                    return False
                role_id = int(row[0])
                delete_cursor = await conn.execute(
                    """
                    DELETE FROM user_roles
                    WHERE user_id = ? AND role_id = ?
                    """,
                    (int(user_id), role_id),
                )
                await conn.commit()
                try:
                    return bool((delete_cursor.rowcount or 0) > 0)
                except AttributeError:
                    cursor = await conn.execute(
                        """
                        SELECT 1
                        FROM user_roles
                        WHERE user_id = ? AND role_id = ?
                        LIMIT 1
                        """,
                        (int(user_id), role_id),
                    )
                    return await cursor.fetchone() is None
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzUsersRepo.remove_role_if_present failed: {exc}")
            raise
