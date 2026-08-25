from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    DuplicateOrganizationError,
    DuplicateTeamError,
)
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    AnchorOwnership,
    MembershipMutation,
    MembershipMutationKind,
    MembershipMutationRelationship,
    MembershipParentRequired,
    MembershipReadError,
    MembershipScopeType,
    MembershipWriteContext,
    MembershipWriter,
    MembershipWriteResult,
)
from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway
from tldw_Server_API.app.core.AuthNZ.transaction_policy import (
    get_authnz_transaction_policy,
)

DEFAULT_BASE_TEAM_NAME = "Default-Base"
DEFAULT_BASE_TEAM_SLUG = "default-base"
DEFAULT_BASE_TEAM_DESCRIPTION = (
    "Automatically managed base team for organization-wide membership."
)


@dataclass
class AuthnzOrgsTeamsRepo:
    """
    Repository for organizations, teams, and membership.

    This repo encapsulates common read/write paths so higher-level orgs/teams
    helpers do not need to embed backend-specific SQL for Postgres vs SQLite.
    """

    db_pool: DatabasePool

    def _is_postgres(self, conn: Any | None = None) -> bool:
        """
        Detect whether the configured backend is PostgreSQL from pool state.
        """
        _ = conn  # Compatibility placeholder for legacy call sites.
        return bool(getattr(self.db_pool, "pool", None))

    def _membership_transaction(self) -> Any:
        """Open a membership transaction with the shared acquisition bound."""

        policy = get_authnz_transaction_policy()
        return self.db_pool.transaction(
            acquire_timeout_seconds=policy.db_pool_acquire_timeout_seconds,
        )

    async def _apply_direct_membership_mutations(
        self,
        *,
        context: MembershipWriteContext,
        mutations: tuple[MembershipMutation, ...],
        operation_time: datetime | None = None,
    ) -> MembershipWriteResult:
        """Own one bounded transaction and one final anchor touch per changed user."""

        sampled_time = operation_time or datetime.now(timezone.utc)
        async with self._membership_transaction() as conn:
            return await MembershipWriter(self.db_pool).apply_membership_mutations(
                conn=conn,
                context=context,
                mutations=mutations,
                anchor_ownership=AnchorOwnership.WRITER_OWNS_ANCHOR,
                operation_time=sampled_time,
            )

    async def _final_touch_membership_results(
        self,
        conn: Any,
        *,
        results: tuple[MembershipWriteResult, ...],
        operation_time: datetime,
    ) -> None:
        floors: dict[int, datetime] = {}
        for result in results:
            for floor in result.version_floors:
                floors[floor.user_id] = max(
                    floors.get(floor.user_id, floor.version_floor),
                    floor.version_floor,
                )
        gateway = VersionedUserWriteGateway(
            "postgres" if self._is_postgres() else "sqlite",
            clock=lambda: operation_time,
        )
        for user_id in sorted(floors):
            await gateway.final_touch(
                conn,
                user_id=user_id,
                version_floor=floors[user_id],
            )

    async def create_organization(
        self,
        *,
        name: str,
        owner_user_id: int | None = None,
        slug: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create an organization row with basic duplicate checks.

        Mirrors the behavior of ``create_organization`` in ``orgs_teams`` but
        centralizes the dialect-specific SQL.
        """
        import json

        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    # PostgreSQL path
                    if slug is not None and slug != "":
                        exists_slug = await conn.fetchrow(
                            "SELECT 1 FROM organizations WHERE LOWER(slug) = LOWER($1)",
                            slug,
                        )
                        if exists_slug:
                            raise DuplicateOrganizationError("slug", str(slug))
                    exists_name = await conn.fetchrow(
                        "SELECT 1 FROM organizations WHERE LOWER(name) = LOWER($1)",
                        name,
                    )
                    if exists_name:
                        raise DuplicateOrganizationError("name", str(name))
                    row = await conn.fetchrow(
                        """
                        INSERT INTO organizations (name, slug, owner_user_id, metadata)
                        VALUES ($1, $2, $3, $4)
                        RETURNING id, name, slug, owner_user_id, is_active, created_at, updated_at
                        """,
                        name,
                        slug,
                        owner_user_id,
                        (metadata if metadata is not None else None),
                    )
                    d = dict(row)
                    try:
                        from datetime import datetime

                        if isinstance(d.get("created_at"), datetime):
                            d["created_at"] = d["created_at"].isoformat()
                        if isinstance(d.get("updated_at"), datetime):
                            d["updated_at"] = d["updated_at"].isoformat()
                    except (TypeError, ValueError, AttributeError) as exc:
                        logger.debug(f"Skipping datetime normalization for org row: {exc}")
                    return d

                # SQLite / aiosqlite path
                if slug is not None and slug != "":
                    cur_chk = await conn.execute(
                        "SELECT 1 FROM organizations WHERE LOWER(slug) = LOWER(?)",
                        (slug,),
                    )
                    if await cur_chk.fetchone():
                        raise DuplicateOrganizationError("slug", str(slug))
                cur_chk2 = await conn.execute(
                    "SELECT 1 FROM organizations WHERE LOWER(name) = LOWER(?)",
                    (name,),
                )
                if await cur_chk2.fetchone():
                    raise DuplicateOrganizationError("name", str(name))
                cur = await conn.execute(
                    "INSERT INTO organizations (name, slug, owner_user_id, metadata) VALUES (?, ?, ?, ?)",
                    (
                        name,
                        slug,
                        owner_user_id,
                        json.dumps(metadata) if metadata else None,
                    ),
                )
                org_id = cur.lastrowid
                cur2 = await conn.execute(
                    """
                    SELECT id, name, slug, owner_user_id, is_active, created_at, updated_at
                    FROM organizations
                    WHERE id = ?
                    """,
                    (org_id,),
                )
                row = await cur2.fetchone()
                return {
                    "id": row[0],
                    "name": row[1],
                    "slug": row[2],
                    "owner_user_id": row[3],
                    "is_active": bool(row[4]),
                    "created_at": row[5],
                    "updated_at": row[6],
                }
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzOrgsTeamsRepo.create_organization failed: {exc}")
            raise

    async def create_team(
        self,
        *,
        org_id: int,
        name: str,
        slug: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a team row with per-organization duplicate checks.

        Mirrors the behavior of ``create_team`` in ``orgs_teams``.
        """
        import json

        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    exists = await conn.fetchrow(
                        "SELECT 1 FROM teams WHERE org_id = $1 AND LOWER(name) = LOWER($2)",
                        org_id,
                        name,
                    )
                    if exists:
                        raise DuplicateTeamError(org_id, "name", str(name))
                    row = await conn.fetchrow(
                        """
                        INSERT INTO teams (org_id, name, slug, description, metadata)
                        VALUES ($1, $2, $3, $4, $5)
                        RETURNING id, org_id, name, slug, description, is_active, created_at, updated_at
                        """,
                        org_id,
                        name,
                        slug,
                        description,
                        (metadata if metadata is not None else None),
                    )
                    d = dict(row)
                    try:
                        from datetime import datetime

                        if isinstance(d.get("created_at"), datetime):
                            d["created_at"] = d["created_at"].isoformat()
                        if isinstance(d.get("updated_at"), datetime):
                            d["updated_at"] = d["updated_at"].isoformat()
                    except (TypeError, ValueError, AttributeError) as exc:
                        logger.debug(f"Skipping datetime normalization for team row: {exc}")
                    return d

                # SQLite path
                curx = await conn.execute(
                    "SELECT 1 FROM teams WHERE org_id = ? AND LOWER(name) = LOWER(?)",
                    (org_id, name),
                )
                if await curx.fetchone():
                    raise DuplicateTeamError(org_id, "name", str(name))
                cur = await conn.execute(
                    "INSERT INTO teams (org_id, name, slug, description, metadata) VALUES (?, ?, ?, ?, ?)",
                    (
                        org_id,
                        name,
                        slug,
                        description,
                        json.dumps(metadata) if metadata else None,
                    ),
                )
                team_id = cur.lastrowid
                cur2 = await conn.execute(
                    """
                    SELECT id, org_id, name, slug, description, is_active, created_at, updated_at
                    FROM teams
                    WHERE id = ?
                    """,
                    (team_id,),
                )
                row = await cur2.fetchone()
                return {
                    "id": row[0],
                    "org_id": row[1],
                    "name": row[2],
                    "slug": row[3],
                    "description": row[4],
                    "is_active": bool(row[5]),
                    "created_at": row[6],
                    "updated_at": row[7],
                }
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzOrgsTeamsRepo.create_team failed: {exc}")
            raise

    async def list_organizations(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        q: str | None = None,
        org_ids: list[int] | None = None,
        with_total: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        List organizations with optional server-side filtering and total count.

        Returns (rows, total).
        """
        try:
            if org_ids is not None and len(org_ids) == 0:
                return [], 0
            if self._is_postgres():
                # Postgres path
                conditions: list[str] = []
                params: list[Any] = []
                param_count = 0
                if org_ids is not None:
                    param_count += 1
                    conditions.append(f"id = ANY(${param_count})")
                    params.append(org_ids)
                if q:
                    param_count += 1
                    like = f"%{str(q).lower()}%"
                    conditions.append(
                        f"(LOWER(name) LIKE ${param_count} OR LOWER(COALESCE(slug, '')) LIKE ${param_count} OR CAST(id AS TEXT) LIKE ${param_count})"
                    )
                    params.append(like)

                where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
                limit_param = param_count + 1
                offset_param = param_count + 2
                rows = await self.db_pool.fetchall(
                    """
                    SELECT id, name, slug, owner_user_id, is_active, created_at, updated_at
                    FROM organizations{where_clause}
                    ORDER BY created_at DESC
                    LIMIT ${limit_param} OFFSET ${offset_param}
                    """.format_map(locals()),  # nosec B608
                    *params,
                    limit,
                    offset,
                )
                total = (
                    await self.db_pool.fetchval(
                        f"SELECT COUNT(*) FROM organizations{where_clause}",  # nosec B608
                        *params,
                    )
                    if with_total
                    else 0
                )

                normalized: list[dict[str, Any]] = []
                for r in rows:
                    d = dict(r)
                    d["is_active"] = bool(d.get("is_active", True))
                    normalized.append(d)
                return normalized, int(total or 0)

            # SQLite / aiosqlite path
            async with self.db_pool.acquire() as conn:
                conditions: list[str] = []
                params: list[Any] = []
                if org_ids is not None:
                    placeholders = ", ".join(["?"] * len(org_ids))
                    conditions.append(f"id IN ({placeholders})")
                    params.extend(org_ids)
                if q:
                    like = f"%{str(q).lower()}%"
                    conditions.append(
                        "(LOWER(name) LIKE ? OR LOWER(COALESCE(slug, '')) LIKE ? OR CAST(id AS TEXT) LIKE ?)"
                    )
                    params.extend([like, like, like])

                where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
                cursor = await conn.execute(
                    """
                    SELECT id, name, slug, owner_user_id, is_active, created_at, updated_at
                    FROM organizations{where_clause}
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """.format_map(locals()),  # nosec B608
                    (*params, limit, offset),
                )
                rows_raw = await cursor.fetchall()
                rows = [
                    {
                        "id": r[0],
                        "name": r[1],
                        "slug": r[2],
                        "owner_user_id": r[3],
                        "is_active": bool(r[4]),
                        "created_at": r[5],
                        "updated_at": r[6],
                    }
                    for r in rows_raw
                ]
                if with_total:
                    cur2 = await conn.execute(
                        f"SELECT COUNT(*) FROM organizations{where_clause}",  # nosec B608
                        params,
                    )
                    total_row = await cur2.fetchone()
                    total = int(total_row[0]) if total_row else 0
                else:
                    total = 0

            return rows, int(total or 0)
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzOrgsTeamsRepo.list_organizations failed: {exc}")
            raise

    async def update_organization(
        self,
        *,
        org_id: int,
        name: str | None = None,
        slug: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Update an organization row.

        Currently supports updating name and slug; additional fields should be
        added here so backend-specific SQL stays encapsulated in the repo.

        Args:
            org_id: Organization ID to update.
            name: New organization name (optional).
            slug: New organization slug (optional).

        Returns:
            Updated organization dict, or None if the organization was not found.

        Raises:
            DuplicateOrganizationError: If name or slug collides with another org.
            ValueError: If no update fields are supplied.
        """
        updates: dict[str, Any] = {}
        if name is not None:
            updates["name"] = name
        if slug is not None:
            updates["slug"] = slug

        if not updates:
            raise ValueError("No fields to update")

        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    if "slug" in updates and updates["slug"] not in (None, ""):
                        exists_slug = await conn.fetchrow(
                            "SELECT 1 FROM organizations WHERE LOWER(slug) = LOWER($1) AND id <> $2",
                            updates["slug"],
                            org_id,
                        )
                        if exists_slug:
                            raise DuplicateOrganizationError("slug", str(updates["slug"]))
                    if "name" in updates:
                        exists_name = await conn.fetchrow(
                            "SELECT 1 FROM organizations WHERE LOWER(name) = LOWER($1) AND id <> $2",
                            updates["name"],
                            org_id,
                        )
                        if exists_name:
                            raise DuplicateOrganizationError("name", str(updates["name"]))

                    set_clause = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates.keys()))
                    params = [org_id] + list(updates.values())
                    row = await conn.fetchrow(
                        """
                        UPDATE organizations
                        SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                        WHERE id = $1
                        RETURNING id, name, slug, owner_user_id, is_active, created_at, updated_at
                        """.format_map(locals()),  # nosec B608
                        *params,
                    )
                    if not row:
                        return None
                    d = dict(row)
                    d["is_active"] = bool(d.get("is_active", True))
                    try:
                        from datetime import datetime

                        for key in ("created_at", "updated_at"):
                            if isinstance(d.get(key), datetime):
                                d[key] = d[key].isoformat()
                    except (TypeError, ValueError, AttributeError) as exc:
                        logger.debug(f"Skipping datetime normalization for org row: {exc}")
                    return d

                if "slug" in updates and updates["slug"] not in (None, ""):
                    cur_chk = await conn.execute(
                        "SELECT 1 FROM organizations WHERE LOWER(slug) = LOWER(?) AND id <> ?",
                        (updates["slug"], org_id),
                    )
                    if await cur_chk.fetchone():
                        raise DuplicateOrganizationError("slug", str(updates["slug"]))
                if "name" in updates:
                    cur_chk2 = await conn.execute(
                        "SELECT 1 FROM organizations WHERE LOWER(name) = LOWER(?) AND id <> ?",
                        (updates["name"], org_id),
                    )
                    if await cur_chk2.fetchone():
                        raise DuplicateOrganizationError("name", str(updates["name"]))

                set_clause = ", ".join(f"{k} = ?" for k in updates)
                params = list(updates.values()) + [org_id]
                await conn.execute(
                    f"UPDATE organizations SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",  # nosec B608
                    tuple(params),
                )
                cur = await conn.execute(
                    "SELECT id, name, slug, owner_user_id, is_active, created_at, updated_at FROM organizations WHERE id = ?",
                    (org_id,),
                )
                row = await cur.fetchone()
                if not row:
                    return None
                return {
                    "id": row[0],
                    "name": row[1],
                    "slug": row[2],
                    "owner_user_id": row[3],
                    "is_active": bool(row[4]),
                    "created_at": row[5],
                    "updated_at": row[6],
                }
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzOrgsTeamsRepo.update_organization failed: {exc}")
            raise

    async def delete_organization_with_provider_secrets(
        self,
        *,
        org_id: int,
    ) -> None:
        """
        Delete an organization and any provider secrets scoped to it or its teams.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    await conn.execute(
                        "DELETE FROM org_provider_secrets WHERE scope_type = 'org' AND scope_id = $1",
                        org_id,
                    )
                    await conn.execute(
                        """
                        DELETE FROM org_provider_secrets
                        WHERE scope_type = 'team'
                          AND scope_id IN (SELECT id FROM teams WHERE org_id = $1)
                        """,
                        org_id,
                    )
                    await conn.execute("DELETE FROM organizations WHERE id = $1", org_id)
                    return

                await conn.execute(
                    "DELETE FROM org_provider_secrets WHERE scope_type = 'org' AND scope_id = ?",
                    (org_id,),
                )
                await conn.execute(
                    """
                    DELETE FROM org_provider_secrets
                    WHERE scope_type = 'team'
                      AND scope_id IN (SELECT id FROM teams WHERE org_id = ?)
                    """,
                    (org_id,),
                )
                await conn.execute("DELETE FROM organizations WHERE id = ?", (org_id,))
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzOrgsTeamsRepo.delete_organization_with_provider_secrets failed: {exc}")
            raise

    async def transfer_organization_ownership(
        self,
        *,
        org_id: int,
        new_owner_user_id: int,
        current_owner_user_id: int,
    ) -> dict[str, Any] | None:
        """
        Transfer organization ownership and update org-member roles atomically.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    await conn.execute(
                        "UPDATE organizations SET owner_user_id = $1, updated_at = CURRENT_TIMESTAMP WHERE id = $2",
                        new_owner_user_id,
                        org_id,
                    )
                    await conn.execute(
                        "UPDATE org_members SET role = 'owner' WHERE org_id = $1 AND user_id = $2",
                        org_id,
                        new_owner_user_id,
                    )
                    await conn.execute(
                        "UPDATE org_members SET role = 'admin' WHERE org_id = $1 AND user_id = $2",
                        org_id,
                        current_owner_user_id,
                    )
                    row = await conn.fetchrow(
                        "SELECT id, name, slug, owner_user_id, is_active, created_at, updated_at FROM organizations WHERE id = $1",
                        org_id,
                    )
                    if not row:
                        return None
                    d = dict(row)
                    d["is_active"] = bool(d.get("is_active", True))
                    try:
                        from datetime import datetime

                        for key in ("created_at", "updated_at"):
                            if isinstance(d.get(key), datetime):
                                d[key] = d[key].isoformat()
                    except (TypeError, ValueError, AttributeError) as exc:
                        logger.debug(f"Skipping datetime normalization for org row: {exc}")
                    return d

                await conn.execute(
                    "UPDATE organizations SET owner_user_id = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (new_owner_user_id, org_id),
                )
                await conn.execute(
                    "UPDATE org_members SET role = 'owner' WHERE org_id = ? AND user_id = ?",
                    (org_id, new_owner_user_id),
                )
                await conn.execute(
                    "UPDATE org_members SET role = 'admin' WHERE org_id = ? AND user_id = ?",
                    (org_id, current_owner_user_id),
                )
                cur = await conn.execute(
                    "SELECT id, name, slug, owner_user_id, is_active, created_at, updated_at FROM organizations WHERE id = ?",
                    (org_id,),
                )
                row = await cur.fetchone()
                if not row:
                    return None
                return {
                    "id": row[0],
                    "name": row[1],
                    "slug": row[2],
                    "owner_user_id": row[3],
                    "is_active": bool(row[4]),
                    "created_at": row[5],
                    "updated_at": row[6],
                }
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzOrgsTeamsRepo.transfer_organization_ownership failed: {exc}")
            raise

    async def update_team(
        self,
        *,
        team_id: int,
        name: str | None = None,
        slug: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Update a team row and return the updated projection.
        """
        updates: dict[str, Any] = {}
        if name is not None:
            updates["name"] = name
        if slug is not None:
            updates["slug"] = slug
        if description is not None:
            updates["description"] = description

        if not updates:
            raise ValueError("No fields to update")

        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    set_clause = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates.keys()))
                    params = [team_id] + list(updates.values())
                    row = await conn.fetchrow(
                        """
                        UPDATE teams
                        SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                        WHERE id = $1
                        RETURNING id, org_id, name, slug, description, is_active, created_at, updated_at
                        """.format_map(locals()),  # nosec B608
                        *params,
                    )
                    if not row:
                        return None
                    d = dict(row)
                    d["is_active"] = bool(d.get("is_active", True))
                    try:
                        from datetime import datetime

                        for key in ("created_at", "updated_at"):
                            if isinstance(d.get(key), datetime):
                                d[key] = d[key].isoformat()
                    except (TypeError, ValueError, AttributeError) as exc:
                        logger.debug(f"Skipping datetime normalization for team row: {exc}")
                    return d

                set_clause = ", ".join(f"{k} = ?" for k in updates)
                params = list(updates.values()) + [team_id]
                await conn.execute(
                    f"UPDATE teams SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",  # nosec B608
                    tuple(params),
                )
                cur = await conn.execute(
                    "SELECT id, org_id, name, slug, description, is_active, created_at, updated_at FROM teams WHERE id = ?",
                    (team_id,),
                )
                row = await cur.fetchone()
                if not row:
                    return None
                return {
                    "id": row[0],
                    "org_id": row[1],
                    "name": row[2],
                    "slug": row[3],
                    "description": row[4],
                    "is_active": bool(row[5]),
                    "created_at": row[6],
                    "updated_at": row[7],
                }
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzOrgsTeamsRepo.update_team failed: {exc}")
            raise

    async def delete_team_with_provider_secrets(
        self,
        *,
        team_id: int,
    ) -> None:
        """
        Delete a team and any team-scoped provider secrets.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    await conn.execute(
                        "DELETE FROM org_provider_secrets WHERE scope_type = 'team' AND scope_id = $1",
                        team_id,
                    )
                    await conn.execute("DELETE FROM teams WHERE id = $1", team_id)
                    return

                await conn.execute(
                    "DELETE FROM org_provider_secrets WHERE scope_type = 'team' AND scope_id = ?",
                    (team_id,),
                )
                await conn.execute("DELETE FROM teams WHERE id = ?", (team_id,))
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzOrgsTeamsRepo.delete_team_with_provider_secrets failed: {exc}")
            raise

    # -------------------------------------------------------------------------
    # Single-record getters
    # -------------------------------------------------------------------------

    async def get_team(self, team_id: int) -> dict[str, Any] | None:
        """
        Get a team by ID.

        Returns team dict with id, org_id, name, slug, description, is_active, etc.
        Returns None if not found.
        """
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres(conn):
                    row = await conn.fetchrow(
                        """
                        SELECT id, org_id, name, slug, description, is_active, created_at, updated_at
                        FROM teams WHERE id = $1
                        """,
                        team_id
                    )
                    if not row:
                        return None
                    d = dict(row)
                    from datetime import datetime
                    for key in ("created_at", "updated_at"):
                        if isinstance(d.get(key), datetime):
                            d[key] = d[key].isoformat()
                    return d
                else:
                    cur = await conn.execute(
                        """
                        SELECT id, org_id, name, slug, description, is_active, created_at, updated_at
                        FROM teams WHERE id = ?
                        """,
                        (team_id,)
                    )
                    row = await cur.fetchone()
                    if not row:
                        return None
                    return {
                        "id": row[0],
                        "org_id": row[1],
                        "name": row[2],
                        "slug": row[3],
                        "description": row[4],
                        "is_active": bool(row[5]),
                        "created_at": row[6],
                        "updated_at": row[7],
                    }
        except Exception as exc:
            logger.error(f"AuthnzOrgsTeamsRepo.get_team failed: {exc}")
            raise

    async def get_org_member(self, org_id: int, user_id: int) -> dict[str, Any] | None:
        """
        Get a specific org membership.

        Returns membership dict with org_id, user_id, role, status, added_at.
        Returns None if user is not a member.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    row = await conn.fetchrow(
                        """
                        SELECT org_id, user_id, role, status, added_at
                        FROM org_members WHERE org_id = $1 AND user_id = $2
                        """,
                        org_id, user_id
                    )
                    if not row:
                        return None
                    d = dict(row)
                    from datetime import datetime
                    if isinstance(d.get("added_at"), datetime):
                        d["added_at"] = d["added_at"].isoformat()
                    return d
                else:
                    cur = await conn.execute(
                        """
                        SELECT org_id, user_id, role, status, added_at
                        FROM org_members WHERE org_id = ? AND user_id = ?
                        """,
                        (org_id, user_id)
                    )
                    row = await cur.fetchone()
                    if not row:
                        return None
                    return {
                        "org_id": row[0],
                        "user_id": row[1],
                        "role": row[2],
                        "status": row[3],
                        "added_at": row[4],
                    }
        except Exception as exc:
            logger.error(f"AuthnzOrgsTeamsRepo.get_org_member failed: {exc}")
            raise

    # -------------------------------------------------------------------------
    # Team membership helpers
    # -------------------------------------------------------------------------

    async def get_team_member(self, team_id: int, user_id: int) -> dict[str, Any] | None:
        """
        Get a specific team membership.

        Returns membership dict with team_id, user_id, role, status, added_at.
        Returns None if user is not a member.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    row = await conn.fetchrow(
                        """
                        SELECT team_id, user_id, role, status, added_at
                        FROM team_members
                        WHERE team_id = $1 AND user_id = $2
                        """,
                        team_id,
                        user_id,
                    )
                    if not row:
                        return None
                    d = dict(row)
                    from datetime import datetime
                    if isinstance(d.get("added_at"), datetime):
                        d["added_at"] = d["added_at"].isoformat()
                    return d
                cur = await conn.execute(
                    """
                    SELECT team_id, user_id, role, status, added_at
                    FROM team_members
                    WHERE team_id = ? AND user_id = ?
                    """,
                    (team_id, user_id),
                )
                row = await cur.fetchone()
                if not row:
                    return None
                return {
                    "team_id": row[0],
                    "user_id": row[1],
                    "role": row[2],
                    "status": row[3],
                    "added_at": row[4],
                }
        except Exception as exc:
            logger.error(f"AuthnzOrgsTeamsRepo.get_team_member failed: {exc}")
            raise

    async def add_team_member(
        self,
        *,
        team_id: int,
        user_id: int,
        context: MembershipWriteContext,
        role: str = "member",
    ) -> dict[str, Any]:
        """
        Add a user to a team (idempotent).

        Returns a dict with ``team_id``, ``user_id``, ``role``, and ``org_id``.
        """
        try:
            mutation = MembershipMutation(
                scope_type=MembershipScopeType.TEAM,
                scope_id=team_id,
                user_id=user_id,
                kind=MembershipMutationKind.ADD,
                role=role,
            )
            result = await self._apply_direct_membership_mutations(
                context=context,
                mutations=(mutation,),
            )
            mutation_result = result.mutation_results[0]
            if mutation_result.error == "org_membership_required":
                raise MembershipParentRequired()
            legacy = mutation_result.to_legacy_result()
            if legacy is None:  # pragma: no cover - impossible for ADD
                raise RuntimeError("Membership add produced no result")
            return legacy
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzOrgsTeamsRepo.add_team_member failed: {exc}")
            raise

    async def list_team_members(self, team_id: int) -> list[dict[str, Any]]:
        """
        List members of a team ordered by ``added_at`` descending.
        """
        try:
            if self._is_postgres():
                rows = await self.db_pool.fetchall(
                    """
                    SELECT user_id, role, status, added_at
                    FROM team_members
                    WHERE team_id = $1
                    ORDER BY added_at DESC
                    """,
                    team_id,
                )
                # Postgres rows are already dict-like
                return [dict(r) for r in rows]

            async with self.db_pool.acquire() as conn:
                cursor = await conn.execute(
                    """
                    SELECT user_id, role, status, added_at
                    FROM team_members
                    WHERE team_id = ?
                    ORDER BY added_at DESC
                    """,
                    (team_id,),
                )
                rows = await cursor.fetchall()
                return [
                    {
                        "user_id": r[0],
                        "role": r[1],
                        "status": r[2],
                        "added_at": r[3],
                    }
                    for r in rows
                ]
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzOrgsTeamsRepo.list_team_members failed: {exc}")
            raise

    async def update_team_member_role(
        self,
        *,
        team_id: int,
        user_id: int,
        role: str,
        context: MembershipWriteContext,
    ) -> dict[str, Any] | None:
        """
        Update a team member's role.
        """
        try:
            result = await self._apply_direct_membership_mutations(
                context=context,
                mutations=(
                    MembershipMutation(
                        scope_type=MembershipScopeType.TEAM,
                        scope_id=team_id,
                        user_id=user_id,
                        kind=MembershipMutationKind.UPDATE_ROLE,
                        role=role,
                    ),
                ),
            )
            return result.mutation_results[0].to_legacy_result()
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzOrgsTeamsRepo.update_team_member_role failed: {exc}")
            raise

    async def list_memberships_for_user(
        self,
        user_id: int,
        *,
        conn: Any | None = None,
    ) -> list[dict[str, Any]]:
        """
        List team memberships (including org_id) for a user.

        Returns dicts with ``team_id``, ``user_id``, ``role``, ``org_id``,
        ``team_name``, and ``org_name``.
        """
        try:
            if self._is_postgres():
                query = """
                SELECT tm.team_id, tm.user_id, tm.role, t.org_id, t.name AS team_name, o.name AS org_name
                FROM team_members tm
                JOIN teams t ON tm.team_id = t.id
                JOIN organizations o ON t.org_id = o.id
                WHERE tm.user_id = $1
                ORDER BY tm.team_id
                """
                rows = (
                    await conn.fetch(query, user_id)
                    if conn is not None
                    else await self.db_pool.fetchall(query, user_id)
                )
                return [dict(r) for r in rows]

            async def _read(sqlite_conn: Any) -> list[dict[str, Any]]:
                cur = await sqlite_conn.execute(
                    """
                    SELECT tm.team_id, tm.user_id, tm.role, t.org_id, t.name, o.name
                    FROM team_members tm
                    JOIN teams t ON tm.team_id = t.id
                    JOIN organizations o ON t.org_id = o.id
                    WHERE tm.user_id = ?
                    ORDER BY tm.team_id
                    """,
                    (user_id,),
                )
                rows = await cur.fetchall()
                return [self._membership_row_to_dict(r) for r in rows]

            if conn is not None:
                return await _read(conn)
            async with self.db_pool.acquire() as acquired_conn:
                return await _read(acquired_conn)
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.bind(
                operation="list_memberships_for_user",
                exception_type=type(exc).__name__,
            ).error("AuthNZ membership read failed")
            raise MembershipReadError() from None

    async def list_active_team_memberships_for_user(
        self,
        user_id: int,
    ) -> list[dict[str, Any]]:
        """
        List active team memberships (including org_id) for a user.

        Returns dicts with ``team_id``, ``user_id``, ``role``, ``org_id``,
        ``team_name``, and ``org_name``.
        """
        try:
            if self._is_postgres():
                rows = await self.db_pool.fetchall(
                    """
                    SELECT tm.team_id, tm.user_id, tm.role, t.org_id, t.name AS team_name, o.name AS org_name
                    FROM team_members tm
                    JOIN teams t ON tm.team_id = t.id
                    JOIN organizations o ON t.org_id = o.id
                    WHERE tm.user_id = $1
                      AND COALESCE(tm.status, 'active') = 'active'
                    ORDER BY tm.team_id
                    """,
                    user_id,
                )
                return [dict(r) for r in rows]

            async with self.db_pool.acquire() as conn:
                cur = await conn.execute(
                    """
                    SELECT tm.team_id, tm.user_id, tm.role, t.org_id, t.name, o.name
                    FROM team_members tm
                    JOIN teams t ON tm.team_id = t.id
                    JOIN organizations o ON t.org_id = o.id
                    WHERE tm.user_id = ?
                      AND COALESCE(tm.status, 'active') = 'active'
                    ORDER BY tm.team_id
                    """,
                    (user_id,),
                )
                rows = await cur.fetchall()
                return [self._membership_row_to_dict(r) for r in rows]
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzOrgsTeamsRepo.list_active_team_memberships_for_user failed: {exc}"
            )
            raise

    async def remove_team_member(
        self,
        *,
        team_id: int,
        user_id: int,
        context: MembershipWriteContext,
    ) -> dict[str, Any]:
        """
        Remove a user from a team.

        Returns ``{\"team_id\", \"user_id\", \"removed\"}``.
        """
        try:
            result = await self._apply_direct_membership_mutations(
                context=context,
                mutations=(
                    MembershipMutation(
                        scope_type=MembershipScopeType.TEAM,
                        scope_id=team_id,
                        user_id=user_id,
                        kind=MembershipMutationKind.REMOVE,
                    ),
                ),
            )
            legacy = result.mutation_results[0].to_legacy_result()
            if legacy is None:  # pragma: no cover - impossible for REMOVE
                raise RuntimeError("Membership remove produced no result")
            return legacy
        except Exception:  # pragma: no cover - surfaced via callers
            logger.exception(
                'AuthnzOrgsTeamsRepo.remove_team_member failed for team_id={} user_id={}',
                team_id,
                user_id,
            )
            raise

    # -------------------------------------------------------------------------
    # Default team helpers (internal)
    # -------------------------------------------------------------------------

    async def _get_or_create_default_team_id(
        self,
        conn: Any,
        org_id: int,
        *,
        create: bool = True,
    ) -> int | None:
        """Fetch (and optionally create) the Default-Base team for an organization."""
        if self._is_postgres():
            row = await conn.fetchrow(
                """
                SELECT id
                FROM public.teams
                WHERE org_id = $1 AND name = $2
                """,
                org_id,
                DEFAULT_BASE_TEAM_NAME,
            )
            if row:
                return int(row["id"])
            if not create:
                return None
            new_row = await conn.fetchrow(
                """
                INSERT INTO public.teams (org_id, name, slug, description, metadata)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (org_id, name) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
                """,
                org_id,
                DEFAULT_BASE_TEAM_NAME,
                DEFAULT_BASE_TEAM_SLUG,
                DEFAULT_BASE_TEAM_DESCRIPTION,
                None,
            )
            return int(new_row["id"])

        # SQLite / aiosqlite connection
        cur = await conn.execute(
            "SELECT id FROM main.teams WHERE org_id = ? AND name = ?",
            (org_id, DEFAULT_BASE_TEAM_NAME),
        )
        row = await cur.fetchone()
        if row:
            return int(row[0])
        if not create:
            return None
        await conn.execute(
            """
            INSERT OR IGNORE INTO main.teams (org_id, name, slug, description, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                org_id,
                DEFAULT_BASE_TEAM_NAME,
                DEFAULT_BASE_TEAM_SLUG,
                DEFAULT_BASE_TEAM_DESCRIPTION,
                None,
            ),
        )
        cur = await conn.execute(
            "SELECT id FROM main.teams WHERE org_id = ? AND name = ?",
            (org_id, DEFAULT_BASE_TEAM_NAME),
        )
        row = await cur.fetchone()
        return int(row[0]) if row else None

    async def _create_default_team_best_effort(
        self,
        conn: Any,
        org_id: int,
    ) -> int | None:
        """Create the default team behind a savepoint on the existing transaction."""

        if self._is_postgres():
            savepoint = conn.transaction()
            await savepoint.start()
            try:
                team_id = await self._get_or_create_default_team_id(
                    conn,
                    org_id,
                    create=True,
                )
            except Exception as exc:
                await savepoint.rollback()
                logger.warning("Default team auto-enroll failed for org_id={}: {}", org_id, exc)
                return None
            await savepoint.commit()
            return team_id

        await conn.create_savepoint("default_team_companion")
        try:
            team_id = await self._get_or_create_default_team_id(
                conn,
                org_id,
                create=True,
            )
        except Exception as exc:
            await conn.rollback_savepoint("default_team_companion")
            await conn.release_savepoint("default_team_companion")
            logger.warning("Default team auto-enroll failed for org_id={}: {}", org_id, exc)
            return None
        await conn.release_savepoint("default_team_companion")
        return team_id

    async def _ensure_user_in_default_team(
        self,
        conn: Any,
        org_id: int,
        user_id: int,
    ) -> None:
        """Ensure the user is enrolled in the organization's Default-Base team."""
        team_id = await self._get_or_create_default_team_id(conn, org_id, create=True)
        if team_id is None:
            return
        if self._is_postgres():
            await conn.execute(
                """
                INSERT INTO team_members (team_id, user_id, role)
                VALUES ($1, $2, $3)
                ON CONFLICT (team_id, user_id) DO NOTHING
                """,
                team_id,
                user_id,
                "member",
            )
        else:
            await conn.execute(
                """
                INSERT OR IGNORE INTO team_members (team_id, user_id, role)
                VALUES (?, ?, ?)
                """,
                (team_id, user_id, "member"),
            )

    async def _remove_user_from_default_team(
        self,
        conn: Any,
        org_id: int,
        user_id: int,
    ) -> None:
        """Remove the user from the organization's Default-Base team if present."""
        team_id = await self._get_or_create_default_team_id(conn, org_id, create=False)
        if team_id is None:
            return
        if self._is_postgres():
            await conn.execute(
                """
                DELETE FROM team_members
                WHERE team_id = $1 AND user_id = $2
                """,
                team_id,
                user_id,
            )
        else:
            await conn.execute(
                """
                DELETE FROM team_members
                WHERE team_id = ? AND user_id = ?
                """,
                (team_id, user_id),
            )

    # -------------------------------------------------------------------------
    # Organization membership helpers
    # -------------------------------------------------------------------------

    async def add_org_member(
        self,
        *,
        org_id: int,
        user_id: int,
        context: MembershipWriteContext,
        role: str = "member",
    ) -> dict[str, Any]:
        """
        Add a user to an organization (idempotent) and ensure default-team membership.
        """
        try:
            operation_time = datetime.now(timezone.utc)
            org_mutation = MembershipMutation(
                scope_type=MembershipScopeType.ORGANIZATION,
                scope_id=org_id,
                user_id=user_id,
                kind=MembershipMutationKind.ADD,
                role=role,
            )
            async with self._membership_transaction() as conn:
                writer = MembershipWriter(self.db_pool)
                default_team_id = await self._get_or_create_default_team_id(
                    conn,
                    org_id,
                    create=False,
                )
                if default_team_id is not None:
                    result = await writer.apply_membership_mutations(
                        conn=conn,
                        context=context,
                        mutations=(
                            org_mutation,
                            MembershipMutation(
                                scope_type=MembershipScopeType.TEAM,
                                scope_id=default_team_id,
                                user_id=user_id,
                                kind=MembershipMutationKind.ADD,
                                role="member",
                                relationship=MembershipMutationRelationship.DEFAULT_TEAM_COMPANION,
                            ),
                        ),
                        anchor_ownership=AnchorOwnership.CALLER_OWNS_ANCHOR,
                        operation_time=operation_time,
                    )
                    results = (result,)
                    primary_result = result.mutation_results[0]
                else:
                    first_result = await writer.apply_membership_mutations(
                        conn=conn,
                        context=context,
                        mutations=(org_mutation,),
                        anchor_ownership=AnchorOwnership.CALLER_OWNS_ANCHOR,
                        operation_time=operation_time,
                    )
                    default_team_id = await self._create_default_team_best_effort(
                        conn,
                        org_id,
                    )
                    results = (first_result,)
                    primary_result = first_result.mutation_results[0]
                    if default_team_id is not None:
                        companion_result = await writer.apply_membership_mutations(
                            conn=conn,
                            context=context,
                            mutations=(
                                org_mutation,
                                MembershipMutation(
                                    scope_type=MembershipScopeType.TEAM,
                                    scope_id=default_team_id,
                                    user_id=user_id,
                                    kind=MembershipMutationKind.ADD,
                                    role="member",
                                    relationship=MembershipMutationRelationship.DEFAULT_TEAM_COMPANION,
                                ),
                            ),
                            anchor_ownership=AnchorOwnership.CALLER_OWNS_ANCHOR,
                            operation_time=operation_time,
                        )
                        results += (companion_result,)
                await self._final_touch_membership_results(
                    conn,
                    results=results,
                    operation_time=operation_time,
                )
            legacy = primary_result.to_legacy_result()
            if legacy is None:  # pragma: no cover - impossible for ADD
                raise RuntimeError("Membership add produced no result")
            return legacy
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzOrgsTeamsRepo.add_org_member failed: {exc}")
            raise

    async def list_org_members(
        self,
        *,
        org_id: int,
        limit: int = 100,
        offset: int = 0,
        role: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List members of an organization with pagination and optional filters.
        """
        try:
            if self._is_postgres():
                conditions = ["org_id = $1"]
                params: list[Any] = [org_id]
                p = 1
                if role:
                    p += 1
                    conditions.append(f"role = ${p}")
                    params.append(role)
                if status:
                    p += 1
                    conditions.append(f"status = ${p}")
                    params.append(status)
                where_clause = " AND ".join(conditions)
                p += 1
                params.append(limit)
                p += 1
                params.append(offset)
                sql = (
                    f"SELECT user_id, role, status, added_at FROM org_members WHERE {where_clause} "  # nosec B608
                    f"ORDER BY added_at DESC LIMIT ${p-1} OFFSET ${p}"
                )
                rows = await self.db_pool.fetchall(sql, *params)
                return [dict(r) for r in rows]

            async with self.db_pool.acquire() as conn:
                conditions = ["org_id = ?"]
                params2: list[Any] = [org_id]
                if role:
                    conditions.append("role = ?")
                    params2.append(role)
                if status:
                    conditions.append("status = ?")
                    params2.append(status)
                where_clause = " AND ".join(conditions)
                sql = (
                    f"SELECT user_id, role, status, added_at FROM org_members WHERE {where_clause} "  # nosec B608
                    f"ORDER BY added_at DESC LIMIT ? OFFSET ?"
                )
                params2.extend([limit, offset])
                cur = await conn.execute(sql, tuple(params2))
                rows = await cur.fetchall()
                return [
                    {
                        "user_id": r[0],
                        "role": r[1],
                        "status": r[2],
                        "added_at": r[3],
                    }
                    for r in rows
                ]
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzOrgsTeamsRepo.list_org_members failed: {exc}")
            raise

    async def remove_org_member(
        self,
        *,
        org_id: int,
        user_id: int,
        context: MembershipWriteContext,
    ) -> dict[str, Any]:
        """
        Remove a user from an organization, enforcing at least one owner.
        """
        try:
            operation_time = datetime.now(timezone.utc)
            org_mutation = MembershipMutation(
                scope_type=MembershipScopeType.ORGANIZATION,
                scope_id=org_id,
                user_id=user_id,
                kind=MembershipMutationKind.REMOVE,
            )
            async with self._membership_transaction() as conn:
                default_team_id = await self._get_or_create_default_team_id(
                    conn,
                    org_id,
                    create=False,
                )
                mutations = [org_mutation]
                if default_team_id is not None:
                    mutations.append(
                        MembershipMutation(
                            scope_type=MembershipScopeType.TEAM,
                            scope_id=default_team_id,
                            user_id=user_id,
                            kind=MembershipMutationKind.REMOVE,
                            relationship=MembershipMutationRelationship.DEFAULT_TEAM_COMPANION,
                        )
                    )
                result = await MembershipWriter(
                    self.db_pool
                ).apply_membership_mutations(
                    conn=conn,
                    context=context,
                    mutations=tuple(mutations),
                    anchor_ownership=AnchorOwnership.CALLER_OWNS_ANCHOR,
                    operation_time=operation_time,
                )
                await self._final_touch_membership_results(
                    conn,
                    results=(result,),
                    operation_time=operation_time,
                )
            legacy = result.mutation_results[0].to_legacy_result()
            if legacy is None:  # pragma: no cover - impossible for REMOVE
                raise RuntimeError("Membership remove produced no result")
            return legacy
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzOrgsTeamsRepo.remove_org_member failed: {exc}")
            raise

    async def update_org_member_role(
        self,
        *,
        org_id: int,
        user_id: int,
        role: str,
        context: MembershipWriteContext,
    ) -> dict[str, Any] | None:
        """
        Update an org member's role, enforcing at least one owner.
        """
        try:
            result = await self._apply_direct_membership_mutations(
                context=context,
                mutations=(
                    MembershipMutation(
                        scope_type=MembershipScopeType.ORGANIZATION,
                        scope_id=org_id,
                        user_id=user_id,
                        kind=MembershipMutationKind.UPDATE_ROLE,
                        role=role,
                    ),
                ),
            )
            return result.mutation_results[0].to_legacy_result()
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzOrgsTeamsRepo.update_org_member_role failed: {exc}"
            )
            raise

    async def list_org_memberships_for_user(
        self,
        user_id: int,
        *,
        conn: Any | None = None,
    ) -> list[dict[str, Any]]:
        """
        List org memberships for a user: ``[{org_id, role, status}]``.
        """
        try:
            if self._is_postgres():
                query = """
                SELECT org_id, role, status
                FROM org_members
                WHERE user_id = $1
                ORDER BY org_id
                """
                rows = (
                    await conn.fetch(query, user_id)
                    if conn is not None
                    else await self.db_pool.fetchall(query, user_id)
                )
                normalized: list[dict[str, Any]] = []
                for r in rows:
                    row_dict = dict(r)
                    normalized.append(
                        {
                            "org_id": int(row_dict.get("org_id")),
                            "role": row_dict.get("role"),
                            "status": row_dict.get("status"),
                        }
                    )
                return normalized

            async def _read(sqlite_conn: Any) -> list[dict[str, Any]]:
                cur = await sqlite_conn.execute(
                    """
                    SELECT org_id, role, status
                    FROM org_members
                    WHERE user_id = ?
                    ORDER BY org_id
                    """,
                    (user_id,),
                )
                rows = await cur.fetchall()
                return [{"org_id": r[0], "role": r[1], "status": r[2]} for r in rows]

            if conn is not None:
                return await _read(conn)
            async with self.db_pool.acquire() as acquired_conn:
                return await _read(acquired_conn)
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.bind(
                operation="list_org_memberships_for_user",
                exception_type=type(exc).__name__,
            ).error("AuthNZ membership read failed")
            raise MembershipReadError() from None

    async def list_organizations_for_user(
        self,
        user_id: int,
        *,
        limit: int = 100,
        offset: int = 0,
        with_total: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        List organizations a given user is a member of with pagination support.

        Returns (rows, total). When with_total=False, total is returned as 0.
        """
        try:
            if self._is_postgres():
                rows = await self.db_pool.fetchall(
                    """
                    SELECT DISTINCT
                        o.id,
                        o.name,
                        o.slug,
                        o.owner_user_id,
                        o.is_active,
                        o.created_at,
                        o.updated_at,
                        m.role AS membership_role
                    FROM organizations o
                    JOIN org_members m ON m.org_id = o.id
                    WHERE m.user_id = $1
                    ORDER BY o.created_at DESC, o.id DESC
                    LIMIT $2 OFFSET $3
                    """,
                    user_id,
                    limit,
                    offset,
                )
                total = (
                    await self.db_pool.fetchval(
                        """
                        SELECT COUNT(DISTINCT o.id)
                        FROM organizations o
                        JOIN org_members m ON m.org_id = o.id
                        WHERE m.user_id = $1
                        """,
                        user_id,
                    )
                    if with_total
                    else 0
                )

                normalized: list[dict[str, Any]] = []
                for r in rows:
                    d = dict(r)
                    d["is_active"] = bool(d.get("is_active", True))
                    normalized.append(d)
                return normalized, int(total or 0)

            async with self.db_pool.acquire() as conn:
                cursor = await conn.execute(
                    """
                    SELECT DISTINCT
                        o.id,
                        o.name,
                        o.slug,
                        o.owner_user_id,
                        o.is_active,
                        o.created_at,
                        o.updated_at,
                        m.role AS membership_role
                    FROM organizations o
                    JOIN org_members m ON m.org_id = o.id
                    WHERE m.user_id = ?
                    ORDER BY o.created_at DESC, o.id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (user_id, limit, offset),
                )
                rows_raw = await cursor.fetchall()
                rows = [
                    {
                        "id": r[0],
                        "name": r[1],
                        "slug": r[2],
                        "owner_user_id": r[3],
                        "is_active": bool(r[4]),
                        "created_at": r[5],
                        "updated_at": r[6],
                        "membership_role": r[7],
                    }
                    for r in rows_raw
                ]

                if with_total:
                    cur2 = await conn.execute(
                        """
                        SELECT COUNT(DISTINCT o.id)
                        FROM organizations o
                        JOIN org_members m ON m.org_id = o.id
                        WHERE m.user_id = ?
                        """,
                        (user_id,),
                    )
                    total_row = await cur2.fetchone()
                    total = int(total_row[0]) if total_row else 0
                else:
                    total = 0

                return rows, int(total or 0)
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzOrgsTeamsRepo.list_organizations_for_user failed: {exc}"
            )
            raise
    @staticmethod
    def _membership_row_to_dict(row: Any) -> dict[str, Any]:
        """Normalize membership rows across legacy/new column projections."""
        if isinstance(row, dict):
            return dict(row)
        result = {
            "team_id": row[0],
            "user_id": row[1],
            "role": row[2],
            "org_id": row[3],
        }
        if len(row) > 4:
            result["team_name"] = row[4]
        if len(row) > 5:
            result["org_name"] = row[5]
        return result
