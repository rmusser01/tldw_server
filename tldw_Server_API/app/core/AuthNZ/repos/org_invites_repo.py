"""
org_invites_repo.py

Repository for organization invite codes and redemption tracking.
"""
from __future__ import annotations

import secrets
import string
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.repos.datetime_utils import _strip_tzinfo
from tldw_Server_API.app.core.AuthNZ.settings import get_settings

# Invite code configuration
INVITE_CODE_LENGTH = 20
INVITE_CODE_PREFIX = "INV-"
INVITE_CODE_ALPHABET = string.ascii_uppercase + string.digits


def generate_invite_code() -> str:
    """Generate a unique invite code like INV-XXXXXXXXXXXXXXXXXXXX."""
    random_part = "".join(secrets.choice(INVITE_CODE_ALPHABET) for _ in range(INVITE_CODE_LENGTH))
    return f"{INVITE_CODE_PREFIX}{random_part}"


@dataclass
class AuthnzOrgInvitesRepo:
    """
    Repository for org_invites and org_invite_redemptions tables.

    Handles invite creation, validation, redemption, and listing.
    Supports both PostgreSQL and SQLite backends.
    """

    db_pool: DatabasePool

    def _is_postgres(self, conn: Any | None = None) -> bool:
        """Detect whether the configured backend is PostgreSQL from pool state."""
        _ = conn  # Compatibility placeholder for legacy call sites.
        return bool(getattr(self.db_pool, "pool", None))

    async def create_invite(
        self,
        *,
        org_id: int,
        created_by: int,
        team_id: int | None = None,
        role_to_grant: str = "member",
        max_uses: int = 1,
        expiry_days: int = 7,
        description: str | None = None,
        allowed_email_domain: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a new organization invite code.

        Args:
            org_id: Organization ID the invite is for
            created_by: User ID of the invite creator
            team_id: Optional team ID (if set, invite is team-specific)
            role_to_grant: Role to assign upon redemption (member, lead, admin)
            max_uses: Maximum number of times this invite can be used
            expiry_days: Number of days until the invite expires
            description: Internal description/note for the invite
            allowed_email_domain: Optional email domain allowlist
            metadata: Optional JSON metadata

        Returns:
            Dict with invite details including the generated code
        """
        import json

        code = generate_invite_code()
        expires_at = datetime.now(timezone.utc) + timedelta(days=expiry_days)

        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    row = await conn.fetchrow(
                        """
                        INSERT INTO org_invites
                        (code, org_id, team_id, role_to_grant, created_by, expires_at,
                         max_uses, description, allowed_email_domain, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        RETURNING id, code, org_id, team_id, role_to_grant, created_by,
                                  created_at, expires_at, max_uses, uses_count, is_active, description,
                                  allowed_email_domain
                        """,
                        code, org_id, team_id, role_to_grant, created_by, _strip_tzinfo(expires_at),
                        max_uses, description, allowed_email_domain, metadata
                    )
                    result = dict(row)
                    # Normalize datetime fields
                    for key in ("created_at", "expires_at"):
                        if isinstance(result.get(key), datetime):
                            result[key] = result[key].isoformat()
                    return result
                else:
                    # SQLite path
                    cur = await conn.execute(
                        """
                        INSERT INTO org_invites
                        (code, org_id, team_id, role_to_grant, created_by, expires_at,
                         max_uses, description, allowed_email_domain, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (code, org_id, team_id, role_to_grant, created_by,
                         expires_at.isoformat(), max_uses, description, allowed_email_domain,
                         json.dumps(metadata) if metadata else None)
                    )
                    invite_id = cur.lastrowid
                    cur2 = await conn.execute(
                        """
                        SELECT id, code, org_id, team_id, role_to_grant, created_by,
                               created_at, expires_at, max_uses, uses_count, is_active, description,
                               allowed_email_domain
                        FROM org_invites WHERE id = ?
                        """,
                        (invite_id,)
                    )
                    row = await cur2.fetchone()
                    return {
                        "id": row[0],
                        "code": row[1],
                        "org_id": row[2],
                        "team_id": row[3],
                        "role_to_grant": row[4],
                        "created_by": row[5],
                        "created_at": row[6],
                        "expires_at": row[7],
                        "max_uses": row[8],
                        "uses_count": row[9],
                        "is_active": bool(row[10]),
                        "description": row[11],
                        "allowed_email_domain": row[12],
                    }
        except Exception as exc:
            logger.error(f"AuthnzOrgInvitesRepo.create_invite failed: {exc}")
            raise

    async def get_invite_by_code(self, code: str) -> dict[str, Any] | None:
        """
        Get invite details by code.

        Returns None if invite doesn't exist.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    row = await conn.fetchrow(
                        """
                        SELECT i.id, i.code, i.org_id, i.team_id, i.role_to_grant, i.created_by,
                               i.created_at, i.expires_at, i.max_uses, i.uses_count, i.is_active,
                               i.description, i.allowed_email_domain, o.name as org_name, o.slug as org_slug,
                               t.name as team_name
                        FROM org_invites i
                        JOIN organizations o ON o.id = i.org_id
                        LEFT JOIN teams t ON t.id = i.team_id
                        WHERE i.code = $1
                        """,
                        code
                    )
                    if not row:
                        return None
                    result = dict(row)
                    for key in ("created_at", "expires_at"):
                        if isinstance(result.get(key), datetime):
                            result[key] = result[key].isoformat()
                    return result
                else:
                    cur = await conn.execute(
                        """
                        SELECT i.id, i.code, i.org_id, i.team_id, i.role_to_grant, i.created_by,
                               i.created_at, i.expires_at, i.max_uses, i.uses_count, i.is_active,
                               i.description, i.allowed_email_domain, o.name as org_name, o.slug as org_slug,
                               t.name as team_name
                        FROM org_invites i
                        JOIN organizations o ON o.id = i.org_id
                        LEFT JOIN teams t ON t.id = i.team_id
                        WHERE i.code = ?
                        """,
                        (code,)
                    )
                    row = await cur.fetchone()
                    if not row:
                        return None
                    return {
                        "id": row[0],
                        "code": row[1],
                        "org_id": row[2],
                        "team_id": row[3],
                        "role_to_grant": row[4],
                        "created_by": row[5],
                        "created_at": row[6],
                        "expires_at": row[7],
                        "max_uses": row[8],
                        "uses_count": row[9],
                        "is_active": bool(row[10]),
                        "description": row[11],
                        "allowed_email_domain": row[12],
                        "org_name": row[13],
                        "org_slug": row[14],
                        "team_name": row[15],
                    }
        except Exception as exc:
            logger.error(f"AuthnzOrgInvitesRepo.get_invite_by_code failed: {exc}")
            raise

    async def get_invite_by_id(self, invite_id: int) -> dict[str, Any] | None:
        """Get invite details by ID."""
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    row = await conn.fetchrow(
                        """
                        SELECT id, code, org_id, team_id, role_to_grant, created_by,
                               created_at, expires_at, max_uses, uses_count, is_active, description,
                               allowed_email_domain
                        FROM org_invites WHERE id = $1
                        """,
                        invite_id
                    )
                    if not row:
                        return None
                    result = dict(row)
                    for key in ("created_at", "expires_at"):
                        if isinstance(result.get(key), datetime):
                            result[key] = result[key].isoformat()
                    return result
                else:
                    cur = await conn.execute(
                        """
                        SELECT id, code, org_id, team_id, role_to_grant, created_by,
                               created_at, expires_at, max_uses, uses_count, is_active, description,
                               allowed_email_domain
                        FROM org_invites WHERE id = ?
                        """,
                        (invite_id,)
                    )
                    row = await cur.fetchone()
                    if not row:
                        return None
                    return {
                        "id": row[0],
                        "code": row[1],
                        "org_id": row[2],
                        "team_id": row[3],
                        "role_to_grant": row[4],
                        "created_by": row[5],
                        "created_at": row[6],
                        "expires_at": row[7],
                        "max_uses": row[8],
                        "uses_count": row[9],
                        "is_active": bool(row[10]),
                        "description": row[11],
                        "allowed_email_domain": row[12],
                    }
        except Exception as exc:
            logger.error(f"AuthnzOrgInvitesRepo.get_invite_by_id failed: {exc}")
            raise

    async def list_org_invites(
        self,
        org_id: int,
        *,
        include_expired: bool = False,
        include_inactive: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        List invites for an organization.

        Returns (list of invites, total count).
        """
        try:
            async with self.db_pool.transaction() as conn:
                # Build WHERE clause
                conditions = ["i.org_id = $1" if self._is_postgres(conn) else "i.org_id = ?"]
                params: list[Any] = [org_id]

                if not include_inactive:
                    if self._is_postgres(conn):
                        conditions.append("i.is_active = TRUE")
                    else:
                        conditions.append("is_active = 1")

                if not include_expired:
                    if self._is_postgres(conn):
                        conditions.append(f"expires_at > ${len(params) + 1}")
                        params.append(_strip_tzinfo(datetime.now(timezone.utc)))
                    else:
                        conditions.append("expires_at > ?")
                        params.append(datetime.now(timezone.utc).isoformat())

                where_clause = " AND ".join(conditions)

                if self._is_postgres(conn):
                    # Get total count
                    count_sql_template = "SELECT COUNT(*) FROM org_invites i WHERE {where_clause}"
                    count_sql = count_sql_template.format_map(locals())  # nosec B608
                    count_row = await conn.fetchrow(
                        count_sql,
                        *params
                    )
                    total = count_row[0] if count_row else 0

                    # Get paginated results
                    limit_placeholder = f"${len(params) + 1}"
                    offset_placeholder = f"${len(params) + 2}"
                    select_sql_template = """
                        SELECT i.id, i.code, i.org_id, i.team_id, i.role_to_grant, i.created_by,
                               i.created_at, i.expires_at, i.max_uses, i.uses_count, i.is_active,
                               i.description, i.allowed_email_domain, t.name as team_name
                        FROM org_invites i
                        LEFT JOIN teams t ON t.id = i.team_id
                        WHERE {where_clause}
                        ORDER BY i.created_at DESC
                        LIMIT {limit_placeholder} OFFSET {offset_placeholder}
                        """
                    select_sql = select_sql_template.format_map(locals())  # nosec B608
                    rows = await conn.fetch(
                        select_sql,
                        *params, limit, offset
                    )
                    results = []
                    for row in rows:
                        d = dict(row)
                        for key in ("created_at", "expires_at"):
                            if isinstance(d.get(key), datetime):
                                d[key] = d[key].isoformat()
                        results.append(d)
                    return results, total
                else:
                    # SQLite path
                    count_sql_template = "SELECT COUNT(*) FROM org_invites i WHERE {where_clause}"
                    count_sql = count_sql_template.format_map(locals())  # nosec B608
                    cur = await conn.execute(
                        count_sql,
                        tuple(params)
                    )
                    count_row = await cur.fetchone()
                    total = count_row[0] if count_row else 0

                    select_sql_template = """
                        SELECT i.id, i.code, i.org_id, i.team_id, i.role_to_grant, i.created_by,
                               i.created_at, i.expires_at, i.max_uses, i.uses_count, i.is_active,
                               i.description, i.allowed_email_domain, t.name as team_name
                        FROM org_invites i
                        LEFT JOIN teams t ON t.id = i.team_id
                        WHERE {where_clause}
                        ORDER BY i.created_at DESC
                        LIMIT ? OFFSET ?
                        """
                    select_sql = select_sql_template.format_map(locals())  # nosec B608
                    cur2 = await conn.execute(
                        select_sql,
                        tuple(params) + (limit, offset)
                    )
                    rows = await cur2.fetchall()
                    results = []
                    for row in rows:
                        results.append({
                            "id": row[0],
                            "code": row[1],
                            "org_id": row[2],
                            "team_id": row[3],
                            "role_to_grant": row[4],
                            "created_by": row[5],
                            "created_at": row[6],
                            "expires_at": row[7],
                            "max_uses": row[8],
                            "uses_count": row[9],
                            "is_active": bool(row[10]),
                            "description": row[11],
                            "allowed_email_domain": row[12],
                            "team_name": row[13],
                        })
                    return results, total
        except Exception as exc:
            logger.error(f"AuthnzOrgInvitesRepo.list_org_invites failed: {exc}")
            raise

    async def increment_uses_count(self, invite_id: int) -> bool:
        """
        Increment the uses_count for an invite.

        Returns True if successful, False if invite not found.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    result = await conn.execute(
                        "UPDATE org_invites SET uses_count = uses_count + 1 WHERE id = $1",
                        invite_id
                    )
                    return "UPDATE 1" in result
                else:
                    cur = await conn.execute(
                        "UPDATE org_invites SET uses_count = uses_count + 1 WHERE id = ?",
                        (invite_id,)
                    )
                    return cur.rowcount > 0
        except Exception as exc:
            logger.error(f"AuthnzOrgInvitesRepo.increment_uses_count failed: {exc}")
            raise

    async def deactivate_invite(self, invite_id: int, org_id: int) -> bool:
        """
        Deactivate an invite (soft revoke).

        Returns True if successful, False if invite not found or doesn't belong to org.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    result = await conn.execute(
                        "UPDATE org_invites SET is_active = FALSE WHERE id = $1 AND org_id = $2",
                        invite_id, org_id
                    )
                    return "UPDATE 1" in result
                else:
                    cur = await conn.execute(
                        "UPDATE org_invites SET is_active = 0 WHERE id = ? AND org_id = ?",
                        (invite_id, org_id)
                    )
                    return cur.rowcount > 0
        except Exception as exc:
            logger.error(f"AuthnzOrgInvitesRepo.deactivate_invite failed: {exc}")
            raise

    async def record_redemption(
        self,
        *,
        invite_id: int,
        user_id: int,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> dict[str, Any]:
        """
        Record that a user redeemed an invite.

        Returns the redemption record.
        Raises if user already redeemed this invite (UNIQUE constraint).
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    row = await conn.fetchrow(
                        """
                        INSERT INTO org_invite_redemptions (invite_id, user_id, ip_address, user_agent)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (invite_id, user_id) DO NOTHING
                        RETURNING id, invite_id, user_id, redeemed_at, ip_address
                        """,
                        invite_id, user_id, ip_address, user_agent
                    )
                    if row is None:
                        settings = get_settings()
                        if settings.PII_REDACT_LOGS:
                            logger.warning("User already redeemed invite (details redacted)")
                        else:
                            logger.warning(f"User {user_id} already redeemed invite {invite_id}")
                        raise ValueError(f"User {user_id} has already redeemed this invite")
                    result = dict(row)
                    if isinstance(result.get("redeemed_at"), datetime):
                        result["redeemed_at"] = result["redeemed_at"].isoformat()
                    return result
                else:
                    cur = await conn.execute(
                        """
                        INSERT OR IGNORE INTO org_invite_redemptions (invite_id, user_id, ip_address, user_agent)
                        VALUES (?, ?, ?, ?)
                        """,
                        (invite_id, user_id, ip_address, user_agent)
                    )
                    if cur.rowcount == 0:
                        settings = get_settings()
                        if settings.PII_REDACT_LOGS:
                            logger.warning("User already redeemed invite (details redacted)")
                        else:
                            logger.warning(f"User {user_id} already redeemed invite {invite_id}")
                        raise ValueError(f"User {user_id} has already redeemed this invite")
                    redemption_id = cur.lastrowid
                    cur2 = await conn.execute(
                        "SELECT id, invite_id, user_id, redeemed_at, ip_address FROM org_invite_redemptions WHERE id = ?",
                        (redemption_id,)
                    )
                    row = await cur2.fetchone()
                    return {
                        "id": row[0],
                        "invite_id": row[1],
                        "user_id": row[2],
                        "redeemed_at": row[3],
                        "ip_address": row[4],
                    }
        except ValueError:
            # Propagate domain-level "already redeemed" error without extra logging
            raise
        except Exception as exc:
            msg = str(exc).lower()
            if (
                "org_invite_redemptions" in msg
                and "invite_id" in msg
                and "user_id" in msg
                and ("unique" in msg or "duplicate" in msg)
            ):
                settings = get_settings()
                if settings.PII_REDACT_LOGS:
                    logger.warning("User already redeemed invite (details redacted)")
                else:
                    logger.warning(f"User {user_id} already redeemed invite {invite_id}")
                raise ValueError(f"User {user_id} has already redeemed this invite") from exc
            logger.error(f"AuthnzOrgInvitesRepo.record_redemption failed: {exc}")
            raise

    async def has_user_redeemed(self, invite_id: int, user_id: int) -> bool:
        """Check if a user has already redeemed a specific invite."""
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    row = await conn.fetchrow(
                        "SELECT 1 FROM org_invite_redemptions WHERE invite_id = $1 AND user_id = $2",
                        invite_id, user_id
                    )
                    return row is not None
                else:
                    cur = await conn.execute(
                        "SELECT 1 FROM org_invite_redemptions WHERE invite_id = ? AND user_id = ?",
                        (invite_id, user_id)
                    )
                    return await cur.fetchone() is not None
        except Exception as exc:
            logger.error(f"AuthnzOrgInvitesRepo.has_user_redeemed failed: {exc}")
            raise

    async def cleanup_expired_invites(self, deactivate_only: bool = True) -> int:
        """
        Clean up expired invites.

        Args:
            deactivate_only: If True, set is_active=0. If False, delete expired invites.

        Returns:
            Number of invites affected.
        """
        try:
            async with self.db_pool.transaction() as conn:
                now = datetime.now(timezone.utc)
                if self._is_postgres(conn):
                    if deactivate_only:
                        result = await conn.execute(
                            "UPDATE org_invites SET is_active = FALSE WHERE expires_at < $1 AND is_active = TRUE",
                            _strip_tzinfo(now)
                        )
                    else:
                        result = await conn.execute(
                            "DELETE FROM org_invites WHERE expires_at < $1",
                            _strip_tzinfo(now)
                        )
                    # Parse affected rows from result string
                    import re
                    match = re.search(r'(\d+)', result)
                    return int(match.group(1)) if match else 0
                else:
                    now_str = now.isoformat()
                    if deactivate_only:
                        cur = await conn.execute(
                            "UPDATE org_invites SET is_active = 0 WHERE expires_at < ? AND is_active = 1",
                            (now_str,)
                        )
                    else:
                        cur = await conn.execute(
                            "DELETE FROM org_invites WHERE expires_at < ?",
                            (now_str,)
                        )
                    return cur.rowcount
        except Exception as exc:
            logger.error(f"AuthnzOrgInvitesRepo.cleanup_expired_invites failed: {exc}")
            raise


# Type alias for backwards compatibility
OrgInvitesRepo = AuthnzOrgInvitesRepo
