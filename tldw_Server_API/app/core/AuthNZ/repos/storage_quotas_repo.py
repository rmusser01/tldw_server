"""Repository for storage quotas management in AuthNZ.

This module provides CRUD operations for the storage_quotas table,
which manages team/org-level storage quotas (shared pool model).
User-level quotas are stored on the users table.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.DB_Management.storage_quota_reservation_store import NativeQuotaReservationStore

# Default quota values (in MB)
DEFAULT_ORG_QUOTA_MB = 10240  # 10 GB
DEFAULT_TEAM_QUOTA_MB = 5120  # 5 GB
DEFAULT_SOFT_LIMIT_PCT = 80
DEFAULT_HARD_LIMIT_PCT = 100


@dataclass
class AuthnzStorageQuotasRepo:
    """
    Repository for AuthNZ storage_quotas records.

    Manages team and org-level storage quotas. User quotas are stored
    directly on the users table (storage_quota_mb, storage_used_mb).
    """

    db_pool: DatabasePool

    def _is_postgres(self) -> bool:
        """Detect whether the current AuthNZ backend is PostgreSQL."""
        return getattr(self.db_pool, "pool", None) is not None

    @staticmethod
    def _normalize_record(row: Any) -> dict[str, Any]:
        """Normalize backend-specific row types to a consistent dict."""
        if row is None:
            return {}
        try:
            record = dict(row) if hasattr(row, "keys") or isinstance(row, dict) else {}
        except Exception:
            record = {}

        # Type conversions
        for field in ("id", "org_id", "team_id", "quota_mb", "soft_limit_pct", "hard_limit_pct"):
            if field in record and record[field] is not None:
                with contextlib.suppress(Exception):
                    record[field] = int(record[field])

        if "used_mb" in record and record["used_mb"] is not None:
            with contextlib.suppress(Exception):
                record["used_mb"] = float(record["used_mb"])

        return record

    async def get_org_quota(self, org_id: int) -> dict[str, Any] | None:
        """Get storage quota for an organization."""
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres():
                    row = await conn.fetchrow(
                        "SELECT * FROM storage_quotas WHERE org_id = $1",
                        org_id,
                    )
                    return self._normalize_record(row) if row else None
                else:
                    cursor = await conn.execute(
                        "SELECT * FROM storage_quotas WHERE org_id = ?",
                        (org_id,),
                    )
                    row = await cursor.fetchone()
                    if not row:
                        return None
                    cols = [desc[0] for desc in cursor.description]
                    return self._normalize_record(dict(zip(cols, row)))
        except Exception as exc:
            logger.error(f"AuthnzStorageQuotasRepo.get_org_quota failed: {exc}")
            raise

    async def get_team_quota(self, team_id: int) -> dict[str, Any] | None:
        """Get storage quota for a team."""
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres():
                    row = await conn.fetchrow(
                        "SELECT * FROM storage_quotas WHERE team_id = $1",
                        team_id,
                    )
                    return self._normalize_record(row) if row else None
                else:
                    cursor = await conn.execute(
                        "SELECT * FROM storage_quotas WHERE team_id = ?",
                        (team_id,),
                    )
                    row = await cursor.fetchone()
                    if not row:
                        return None
                    cols = [desc[0] for desc in cursor.description]
                    return self._normalize_record(dict(zip(cols, row)))
        except Exception as exc:
            logger.error(f"AuthnzStorageQuotasRepo.get_team_quota failed: {exc}")
            raise

    async def upsert_org_quota(
        self,
        org_id: int,
        *,
        quota_mb: int = DEFAULT_ORG_QUOTA_MB,
        soft_limit_pct: int = DEFAULT_SOFT_LIMIT_PCT,
        hard_limit_pct: int = DEFAULT_HARD_LIMIT_PCT,
    ) -> dict[str, Any]:
        """Create or update storage quota for an organization."""
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres():
                    row = await conn.fetchrow(
                        """
                        INSERT INTO storage_quotas (org_id, quota_mb, soft_limit_pct, hard_limit_pct, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        ON CONFLICT (org_id) WHERE org_id IS NOT NULL
                        DO UPDATE SET quota_mb = EXCLUDED.quota_mb,
                                      soft_limit_pct = EXCLUDED.soft_limit_pct,
                                      hard_limit_pct = EXCLUDED.hard_limit_pct,
                                      updated_at = CURRENT_TIMESTAMP
                        RETURNING *
                        """,
                        org_id, quota_mb, soft_limit_pct, hard_limit_pct,
                    )
                    return self._normalize_record(row)
                else:
                    # Check if exists
                    cursor = await conn.execute(
                        "SELECT id FROM storage_quotas WHERE org_id = ?",
                        (org_id,),
                    )
                    existing = await cursor.fetchone()

                    if existing:
                        await conn.execute(
                            """
                            UPDATE storage_quotas
                            SET quota_mb = ?, soft_limit_pct = ?, hard_limit_pct = ?, updated_at = ?
                            WHERE org_id = ?
                            """,
                            (quota_mb, soft_limit_pct, hard_limit_pct, now_iso, org_id),
                        )
                    else:
                        await conn.execute(
                            """
                            INSERT INTO storage_quotas (org_id, quota_mb, soft_limit_pct, hard_limit_pct, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (org_id, quota_mb, soft_limit_pct, hard_limit_pct, now_iso, now_iso),
                        )

            return await self.get_org_quota(org_id) or {}

        except Exception as exc:
            logger.error(f"AuthnzStorageQuotasRepo.upsert_org_quota failed: {exc}")
            raise

    async def upsert_team_quota(
        self,
        team_id: int,
        *,
        quota_mb: int = DEFAULT_TEAM_QUOTA_MB,
        soft_limit_pct: int = DEFAULT_SOFT_LIMIT_PCT,
        hard_limit_pct: int = DEFAULT_HARD_LIMIT_PCT,
    ) -> dict[str, Any]:
        """Create or update storage quota for a team."""
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres():
                    row = await conn.fetchrow(
                        """
                        INSERT INTO storage_quotas (team_id, quota_mb, soft_limit_pct, hard_limit_pct, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        ON CONFLICT (team_id) WHERE team_id IS NOT NULL
                        DO UPDATE SET quota_mb = EXCLUDED.quota_mb,
                                      soft_limit_pct = EXCLUDED.soft_limit_pct,
                                      hard_limit_pct = EXCLUDED.hard_limit_pct,
                                      updated_at = CURRENT_TIMESTAMP
                        RETURNING *
                        """,
                        team_id, quota_mb, soft_limit_pct, hard_limit_pct,
                    )
                    return self._normalize_record(row)
                else:
                    # Check if exists
                    cursor = await conn.execute(
                        "SELECT id FROM storage_quotas WHERE team_id = ?",
                        (team_id,),
                    )
                    existing = await cursor.fetchone()

                    if existing:
                        await conn.execute(
                            """
                            UPDATE storage_quotas
                            SET quota_mb = ?, soft_limit_pct = ?, hard_limit_pct = ?, updated_at = ?
                            WHERE team_id = ?
                            """,
                            (quota_mb, soft_limit_pct, hard_limit_pct, now_iso, team_id),
                        )
                    else:
                        await conn.execute(
                            """
                            INSERT INTO storage_quotas (team_id, quota_mb, soft_limit_pct, hard_limit_pct, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (team_id, quota_mb, soft_limit_pct, hard_limit_pct, now_iso, now_iso),
                        )

            return await self.get_team_quota(team_id) or {}

        except Exception as exc:
            logger.error(f"AuthnzStorageQuotasRepo.upsert_team_quota failed: {exc}")
            raise

    async def update_org_used_mb(self, org_id: int, used_mb: float) -> None:
        """Update the used_mb for an organization."""
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres():
                    await conn.execute(
                        """
                        UPDATE storage_quotas
                        SET used_mb = $1, updated_at = CURRENT_TIMESTAMP
                        WHERE org_id = $2
                        """,
                        used_mb, org_id,
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE storage_quotas
                        SET used_mb = ?, updated_at = ?
                        WHERE org_id = ?
                        """,
                        (used_mb, now_iso, org_id),
                    )
        except Exception as exc:
            logger.error(f"AuthnzStorageQuotasRepo.update_org_used_mb failed: {exc}")
            raise

    async def update_team_used_mb(self, team_id: int, used_mb: float) -> None:
        """Update the used_mb for a team."""
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres():
                    await conn.execute(
                        """
                        UPDATE storage_quotas
                        SET used_mb = $1, updated_at = CURRENT_TIMESTAMP
                        WHERE team_id = $2
                        """,
                        used_mb, team_id,
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE storage_quotas
                        SET used_mb = ?, updated_at = ?
                        WHERE team_id = ?
                        """,
                        (used_mb, now_iso, team_id),
                    )
        except Exception as exc:
            logger.error(f"AuthnzStorageQuotasRepo.update_team_used_mb failed: {exc}")
            raise

    async def increment_org_used_mb(self, org_id: int, delta_mb: float) -> float:
        """Atomically increment used_mb for an org. Returns new value."""
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres():
                    new_value = await conn.fetchval(
                        """
                        UPDATE storage_quotas
                        SET used_mb = COALESCE(used_mb, 0) + $1, updated_at = CURRENT_TIMESTAMP
                        WHERE org_id = $2
                        RETURNING used_mb
                        """,
                        delta_mb, org_id,
                    )
                    return float(new_value) if new_value else 0.0
                else:
                    await conn.execute(
                        """
                        UPDATE storage_quotas
                        SET used_mb = COALESCE(used_mb, 0) + ?, updated_at = ?
                        WHERE org_id = ?
                        """,
                        (delta_mb, now_iso, org_id),
                    )
                    cursor = await conn.execute(
                        "SELECT used_mb FROM storage_quotas WHERE org_id = ?",
                        (org_id,),
                    )
                    row = await cursor.fetchone()
                    return float(row[0]) if row and row[0] else 0.0
        except Exception as exc:
            logger.error(f"AuthnzStorageQuotasRepo.increment_org_used_mb failed: {exc}")
            raise

    async def increment_team_used_mb(self, team_id: int, delta_mb: float) -> float:
        """Atomically increment used_mb for a team. Returns new value."""
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres():
                    new_value = await conn.fetchval(
                        """
                        UPDATE storage_quotas
                        SET used_mb = COALESCE(used_mb, 0) + $1, updated_at = CURRENT_TIMESTAMP
                        WHERE team_id = $2
                        RETURNING used_mb
                        """,
                        delta_mb, team_id,
                    )
                    return float(new_value) if new_value else 0.0
                else:
                    await conn.execute(
                        """
                        UPDATE storage_quotas
                        SET used_mb = COALESCE(used_mb, 0) + ?, updated_at = ?
                        WHERE team_id = ?
                        """,
                        (delta_mb, now_iso, team_id),
                    )
                    cursor = await conn.execute(
                        "SELECT used_mb FROM storage_quotas WHERE team_id = ?",
                        (team_id,),
                    )
                    row = await cursor.fetchone()
                    return float(row[0]) if row and row[0] else 0.0
        except Exception as exc:
            logger.error(f"AuthnzStorageQuotasRepo.increment_team_used_mb failed: {exc}")
            raise

    async def delete_org_quota(self, org_id: int) -> bool:
        """Delete storage quota for an organization."""
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres():
                    result = await conn.execute(
                        "DELETE FROM storage_quotas WHERE org_id = $1",
                        org_id,
                    )
                    return "DELETE 1" in str(result)
                else:
                    cursor = await conn.execute(
                        "DELETE FROM storage_quotas WHERE org_id = ?",
                        (org_id,),
                    )
                    return cursor.rowcount > 0
        except Exception as exc:
            logger.error(f"AuthnzStorageQuotasRepo.delete_org_quota failed: {exc}")
            raise

    async def delete_team_quota(self, team_id: int) -> bool:
        """Delete storage quota for a team."""
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres():
                    result = await conn.execute(
                        "DELETE FROM storage_quotas WHERE team_id = $1",
                        team_id,
                    )
                    return "DELETE 1" in str(result)
                else:
                    cursor = await conn.execute(
                        "DELETE FROM storage_quotas WHERE team_id = ?",
                        (team_id,),
                    )
                    return cursor.rowcount > 0
        except Exception as exc:
            logger.error(f"AuthnzStorageQuotasRepo.delete_team_quota failed: {exc}")
            raise

    async def list_all_quotas(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List all storage quotas (for admin purposes)."""
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres():
                    rows = await conn.fetch(
                        """
                        SELECT sq.*, o.name as org_name, t.name as team_name
                        FROM storage_quotas sq
                        LEFT JOIN organizations o ON sq.org_id = o.id
                        LEFT JOIN teams t ON sq.team_id = t.id
                        ORDER BY sq.id
                        OFFSET $1 LIMIT $2
                        """,
                        offset, limit,
                    )
                    records = [self._normalize_record(row) for row in rows]
                else:
                    cursor = await conn.execute(
                        """
                        SELECT sq.*, o.name as org_name, t.name as team_name
                        FROM storage_quotas sq
                        LEFT JOIN organizations o ON sq.org_id = o.id
                        LEFT JOIN teams t ON sq.team_id = t.id
                        ORDER BY sq.id
                        LIMIT ? OFFSET ?
                        """,
                        (limit, offset),
                    )
                    rows = await cursor.fetchall()
                    cols = [desc[0] for desc in cursor.description] if cursor.description else []
                    records = [self._normalize_record(dict(zip(cols, row))) for row in rows]
            for record in records:
                native = await NativeQuotaReservationStore(self.db_pool).usage(
                    org_id=record.get('org_id'), team_id=record.get('team_id'),
                )
                record['legacy_used_mb'] = record['used_mb']
                record['native_used_mb'] = native['bytes'] / 1048576
                record['used_mb'] += record['native_used_mb']
            return records
        except Exception as exc:
            logger.error(f"AuthnzStorageQuotasRepo.list_all_quotas failed: {exc}")
            raise

    async def check_quota_status(
        self,
        *,
        org_id: int | None = None,
        team_id: int | None = None,
    ) -> dict[str, Any]:
        """
        Check quota status for an org or team.

        Returns:
            Dict with: quota_mb, used_mb, remaining_mb, usage_pct,
                       at_soft_limit, at_hard_limit
        """
        quota = None
        if org_id:
            quota = await self.get_org_quota(org_id)
        elif team_id:
            quota = await self.get_team_quota(team_id)

        native = await NativeQuotaReservationStore(self.db_pool).usage(org_id=org_id, team_id=team_id)
        native_mb = native['bytes'] / 1048576
        if not quota:
            # Return default status (no quota set = unlimited)
            return {
                "quota_mb": None,
                "used_mb": native_mb,
                "legacy_used_mb": 0.0,
                "native_used_mb": native_mb,
                "remaining_mb": None,
                "usage_pct": 0.0,
                "at_soft_limit": False,
                "at_hard_limit": False,
                "has_quota": False,
            }

        quota_mb = quota.get("quota_mb", 0)
        legacy_mb = quota.get('used_mb', 0.0)
        used_mb = legacy_mb + native_mb
        soft_limit_pct = quota.get("soft_limit_pct", DEFAULT_SOFT_LIMIT_PCT)
        hard_limit_pct = quota.get("hard_limit_pct", DEFAULT_HARD_LIMIT_PCT)

        remaining_mb = max(0, quota_mb - used_mb)
        usage_pct = (used_mb / quota_mb * 100) if quota_mb > 0 else 0.0

        return {
            "quota_mb": quota_mb,
            "used_mb": used_mb,
            "legacy_used_mb": legacy_mb,
            "native_used_mb": native_mb,
            "remaining_mb": remaining_mb,
            "usage_pct": round(usage_pct, 2),
            "at_soft_limit": usage_pct >= soft_limit_pct,
            "at_hard_limit": usage_pct >= hard_limit_pct,
            "has_quota": True,
        }

    async def can_allocate(
        self,
        size_bytes: int,
        *,
        org_id: int | None = None,
        team_id: int | None = None,
    ) -> tuple[bool, str]:
        """
        Check if additional storage can be allocated.

        Args:
            size_bytes: Size to allocate in bytes
            org_id: Organization ID (optional)
            team_id: Team ID (optional)

        Returns:
            Tuple of (can_allocate, reason)
        """
        size_mb = size_bytes / (1024 * 1024)

        status = await self.check_quota_status(org_id=org_id, team_id=team_id)

        if not status["has_quota"]:
            return True, "No quota limit set"

        remaining_mb = status["remaining_mb"]
        if remaining_mb is None:
            return True, "No quota limit set"

        if size_mb > remaining_mb:
            return False, f"Insufficient storage quota. Need {size_mb:.2f} MB, only {remaining_mb:.2f} MB available"

        if status["at_hard_limit"]:
            return False, "Storage quota exceeded (at hard limit)"

        if status["at_soft_limit"]:
            return True, "Warning: Approaching storage limit (soft limit reached)"

        return True, "Quota check passed"


# Type alias for import convenience
