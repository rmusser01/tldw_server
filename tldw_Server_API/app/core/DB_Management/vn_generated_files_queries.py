"""VN registration SQL on a caller-owned AuthNZ connection.

The caller owns acquisition, transaction admission, commit and rollback. These
queries never open a pool connection or manage the transaction lifecycle.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass
class VNGeneratedFilesQueries:
    """Lock and locate VN generated files using bound data values only."""

    conn: Any
    postgres: bool

    async def lock_item(self, *, user_id: int, source_ref: str) -> None:
        """Hold the owner/item advisory lock until the caller's PostgreSQL commit."""
        if not self.postgres:
            # The owning SQLite transaction already uses BEGIN IMMEDIATE.
            return
        lock_key = int.from_bytes(
            hashlib.sha256(f"vn_asset_file:{user_id}:{source_ref}".encode()).digest()[:8],
            byteorder="big", signed=True,
        )
        await self.conn.fetchrow("SELECT pg_advisory_xact_lock($1)", lock_key)

    async def lock_quota_scopes(
        self, *, user_id: int, org_id: int | None, team_id: int | None,
    ) -> None:
        """Lock admission counters in user/org/team order on the same connection."""
        if not self.postgres:
            return
        await self.conn.fetchrow("SELECT id FROM users WHERE id = $1 FOR UPDATE", user_id)
        if org_id:
            await self.conn.fetchrow("SELECT id FROM storage_quotas WHERE org_id = $1 FOR UPDATE", org_id)
        if team_id:
            await self.conn.fetchrow("SELECT id FROM storage_quotas WHERE team_id = $1 FOR UPDATE", team_id)

    async def find_live_by_source_ref(
        self, *, user_id: int, source_feature: str, source_ref: str,
    ) -> dict[str, Any] | None:
        """Return the newest live file for the exact owner, feature and source."""
        if self.postgres:
            row = await self.conn.fetchrow(
                """
                SELECT * FROM generated_files
                WHERE user_id = $1 AND source_feature = $2 AND source_ref = $3
                  AND is_deleted = FALSE ORDER BY id DESC LIMIT 1
                """,
                user_id, source_feature, source_ref,
            )
            return dict(row) if row is not None else None
        cursor = await self.conn.execute(
            """
            SELECT * FROM generated_files
            WHERE user_id = ? AND source_feature = ? AND source_ref = ?
              AND is_deleted = 0 ORDER BY id DESC LIMIT 1
            """,
            (user_id, source_feature, source_ref),
        )
        row = await cursor.fetchone()
        return dict(zip((col[0] for col in cursor.description), row)) if row is not None else None

    async def find_live_by_storage_path(
        self, *, user_id: int, storage_path: str,
    ) -> dict[str, Any] | None:
        """Find the newest owned live byte reference before VN cleanup."""
        if self.postgres:
            row = await self.conn.fetchrow(
                """
                SELECT * FROM generated_files
                WHERE user_id = $1 AND storage_path = $2 AND is_deleted = FALSE
                ORDER BY id DESC LIMIT 1
                """,
                user_id, storage_path,
            )
            return dict(row) if row is not None else None
        cursor = await self.conn.execute(
            """
            SELECT * FROM generated_files
            WHERE user_id = ? AND storage_path = ? AND is_deleted = 0
            ORDER BY id DESC LIMIT 1
            """,
            (user_id, storage_path),
        )
        row = await cursor.fetchone()
        return dict(zip((col[0] for col in cursor.description), row)) if row is not None else None
