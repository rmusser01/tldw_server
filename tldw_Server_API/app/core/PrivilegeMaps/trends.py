from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.PrivilegeMaps.db_utils import execute_transaction_sql


@dataclass
class TrendBaseline:
    users: int
    endpoints: int
    scopes: int
    generated_at: datetime | None


class PrivilegeTrendStore:
    """Persists privilege summary history to support trend calculations."""

    def __init__(self, pool: DatabasePool | None = None) -> None:
        self._pool = pool
        self._initialized = False

    async def record_snapshot(
        self,
        *,
        scope: str,
        group_by: str,
        catalog_version: str,
        generated_at: datetime,
        buckets: Sequence[dict[str, Any]],
        team_id: str | None = None,
        org_id: str | None = None,
    ) -> None:
        if not buckets:
            return
        pool = await self._get_pool()
        await self._ensure_schema(pool)
        team_bucket = self._normalize_bucket(team_id, default="__none__")
        org_bucket = self._normalize_bucket(org_id, default="__global__")
        timestamp = self._to_iso(generated_at)

        async with pool.transaction() as conn:
            for bucket in buckets:
                key = str(bucket.get("key") or "").strip()
                if not key:
                    continue
                users = int(bucket.get("users") or 0)
                endpoints = int(bucket.get("endpoints") or 0)
                scopes = int(bucket.get("scopes") or 0)
                metadata = {
                    k: v
                    for k, v in bucket.items()
                    if k not in {"key", "users", "endpoints", "scopes"}
                }
                metadata_json = json.dumps(metadata, default=str) if metadata else None
                # Deduplicate exact timestamp snapshots for same bucket.
                await execute_transaction_sql(
                    pool,
                    conn,
                    """
                    DELETE FROM privilege_trend_history
                    WHERE scope = ?
                      AND group_by = ?
                      AND bucket_key = ?
                      AND team_bucket = ?
                      AND org_bucket = ?
                      AND generated_at = ?
                    """,
                    (
                        scope,
                        group_by,
                        key,
                        team_bucket,
                        org_bucket,
                        timestamp,
                    ),
                )
                await execute_transaction_sql(
                    pool,
                    conn,
                    """
                    INSERT INTO privilege_trend_history (
                        scope,
                        group_by,
                        bucket_key,
                        team_bucket,
                        org_bucket,
                        generated_at,
                        users,
                        endpoints,
                        scopes,
                        catalog_version,
                        metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scope,
                        group_by,
                        key,
                        team_bucket,
                        org_bucket,
                        timestamp,
                        users,
                        endpoints,
                        scopes,
                        catalog_version,
                        metadata_json,
                    ),
                )

    async def compute_trends(
        self,
        *,
        scope: str,
        group_by: str,
        bucket_counts: dict[str, dict[str, int]],
        window_start: datetime,
        window_end: datetime,
        team_id: str | None = None,
        org_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not bucket_counts:
            return []
        pool = await self._get_pool()
        await self._ensure_schema(pool)
        team_bucket = self._normalize_bucket(team_id, default="__none__")
        org_bucket = self._normalize_bucket(org_id, default="__global__")
        window_start_iso = self._to_iso(window_start)
        window_end_iso = self._to_iso(window_end)

        trends: list[dict[str, Any]] = []
        for key, counts in bucket_counts.items():
            baseline = await self._baseline_for_bucket(
                pool=pool,
                scope=scope,
                group_by=group_by,
                bucket_key=key,
                team_bucket=team_bucket,
                org_bucket=org_bucket,
                window_start_iso=window_start_iso,
            )
            if baseline is None:
                baseline = TrendBaseline(users=0, endpoints=0, scopes=0, generated_at=None)
            delta_users = int(counts.get("users", 0)) - baseline.users
            delta_endpoints = int(counts.get("endpoints", 0)) - baseline.endpoints
            delta_scopes = int(counts.get("scopes", 0)) - baseline.scopes
            trends.append(
                {
                    "key": key,
                    "window": {
                        "start": baseline.generated_at.isoformat() if baseline.generated_at else window_start_iso,
                        "end": window_end_iso,
                    },
                    "delta_users": delta_users,
                    "delta_endpoints": delta_endpoints,
                    "delta_scopes": delta_scopes,
                }
            )
        return trends

    async def _baseline_for_bucket(
        self,
        *,
        pool: DatabasePool,
        scope: str,
        group_by: str,
        bucket_key: str,
        team_bucket: str,
        org_bucket: str,
        window_start_iso: str,
    ) -> TrendBaseline | None:
        # Prefer most recent record at or before window start.
        row = await pool.fetchone(
            """
            SELECT users, endpoints, scopes, generated_at
            FROM privilege_trend_history
            WHERE scope = ?
              AND group_by = ?
              AND bucket_key = ?
              AND team_bucket = ?
              AND org_bucket = ?
              AND generated_at <= ?
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            (
                scope,
                group_by,
                bucket_key,
                team_bucket,
                org_bucket,
                window_start_iso,
            ),
        )
        if not row:
            # Fallback: earliest record after the window start.
            row = await pool.fetchone(
                """
                SELECT users, endpoints, scopes, generated_at
                FROM privilege_trend_history
                WHERE scope = ?
                  AND group_by = ?
                  AND bucket_key = ?
                  AND team_bucket = ?
                  AND org_bucket = ?
                  AND generated_at > ?
                ORDER BY generated_at ASC
                LIMIT 1
                """,
                (
                    scope,
                    group_by,
                    bucket_key,
                    team_bucket,
                    org_bucket,
                    window_start_iso,
                ),
            )
            if not row:
                return None
        record = self._row_to_dict(row)
        generated_at = self._parse_datetime(record.get("generated_at"))
        return TrendBaseline(
            users=int(record.get("users") or 0),
            endpoints=int(record.get("endpoints") or 0),
            scopes=int(record.get("scopes") or 0),
            generated_at=generated_at,
        )

    async def purge_older_than(self, *, cutoff: datetime) -> int:
        """Remove history entries older than the supplied cutoff."""
        pool = await self._get_pool()
        await self._ensure_schema(pool)
        cutoff_iso = self._to_iso(cutoff)
        async with pool.transaction() as conn:
            result = await execute_transaction_sql(
                pool,
                conn,
                "DELETE FROM privilege_trend_history WHERE generated_at < ?",
                (cutoff_iso,),
            )
            try:
                # sqlite3 cursor returns rowcount; asyncpg returns str
                return int(getattr(result, "rowcount", int(str(result).split()[-1])))
            except Exception:
                return 0

    async def _get_pool(self) -> DatabasePool:
        if self._pool is None:
            self._pool = await get_db_pool()
        return self._pool

    async def _ensure_schema(self, pool: DatabasePool) -> None:
        if self._initialized:
            return
        async with pool.transaction() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS privilege_trend_history (
                    scope TEXT NOT NULL,
                    group_by TEXT NOT NULL,
                    bucket_key TEXT NOT NULL,
                    team_bucket TEXT NOT NULL,
                    org_bucket TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    users INTEGER NOT NULL,
                    endpoints INTEGER NOT NULL,
                    scopes INTEGER NOT NULL,
                    catalog_version TEXT NOT NULL,
                    metadata_json TEXT,
                    PRIMARY KEY (scope, group_by, bucket_key, team_bucket, org_bucket, generated_at)
                )
                """
            )
        try:
            async with pool.transaction() as conn:
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_priv_trend_lookup
                    ON privilege_trend_history(scope, group_by, team_bucket, org_bucket, generated_at)
                    """
                )
        except Exception as exc:
            logger.debug("Privilege trend index creation skipped: {}", exc)
        self._initialized = True

    @staticmethod
    def _normalize_bucket(value: str | None, *, default: str) -> str:
        raw = str(value).strip() if value else ""
        return raw or default

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        if row is None:
            return {}
        if isinstance(row, dict):
            return row
        if hasattr(row, "keys"):
            return {key: row[key] for key in row}
        if hasattr(row, "_mapping"):
            return dict(row._mapping)  # type: ignore[attr-defined]
        return {}

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None

    @staticmethod
    def _to_iso(value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()


@lru_cache
def get_privilege_trend_store() -> PrivilegeTrendStore:
    return PrivilegeTrendStore()
