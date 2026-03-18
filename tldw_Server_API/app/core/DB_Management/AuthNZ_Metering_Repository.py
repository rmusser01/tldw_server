"""Persistence boundary for AuthNZ metering reads and sync-log writes.

The Stripe metering service delegates all database access in this module so the
service layer only coordinates usage lookup, subscription resolution, Stripe
calls, and reconciliation. Each repository normalizes backend-specific row
shapes before returning dictionaries to the orchestration layer.
"""

from __future__ import annotations

import inspect
from datetime import date, datetime
from typing import Any


class DuplicateActiveSubscriptionError(RuntimeError):
    """Raised when a user resolves to more than one active billing subscription."""


class _AuthNZMeteringRepositoryBase:
    """Shared pool-loading helpers for AuthNZ metering repositories."""

    def __init__(self, *, db_pool: Any | None = None) -> None:
        """Store an optional injected pool for tests or alternate composition."""
        self._db_pool = db_pool

    async def _get_db_pool(self) -> Any:
        """Lazily resolve the shared AuthNZ DB pool on first use."""
        if self._db_pool is None:
            from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

            self._db_pool = await get_db_pool()
        return self._db_pool

    @staticmethod
    def _is_postgres(conn: Any) -> bool:
        """Detect whether the acquired connection exposes asyncpg-style methods."""
        return hasattr(conn, "fetchrow")

    @staticmethod
    def _coerce_day(value: str | date) -> date:
        """Return a concrete date object for PostgreSQL DATE bindings."""
        if isinstance(value, date):
            return value
        return datetime.strptime(str(value), "%Y-%m-%d").date()

    @staticmethod
    async def _sqlite_fetch_rows(cur: Any) -> list[Any]:
        """Fetch SQLite cursor rows from adapters that expose fetchall or fetchone."""
        fetchall = getattr(cur, "fetchall", None)
        if callable(fetchall):
            rows = fetchall()
            if inspect.isawaitable(rows):
                rows = await rows
            return list(rows)

        fetchone = getattr(cur, "fetchone", None)
        if not callable(fetchone):
            return []
        row = fetchone()
        if inspect.isawaitable(row):
            row = await row
        return [] if row is None else [row]


class AuthNZUsageDailyRepository(_AuthNZMeteringRepositoryBase):
    """Read normalized daily usage rows from the AuthNZ billing database."""

    @staticmethod
    def _is_missing_usage_column_error(exc: Exception) -> bool:
        """Identify legacy-schema errors where `bytes_in_total` is unavailable."""
        message = str(exc).lower()
        return "bytes_in_total" in message and (
            "no such column" in message
            or "does not exist" in message
            or "undefined column" in message
        )

    @staticmethod
    def _sqlite_rows_to_dicts(
        raw_rows: list[tuple[Any, ...]],
        description: list[tuple[Any, ...]] | None,
        *,
        include_bytes_in_total: bool,
    ) -> list[dict[str, Any]]:
        """Map SQLite cursor output to normalized usage dictionaries."""
        if not raw_rows:
            return []
        columns = [col[0] for col in (description or [])]
        rows = [dict(zip(columns, row)) for row in raw_rows]
        if not include_bytes_in_total:
            for row in rows:
                row["bytes_in_total"] = 0
        return rows

    async def fetch_usage_for_date(self, target_date: str | date) -> list[dict[str, Any]]:
        """Return per-user usage rows for a billing date with legacy fallback."""
        pool = await self._get_db_pool()
        async with pool.acquire() as conn:
            if self._is_postgres(conn):
                pg_day = self._coerce_day(target_date)
                try:
                    rows = await conn.fetch(
                        "SELECT user_id, requests, errors, bytes_total, "
                        "COALESCE(bytes_in_total, 0) AS bytes_in_total, latency_avg_ms "
                        "FROM usage_daily WHERE day = $1",
                        pg_day,
                    )
                    return [dict(r) for r in rows]
                except Exception as exc:
                    if not self._is_missing_usage_column_error(exc):
                        raise
                    rows = await conn.fetch(
                        "SELECT user_id, requests, errors, bytes_total, latency_avg_ms "
                        "FROM usage_daily WHERE day = $1",
                        pg_day,
                    )
                    legacy_rows = [dict(r) for r in rows]
                    for row in legacy_rows:
                        row["bytes_in_total"] = 0
                    return legacy_rows

            try:
                cur = await conn.execute(
                    "SELECT user_id, requests, errors, bytes_total, "
                    "COALESCE(bytes_in_total, 0) AS bytes_in_total, latency_avg_ms "
                    "FROM usage_daily WHERE day = ?",
                    (target_date,),
                )
                raw_rows = await cur.fetchall()
                return self._sqlite_rows_to_dicts(
                    raw_rows,
                    cur.description,
                    include_bytes_in_total=True,
                )
            except Exception as exc:
                if not self._is_missing_usage_column_error(exc):
                    raise
                cur = await conn.execute(
                    "SELECT user_id, requests, errors, bytes_total, latency_avg_ms "
                    "FROM usage_daily WHERE day = ?",
                    (target_date,),
                )
                raw_rows = await cur.fetchall()
                return self._sqlite_rows_to_dicts(
                    raw_rows,
                    cur.description,
                    include_bytes_in_total=False,
                )


class AuthNZBillingSubscriptionRepository(_AuthNZMeteringRepositoryBase):
    """Resolve the active Stripe subscription that should receive metered usage."""

    @staticmethod
    def _raise_on_duplicate_matches(
        rows: list[dict[str, Any]],
        *,
        user_id: int,
        source: str,
    ) -> None:
        """Fail fast when a lookup branch returns more than one active subscription."""
        if len(rows) > 1:
            raise DuplicateActiveSubscriptionError(
                f"Found multiple active subscriptions for user {user_id} via {source} lookup"
            )

    @staticmethod
    def _coalesce_unique_match(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Collapse member/owner matches into a single unique subscription row."""
        unique_rows: dict[tuple[Any, Any], dict[str, Any]] = {}
        for row in rows:
            unique_rows[(row.get("org_id"), row.get("stripe_subscription_id"))] = row
        if not unique_rows:
            return None
        if len(unique_rows) > 1:
            raise DuplicateActiveSubscriptionError(
                "Found multiple active subscriptions for a single user across membership ownership lookups"
            )
        return next(iter(unique_rows.values()))

    async def get_active_subscription_for_user(
        self,
        user_id: int,
    ) -> dict[str, Any] | None:
        """Return the active subscription for a member or org owner, if any."""
        pool = await self._get_db_pool()
        async with pool.acquire() as conn:
            if self._is_postgres(conn):
                member_rows = [
                    dict(row)
                    for row in await conn.fetch(
                        """
                        SELECT os.stripe_customer_id,
                               os.stripe_subscription_id,
                               os.org_id
                        FROM org_subscriptions os
                        JOIN org_members om ON om.org_id = os.org_id
                        WHERE om.user_id = $1
                          AND om.status = 'active'
                          AND os.status = 'active'
                          AND os.stripe_subscription_id IS NOT NULL
                        LIMIT 2
                        """,
                        user_id,
                    )
                ]
                self._raise_on_duplicate_matches(member_rows, user_id=user_id, source="member")
                owner_rows = [
                    dict(row)
                    for row in await conn.fetch(
                        """
                        SELECT os.stripe_customer_id,
                               os.stripe_subscription_id,
                               os.org_id
                        FROM org_subscriptions os
                        JOIN organizations o ON o.id = os.org_id
                        WHERE o.owner_user_id = $1
                          AND os.status = 'active'
                          AND os.stripe_subscription_id IS NOT NULL
                        LIMIT 2
                        """,
                        user_id,
                    )
                ]
                self._raise_on_duplicate_matches(owner_rows, user_id=user_id, source="owner")
                return self._coalesce_unique_match(member_rows + owner_rows)

            cur = await conn.execute(
                """
                SELECT os.stripe_customer_id,
                       os.stripe_subscription_id,
                       os.org_id
                FROM org_subscriptions os
                JOIN org_members om ON om.org_id = os.org_id
                WHERE om.user_id = ?
                  AND om.status = 'active'
                  AND os.status = 'active'
                  AND os.stripe_subscription_id IS NOT NULL
                LIMIT 2
                """,
                (user_id,),
            )
            member_rows_raw = await self._sqlite_fetch_rows(cur)
            member_columns = [col[0] for col in cur.description]
            member_rows = [dict(zip(member_columns, row)) for row in member_rows_raw]
            self._raise_on_duplicate_matches(member_rows, user_id=user_id, source="member")

            cur = await conn.execute(
                """
                SELECT os.stripe_customer_id,
                       os.stripe_subscription_id,
                       os.org_id
                FROM org_subscriptions os
                JOIN organizations o ON o.id = os.org_id
                WHERE o.owner_user_id = ?
                  AND os.status = 'active'
                  AND os.stripe_subscription_id IS NOT NULL
                LIMIT 2
                """,
                (user_id,),
            )
            owner_rows_raw = await self._sqlite_fetch_rows(cur)
            owner_columns = [col[0] for col in cur.description]
            owner_rows = [dict(zip(owner_columns, row)) for row in owner_rows_raw]
            self._raise_on_duplicate_matches(owner_rows, user_id=user_id, source="owner")
            return self._coalesce_unique_match(member_rows + owner_rows)


class AuthNZMeteringSyncLogRepository(_AuthNZMeteringRepositoryBase):
    """Own metering sync-log schema management and sync-state persistence."""

    async def ensure_schema(self) -> None:
        """Create the metering sync-log table if it does not already exist."""
        pool = await self._get_db_pool()
        async with pool.acquire() as conn:
            if self._is_postgres(conn):
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS metering_sync_log (
                        user_id INTEGER NOT NULL,
                        day DATE NOT NULL,
                        stripe_subscription_id TEXT NOT NULL,
                        requests_synced INTEGER DEFAULT 0,
                        bytes_synced BIGINT DEFAULT 0,
                        synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_id, day, stripe_subscription_id)
                    )
                    """
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_metering_sync_log_day "
                    "ON metering_sync_log (day)"
                )
                return

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metering_sync_log (
                    user_id INTEGER NOT NULL,
                    day DATE NOT NULL,
                    stripe_subscription_id TEXT NOT NULL,
                    requests_synced INTEGER DEFAULT 0,
                    bytes_synced INTEGER DEFAULT 0,
                    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, day, stripe_subscription_id)
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metering_sync_log_day "
                "ON metering_sync_log (day)"
            )
            await conn.commit()

    async def already_synced(
        self,
        *,
        user_id: int,
        day: str | date,
        subscription_id: str,
    ) -> bool:
        """Check whether a subscription/day pair has already been synced."""
        pool = await self._get_db_pool()
        async with pool.acquire() as conn:
            if self._is_postgres(conn):
                pg_day = self._coerce_day(day)
                row = await conn.fetchval(
                    "SELECT 1 FROM metering_sync_log "
                    "WHERE user_id = $1 AND day = $2 AND stripe_subscription_id = $3",
                    user_id,
                    pg_day,
                    subscription_id,
                )
                return row is not None

            cur = await conn.execute(
                "SELECT 1 FROM metering_sync_log "
                "WHERE user_id = ? AND day = ? AND stripe_subscription_id = ?",
                (user_id, day, subscription_id),
            )
            return (await cur.fetchone()) is not None

    async def record_sync(
        self,
        *,
        user_id: int,
        day: str | date,
        subscription_id: str,
        requests: int,
        bytes_total: int,
    ) -> None:
        """Persist the latest synced totals for a user/subscription/day tuple."""
        pool = await self._get_db_pool()
        async with pool.acquire() as conn:
            if self._is_postgres(conn):
                pg_day = self._coerce_day(day)
                await conn.execute(
                    "INSERT INTO metering_sync_log "
                    "(user_id, day, stripe_subscription_id, requests_synced, bytes_synced) "
                    "VALUES ($1, $2, $3, $4, $5) "
                    "ON CONFLICT (user_id, day, stripe_subscription_id) DO UPDATE "
                    "SET requests_synced = EXCLUDED.requests_synced, "
                    "    bytes_synced = EXCLUDED.bytes_synced, "
                    "    synced_at = CURRENT_TIMESTAMP",
                    user_id,
                    pg_day,
                    subscription_id,
                    requests,
                    bytes_total,
                )
                return

            await conn.execute(
                "INSERT OR REPLACE INTO metering_sync_log "
                "(user_id, day, stripe_subscription_id, requests_synced, bytes_synced) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, day, subscription_id, requests, bytes_total),
            )
            await conn.commit()

    async def fetch_sync_totals(self, target_date: str | date) -> list[dict[str, Any]]:
        """Return previously recorded sync totals for reconciliation output."""
        pool = await self._get_db_pool()
        async with pool.acquire() as conn:
            if self._is_postgres(conn):
                pg_day = self._coerce_day(target_date)
                rows = await conn.fetch(
                    "SELECT user_id, stripe_subscription_id, requests_synced, bytes_synced "
                    "FROM metering_sync_log WHERE day = $1",
                    pg_day,
                )
                return [dict(r) for r in rows]

            cur = await conn.execute(
                "SELECT user_id, stripe_subscription_id, requests_synced, bytes_synced "
                "FROM metering_sync_log WHERE day = ?",
                (target_date,),
            )
            raw = await cur.fetchall()
            if not raw:
                return []
            columns = [col[0] for col in cur.description]
            return [dict(zip(columns, row)) for row in raw]
