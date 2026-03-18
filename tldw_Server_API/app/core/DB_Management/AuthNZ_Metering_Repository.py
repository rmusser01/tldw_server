from __future__ import annotations

from typing import Any


class _AuthNZMeteringRepositoryBase:
    def __init__(self, *, db_pool: Any | None = None) -> None:
        self._db_pool = db_pool

    async def _get_db_pool(self) -> Any:
        if self._db_pool is None:
            from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

            self._db_pool = await get_db_pool()
        return self._db_pool

    @staticmethod
    def _is_postgres(conn: Any) -> bool:
        return hasattr(conn, "fetchrow")


class AuthNZUsageDailyRepository(_AuthNZMeteringRepositoryBase):
    @staticmethod
    def _is_missing_usage_column_error(exc: Exception) -> bool:
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
        if not raw_rows:
            return []
        columns = [col[0] for col in (description or [])]
        rows = [dict(zip(columns, row)) for row in raw_rows]
        if not include_bytes_in_total:
            for row in rows:
                row["bytes_in_total"] = 0
        return rows

    async def fetch_usage_for_date(self, target_date: str) -> list[dict[str, Any]]:
        pool = await self._get_db_pool()
        async with pool.acquire() as conn:
            if self._is_postgres(conn):
                try:
                    rows = await conn.fetch(
                        "SELECT user_id, requests, errors, bytes_total, "
                        "COALESCE(bytes_in_total, 0) AS bytes_in_total, latency_avg_ms "
                        "FROM usage_daily WHERE day = $1",
                        target_date,
                    )
                    return [dict(r) for r in rows]
                except Exception as exc:
                    if not self._is_missing_usage_column_error(exc):
                        raise
                    rows = await conn.fetch(
                        "SELECT user_id, requests, errors, bytes_total, latency_avg_ms "
                        "FROM usage_daily WHERE day = $1",
                        target_date,
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
    async def get_active_subscription_for_user(
        self,
        user_id: int,
    ) -> dict[str, Any] | None:
        pool = await self._get_db_pool()
        async with pool.acquire() as conn:
            if self._is_postgres(conn):
                row = await conn.fetchrow(
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
                    LIMIT 1
                    """,
                    user_id,
                )
                if row:
                    return dict(row)
                row = await conn.fetchrow(
                    """
                    SELECT os.stripe_customer_id,
                           os.stripe_subscription_id,
                           os.org_id
                    FROM org_subscriptions os
                    JOIN organizations o ON o.id = os.org_id
                    WHERE o.owner_user_id = $1
                      AND os.status = 'active'
                      AND os.stripe_subscription_id IS NOT NULL
                    LIMIT 1
                    """,
                    user_id,
                )
                return dict(row) if row else None

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
                LIMIT 1
                """,
                (user_id,),
            )
            row = await cur.fetchone()
            if row:
                columns = [col[0] for col in cur.description]
                return dict(zip(columns, row))

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
                LIMIT 1
                """,
                (user_id,),
            )
            row = await cur.fetchone()
            if not row:
                return None
            columns = [col[0] for col in cur.description]
            return dict(zip(columns, row))


class AuthNZMeteringSyncLogRepository(_AuthNZMeteringRepositoryBase):
    async def ensure_schema(self) -> None:
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
            await conn.commit()

    async def already_synced(
        self,
        *,
        user_id: int,
        day: str,
        subscription_id: str,
    ) -> bool:
        pool = await self._get_db_pool()
        async with pool.acquire() as conn:
            if self._is_postgres(conn):
                row = await conn.fetchval(
                    "SELECT 1 FROM metering_sync_log "
                    "WHERE user_id = $1 AND day = $2 AND stripe_subscription_id = $3",
                    user_id,
                    day,
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
        day: str,
        subscription_id: str,
        requests: int,
        bytes_total: int,
    ) -> None:
        pool = await self._get_db_pool()
        async with pool.acquire() as conn:
            if self._is_postgres(conn):
                await conn.execute(
                    "INSERT INTO metering_sync_log "
                    "(user_id, day, stripe_subscription_id, requests_synced, bytes_synced) "
                    "VALUES ($1, $2, $3, $4, $5) "
                    "ON CONFLICT (user_id, day, stripe_subscription_id) DO UPDATE "
                    "SET requests_synced = EXCLUDED.requests_synced, "
                    "    bytes_synced = EXCLUDED.bytes_synced, "
                    "    synced_at = CURRENT_TIMESTAMP",
                    user_id,
                    day,
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

    async def fetch_sync_totals(self, target_date: str) -> list[dict[str, Any]]:
        pool = await self._get_db_pool()
        async with pool.acquire() as conn:
            if self._is_postgres(conn):
                rows = await conn.fetch(
                    "SELECT user_id, stripe_subscription_id, requests_synced, bytes_synced "
                    "FROM metering_sync_log WHERE day = $1",
                    target_date,
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
