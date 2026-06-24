from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool


@dataclass(frozen=True)
class LedgerEntry:
    entity_scope: str
    entity_value: str
    category: str
    units: int
    op_id: str
    occurred_at: datetime


@dataclass(frozen=True)
class DailyCapConsumeResult:
    """Result of an atomic daily-cap consume attempt."""

    allowed: bool
    used: int
    inserted: bool


class ResourceDailyLedger:
    """
    Generic daily ledger for resource accounting (minutes, tokens_per_day, etc.).

    Uses the AuthNZ DatabasePool for persistence. Methods are safe for both
    PostgreSQL and SQLite backends.
    """

    def __init__(self, db_pool: DatabasePool | None = None) -> None:
        self.db_pool = db_pool
        self._initialized = False

    async def _using_postgres_backend(self) -> bool:
        """Return True when this ledger's DatabasePool is backed by PostgreSQL."""
        if self.db_pool is None:
            self.db_pool = await get_db_pool()
        try:
            await self.db_pool.initialize()
        except Exception as exc:
            logger.debug("ResourceDailyLedger: db_pool initialization failed; assuming SQLite: {}", exc)
            return False
        return getattr(self.db_pool, "pool", None) is not None

    async def initialize(self) -> None:
        if self._initialized:
            return
        if not self.db_pool:
            self.db_pool = await get_db_pool()

        is_pg = await self._using_postgres_backend()
        try:
            async with self.db_pool.transaction() as conn:
                if is_pg:
                    await conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS resource_daily_ledger (
                          id BIGSERIAL PRIMARY KEY,
                          day_utc DATE NOT NULL,
                          entity_scope TEXT NOT NULL,
                          entity_value TEXT NOT NULL,
                          category TEXT NOT NULL,
                          units BIGINT NOT NULL CHECK (units >= 0),
                          op_id TEXT NOT NULL,
                          occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                        """
                    )
                    await conn.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS uq_ledger_op ON resource_daily_ledger (day_utc, entity_scope, entity_value, category, op_id)"
                    )
                    await conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_ledger_lookup ON resource_daily_ledger (entity_scope, entity_value, category, day_utc)"
                    )
                else:
                    await conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS resource_daily_ledger (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          day_utc TEXT NOT NULL,
                          entity_scope TEXT NOT NULL,
                          entity_value TEXT NOT NULL,
                          category TEXT NOT NULL,
                          units INTEGER NOT NULL,
                          op_id TEXT NOT NULL,
                          occurred_at TEXT NOT NULL,
                          created_at TEXT NOT NULL
                        )
                        """
                    )
                    await conn.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS uq_ledger_op ON resource_daily_ledger (day_utc, entity_scope, entity_value, category, op_id)"
                    )
                    await conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_ledger_lookup ON resource_daily_ledger (entity_scope, entity_value, category, day_utc)"
                    )
            self._initialized = True
            logger.info("ResourceDailyLedger initialized (table ensured)")
        except Exception as e:
            logger.error(f"ResourceDailyLedger initialize failed: {e}")
            raise

    @staticmethod
    def _to_day_utc(dt: datetime | None = None) -> str:
        d = (dt or datetime.now(timezone.utc)).astimezone(timezone.utc)
        return d.strftime("%Y-%m-%d")

    @staticmethod
    def _advisory_lock_key(*parts: str) -> int:
        """Return a stable signed 64-bit key for Postgres advisory locks."""
        material = "\x1f".join(str(part) for part in parts).encode("utf-8", errors="surrogatepass")
        digest = hashlib.blake2b(material, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=True)

    @staticmethod
    def _affected_rows(result: Any) -> int | None:
        """Best-effort affected row count across asyncpg and aiosqlite results."""
        rowcount = getattr(result, "rowcount", None)
        if isinstance(rowcount, int):
            return rowcount
        text = str(result or "")
        try:
            tail = text.rsplit(" ", 1)[-1]
            if tail.isdigit():
                return int(tail)
        except (IndexError, TypeError, ValueError):
            return None
        return None

    @staticmethod
    async def _fetchval_from_conn(conn: Any, query: str, *args: Any, is_pg: bool) -> Any:
        """Fetch one scalar using either asyncpg or the SQLite transaction shim."""
        if is_pg:
            return await conn.fetchval(query, *args)
        cursor = await conn.execute(query, *args)
        row = await cursor.fetchone()
        return row[0] if row else None

    async def consume_if_within_cap(
        self,
        entry: LedgerEntry,
        *,
        daily_cap: int,
        day_utc: str | None = None,
    ) -> DailyCapConsumeResult:
        """
        Idempotently insert ``entry`` only when the daily cap has headroom.

        The check and insert are serialized per (day, scope, value, category):
        Postgres uses an advisory transaction lock, while SQLite uses the
        AuthNZ transaction wrapper's ``BEGIN IMMEDIATE`` write transaction.
        """
        if not self._initialized:
            await self.initialize()

        cap = int(daily_cap)
        units = max(0, int(entry.units))
        day = str(day_utc or self._to_day_utc(entry.occurred_at))
        is_pg = await self._using_postgres_backend()

        async with self.db_pool.transaction() as conn:
            if is_pg:
                day_param: date = date.fromisoformat(day)
                await conn.execute(
                    "SELECT pg_advisory_xact_lock($1)",
                    self._advisory_lock_key(day, entry.entity_scope, entry.entity_value, entry.category),
                )
                existing_q = (
                    "SELECT units FROM resource_daily_ledger "
                    "WHERE day_utc = $1 AND entity_scope = $2 AND entity_value = $3 AND category = $4 AND op_id = $5"
                )
                total_q = (
                    "SELECT COALESCE(SUM(units), 0) FROM resource_daily_ledger "
                    "WHERE day_utc = $1 AND entity_scope = $2 AND entity_value = $3 AND category = $4"
                )
                insert_q = (
                    "INSERT INTO resource_daily_ledger "
                    "(day_utc, entity_scope, entity_value, category, units, op_id, occurred_at, created_at) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, NOW()) "
                    "ON CONFLICT (day_utc, entity_scope, entity_value, category, op_id) DO NOTHING"
                )
                params = (day_param, entry.entity_scope, entry.entity_value, entry.category)
                existing = await conn.fetchval(existing_q, *params, entry.op_id)
                used = int(await conn.fetchval(total_q, *params) or 0)
                if existing is not None:
                    return DailyCapConsumeResult(allowed=True, used=used, inserted=False)
                if max(0, cap - used) < units:
                    return DailyCapConsumeResult(allowed=False, used=used, inserted=False)
                result = await conn.execute(
                    insert_q,
                    day_param,
                    entry.entity_scope,
                    entry.entity_value,
                    entry.category,
                    units,
                    entry.op_id,
                    entry.occurred_at,
                )
                inserted = (self._affected_rows(result) or 0) > 0
                used_after = used + units if inserted else int(await conn.fetchval(total_q, *params) or used)
                return DailyCapConsumeResult(allowed=True, used=used_after, inserted=inserted)

            existing_q = (
                "SELECT units FROM resource_daily_ledger "
                "WHERE day_utc = ? AND entity_scope = ? AND entity_value = ? AND category = ? AND op_id = ?"
            )
            total_q = (
                "SELECT COALESCE(SUM(units), 0) FROM resource_daily_ledger "
                "WHERE day_utc = ? AND entity_scope = ? AND entity_value = ? AND category = ?"
            )
            insert_q = (
                "INSERT OR IGNORE INTO resource_daily_ledger "
                "(day_utc, entity_scope, entity_value, category, units, op_id, occurred_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))"
            )
            params = (day, entry.entity_scope, entry.entity_value, entry.category)
            existing = await self._fetchval_from_conn(conn, existing_q, *params, entry.op_id, is_pg=False)
            used = int(await self._fetchval_from_conn(conn, total_q, *params, is_pg=False) or 0)
            if existing is not None:
                return DailyCapConsumeResult(allowed=True, used=used, inserted=False)
            if max(0, cap - used) < units:
                return DailyCapConsumeResult(allowed=False, used=used, inserted=False)
            result = await conn.execute(
                insert_q,
                day,
                entry.entity_scope,
                entry.entity_value,
                entry.category,
                units,
                entry.op_id,
                entry.occurred_at.isoformat(),
            )
            inserted = (self._affected_rows(result) or 0) > 0
            used_after = used + units if inserted else int(await self._fetchval_from_conn(conn, total_q, *params, is_pg=False) or used)
            return DailyCapConsumeResult(allowed=True, used=used_after, inserted=inserted)

    async def add(self, entry: LedgerEntry) -> bool:
        """
        Add a ledger entry (idempotent on (day_utc, scope, value, category, op_id)).
        Returns True if inserted; False if already present.
        """
        if not self._initialized:
            await self.initialize()

        day = self._to_day_utc(entry.occurred_at)
        is_pg = await self._using_postgres_backend()
        try:
            if is_pg:
                q = (
                    "INSERT INTO resource_daily_ledger (day_utc, entity_scope, entity_value, category, units, op_id, occurred_at, created_at) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, NOW()) ON CONFLICT (day_utc, entity_scope, entity_value, category, op_id) DO NOTHING"
                )
                # asyncpg expects a Python date for DATE columns
                day_param: date = date.fromisoformat(day)
                res = await self.db_pool.execute(
                    q,
                    day_param,
                    entry.entity_scope,
                    entry.entity_value,
                    entry.category,
                    int(entry.units),
                    entry.op_id,
                    entry.occurred_at,
                )
                # asyncpg returns 'INSERT 0 1' on insert; 'INSERT 0 0' on conflict/no-op
                return str(res).endswith(" 1")
            else:
                # Robust idempotency for SQLite: check existence before insert.
                exists_q = (
                    "SELECT 1 FROM resource_daily_ledger WHERE day_utc = ? AND entity_scope = ? AND entity_value = ? AND category = ? AND op_id = ? LIMIT 1"
                )
                exists = await self.db_pool.fetchval(
                    exists_q,
                    day,
                    entry.entity_scope,
                    entry.entity_value,
                    entry.category,
                    entry.op_id,
                )
                if exists:
                    return False
                q = (
                    "INSERT INTO resource_daily_ledger (day_utc, entity_scope, entity_value, category, units, op_id, occurred_at, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))"
                )
                await self.db_pool.execute(
                    q,
                    day,
                    entry.entity_scope,
                    entry.entity_value,
                    entry.category,
                    int(entry.units),
                    entry.op_id,
                    entry.occurred_at.isoformat(),
                )
                return True
        except Exception as e:
            logger.error(f"ResourceDailyLedger.add failed: {e}")
            raise

    async def consume_if_available(self, entry: LedgerEntry, daily_cap: int) -> tuple[bool, int]:
        """
        Add a ledger entry only when it fits within the daily cap.

        Returns ``(allowed, remaining_after_units)``. Duplicate ``op_id`` values
        are treated as already consumed and return the current remaining value.
        """
        if not self._initialized:
            await self.initialize()

        day = self._to_day_utc(entry.occurred_at)
        cap = int(max(0, daily_cap))
        units = int(max(0, entry.units))
        is_pg = await self._using_postgres_backend()
        try:
            async with self.db_pool.transaction() as conn:
                if is_pg:
                    day_param: date = date.fromisoformat(day)
                    lock_key = (
                        f"{entry.entity_scope}:{entry.entity_value}:"
                        f"{entry.category}:{day}"
                    )
                    await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", lock_key)
                    existing = await conn.fetchval(
                        """
                        SELECT 1 FROM resource_daily_ledger
                        WHERE day_utc = $1 AND entity_scope = $2 AND entity_value = $3
                          AND category = $4 AND op_id = $5
                        LIMIT 1
                        """,
                        day_param,
                        entry.entity_scope,
                        entry.entity_value,
                        entry.category,
                        entry.op_id,
                    )
                    used = int(
                        await conn.fetchval(
                            """
                            SELECT COALESCE(SUM(units), 0)
                            FROM resource_daily_ledger
                            WHERE day_utc = $1 AND entity_scope = $2 AND entity_value = $3 AND category = $4
                            """,
                            day_param,
                            entry.entity_scope,
                            entry.entity_value,
                            entry.category,
                        )
                        or 0
                    )
                    if existing:
                        return True, max(0, cap - used)
                    remaining_before = max(0, cap - used)
                    if units > remaining_before:
                        return False, remaining_before
                    await conn.execute(
                        """
                        INSERT INTO resource_daily_ledger
                          (day_utc, entity_scope, entity_value, category, units, op_id, occurred_at, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                        ON CONFLICT (day_utc, entity_scope, entity_value, category, op_id) DO NOTHING
                        """,
                        day_param,
                        entry.entity_scope,
                        entry.entity_value,
                        entry.category,
                        units,
                        entry.op_id,
                        entry.occurred_at,
                    )
                    return True, max(0, remaining_before - units)

                existing_cursor = await conn.execute(
                    """
                    SELECT 1 FROM resource_daily_ledger
                    WHERE day_utc = ? AND entity_scope = ? AND entity_value = ?
                      AND category = ? AND op_id = ?
                    LIMIT 1
                    """,
                    day,
                    entry.entity_scope,
                    entry.entity_value,
                    entry.category,
                    entry.op_id,
                )
                existing = await existing_cursor.fetchone()
                used_cursor = await conn.execute(
                    """
                    SELECT COALESCE(SUM(units), 0)
                    FROM resource_daily_ledger
                    WHERE day_utc = ? AND entity_scope = ? AND entity_value = ? AND category = ?
                    """,
                    day,
                    entry.entity_scope,
                    entry.entity_value,
                    entry.category,
                )
                used_row = await used_cursor.fetchone()
                used = int((used_row[0] if used_row else 0) or 0)
                if existing:
                    return True, max(0, cap - used)
                remaining_before = max(0, cap - used)
                if units > remaining_before:
                    return False, remaining_before
                await conn.execute(
                    """
                    INSERT INTO resource_daily_ledger
                      (day_utc, entity_scope, entity_value, category, units, op_id, occurred_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    """,
                    day,
                    entry.entity_scope,
                    entry.entity_value,
                    entry.category,
                    units,
                    entry.op_id,
                    entry.occurred_at.isoformat(),
                )
                return True, max(0, remaining_before - units)
        except Exception as e:
            logger.error(f"ResourceDailyLedger.consume_if_available failed: {e}")
            raise

    async def add_if_within_daily_cap(self, entry: LedgerEntry, daily_cap: int) -> tuple[bool, int]:
        """Backward-compatible alias for atomic capped ledger consumption."""
        return await self.consume_if_available(entry, daily_cap=daily_cap)

    async def total_for_day(self, entity_scope: str, entity_value: str, category: str, day_utc: str | None = None) -> int:
        if not self._initialized:
            await self.initialize()
        day = day_utc or self._to_day_utc()
        try:
            q = (
                "SELECT COALESCE(SUM(units), 0) FROM resource_daily_ledger WHERE day_utc = ? AND entity_scope = ? AND entity_value = ? AND category = ?"
            )
            # DatabasePool will adapt '?' to '$N' when using Postgres; for Postgres send a Python date
            if await self._using_postgres_backend():
                day_param: date = date.fromisoformat(day)
                val = await self.db_pool.fetchval(q, day_param, entity_scope, entity_value, category)
            else:
                val = await self.db_pool.fetchval(q, day, entity_scope, entity_value, category)
            return int(val or 0)
        except Exception as e:
            logger.error(f"ResourceDailyLedger.total_for_day failed: {e}")
            raise

    async def remaining_for_day(
        self,
        entity_scope: str,
        entity_value: str,
        category: str,
        daily_cap: int,
        day_utc: str | None = None,
    ) -> int:
        """
        Convenience helper: returns max(0, daily_cap - total_for_day(...)).
        """
        used = await self.total_for_day(entity_scope, entity_value, category, day_utc)
        rem = int(daily_cap) - int(used)
        return rem if rem > 0 else 0

    async def peek_range(
        self,
        entity_scope: str,
        entity_value: str,
        category: str,
        start_day_utc: str,
        end_day_utc: str,
    ) -> dict[str, Any]:
        """
        Return daily totals and grand total for an inclusive UTC day range.

        Example return:
        {"days": [{"day_utc": "2025-01-01", "units": 5}, ...], "total": 12}
        """
        if not self._initialized:
            await self.initialize()
        try:
            q = (
                "SELECT day_utc, COALESCE(SUM(units), 0) AS units "
                "FROM resource_daily_ledger "
                "WHERE entity_scope = ? AND entity_value = ? AND category = ? AND day_utc BETWEEN ? AND ? "
                "GROUP BY day_utc ORDER BY day_utc"
            )
            if await self._using_postgres_backend():
                start_param: date = date.fromisoformat(start_day_utc)
                end_param: date = date.fromisoformat(end_day_utc)
                rows = await self.db_pool.fetchall(
                    q, entity_scope, entity_value, category, start_param, end_param
                )
            else:
                rows = await self.db_pool.fetchall(
                    q, entity_scope, entity_value, category, start_day_utc, end_day_utc
                )
            days: list[dict[str, Any]] = []
            total = 0
            for r in rows:
                # rows are dicts (PG) or aiosqlite.Row
                d = r["day_utc"] if isinstance(r, dict) else r[0]
                u = int(r["units"] if isinstance(r, dict) else r[1] or 0)
                days.append({"day_utc": str(d), "units": u})
                total += u
            return {"days": days, "total": total}
        except Exception as e:
            logger.error(f"ResourceDailyLedger.peek_range failed: {e}")
            raise
