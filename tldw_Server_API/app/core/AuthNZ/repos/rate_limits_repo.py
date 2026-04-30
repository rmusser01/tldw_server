from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool


@dataclass
class AuthnzRateLimitsRepo:
    """
    Repository for AuthNZ rate-limiter storage.

    This repo encapsulates common read/write paths for ``rate_limits``,
    ``failed_attempts``, and ``account_lockouts`` where backend-specific
    SQL is required, so higher-level logic can remain dialect-agnostic.
    """

    db_pool: DatabasePool

    def _is_postgres_backend(self) -> bool:
        """Return True when the underlying DatabasePool is using PostgreSQL."""
        return bool(getattr(self.db_pool, "pool", None))

    async def _ensure_sqlite_account_lockouts_schema(self, conn: Any) -> None:
        """Upgrade legacy identifier-only lockout tables to attempt-type scope (SQLite)."""
        lockout_ddl = """
            CREATE TABLE IF NOT EXISTS account_lockouts (
                identifier TEXT NOT NULL,
                attempt_type TEXT NOT NULL,
                locked_until TEXT NOT NULL,
                reason TEXT,
                PRIMARY KEY (identifier, attempt_type)
            )
            """
        cursor = await conn.execute("PRAGMA table_info(account_lockouts)")
        rows = await cursor.fetchall()
        if not rows:
            # Table does not exist yet; create it fresh.
            await conn.execute(lockout_ddl)
            return

        column_names = {row[1] if not isinstance(row, dict) else row["name"] for row in rows}
        if "attempt_type" in column_names:
            return

        # Legacy table lacks attempt_type -- rebuild with the new schema.
        await conn.execute("ALTER TABLE account_lockouts RENAME TO account_lockouts_legacy")
        await conn.execute(lockout_ddl)
        await conn.execute(
            """
            INSERT INTO account_lockouts (identifier, attempt_type, locked_until, reason)
            SELECT identifier, 'login', locked_until, reason
            FROM account_lockouts_legacy
            """
        )
        await conn.execute("DROP TABLE IF EXISTS account_lockouts_legacy")


    async def _ensure_postgres_account_lockouts_schema(self, conn: Any) -> None:
        """Upgrade legacy identifier-only lockout tables to attempt-type scope."""
        lockout_ddl = """
            CREATE TABLE IF NOT EXISTS account_lockouts (
                identifier TEXT NOT NULL,
                attempt_type TEXT NOT NULL,
                locked_until TIMESTAMPTZ NOT NULL,
                reason TEXT,
                PRIMARY KEY (identifier, attempt_type)
            )
            """
        columns = await conn.fetch(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'account_lockouts'
            """
        )
        if not columns:
            await conn.execute(lockout_ddl)
            return

        column_names = {row["column_name"] for row in columns}
        if "attempt_type" in column_names:
            return

        await conn.execute("LOCK TABLE account_lockouts IN ACCESS EXCLUSIVE MODE")
        columns = await conn.fetch(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'account_lockouts'
            """
        )
        column_names = {row["column_name"] for row in columns}
        if "attempt_type" in column_names:
            return

        await conn.execute("ALTER TABLE account_lockouts RENAME TO account_lockouts_legacy")
        await conn.execute(lockout_ddl)
        await conn.execute(
            """
            INSERT INTO account_lockouts (identifier, attempt_type, locked_until, reason)
            SELECT identifier, 'login', locked_until, reason
            FROM account_lockouts_legacy
            ON CONFLICT (identifier, attempt_type) DO NOTHING
            """
        )
        await conn.execute("DROP TABLE IF EXISTS account_lockouts_legacy")

    async def ensure_schema(self) -> None:
        """
        Ensure the AuthNZ rate-limiter tables exist for the configured backend.

        This is an idempotent bootstrap/backstop helper for deployments and
        test setups that may not have run the full AuthNZ migrations yet.
        """
        ddl_sqlite = [
            # Per-identifier request counts per window
            """
            CREATE TABLE IF NOT EXISTS rate_limits (
                identifier TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                request_count INTEGER NOT NULL,
                window_start TEXT NOT NULL,
                PRIMARY KEY (identifier, endpoint, window_start)
            )
            """,
            # Failed attempts for lockout
            """
            CREATE TABLE IF NOT EXISTS failed_attempts (
                identifier TEXT NOT NULL,
                attempt_type TEXT NOT NULL,
                attempt_count INTEGER NOT NULL,
                window_start TEXT NOT NULL,
                PRIMARY KEY (identifier, attempt_type)
            )
            """,
            # Account lockouts
            """
            CREATE TABLE IF NOT EXISTS account_lockouts (
                identifier TEXT NOT NULL,
                attempt_type TEXT NOT NULL,
                locked_until TEXT NOT NULL,
                reason TEXT,
                PRIMARY KEY (identifier, attempt_type)
            )
            """,
            # Helpful index for queries by identifier
            "CREATE INDEX IF NOT EXISTS idx_rate_limits_identifier ON rate_limits(identifier)",
        ]

        ddl_postgres = [
            """
            CREATE TABLE IF NOT EXISTS rate_limits (
                identifier TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                request_count INTEGER NOT NULL,
                window_start TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (identifier, endpoint, window_start)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS failed_attempts (
                identifier TEXT NOT NULL,
                attempt_type TEXT NOT NULL,
                attempt_count INTEGER NOT NULL,
                window_start TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (identifier, attempt_type)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS account_lockouts (
                identifier TEXT NOT NULL,
                attempt_type TEXT NOT NULL,
                locked_until TIMESTAMPTZ NOT NULL,
                reason TEXT,
                PRIMARY KEY (identifier, attempt_type)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_rate_limits_identifier ON rate_limits(identifier)",
        ]

        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    for sql in ddl_postgres[:-1]:
                        await conn.execute(sql)
                    await self._ensure_postgres_account_lockouts_schema(conn)
                    for sql in ddl_postgres[-1:]:
                        await conn.execute(sql)
                else:
                    # Run rate_limits and failed_attempts DDL, then
                    # handle account_lockouts with legacy upgrade logic.
                    for sql in ddl_sqlite[:2]:
                        await conn.execute(sql)
                    await self._ensure_sqlite_account_lockouts_schema(conn)
                    # Create the index
                    for sql in ddl_sqlite[3:]:
                        await conn.execute(sql)
                    try:
                        await conn.commit()
                    except Exception as commit_error:
                        # aiosqlite transaction manager may commit outside; ignore
                        logger.bind(error_type=type(commit_error).__name__).debug(
                            "Rate limits repo explicit commit failed; transaction manager likely committed"
                        )
        except Exception as exc:
            logger.error(f"AuthnzRateLimitsRepo.ensure_schema failed: {exc}")
            raise

    async def cleanup_rate_limits_older_than(
        self,
        cutoff: datetime,
    ) -> int:
        """
        Delete ``rate_limits`` rows with ``window_start`` older than the cutoff.

        Returns the number of deleted rows (best-effort when the backend does
        not report an accurate rowcount).
        """
        try:
            if self._is_postgres_backend():
                rows = await self.db_pool.fetch(
                    """
                    DELETE FROM rate_limits
                    WHERE window_start < $1
                    RETURNING 1
                    """,
                    cutoff,
                )
                return len(rows)

            async with self.db_pool.acquire() as conn:
                cursor = await conn.execute(
                    """
                    DELETE FROM rate_limits
                    WHERE datetime(window_start) < datetime(?)
                    """,
                    (cutoff.isoformat(),),
                )
                deleted = getattr(cursor, "rowcount", 0) or 0
                try:
                    await conn.commit()
                except Exception as exc:
                    logger.warning(
                        f"AuthnzRateLimitsRepo SQLite commit failed during cleanup: {exc}"
                    )
                return int(deleted)
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzRateLimitsRepo.cleanup_rate_limits_older_than failed: {exc}")
            raise

    async def increment_rate_limit_window(
        self,
        *,
        identifier: str,
        endpoint: str,
        window_start: datetime,
    ) -> int:
        """
        Increment the ``request_count`` for a given (identifier, endpoint, window_start)
        bucket in ``rate_limits`` and return the updated count.

        Mirrors the upsert semantics previously implemented in
        ``RateLimiter._check_database_rate_limit``.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    result = await conn.fetchval(
                        """
                        INSERT INTO rate_limits (identifier, endpoint, request_count, window_start)
                        VALUES ($1, $2, 1, $3)
                        ON CONFLICT (identifier, endpoint, window_start)
                        DO UPDATE SET request_count = rate_limits.request_count + 1
                        RETURNING request_count
                        """,
                        identifier,
                        endpoint,
                        window_start,
                    )
                    return int(result or 0)

                cursor = await conn.execute(
                    """
                    SELECT request_count
                    FROM rate_limits
                    WHERE identifier = ? AND endpoint = ? AND window_start = ?
                    """,
                    (identifier, endpoint, window_start.isoformat()),
                )
                row = await cursor.fetchone()

                if row:
                    current_count = int(row[0]) + 1
                    await conn.execute(
                        """
                        UPDATE rate_limits
                        SET request_count = ?
                        WHERE identifier = ? AND endpoint = ? AND window_start = ?
                        """,
                        (current_count, identifier, endpoint, window_start.isoformat()),
                    )
                else:
                    current_count = 1
                    await conn.execute(
                        """
                        INSERT INTO rate_limits (identifier, endpoint, request_count, window_start)
                        VALUES (?, ?, ?, ?)
                        """,
                        (identifier, endpoint, 1, window_start.isoformat()),
                    )
                return int(current_count)
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzRateLimitsRepo.increment_rate_limit_window failed: {exc}"
            )
            raise

    async def get_rate_limit_count(
        self,
        *,
        identifier: str,
        endpoint: str,
        window_start: datetime,
    ) -> int:
        """
        Fetch the ``request_count`` for a specific rate-limit bucket.
        """
        try:
            if self._is_postgres_backend():
                value = await self.db_pool.fetchval(
                    """
                    SELECT request_count
                    FROM rate_limits
                    WHERE identifier = $1 AND endpoint = $2 AND window_start = $3
                    """,
                    identifier,
                    endpoint,
                    window_start,
                )
                return int(value or 0)

            async with self.db_pool.acquire() as conn:
                cursor = await conn.execute(
                    """
                    SELECT request_count
                    FROM rate_limits
                    WHERE identifier = ? AND endpoint = ? AND window_start = ?
                    """,
                    (identifier, endpoint, window_start.isoformat()),
                )
                row = await cursor.fetchone()
                return int(row[0]) if row else 0
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzRateLimitsRepo.get_rate_limit_count failed: {exc}"
            )
            raise

    async def list_rate_limit_endpoints_for_identifier(
        self,
        *,
        identifier: str,
    ) -> tuple[str, ...]:
        """
        Return a tuple of distinct endpoints seen for an identifier in ``rate_limits``.
        """
        try:
            if self._is_postgres_backend():
                rows = await self.db_pool.fetch(
                    "SELECT DISTINCT endpoint FROM rate_limits WHERE identifier = $1",
                    identifier,
                )
                return tuple(str(r["endpoint"]) for r in rows or [])

            async with self.db_pool.acquire() as conn:
                cursor = await conn.execute(
                    "SELECT DISTINCT endpoint FROM rate_limits WHERE identifier = ?",
                    (identifier,),
                )
                rows = await cursor.fetchall()
                return tuple(str(r[0]) for r in rows or [])
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzRateLimitsRepo.list_rate_limit_endpoints_for_identifier failed: {exc}"
            )
            raise

    async def delete_rate_limits_for_identifier(
        self,
        *,
        identifier: str,
        endpoint: str | None = None,
    ) -> int:
        """
        Delete ``rate_limits`` rows for an identifier (optionally scoped to an endpoint).

        Returns the number of rows deleted (best-effort).
        """
        try:
            async with self.db_pool.transaction() as conn:
                deleted = 0
                if self._is_postgres_backend():
                    if endpoint:
                        result = await conn.execute(
                            """
                            DELETE FROM rate_limits
                            WHERE identifier = $1 AND endpoint = $2
                            """,
                            identifier,
                            endpoint,
                        )
                    else:
                        result = await conn.execute(
                            """
                            DELETE FROM rate_limits
                            WHERE identifier = $1
                            """,
                            identifier,
                        )
                    try:
                        deleted = (
                            int(result.split()[-1])
                            if isinstance(result, str)
                            else 0
                        )
                    except (ValueError, AttributeError, IndexError) as exc:
                        logger.debug(
                            f"AuthnzRateLimitsRepo asyncpg DELETE result parse failed: result={result!r}, error={exc}"
                        )
                        deleted = 0
                else:
                    if endpoint:
                        cursor = await conn.execute(
                            """
                            DELETE FROM rate_limits
                            WHERE identifier = ? AND endpoint = ?
                            """,
                            (identifier, endpoint),
                        )
                    else:
                        cursor = await conn.execute(
                            """
                            DELETE FROM rate_limits
                            WHERE identifier = ?
                            """,
                            (identifier,),
                        )
                    deleted = getattr(cursor, "rowcount", 0) or 0
                    try:
                        await conn.commit()
                    except Exception as exc:
                        logger.debug(
                            f"AuthnzRateLimitsRepo SQLite commit failed during delete: {exc}"
                        )
            return int(deleted)
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzRateLimitsRepo.delete_rate_limits_for_identifier failed: {exc}"
            )
            raise

    async def list_recent_violations(
        self,
        *,
        cutoff: datetime,
        rate_threshold: int,
        limit: int = 20,
    ) -> tuple[dict[str, Any], ...]:
        """
        Return aggregated rate-limit buckets that exceed the given threshold.

        The result mirrors the query used by the AuthNZ scheduler's
        Legacy scheduler monitoring used this helper; it remains backend-agnostic.

        Each returned row is a mapping with keys:
        - ``identifier``
        - ``endpoint``
        - ``total_requests``
        - ``window_count``
        """
        try:
            # SQLite stores window_start as TEXT; Postgres uses TIMESTAMPTZ.
            is_postgres = self._is_postgres_backend()
            cutoff_param: Any = cutoff if is_postgres else cutoff.isoformat()

            rows_raw = await self.db_pool.fetch(
                """
                SELECT
                    identifier,
                    endpoint,
                    SUM(request_count) AS total_requests,
                    COUNT(*) AS window_count
                FROM rate_limits
                WHERE window_start > ?
                GROUP BY identifier, endpoint
                HAVING SUM(request_count) > ?
                ORDER BY total_requests DESC
                LIMIT ?
                """,
                cutoff_param,
                int(rate_threshold),
                int(limit),
            )

            results: list[dict[str, Any]] = []
            for r in rows_raw or []:
                try:
                    if isinstance(r, dict):
                        results.append(
                            {
                                "identifier": r.get("identifier"),
                                "endpoint": r.get("endpoint"),
                                "total_requests": int(r.get("total_requests", 0)),
                                "window_count": int(r.get("window_count", 0)),
                            }
                        )
                    else:
                        # aiosqlite.Row-style access
                        results.append(
                            {
                                "identifier": r["identifier"],
                                "endpoint": r["endpoint"],
                                "total_requests": int(r["total_requests"]),
                                "window_count": int(r["window_count"]),
                            }
                        )
                except Exception as exc:
                    logger.debug(
                        f"Skipping malformed rate_limit row: {exc}"
                    )
                    continue

            return tuple(results)
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzRateLimitsRepo.list_recent_violations failed: {exc}"
            )
            raise

    async def record_failed_attempt_and_lockout(
        self,
        *,
        identifier: str,
        attempt_type: str,
        now: datetime,
        lockout_threshold: int,
        lockout_duration_minutes: int,
    ) -> dict[str, Any]:
        """
        Increment the failed-attempt counter and, if necessary, record a lockout.

        Returns a dict containing:
        - ``attempt_count`` (int)
        - ``is_locked`` (bool)
        - ``lockout_expires`` (Optional[datetime])
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    # PostgreSQL - use ON CONFLICT with window-reset semantics
                    result = await conn.fetchrow(
                        """
                        INSERT INTO failed_attempts (identifier, attempt_type, attempt_count, window_start)
                        VALUES ($1, $2, 1, $3)
                        ON CONFLICT (identifier, attempt_type)
                        DO UPDATE SET
                            attempt_count = CASE
                                WHEN failed_attempts.window_start + ($4 * INTERVAL '1 minute') < $3
                                THEN 1
                                ELSE failed_attempts.attempt_count + 1
                            END,
                            window_start = CASE
                                WHEN failed_attempts.window_start + ($4 * INTERVAL '1 minute') < $3
                                THEN $3
                                ELSE failed_attempts.window_start
                            END
                        RETURNING attempt_count, window_start
                        """,
                        identifier,
                        attempt_type,
                        now,
                        int(lockout_duration_minutes),
                    )
                    attempt_count = int(result["attempt_count"])
                else:
                    # SQLite path with equivalent window-reset behavior
                    cursor = await conn.execute(
                        """
                        INSERT INTO failed_attempts (identifier, attempt_type, attempt_count, window_start)
                        VALUES (?, ?, 1, ?)
                        ON CONFLICT (identifier, attempt_type)
                        DO UPDATE SET
                            attempt_count = CASE
                                WHEN datetime(window_start, '+' || ? || ' minutes') < ?
                                THEN 1
                                ELSE attempt_count + 1
                            END,
                            window_start = CASE
                                WHEN datetime(window_start, '+' || ? || ' minutes') < ?
                                THEN ?
                                ELSE window_start
                            END
                        """,
                        (
                            identifier,
                            attempt_type,
                            now.isoformat(),
                            lockout_duration_minutes,
                            now.isoformat(),
                            lockout_duration_minutes,
                            now.isoformat(),
                            now.isoformat(),
                        ),
                    )
                    _ = cursor  # unused
                    cursor = await conn.execute(
                        """
                        SELECT attempt_count
                        FROM failed_attempts
                        WHERE identifier = ? AND attempt_type = ?
                        """,
                        (identifier, attempt_type),
                    )
                    row = await cursor.fetchone()
                    attempt_count = int(row[0]) if row else 1

                is_locked = attempt_count >= lockout_threshold
                lockout_expires: datetime | None = None

                if is_locked:
                    lockout_expires = now + timedelta(minutes=lockout_duration_minutes)
                    reason = f"Too many failed {attempt_type} attempts"
                    if self._is_postgres_backend():
                        await conn.execute(
                            """
                            INSERT INTO account_lockouts (identifier, attempt_type, locked_until, reason)
                            VALUES ($1, $2, $3, $4)
                            ON CONFLICT (identifier, attempt_type) DO UPDATE SET
                                locked_until = $3,
                                reason = $4
                            """,
                            identifier,
                            attempt_type,
                            lockout_expires,
                            reason,
                        )
                    else:
                        await conn.execute(
                            """
                            INSERT INTO account_lockouts (identifier, attempt_type, locked_until, reason)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(identifier, attempt_type) DO UPDATE SET
                                locked_until = excluded.locked_until,
                                reason = excluded.reason
                            """,
                            (identifier, attempt_type, lockout_expires.isoformat(), reason),
                        )
                return {
                    "attempt_count": int(attempt_count),
                    "is_locked": bool(is_locked),
                    "lockout_expires": lockout_expires,
                }
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzRateLimitsRepo.record_failed_attempt_and_lockout failed: {exc}")
            raise

    async def get_active_lockout(
        self,
        *,
        identifier: str,
        attempt_type: str = "login",
        now: datetime,
    ) -> datetime | None:
        """
        Return the active lockout expiry for an identifier, pruning expired rows.
        """
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres_backend():
                    row = await conn.fetchrow(
                        """
                        SELECT locked_until
                        FROM account_lockouts
                        WHERE identifier = $1 AND attempt_type = $2 AND locked_until > $3
                        """,
                        identifier,
                        attempt_type,
                        now,
                    )
                    if row:
                        return row["locked_until"]
                    await conn.execute(
                        "DELETE FROM account_lockouts WHERE identifier = $1 AND attempt_type = $2 AND locked_until <= $3",
                        identifier,
                        attempt_type,
                        now,
                    )
                    return None

                cursor = await conn.execute(
                    """
                    SELECT locked_until
                    FROM account_lockouts
                    WHERE identifier = ? AND attempt_type = ? AND locked_until > ?
                    """,
                    (identifier, attempt_type, now.isoformat()),
                )
                row = await cursor.fetchone()
                if row:
                    return datetime.fromisoformat(row[0])
                await conn.execute(
                    "DELETE FROM account_lockouts WHERE identifier = ? AND attempt_type = ? AND locked_until <= ?",
                    (identifier, attempt_type, now.isoformat()),
                )
                try:
                    await conn.commit()
                except Exception as exc:
                    logger.debug(
                        f"AuthnzRateLimitsRepo SQLite commit failed during delete: {exc}"
                    )
                return None
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzRateLimitsRepo.get_active_lockout failed: {exc}")
            raise

    async def reset_failed_attempts_and_lockout(
        self,
        *,
        identifier: str,
        attempt_type: str,
    ) -> None:
        """
        Clear failed-attempt counters and account lockout rows for an identifier.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    await conn.execute(
                        """
                        DELETE FROM failed_attempts
                        WHERE identifier = $1 AND attempt_type = $2
                        """,
                        identifier,
                        attempt_type,
                    )
                    await conn.execute(
                        "DELETE FROM account_lockouts WHERE identifier = $1 AND attempt_type = $2",
                        identifier,
                        attempt_type,
                    )
                else:
                    await conn.execute(
                        "DELETE FROM failed_attempts WHERE identifier = ? AND attempt_type = ?",
                        (identifier, attempt_type),
                    )
                    await conn.execute(
                        "DELETE FROM account_lockouts WHERE identifier = ? AND attempt_type = ?",
                        (identifier, attempt_type),
                    )
                    try:
                        await conn.commit()
                    except Exception as exc:
                        logger.debug(
                            f"AuthnzRateLimitsRepo SQLite commit failed during reset_failed_attempts_and_lockout: {exc}"
                        )
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzRateLimitsRepo.reset_failed_attempts_and_lockout failed: {exc}")
            raise
