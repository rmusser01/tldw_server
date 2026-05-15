from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool

_SQLITE_CORRUPTION_SIGNATURES = (
    "database disk image is malformed",
    "malformed database schema",
    "file is not a database",
)
_SQLITE_CORRUPTION_WARNING_KEYS: set[str] = set()


def _looks_like_sqlite_corruption(exc: Exception) -> bool:
    text = str(exc or "").strip().lower()
    if not text:
        return False
    return any(signature in text for signature in _SQLITE_CORRUPTION_SIGNATURES)


def _sqlite_pool_label(db_pool: DatabasePool) -> str:
    raw = getattr(db_pool, "_sqlite_fs_path", None) or getattr(db_pool, "db_path", None) or "unknown"
    try:
        text = str(raw).strip()
        return text or "unknown"
    except (TypeError, ValueError):
        return "unknown"


def _log_sqlite_corruption_skip_once(*, operation: str, db_pool: DatabasePool, exc: Exception) -> None:
    db_label = _sqlite_pool_label(db_pool)
    key = f"{operation}:{db_label}"
    if key in _SQLITE_CORRUPTION_WARNING_KEYS:
        logger.debug(
            "AuthnzUsageRepo.{} skipping due to previously detected sqlite corruption ({}): {}",
            operation,
            db_label,
            exc,
        )
        return
    _SQLITE_CORRUPTION_WARNING_KEYS.add(key)
    logger.warning(
        "AuthnzUsageRepo.{} detected sqlite corruption at {}; skipping aggregate until DB is repaired: {}",
        operation,
        db_label,
        exc,
    )


@dataclass
class AuthnzUsageRepo:
    """
    Repository for AuthNZ LLM usage accounting tables.

    This class centralizes common aggregate queries over ``llm_usage_log``
    and related tables so callers do not need to embed backend-specific
    SQL or timestamp handling logic.
    """

    db_pool: DatabasePool

    def _is_postgres_backend(self) -> bool:
        """
        Return True when the underlying DatabasePool is using PostgreSQL.

        Backend routing should rely on pool state rather than probing
        connection capabilities.
        """
        return bool(getattr(self.db_pool, "pool", None))

    async def summarize_key_day(
        self,
        *,
        key_id: int,
        day: date | None = None,
    ) -> dict[str, Any]:
        """
        Summarize token and USD usage for a key over a specific UTC day.

        Returns a dict with:
        - ``tokens`` (int)
        - ``usd`` (float)
        """
        try:
            day_val: date
            if isinstance(day, date):
                day_val = day
            else:
                # Default to "today" in UTC; Date() on Postgres is always UTC.
                day_val = datetime.now(timezone.utc).date()

            if getattr(self.db_pool, "pool", None) is not None:
                total_tokens = await self.db_pool.fetchval(
                    """
                    SELECT COALESCE(SUM(total_tokens),0)
                    FROM llm_usage_log
                    WHERE date(ts AT TIME ZONE 'UTC') = $1
                      AND key_id = $2
                    """,
                    day_val,
                    key_id,
                )
                total_cost = await self.db_pool.fetchval(
                    """
                    SELECT COALESCE(SUM(total_cost_usd),0)
                    FROM llm_usage_log
                    WHERE date(ts AT TIME ZONE 'UTC') = $1
                      AND key_id = $2
                    """,
                    day_val,
                    key_id,
                )
            else:
                day_str = day_val.isoformat()
                total_tokens = await self.db_pool.fetchval(
                    """
                    SELECT COALESCE(SUM(total_tokens),0)
                    FROM llm_usage_log
                    WHERE DATE(datetime(ts)) = ?
                      AND key_id = ?
                    """,
                    day_str,
                    key_id,
                )
                total_cost = await self.db_pool.fetchval(
                    """
                    SELECT COALESCE(SUM(total_cost_usd),0)
                    FROM llm_usage_log
                    WHERE DATE(datetime(ts)) = ?
                      AND key_id = ?
                    """,
                    day_str,
                    key_id,
                )

            return {
                "tokens": int(total_tokens or 0),
                "usd": float(total_cost or 0.0),
            }
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzUsageRepo.summarize_key_day failed: {exc}")
            raise

    async def summarize_user_day(
        self,
        *,
        user_id: int,
        day: date | None = None,
    ) -> dict[str, Any]:
        """
        Summarize token and USD usage for a user over a specific UTC day.

        Returns a dict with:
        - ``tokens`` (int)
        - ``usd`` (float)
        """
        try:
            day_val: date
            day_val = day if isinstance(day, date) else datetime.now(timezone.utc).date()

            if getattr(self.db_pool, "pool", None) is not None:
                total_tokens = await self.db_pool.fetchval(
                    """
                    SELECT COALESCE(SUM(total_tokens),0)
                    FROM llm_usage_log
                    WHERE date(ts AT TIME ZONE 'UTC') = $1
                      AND user_id = $2
                    """,
                    day_val,
                    user_id,
                )
                total_cost = await self.db_pool.fetchval(
                    """
                    SELECT COALESCE(SUM(total_cost_usd),0)
                    FROM llm_usage_log
                    WHERE date(ts AT TIME ZONE 'UTC') = $1
                      AND user_id = $2
                    """,
                    day_val,
                    user_id,
                )
            else:
                day_str = day_val.isoformat()
                total_tokens = await self.db_pool.fetchval(
                    """
                    SELECT COALESCE(SUM(total_tokens),0)
                    FROM llm_usage_log
                    WHERE DATE(datetime(ts)) = ?
                      AND user_id = ?
                    """,
                    day_str,
                    user_id,
                )
                total_cost = await self.db_pool.fetchval(
                    """
                    SELECT COALESCE(SUM(total_cost_usd),0)
                    FROM llm_usage_log
                    WHERE DATE(datetime(ts)) = ?
                      AND user_id = ?
                    """,
                    day_str,
                    user_id,
                )

            return {
                "tokens": int(total_tokens or 0),
                "usd": float(total_cost or 0.0),
            }
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzUsageRepo.summarize_user_day failed: {exc}")
            raise

    async def summarize_key_rolling_window(
        self,
        *,
        key_id: int,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Summarize token and USD usage for a key over a rolling UTC window.

        The window is defined as ``[now - days, now)`` in UTC.

        Returns a dict with:
        - ``tokens`` (int)
        - ``usd`` (float)
        """
        window_days = max(1, int(days))
        try:
            now = datetime.now(timezone.utc)
            start_dt = now - timedelta(days=window_days)
            end_dt = now

            if getattr(self.db_pool, "pool", None) is not None:
                # Postgres path: ensure naive UTC timestamps for comparison
                _start = start_dt.replace(tzinfo=None)
                _end = end_dt.replace(tzinfo=None)
                row = await self.db_pool.fetchone(
                    """
                    SELECT
                        COALESCE(SUM(total_tokens),0) AS tokens,
                        COALESCE(SUM(total_cost_usd),0.0) AS usd
                    FROM llm_usage_log
                    WHERE ts >= $1 AND ts < $2 AND key_id = $3
                    """,
                    _start,
                    _end,
                    key_id,
                )
            else:
                # SQLite path: normalize timestamps to a consistent naive UTC string
                def _sqlite_fmt(value: datetime) -> str:
                    dt = value.astimezone(timezone.utc).replace(tzinfo=None)
                    return dt.strftime("%Y-%m-%d %H:%M:%S")

                start_str = _sqlite_fmt(start_dt)
                end_str = _sqlite_fmt(end_dt)
                row = await self.db_pool.fetchone(
                    """
                    SELECT
                        COALESCE(SUM(total_tokens),0) AS tokens,
                        COALESCE(SUM(total_cost_usd),0.0) AS usd
                    FROM llm_usage_log
                    WHERE datetime(ts) >= ?
                      AND datetime(ts) < ?
                      AND key_id = ?
                    """,
                    start_str,
                    end_str,
                    key_id,
                )

            tokens = 0
            usd = 0.0
            if row:
                if hasattr(row, "get"):
                    tokens = int(row.get("tokens") or 0)
                    usd = float(row.get("usd") or 0.0)
                else:
                    tokens = int((row["tokens"] if "tokens" in row else row[0]) or 0)
                    usd = float((row["usd"] if "usd" in row else row[1]) or 0.0)

            return {"tokens": tokens, "usd": usd}
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzUsageRepo.summarize_key_rolling_window failed: {exc}")
            raise

    async def prune_llm_usage_log_before(self, cutoff: datetime) -> int:
        """
        Delete ``llm_usage_log`` rows older than the given cutoff timestamp.

        Returns the number of deleted rows (best-effort when the backend does
        not report an accurate rowcount).
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    cutoff_param = cutoff.replace(tzinfo=None) if getattr(cutoff, "tzinfo", None) else cutoff
                    rows = await conn.fetch(
                        "DELETE FROM llm_usage_log WHERE ts < $1 RETURNING 1",
                        cutoff_param,
                    )
                    return len(rows)
                # SQLite path
                cursor = await conn.execute(
                    "DELETE FROM llm_usage_log WHERE ts < ?",
                    (cutoff.isoformat(),),
                )
                deleted = getattr(cursor, "rowcount", 0) or 0
                return int(deleted)
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzUsageRepo.prune_llm_usage_log_before failed: {exc}")
            raise

    async def prune_usage_log_before(self, cutoff: datetime) -> int:
        """
        Delete ``usage_log`` rows older than the given cutoff timestamp.

        Returns the number of deleted rows (best-effort).
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    cutoff_param = cutoff.replace(tzinfo=None) if getattr(cutoff, "tzinfo", None) else cutoff
                    rows = await conn.fetch(
                        "DELETE FROM usage_log WHERE ts < $1 RETURNING 1",
                        cutoff_param,
                    )
                    return len(rows)

                cursor = await conn.execute(
                    "DELETE FROM usage_log WHERE ts < ?",
                    (cutoff.isoformat(),),
                )
                deleted = getattr(cursor, "rowcount", 0) or 0
                return int(deleted)
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzUsageRepo.prune_usage_log_before failed: {exc}")
            raise

    async def prune_usage_daily_before(self, cutoff_day: date) -> int:
        """
        Delete ``usage_daily`` rows older than the given cutoff day.

        Returns the number of deleted rows (best-effort).
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    rows = await conn.fetch(
                        "DELETE FROM usage_daily WHERE day < $1::date RETURNING 1",
                        cutoff_day,
                    )
                    return len(rows)
                cursor = await conn.execute(
                    "DELETE FROM usage_daily WHERE day < ?",
                    (cutoff_day.isoformat(),),
                )
                deleted = getattr(cursor, "rowcount", 0) or 0
                return int(deleted)
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzUsageRepo.prune_usage_daily_before failed: {exc}")
            raise

    async def insert_usage_log(
        self,
        *,
        user_id: int | None,
        key_id: int | None,
        endpoint: str,
        status: int,
        latency_ms: int,
        bytes_out: int | None,
        bytes_in: int | None,
        meta: str,
        request_id: str | None,
    ) -> None:
        """
        Insert a single row into ``usage_log``.

        This mirrors the insert logic previously embedded in
        ``UsageLoggingMiddleware`` while centralizing dialect differences
        and fallback behavior (with/without ``bytes_in``) in one place.
        """
        try:
            # Prefer the extended schema including bytes_in when available;
            # fall back to the legacy schema when the column is missing.
            try:
                await self.db_pool.execute(
                    """
                    INSERT INTO usage_log (
                        user_id, key_id, endpoint, status, latency_ms,
                        bytes, bytes_in, meta, request_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    user_id,
                    key_id,
                    endpoint,
                    int(status),
                    int(latency_ms),
                    int(bytes_out) if bytes_out is not None else None,
                    int(bytes_in) if bytes_in is not None else None,
                    meta,
                    request_id,
                )
            except Exception:
                await self.db_pool.execute(
                    """
                    INSERT INTO usage_log (
                        user_id, key_id, endpoint, status, latency_ms,
                        bytes, meta, request_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    user_id,
                    key_id,
                    endpoint,
                    int(status),
                    int(latency_ms),
                    int(bytes_out) if bytes_out is not None else None,
                    meta,
                    request_id,
                )
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzUsageRepo.insert_usage_log failed: {exc}")
            raise

    async def insert_llm_usage_log(
        self,
        *,
        user_id: int | None,
        key_id: int | None,
        endpoint: str,
        operation: str,
        provider: str,
        model: str,
        status: int,
        latency_ms: int,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        prompt_cost_usd: float,
        completion_cost_usd: float,
        total_cost_usd: float,
        currency: str = "USD",
        estimated: bool = False,
        request_id: str | None = None,
        remote_ip: str | None = None,
        user_agent: str | None = None,
        token_name: str | None = None,
        conversation_id: str | None = None,
        cached_input_tokens: int | None = None,
        cache_write_input_tokens: int | None = None,
        cache_read_input_tokens: int | None = None,
        billable_input_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        choice_count: int | None = None,
        estimate_source: str | None = None,
        prompt_fingerprint: str | None = None,
        prompt_fingerprint_version: str | None = None,
        world_book_fingerprint: str | None = None,
        raw_usage_metadata_json: str | None = None,
    ) -> None:
        """
        Insert a single row into ``llm_usage_log``.

        This centralizes dialect handling so callers (e.g., usage_tracker)
        do not embed backend-specific SQL.
        """
        try:
            try:
                await self.db_pool.execute(
                    """
                    INSERT INTO llm_usage_log (
                        ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
                        prompt_tokens, completion_tokens, total_tokens,
                        prompt_cost_usd, completion_cost_usd, total_cost_usd, currency, estimated, request_id,
                        remote_ip, user_agent, token_name, conversation_id,
                        cached_input_tokens, cache_write_input_tokens, cache_read_input_tokens,
                        billable_input_tokens, reasoning_tokens, choice_count, estimate_source,
                        prompt_fingerprint, prompt_fingerprint_version, world_book_fingerprint,
                        raw_usage_metadata_json
                    ) VALUES (
                        CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?
                    )
                    """,
                    user_id,
                    key_id,
                    endpoint,
                    operation,
                    provider,
                    model,
                    int(status),
                    int(latency_ms),
                    int(prompt_tokens),
                    int(completion_tokens),
                    int(total_tokens),
                    float(prompt_cost_usd),
                    float(completion_cost_usd),
                    float(total_cost_usd),
                    currency,
                    bool(estimated),
                    request_id,
                    remote_ip,
                    user_agent,
                    token_name,
                    conversation_id,
                    int(cached_input_tokens) if cached_input_tokens is not None else None,
                    int(cache_write_input_tokens) if cache_write_input_tokens is not None else None,
                    int(cache_read_input_tokens) if cache_read_input_tokens is not None else None,
                    int(billable_input_tokens) if billable_input_tokens is not None else None,
                    int(reasoning_tokens) if reasoning_tokens is not None else None,
                    int(choice_count) if choice_count is not None else None,
                    estimate_source,
                    prompt_fingerprint,
                    prompt_fingerprint_version,
                    world_book_fingerprint,
                    raw_usage_metadata_json,
                )
            except Exception:
                try:
                    # Backward-compatible fallback for pre-088 schemas.
                    await self.db_pool.execute(
                        """
                        INSERT INTO llm_usage_log (
                            ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
                            prompt_tokens, completion_tokens, total_tokens,
                            prompt_cost_usd, completion_cost_usd, total_cost_usd, currency, estimated, request_id,
                            remote_ip, user_agent, token_name, conversation_id
                        ) VALUES (
                            CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?,
                            ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?
                        )
                        """,
                        user_id,
                        key_id,
                        endpoint,
                        operation,
                        provider,
                        model,
                        int(status),
                        int(latency_ms),
                        int(prompt_tokens),
                        int(completion_tokens),
                        int(total_tokens),
                        float(prompt_cost_usd),
                        float(completion_cost_usd),
                        float(total_cost_usd),
                        currency,
                        bool(estimated),
                        request_id,
                        remote_ip,
                        user_agent,
                        token_name,
                        conversation_id,
                    )
                except Exception:
                    # Backward-compatible fallback for pre-054 schemas.
                    await self.db_pool.execute(
                        """
                        INSERT INTO llm_usage_log (
                            ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
                            prompt_tokens, completion_tokens, total_tokens,
                            prompt_cost_usd, completion_cost_usd, total_cost_usd, currency, estimated, request_id
                        ) VALUES (
                            CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?,
                            ?, ?, ?, ?, ?, ?
                        )
                        """,
                        user_id,
                        key_id,
                        endpoint,
                        operation,
                        provider,
                        model,
                        int(status),
                        int(latency_ms),
                        int(prompt_tokens),
                        int(completion_tokens),
                        int(total_tokens),
                        float(prompt_cost_usd),
                        float(completion_cost_usd),
                        float(total_cost_usd),
                        currency,
                        bool(estimated),
                        request_id,
                    )
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzUsageRepo.insert_llm_usage_log failed: {exc}")
            raise

    async def get_api_key_name(self, *, key_id: int) -> str | None:
        """Fetch api_keys.name for a key id."""
        try:
            if self._is_postgres_backend():
                val = await self.db_pool.fetchval(
                    "SELECT name FROM api_keys WHERE id = $1",
                    int(key_id),
                )
            else:
                val = await self.db_pool.fetchval(
                    "SELECT name FROM api_keys WHERE id = ?",
                    int(key_id),
                )
            if val is None:
                return None
            text = str(val).strip()
            return text or None
        except Exception as exc:
            logger.debug(f"AuthnzUsageRepo.get_api_key_name skipped/failed: {exc}")
            return None

    async def aggregate_usage_daily_for_day(self, *, day: date | None = None) -> None:
        """
        Aggregate per-request usage from ``usage_log`` into ``usage_daily`` for a UTC day.

        This mirrors the logic previously in app/services/usage_aggregator.py.
        """
        try:
            day_val = day if isinstance(day, date) else datetime.now(timezone.utc).date()
            day_str = day_val.isoformat()

            if getattr(self.db_pool, "pool", None) is not None:
                # Postgres: use date(ts AT TIME ZONE 'UTC') and ON CONFLICT upsert.
                try:
                    await self.db_pool.execute(
                        """
                        INSERT INTO usage_daily (user_id, day, requests, errors, bytes_total, bytes_in_total, latency_avg_ms)
                        SELECT
                            user_id as user_id,
                            ?::date as day,
                            COUNT(*) as requests,
                            SUM(CASE WHEN status >= 400 THEN 1 ELSE 0 END) as errors,
                            COALESCE(SUM(COALESCE(bytes, 0)), 0) as bytes_total,
                            COALESCE(SUM(COALESCE(bytes_in, 0)), 0) as bytes_in_total,
                            AVG(latency_ms)::float as latency_avg_ms
                        FROM usage_log
                        WHERE user_id IS NOT NULL AND date(ts AT TIME ZONE 'UTC') = ?::date
                        GROUP BY user_id
                        ON CONFLICT (user_id, day) DO UPDATE SET
                            requests = EXCLUDED.requests,
                            errors = EXCLUDED.errors,
                            bytes_total = EXCLUDED.bytes_total,
                            bytes_in_total = EXCLUDED.bytes_in_total,
                            latency_avg_ms = EXCLUDED.latency_avg_ms
                        """,
                        day_val,
                        day_val,
                    )
                except Exception:
                    # Fallback to legacy schema without bytes_in_total
                    await self.db_pool.execute(
                        """
                        INSERT INTO usage_daily (user_id, day, requests, errors, bytes_total, latency_avg_ms)
                        SELECT
                            user_id as user_id,
                            ?::date as day,
                            COUNT(*) as requests,
                            SUM(CASE WHEN status >= 400 THEN 1 ELSE 0 END) as errors,
                            COALESCE(SUM(COALESCE(bytes, 0)), 0) as bytes_total,
                            AVG(latency_ms)::float as latency_avg_ms
                        FROM usage_log
                        WHERE user_id IS NOT NULL AND date(ts AT TIME ZONE 'UTC') = ?::date
                        GROUP BY user_id
                        ON CONFLICT (user_id, day) DO UPDATE SET
                            requests = EXCLUDED.requests,
                            errors = EXCLUDED.errors,
                            bytes_total = EXCLUDED.bytes_total,
                            latency_avg_ms = EXCLUDED.latency_avg_ms
                        """,
                        day_val,
                        day_val,
                    )
            else:
                # SQLite: INSERT OR REPLACE grouped aggregates.
                try:
                    await self.db_pool.execute(
                        """
                        INSERT OR REPLACE INTO usage_daily (user_id, day, requests, errors, bytes_total, bytes_in_total, latency_avg_ms)
                        SELECT
                            user_id as user_id,
                            ? as day,
                            COUNT(*) as requests,
                            SUM(CASE WHEN status >= 400 THEN 1 ELSE 0 END) as errors,
                            IFNULL(SUM(IFNULL(bytes, 0)), 0) as bytes_total,
                            IFNULL(SUM(IFNULL(bytes_in, 0)), 0) as bytes_in_total,
                            AVG(latency_ms) as latency_avg_ms
                        FROM usage_log
                        WHERE user_id IS NOT NULL AND DATE(ts) = ?
                        GROUP BY user_id
                        """,
                        day_str,
                        day_str,
                    )
                except Exception:
                    await self.db_pool.execute(
                        """
                        INSERT OR REPLACE INTO usage_daily (user_id, day, requests, errors, bytes_total, latency_avg_ms)
                        SELECT
                            user_id as user_id,
                            ? as day,
                            COUNT(*) as requests,
                            SUM(CASE WHEN status >= 400 THEN 1 ELSE 0 END) as errors,
                            IFNULL(SUM(IFNULL(bytes, 0)), 0) as bytes_total,
                            AVG(latency_ms) as latency_avg_ms
                        FROM usage_log
                        WHERE user_id IS NOT NULL AND DATE(ts) = ?
                        GROUP BY user_id
                        """,
                        day_str,
                        day_str,
                    )
        except Exception as exc:  # pragma: no cover - surfaced via callers
            if _looks_like_sqlite_corruption(exc):
                _log_sqlite_corruption_skip_once(
                    operation="aggregate_usage_daily_for_day",
                    db_pool=self.db_pool,
                    exc=exc,
                )
                return
            logger.error(f"AuthnzUsageRepo.aggregate_usage_daily_for_day failed: {exc}")
            raise

    async def aggregate_llm_usage_daily_for_day(self, *, day: date | None = None) -> None:
        """
        Aggregate per-request LLM usage from ``llm_usage_log`` into ``llm_usage_daily`` for a UTC day.

        Mirrors app/services/llm_usage_aggregator.py.
        """
        try:
            day_val = day if isinstance(day, date) else datetime.now(timezone.utc).date()
            day_str = day_val.isoformat()

            if getattr(self.db_pool, "pool", None) is not None:
                await self.db_pool.execute(
                    """
                    INSERT INTO llm_usage_daily (
                        day, user_id, operation, provider, model,
                        requests, errors, input_tokens, output_tokens, total_tokens, total_cost_usd, latency_avg_ms
                    )
                    SELECT
                        ?::date as day,
                        user_id as user_id,
                        COALESCE(operation,'') as operation,
                        COALESCE(provider,'') as provider,
                        COALESCE(model,'') as model,
                        COUNT(*) as requests,
                        SUM(CASE WHEN status >= 400 THEN 1 ELSE 0 END) as errors,
                        COALESCE(SUM(COALESCE(prompt_tokens,0)),0) as input_tokens,
                        COALESCE(SUM(COALESCE(completion_tokens,0)),0) as output_tokens,
                        COALESCE(SUM(COALESCE(total_tokens,0)),0) as total_tokens,
                        COALESCE(SUM(COALESCE(total_cost_usd,0)),0) as total_cost_usd,
                        AVG(latency_ms)::float as latency_avg_ms
                    FROM llm_usage_log
                    WHERE user_id IS NOT NULL AND date(ts AT TIME ZONE 'UTC') = ?::date
                    GROUP BY user_id, COALESCE(operation,''), COALESCE(provider,''), COALESCE(model,'')
                    ON CONFLICT (day, user_id, operation, provider, model) DO UPDATE SET
                        requests = EXCLUDED.requests,
                        errors = EXCLUDED.errors,
                        input_tokens = EXCLUDED.input_tokens,
                        output_tokens = EXCLUDED.output_tokens,
                        total_tokens = EXCLUDED.total_tokens,
                        total_cost_usd = EXCLUDED.total_cost_usd,
                        latency_avg_ms = EXCLUDED.latency_avg_ms
                    """,
                    day_val,
                    day_val,
                )
            else:
                await self.db_pool.execute(
                    """
                    INSERT OR REPLACE INTO llm_usage_daily (
                        day, user_id, operation, provider, model,
                        requests, errors, input_tokens, output_tokens, total_tokens, total_cost_usd, latency_avg_ms
                    )
                    SELECT
                        ? as day,
                        user_id as user_id,
                        IFNULL(operation,'') as operation,
                        IFNULL(provider,'') as provider,
                        IFNULL(model,'') as model,
                        COUNT(*) as requests,
                        SUM(CASE WHEN status >= 400 THEN 1 ELSE 0 END) as errors,
                        IFNULL(SUM(IFNULL(prompt_tokens,0)),0) as input_tokens,
                        IFNULL(SUM(IFNULL(completion_tokens,0)),0) as output_tokens,
                        IFNULL(SUM(IFNULL(total_tokens,0)),0) as total_tokens,
                        IFNULL(SUM(IFNULL(total_cost_usd,0)),0) as total_cost_usd,
                        AVG(latency_ms) as latency_avg_ms
                    FROM llm_usage_log
                    WHERE user_id IS NOT NULL AND DATE(ts) = ?
                    GROUP BY user_id, IFNULL(operation,''), IFNULL(provider,''), IFNULL(model,'')
                    """,
                    day_str,
                    day_str,
                )
        except Exception as exc:  # pragma: no cover - surfaced via callers
            if _looks_like_sqlite_corruption(exc):
                _log_sqlite_corruption_skip_once(
                    operation="aggregate_llm_usage_daily_for_day",
                    db_pool=self.db_pool,
                    exc=exc,
                )
                return
            logger.error(f"AuthnzUsageRepo.aggregate_llm_usage_daily_for_day failed: {exc}")
            raise

    async def prune_llm_usage_daily_before(self, cutoff_day: date) -> int:
        """
        Delete ``llm_usage_daily`` rows older than the given cutoff day.

        Returns the number of deleted rows (best-effort).
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    result = await conn.execute(
                        "DELETE FROM llm_usage_daily WHERE day < $1::date",
                        cutoff_day,
                    )
                    try:
                        return int(result.split()[-1]) if isinstance(result, str) else 0
                    except Exception:
                        return 0
                cursor = await conn.execute(
                    "DELETE FROM llm_usage_daily WHERE day < ?",
                    (cutoff_day.isoformat(),),
                )
                deleted = getattr(cursor, "rowcount", 0) or 0
                return int(deleted)
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzUsageRepo.prune_llm_usage_daily_before failed: {exc}")
            raise
