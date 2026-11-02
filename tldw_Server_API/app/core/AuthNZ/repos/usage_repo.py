from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_CEILING, Decimal
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.repos._dual_backend import execute, fetch_all, fetch_one, fetch_value
from tldw_Server_API.app.core.AuthNZ.repos.datetime_utils import _strip_tzinfo
from tldw_Server_API.app.core.AuthNZ.repos.provider_usage_reservations_repo import (
    BillingScope,
    ProviderUsageReservation,
    ReservationActuals,
)

COST_UNITS_PER_USD = 1_000_000_000
_MAX_ACCOUNTING_UNITS = (1 << 63) - 1


def _accounting_count(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_ACCOUNTING_UNITS:
        raise ValueError("Invalid accounting count")
    return value


def _accounting_aggregate(value: Any) -> int:
    """Accept PostgreSQL's exact NUMERIC sums without coercing invalid counts."""
    if type(value) is Decimal:
        if not value.is_finite() or not 0 <= value <= _MAX_ACCOUNTING_UNITS or value != value.to_integral_value():
            raise ValueError("Invalid accounting aggregate")
        value = int(value)
    return _accounting_count(value)


def _accounting_totals(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Read exact MCP settlement costs and conservatively round legacy costs."""
    tokens = costs = 0
    for row in rows:
        count = 0 if row["total_tokens"] is None else row["total_tokens"]
        tokens = _accounting_count(tokens + _accounting_count(count))
        if row["operation"] == "mcp_model_completion":
            if row["reservation_state"] != "reconciled":
                raise ValueError("MCP usage lacks atomic settlement")
            units = _accounting_count(row["actual_cost_units"])
        else:
            raw_cost = 0 if row["total_cost_usd"] is None else row["total_cost_usd"]
            cost = Decimal(str(raw_cost))
            if not cost.is_finite() or cost < 0:
                raise ValueError("Invalid accounting cost")
            units = int((cost * COST_UNITS_PER_USD).to_integral_value(rounding=ROUND_CEILING))
        costs = _accounting_count(costs + _accounting_count(units))
    return tokens, costs


def _scope_usage_parts(scope: BillingScope) -> tuple[str, str, str, tuple[int, ...]]:
    if type(scope.value) is not int or scope.value <= 0 or scope.kind not in {"user", "team", "org"}:
        raise ValueError("Invalid accounting scope")
    joins = """
        FROM llm_usage_log AS l
        LEFT JOIN provider_usage_reservations AS r
          ON l.operation = 'mcp_model_completion' AND r.execution_id = l.request_id
    """
    if scope.kind == "team":
        return "", joins, "r.billing_scope_type = 'team' AND r.billing_scope_id = ?", (scope.value,)
    if scope.kind == "user":
        return (
            "",
            joins,
            """l.user_id = ? AND l.billing_org_id IS NULL
            AND (r.billing_scope_type IS NULL OR r.billing_scope_type = 'user')""",
            (scope.value,),
        )
    prefix = """
        WITH primary_org AS (
            SELECT user_id, org_id FROM (
                SELECT user_id, org_id, ROW_NUMBER() OVER (
                    PARTITION BY user_id ORDER BY added_at ASC, org_id ASC
                ) AS rn FROM org_members WHERE added_at IS NOT NULL
            ) ranked WHERE rn = 1
        )
    """
    joins += """
        LEFT JOIN primary_org AS po ON l.user_id = po.user_id
        LEFT JOIN api_keys AS ak ON l.key_id = ak.id
    """
    predicate = """(l.billing_org_id = ? OR (l.billing_org_id IS NULL
        AND COALESCE(l.operation, '') <> 'mcp_model_completion'
        AND (po.org_id = ? OR ak.org_id = ?)))"""
    return prefix, joins, predicate, (scope.value, scope.value, scope.value)


def _scope_usage_query(scope: BillingScope, selection: str, time_filter: str) -> tuple[str, tuple[int, ...]]:
    prefix, joins, predicate, args = _scope_usage_parts(scope)
    return prefix + selection + joins + " WHERE " + predicate + " AND " + time_filter, args


# LLM quota counters leave embeddings out. ``operation`` is nullable, so a bare
# ``<>`` would also drop rows with no operation. This is a constant, never user
# input, so the queries that interpolate it carry ``nosec B608``.
_LLM_GENERATION_PREDICATE = "COALESCE(operation, '') <> 'embeddings'"
# Nor do they count calls paid by the user's own provider key (TASK-13955). NULL (unknown, and every
# row written before the column existed) counts.
_OPERATOR_PAID_PREDICATE = "COALESCE(credential_source, '') <> 'user'"
_MISSING_SOURCE_WARNED: set[str] = set()


def missing_credential_source(exc: BaseException, where: str) -> bool:
    """True when ``exc`` names ``llm_usage_log.credential_source``: the schema predates the column.

    The caller then retries without the column, so every row counts. The first
    time per ``where`` it warns, because the LLM limits then count calls on the
    user's own key.
    """
    if "credential_source" not in str(exc):
        return False
    if where not in _MISSING_SOURCE_WARNED:
        _MISSING_SOURCE_WARNED.add(where)
        logger.bind(exception_type=type(exc).__name__).warning(
            "llm_usage_log has no credential_source column, so calls on users' own keys count ({}); "
            "apply AuthNZ migration 104, or check the PostgreSQL usage-table bootstrap",
            where,
        )
    return True


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

    async def read_provider_scope_usage(
        self,
        conn: Any,
        scope: BillingScope,
        month_start: datetime,
        *,
        operator_paid_only: bool = False,
    ) -> tuple[int, int]:
        """Read strict canonical actuals on the caller's locked transaction.

        Unresolved reservations are added by the admission repository. Convert
        legacy cost rows individually. MCP costs use the exact integer audit
        value committed atomically with the canonical usage record.
        """
        is_pg = self._is_postgres_backend()
        sql, args = _scope_usage_query(
            scope,
            """SELECT l.total_tokens, l.total_cost_usd, l.operation,
                r.state AS reservation_state, r.actual_cost_units""",
            ("l.ts >= ?" if is_pg else "datetime(l.ts) >= datetime(?)")
            + (" AND COALESCE(l.credential_source, '') <> 'user'" if operator_paid_only else ""),
        )
        since = _strip_tzinfo(month_start) if is_pg else _strip_tzinfo(month_start).isoformat(" ", timespec="seconds")
        rows = await fetch_all(
            conn,
            is_pg,
            sql,
            (*args, since),
            (
                "total_tokens",
                "total_cost_usd",
                "operation",
                "reservation_state",
                "actual_cost_units",
            ),
        )
        return _accounting_totals(rows)

    async def read_user_llm_exposure(self, conn: Any, user_id: int, since: datetime) -> tuple[int, int]:
        """Count native paid usage plus unresolved MCP exposure across all scopes.

        The caller holds the durable user lock also taken by MCP settlement.
        Completed usage is period-bound; unresolved exposure never ages out.
        Legacy Chat remains outside this reservation-lock protocol.
        """
        if _accounting_count(user_id) == 0:
            raise ValueError("Invalid accounting user")
        is_pg = self._is_postgres_backend()
        time_filter = "l.ts >= ?" if is_pg else "datetime(l.ts) >= datetime(?)"
        bound = _strip_tzinfo(since) if is_pg else _strip_tzinfo(since).isoformat(" ", timespec="seconds")
        # Only the two fixed timestamp predicates vary; all values remain bound.
        sql = (
            "SELECT l.total_tokens, l.total_cost_usd, l.operation, "  # nosec B608
            """r.state AS reservation_state, r.actual_cost_units
               FROM llm_usage_log AS l
               LEFT JOIN provider_usage_reservations AS r
                 ON l.operation = 'mcp_model_completion' AND r.execution_id = l.request_id
               WHERE l.user_id = ? AND """
            + time_filter
            + " AND COALESCE(l.operation, '') <> 'embeddings' AND COALESCE(l.credential_source, '') <> 'user'"
        )
        rows = await fetch_all(conn, is_pg, sql, (user_id, bound), ())
        tokens, costs = _accounting_totals(rows)
        unresolved = await fetch_all(
            conn,
            is_pg,
            """SELECT reserved_input_tokens, reserved_output_tokens, reserved_cost_units
               FROM provider_usage_reservations
               WHERE user_id = ? AND state IN ('reserved', 'dispatched', 'ambiguous')
                 AND COALESCE(credential_source, '') <> 'user'""",
            (user_id,),
            (),
        )
        for row in unresolved:
            tokens = _accounting_count(
                tokens
                + _accounting_count(row["reserved_input_tokens"])
                + _accounting_count(row["reserved_output_tokens"])
            )
            costs = _accounting_count(costs + _accounting_count(row["reserved_cost_units"]))
        return tokens, costs

    async def read_org_token_exposure(self, conn: Any, org_id: int, month_start: datetime) -> int:
        """Include unresolved MCP exposure while preserving ADR-083's own-key exclusion."""
        is_pg = self._is_postgres_backend()
        sql, args = _scope_usage_query(
            BillingScope("org", org_id),
            """SELECT COALESCE(SUM(l.total_tokens), 0) + (
                SELECT COALESCE(SUM(reserved_input_tokens + reserved_output_tokens), 0)
                FROM provider_usage_reservations
                WHERE billing_scope_type = 'org' AND billing_scope_id = ?
                  AND state IN ('reserved', 'dispatched', 'ambiguous')
                  AND COALESCE(credential_source, '') <> 'user'
            )""",
            ("l.ts >= ?" if is_pg else "datetime(l.ts) >= datetime(?)")
            + " AND COALESCE(l.credential_source, '') <> 'user'",
        )
        since = _strip_tzinfo(month_start) if is_pg else _strip_tzinfo(month_start).isoformat(" ", timespec="seconds")
        return _accounting_aggregate(await fetch_value(conn, is_pg, sql, (org_id, *args, since)))

    async def insert_mcp_completion_usage(
        self,
        conn: Any,
        reservation: ProviderUsageReservation,
        actuals: ReservationActuals,
        *,
        estimated: bool,
        input_cost_units: int = 0,
    ) -> None:
        """Insert minimized usage atomically with reservation settlement, without fallback."""
        is_pg = self._is_postgres_backend()
        if type(estimated) is not bool:
            raise ValueError("Invalid usage estimation flag")
        input_tokens = _accounting_count(actuals.input_tokens)
        output_tokens = _accounting_count(actuals.output_tokens)
        total_tokens = _accounting_count(input_tokens + output_tokens)
        cost_units = _accounting_count(actuals.cost_units)
        input_cost_units = _accounting_count(input_cost_units)
        if input_cost_units > cost_units:
            raise ValueError("Invalid accounting cost split")
        org_id = reservation.billing_scope.value if reservation.billing_scope.kind == "org" else None
        total_cost = float(Decimal(cost_units) / COST_UNITS_PER_USD)
        values = (
            reservation.user_id,
            org_id,
            reservation.provider,
            reservation.model,
            input_tokens,
            output_tokens,
            total_tokens,
            float(Decimal(input_cost_units) / COST_UNITS_PER_USD),
            float(Decimal(cost_units - input_cost_units) / COST_UNITS_PER_USD),
            total_cost,
            bool(estimated),
            reservation.execution_id,
            "mcp_conservative_ceiling" if estimated else "mcp_provider_counts",
            getattr(reservation, "credential_source", None),
        )
        now = _strip_tzinfo(datetime.now(timezone.utc))
        timestamp = now if is_pg else now.isoformat(" ", timespec="seconds")
        await execute(
            conn,
            is_pg,
            """
            INSERT INTO llm_usage_log (
                ts, user_id, billing_org_id, endpoint, operation, provider, model, status, latency_ms,
                prompt_tokens, completion_tokens, total_tokens, prompt_cost_usd, completion_cost_usd,
                total_cost_usd, currency, estimated, request_id, estimate_source, credential_source
            ) VALUES (?, ?, ?, '/mcp/model_completion', 'mcp_model_completion',
                      ?, ?, 200, 0, ?, ?, ?, ?, ?, ?, 'USD', ?, ?, ?, ?)
            ON CONFLICT (request_id) WHERE operation = 'mcp_model_completion' AND request_id IS NOT NULL
            DO NOTHING
        """,
            (timestamp, *values),
        )
        row = await fetch_one(
            conn,
            is_pg,
            """
            SELECT user_id, billing_org_id, provider, model, prompt_tokens, completion_tokens,
                   total_tokens, prompt_cost_usd, completion_cost_usd, total_cost_usd,
                   endpoint, status, latency_ms, currency, estimated, estimate_source, credential_source
            FROM llm_usage_log
            WHERE request_id = ? AND operation = 'mcp_model_completion'
        """,
            (reservation.execution_id,),
            (
                "user_id",
                "billing_org_id",
                "provider",
                "model",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "prompt_cost_usd",
                "completion_cost_usd",
                "total_cost_usd",
                "endpoint",
                "status",
                "latency_ms",
                "currency",
                "estimated",
                "estimate_source",
                "credential_source",
            ),
        )
        expected = {
            "user_id": reservation.user_id,
            "billing_org_id": org_id,
            "provider": reservation.provider,
            "model": reservation.model,
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
            "prompt_cost_usd": values[7],
            "completion_cost_usd": values[8],
            "total_cost_usd": total_cost,
            "endpoint": "/mcp/model_completion",
            "status": 200,
            "latency_ms": 0,
            "currency": "USD",
            "estimated": estimated,
            "estimate_source": values[12],
            "credential_source": values[13],
        }
        for key in ("prompt_cost_usd", "completion_cost_usd", "total_cost_usd"):
            expected[key] = Decimal(str(expected[key]))
            if row is not None:
                row[key] = Decimal(str(row[key]))
        if row != expected:
            raise ValueError("Conflicting MCP usage replay")

    async def sum_org_api_requests(self, *, org_id: int, day: date) -> int:
        """Sum the UTC daily rollup, assigning each user to one primary org.

        The aggregate stores user/day counts, not request-time org or API-key
        attribution. Use active memberships only and follow billing's user-scoped
        LLM convention: earliest dated membership, then lowest org ID. Undated
        legacy memberships are ignored, as in the existing SQLite LLM attribution query.
        Counts reflect rollup freshness.
        Source failures propagate so the caller retains its failure policy.
        """
        day_param = day if self._is_postgres_backend() else day.isoformat()
        result = await self.db_pool.fetchval(
            """
            WITH primary_org AS (
                SELECT user_id, org_id,
                       ROW_NUMBER() OVER (
                           PARTITION BY user_id ORDER BY added_at ASC, org_id ASC
                       ) AS rank
                FROM org_members
                WHERE added_at IS NOT NULL
                  AND LOWER(TRIM(status)) = 'active'
            )
            SELECT COALESCE(SUM(u.requests), 0)
            FROM usage_daily AS u
            JOIN primary_org AS po ON po.user_id = u.user_id AND po.rank = 1
            WHERE po.org_id = ? AND u.day = ?
            """,
            org_id, day_param,
        )
        return int(result or 0)

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

    async def _sum_llm_column_since(self, column: str, *, user_id: int, since: datetime) -> float:
        """Sum one ``llm_usage_log`` column for a user from ``since`` (inclusive), embeddings excluded.

        Calls made with the user's own provider key are excluded too, except on
        a schema without ``credential_source``, where every row counts.
        ``since`` is bound naive on Postgres: ``llm_usage_log.ts`` is TIMESTAMP
        without time zone there, so an aware bound would never match.
        ``column`` is a constant chosen by the callers below, never user input.
        """
        bound: Any = _strip_tzinfo(since) if self._is_postgres_backend() else since.strftime("%Y-%m-%d %H:%M:%S")
        query = (
            f"SELECT COALESCE(SUM({column}), 0) FROM llm_usage_log "  # nosec B608
            f"WHERE user_id = ? AND ts >= ? AND {_LLM_GENERATION_PREDICATE}"
        )
        try:
            value = await self.db_pool.fetchval(f"{query} AND {_OPERATOR_PAID_PREDICATE}", int(user_id), bound)
        except Exception as exc:
            if not missing_credential_source(exc, "the per-user LLM limits"):
                raise
            value = await self.db_pool.fetchval(query, int(user_id), bound)  # never fail open: count every row
        return float(value or 0)

    async def sum_user_llm_tokens_since(self, *, user_id: int, since: datetime) -> float:
        """Sum ``llm_usage_log.total_tokens`` for a user from ``since`` onward (LLM usage accounting spec §6)."""
        return await self._sum_llm_column_since("total_tokens", user_id=user_id, since=since)

    async def sum_user_llm_cost_since(self, *, user_id: int, since: datetime) -> float:
        """Sum ``llm_usage_log.total_cost_usd`` (USD) for a user from ``since`` onward (LLM usage accounting spec §6)."""
        return await self._sum_llm_column_since("total_cost_usd", user_id=user_id, since=since)

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
                _start = _strip_tzinfo(start_dt)
                _end = _strip_tzinfo(end_dt)
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
                    cutoff_param = _strip_tzinfo(cutoff)
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
                    cutoff_param = _strip_tzinfo(cutoff)
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
        credential_source: str | None = None,
    ) -> None:
        """
        Insert a single row into ``llm_usage_log``.

        This centralizes dialect handling so callers (e.g., usage_tracker)
        do not embed backend-specific SQL. A row with a ``credential_source`` on a
        schema without that column is written without it.
        """

        async def _insert_full(with_source: bool) -> None:
            """The current statement, with or without ``credential_source``."""
            source_column = ", credential_source" if with_source else ""
            source_value = ", ?" if with_source else ""
            source_args = (credential_source,) if with_source else ()
            await self.db_pool.execute(
                f"""
                INSERT INTO llm_usage_log (
                    ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
                    prompt_tokens, completion_tokens, total_tokens,
                    prompt_cost_usd, completion_cost_usd, total_cost_usd, currency, estimated, request_id,
                    remote_ip, user_agent, token_name, conversation_id,
                    cached_input_tokens, cache_write_input_tokens, cache_read_input_tokens,
                    billable_input_tokens, reasoning_tokens, choice_count, estimate_source,
                    prompt_fingerprint, prompt_fingerprint_version, world_book_fingerprint,
                    raw_usage_metadata_json{source_column}
                ) VALUES (
                    CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?{source_value}
                )
                """,  # nosec B608 - the interpolated fragments are constants
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
                *source_args,
            )

        try:
            try:
                try:
                    await _insert_full(credential_source is not None)
                except Exception as exc:
                    if credential_source is None:
                        raise
                    missing_credential_source(exc, "usage rows")  # warns once when the column is missing
                    await _insert_full(False)  # keep the row, without its source: it counts
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
