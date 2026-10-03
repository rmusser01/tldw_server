"""Usage-quota checks shared by enforcement sites (spec 2 §4).

``check_usage`` runs a site's counter only when the user has a limit, so
unlimited users cost no query. Sites raise their own existing errors.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from tldw_Server_API.app.core.Usage.quota_resolver import user_quota

RAG_QUERIES_CATEGORY = "rag_queries"


@dataclass(frozen=True)
class QuotaDecision:
    """Whether a request fits the user's quota; ``limit`` None means unlimited."""

    allowed: bool
    limit: int | float | None
    used: float


UNLIMITED = QuotaDecision(allowed=True, limit=None, used=0.0)


def as_quota_user_id(value: Any) -> int | None:
    """The numeric user id quotas are keyed by, or None (treated as unlimited)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def check_usage(
    user_id: int | None,
    key: str,
    requested: float,
    counter: Callable[[], Awaitable[float]],
) -> QuotaDecision:
    """Admit when the user has no ``key`` limit, or when used + requested stays within it."""
    limit = await user_quota(user_id, key)
    if limit is None:
        return UNLIMITED
    used = float(await counter())
    return QuotaDecision(allowed=used + float(requested) <= float(limit), limit=limit, used=used)


def seconds_until_utc_midnight() -> int:
    """Seconds until the daily counters reset (UTC), at least 1."""
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((tomorrow - now).total_seconds()))


def _utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


async def _ledger() -> Any:
    from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import ResourceDailyLedger

    ledger = ResourceDailyLedger()
    await ledger.initialize()
    return ledger


async def ledger_used_today(entity_value: str, category: str) -> float:
    """Today's (UTC) ledger total for ``user:<entity_value>`` in ``category``."""
    ledger = await _ledger()
    return float(await ledger.total_for_day("user", str(entity_value), category))


async def ledger_used_this_month(entity_value: str, category: str) -> float:
    """This calendar month's (UTC) ledger total for ``user:<entity_value>`` in ``category``."""
    ledger = await _ledger()
    month_start = datetime.now(timezone.utc).date().replace(day=1).isoformat()
    result = await ledger.peek_range("user", str(entity_value), category, month_start, _utc_today())
    return float(result.get("total") or 0)


async def llm_tokens_this_month(user_id: int) -> float:
    """This calendar month's (UTC) ``llm_usage_log`` token total for the user."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    pool = await get_db_pool()
    month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    # llm_usage_log.ts is TIMESTAMP without time zone on Postgres: bind naive UTC there.
    bound: Any = (
        month_start.replace(tzinfo=None)
        if getattr(pool, "pool", None) is not None
        else month_start.strftime("%Y-%m-%d %H:%M:%S")
    )
    value = await pool.fetchval(
        "SELECT COALESCE(SUM(total_tokens), 0) FROM llm_usage_log WHERE user_id = ? AND ts >= ?",
        int(user_id),
        bound,
    )
    return float(value or 0)


async def rag_queries_decision(user_id: Any, units: int) -> QuotaDecision:
    """The user's daily RAG-query allowance (``limits.rag_queries_per_day``)."""
    uid = as_quota_user_id(user_id)
    return await check_usage(
        uid,
        "limits.rag_queries_per_day",
        units,
        lambda: ledger_used_today(str(uid), RAG_QUERIES_CATEGORY),
    )


async def workflows_runs_decision(user_id: Any) -> QuotaDecision:
    """The user's daily workflow-run allowance (``limits.workflows_runs_per_day``)."""
    from tldw_Server_API.app.core.Workflows.daily_ledger import workflows_ledger_category

    uid = as_quota_user_id(user_id)
    return await check_usage(
        uid,
        "limits.workflows_runs_per_day",
        1,
        lambda: ledger_used_today(str(uid), workflows_ledger_category()),
    )
