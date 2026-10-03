"""Usage-quota checks shared by enforcement sites (spec 2 §4).

``check_usage`` runs a site's counter only when the user has a limit, so
unlimited users cost no query. Sites raise their own existing errors.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

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
    try:
        used = float(await counter())
    except Exception:  # noqa: BLE001 - a counter failure must not block requests (spec 2 §2)
        logger.opt(exception=True).warning(
            "Usage quota counter failed for key={} user_id={}; treating as unlimited", key, user_id
        )
        return UNLIMITED
    return QuotaDecision(allowed=used + float(requested) <= float(limit), limit=limit, used=used)


def seconds_until_utc_midnight() -> int:
    """Seconds until the daily counters reset (UTC), at least 1."""
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((tomorrow - now).total_seconds()))


def seconds_until_utc_month_start() -> int:
    """Seconds until the monthly counters reset (UTC), at least 1."""
    now = datetime.now(timezone.utc)
    if now.month == 12:
        next_month_start = now.replace(year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        next_month_start = now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((next_month_start - now).total_seconds()))


def _utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


_cached_ledger: Any | None = None
_cached_ledger_lock = asyncio.Lock()


async def _ledger() -> Any:
    """The shared, lazily-initialized ResourceDailyLedger (cached at module level).

    Re-running ``initialize()`` (DDL in a write transaction, plus an INFO log)
    on every call was wasteful; cache one instance, the way
    ``Workflows.daily_ledger.get_workflows_daily_ledger`` does.
    """
    global _cached_ledger
    if _cached_ledger is not None:
        return _cached_ledger
    async with _cached_ledger_lock:
        if _cached_ledger is not None:
            return _cached_ledger
        from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import ResourceDailyLedger

        ledger = ResourceDailyLedger()
        await ledger.initialize()
        _cached_ledger = ledger
        return _cached_ledger


def reset_ledger_cache() -> None:
    """Test hook: drop the cached ledger instance so a new DB target takes effect."""
    global _cached_ledger
    _cached_ledger = None


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
    from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo

    pool = await get_db_pool()
    month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return await AuthnzUsageRepo(pool).sum_user_llm_tokens_since(user_id=user_id, since=month_start)


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
    """The user's daily workflow-run allowance (``limits.workflows_runs_per_day``).

    UNLIMITED when ``WORKFLOWS_DISABLE_QUOTAS`` is set, so every caller (the
    endpoint and the scheduler's direct call for scheduled runs) honors the
    same escape hatch.
    """
    from tldw_Server_API.app.core.testing import env_flag_enabled
    from tldw_Server_API.app.core.Workflows.daily_ledger import workflows_ledger_category

    if env_flag_enabled("WORKFLOWS_DISABLE_QUOTAS"):
        return UNLIMITED
    uid = as_quota_user_id(user_id)
    return await check_usage(
        uid,
        "limits.workflows_runs_per_day",
        1,
        lambda: ledger_used_today(str(uid), workflows_ledger_category()),
    )


async def workflows_runs_consume(user_id: Any, run_id: str) -> QuotaDecision:
    """Atomically admit and record one workflow run.

    Unlike ``workflows_runs_decision`` (read-only), this performs the
    admission check and the ledger write as one atomic operation
    (``ResourceDailyLedger.add_if_within_daily_cap``), keyed by ``run_id``, so
    two concurrent runs cannot both pass against the same remaining slot.

    ``WORKFLOWS_DISABLE_QUOTAS`` and an unset ``limits.workflows_runs_per_day``
    both mean unlimited: the run is admitted and still shadow-recorded (gate
    the check, never the record).
    """
    from tldw_Server_API.app.core.testing import env_flag_enabled
    from tldw_Server_API.app.core.Workflows.daily_ledger import consume_workflow_run_if_within_cap

    uid = as_quota_user_id(user_id)
    if env_flag_enabled("WORKFLOWS_DISABLE_QUOTAS"):
        await consume_workflow_run_if_within_cap(
            entity_scope="user", entity_value=str(uid), run_id=run_id, daily_cap=None
        )
        return UNLIMITED

    limit = await user_quota(uid, "limits.workflows_runs_per_day")
    if limit is None:
        await consume_workflow_run_if_within_cap(
            entity_scope="user", entity_value=str(uid), run_id=run_id, daily_cap=None
        )
        return UNLIMITED

    allowed, remaining = await consume_workflow_run_if_within_cap(
        entity_scope="user", entity_value=str(uid), run_id=run_id, daily_cap=int(limit)
    )
    used = max(0, int(limit) - int(remaining))
    return QuotaDecision(allowed=allowed, limit=limit, used=used)
