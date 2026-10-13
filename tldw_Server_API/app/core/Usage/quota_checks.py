"""Usage-quota checks shared by enforcement sites (spec 2 §4).

``check_usage`` runs a site's counter only when the user has a limit, so
unlimited users cost no query. Sites raise their own existing errors.
"""

from __future__ import annotations

import asyncio
import time
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

LLM_TOKENS_PER_MONTH_KEY = "limits.llm_tokens_per_month"
LLM_COST_PER_DAY_KEY = "limits.llm_cost_per_day_usd"
LLM_COST_PER_MONTH_KEY = "limits.llm_cost_per_month_usd"


@dataclass(frozen=True)
class LLMQuotaRefusal:
    """Why LLM work for a user is refused (LLM usage accounting spec §4)."""

    category: str
    current: float
    limit: int | float
    message: str
    retry_after: int


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


def rate_limit_headers(limit: int, remaining: int, reset_seconds: int) -> dict[str, str]:
    """RFC-style RateLimit-* (reset as delta seconds) plus legacy X-RateLimit-* (reset as epoch)."""
    delta = max(0, int(reset_seconds))
    left = str(max(0, int(remaining)))
    return {
        "RateLimit-Limit": str(int(limit)),
        "RateLimit-Remaining": left,
        "RateLimit-Reset": str(delta),
        "Retry-After": str(delta),
        "X-RateLimit-Limit": str(int(limit)),
        "X-RateLimit-Remaining": left,
        "X-RateLimit-Reset": str(int(time.time()) + delta),
    }


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


def _utc_day_start() -> datetime:
    """Midnight UTC today."""
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


async def _usage_repo() -> Any:
    """The AuthNZ usage repository on the shared pool."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo

    return AuthnzUsageRepo(await get_db_pool())


async def llm_tokens_this_month(user_id: int) -> float:
    """This calendar month's (UTC) ``llm_usage_log`` token total for the user."""
    repo = await _usage_repo()
    return await repo.sum_user_llm_tokens_since(user_id=user_id, since=_utc_day_start().replace(day=1))


async def llm_cost_today(user_id: int) -> float:
    """Today's (UTC) LLM spend in USD for the user."""
    repo = await _usage_repo()
    return await repo.sum_user_llm_cost_since(user_id=user_id, since=_utc_day_start())


async def llm_cost_this_month(user_id: int) -> float:
    """This calendar month's (UTC) LLM spend in USD for the user."""
    repo = await _usage_repo()
    return await repo.sum_user_llm_cost_since(user_id=user_id, since=_utc_day_start().replace(day=1))


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


def _count_refusal(category: str) -> None:
    """Count one refusal in ``llm_quota_refusals_total``; metrics never decide a request."""
    try:
        from tldw_Server_API.app.core.Metrics import increment_counter

        increment_counter("llm_quota_refusals_total", 1, labels={"category": category})
    except Exception:  # noqa: BLE001 - a metrics failure must not change the decision
        logger.debug("LLM quota refusal metric failed")


async def _cost_refusal(
    uid: int | None,
    key: str,
    counter: Callable[[int], Awaitable[float]],
    category: str,
    message: str,
    retry_after: Callable[[], int],
) -> LLMQuotaRefusal | None:
    """Refuse when the spend so far has reached the cap. An unset cap, or a failed counter, admits."""
    limit = await user_quota(uid, key)
    if limit is None or uid is None:
        return None
    try:
        used = float(await counter(uid))
    except Exception:  # noqa: BLE001 - a counter failure must not block requests (spec 2 §2)
        logger.opt(exception=True).warning(
            "Usage quota counter failed for key={} user_id={}; treating as unlimited", key, uid
        )
        return None
    if used < float(limit):
        return None
    return LLMQuotaRefusal(category, round(used, 4), limit, message, retry_after())


async def llm_quota_refusal(user_id: Any, requested_tokens: int = 1) -> LLMQuotaRefusal | None:
    """The first LLM quota the user is over, or None (LLM usage accounting spec §4).

    Tokens refuse when this month's tokens plus ``requested_tokens`` would pass
    the limit; with 1 requested, that is when the user is at or over it. Cost
    refuses when today's or this month's spend has reached its limit. Quotas
    off, unset limits and counter failures admit.
    """
    uid = as_quota_user_id(user_id)
    tokens = await check_usage(
        uid, LLM_TOKENS_PER_MONTH_KEY, max(1, int(requested_tokens)), lambda: llm_tokens_this_month(uid)
    )
    refusal: LLMQuotaRefusal | None = None
    if not tokens.allowed:
        refusal = LLMQuotaRefusal(
            "llm_tokens_month", int(tokens.used), tokens.limit, "Monthly LLM token limit reached",
            seconds_until_utc_month_start(),
        )
    if refusal is None:
        refusal = await _cost_refusal(
            uid, LLM_COST_PER_DAY_KEY, llm_cost_today, "llm_cost_day", "Daily LLM cost limit reached",
            seconds_until_utc_midnight,
        )
    if refusal is None:
        refusal = await _cost_refusal(
            uid, LLM_COST_PER_MONTH_KEY, llm_cost_this_month, "llm_cost_month", "Monthly LLM cost limit reached",
            seconds_until_utc_month_start,
        )
    if refusal is not None:
        _count_refusal(refusal.category)
    return refusal


_LLM_QUOTA_SETTINGS_WARNED = False


def warn_if_llm_quotas_defeated() -> list[str]:
    """Log, once per process, a setting that silently disables the LLM quotas while quotas are on (spec §6).

    Returns the problems it logged, for tests. Never raises.
    """
    global _LLM_QUOTA_SETTINGS_WARNED
    problems: list[str] = []
    try:
        from tldw_Server_API.app.core.config import usage_quotas_enabled

        if _LLM_QUOTA_SETTINGS_WARNED or not usage_quotas_enabled():
            return problems
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings
        from tldw_Server_API.app.core.Usage.usage_tracker import _enabled as llm_usage_recording_enabled

        if not llm_usage_recording_enabled():
            problems.append(
                "LLM_USAGE_ENABLED is off, so no LLM usage is recorded and the LLM quotas never refuse anything."
            )
        retention = int(getattr(get_settings(), "LLM_USAGE_LOG_RETENTION_DAYS", 180))
        if retention < 31:
            problems.append(
                f"LLM_USAGE_LOG_RETENTION_DAYS is {retention}, so the monthly LLM quotas count less than a month. "
                "Set it to 31 or more."
            )
        _LLM_QUOTA_SETTINGS_WARNED = True
        for problem in problems:
            logger.warning("Usage quotas are on, but {}", problem)
    except Exception:  # noqa: BLE001 - a startup diagnostic must never stop the server
        logger.opt(exception=True).debug("LLM quota settings check failed")
    return problems


async def llm_quota_summary(user_id: int) -> dict[str, Any]:
    """The user's LLM usage this month and today, and their LLM limits, for the profile (spec §9)."""
    return {
        "tokens_month_used": int(await llm_tokens_this_month(user_id)),
        "tokens_month_limit": await user_quota(user_id, LLM_TOKENS_PER_MONTH_KEY),
        "cost_day_usd_used": round(await llm_cost_today(user_id), 4),
        "cost_day_usd_limit": await user_quota(user_id, LLM_COST_PER_DAY_KEY),
        "cost_month_usd_used": round(await llm_cost_this_month(user_id), 4),
        "cost_month_usd_limit": await user_quota(user_id, LLM_COST_PER_MONTH_KEY),
    }
