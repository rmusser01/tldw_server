"""
enforcement.py

Billing limit enforcement that integrates with the Resource Governor system.
Provides dependencies for checking and enforcing subscription limits.
"""
from __future__ import annotations

import importlib
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Resource_Governance.governor import RGRequest

# Environment variable for cache TTL (default 60 seconds)
BILLING_CACHE_TTL_SECONDS = float(os.environ.get("BILLING_CACHE_TTL_SECONDS", "60.0"))

from tldw_Server_API.app.core.Billing.overage_config import OveragePolicy
from tldw_Server_API.app.core.Billing.plan_limits import SOFT_LIMIT_PERCENT
from tldw_Server_API.app.core.Billing.runtime_flags import is_billing_enabled

_BILLING_ENFORCEMENT_COERCE_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
)

_BILLING_ENFORCEMENT_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    sqlite3.Error,
)

_BILLING_ENFORCEMENT_FAILURE_MODE_ENV = "BILLING_ENFORCEMENT_FAILURE_MODE"
_BILLING_ENFORCEMENT_FAILURE_MODE_OPEN = "open"
_BILLING_ENFORCEMENT_FAILURE_MODE_CLOSED = "closed"
_BILLING_ENFORCEMENT_POLICY_UNSET = object()

try:  # pragma: no cover - optional dependency guard
    import asyncpg  # type: ignore
except Exception:  # pragma: no cover - asyncpg may be absent in SQLite-only setups
    asyncpg = None  # type: ignore[assignment]
else:
    _BILLING_ENFORCEMENT_NONCRITICAL_EXCEPTIONS = (
        *_BILLING_ENFORCEMENT_NONCRITICAL_EXCEPTIONS,
        asyncpg.PostgresError,  # type: ignore[attr-defined]
    )


class LimitCategory(str, Enum):
    """Categories of limits that can be enforced."""
    API_CALLS_DAY = "api_calls_day"
    LLM_TOKENS_MONTH = "llm_tokens_month"
    STORAGE_MB = "storage_mb"
    TEAM_MEMBERS = "team_members"
    TRANSCRIPTION_MINUTES_MONTH = "transcription_minutes_month"
    RAG_QUERIES_DAY = "rag_queries_day"
    CONCURRENT_JOBS = "concurrent_jobs"


class EnforcementAction(str, Enum):
    """Action to take when limit is approached/exceeded."""
    ALLOW = "allow"
    WARN = "warn"  # Allow but add warning header
    SOFT_BLOCK = "soft_block"  # Block with upgrade prompt
    HARD_BLOCK = "hard_block"  # Block completely


@dataclass
class LimitCheckResult:
    """Result of checking a limit."""
    category: str
    action: EnforcementAction
    current: int
    limit: int
    percent_used: float
    unlimited: bool = False
    message: str | None = None
    retry_after: int | None = None  # For rate limits

    @property
    def should_block(self) -> bool:
        return self.action in (EnforcementAction.SOFT_BLOCK, EnforcementAction.HARD_BLOCK)

    @property
    def should_warn(self) -> bool:
        return self.action == EnforcementAction.WARN


@dataclass
class UsageSummary:
    """Summary of current usage for an organization."""
    org_id: int
    api_calls_today: int = 0
    llm_tokens_month: int = 0
    storage_bytes: int = 0
    team_members: int = 0
    transcription_minutes_month: int = 0
    rag_queries_today: int = 0
    concurrent_jobs: int = 0


class BillingEnforcer:
    """
    Enforces billing limits by integrating with the subscription service
    and usage tracking systems.

    This class provides the bridge between:
    - Subscription limits (from billing service)
    - Current usage (from usage repos)
    - Enforcement decisions (allow/warn/block)
    """

    def __init__(
        self,
        *,
        soft_limit_percent: float = SOFT_LIMIT_PERCENT,
        grace_period_days: int = 3,
        cache_ttl: float | None = None,
    ):
        self.soft_limit_percent = soft_limit_percent
        self.grace_period_days = grace_period_days
        self._usage_cache: dict[int, tuple[UsageSummary, float]] = {}
        self._limits_cache: dict[int, tuple[dict[str, Any], float]] = {}
        # Use provided TTL, env var, or default to 60s
        self._cache_ttl = cache_ttl if cache_ttl is not None else BILLING_CACHE_TTL_SECONDS
        self._overage_policy: Any = _BILLING_ENFORCEMENT_POLICY_UNSET

    @staticmethod
    def _is_postgres_pool(pool: Any) -> bool:
        """
        Return True when the provided DatabasePool is backed by PostgreSQL.

        Backend selection should derive from pool state, not connection
        method-presence probes, because wrapper/shim connections can expose
        methods that do not indicate the active backend.
        """
        if pool is None:
            return False
        if getattr(pool, "pool", None):
            return True
        backend = getattr(pool, "backend", None)
        if isinstance(backend, str):
            return backend.strip().lower() in {"postgres", "postgresql", "pg"}
        return False

    @staticmethod
    def _get_failure_mode() -> str:
        """Return enforcement fallback mode for data-source failures."""
        raw_mode = os.environ.get(
            _BILLING_ENFORCEMENT_FAILURE_MODE_ENV,
            _BILLING_ENFORCEMENT_FAILURE_MODE_OPEN,
        )
        mode = str(raw_mode).strip().lower()
        if mode in {_BILLING_ENFORCEMENT_FAILURE_MODE_OPEN, _BILLING_ENFORCEMENT_FAILURE_MODE_CLOSED}:
            return mode

        logger.warning(
            "Invalid {}={!r}; defaulting to {}",
            _BILLING_ENFORCEMENT_FAILURE_MODE_ENV,
            raw_mode,
            _BILLING_ENFORCEMENT_FAILURE_MODE_OPEN,
        )
        return _BILLING_ENFORCEMENT_FAILURE_MODE_OPEN

    def _get_overage_policy(self) -> OveragePolicy | None:
        """Load the configured overage policy once per enforcer instance."""
        if self._overage_policy is _BILLING_ENFORCEMENT_POLICY_UNSET:
            try:
                self._overage_policy = OveragePolicy.from_env()
            except _BILLING_ENFORCEMENT_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(f"Overage policy initialization skipped: {exc}")
                self._overage_policy = None
        return self._overage_policy

    @classmethod
    def _fail_closed_on_data_error(cls) -> bool:
        return cls._get_failure_mode() == _BILLING_ENFORCEMENT_FAILURE_MODE_CLOSED

    @staticmethod
    def _permissive_limit_fallbacks() -> dict[str, Any]:
        """Fail-open fallback limits when billing data sources are unavailable."""
        return {
            "api_calls_day": 1000,
            "llm_tokens_month": 10_000_000,
            "storage_mb": 10240,
            "team_members": -1,
            "transcription_minutes_month": 600,
            "rag_queries_day": 500,
            "concurrent_jobs": 5,
        }

    @staticmethod
    def _restrictive_limit_fallbacks() -> dict[str, Any]:
        """Fail-closed fallback limits when billing data sources are unavailable."""
        return {
            "api_calls_day": 0,
            "llm_tokens_month": 0,
            "storage_mb": 0,
            "team_members": 0,
            "transcription_minutes_month": 0,
            "rag_queries_day": 0,
            "concurrent_jobs": 0,
            "advanced_analytics": False,
            "priority_support": False,
            "custom_models": False,
            "api_access": False,
            "sso_enabled": False,
            "audit_logs": False,
        }

    @staticmethod
    def _restrictive_usage_fallback(org_id: int) -> UsageSummary:
        """Fail-closed usage snapshot that forces blocking checks."""
        max_usage = 2_147_483_647
        return UsageSummary(
            org_id=org_id,
            api_calls_today=max_usage,
            llm_tokens_month=max_usage,
            storage_bytes=max_usage * (1024 ** 2),
            team_members=max_usage,
            transcription_minutes_month=max_usage,
            rag_queries_today=max_usage,
            concurrent_jobs=max_usage,
        )

    async def get_org_limits(self, org_id: int) -> dict[str, Any]:
        """Get subscription limits for an organization with caching."""
        now = time.time()

        # Check cache
        if org_id in self._limits_cache:
            cached, cached_at = self._limits_cache[org_id]
            if now - cached_at < self._cache_ttl:
                return cached

        # Fetch from service
        try:
            from tldw_Server_API.app.core.Billing.subscription_service import get_subscription_service
            service = await get_subscription_service()
            limits = await service.get_org_limits(org_id)
            self._limits_cache[org_id] = (limits, now)
            return limits
        except _BILLING_ENFORCEMENT_NONCRITICAL_EXCEPTIONS as exc:
            if self._fail_closed_on_data_error():
                logger.error(
                    "Failed to get org limits for {}: {}. Applying fail-closed fallback.",
                    org_id,
                    exc,
                )
                return self._restrictive_limit_fallbacks()
            logger.warning(
                "Failed to get org limits for {}: {}. Applying fail-open fallback.",
                org_id,
                exc,
            )
            return self._permissive_limit_fallbacks()

    async def get_org_usage(self, org_id: int) -> UsageSummary:
        """
        Get current usage summary for an organization.

        Aggregates from various usage tracking sources.
        """
        now = time.time()

        # Check cache
        if org_id in self._usage_cache:
            cached, cached_at = self._usage_cache[org_id]
            if now - cached_at < self._cache_ttl:
                return cached

        summary = UsageSummary(org_id=org_id)

        try:
            # Get API call count for today
            summary.api_calls_today = await self._get_api_calls_today(org_id)

            # Get LLM token usage for this month
            summary.llm_tokens_month = await self._get_llm_tokens_month(org_id)

            # Get team member count
            summary.team_members = await self._get_team_member_count(org_id)

            # Get concurrent jobs
            summary.concurrent_jobs = await self._get_concurrent_jobs(org_id)

            # Get storage usage
            summary.storage_bytes = await self._get_storage_bytes(org_id)

            # Get transcription minutes for this month
            summary.transcription_minutes_month = await self._get_transcription_minutes_month(org_id)

            # Get RAG queries for today
            summary.rag_queries_today = await self._get_rag_queries_today(org_id)

            # Cache the result
            self._usage_cache[org_id] = (summary, now)

        except _BILLING_ENFORCEMENT_NONCRITICAL_EXCEPTIONS as exc:
            # Log at ERROR level since this affects billing enforcement
            logger.error(f"Failed to get org usage for {org_id}: {exc}")
            # Try to return cached value if available (even if expired)
            if org_id in self._usage_cache:
                cached, _ = self._usage_cache[org_id]
                logger.warning(f"Using stale cached usage for org {org_id}")
                return cached
            if self._fail_closed_on_data_error():
                logger.error(
                    "No cached usage for org {}; returning restrictive usage (fail-closed).",
                    org_id,
                )
                return self._restrictive_usage_fallback(org_id)
            # Otherwise return zeros (fail-open) but log warning
            logger.warning(f"No cached usage for org {org_id}, returning zeros (fail-open)")

        return summary

    async def _get_api_calls_today(self, org_id: int) -> int:
        """Get API call count for today from usage_daily table."""
        try:
            from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

            pool = await get_db_pool()
            today = datetime.now(timezone.utc).date().isoformat()
            is_postgres = self._is_postgres_pool(pool)

            async with pool.acquire() as conn:
                if is_postgres:
                    query_variants: tuple[tuple[str, tuple[Any, ...]], ...] = (
                        (
                            """
                            SELECT COALESCE(SUM(requests), 0)
                            FROM usage_daily
                            WHERE org_id = $1 AND day = $2
                            """,
                            (org_id, today),
                        ),
                        (
                            """
                            SELECT COALESCE(SUM(request_count), 0)
                            FROM usage_daily
                            WHERE org_id = $1 AND day = $2
                            """,
                            (org_id, today),
                        ),
                    )
                    result = 0
                    for query, params in query_variants:
                        try:
                            fetched = await conn.fetchval(query, *params)
                            result = int(fetched or 0)
                            break
                        except Exception as exc:  # noqa: BLE001 - schema variance fallback
                            logger.debug("usage_daily PG query variant failed: {}", exc)
                            continue
                else:
                    query_variants = (
                        (
                            """
                            SELECT COALESCE(SUM(requests), 0)
                            FROM usage_daily
                            WHERE org_id = ? AND day = ?
                            """,
                            (org_id, today),
                        ),
                        (
                            """
                            SELECT COALESCE(SUM(request_count), 0)
                            FROM usage_daily
                            WHERE org_id = ? AND day = ?
                            """,
                            (org_id, today),
                        ),
                    )
                    result = 0
                    for query, params in query_variants:
                        try:
                            cur = await conn.execute(query, params)
                            row = await cur.fetchone()
                            result = int((row[0] if row else 0) or 0)
                            break
                        except Exception as exc:  # noqa: BLE001 - schema variance fallback
                            logger.debug("usage_daily SQLite query variant failed: {}", exc)
                            continue

                return int(result or 0)
        except _BILLING_ENFORCEMENT_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Failed to get API calls for org {org_id}: {exc}")
            return 0

    async def _get_llm_tokens_month(self, org_id: int) -> int:
        """Get LLM token usage for current month for an organization.

        This aggregates usage from both user-scoped and API-key scoped calls.
        Each llm_usage_log row is counted at most once per org.

        User-scoped calls are attributed to a user's primary org (earliest
        org_members.added_at, tie-broken by lowest org_id) to avoid
        double-counting when a user belongs to multiple orgs.
        """
        try:
            from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

            pool = await get_db_pool()
            now = datetime.now(timezone.utc)
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            is_postgres = self._is_postgres_pool(pool)

            async with pool.acquire() as conn:
                if is_postgres:
                    # PostgreSQL: single-pass attribution, no UNION path.
                    result = await conn.fetchval(
                        """
                        WITH primary_org AS (
                            SELECT user_id, org_id
                            FROM (
                                SELECT user_id, org_id,
                                       ROW_NUMBER() OVER (
                                           PARTITION BY user_id
                                           ORDER BY added_at ASC, org_id ASC
                                       ) AS rn
                                FROM org_members
                            ) ranked
                            WHERE rn = 1
                        )
                        SELECT COALESCE(SUM(l.total_tokens), 0)
                        FROM llm_usage_log AS l
                        LEFT JOIN primary_org AS po ON l.user_id = po.user_id
                        LEFT JOIN api_keys AS ak ON l.key_id = ak.id
                        WHERE l.ts >= $2
                          AND (
                                po.org_id = $1
                                OR ak.org_id = $1
                              )
                        """,
                        org_id, month_start,
                    )
                else:
                    # SQLite: single-pass attribution, no UNION path.
                    month_start_str = month_start.strftime("%Y-%m-%d %H:%M:%S")
                    cur = await conn.execute(
                        """
                        WITH primary_org AS (
                            SELECT om.user_id, MIN(om.added_at) AS min_added
                            FROM org_members om
                            GROUP BY om.user_id
                        ),
                        primary_org_resolved AS (
                            SELECT om.user_id, MIN(om.org_id) AS org_id
                            FROM org_members om
                            JOIN primary_org po
                              ON po.user_id = om.user_id AND po.min_added = om.added_at
                            GROUP BY om.user_id
                        )
                        SELECT COALESCE(SUM(l.total_tokens), 0)
                        FROM llm_usage_log AS l
                        LEFT JOIN primary_org_resolved po ON l.user_id = po.user_id
                        LEFT JOIN api_keys AS ak ON l.key_id = ak.id
                        WHERE l.ts >= ?
                          AND (
                                po.org_id = ?
                                OR ak.org_id = ?
                              )
                        """,
                        (month_start_str, org_id, org_id),
                    )
                    row = await cur.fetchone()
                    result = row[0] if row else 0

                return int(result or 0)
        except _BILLING_ENFORCEMENT_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Failed to get LLM tokens for org {org_id}: {exc}")
            return 0

    async def _get_team_member_count(self, org_id: int) -> int:
        """Get current team member count for organization."""
        try:
            from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
            from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo

            pool = await get_db_pool()
            repo = AuthnzOrgsTeamsRepo(db_pool=pool)
            total_members = 0
            offset = 0
            batch_size = 1000

            while True:
                members = await repo.list_org_members(
                    org_id=org_id,
                    limit=batch_size,
                    offset=offset,
                    status="active",
                )
                if not members:
                    break

                total_members += len(members)
                if len(members) < batch_size:
                    break
                offset += batch_size

            return total_members
        except _BILLING_ENFORCEMENT_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Failed to get team members for org {org_id}: {exc}")
            return 0

    async def _get_concurrent_jobs(self, org_id: int) -> int:
        """
        Get count of currently running jobs for an organization.

        This aggregates concurrent embeddings jobs per org member from the
        core Jobs table by summing processing jobs for the embeddings domain.
        Other job systems (audio/chatbooks) continue to enforce their own caps
        and are not yet included in this summary.
        """
        try:
            from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
            from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
            from tldw_Server_API.app.core.Jobs.manager import JobManager

            pool = await get_db_pool()
            repo = AuthnzOrgsTeamsRepo(db_pool=pool)

            member_ids: set[str] = set()
            offset = 0
            batch_size = 500

            while True:
                members = await repo.list_org_members(
                    org_id=org_id,
                    limit=batch_size,
                    offset=offset,
                    status="active",
                )
                if not members:
                    break

                for member in members:
                    user_id = member.get("user_id")
                    if user_id is None:
                        continue
                    member_ids.add(str(user_id))

                if len(members) < batch_size:
                    break
                offset += batch_size

            if not member_ids:
                return 0

            jm = JobManager()
            JobManager.set_rls_context(is_admin=True, domain_allowlist="embeddings", owner_user_id=None)
            try:
                summary = jm.summarize_by_owner_and_status(domain="embeddings")
            finally:
                JobManager.clear_rls_context()

            total_active = 0
            for row in summary:
                owner = row.get("owner_user_id")
                if owner is None or str(owner) not in member_ids:
                    continue
                if str(row.get("status")) != "processing":
                    continue
                try:
                    total_active += int(row.get("count") or 0)
                except (TypeError, ValueError):
                    continue

            return int(total_active)
        except _BILLING_ENFORCEMENT_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Failed to get concurrent jobs for org {org_id}: {exc}")
            return 0

    async def _get_storage_bytes(self, org_id: int) -> int:
        """Sum storage_used_mb from users table for all org members."""
        try:
            from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
            from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo

            pool = await get_db_pool()
            repo = AuthnzOrgsTeamsRepo(db_pool=pool)
            total_mb = 0
            offset = 0
            batch_size = 500  # keep below SQLite's max parameter count
            is_postgres = self._is_postgres_pool(pool)

            def _chunks(values: list[int], size: int) -> list[list[int]]:
                return [values[i:i + size] for i in range(0, len(values), size)]

            while True:
                members = await repo.list_org_members(
                    org_id=org_id,
                    limit=batch_size,
                    offset=offset,
                    status="active",
                )
                if not members:
                    break

                user_ids = [m["user_id"] for m in members if m.get("user_id")]
                if user_ids:
                    for chunk in _chunks(user_ids, batch_size):
                        async with pool.acquire() as conn:
                            if is_postgres:
                                # PostgreSQL
                                result = await conn.fetchval(
                                    "SELECT COALESCE(SUM(storage_used_mb), 0) FROM users WHERE id = ANY($1)",
                                    chunk,
                                )
                            else:
                                # SQLite
                                placeholders = ",".join("?" * len(chunk))
                                user_ids_clause = f"({placeholders})"
                                usage_sum_sql_template = (
                                    "SELECT COALESCE(SUM(storage_used_mb), 0) FROM users WHERE id IN {user_ids_clause}"
                                )
                                usage_sum_sql = usage_sum_sql_template.format_map(locals())  # nosec B608
                                cur = await conn.execute(
                                    usage_sum_sql,
                                    tuple(chunk),
                                )
                                row = await cur.fetchone()
                                result = row[0] if row else 0
                        total_mb += int(result or 0)

                if len(members) < batch_size:
                    break
                offset += batch_size

            # Convert MB to bytes
            return int(total_mb * 1024 * 1024)
        except _BILLING_ENFORCEMENT_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Failed to get storage for org {org_id}: {exc}")
            return 0

    async def _get_transcription_minutes_month(self, org_id: int) -> int:
        """Get transcription minutes for current month from resource ledger."""
        try:
            from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import ResourceDailyLedger

            ledger = ResourceDailyLedger()
            await ledger.initialize()

            now = datetime.now(timezone.utc)
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).date()
            today = now.date()

            # Query ledger for org-scoped minutes entries over the month-to-date window
            window = await ledger.peek_range(
                entity_scope="org",
                entity_value=str(org_id),
                category="minutes",
                start_day_utc=month_start.isoformat(),
                end_day_utc=today.isoformat(),
            )

            return int(window.get("total", 0)) if window else 0
        except _BILLING_ENFORCEMENT_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Failed to get transcription minutes for org {org_id}: {exc}")
            return 0

    async def _get_rag_queries_today(self, org_id: int) -> int:
        """
        Get RAG query count for today.

        RAG queries are tracked via the shared ResourceDailyLedger using the
        "rag_queries" category scoped to the organization. Each RAG endpoint
        invocation can record one or more units for the active org.
        """
        try:
            from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import ResourceDailyLedger

            ledger = ResourceDailyLedger()
            await ledger.initialize()

            today = datetime.now(timezone.utc).date().isoformat()
            total = await ledger.total_for_day(
                entity_scope="org",
                entity_value=str(org_id),
                category="rag_queries",
                day_utc=today,
            )

            return int(total or 0)
        except _BILLING_ENFORCEMENT_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Failed to get RAG queries for org {org_id}: {exc}")
            return 0

    async def check_limit(
        self,
        org_id: int,
        category: LimitCategory,
        *,
        requested_units: int = 1,
    ) -> LimitCheckResult:
        """
        Check if an operation is allowed given the org's limits.

        Args:
            org_id: Organization ID
            category: Limit category to check
            requested_units: Number of units the operation will consume

        Returns:
            LimitCheckResult with action to take
        """
        limits = await self.get_org_limits(org_id)
        usage = await self.get_org_usage(org_id)

        # Map category to current usage and limit
        usage_map = {
            LimitCategory.API_CALLS_DAY: usage.api_calls_today,
            LimitCategory.LLM_TOKENS_MONTH: usage.llm_tokens_month,
            LimitCategory.TEAM_MEMBERS: usage.team_members,
            LimitCategory.CONCURRENT_JOBS: usage.concurrent_jobs,
            LimitCategory.RAG_QUERIES_DAY: usage.rag_queries_today,
            LimitCategory.TRANSCRIPTION_MINUTES_MONTH: usage.transcription_minutes_month,
            LimitCategory.STORAGE_MB: usage.storage_bytes // (1024 ** 2),  # Convert bytes to MB
        }

        current = usage_map.get(category, 0)
        limit_value_raw = limits.get(category.value, -1)
        limit_value = -1
        try:
            if limit_value_raw is None or isinstance(limit_value_raw, bool):
                raise TypeError("invalid limit value")
            limit_value = int(limit_value_raw)
        except _BILLING_ENFORCEMENT_COERCE_EXCEPTIONS:
            logger.warning(
                f"Invalid limit value for {category.value} (org_id={org_id}): "
                f"{limit_value_raw!r}; treating as unlimited"
            )
            limit_value = -1

        if limit_value < -1:
            limit_value = -1

        # Unlimited (-1)
        if limit_value == -1:
            return LimitCheckResult(
                category=category.value,
                action=EnforcementAction.ALLOW,
                current=current,
                limit=limit_value,
                percent_used=0,
                unlimited=True,
            )

        # Calculate after operation
        after_operation = current + requested_units
        percent_used = (after_operation / limit_value * 100) if limit_value > 0 else 100

        # Determine base action
        if after_operation > limit_value:
            # Would exceed limit
            action = EnforcementAction.SOFT_BLOCK
            message = f"Limit exceeded for {category.value}. Current: {current}, Limit: {limit_value}"
        elif percent_used >= self.soft_limit_percent:
            # Approaching limit - warn
            action = EnforcementAction.WARN
            message = f"Approaching limit for {category.value}: {percent_used:.0f}% used"
        else:
            # Well within limits
            action = EnforcementAction.ALLOW
            message = None

        # Apply overage policy to potentially upgrade the enforcement action
        try:
            overage_policy = self._get_overage_policy()
            if overage_policy is None:
                raise RuntimeError("overage policy unavailable")
            overage_result = overage_policy.evaluate(percent_used)

            if overage_result["blocked"]:
                action = EnforcementAction.HARD_BLOCK
                message = (
                    f"Hard blocked by overage policy for {category.value}: "
                    f"{percent_used:.0f}% used (mode={overage_result['mode']})"
                )
            elif overage_result["degraded"]:
                action = EnforcementAction.SOFT_BLOCK
                message = (
                    f"Degraded by overage policy for {category.value}: "
                    f"{percent_used:.0f}% used (mode={overage_result['mode']})"
                )
            elif overage_result["notify"] and action == EnforcementAction.ALLOW:
                action = EnforcementAction.WARN
                message = (
                    f"Overage notification for {category.value}: "
                    f"{percent_used:.0f}% used"
                )
        except _BILLING_ENFORCEMENT_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"Overage policy evaluation skipped: {exc}")

        return LimitCheckResult(
            category=category.value,
            action=action,
            current=current,
            limit=limit_value,
            percent_used=percent_used,
            message=message,
        )

    async def check_feature_access(
        self,
        org_id: int,
        feature: str,
    ) -> bool:
        """
        Check if an organization has access to a specific feature.

        Args:
            org_id: Organization ID
            feature: Feature name (e.g., "advanced_analytics", "sso_enabled")

        Returns:
            True if feature is enabled, False otherwise
        """
        limits = await self.get_org_limits(org_id)
        return bool(limits.get(feature, False))

    def invalidate_cache(self, org_id: int | None = None) -> None:
        """Invalidate usage/limits cache for an org or all orgs."""
        if org_id is not None:
            self._usage_cache.pop(org_id, None)
            self._limits_cache.pop(org_id, None)
        else:
            self._usage_cache.clear()
            self._limits_cache.clear()

    def apply_usage_delta(self, org_id: int, category: LimitCategory, units: int) -> bool:
        """
        Apply an in-memory usage delta for an organization.

        This helper updates the cached UsageSummary for the given org_id so
        that subsequent limit checks within the cache TTL see the most recent
        usage without needing to hit the database again.

        Persistent usage remains the responsibility of the existing logging
        flows (usage_log, llm_usage_log, audio minutes ledger, etc.).
        """
        try:
            delta = int(units)
        except _BILLING_ENFORCEMENT_COERCE_EXCEPTIONS:
            return False

        if delta <= 0:
            return False

        cached = self._usage_cache.get(org_id)
        if not cached:
            return False

        summary, cached_at = cached
        if not isinstance(summary, UsageSummary):
            return False

        updated = True
        if category == LimitCategory.API_CALLS_DAY:
            summary.api_calls_today += delta
        elif category == LimitCategory.LLM_TOKENS_MONTH:
            summary.llm_tokens_month += delta
        elif category == LimitCategory.TEAM_MEMBERS:
            summary.team_members += delta
        elif category == LimitCategory.RAG_QUERIES_DAY:
            summary.rag_queries_today += delta
        elif category == LimitCategory.CONCURRENT_JOBS:
            summary.concurrent_jobs += delta
        elif category == LimitCategory.TRANSCRIPTION_MINUTES_MONTH:
            summary.transcription_minutes_month += delta
        elif category == LimitCategory.STORAGE_MB:
            summary.storage_bytes += delta * (1024 ** 2)
        else:
            updated = False

        if not updated:
            return False

        self._usage_cache[org_id] = (summary, cached_at)
        return True


# Singleton instance with thread-safe initialization
_billing_enforcer: BillingEnforcer | None = None
_billing_enforcer_lock = threading.Lock()


def get_billing_enforcer() -> BillingEnforcer:
    """Get or create the billing enforcer singleton (thread-safe)."""
    global _billing_enforcer
    if _billing_enforcer is None:
        with _billing_enforcer_lock:
            # Double-check pattern for thread safety
            if _billing_enforcer is None:
                _billing_enforcer = BillingEnforcer()
    return _billing_enforcer


def reset_billing_enforcer() -> None:
    """Reset the billing enforcer singleton (primarily for tests)."""
    global _billing_enforcer
    _billing_enforcer = None


# =============================================================================
# Resource Governor Integration
# =============================================================================

async def create_billing_rg_request(
    *,
    org_id: int,
    category: LimitCategory,
    units: int = 1,
    endpoint: str | None = None,
) -> RGRequest:
    """
    Create a Resource Governor request for billing enforcement.

    This allows billing limits to be enforced through the RG system.
    """
    try:
        from tldw_Server_API.app.core.Resource_Governance.governor import RGRequest
    except ImportError:
        raise RuntimeError("Resource Governor not available") from None

    return RGRequest(
        entity=f"org:{org_id}",
        categories={
            category.value: {"units": units},
        },
        tags={
            "org_id": str(org_id),
            "endpoint": endpoint or "unknown",
            "policy_id": f"billing.{category.value}",
        },
    )


async def check_billing_with_rg(
    org_id: int,
    category: LimitCategory,
    units: int = 1,
) -> bool:
    """
    Check billing limits using the Resource Governor if available.

    Falls back to direct enforcement if RG is not available.
    """
    try:
        importlib.import_module("tldw_Server_API.app.core.Resource_Governance.governor")

        # Try to get RG from app state (if we're in a request context)
        # Otherwise use a local instance
        enforcer = get_billing_enforcer()
        result = await enforcer.check_limit(org_id, category, requested_units=units)
        return not result.should_block

    except ImportError:
        # RG not available, use direct enforcement
        enforcer = get_billing_enforcer()
        result = await enforcer.check_limit(org_id, category, requested_units=units)
        return not result.should_block
    except _BILLING_ENFORCEMENT_NONCRITICAL_EXCEPTIONS:
        fail_closed = BillingEnforcer._fail_closed_on_data_error()
        if fail_closed:
            logger.error("Billing RG check failed, denying (fail-closed).")
            return False
        logger.warning("Billing RG check failed, allowing (fail-open).")
        return True


# Utility helpers

# Alias the global billing feature flag so callers importing from this module
# or from the Stripe client see consistent behavior.
billing_enabled = is_billing_enabled


def enforcement_enabled() -> bool:
    """Check if limit enforcement is enabled (can be separate from billing)."""
    # Can have enforcement without Stripe billing (e.g., for usage caps)
    return os.environ.get("LIMIT_ENFORCEMENT_ENABLED", "true").lower() == "true"
