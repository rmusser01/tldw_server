"""
Per-user rate limiting for Evaluations module.

**Phase 2 Deprecation Notice**:
- When ``RG_ENABLED=true``, evaluations/minute and evaluations/tokens daily
  caps are enforced by the ResourceGovernor via ``_maybe_enforce_with_rg_evaluations``.
- Cost caps (``max_cost_per_day``, ``max_cost_per_month``) remain **legacy-only**
  and are enforced regardless of RG status.
- When RG is disabled or unavailable, minute/daily eval/token checks are
  skipped (fail-open with deprecation warning). Only cost caps are enforced.
- CUSTOM tier bypasses RG; only cost caps are enforced with a deprecation warning.
- This shim will be removed in a future release.
"""

import asyncio
import contextlib
import json
import os
import sqlite3
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Evaluations.config_manager import (
    get_rate_limit_config,
)
from tldw_Server_API.app.core.Evaluations.identity import canonical_evaluations_user_scope
from tldw_Server_API.app.core.testing import is_test_mode

# Narrowed exception tuple for BLE001 fixes
_USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
    sqlite3.Error,
    json.JSONDecodeError,
)

# Optional Resource Governor integration (gated by global RG_ENABLED/config)
try:  # pragma: no cover - RG is optional
    from tldw_Server_API.app.core.config import rg_enabled  # type: ignore
    from tldw_Server_API.app.core.Resource_Governance import (  # type: ignore
        MemoryResourceGovernor,
        RedisResourceGovernor,
        RGRequest,
    )
    from tldw_Server_API.app.core.Resource_Governance.policy_loader import (  # type: ignore
        PolicyLoader,
        PolicyReloadConfig,
        default_policy_loader,
    )
except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:  # pragma: no cover - safe fallback when RG not installed
    MemoryResourceGovernor = None  # type: ignore
    RedisResourceGovernor = None  # type: ignore
    RGRequest = None  # type: ignore
    PolicyLoader = None  # type: ignore
    PolicyReloadConfig = None  # type: ignore
    default_policy_loader = None  # type: ignore
    rg_enabled = None  # type: ignore

# Optional generic daily ledger integration (v1.1 shadow + enforcement)
try:  # pragma: no cover - ledger is optional during upgrades/tests
    from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import (  # type: ignore
        LedgerEntry,
        ResourceDailyLedger,
    )
except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:  # pragma: no cover - safe fallback
    LedgerEntry = None  # type: ignore
    ResourceDailyLedger = None  # type: ignore

# Import connection pool
# from tldw_Server_API.app.core.Evaluations.connection_pool import get_connection  # unused

_EVALS_DEPRECATION_WARNED = False


def _emit_evals_legacy_deprecation(context: str) -> None:
    global _EVALS_DEPRECATION_WARNED
    if _EVALS_DEPRECATION_WARNED:
        return
    _EVALS_DEPRECATION_WARNED = True
    msg = (
        "Evaluations legacy rate limiter is deprecated (Phase 2). "
        f"Context: {context}. Enable RG_ENABLED=true for enforcement. "
        "This shim will be removed in a future release."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
    logger.warning(msg)


class UserTier(Enum):
    """User subscription tiers."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


@dataclass
class RateLimitConfig:
    """Rate limit configuration for a user tier."""
    tier: UserTier
    evaluations_per_minute: int
    batch_evaluations_per_minute: int
    evaluations_per_day: int
    total_tokens_per_day: int
    burst_size: int
    max_cost_per_day: float
    max_cost_per_month: float

    @classmethod
    def for_tier(cls, tier: UserTier) -> "RateLimitConfig":
        """Get configuration for a tier from external config."""
        # Try to get configuration from external config first
        tier_config = get_rate_limit_config(tier.value)

        if tier_config:
            return cls(
                tier=tier,
                evaluations_per_minute=tier_config.evaluations_per_minute,
                batch_evaluations_per_minute=tier_config.batch_evaluations_per_minute,
                evaluations_per_day=tier_config.evaluations_per_day,
                total_tokens_per_day=tier_config.total_tokens_per_day,
                burst_size=tier_config.burst_size,
                max_cost_per_day=tier_config.max_cost_per_day,
                max_cost_per_month=tier_config.max_cost_per_month
            )

        # Fallback to hardcoded defaults if external config not available
        fallback_configs = {
            UserTier.FREE: cls(
                tier=UserTier.FREE,
                evaluations_per_minute=10,
                batch_evaluations_per_minute=2,
                evaluations_per_day=100,
                total_tokens_per_day=100_000,
                burst_size=5,
                max_cost_per_day=1.0,
                max_cost_per_month=10.0
            ),
            UserTier.BASIC: cls(
                tier=UserTier.BASIC,
                evaluations_per_minute=30,
                batch_evaluations_per_minute=5,
                evaluations_per_day=1000,
                total_tokens_per_day=1_000_000,
                burst_size=10,
                max_cost_per_day=10.0,
                max_cost_per_month=100.0
            ),
            UserTier.PREMIUM: cls(
                tier=UserTier.PREMIUM,
                evaluations_per_minute=100,
                batch_evaluations_per_minute=20,
                evaluations_per_day=10000,
                total_tokens_per_day=10_000_000,
                burst_size=25,
                max_cost_per_day=100.0,
                max_cost_per_month=1000.0
            ),
            UserTier.ENTERPRISE: cls(
                tier=UserTier.ENTERPRISE,
                evaluations_per_minute=500,
                batch_evaluations_per_minute=100,
                evaluations_per_day=100000,
                total_tokens_per_day=100_000_000,
                burst_size=100,
                max_cost_per_day=1000.0,
                max_cost_per_month=10000.0
            ),
            UserTier.CUSTOM: cls(
                tier=UserTier.CUSTOM,
                evaluations_per_minute=10,
                batch_evaluations_per_minute=2,
                evaluations_per_day=100,
                total_tokens_per_day=100_000,
                burst_size=5,
                max_cost_per_day=1.0,
                max_cost_per_month=10.0
            )
        }

        logger.warning(f"Using fallback configuration for tier {tier.value}")
        return fallback_configs.get(tier, fallback_configs[UserTier.FREE])


class UserRateLimiter:
    """Per-user rate limiter with tier-based limits."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize user rate limiter.

        Args:
            db_path: Path to rate limiting database
        """
        if db_path is None:
            # Default to canonical per-user evaluations DB (single-user ID in legacy contexts)
            db_path = DatabasePaths.get_evaluations_db_path(DatabasePaths.get_single_user_id())

        self.db_path = str(db_path)
        self._init_database()

        # In-memory cache for rate limit data (with TTL)
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_ttl = 60  # seconds

    def _init_database(self):
        """Initialize rate limiting tables."""
        # Register explicit adapters to avoid deprecated defaults on Python 3.12+
        with contextlib.suppress(_USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS):
            sqlite3.register_adapter(datetime, lambda d: d.isoformat(sep=" "))
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES) as conn:
            # User rate limits table (created in migration)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_rate_limits (
                    user_id TEXT PRIMARY KEY,
                    tier TEXT NOT NULL DEFAULT 'free',
                    evaluations_per_minute INTEGER DEFAULT 10,
                    batch_evaluations_per_minute INTEGER DEFAULT 2,
                    evaluations_per_day INTEGER DEFAULT 100,
                    total_tokens_per_day INTEGER DEFAULT 100000,
                    burst_size INTEGER DEFAULT 5,
                    max_cost_per_day REAL DEFAULT 10.0,
                    max_cost_per_month REAL DEFAULT 100.0,
                    custom_limits TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    expires_at TEXT,
                    notes TEXT
                )
            """)

            # Rate limit tracking table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rate_limit_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tokens_used INTEGER DEFAULT 0,
                    cost REAL DEFAULT 0.0
                )
            """)

            # Create indexes separately
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tracking_user ON rate_limit_tracking(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tracking_time ON rate_limit_tracking(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tracking_endpoint ON rate_limit_tracking(endpoint)")

            # Daily usage summary
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_usage (
                    user_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    total_evaluations INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0.0,
                    PRIMARY KEY (user_id, date)
                )
            """)

            conn.commit()

    async def check_rate_limit(
        self,
        user_id: str,
        endpoint: str,
        is_batch: bool = False,
        tokens_requested: int = 0,
        estimated_cost: float = 0.0
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check if user can make request based on their tier limits.

        Args:
            user_id: User identifier
            endpoint: API endpoint being accessed
            is_batch: Whether this is a batch evaluation
            tokens_requested: Estimated tokens for request
            estimated_cost: Estimated cost for request

        Returns:
            Tuple of (is_allowed, metadata)
            - is_allowed: Whether request is allowed
            - metadata: Rate limit information and headers
        """
        config = await self._get_user_config(user_id)

        if _rg_evaluations_enabled() and config.tier != UserTier.CUSTOM:
            policy_id = _policy_id_for_config(config, is_batch=is_batch)
            rg_decision = await _maybe_enforce_with_rg_evaluations(
                user_id=user_id,
                endpoint=endpoint,
                is_batch=is_batch,
                tokens_requested=tokens_requested,
                estimated_cost=estimated_cost,
                policy_id=policy_id,
            )
            if rg_decision is not None:
                if not rg_decision["allowed"]:
                    retry_after = rg_decision.get("retry_after") or 1
                    return False, {
                        "error": "Rate limit exceeded (ResourceGovernor)",
                        "policy_id": rg_decision.get("policy_id", policy_id),
                        "retry_after": retry_after,
                        "rate_limit_source": "resource_governor",
                    }

                # Enforce cost caps locally (RG does not handle cost limits).
                cost_ok, cost_meta = await self._check_cost_limits(user_id, estimated_cost, config)
                if not cost_ok:
                    return False, cost_meta

                # Record the request (usage + ledger shadow writes).
                try:
                    await self._record_request(user_id, endpoint, tokens_requested, estimated_cost)
                except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:
                    # Best-effort only; never block allow path.
                    pass

                metadata: dict[str, Any] = {
                    "policy_id": rg_decision.get("policy_id", policy_id),
                    "rate_limit_source": "resource_governor",
                }
                return True, metadata

            _log_rg_evals_fallback("rg_decision_unavailable")
            _emit_evals_legacy_deprecation("rg_decision_unavailable")
            # Fail-open for eval/token caps; still enforce cost.
            cost_ok, cost_meta = await self._check_cost_limits(user_id, estimated_cost, config)
            if not cost_ok:
                return False, cost_meta
            try:
                await self._record_request(user_id, endpoint, tokens_requested, estimated_cost)
            except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:
                pass
            return True, {
                "policy_id": policy_id,
                "rate_limit_source": "resource_governor",
            }

        # RG disabled or CUSTOM tier → skip minute/daily checks (Phase 2),
        # enforce cost caps only.
        _emit_evals_legacy_deprecation("rg_disabled_or_custom_tier")
        cost_ok, cost_meta = await self._check_cost_limits(user_id, estimated_cost, config)
        if not cost_ok:
            return False, cost_meta

        await self._record_request(user_id, endpoint, tokens_requested, estimated_cost)
        return True, {
            "rate_limit_source": "legacy_cost_only",
            "tier": config.tier.value,
        }

    async def _get_user_config(self, user_id: str) -> RateLimitConfig:
        """Get user's rate limit configuration."""
        # Check cache first
        cache_key = f"config_{user_id}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() - cached["timestamp"] < self._cache_ttl:
                return cached["config"]

        # Query database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT tier, evaluations_per_minute, batch_evaluations_per_minute,
                       evaluations_per_day, total_tokens_per_day, burst_size,
                       max_cost_per_day, max_cost_per_month, expires_at
                FROM user_rate_limits
                WHERE user_id = ?
            """, (user_id,))

            row = cursor.fetchone()

            if row:
                # Check if custom limits have expired (tolerate naive or aware timestamps)
                expired = False
                if row[8]:
                    try:
                        _exp_raw = row[8]
                        exp_dt = datetime.fromisoformat(_exp_raw)
                        if exp_dt.tzinfo is None:
                            # Compare naive timestamps in UTC
                            expired = exp_dt < datetime.utcnow()
                        else:
                            expired = exp_dt < datetime.now(timezone.utc)
                    except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:
                        expired = False
                if expired:
                    # Reset to default tier
                    tier = UserTier.FREE
                    config = RateLimitConfig.for_tier(tier)

                    # Update database
                    cursor.execute("""
                        UPDATE user_rate_limits
                        SET tier = ?, expires_at = NULL
                        WHERE user_id = ?
                    """, (tier.value, user_id))
                    conn.commit()
                else:
                    # Use stored configuration
                    config = RateLimitConfig(
                        tier=UserTier(row[0]),
                        evaluations_per_minute=row[1],
                        batch_evaluations_per_minute=row[2],
                        evaluations_per_day=row[3],
                        total_tokens_per_day=row[4],
                        burst_size=row[5],
                        max_cost_per_day=row[6],
                        max_cost_per_month=row[7]
                    )
            else:
                # Create default configuration for new user
                config = RateLimitConfig.for_tier(UserTier.FREE)

                cursor.execute("""
                    INSERT INTO user_rate_limits (
                        user_id, tier, evaluations_per_minute, batch_evaluations_per_minute,
                        evaluations_per_day, total_tokens_per_day, burst_size,
                        max_cost_per_day, max_cost_per_month
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id, config.tier.value, config.evaluations_per_minute,
                    config.batch_evaluations_per_minute, config.evaluations_per_day,
                    config.total_tokens_per_day, config.burst_size,
                    config.max_cost_per_day, config.max_cost_per_month
                ))
                conn.commit()

        # Cache the configuration
        self._cache[cache_key] = {
            "config": config,
            "timestamp": time.time()
        }

        return config

    async def _get_daily_ledger(self, user_id: str) -> Optional["ResourceDailyLedger"]:
        """
        Lazily initialize the shared ResourceDailyLedger and backfill today's
        legacy daily_usage totals for this user once per process.

        The ledger is used as the canonical store for daily evaluations/tokens
        caps when available. Failures are best-effort and never block callers.
        """
        global _evals_daily_ledger
        if ResourceDailyLedger is None or LedgerEntry is None:
            return None
        if _evals_daily_ledger is None:
            async with _evals_daily_ledger_lock:
                if _evals_daily_ledger is None:
                    try:
                        ledger = ResourceDailyLedger()  # type: ignore[call-arg]
                        await ledger.initialize()
                        _evals_daily_ledger = ledger
                    except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover - defensive
                        logger.debug(f"Evaluations: ResourceDailyLedger init failed; using legacy daily_usage: {exc}")
                        _evals_daily_ledger = None
                        return None
        ledger = _evals_daily_ledger

        # Best-effort one-time backfill for today's legacy totals.
        try:
            day = datetime.now(timezone.utc).date().isoformat()
            if user_id not in _evals_legacy_backfill_done:
                await self._backfill_legacy_daily_usage_to_ledger(ledger, user_id=user_id, day_utc=day)
                _evals_legacy_backfill_done.add(user_id)
        except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:
            pass
        return ledger

    async def _backfill_legacy_daily_usage_to_ledger(
        self,
        ledger: "ResourceDailyLedger",
        *,
        user_id: str,
        day_utc: str,
    ) -> None:
        """
        Mirror today's legacy daily_usage totals into ResourceDailyLedger.

        This preserves in-progress daily caps after upgrades. The backfill is
        idempotent via deterministic op_id values.
        """
        if LedgerEntry is None:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT total_evaluations, total_tokens FROM daily_usage WHERE user_id = ? AND date = ?",
                    (user_id, str(day_utc)),
                )
                row = cursor.fetchone()
            if not row:
                return
            total_evaluations = int(row[0] or 0)
            total_tokens = int(row[1] or 0)
            ts = datetime.now(timezone.utc)
            if total_evaluations > 0:
                entry = LedgerEntry(  # type: ignore[call-arg]
                    entity_scope="user",
                    entity_value=str(user_id),
                    category="evaluations",
                    units=total_evaluations,
                    op_id=f"evals-legacy:{user_id}:{day_utc}",
                    occurred_at=ts,
                )
                await ledger.add(entry)
            if total_tokens > 0:
                entry = LedgerEntry(  # type: ignore[call-arg]
                    entity_scope="user",
                    entity_value=str(user_id),
                    category="tokens",
                    units=total_tokens,
                    op_id=f"evals-legacy-tokens:{user_id}:{day_utc}",
                    occurred_at=ts,
                )
                await ledger.add(entry)
        except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover - defensive
            logger.debug(f"Evaluations: legacy daily_usage backfill skipped: {exc}")

    async def _check_cost_limits(
        self,
        user_id: str,
        estimated_cost: float,
        config: RateLimitConfig,
    ) -> tuple[bool, dict[str, Any]]:
        """Check daily cost limits only (RG handles evaluations/tokens)."""
        try:
            cost = float(estimated_cost or 0.0)
        except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:
            cost = 0.0
        if cost <= 0:
            return True, {}

        today = datetime.now(timezone.utc).date()
        day_str = str(today)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT total_cost FROM daily_usage WHERE user_id = ? AND date = ?",
                (user_id, day_str),
            )
            row = cursor.fetchone()
            total_cost = float(row[0] or 0.0) if row else 0.0

        if total_cost + cost > float(config.max_cost_per_day):
            reset_at = datetime.combine(today + timedelta(days=1), datetime.min.time())
            retry_after = max(1, int((reset_at - datetime.now(timezone.utc)).total_seconds()))
            return False, {
                "error": "Daily cost limit exceeded",
                "limit": config.max_cost_per_day,
                "used": total_cost,
                "requested": cost,
                "retry_after": retry_after,
                "resets_at": reset_at.isoformat(),
            }

        return True, {
            "cost_remaining": float(config.max_cost_per_day) - total_cost - cost,
        }

    async def _record_request(
        self,
        user_id: str,
        endpoint: str,
        tokens_used: int,
        cost: float
    ):
        """Record a request for tracking."""
        now = datetime.now(timezone.utc)
        today = now.date()

        with sqlite3.connect(self.db_path) as conn:
            # Record in tracking table
            conn.execute(
                "INSERT INTO rate_limit_tracking (user_id, endpoint, timestamp, tokens_used, cost) VALUES (?, ?, ?, ?, ?)",
                (user_id, endpoint, now.isoformat(), tokens_used, cost)
            )

            # Update daily usage
            conn.execute("""
                INSERT INTO daily_usage (user_id, date, total_evaluations, total_tokens, total_cost)
                VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(user_id, date) DO UPDATE SET
                    total_evaluations = total_evaluations + 1,
                    total_tokens = total_tokens + ?,
                    total_cost = total_cost + ?
            """, (user_id, str(today), tokens_used, cost, tokens_used, cost))

            conn.commit()

        # Shadow-write daily usage into the shared ResourceDailyLedger so RG can
        # enforce tokens/evaluations caps cross-module in v1.1.
        try:
            ledger = await self._get_daily_ledger(user_id)
            if ledger is not None and LedgerEntry is not None:
                base_oid = f"evals:{user_id}:{endpoint}:{int(now.timestamp())}:{time.time_ns()}"
                try:
                    entry_eval = LedgerEntry(  # type: ignore[call-arg]
                        entity_scope="user",
                        entity_value=str(user_id),
                        category="evaluations",
                        units=1,
                        op_id=f"{base_oid}:eval",
                        occurred_at=now,
                    )
                    await ledger.add(entry_eval)
                except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:
                    pass
                if int(tokens_used or 0) > 0:
                    try:
                        entry_tokens = LedgerEntry(  # type: ignore[call-arg]
                            entity_scope="user",
                            entity_value=str(user_id),
                            category="tokens",
                            units=int(tokens_used),
                            op_id=f"{base_oid}:tokens",
                            occurred_at=now,
                        )
                        await ledger.add(entry_tokens)
                    except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:
                        pass
        except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover - defensive
            logger.debug(f"Evaluations: ResourceDailyLedger shadow write failed; ignoring: {exc}")

    def _generate_rate_limit_headers(
        self,
        user_id: str,
        config: RateLimitConfig,
        minute_metadata: dict[str, Any],
        daily_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate rate limit headers for response."""
        return {
            "headers": {
                "X-RateLimit-Tier": config.tier.value,
                "X-RateLimit-Limit": str(minute_metadata.get("limit", 0)),
                "X-RateLimit-Remaining": str(minute_metadata.get("requests_remaining", 0)),
                # Align reset with computed remaining seconds in current window
                "X-RateLimit-Reset": str(int(time.time()) + int(minute_metadata.get("reset_seconds", 60))),
                "X-RateLimit-Daily-Limit": str(config.evaluations_per_day),
                "X-RateLimit-Daily-Remaining": str(daily_metadata.get("evaluations_remaining", 0)),
                "X-RateLimit-Tokens-Remaining": str(daily_metadata.get("tokens_remaining", 0)),
                "X-RateLimit-Cost-Remaining": f"{daily_metadata.get('cost_remaining', 0):.2f}"
            },
            "tier": config.tier.value,
            "limits": {
                "per_minute": minute_metadata,
                "daily": daily_metadata
            }
        }

    async def upgrade_user_tier(
        self,
        user_id: str,
        new_tier: UserTier,
        expires_at: Optional[datetime] = None,
        custom_limits: Optional[dict[str, Any]] = None
    ) -> bool:
        """
        Upgrade a user's tier.

        Args:
            user_id: User identifier
            new_tier: New tier to assign
            expires_at: Optional expiration for temporary upgrades
            custom_limits: Optional custom limit overrides

        Returns:
            True if upgrade successful
        """
        try:
            config = RateLimitConfig.for_tier(new_tier)

            # Apply custom limits if provided
            if custom_limits:
                # If moving to CUSTOM with partial overrides, default to no burst unless specified
                if new_tier == UserTier.CUSTOM and 'burst_size' not in custom_limits:
                    config.burst_size = 0
                for key, value in custom_limits.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE user_rate_limits
                    SET tier = ?, evaluations_per_minute = ?, batch_evaluations_per_minute = ?,
                        evaluations_per_day = ?, total_tokens_per_day = ?, burst_size = ?,
                        max_cost_per_day = ?, max_cost_per_month = ?,
                        expires_at = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (
                    new_tier.value, config.evaluations_per_minute, config.batch_evaluations_per_minute,
                    config.evaluations_per_day, config.total_tokens_per_day, config.burst_size,
                    config.max_cost_per_day, config.max_cost_per_month,
                    expires_at, user_id
                ))

                if cursor.rowcount == 0:
                    # User doesn't exist, create new entry
                    cursor.execute("""
                        INSERT INTO user_rate_limits (
                            user_id, tier, evaluations_per_minute, batch_evaluations_per_minute,
                            evaluations_per_day, total_tokens_per_day, burst_size,
                            max_cost_per_day, max_cost_per_month, expires_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        user_id, new_tier.value, config.evaluations_per_minute,
                        config.batch_evaluations_per_minute, config.evaluations_per_day,
                        config.total_tokens_per_day, config.burst_size,
                        config.max_cost_per_day, config.max_cost_per_month,
                        expires_at
                    ))

                conn.commit()

            # Clear cache
            cache_key = f"config_{user_id}"
            if cache_key in self._cache:
                del self._cache[cache_key]

            logger.info(f"Upgraded user {user_id} to tier {new_tier.value}")
            return True

        except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to upgrade user tier: {e}")
            return False

    async def get_usage_summary(self, user_id: str) -> dict[str, Any]:
        """
        Get usage summary for a user.

        Args:
            user_id: User identifier

        Returns:
            Usage summary with current limits and consumption
        """
        config = await self._get_user_config(user_id)
        today = datetime.now(timezone.utc).date()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get today's usage
            cursor.execute("""
                SELECT total_evaluations, total_tokens, total_cost
                FROM daily_usage
                WHERE user_id = ? AND date = ?
            """, (user_id, str(today)))

            row = cursor.fetchone()

            if row:
                total_evaluations, total_tokens, total_cost = row
            else:
                total_evaluations = total_tokens = total_cost = 0

            # Get monthly cost
            month_start = today.replace(day=1)
            cursor.execute("""
                SELECT SUM(total_cost)
                FROM daily_usage
                WHERE user_id = ? AND date >= ?
            """, (user_id, str(month_start)))

            monthly_cost = cursor.fetchone()[0] or 0.0

        return {
            "user_id": user_id,
            "tier": config.tier.value,
            "limits": {
                "per_minute": {
                    "evaluations": config.evaluations_per_minute,
                    "batch_evaluations": config.batch_evaluations_per_minute,
                    "burst_size": config.burst_size
                },
                "daily": {
                    "evaluations": config.evaluations_per_day,
                    "tokens": config.total_tokens_per_day,
                    "cost": config.max_cost_per_day
                },
                "monthly": {
                    "cost": config.max_cost_per_month
                }
            },
            "usage": {
                "today": {
                    "evaluations": total_evaluations,
                    "tokens": total_tokens,
                    "cost": total_cost
                },
                "month": {
                    "cost": monthly_cost
                }
            },
            "remaining": {
                "daily_evaluations": config.evaluations_per_day - total_evaluations,
                "daily_tokens": config.total_tokens_per_day - total_tokens,
                "daily_cost": config.max_cost_per_day - total_cost,
                "monthly_cost": config.max_cost_per_month - monthly_cost
            }
        }

    async def record_actual_usage(self, user_id: str, endpoint: str, tokens_used: int, cost: float = 0.0) -> None:
        """Record actual usage after a request completes (if provider returns usage).

        Safe no-op on failure.
        """
        try:
            await self._record_request(user_id, endpoint, max(0, int(tokens_used or 0)), float(cost or 0.0))
        except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:
            # Non-fatal; logging here could be noisy for hot paths
            pass


# Global instance
# Global default (legacy) instance
user_rate_limiter = UserRateLimiter()

# Per-user instances cache
_user_rate_limiter_instances: dict = {}
_user_rate_limiter_lock: Optional[threading.Lock] = None


def get_user_rate_limiter_for_user(user_id: str | int) -> UserRateLimiter:
    """Return a UserRateLimiter bound to the user's evaluations DB."""
    # In test environments, fall back to legacy global instance for compatibility with existing tests/mocks
    try:
        import os as _os
        if is_test_mode() or "PYTEST_CURRENT_TEST" in _os.environ:
            return user_rate_limiter
    except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:
        pass
    global _user_rate_limiter_lock
    if _user_rate_limiter_lock is None:
        _user_rate_limiter_lock = threading.Lock()
    with _user_rate_limiter_lock:
        uid_key = canonical_evaluations_user_scope(user_id)
        inst = _user_rate_limiter_instances.get(uid_key)
        if inst is not None:
            return inst
        legacy_numeric_key: int | None = None
        try:
            legacy_numeric_key = int(uid_key)
        except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:
            legacy_numeric_key = None
        # Temporary migration path: older in-process callers cached limiters under
        # numeric user ids before canonical string scopes landed. Remove this branch
        # after all callers and long-lived workers have restarted on string scopes.
        if legacy_numeric_key is not None:
            inst = _user_rate_limiter_instances.get(legacy_numeric_key)
            if inst is not None:
                _user_rate_limiter_instances[uid_key] = inst
                return inst
        db_path = str(DatabasePaths.get_evaluations_db_path(uid_key))
        inst = UserRateLimiter(db_path=db_path)
        _user_rate_limiter_instances[uid_key] = inst
        return inst


# --- Resource Governor plumbing (optional) ---------------------------------
_rg_evals_governor = None
_rg_evals_loader = None
_rg_evals_lock = asyncio.Lock()
_rg_evals_init_error: Optional[str] = None
_rg_evals_init_error_logged = False
_rg_evals_fallback_logged = False


def _rg_evals_context() -> dict[str, str]:
    policy_path = os.getenv(
        "RG_POLICY_PATH",
        "tldw_Server_API/Config_Files/resource_governor_policies.yaml",
    )
    try:
        policy_path_resolved = os.path.abspath(policy_path)
    except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:
        policy_path_resolved = policy_path
    return {
        "backend": os.getenv("RG_BACKEND", "memory"),
        "policy_path": policy_path,
        "policy_path_resolved": policy_path_resolved,
        "policy_store": os.getenv("RG_POLICY_STORE", ""),
        "policy_reload_enabled": os.getenv("RG_POLICY_RELOAD_ENABLED", ""),
        "policy_reload_interval": os.getenv("RG_POLICY_RELOAD_INTERVAL_SEC", ""),
        "cwd": os.getcwd(),
    }


def _policy_id_for_config(config: RateLimitConfig, *, is_batch: bool) -> str:
    override = os.getenv("RG_EVALUATIONS_POLICY_ID")
    if override:
        return override
    tier = getattr(config.tier, "value", str(config.tier))
    suffix = "batch" if is_batch else None
    if tier:
        if suffix:
            return f"evals.{tier}.batch"
        return f"evals.{tier}"
    return "evals.default"


def _log_rg_evals_init_failure(exc: Exception) -> None:
    global _rg_evals_init_error, _rg_evals_init_error_logged
    _rg_evals_init_error = repr(exc)
    if _rg_evals_init_error_logged:
        return
    _rg_evals_init_error_logged = True
    ctx = _rg_evals_context()
    logger.exception(
        "Evaluations ResourceGovernor init failed; using diagnostics-only compatibility shim (no enforcement). "
        "backend={backend} policy_path={policy_path} policy_path_resolved={policy_path_resolved} "
        "policy_store={policy_store} reload_enabled={policy_reload_enabled} "
        "reload_interval={policy_reload_interval} cwd={cwd}",
        **ctx,
    )


def _log_rg_evals_fallback(reason: str) -> None:
    global _rg_evals_fallback_logged
    if _rg_evals_fallback_logged:
        return
    _rg_evals_fallback_logged = True
    ctx = _rg_evals_context()
    logger.error(
        "Evaluations ResourceGovernor unavailable; using diagnostics-only compatibility shim (no enforcement). "
        "reason={} init_error={} backend={backend} policy_path={policy_path} "
        "policy_path_resolved={policy_path_resolved} policy_store={policy_store} "
        "reload_enabled={policy_reload_enabled} reload_interval={policy_reload_interval} cwd={cwd}",
        reason,
        _rg_evals_init_error,
        **ctx,
    )


# --- Generic daily ledger plumbing (optional) ------------------------------
_evals_daily_ledger: Optional["ResourceDailyLedger"] = None  # type: ignore[name-defined]
_evals_daily_ledger_lock = asyncio.Lock()
# Track per-user backfill attempts for today's legacy daily_usage
_evals_legacy_backfill_done: set[str] = set()


def _rg_evaluations_enabled() -> bool:
    """Return True when RG should gate Evaluations requests."""
    if rg_enabled is not None:
        try:
            return bool(rg_enabled(True))  # type: ignore[func-returns-value]
        except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:
            return False
    return False


async def _get_evaluations_rg_governor():
    """Lazily initialize a ResourceGovernor instance for Evaluations."""
    global _rg_evals_governor, _rg_evals_loader
    if not _rg_evaluations_enabled():
        return None
    if RGRequest is None or PolicyLoader is None:
        _log_rg_evals_fallback("rg_components_unavailable")
        return None
    if _rg_evals_governor is not None:
        return _rg_evals_governor
    async with _rg_evals_lock:
        if _rg_evals_governor is not None:
            return _rg_evals_governor
        try:
            loader = (
                default_policy_loader()
                if default_policy_loader
                else PolicyLoader(
                    os.getenv(
                        "RG_POLICY_PATH",
                        "tldw_Server_API/Config_Files/resource_governor_policies.yaml",
                    ),
                    PolicyReloadConfig(
                        enabled=True,
                        interval_sec=int(
                            os.getenv("RG_POLICY_RELOAD_INTERVAL_SEC", "10") or "10"
                        ),
                    ),
                )
            )
            await loader.load_once()
            _rg_evals_loader = loader
            backend = os.getenv("RG_BACKEND", "memory").lower()
            if backend == "redis" and RedisResourceGovernor is not None:
                gov = RedisResourceGovernor(policy_loader=loader)  # type: ignore[call-arg]
            else:
                gov = MemoryResourceGovernor(policy_loader=loader)  # type: ignore[call-arg]
            _rg_evals_governor = gov
            return gov
        except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover - optional path
            _log_rg_evals_init_failure(exc)
            return None


async def _maybe_enforce_with_rg_evaluations(
    user_id: str,
    endpoint: str,
    is_batch: bool,
    tokens_requested: int,
    estimated_cost: float,
    policy_id: str,
) -> Optional[dict[str, object]]:
    """
    Optionally enforce Evaluations request limits via ResourceGovernor.

    Returns a decision dict when RG is used, or None when RG is
    unavailable or disabled.
    """
    gov = await _get_evaluations_rg_governor()
    if gov is None:
        return None
    op_id = f"evals-{user_id}-{time.time_ns()}"
    try:
        categories: dict[str, dict[str, int]] = {
            "evaluations": {"units": 1},
        }
        try:
            tu = int(tokens_requested or 0)
        except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:
            tu = 0
        if tu > 0:
            categories["tokens"] = {"units": tu}
        # Cost/day caps remain legacy-only (and unified eval endpoints currently
        # pass estimated_cost=0.0). RG-first enforcement covers evaluations/tokens;
        # keep estimated_cost in tags for observability only.

        decision, handle = await gov.reserve(
            RGRequest(
                entity=f"user:{user_id}",
                categories=categories,
                tags={
                    "policy_id": policy_id,
                    "module": "evaluations",
                    "endpoint": endpoint,
                    "is_batch": str(bool(is_batch)).lower(),
                    "tokens_est": str(int(tu)),
                    "cost_est": str(float(estimated_cost or 0.0)),
                },
            ),
            op_id=op_id,
        )
        if decision.allowed:
            if handle:
                try:
                    await gov.commit(handle, None, op_id=op_id)
                except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:
                    logger.debug("Evaluations RG commit failed", exc_info=True)
            # Expose decision details for callers that want legacy-style headers.
            details = {}
            try:
                details = dict(getattr(decision, "details", {}) or {})
            except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:
                details = {}
            return {
                "allowed": True,
                "retry_after": None,
                "policy_id": policy_id,
                "details": details,
            }
        return {
            "allowed": False,
            "retry_after": decision.retry_after or 1,
            "policy_id": policy_id,
        }
    except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "Evaluations RG reserve failed: {}", exc
        )
        return None
