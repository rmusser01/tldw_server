"""
Audio usage quotas and tracking.

Provides per-user tiered limits for:
- daily transcription minutes
- concurrent streaming connections
- concurrent audio jobs (batch pipeline)
- per-request max file size

Backed by the AuthNZ database via DatabasePool for durable daily minute tracking.
In-process maps are used for concurrency caps (MVP, single-process safety).
"""

from __future__ import annotations

import asyncio
import configparser
import contextlib
import os
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.config import usage_quotas_enabled
from tldw_Server_API.app.core.exceptions import AudioQuotaStoreUnavailable
from tldw_Server_API.app.core.Usage.quota_checks import ledger_used_this_month, ledger_used_today
from tldw_Server_API.app.core.Usage.quota_resolver import user_quota

try:
    from tldw_Server_API.app.core.Metrics.metrics_manager import MetricDefinition, MetricType, get_metrics_registry
except ImportError:  # pragma: no cover
    get_metrics_registry = None  # type: ignore
    MetricDefinition = None  # type: ignore
    MetricType = None  # type: ignore

try:
    # Generic daily ledger (canonical store for daily minutes when available)
    from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import (
        LedgerEntry,
        ResourceDailyLedger,
    )
except ImportError:  # pragma: no cover
    ResourceDailyLedger = None  # type: ignore
    LedgerEntry = None  # type: ignore

_AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS = (
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    configparser.Error,
)


# Default tier limits (can be extended later via configuration or DB)
TIER_LIMITS = {
    "free": {
        "daily_minutes": 30.0,
        "concurrent_streams": 1,
        "concurrent_jobs": 1,
        "max_file_size_mb": 25,
    },
    "standard": {
        "daily_minutes": 300.0,
        "concurrent_streams": 3,
        "concurrent_jobs": 3,
        "max_file_size_mb": 100,
    },
    "premium": {
        "daily_minutes": None,  # unlimited
        "concurrent_streams": 10,
        "concurrent_jobs": 10,
        "max_file_size_mb": 500,
    },
}


_audio_minutes_consume_locks: dict[int, asyncio.Lock] = {}
_audio_minutes_consume_locks_lock = asyncio.Lock()


async def _get_audio_minutes_consume_lock(user_id: int) -> asyncio.Lock:
    """Return the per-user lock used by fallback minute consumption."""
    uid = int(user_id)
    async with _audio_minutes_consume_locks_lock:
        lock = _audio_minutes_consume_locks.get(uid)
        if lock is None:
            lock = asyncio.Lock()
            _audio_minutes_consume_locks[uid] = lock
        return lock


async def _cleanup_audio_minutes_consume_lock(user_id: int, expected_lock: asyncio.Lock) -> None:
    """Remove an idle fallback minute-consume lock from the per-user lock map."""
    uid = int(user_id)
    async with _audio_minutes_consume_locks_lock:
        lock = _audio_minutes_consume_locks.get(uid)
        if lock is expected_lock and not lock.locked():
            _audio_minutes_consume_locks.pop(uid, None)


@lru_cache(maxsize=1)
def _get_stream_ttl_seconds() -> int:
    """
    Determine the TTL (in seconds) to use for Redis stream counters.

    Checks for a value in this order: the AUDIO_STREAM_TTL_SECONDS environment variable, the
    Audio-Quota stream_ttl_seconds config setting, then a hard default of 120. The resulting
    value is clamped to the inclusive range 30-3600.

    Returns:
        int: TTL in seconds (clamped to 30-3600).
    """
    # 1) Environment variable override
    val_env = os.getenv("AUDIO_STREAM_TTL_SECONDS")
    if val_env:
        try:
            v = int(val_env)
            return max(30, min(3600, v))
        except (TypeError, ValueError):
            logger.debug("Audio stream TTL: invalid AUDIO_STREAM_TTL_SECONDS")
    # 2) Config default
    try:
        from tldw_Server_API.app.core.config import load_comprehensive_config  # lazy import
        cfg = load_comprehensive_config()
        if cfg and cfg.has_section('Audio-Quota'):
            try:
                v = int(cfg.get('Audio-Quota', 'stream_ttl_seconds', fallback='120'))
            except (TypeError, ValueError, configparser.Error):
                logger.debug("Audio stream TTL: invalid config value")
                v = 120
            return max(30, min(3600, v))
    except (OSError, RuntimeError, configparser.Error, TypeError, ValueError):
        logger.debug("Audio stream TTL: failed to load config")
    # 3) Hard default
    return 120


def clear_stream_ttl_cache() -> None:
    """Clear the cached TTL value so subsequent calls re-read configuration.

    Use this after reloading application configuration or changing
    AUDIO_STREAM_TTL_SECONDS at runtime.
    """
    try:
        _get_stream_ttl_seconds.cache_clear()  # type: ignore[attr-defined]
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        # If decoration is missing for any reason, ignore
        pass


def _metrics_set_gauge(name: str, value: float, labels: dict[str, str]) -> None:
    """
    Register and update a gauge metric under both underscore and dot-name variants and set its value with provided labels.

    Attempts to register and set the gauge for the canonical metric name (underscores) and a backward-compatible alias (dots). Any errors during registration or setting are suppressed so the call never raises. `labels` must map label names to their string values; `value` is converted to float before setting.
    """
    try:
        if not get_metrics_registry or not MetricDefinition or not MetricType:
            return
        reg = get_metrics_registry()
        # Canonical metric name: underscores
        canonical = name
        # Backward-compat alias with dots
        alias = name.replace('_', '.') if '_' in name else name
        for metric_name in (canonical, alias):
            with contextlib.suppress(_AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS):
                reg.register_metric(
                    MetricDefinition(
                        name=metric_name,
                        type=MetricType.GAUGE,
                        description=metric_name,
                        labels=list(labels.keys()),
                    )
                )
            reg.set_gauge(metric_name, float(value), labels)
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        pass


def _metrics_increment(name: str, labels: dict[str, str]) -> None:
    """
    Increment a counter metric in the metrics registry for a given metric name and label set.

    Attempts to register and increment two metric name variants: the provided name and a dot-separated alias (underscores replaced with dots). If a metrics registry is unavailable or any error occurs, the function does nothing and does not raise.

    Parameters:
        name (str): Base metric name to increment (e.g., "audio_jobs_active").
        labels (Dict[str, str]): Mapping of label names to values to attach to the metric.
    """
    try:
        if not get_metrics_registry or not MetricDefinition or not MetricType:
            return
        reg = get_metrics_registry()
        canonical = name
        alias = name.replace('_', '.') if '_' in name else name
        for metric_name in (canonical, alias):
            with contextlib.suppress(_AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS):
                reg.register_metric(
                    MetricDefinition(
                        name=metric_name,
                        type=MetricType.COUNTER,
                        description=metric_name,
                        labels=list(labels.keys()),
                    )
                )
            reg.increment(metric_name, 1, labels)
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        pass


async def _ensure_tables(pool: DatabasePool) -> None:
    """Ensure audio usage tables exist."""
    try:
        # Create tables separately to satisfy SQLite single-statement execution
        if pool.pool:
            await pool.execute(
                """
                CREATE TABLE IF NOT EXISTS audio_usage_daily (
                    user_id INTEGER NOT NULL,
                    day DATE NOT NULL,
                    minutes_used DOUBLE PRECISION NOT NULL DEFAULT 0,
                    jobs_started INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (user_id, day)
                )
                """
            )
            await pool.execute(
                """
                CREATE TABLE IF NOT EXISTS audio_user_tiers (
                    user_id INTEGER PRIMARY KEY,
                    tier TEXT NOT NULL
                )
                """
            )
        else:
            await pool.execute(
                """
                CREATE TABLE IF NOT EXISTS audio_usage_daily (
                    user_id INTEGER NOT NULL,
                    day TEXT NOT NULL,
                    minutes_used REAL NOT NULL DEFAULT 0,
                    jobs_started INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (user_id, day)
                )
                """
            )
            await pool.execute(
                """
                CREATE TABLE IF NOT EXISTS audio_user_tiers (
                    user_id INTEGER PRIMARY KEY,
                    tier TEXT NOT NULL
                )
                """
            )
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        logger.debug("audio_usage_daily ensure failed")


async def get_user_tier(user_id: int) -> str:
    """Return user tier string from DB if set, else 'free'."""
    try:
        pool = await get_db_pool()
        await _ensure_tables(pool)
        # Use portable fetchone helper (supports both backends)
        row = await pool.fetchone("SELECT tier FROM audio_user_tiers WHERE user_id = ?", int(user_id))
        if row and row.get("tier"):
            return str(row["tier"]).strip()
        return "free"
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        logger.debug("get_user_tier failed")
        return "free"


async def set_user_tier(user_id: int, tier: str) -> None:
    """Set or update a user's audio tier in the DB."""
    try:
        pool = await get_db_pool()
        await _ensure_tables(pool)
        if pool.pool:
            await pool.execute(
                "INSERT INTO audio_user_tiers (user_id, tier) VALUES ($1, $2) ON CONFLICT (user_id) DO UPDATE SET tier = EXCLUDED.tier",
                int(user_id),
                tier,
            )
        else:
            await pool.execute(
                "INSERT INTO audio_user_tiers (user_id, tier) VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET tier=excluded.tier",
                int(user_id),
                tier,
            )
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        logger.debug("set_user_tier failed")
        raise


# Every audio limit when usage quotas are off: no tier applies (spec 2).
_UNLIMITED_AUDIO_LIMITS: dict[str, float | None] = {
    "daily_minutes": None,
    "monthly_minutes": None,
    "concurrent_streams": None,
    "concurrent_jobs": None,
    "max_file_size_mb": None,
}


async def get_limits_for_user(user_id: int) -> dict[str, float | None]:
    """The user's audio limits (spec 2): their limits.* values; None is unlimited."""
    limits = dict(_UNLIMITED_AUDIO_LIMITS)
    if not usage_quotas_enabled():
        return limits
    uid = int(user_id)
    limits["daily_minutes"] = await user_quota(uid, "limits.audio_daily_minutes")
    limits["monthly_minutes"] = await user_quota(uid, "limits.transcription_minutes_per_month")
    limits["concurrent_jobs"] = await user_quota(uid, "limits.audio_concurrent_jobs")
    return limits


async def _monthly_minutes_exhausted(user_id: int, monthly_limit: float | None, minutes_requested: float) -> bool:
    """True when this request would take the user past their calendar-month (UTC) minutes."""
    if monthly_limit is None:
        return False
    try:
        used_seconds = await ledger_used_this_month(str(int(user_id)), "minutes")
    except Exception:  # noqa: BLE001 - a counter failure must not block requests (spec 2 §2)
        logger.opt(exception=True).warning(
            "Audio monthly-minutes counter failed for user_id={}; treating month as not exhausted", user_id
        )
        return False
    return used_seconds + _audio_minutes_units(minutes_requested) > _audio_minutes_units(float(monthly_limit))


async def get_daily_minutes_used(user_id: int) -> float:
    """Audio minutes used today (UTC), from the resource ledger; 0.0 if it can't be read."""
    try:
        return float(await ledger_used_today(str(int(user_id)), "minutes")) / 60.0
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        logger.debug("get_daily_minutes_used failed")
        return 0.0


async def get_monthly_minutes_used(user_id: int) -> float:
    """Audio minutes used this calendar month (UTC), from the resource ledger; 0.0 if it can't be read."""
    try:
        return float(await ledger_used_this_month(str(int(user_id)), "minutes")) / 60.0
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        logger.debug("get_monthly_minutes_used failed")
        return 0.0


async def monthly_minutes_exhausted(user_id: int, minutes_requested: float) -> bool:
    """True when this request would take the user past a monthly minutes limit (False when none is set or the counter fails)."""
    limits = await get_limits_for_user(user_id)
    return await _monthly_minutes_exhausted(user_id, limits.get("monthly_minutes"), minutes_requested)


_daily_ledger: ResourceDailyLedger | None = None  # type: ignore[assignment]
_daily_ledger_lock = asyncio.Lock()
# Tracks whether we have attempted to backfill legacy audio_usage_daily rows
# into the shared ResourceDailyLedger for the current process.
_audio_minutes_legacy_backfill_done = False


async def _get_daily_ledger() -> ResourceDailyLedger | None:
    """
    Lazily initialize the shared ResourceDailyLedger for audio minutes.

    When available, the ledger is the canonical source of truth for daily
    minutes caps: callers write new usage via ``add_daily_minutes`` or
    ``consume_daily_minutes`` and read remaining quota via
    ``_ledger_remaining_minutes``. The legacy ``audio_usage_daily`` table is
    consulted only for a one-time backfill on first use (per process).
    """
    global _daily_ledger
    # If the ledger implementation is not available, skip silently.
    if ResourceDailyLedger is None or LedgerEntry is None:
        return None
    if _daily_ledger is not None:
        return _daily_ledger
    async with _daily_ledger_lock:
        if _daily_ledger is not None:
            return _daily_ledger
        try:
            ledger = ResourceDailyLedger()  # type: ignore[call-arg]
            await ledger.initialize()
            try:
                # Best-effort backfill for upgrades: if legacy audio_usage_daily
                # rows exist for the current UTC day, mirror them into the
                # generic ResourceDailyLedger so that daily minutes caps remain
                # accurate immediately after deploy. This runs once per process
                # and is idempotent via LedgerEntry.op_id.
                await _backfill_audio_usage_daily_to_ledger(ledger)
            except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:  # pragma: no cover - defensive
                logger.debug(
                    "Audio quotas: legacy audio_usage_daily backfill failed; continuing without backfill"
                )
            _daily_ledger = ledger
            return ledger
        except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:  # pragma: no cover - best-effort shadow path
            logger.debug("Audio quotas ResourceDailyLedger init failed; continuing without ledger")
            _daily_ledger = None
            return None


async def _backfill_audio_usage_daily_to_ledger(ledger: ResourceDailyLedger) -> None:
    """
    Best-effort migration helper: mirror today's audio_usage_daily minutes
    into ResourceDailyLedger once per process.

    This preserves in-progress daily minutes caps when upgrading from older
    versions that only wrote to audio_usage_daily.
    """
    global _audio_minutes_legacy_backfill_done
    if _audio_minutes_legacy_backfill_done:
        return
    try:
        pool = await get_db_pool()
        # Ensure legacy tables exist if callers created them previously; this
        # is a no-op when they do not.
        await _ensure_tables(pool)
        day = datetime.now(timezone.utc).date()
        rows = []
        if pool.pool:
            try:
                rows = await pool.fetch(
                    "SELECT user_id, minutes_used FROM audio_usage_daily WHERE day=$1",
                    day,
                )
                iterable = [(int(r["user_id"]), float(r["minutes_used"])) for r in rows or []]
            except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
                logger.debug("Audio quotas: legacy backfill query (Postgres) failed")
                iterable = []
        else:
            try:
                rows = await pool.fetch(
                    "SELECT user_id, minutes_used FROM audio_usage_daily WHERE user_id IS NOT NULL AND day=?",
                    day.isoformat(),
                )
                iterable = [(int(r[0]), float(r[1])) for r in rows or []]
            except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
                logger.debug("Audio quotas: legacy backfill query (SQLite) failed")
                iterable = []

        for user_id, minutes_used in iterable:
            units = int(max(0, round(float(minutes_used) * 60.0)))
            if units <= 0:
                continue
            entry = LedgerEntry(  # type: ignore[call-arg]
                entity_scope="user",
                entity_value=str(user_id),
                category="minutes",
                units=units,
                op_id=f"audio-minutes-legacy:{user_id}:{day}",
                occurred_at=datetime.now(timezone.utc),
            )
            try:
                await ledger.add(entry)
            except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
                logger.debug(
                    "Audio quotas: ResourceDailyLedger legacy backfill add failed"
                )
        _audio_minutes_legacy_backfill_done = True
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        logger.debug(
            "Audio quotas: legacy audio_usage_daily backfill to ResourceDailyLedger failed; continuing without backfill"
        )


def _audio_minutes_units(minutes: float) -> int:
    """Convert fractional minutes into the integer second units used by the ledger."""
    return int(max(0, round(float(minutes) * 60.0)))


def _audio_minutes_op_id(user_id: int, day: str, units: int, operation_id: str | None = None) -> str:
    """Return a stable caller operation id or generate a unique audio-minute id."""
    if operation_id is not None:
        op_id = str(operation_id).strip()
        if op_id:
            return op_id
    return f"audio-minutes:{int(user_id)}:{day}:{int(units)}:{uuid.uuid4().hex}"


async def _require_daily_ledger() -> ResourceDailyLedger:
    """Return the canonical daily ledger or raise when quota storage is unavailable."""
    ledger = await _get_daily_ledger()
    if ledger is None or LedgerEntry is None:
        raise AudioQuotaStoreUnavailable("audio quota daily ledger is unavailable")
    return ledger


def _build_audio_minutes_entry(user_id: int, units: int, operation_id: str | None = None) -> LedgerEntry:
    """Build a daily-ledger entry for audio minute consumption."""
    day = datetime.now(timezone.utc).date().isoformat()
    return LedgerEntry(  # type: ignore[call-arg, return-value]
        entity_scope="user",
        entity_value=str(int(user_id)),
        category="minutes",
        units=int(units),
        op_id=_audio_minutes_op_id(user_id=int(user_id), day=day, units=int(units), operation_id=operation_id),
        occurred_at=datetime.now(timezone.utc),
    )


async def add_daily_minutes(
    user_id: int,
    minutes: float,
    *,
    operation_id: str | None = None,
    op_id: str | None = None,
) -> None:
    units = _audio_minutes_units(minutes)
    if units <= 0:
        return

    try:
        ledger = await _get_daily_ledger()
        if ledger is not None and LedgerEntry is not None:
            entry = _build_audio_minutes_entry(
                user_id=int(user_id),
                units=units,
                operation_id=operation_id or op_id,
            )
            try:
                await ledger.add(entry)
            except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
                logger.debug("Audio quotas ResourceDailyLedger add failed; shadow-only")
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        logger.debug("Audio quotas: ResourceDailyLedger shadow path failed; ignoring")

    # Legacy audio_usage_daily writes have been removed; usage is tracked
    # solely via ResourceDailyLedger for new events.


async def consume_daily_minutes(
    user_id: int,
    minutes_requested: float,
    *,
    operation_id: str | None = None,
    op_id: str | None = None,
) -> tuple[bool, float | None]:
    """
    Atomically enforce and record audio daily-minute usage.

    Returns ``(allowed, remaining_after)``. ``remaining_after`` is ``None`` for
    unlimited tiers or a no-op request whose remaining quota cannot be computed.
    If the canonical ledger is unavailable for a positive consume, raises
    ``AudioQuotaStoreUnavailable`` so API layers can use bounded fail-open.
    """
    units = _audio_minutes_units(minutes_requested)
    if units <= 0:
        limits = await get_limits_for_user(user_id)
        limit = limits.get("daily_minutes")
        if limit is None:
            return True, None
        remaining = await _ledger_remaining_minutes(user_id=int(user_id), daily_limit_minutes=float(limit))
        if remaining is None:
            return True, None
        return True, remaining

    limits = await get_limits_for_user(user_id)
    limit = limits.get("daily_minutes")
    if await _monthly_minutes_exhausted(user_id, limits.get("monthly_minutes"), minutes_requested):
        _metrics_increment("audio_quota_violations_total", {"type": "monthly_minutes"})
        return False, 0.0
    ledger = await _require_daily_ledger()
    entry = _build_audio_minutes_entry(user_id=int(user_id), units=units, operation_id=operation_id or op_id)

    if limit is None:
        try:
            await ledger.add(entry)
            return True, None
        except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Audio quotas unlimited daily-minute add failed")
            raise AudioQuotaStoreUnavailable("audio quota daily ledger add failed") from exc

    cap_units = _audio_minutes_units(float(limit))
    if cap_units <= 0:
        _metrics_increment("audio_quota_violations_total", {"type": "daily_minutes"})
        return False, 0.0

    try:
        consume_if_available = getattr(ledger, "consume_if_available", None)
        if callable(consume_if_available):
            allowed, remaining_units = await consume_if_available(entry, daily_cap=cap_units)
            if not allowed:
                _metrics_increment("audio_quota_violations_total", {"type": "daily_minutes"})
            return bool(allowed), float(max(0, int(remaining_units))) / 60.0

        # Compatibility fallback for tests/lightweight fakes. Real
        # ResourceDailyLedger provides consume_if_available for DB-backed
        # atomicity; this lock avoids same-process races when that API is not
        # present.
        consume_lock = await _get_audio_minutes_consume_lock(int(user_id))
        try:
            async with consume_lock:
                remaining_units = await ledger.remaining_for_day(
                    entity_scope="user",
                    entity_value=str(int(user_id)),
                    category="minutes",
                    daily_cap=cap_units,
                )
                if units > remaining_units:
                    _metrics_increment("audio_quota_violations_total", {"type": "daily_minutes"})
                    return False, float(max(0, int(remaining_units))) / 60.0
                await ledger.add(entry)
                return True, float(max(0, int(remaining_units) - units)) / 60.0
        finally:
            await _cleanup_audio_minutes_consume_lock(int(user_id), consume_lock)
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Audio quotas consume failed")
        raise AudioQuotaStoreUnavailable("audio quota daily ledger consume failed") from exc


AudioQuotaCheckFn = Callable[[int, float], Awaitable[tuple[bool, float | None]]]
AudioQuotaAddFn = Callable[[int, float], Awaitable[Any]]
AudioQuotaConsumeFn = Callable[..., Awaitable[tuple[bool, float | None]]]
AudioQuotaHelperResolver = Callable[[str], Any]


def _resolve_audio_quota_helper(
    resolver: AudioQuotaHelperResolver | None,
    name: str,
    default: Any,
    *,
    optional: bool = False,
) -> Any:
    """Resolve an endpoint shim helper while preserving core defaults."""
    if resolver is None:
        return default
    try:
        return resolver(name)
    except (AttributeError, KeyError, NameError):
        if optional:
            return None
        return default


async def consume_daily_minutes_with_compat(
    user_id: int,
    minutes: float,
    *,
    operation_id: str | None = None,
    quota_helper_resolver: AudioQuotaHelperResolver | None = None,
) -> tuple[bool, float | None]:
    """Consume daily minutes while honoring legacy endpoint quota shims."""
    check_fn: AudioQuotaCheckFn = _resolve_audio_quota_helper(
        quota_helper_resolver,
        "check_daily_minutes_allow",
        check_daily_minutes_allow,
    )
    add_fn: AudioQuotaAddFn = _resolve_audio_quota_helper(
        quota_helper_resolver,
        "add_daily_minutes",
        add_daily_minutes,
    )
    consume_fn: AudioQuotaConsumeFn | None = _resolve_audio_quota_helper(
        quota_helper_resolver,
        "consume_daily_minutes",
        consume_daily_minutes,
        optional=True,
    )
    legacy_quota_override = (
        consume_fn is None
        or (
            consume_fn is consume_daily_minutes
            and (check_fn is not check_daily_minutes_allow or add_fn is not add_daily_minutes)
        )
    )
    if legacy_quota_override:
        allowed, remaining_after = await check_fn(user_id, minutes)
        if allowed:
            await add_fn(user_id, minutes)
        return allowed, remaining_after
    return await consume_fn(user_id, minutes, operation_id=operation_id)


async def consume_daily_minutes_if_allowed(
    user_id: int,
    minutes: float,
    *,
    op_id: str | None = None,
    operation_id: str | None = None,
) -> tuple[bool, float | None]:
    """Backward-compatible alias for atomic daily-minute consumption."""
    return await consume_daily_minutes(
        user_id=user_id,
        minutes_requested=minutes,
        operation_id=operation_id or op_id,
    )


async def _ledger_remaining_minutes(user_id: int, daily_limit_minutes: float) -> float | None:
    """
    Compute remaining minutes using the ResourceDailyLedger when available.

    Returns None when the ledger is unavailable; otherwise returns remaining
    minutes (float) based on a per-day cap (converted to seconds internally).
    """
    ledger = await _get_daily_ledger()
    if ledger is None:
        return None
    try:
        cap_units = int(max(0, round(daily_limit_minutes * 60.0)))
        remaining_units = await ledger.remaining_for_day(
            entity_scope="user",
            entity_value=str(int(user_id)),
            category="minutes",
            daily_cap=cap_units,
        )
        return float(remaining_units) / 60.0
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        logger.debug("Audio quotas ledger remaining check failed; fallback to legacy")
        return None


async def increment_jobs_started(user_id: int) -> None:
    pool = await get_db_pool()
    await _ensure_tables(pool)
    day = datetime.now(timezone.utc).date()
    try:
        if pool.pool:
            await pool.execute(
                """
                INSERT INTO audio_usage_daily (user_id, day, minutes_used, jobs_started)
                VALUES ($1, $2, 0, 1)
                ON CONFLICT (user_id, day) DO UPDATE SET jobs_started = audio_usage_daily.jobs_started + 1
                """,
                user_id,
                day,
            )
        else:
            await pool.execute(
                """
                INSERT INTO audio_usage_daily (user_id, day, minutes_used, jobs_started)
                VALUES (?, ?, 0, 1)
                ON CONFLICT(user_id, day) DO UPDATE SET jobs_started = jobs_started + 1
                """,
                user_id,
                day.isoformat(),
            )
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        logger.debug("increment_jobs_started failed")


async def can_start_job(user_id: int) -> tuple[bool, str]:
    """Per-user synchronous concurrency is deferred (spec 2 Non-goals); always admits."""
    return True, "OK"


async def finish_job(user_id: int) -> None:
    """No-op hook: per-user job concurrency isn't tracked (see TASK-13435)."""
    return None


async def can_start_stream(user_id: int) -> tuple[bool, str]:
    """Per-user synchronous concurrency is deferred (spec 2 Non-goals); always admits."""
    return True, "OK"


async def finish_stream(user_id: int) -> None:
    """No-op hook: per-user stream concurrency isn't tracked (see TASK-13435)."""
    return None


async def check_daily_minutes_allow(user_id: int, minutes_requested: float) -> tuple[bool, float | None]:
    """
    Check whether the requested daily transcription minutes can be consumed and report the remaining minutes.

    Parameters:
        user_id (int): ID of the user whose quota is being checked.
        minutes_requested (float): Minutes requested to consume from today's quota.

    Returns:
        Tuple[bool, Optional[float]]:
            allowed: `True` if the requested minutes can be consumed, `False` otherwise.
            remaining_after: Remaining minutes for the current UTC day after the request, or `None` if the user's daily limit is unlimited.

    Notes:
        When the request is denied due to insufficient remaining minutes, a quota violation metric is recorded.
    """
    limits = await get_limits_for_user(user_id)
    if await _monthly_minutes_exhausted(user_id, limits.get("monthly_minutes"), minutes_requested):
        _metrics_increment("audio_quota_violations_total", {"type": "monthly_minutes"})
        return False, 0.0
    limit = limits.get("daily_minutes")
    if limit is None:
        return True, None

    # Enforce new usage against the shared ResourceDailyLedger. Legacy
    # audio_usage_daily rows are backfilled into the ledger during
    # initialization and are not a safe fallback after new writes moved to the
    # ledger-only path.
    ledger_remaining = await _ledger_remaining_minutes(user_id=int(user_id), daily_limit_minutes=float(limit))
    if ledger_remaining is not None:
        if minutes_requested > ledger_remaining:
            _metrics_increment("audio_quota_violations_total", {"type": "daily_minutes"})
            return False, max(0.0, ledger_remaining)
        return True, ledger_remaining - minutes_requested

    raise AudioQuotaStoreUnavailable("audio quota daily ledger is unavailable")


def bytes_to_seconds(byte_count: int, sample_rate: int) -> float:
    # Float32 mono: 4 bytes per sample
    """
    Convert a byte count of Float32 mono audio into playback duration in seconds.

    Treats audio as mono Float32 (4 bytes per sample). Negative byte counts are treated as zero. If `sample_rate` is zero or otherwise falsy, a default sample rate of 16000 Hz is used.

    Parameters:
        byte_count (int): Number of bytes of audio data.
        sample_rate (int): Samples per second for the audio; if falsy, 16000 is used.

    Returns:
        float: Duration in seconds represented by the given byte count.
    """
    samples = max(0, int(byte_count // 4))
    return float(samples) / float(sample_rate or 16000)


async def heartbeat_stream(user_id: int) -> None:
    """No-op hook: per-user stream concurrency isn't tracked (see TASK-13435)."""
    return None


def _get_job_ttl_seconds() -> int:
    """Determine TTL for RG job leases; defaults to 600 seconds."""
    val_env = os.getenv("AUDIO_JOB_TTL_SECONDS")
    if val_env:
        try:
            v = int(val_env)
            return max(30, min(3600, v))
        except (TypeError, ValueError):
            logger.debug("Audio job TTL: invalid AUDIO_JOB_TTL_SECONDS")
    try:
        from tldw_Server_API.app.core.config import load_comprehensive_config  # lazy import

        cfg = load_comprehensive_config()
        if cfg and cfg.has_section("Audio-Quota"):
            try:
                v = int(cfg.get("Audio-Quota", "job_ttl_seconds", fallback="600"))
            except (TypeError, ValueError, configparser.Error):
                logger.debug("Audio job TTL: invalid config value")
                v = 600
            return max(30, min(3600, v))
    except (OSError, RuntimeError, configparser.Error, TypeError, ValueError):
        logger.debug("Audio job TTL: failed to load config")
    return 600


def get_job_heartbeat_interval_seconds() -> int:
    """Return a safe heartbeat interval derived from the job TTL."""
    ttl = _get_job_ttl_seconds()
    if ttl <= 10:
        return ttl
    # Refresh roughly twice per TTL window while avoiding overly chatty loops.
    return max(10, ttl // 2)


async def heartbeat_jobs(user_id: int) -> None:
    """No-op hook: per-user job concurrency isn't tracked (see TASK-13435)."""
    return None


_tier_overrides_deprecation_warned = False


def _apply_tier_overrides_from_config(base: dict[str, dict[str, float | None]]) -> dict[str, dict[str, float | None]]:
    """
    Merge a base per-tier limits mapping with overrides from the environment and configuration.

    Checks the AUDIO_TIER_LIMITS_JSON environment variable (JSON object mapping tier names to partial
    limit objects) first, then the application's [Audio-Quota] config section. Keys recognized per tier
    are: `daily_minutes`, `concurrent_streams`, `concurrent_jobs`, and `max_file_size_mb`. Only tiers
    present in the base mapping are updated; other entries in the environment JSON are ignored.

    Environment JSON takes precedence over config file values. For `daily_minutes`, the string values
    "none", "unlimited", or "-1" in the config are treated as `None` (unlimited).

    Parameters:
        base (Dict[str, Dict[str, Optional[float]]]): Original tier limits mapping to copy and merge into.

    Returns:
        Dict[str, Dict[str, Optional[float]]]: A new mapping with overrides applied.
    """
    merged = {k: v.copy() for k, v in base.items()}
    # Env JSON has priority
    import json as _json
    try:
        j = os.getenv("AUDIO_TIER_LIMITS_JSON")
        if j:
            data = _json.loads(j)
            if isinstance(data, dict):
                for tier, vals in data.items():
                    if tier in merged and isinstance(vals, dict):
                        merged[tier].update(vals)
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        logger.debug("AUDIO_TIER_LIMITS_JSON parse failed")
    # Config file overrides
    try:
        from tldw_Server_API.app.core.config import load_comprehensive_config
        cfg = load_comprehensive_config()
        if cfg and cfg.has_section('Audio-Quota'):
            for tier in ("free", "standard", "premium"):
                for key in ("daily_minutes", "concurrent_streams", "concurrent_jobs", "max_file_size_mb"):
                    opt = f"{tier}_{key}"
                    if cfg.has_option('Audio-Quota', opt):
                        val = cfg.get('Audio-Quota', opt)
                        # Coerce None for 'unlimited'
                        if str(val).strip().lower() in {"none", "unlimited", "-1"} and key == "daily_minutes":
                            merged[tier][key] = None
                        else:
                            try:
                                merged[tier][key] = float(val) if key == "daily_minutes" else int(val)
                            except (TypeError, ValueError):
                                logger.debug("Audio-Quota override parse failed")
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        logger.debug("Audio-Quota config overrides failed")

    global _tier_overrides_deprecation_warned
    if not _tier_overrides_deprecation_warned and merged != base:
        _tier_overrides_deprecation_warned = True
        logger.warning(
            "AUDIO_TIER_LIMITS_JSON / [Audio-Quota] {tier}_* no longer set limits; "
            "use the limits.audio_* UserProfiles values (spec 2)"
        )
    return merged


# Apply overrides on import
TIER_LIMITS = _apply_tier_overrides_from_config(TIER_LIMITS)
