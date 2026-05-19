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
from datetime import datetime, timezone
from functools import lru_cache

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool

try:
    from tldw_Server_API.app.core.Metrics.metrics_manager import MetricDefinition, MetricType, get_metrics_registry
except ImportError:  # pragma: no cover
    get_metrics_registry = None  # type: ignore
    MetricDefinition = None  # type: ignore
    MetricType = None  # type: ignore

try:
    # Resource Governor (optional, guarded by global RG_ENABLED/config)
    from tldw_Server_API.app.core.config import (
        rg_backend,
        rg_enabled,
        rg_policy_path,
        rg_policy_reload_enabled,
        rg_policy_reload_interval_sec,
        rg_policy_store,
    )
    from tldw_Server_API.app.core.Resource_Governance import (
        MemoryResourceGovernor,
        RedisResourceGovernor,
        RGRequest,
    )
    from tldw_Server_API.app.core.Resource_Governance.authnz_policy_store import (
        AuthNZPolicyStore,
    )
    from tldw_Server_API.app.core.Resource_Governance.policy_loader import (
        PolicyLoader,
        PolicyReloadConfig,
        db_policy_loader,
    )
except ImportError:  # pragma: no cover - RG is optional for audio quotas
    RGRequest = None  # type: ignore
    MemoryResourceGovernor = None  # type: ignore
    RedisResourceGovernor = None  # type: ignore
    PolicyLoader = None  # type: ignore
    PolicyReloadConfig = None  # type: ignore
    db_policy_loader = None  # type: ignore
    AuthNZPolicyStore = None  # type: ignore
    rg_enabled = None  # type: ignore
    rg_policy_store = None  # type: ignore
    rg_policy_reload_enabled = None  # type: ignore
    rg_policy_reload_interval_sec = None  # type: ignore
    rg_policy_path = None  # type: ignore
    rg_backend = None  # type: ignore

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
    asyncio.CancelledError,
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


def _rg_audio_enabled() -> bool:
    """
    Return True when audio quotas should use the shared Resource Governor for
    streams/jobs concurrency instead of Redis/in-process counters.

    This is controlled by the global ResourceGovernor enablement
    (`RG_ENABLED` / config.txt). When RG is disabled, concurrency limiting is
    treated as disabled (legacy counters are retired).
    """
    if rg_enabled is not None:
        try:
            return bool(rg_enabled(True))  # type: ignore[func-returns-value]
        except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
            return False
    return False


_rg_audio_governor = None
_rg_audio_loader = None
_rg_audio_lock = asyncio.Lock()
_rg_stream_handles: dict[int, list[str]] = {}
_rg_job_handles: dict[int, list[str]] = {}
_rg_job_handle_locks: dict[int, asyncio.Lock] = {}
_rg_job_handle_locks_lock = asyncio.Lock()
_rg_audio_init_error: str | None = None
_rg_audio_init_error_logged = False
_rg_audio_fallback_logged = False


def _safe_config_or_env(name: str, config_fn, env_key: str, default: str = "") -> str:
    """Safely retrieve a config value or fall back to an environment variable."""
    if not callable(config_fn):
        return os.getenv(env_key, default)
    try:
        return str(config_fn())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, configparser.Error):
        logger.debug(f"RG audio context failed to resolve {name}")
        return os.getenv(env_key, default)


def _rg_audio_context() -> dict[str, str]:
    backend = _safe_config_or_env("backend", rg_backend, "RG_BACKEND", "memory")  # type: ignore[arg-type]
    store = _safe_config_or_env("policy_store", rg_policy_store, "RG_POLICY_STORE", "")  # type: ignore[arg-type]
    policy_path = _safe_config_or_env(
        "policy_path",
        rg_policy_path,
        "RG_POLICY_PATH",
        "tldw_Server_API/Config_Files/resource_governor_policies.yaml",
    )
    try:
        policy_path_resolved = os.path.abspath(policy_path)
    except OSError:
        logger.debug("RG audio context failed to resolve policy_path_resolved")
        policy_path_resolved = policy_path
    reload_enabled = _safe_config_or_env(
        "policy_reload_enabled",
        rg_policy_reload_enabled,
        "RG_POLICY_RELOAD_ENABLED",
        "",
    )  # type: ignore[arg-type]
    reload_interval = _safe_config_or_env(
        "policy_reload_interval",
        rg_policy_reload_interval_sec,
        "RG_POLICY_RELOAD_INTERVAL_SEC",
        "",
    )  # type: ignore[arg-type]
    try:
        cwd = os.getcwd()
    except OSError:
        logger.debug("RG audio context failed to resolve cwd")
        cwd = ""
    return {
        "backend": str(backend),
        "policy_path": str(policy_path),
        "policy_path_resolved": str(policy_path_resolved),
        "policy_store": str(store),
        "policy_reload_enabled": str(reload_enabled),
        "policy_reload_interval": str(reload_interval),
        "cwd": str(cwd),
    }


async def _log_rg_audio_init_failure(exc: Exception) -> None:
    global _rg_audio_init_error, _rg_audio_init_error_logged
    ctx = _rg_audio_context()
    async with _rg_audio_lock:
        _rg_audio_init_error = repr(exc)
        if _rg_audio_init_error_logged:
            return
        _rg_audio_init_error_logged = True
        logger.opt(exception=exc).error(
            "Audio ResourceGovernor init failed; using diagnostics-only compatibility shim (no enforcement/counters). "
            "backend={backend} policy_path={policy_path} policy_path_resolved={policy_path_resolved} "
            "policy_store={policy_store} reload_enabled={policy_reload_enabled} "
            "reload_interval={policy_reload_interval} cwd={cwd}",
            **ctx,
        )


async def _log_rg_audio_fallback(reason: str) -> None:
    global _rg_audio_fallback_logged
    ctx = _rg_audio_context()
    async with _rg_audio_lock:
        if _rg_audio_fallback_logged:
            return
        _rg_audio_fallback_logged = True
        init_error = _rg_audio_init_error
        logger.error(
            "Audio ResourceGovernor unavailable; using diagnostics-only compatibility shim (no enforcement/counters). "
            "reason={} init_error={} backend={backend} policy_path={policy_path} "
            "policy_path_resolved={policy_path_resolved} policy_store={policy_store} "
            "reload_enabled={policy_reload_enabled} reload_interval={policy_reload_interval} cwd={cwd}",
            reason,
            init_error,
            **ctx,
        )


def _reset_in_process_counters_for_tests() -> None:
    """
    Reset in-process RG handle tracking state for tests.

    This helper clears in-process handle registries used for stream and job
    concurrency so integration tests can start from a clean slate. It is not
    intended for application runtime use.
    """
    _rg_stream_handles.clear()
    _rg_job_handles.clear()
    _rg_job_handle_locks.clear()


async def _get_audio_rg_governor():
    """
    Lazily initialize a process-local ResourceGovernor instance for audio
    streams/jobs using the same configuration helpers as app.main.

    On failure, returns None and callers continue in diagnostics-only mode
    (fail-open), even when RG is enabled.
    """
    global _rg_audio_governor, _rg_audio_loader
    if not _rg_audio_enabled():
        return None
    # If RG is not available in this environment, continue fail-open.
    if RGRequest is None or PolicyLoader is None or PolicyReloadConfig is None or rg_policy_store is None:
        await _log_rg_audio_fallback("rg_components_unavailable")
        return None
    if _rg_audio_governor is not None:
        return _rg_audio_governor
    init_error: Exception | None = None
    async with _rg_audio_lock:
        if _rg_audio_governor is not None:
            return _rg_audio_governor
        try:
            store_mode = rg_policy_store()  # type: ignore[operator]
        except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
            store_mode = "file"
        try:
            if store_mode == "db" and AuthNZPolicyStore is not None and db_policy_loader is not None:
                store = AuthNZPolicyStore()  # type: ignore[call-arg]
                interval = rg_policy_reload_interval_sec() if rg_policy_reload_interval_sec else 10  # type: ignore[operator]
                loader = db_policy_loader(store, PolicyReloadConfig(enabled=True, interval_sec=interval))  # type: ignore[call-arg]
            else:
                # File-based loader mirroring app.main behavior
                enabled = rg_policy_reload_enabled() if rg_policy_reload_enabled else True  # type: ignore[operator]
                interval = rg_policy_reload_interval_sec() if rg_policy_reload_interval_sec else 10  # type: ignore[operator]
                path = rg_policy_path() if rg_policy_path else "Config_Files/resource_governor_policies.yaml"  # type: ignore[operator]
                loader = PolicyLoader(path, PolicyReloadConfig(enabled=enabled, interval_sec=interval))  # type: ignore[call-arg]
            await loader.load_once()
            _rg_audio_loader = loader
            try:
                backend = rg_backend() if rg_backend else "memory"  # type: ignore[operator]
            except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
                backend = "memory"
            if backend == "redis" and RedisResourceGovernor is not None:
                gov = RedisResourceGovernor(policy_loader=loader)  # type: ignore[call-arg]
            else:
                gov = MemoryResourceGovernor(policy_loader=loader)  # type: ignore[call-arg]
            _rg_audio_governor = gov
            return gov
        except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS as e:
            init_error = e
            _rg_audio_governor = None
            _rg_audio_loader = None
    if init_error is not None:
        await _log_rg_audio_init_failure(init_error)
    return None


async def _get_job_handle_lock(user_id: int) -> asyncio.Lock:
    """Return (and lazily create) the per-user lock protecting RG job handles."""
    uid = int(user_id)
    async with _rg_job_handle_locks_lock:
        lock = _rg_job_handle_locks.get(uid)
        if lock is None:
            lock = asyncio.Lock()
            _rg_job_handle_locks[uid] = lock
        return lock


async def _cleanup_job_handle_lock(user_id: int) -> None:
    """Remove a per-user job handle lock when no handles remain."""
    uid = int(user_id)
    async with _rg_job_handle_locks_lock:
        # If handles were added after the pop, keep the lock.
        if _rg_job_handles.get(uid):
            return
        lock = _rg_job_handle_locks.get(uid)
        if lock and not lock.locked():
            _rg_job_handle_locks.pop(uid, None)


async def _cleanup_stream_handles(user_id: int) -> None:
    """Prune empty stream handle entries to avoid unbounded growth."""
    uid = int(user_id)
    async with _rg_audio_lock:
        handles = _rg_stream_handles.get(uid)
        if handles:
            return
        _rg_stream_handles.pop(uid, None)


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


async def get_limits_for_user(user_id: int) -> dict[str, float | None]:
    tier = await get_user_tier(user_id)
    limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"]).copy()
    overrides = await _get_user_override_limits(user_id)
    for key, value in overrides.items():
        if value is not None:
            limits[key] = value
    return limits


async def _get_user_override_limits(user_id: int) -> dict[str, float | None]:
    try:
        from tldw_Server_API.app.core.UserProfiles.overrides_repo import UserProfileOverridesRepo

        pool = await get_db_pool()
        repo = UserProfileOverridesRepo(pool)
        await repo.ensure_tables()
        rows = await repo.list_overrides_for_user(int(user_id))
        overrides: dict[str, float | None] = {}
        for row in rows:
            key = str(row.get("key") or "")
            value = row.get("value")
            if value is None:
                continue
            if key == "limits.audio_daily_minutes":
                overrides["daily_minutes"] = float(value)
            elif key == "limits.audio_concurrent_jobs":
                overrides["concurrent_jobs"] = int(value)
        return overrides
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        logger.debug("Audio quota overrides unavailable")
        return {}


async def get_daily_minutes_used(user_id: int) -> float:
    pool = await get_db_pool()
    await _ensure_tables(pool)
    day = datetime.now(timezone.utc).date().isoformat()
    try:
        if pool.pool:
            row = await pool.fetchrow(
                "SELECT minutes_used FROM audio_usage_daily WHERE user_id=$1 AND day=$2",
                user_id,
                day,
            )
            return float(row["minutes_used"]) if row else 0.0
        else:
            rows = await pool.fetch(
                "SELECT minutes_used FROM audio_usage_daily WHERE user_id=? AND day=?",
                user_id,
                day,
            )
            return float(rows[0][0]) if rows else 0.0
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        logger.debug("get_daily_minutes_used failed")
        return 0.0


_daily_ledger: ResourceDailyLedger | None = None  # type: ignore[assignment]
_daily_ledger_lock = asyncio.Lock()
# Tracks whether we have attempted to backfill legacy audio_usage_daily rows
# into the shared ResourceDailyLedger for the current process.
_audio_minutes_legacy_backfill_done = False


async def _get_daily_ledger() -> ResourceDailyLedger | None:
    """
    Lazily initialize the shared ResourceDailyLedger for audio minutes.

    When available, the ledger is the canonical source of truth for daily
    minutes caps: callers write new usage via ``add_daily_minutes`` and read
    remaining quota via ``_ledger_remaining_minutes``. The legacy
    ``audio_usage_daily`` table is consulted only for a one-time backfill on
    first use (per process) and as a fallback when the ledger cannot be
    initialized.
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
        day = datetime.now(timezone.utc).date().isoformat()
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
                    day,
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


async def add_daily_minutes(user_id: int, minutes: float) -> None:
    if minutes <= 0:
        return
    day = datetime.now(timezone.utc).date().isoformat()

    # Record against the shared ResourceDailyLedger; this is the canonical
    # source of truth for audio daily minutes. Legacy audio_usage_daily
    # counters are no longer written to and remain as compatibility state
    # only for pre-upgrade rows.
    units = int(max(0, round(float(minutes) * 60.0)))
    try:
        ledger = await _get_daily_ledger()
        if ledger is not None and LedgerEntry is not None and units > 0:
            entry = LedgerEntry(  # type: ignore[call-arg]
                entity_scope="user",
                entity_value=str(int(user_id)),
                category="minutes",
                units=units,
                op_id=f"audio-minutes:{int(user_id)}:{day}:{units}",
                occurred_at=datetime.now(timezone.utc),
            )
            try:
                await ledger.add(entry)
            except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
                logger.debug("Audio quotas ResourceDailyLedger add failed; shadow-only")
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        logger.debug("Audio quotas: ResourceDailyLedger shadow path failed; ignoring")

    # Legacy audio_usage_daily writes have been removed; usage is tracked
    # solely via ResourceDailyLedger for new events.


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
    day = datetime.now(timezone.utc).date().isoformat()
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
                day,
            )
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        logger.debug("increment_jobs_started failed")


async def can_start_job(user_id: int) -> tuple[bool, str]:
    """
    Determine whether a new concurrent audio job may be started for the given user and, if allowed, increment the active-job counter.

    If RG-based concurrency is available, the function enforces it and updates metrics. When RG is
    enabled but unavailable or reserve fails, the function logs a fallback and allows the job
    (fail-open, no concurrency tracking). When the limit is exceeded the function reports a
    descriptive message.

    Returns:
        (bool, str): `True` and `"OK"` if the job may start and the active-job counter was incremented; `False` and an explanatory message like `"Concurrent job limit reached (<max>)"` if starting the job would exceed the user's concurrency limit.
    """
    # When RG audio integration is enabled and available, enforce streams/jobs
    # concurrency via the shared governor. Legacy Redis/in-process counters are
    # retired.
    gov = await _get_audio_rg_governor()
    if gov is not None and RGRequest is not None:
        try:
            user_key = int(user_id)
            entity = f"user:{user_key}"
            policy_id = "audio.default"
            req = RGRequest(
                entity=entity,
                categories={"jobs": {"units": 1}},  # type: ignore[call-arg]
                tags={"policy_id": policy_id, "endpoint": "audio.jobs"},
            )
            dec, handle_id = await gov.reserve(req, op_id=f"audio-job:{entity}")
            if not dec.allowed or not handle_id:
                _metrics_increment("audio_quota_violations_total", {"type": "concurrent_jobs"})
                return False, "Concurrent job limit reached"
            # Track handle for explicit release in finish_job
            job_lock = await _get_job_handle_lock(user_key)
            async with job_lock:
                handles = _rg_job_handles.setdefault(user_key, [])
                handles.append(handle_id)
                # Approximate active count via local handles for metrics
                _metrics_set_gauge("audio_jobs_active", float(len(handles)), {"user_id": str(user_key)})
            return True, "OK"
        # Fail-open is intentional: if RG is temporarily unavailable, don't deny audio jobs.
        # Operators control overall service availability via infrastructure decisions, not exceptions here.
        except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
            logger.debug("RG reserve failed for jobs, failing open")
            await _log_rg_audio_fallback("rg_reserve_failed_jobs")
            return True, "OK"

    if _rg_audio_enabled():
        await _log_rg_audio_fallback("rg_governor_unavailable_jobs")
        return True, "OK"

    # RG disabled → treat as unlimited.
    return True, "OK"


async def finish_job(user_id: int) -> None:
    """
    Decrement the user's active audio job counter and update metrics.

    If a Redis client is available, decrement the per-user Redis counter and clamp it to zero; otherwise decrement the in-process counter protected by the module lock. Updates the `audio_jobs_active` gauge with the new value.

    Parameters:
        user_id (int): Numeric identifier of the user whose active-job count should be decremented.
    """
    # When RG audio integration is enabled, release one jobs concurrency lease
    # for this user if present. Legacy Redis/in-process counters are retired.
    gov = await _get_audio_rg_governor()
    if gov is not None:
        try:
            user_key = int(user_id)
            job_lock = await _get_job_handle_lock(user_key)
            should_cleanup_lock = False
            async with job_lock:
                handles = _rg_job_handles.get(user_key)
                if handles:
                    handle_id = handles.pop()
                    try:
                        await gov.release(handle_id)
                    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
                        logger.debug("RG finish_job release failed")
                remaining_handles = _rg_job_handles.get(user_key)
                remaining = len(remaining_handles or [])
                if remaining == 0:
                    _rg_job_handles.pop(user_key, None)
                    should_cleanup_lock = True
                _metrics_set_gauge("audio_jobs_active", float(remaining), {"user_id": str(user_key)})
            if should_cleanup_lock:
                await _cleanup_job_handle_lock(user_key)
            # Even when RG is in use, do not touch Redis/in-process counters here
            # to avoid double-decrement; legacy state is not used when RG is active.
            return
        except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
            logger.debug("RG error in finish_job")
            return
    return


async def can_start_stream(user_id: int) -> tuple[bool, str]:
    """
    Determines whether the user may start a new concurrent audio stream and reserves a slot if allowed.

    Attempts to allocate a concurrent-stream slot for the given user; if a slot is available the
    function records the reservation and returns success, otherwise it reports the denial reason.
    When RG is enabled but unavailable or reserve fails, the function allows the stream (fail-open).

    Returns:
        (bool, str): `True` and `"OK"` if a slot was reserved and the stream may start, `False` and a human-readable reason (e.g., "Concurrent streams limit reached (<n>)") otherwise.
    """
    # Prefer ResourceGovernor-based concurrency when enabled for audio.
    gov = await _get_audio_rg_governor()
    if gov is not None and RGRequest is not None:
        try:
            entity = f"user:{int(user_id)}"
            policy_id = "audio.default"
            req = RGRequest(
                entity=entity,
                categories={"streams": {"units": 1}},  # type: ignore[call-arg]
                tags={"policy_id": policy_id, "endpoint": "audio.stream"},
            )
            dec, handle_id = await gov.reserve(req, op_id=f"audio-stream:{entity}")
            if not dec.allowed or not handle_id:
                _metrics_increment("audio_quota_violations_total", {"type": "concurrent_streams"})
                return False, "Concurrent streams limit reached"
            handles = _rg_stream_handles.setdefault(int(user_id), [])
            handles.append(handle_id)
            _metrics_set_gauge("audio_streaming_active", float(len(handles)), {"user_id": str(int(user_id))})
            return True, "OK"
        except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
            logger.debug("RG reserve failed for streams, failing open")
            await _log_rg_audio_fallback("rg_reserve_failed_streams")
            return True, "OK"

    if _rg_audio_enabled():
        await _log_rg_audio_fallback("rg_governor_unavailable_streams")
        return True, "OK"

    # RG disabled → treat as unlimited.
    return True, "OK"


async def finish_stream(user_id: int) -> None:
    """
    Decrement the user's active audio stream count and update the corresponding metric.

    If a Redis client is available, the Redis counter for the user is decremented and clamped to zero; otherwise an in-process counter protected by an async lock is decremented and clamped to zero. Always emits the updated `audio_streaming_active` gauge with the user's id as a label.
    """
    # When RG audio integration is enabled, release one streams concurrency
    # lease for this user if present and update metrics from local handles.
    gov = await _get_audio_rg_governor()
    if gov is not None:
        try:
            handles = _rg_stream_handles.get(int(user_id)) or []
            if handles:
                handle_id = handles.pop()
                try:
                    await gov.release(handle_id)
                except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
                    logger.debug("RG finish_stream release failed")
            remaining = len(_rg_stream_handles.get(int(user_id)) or [])
            if remaining == 0:
                await _cleanup_stream_handles(int(user_id))
            _metrics_set_gauge("audio_streaming_active", float(remaining), {"user_id": str(int(user_id))})
            return
        except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
            logger.debug("RG error in finish_stream")
            return
    return


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
    limit = limits.get("daily_minutes")
    if limit is None:
        return True, None

    # Prefer the shared ResourceDailyLedger when available to enforce daily caps.
    ledger_remaining = await _ledger_remaining_minutes(user_id=int(user_id), daily_limit_minutes=float(limit))
    if ledger_remaining is not None:
        if minutes_requested > ledger_remaining:
            _metrics_increment("audio_quota_violations_total", {"type": "daily_minutes"})
            return False, max(0.0, ledger_remaining)
        return True, ledger_remaining - minutes_requested

    # Fallback to legacy audio_usage_daily enforcement.
    used = await get_daily_minutes_used(user_id)
    remaining = float(limit) - float(used)
    if minutes_requested > remaining:
        _metrics_increment("audio_quota_violations_total", {"type": "daily_minutes"})
        return False, max(0.0, remaining)
    return True, remaining - minutes_requested


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
    """
    Refresh the TTL for a user's active audio stream counter to prevent stale/ leaked keys.

    If Redis is unavailable this is a no-op; does nothing for in-process counters.
    """
    # When RG audio integration is enabled, renew any active RG streams leases
    # for this user using the configured stream TTL. Legacy Redis TTL refresh
    # is retired.
    gov = await _get_audio_rg_governor()
    if gov is not None:
        try:
            ttl = _get_stream_ttl_seconds()
            for handle_id in list(_rg_stream_handles.get(int(user_id)) or []):
                try:
                    await gov.renew(handle_id, ttl_s=ttl)
                except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
                    logger.debug("RG heartbeat_stream renew failed")
        except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
            logger.debug("RG error in heartbeat_stream")
    return


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
    """
    Renew active RG job leases to prevent premature expiry during long-running jobs.

    Legacy Redis/in-process counters are not renewed here; they do not rely on TTL.
    """
    gov = await _get_audio_rg_governor()
    if gov is None:
        return
    try:
        ttl = _get_job_ttl_seconds()
        handles = list(_rg_job_handles.get(int(user_id)) or [])
        if not handles:
            return
        job_lock = await _get_job_handle_lock(int(user_id))
        async with job_lock:
            for handle_id in list(_rg_job_handles.get(int(user_id)) or []):
                try:
                    await gov.renew(handle_id, ttl_s=ttl)
                except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
                    logger.debug("RG heartbeat_jobs renew failed")
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        logger.debug("RG error in heartbeat_jobs")


async def active_streams_count(user_id: int) -> int:
    """
    Get the number of currently active audio streams for a user.

    When Redis is available, reads the user's Redis counter key; otherwise returns the in-process counter. If the Redis value is missing or cannot be parsed as an integer, returns 0.

    Returns:
        int: Active stream count for the user.
    """
    # When RG audio integration is enabled, approximate active streams using
    # the in-process handle registry; peeking via the governor is possible but
    # not required for the status/limits endpoint semantics.
    gov = await _get_audio_rg_governor()
    if gov is not None:
        try:
            return len(_rg_stream_handles.get(int(user_id)) or [])
        except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
            return 0
    return 0


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
    return merged


# Apply overrides on import
TIER_LIMITS = _apply_tier_overrides_from_config(TIER_LIMITS)
