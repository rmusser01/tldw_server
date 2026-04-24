"""
Manages user-specific audit service instances for dependency injection.
"""

import asyncio
import concurrent.futures
import contextlib
import os
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from fastapi import Depends, HTTPException, status
from loguru import logger

try:
    from cachetools import Cache, LRUCache

    _HAS_CACHETOOLS = True
except ImportError:
    _HAS_CACHETOOLS = False
    logger.warning(
        "cachetools not found. Using bounded fallback LRU. Install with: pip install cachetools"
    )

# Local Imports
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditShutdownError,
    UnifiedAuditService,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.exceptions import (
    ServiceInitializationError,
    ServiceInitializationTimeoutError,
)
from tldw_Server_API.app.core.testing import is_test_mode, is_truthy

#######################################################################################################################

_FALSEY = {"0", "false", "no", "n", "off"}
_VALID_STORAGE_MODES = {"per_user", "shared"}


@dataclass(frozen=True)
class AuditShutdownSummary:
    """Summarize how many audit services shutdown attempted, stopped, or failed."""

    requested: int = 0
    stopped: int = 0
    timeout_count: int = 0
    error_count: int = 0
    errors: tuple[str, ...] = ()


def _settings_int(
    key: str,
    default: int,
    *,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    """Coerce a potentially loosely-typed settings value to int with clamping."""
    raw = settings.get(key, default)
    try:
        if isinstance(raw, bool):
            raise TypeError("bool is not a valid int setting")
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(f"Invalid {key}={raw!r}; using default {default}")
        value = default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _settings_float(
    key: str,
    default: float,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    """Coerce a potentially loosely-typed settings value to float with clamping."""
    raw = settings.get(key, default)
    try:
        if isinstance(raw, bool):
            raise TypeError("bool is not a valid float setting")
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(f"Invalid {key}={raw!r}; using default {default}")
        value = default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _settings_bool(key: str, default: bool) -> bool:
    """Coerce a settings value to bool, accepting common string representations."""
    raw = settings.get(key, default)
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if is_truthy(s):
        return True
    if s in _FALSEY:
        return False
    logger.warning(f"Invalid {key}={raw!r}; using default {default}")
    return default


def _resolve_audit_storage_mode() -> str:
    """Resolve audit storage mode with rollback precedence."""
    if _settings_bool("AUDIT_STORAGE_ROLLBACK", False):
        return "per_user"
    raw = settings.get("AUDIT_STORAGE_MODE", "per_user")
    try:
        mode = str(raw).strip().lower()
    except (TypeError, ValueError):
        mode = "per_user"
    if mode not in _VALID_STORAGE_MODES:
        logger.warning(f"Invalid AUDIT_STORAGE_MODE={raw!r}; using per_user")
        return "per_user"
    return mode


def _shared_audit_db_path() -> str:
    """Resolve shared audit DB path from settings."""
    return str(DatabasePaths.get_shared_audit_db_path())


def _shutdown_cache_keys(user_id: Optional[Union[int, str]]) -> list[Optional[Union[int, str]]]:
    if _resolve_audit_storage_mode() == "shared":
        return [None]
    return [user_id]


def _finalize_shutdown_summary(
    *,
    summary: AuditShutdownSummary,
    raise_on_error: bool,
    message: str,
    first_exception: BaseException | None,
) -> AuditShutdownSummary:
    if raise_on_error and (summary.timeout_count or summary.error_count):
        raise AuditShutdownError(f"{message}: {summary}") from first_exception
    return summary


async def _stop_audit_service_instance(
    service: UnifiedAuditService,
    *,
    label: str,
    timeout_s: Optional[float] = None,
) -> tuple[bool, bool, Optional[str], Optional[BaseException]]:
    owner_loop = getattr(service, "owner_loop", None)
    future: concurrent.futures.Future[Any] | None = None
    if owner_loop:
        try:
            if owner_loop.is_closed():
                owner_loop = None
            elif not owner_loop.is_running():
                logger.warning(
                    f"Audit service owner loop not running; forcing shutdown on current loop ({label})"
                )
                owner_loop = None
        except (AttributeError, RuntimeError):
            owner_loop = None

    try:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if owner_loop and current_loop is not owner_loop:
            future = asyncio.run_coroutine_threadsafe(service.stop(), owner_loop)
            awaitable = asyncio.wrap_future(future)
        else:
            awaitable = service.stop()

        if timeout_s is not None and timeout_s > 0:
            await asyncio.wait_for(awaitable, timeout=timeout_s)
        else:
            await awaitable
        return True, False, None, None
    except asyncio.TimeoutError as exc:
        if future is not None:
            with contextlib.suppress(Exception):
                future.cancel()
        timeout_label = (
            f"Audit service shutdown timed out after {timeout_s:.2f}s ({label})"
            if timeout_s is not None
            else f"Audit service shutdown timed out ({label})"
        )
        logger.error(timeout_label)
        return False, True, timeout_label, exc
    except Exception as exc:
        logger.error(f"Error shutting down audit service ({label}): {exc}", exc_info=True)
        return False, False, f"{type(exc).__name__}: {exc}", exc


def _is_test_context() -> bool:
    try:
        return bool(os.getenv("PYTEST_CURRENT_TEST")) or is_test_mode()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Failed to determine test context; defaulting to False: {}", exc)
        return False


def _mark_service_used(service: UnifiedAuditService) -> None:
    try:
        service._last_used_ts = time.monotonic()  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError, TypeError):
        logger.debug("Failed to mark service used timestamp")


def _schedule_service_stop(user_id: Optional[Union[int, str]], service: UnifiedAuditService, reason: str) -> None:
    """Schedule graceful shutdown of an audit service instance."""
    if service is None:
        return

    if getattr(service, "_tldw_stop_scheduled", False):
        return

    service._tldw_stop_scheduled = True  # type: ignore[attr-defined]
    evicted_at = time.monotonic()
    try:
        service._tldw_evicted_at = evicted_at  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError, TypeError) as exc:
        logger.debug("Failed to set eviction timestamp for user {}: {}", user_id, exc)
    service_key = id(service)

    with _services_stopping_lock:
        if service_key in _services_stopping:
            return
        _services_stopping.add(service_key)

    def _clear_stopping_flag():
        """Clear the stopping flag after service is stopped."""
        with _services_stopping_lock:
            _services_stopping.discard(service_key)
        try:
            service._tldw_stop_scheduled = False  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("Failed to clear stop-scheduled flag on audit service: {}", exc)

    async def _stop():
        try:
            eviction_deadline = time.monotonic() + EVICTION_MAX_WAIT_SECONDS
            while True:
                if time.monotonic() >= eviction_deadline:
                    logger.warning(
                        "Audit service for user {} eviction wait exceeded ({}s); forcing stop.",
                        user_id,
                        EVICTION_MAX_WAIT_SECONDS,
                    )
                    break
                if EVICTION_GRACE_SECONDS > 0:
                    remaining = eviction_deadline - time.monotonic()
                    await asyncio.sleep(min(EVICTION_GRACE_SECONDS, max(0.0, remaining)))
                else:
                    # Yield control to avoid a tight loop if grace is disabled.
                    await asyncio.sleep(0)
                last_used = getattr(service, "_last_used_ts", None)
                evicted_ts = getattr(service, "_tldw_evicted_at", None)
                if (
                    last_used is not None
                    and evicted_ts is not None
                    and last_used > evicted_ts
                ):
                    try:
                        service._tldw_evicted_at = last_used  # type: ignore[attr-defined]
                    except (AttributeError, RuntimeError, TypeError) as exc:
                        logger.debug(
                            "Failed to update eviction timestamp for user {}: {}",
                            user_id,
                            exc,
                        )
                    if time.monotonic() >= eviction_deadline:
                        logger.warning(
                            "Audit service for user {} eviction wait exceeded ({}s); forcing stop.",
                            user_id,
                            EVICTION_MAX_WAIT_SECONDS,
                        )
                        break
                    logger.debug(
                        "Audit service for user {} reused after eviction; delaying stop.",
                        user_id,
                    )
                    continue
                break
            await service.stop()
            logger.info(f"Audit service for user {user_id} stopped ({reason}).")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.error(
                f"Failed to stop audit service for user {user_id} ({reason}): {exc}",
                exc_info=True,
            )
        finally:
            _clear_stopping_flag()

    owner_loop = getattr(service, "owner_loop", None)
    if owner_loop:
        try:
            if owner_loop.is_closed():
                owner_loop = None
            elif not owner_loop.is_running():
                logger.warning(
                    "Audit service owner loop not running; shutting down on current loop."
                )
                owner_loop = None
        except (AttributeError, RuntimeError):
            owner_loop = None

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    def _run():
        try:
            asyncio.run(_stop())
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            logger.error(
                f"Audit service stop failed for user {user_id} in fallback thread: {type(e).__name__}: {e}"
            )

    if owner_loop and owner_loop is not current_loop:
        try:
            future = asyncio.run_coroutine_threadsafe(_stop(), owner_loop)
        except RuntimeError:
            # Owner loop closed between our check and the submission; fall through
            # to the current-loop or thread-based fallback below.
            pass
        else:
            _track_scheduled_stop_future(future)

            def _log_result(fut):
                exc = fut.exception()
                if exc:
                    logger.error(
                        f"Failed to stop audit service for user {user_id} ({reason}): {exc}",
                        exc_info=True,
                    )

            future.add_done_callback(_log_result)
            return

    if current_loop:
        # In test contexts, avoid leaving background stop tasks pending on loop shutdown.
        if _is_test_context():
            threading.Thread(target=_run, name=f"audit-stop-{user_id}", daemon=True).start()
            return
        task = current_loop.create_task(_stop())
        _track_scheduled_stop_future(task)
        return

    threading.Thread(target=_run, name=f"audit-stop-{user_id}", daemon=True).start()


_services_stopping: set[int] = set()
_services_stopping_lock = threading.Lock()  # Protects _services_stopping service-id set
_scheduled_stop_futures: set[Union[concurrent.futures.Future[Any], asyncio.Task[Any]]] = set()
_scheduled_stop_lock = threading.Lock()


def _track_scheduled_stop_future(future: Union[concurrent.futures.Future[Any], asyncio.Task[Any]]) -> None:
    """Track a cross-loop audit stop future or same-loop task until completion."""
    with _scheduled_stop_lock:
        _scheduled_stop_futures.add(future)

    def _cleanup(done_future: Union[concurrent.futures.Future[Any], asyncio.Task[Any]]) -> None:
        with _scheduled_stop_lock:
            _scheduled_stop_futures.discard(done_future)

    future.add_done_callback(_cleanup)


async def _drain_scheduled_audit_stops(timeout: Optional[float] = None) -> None:
    """Wait for any tracked cross-loop audit stop futures to finish."""
    with _scheduled_stop_lock:
        pending = [future for future in _scheduled_stop_futures if not future.done()]

    if not pending:
        return

    # asyncio.Task instances are already asyncio futures; concurrent.futures.Future
    # instances need wrapping so they can be awaited on this event loop.
    awaitables = []
    for future in pending:
        if isinstance(future, asyncio.Task):
            awaitables.append(future)
        else:
            awaitables.append(asyncio.wrap_future(future))

    waiter = asyncio.gather(*awaitables, return_exceptions=True)
    if timeout is None or timeout <= 0:
        await waiter
    else:
        await asyncio.wait_for(waiter, timeout=timeout)


def _handle_cache_eviction(user_id: Optional[Union[int, str]], service: UnifiedAuditService, reason: str) -> None:
    """Handle cache eviction/removal by scheduling a stop unless manually managed."""
    if service is None:
        return
    logger.debug(f"Evicting audit service for user {user_id} ({reason}).")
    _schedule_service_stop(user_id, service, reason)


class _SmallLRUCache:
    """Minimal LRU cache with eviction callback."""

    def __init__(
        self,
        maxsize: int,
        on_evict: Callable[[Optional[Union[int, str]], UnifiedAuditService, str], None],
    ):
        self.maxsize = maxsize
        self._on_evict = on_evict
        self._data: OrderedDict[Optional[Union[int, str]], UnifiedAuditService] = OrderedDict()

    def get(self, key: Optional[Union[int, str]]) -> Optional[UnifiedAuditService]:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def __setitem__(self, key: Optional[Union[int, str]], value: UnifiedAuditService) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            evicted_key, evicted_value = self._data.popitem(last=False)
            self._on_evict(evicted_key, evicted_value, "capacity")

    def pop(
        self,
        key: Optional[Union[int, str]],
        default: Optional[UnifiedAuditService] = None,
    ) -> Optional[UnifiedAuditService]:
        if key in self._data:
            value = self._data.pop(key)
            self._on_evict(key, value, "removed")
            return value
        return default

    def pop_no_callback(
        self,
        key: Optional[Union[int, str]],
        default: Optional[UnifiedAuditService] = None,
    ) -> Optional[UnifiedAuditService]:
        """Pop without triggering eviction callback (for manual shutdown)."""
        return self._data.pop(key, default)

    def keys(self):
        return list(self._data.keys())


if _HAS_CACHETOOLS:

    class _EvictingLRUCache(LRUCache):
        """LRUCache variant that fires a callback whenever items are removed."""

        _MISSING = object()

        def __init__(
            self,
            maxsize: int,
            on_evict: Callable[[Optional[Union[int, str]], UnifiedAuditService, str], None],
        ):
            super().__init__(maxsize=maxsize)
            self._on_evict = on_evict

        def popitem(self):
            key, value = super().popitem()
            self._on_evict(key, value, "capacity")
            return key, value

        def __delitem__(self, key):
            value = self[key]
            super().__delitem__(key)
            self._on_evict(key, value, "removed")

        def pop(self, key, default=_MISSING):
            if key in self:
                value = Cache.pop(self, key)
                self._on_evict(key, value, "removed")
                return value
            if default is self._MISSING:
                raise KeyError(key)
            return default

        def pop_no_callback(self, key, default=None):
            """Pop without triggering eviction callback (for manual shutdown)."""
            if key in self:
                return Cache.pop(self, key)
            return default


# --- Configuration ---
MAX_CACHED_AUDIT_INSTANCES = _settings_int("MAX_CACHED_AUDIT_INSTANCES", 20, min_value=1, max_value=1000)
EVICTION_GRACE_SECONDS = _settings_float(
    "AUDIT_EVICTION_GRACE_SECONDS",
    2.0,
    min_value=0.0,
    max_value=300.0,
)
EVICTION_MAX_WAIT_SECONDS = _settings_float(
    "AUDIT_EVICTION_MAX_WAIT_SECONDS",
    max(EVICTION_GRACE_SECONDS * 5.0, 10.0),
    min_value=1.0,
    max_value=600.0,
)

_CACHE_IMPL = "cachetools.LRUCache" if _HAS_CACHETOOLS else "_SmallLRUCache"
logger.info(
    f"Using {_CACHE_IMPL} for audit service instances (maxsize={MAX_CACHED_AUDIT_INSTANCES})."
)


def _build_cache() -> Any:
    """Construct a cache instance for a single event loop."""
    if _HAS_CACHETOOLS:
        return _EvictingLRUCache(
            maxsize=MAX_CACHED_AUDIT_INSTANCES,
            on_evict=_handle_cache_eviction,
        )
    return _SmallLRUCache(
        MAX_CACHED_AUDIT_INSTANCES,
        on_evict=_handle_cache_eviction,
    )


@dataclass
class _LoopState:
    """Per-event-loop cache and initialization state."""
    cache: Any
    loop: Optional[asyncio.AbstractEventLoop] = None
    cache_lock: threading.Lock = field(default_factory=threading.Lock)
    init_lock: threading.Lock = field(default_factory=threading.Lock)
    initializing_users: set[Optional[Union[int, str]]] = field(default_factory=set)
    initializing_events: dict[Optional[Union[int, str]], asyncio.Event] = field(default_factory=dict)
    shutting_down_keys: set[Optional[Union[int, str]]] = field(default_factory=set)


_STATE_LOCK = threading.Lock()
_STATE_BY_LOOP: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _LoopState]" = weakref.WeakKeyDictionary()
_GLOBAL_AUDIT_SHUTDOWN_LOCK = threading.Lock()
_GLOBAL_AUDIT_SHUTDOWN_COUNT = 0


def _state_for_loop() -> _LoopState:
    """Return (or create) per-loop state for audit services."""
    loop = asyncio.get_running_loop()
    with _STATE_LOCK:
        state = _STATE_BY_LOOP.get(loop)
        if state is None:
            state = _LoopState(loop=loop, cache=_build_cache())
            _STATE_BY_LOOP[loop] = state
        return state


def _all_loop_states() -> list[_LoopState]:
    """Snapshot all known loop states."""
    with _STATE_LOCK:
        return list(_STATE_BY_LOOP.values())


def _signal_initializing_event(state: _LoopState, event: asyncio.Event) -> None:
    """Wake audit initialization waiters on the event loop that owns the event."""
    owner_loop = state.loop
    if owner_loop is None:
        event.set()
        return

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if owner_loop is current_loop:
        event.set()
        return

    try:
        if owner_loop.is_closed():
            logger.debug("Skipping audit init waiter wake-up because owner loop is closed.")
            return
        owner_loop.call_soon_threadsafe(event.set)
    except RuntimeError as exc:
        logger.debug("Failed to signal audit init waiter on owner loop: {}", exc)


def _begin_global_audit_shutdown() -> None:
    global _GLOBAL_AUDIT_SHUTDOWN_COUNT
    with _GLOBAL_AUDIT_SHUTDOWN_LOCK:
        _GLOBAL_AUDIT_SHUTDOWN_COUNT += 1


def _end_global_audit_shutdown() -> None:
    global _GLOBAL_AUDIT_SHUTDOWN_COUNT
    with _GLOBAL_AUDIT_SHUTDOWN_LOCK:
        _GLOBAL_AUDIT_SHUTDOWN_COUNT = max(0, _GLOBAL_AUDIT_SHUTDOWN_COUNT - 1)


def _is_global_audit_shutdown_in_progress() -> bool:
    with _GLOBAL_AUDIT_SHUTDOWN_LOCK:
        return _GLOBAL_AUDIT_SHUTDOWN_COUNT > 0


def _raise_if_global_audit_shutdown_in_progress(user_id: Optional[Union[int, str]]) -> None:
    if _is_global_audit_shutdown_in_progress():
        msg = f"Audit service initialization aborted during global shutdown for user {user_id}"
        logger.warning(msg)
        raise ServiceInitializationError(msg)

#######################################################################################################################

# --- Helper Functions ---

async def _create_audit_service_for_user(user_id: Union[int, str]) -> UnifiedAuditService:
    """
    Create a new audit service instance for a specific user.

    Args:
        user_id: The user's ID

    Returns:
        Initialized UnifiedAuditService instance
    """
    storage_mode = _resolve_audit_storage_mode()
    if storage_mode == "shared":
        db_path = _shared_audit_db_path()
        logger.info(f"Creating shared audit service for user {user_id} at path: {db_path}")
    else:
        db_path = DatabasePaths.get_audit_db_path(user_id)
        logger.info(f"Creating audit service for user {user_id} at path: {db_path}")

    # Create the service with user-specific database
    service = UnifiedAuditService(
        db_path=str(db_path),
        storage_mode=storage_mode,
        retention_days=_settings_int("AUDIT_RETENTION_DAYS", 30, min_value=1, max_value=3650),
        enable_pii_detection=_settings_bool("AUDIT_ENABLE_PII_DETECTION", True),
        enable_risk_scoring=_settings_bool("AUDIT_ENABLE_RISK_SCORING", True),
        buffer_size=_settings_int("AUDIT_BUFFER_SIZE", 100, min_value=1, max_value=100000),
        flush_interval=_settings_float("AUDIT_FLUSH_INTERVAL", 5.0, min_value=0.1, max_value=3600.0),
    )

    # Initialize the service (creates database, starts background tasks)
    await service.initialize()

    logger.info(f"Audit service initialized successfully for user {user_id}")
    return service


async def _create_default_audit_service() -> UnifiedAuditService:
    """Create a shared default audit service (no user-specific DB path)."""
    storage_mode = _resolve_audit_storage_mode()
    db_path: Optional[str]
    if storage_mode == "shared":
        db_path = _shared_audit_db_path()
        logger.info(f"Creating shared default audit service at path: {db_path}")
    else:
        db_path = None
        logger.info("Creating default audit service.")
    service = UnifiedAuditService(
        db_path=db_path,
        storage_mode=storage_mode,
        retention_days=_settings_int("AUDIT_RETENTION_DAYS", 30, min_value=1, max_value=3650),
        enable_pii_detection=_settings_bool("AUDIT_ENABLE_PII_DETECTION", True),
        enable_risk_scoring=_settings_bool("AUDIT_ENABLE_RISK_SCORING", True),
        buffer_size=_settings_int("AUDIT_BUFFER_SIZE", 100, min_value=1, max_value=100000),
        flush_interval=_settings_float("AUDIT_FLUSH_INTERVAL", 5.0, min_value=0.1, max_value=3600.0),
    )
    await service.initialize()
    logger.info("Default audit service initialized successfully.")
    return service


async def _get_or_create_audit_service_for_key(user_id: Optional[Union[int, str]]) -> UnifiedAuditService:
    """Internal helper to get or create a cached audit service for a cache key."""
    storage_mode = _resolve_audit_storage_mode()
    cache_key = None if storage_mode == "shared" else user_id
    _raise_if_global_audit_shutdown_in_progress(user_id)
    state = _state_for_loop()
    service_instance: Optional[UnifiedAuditService] = None

    # Check cache
    with state.cache_lock:
        service_instance = state.cache.get(cache_key)

    if service_instance:
        _mark_service_used(service_instance)
        logger.debug(f"Using cached audit service instance for user_id: {user_id}")
        return service_instance

    key_label = "shared" if storage_mode == "shared" else "default" if user_id is None else f"user_id {user_id}"
    logger.info(f"No cached audit service found for {key_label}. Initializing.")

    init_timeout_s = _settings_float("AUDIT_INIT_TIMEOUT_SECONDS", 30.0, min_value=0.1, max_value=300.0)
    deadline = time.monotonic() + init_timeout_s

    # Check if another task is already initializing this service
    wait_event: Optional[asyncio.Event] = None
    should_initialize = False

    while True:
        with state.init_lock:
            _raise_if_global_audit_shutdown_in_progress(user_id)
            if cache_key in state.shutting_down_keys:
                msg = f"Audit service initialization aborted during shutdown for user {user_id}"
                logger.warning(msg)
                raise ServiceInitializationError(msg)
            if cache_key in state.initializing_users:
                if cache_key not in state.initializing_events:
                    state.initializing_events[cache_key] = asyncio.Event()
                wait_event = state.initializing_events[cache_key]
                should_initialize = False
            else:
                state.initializing_users.add(cache_key)
                should_initialize = True
                wait_event = None

        if should_initialize:
            break

        if wait_event is None:
            msg = f"Missing wait event for audit service initialization (user {user_id})"
            logger.warning(msg)
            raise ServiceInitializationError(msg)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            msg = f"Timeout waiting for audit service initialization for user {user_id}"
            logger.warning(msg)
            raise ServiceInitializationTimeoutError(msg)

        try:
            await asyncio.wait_for(wait_event.wait(), timeout=remaining)
        except asyncio.TimeoutError as exc:
            msg = f"Timeout waiting for audit service initialization for user {user_id}"
            logger.warning(msg)
            raise ServiceInitializationTimeoutError(msg) from exc

        # Check cache again after waiting
        with state.cache_lock:
            service_instance = state.cache.get(cache_key)
            if service_instance:
                _mark_service_used(service_instance)
                return service_instance

    if should_initialize:
        try:
            # Double-check cache in case another request created it
            with state.cache_lock:
                service_instance = state.cache.get(cache_key)
                if service_instance:
                    _mark_service_used(service_instance)
                    logger.debug(f"Audit service for user {user_id} created concurrently.")
                    return service_instance

            if storage_mode == "shared" or user_id is None:
                service_instance = await _create_default_audit_service()
            else:
                service_instance = await _create_audit_service_for_user(user_id)

            # Store in cache
            with state.init_lock:
                _raise_if_global_audit_shutdown_in_progress(user_id)
                if cache_key in state.shutting_down_keys:
                    msg = f"Audit service initialization aborted during shutdown for user {user_id}"
                    logger.warning(msg)
                    raise ServiceInitializationError(msg)
            with state.cache_lock:
                state.cache[cache_key] = service_instance

            _mark_service_used(service_instance)
            logger.info(f"Audit service created and cached successfully for user {user_id}")

        except Exception as e:
            logger.error(f"Failed to initialize audit service for user {user_id}: {e}", exc_info=True)
            raise
        finally:
            # Clean up initialization tracking and signal waiters
            with state.init_lock:
                state.initializing_users.discard(cache_key)
                event = state.initializing_events.pop(cache_key, None)
                if event:
                    _signal_initializing_event(state, event)
                should_clear_shutdown_key = (
                    cache_key in state.shutting_down_keys
                    and cache_key not in state.initializing_users
                )
            if should_clear_shutdown_key:
                with state.cache_lock:
                    has_cached_service = cache_key in state.cache
                if not has_cached_service:
                    with state.init_lock:
                        if cache_key not in state.initializing_users:
                            state.shutting_down_keys.discard(cache_key)

    if service_instance is None:
        # Defensive: should not happen, but avoid returning None.
        raise ServiceInitializationError(f"Could not initialize audit service for user {user_id}")

    return service_instance


async def get_or_create_audit_service_for_user_id(user_id: int) -> UnifiedAuditService:
    """Get (or create) a cached UnifiedAuditService for a concrete user_id.

    This is the core implementation behind the FastAPI dependency
    `get_audit_service_for_user`, but is also usable from non-DI code paths
    where you already have a user id (e.g., post-authentication flows).
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an int")
    return await _get_or_create_audit_service_for_key(user_id)


async def get_or_create_default_audit_service() -> UnifiedAuditService:
    """Return a shared default audit service (used when no user_id is available)."""
    return await _get_or_create_audit_service_for_key(None)


async def get_or_create_audit_service_for_user_id_optional(
    user_id: Optional[Union[int, str]]
) -> UnifiedAuditService:
    """Get an audit service for a user id or fall back to the default service."""
    storage_mode = _resolve_audit_storage_mode()
    if storage_mode == "shared":
        # Shared mode uses a single DB; accept any user_id (including non-numeric).
        return await get_or_create_default_audit_service()
    if user_id is None:
        return await get_or_create_default_audit_service()
    try:
        if isinstance(user_id, bool):
            raise TypeError("bool is not a valid user id")
        uid_int = int(user_id)
    except (TypeError, ValueError):
        raw_id = str(user_id).strip().lower()
        if raw_id in {"system", "unidentified_user"}:
            logger.warning("Non-numeric system user_id {!r}; routing to default audit service.", user_id)
            return await get_or_create_default_audit_service()
        if _is_test_context():
            # In tests we allow non-numeric ids to map to per-user storage paths.
            return await _get_or_create_audit_service_for_key(str(user_id))
        msg = f"Invalid non-numeric user_id {user_id!r} for audit service"
        logger.error(msg)
        raise ServiceInitializationError(msg) from None
    return await get_or_create_audit_service_for_user_id(uid_int)

# --- Main Dependency Function ---

async def get_audit_service_for_user(
    current_user: User = Depends(get_request_user)
) -> UnifiedAuditService:
    """
    FastAPI dependency to get the UnifiedAuditService instance for the identified user.

    Handles caching, initialization, and lifecycle management.
    Uses configuration values from the 'settings' dictionary.

    Args:
        current_user: The User object provided by `get_request_user`.

    Returns:
        A UnifiedAuditService instance for the user.

    Raises:
        HTTPException: If the service cannot be initialized.
    """
    if current_user is None:
        logger.error("get_audit_service_for_user called without a valid User object.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User identification failed for audit service.",
        )

    try:
        raw_id: Optional[Union[int, str]] = getattr(current_user, "id_int", None)
        if raw_id is None:
            raw_id = getattr(current_user, "id", None)
        return await get_or_create_audit_service_for_user_id_optional(raw_id)
    except Exception as e:
        logger.error(
            "Failed to initialize audit service for user {}: {}",
            getattr(current_user, "id", None),
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not initialize audit service.",
        ) from e

# --- Cleanup Functions ---

async def shutdown_user_audit_service(user_id: int) -> AuditShutdownSummary:
    """
    Shutdown audit service for a specific user.

    Args:
        user_id: The user's ID
    """
    services: list[UnifiedAuditService] = []
    errors: list[str] = []
    requested = 0
    stopped = 0

    cache_keys = _shutdown_cache_keys(user_id)
    states = list(_all_loop_states())
    try:
        for state in states:
            waiters: list[asyncio.Event] = []
            with state.init_lock:
                for cache_key in cache_keys:
                    state.shutting_down_keys.add(cache_key)
                    event = state.initializing_events.get(cache_key)
                    if event is not None:
                        waiters.append(event)
            for event in waiters:
                _signal_initializing_event(state, event)

            with state.cache_lock:
                for cache_key in cache_keys:
                    existing = state.cache.get(cache_key)
                    if existing:
                        pop_no_cb = getattr(state.cache, "pop_no_callback", None)
                        service = pop_no_cb(cache_key, None) if callable(pop_no_cb) else state.cache.pop(cache_key, None)
                        if service:
                            services.append(service)

        if not services:
            return AuditShutdownSummary()

        requested = len(services)
        for service in services:
            stopped_ok, _, error_message, _exc = await _stop_audit_service_instance(
                service,
                label=f"user {user_id}",
            )
            if stopped_ok:
                stopped += 1
                logger.info(f"Shut down audit service for user {user_id}")
            elif error_message:
                errors.append(error_message)

        return AuditShutdownSummary(
            requested=requested,
            stopped=stopped,
            timeout_count=0,
            error_count=len(errors),
            errors=tuple(errors),
        )
    finally:
        for state in states:
            with state.init_lock:
                for cache_key in cache_keys:
                    if cache_key not in state.initializing_users:
                        state.shutting_down_keys.discard(cache_key)
                        state.initializing_events.pop(cache_key, None)


async def shutdown_all_audit_services(*, raise_on_error: bool = True) -> AuditShutdownSummary:
    """
    Shutdown all cached audit service instances.
    Useful for application shutdown.
    """
    services: list[UnifiedAuditService] = []
    total_instances = 0
    shutdown_timeout_s = _settings_float(
        "AUDIT_SHUTDOWN_TIMEOUT_SECONDS",
        10.0,
        min_value=0.0,
        max_value=300.0,
    )
    if shutdown_timeout_s <= 0:
        shutdown_timeout_s = 0.0
    timeout_count = 0
    error_count = 0
    stopped_count = 0
    errors: list[str] = []
    first_exception: BaseException | None = None

    _begin_global_audit_shutdown()
    states = list(_all_loop_states())
    per_state_shutdown_keys: list[tuple[_LoopState, set[Optional[Union[int, str]]]]] = []
    try:
        for state in states:
            waiters: list[asyncio.Event] = []
            with state.init_lock:
                shutdown_keys = set(state.initializing_users) | set(state.initializing_events.keys())
            with state.cache_lock:
                cache_keys = list(state.cache.keys())
                total_instances += len(cache_keys)
                shutdown_keys.update(cache_keys)
                for key in cache_keys:
                    pop_no_cb = getattr(state.cache, "pop_no_callback", None)
                    service = pop_no_cb(key, None) if callable(pop_no_cb) else state.cache.pop(key, None)
                    if service:
                        services.append(service)
            with state.init_lock:
                state.shutting_down_keys.update(shutdown_keys)
                for key in shutdown_keys:
                    event = state.initializing_events.get(key)
                    if event is not None:
                        waiters.append(event)
            for event in waiters:
                _signal_initializing_event(state, event)
            per_state_shutdown_keys.append((state, shutdown_keys))

        logger.info(f"Shutting down audit services for {total_instances} instances...")

        def _service_label(service: UnifiedAuditService) -> str:
            db_path = getattr(service, "db_path", None)
            storage_mode = getattr(service, "storage_mode", None)
            return f"id={id(service)} db_path={db_path} storage_mode={storage_mode}"

        if services:
            stop_tasks = [
                asyncio.create_task(
                    _stop_audit_service_instance(
                        service,
                        label=_service_label(service),
                        timeout_s=shutdown_timeout_s if shutdown_timeout_s > 0 else None,
                    )
                )
                for service in services
            ]
            stop_results = await asyncio.gather(*stop_tasks)
            for stopped_ok, timeout_hit, error_message, exc in stop_results:
                if stopped_ok:
                    stopped_count += 1
                elif timeout_hit:
                    timeout_count += 1
                    if error_message:
                        errors.append(error_message)
                    if first_exception is None:
                        first_exception = exc
                else:
                    error_count += 1
                    if error_message:
                        errors.append(error_message)
                    if first_exception is None:
                        first_exception = exc

        try:
            await _drain_scheduled_audit_stops(timeout=shutdown_timeout_s if shutdown_timeout_s > 0 else None)
        except asyncio.TimeoutError as exc:
            timeout_count += 1
            drain_message = f"TimeoutError: scheduled audit stop drain timed out after {shutdown_timeout_s:.2f}s"
            errors.append(drain_message)
            if first_exception is None:
                first_exception = exc
            logger.error(f"Scheduled audit stop drain timed out after {shutdown_timeout_s:.2f}s")

        summary = AuditShutdownSummary(
            requested=total_instances,
            stopped=stopped_count,
            timeout_count=timeout_count,
            error_count=error_count,
            errors=tuple(errors),
        )

        if timeout_count or error_count:
            logger.warning(
                f"Audit services shutdown completed with issues (timeouts={timeout_count}, errors={error_count})."
            )
        else:
            logger.info("All audit services shut down successfully.")

        return _finalize_shutdown_summary(
            summary=summary,
            raise_on_error=raise_on_error,
            message="Audit services shutdown completed with issues",
            first_exception=first_exception,
        )
    finally:
        for state, shutdown_keys in per_state_shutdown_keys:
            with state.init_lock:
                for key in shutdown_keys:
                    if key not in state.initializing_users:
                        state.shutting_down_keys.discard(key)
                        state.initializing_events.pop(key, None)
        _end_global_audit_shutdown()

# Example of how to register for shutdown event in FastAPI:
# from fastapi import FastAPI
# app = FastAPI()
# @app.on_event("shutdown")
# async def shutdown_event():
#     await shutdown_all_audit_services()

#
# End of Audit_DB_Deps.py
########################################################################################################################
