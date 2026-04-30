# tldw_Server_API/app/api/v1/API_Deps/kanban_deps.py
"""
FastAPI dependency injection for Kanban database access.

Provides per-user KanbanDB instances with caching for performance.
"""
import asyncio
import os
import sqlite3
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from cachetools import LRUCache
from fastapi import Depends, HTTPException, status
from loguru import logger

from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.Kanban_DB import (
    ConflictError,
    InputError,
    KanbanDB,
    KanbanDBError,
    NotFoundError,
)
from tldw_Server_API.app.core.testing import (
    is_explicit_pytest_runtime as _is_explicit_pytest_runtime,
)
from tldw_Server_API.app.core.testing import is_test_mode as _is_test_mode

# --- Configuration ---
_KANBAN_EXECUTOR: Optional[ThreadPoolExecutor] = None
_KANBAN_EXECUTOR_SHUTDOWN: bool = False
_KANBAN_EXECUTOR_LOCK = threading.Lock()
_KANBAN_EXECUTOR_MAX_WORKERS = max(1, int(settings.get("KANBAN_EXECUTOR_MAX_WORKERS", "4")))
_KANBAN_INIT_TIMEOUT_SECS = float(settings.get("KANBAN_INIT_TIMEOUT_SECS", "5"))
_KANBAN_INSTANCE_HEALTHCHECK_TTL_SECS = max(
    0.0, float(settings.get("KANBAN_INSTANCE_HEALTHCHECK_TTL_SECS", "30"))
)
_KANBAN_INSTANCE_HEALTHCHECK_TIMEOUT_SECS = max(
    0.1, float(settings.get("KANBAN_INSTANCE_HEALTHCHECK_TIMEOUT_SECS", "1.0"))
)
_KANBAN_HEALTH_DEGRADED_WINDOW_SECS = max(
    0.0, float(settings.get("KANBAN_HEALTH_DEGRADED_WINDOW_SECS", "300"))
)
_KANBAN_HEALTH_MAX_RECENT_FAILURES = max(
    1, int(settings.get("KANBAN_HEALTH_MAX_RECENT_FAILURES", "100"))
)

# --- Global Cache for KanbanDB Instances ---
MAX_CACHED_KANBAN_DB_INSTANCES = int(settings.get("MAX_CACHED_KANBAN_DB_INSTANCES", "50"))
_kanban_db_instances: LRUCache = LRUCache(maxsize=MAX_CACHED_KANBAN_DB_INSTANCES)
_kanban_db_health_checks: dict[str, float] = {}
_kanban_db_lock = threading.Lock()

# --- Health Tracking ---
_KANBAN_HEALTH_LOCK = threading.Lock()
_KANBAN_HEALTH: dict[str, Any] = {
    "init_attempts": 0,
    "init_failures": 0,
    "last_init_ms": None,
    "last_error": None,
    "last_success_ts": None,
    "last_failure_ts": None,
    "cached_instances": 0,
}
_KANBAN_RECENT_INIT_FAILURES = deque(maxlen=_KANBAN_HEALTH_MAX_RECENT_FAILURES)


def _safe_kanban_health_error(error: Optional[Exception]) -> str:
    return type(error).__name__ if error else "unknown error"


def _get_kanban_executor() -> ThreadPoolExecutor:
    """
    Return a live executor for Kanban DB work.

    Recreates the executor if it has been shut down.
    """
    global _KANBAN_EXECUTOR, _KANBAN_EXECUTOR_SHUTDOWN
    with _KANBAN_EXECUTOR_LOCK:
        if _KANBAN_EXECUTOR is None or _KANBAN_EXECUTOR_SHUTDOWN:
            _KANBAN_EXECUTOR = ThreadPoolExecutor(
                max_workers=_KANBAN_EXECUTOR_MAX_WORKERS,
                thread_name_prefix="kanban-db",
            )
            _KANBAN_EXECUTOR_SHUTDOWN = False
        return _KANBAN_EXECUTOR


def _record_init(duration_ms: float, success: bool, error: Optional[Exception] = None) -> None:
    """Record initialization metrics."""
    now_ts = time.time()
    with _KANBAN_HEALTH_LOCK:
        _KANBAN_HEALTH["init_attempts"] += 1
        _KANBAN_HEALTH["last_init_ms"] = duration_ms
        _KANBAN_HEALTH["cached_instances"] = len(_kanban_db_instances)
        if success:
            _KANBAN_HEALTH["last_error"] = None
            _KANBAN_HEALTH["last_success_ts"] = now_ts
        else:
            _KANBAN_HEALTH["init_failures"] += 1
            _KANBAN_HEALTH["last_error"] = _safe_kanban_health_error(error)
            _KANBAN_HEALTH["last_failure_ts"] = now_ts
            _KANBAN_RECENT_INIT_FAILURES.append(now_ts)


def get_kanban_health_snapshot() -> dict[str, Any]:
    """Get current health metrics snapshot."""
    with _KANBAN_HEALTH_LOCK:
        now_ts = time.time()
        if _KANBAN_HEALTH_DEGRADED_WINDOW_SECS:
            while (
                _KANBAN_RECENT_INIT_FAILURES
                and (now_ts - _KANBAN_RECENT_INIT_FAILURES[0]) > _KANBAN_HEALTH_DEGRADED_WINDOW_SECS
            ):
                _KANBAN_RECENT_INIT_FAILURES.popleft()
        recent_failures = len(_KANBAN_RECENT_INIT_FAILURES)
        return {
            "status": "degraded" if recent_failures else "healthy",
            "init_attempts": _KANBAN_HEALTH.get("init_attempts"),
            "init_failures": _KANBAN_HEALTH.get("init_failures"),
            "recent_init_failures": recent_failures,
            "recent_failure_window_secs": _KANBAN_HEALTH_DEGRADED_WINDOW_SECS,
            "last_init_ms": _KANBAN_HEALTH.get("last_init_ms"),
            "last_error": _KANBAN_HEALTH.get("last_error"),
            "last_success_ts": _KANBAN_HEALTH.get("last_success_ts"),
            "last_failure_ts": _KANBAN_HEALTH.get("last_failure_ts"),
            "cached_instances": len(_kanban_db_instances),
        }


def _create_kanban_db(user_id: int) -> KanbanDB:
    """
    Create a KanbanDB instance for a user.

    This runs in the executor thread pool.
    """
    db_path = DatabasePaths.get_kanban_db_path(user_id)
    logger.info(f"Initializing KanbanDB instance for user {user_id} at path: {db_path}")
    db_instance = KanbanDB(db_path=str(db_path), user_id=str(user_id))
    return db_instance


def _map_kanban_init_db_error(exc: Exception) -> HTTPException:
    return map_db_error_to_http(
        exc,
        default_detail="Kanban DB unavailable",
        input_status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        conflict_status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        conflict_detail="Kanban DB unavailable",
    )


def _health_check_instance(db_instance: KanbanDB) -> bool:
    """Quick health check for a cached instance."""
    try:
        conn = sqlite3.connect(db_instance.db_path, timeout=0.1, isolation_level=None)
        try:
            cur = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='kanban_boards'",
            )
            if cur.fetchone() is None:
                return False
        finally:
            conn.close()
        return True
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower():
            return True
        logger.warning("Kanban health probe failed")
        return False
    except Exception:
        logger.warning("Kanban health probe failed")
        return False


async def _is_instance_healthy(db_instance: KanbanDB) -> bool:
    """Async wrapper for health check."""
    try:
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(_get_kanban_executor(), _health_check_instance, db_instance),
            timeout=_KANBAN_INSTANCE_HEALTHCHECK_TIMEOUT_SECS,
        )
        return bool(result)
    except Exception:
        return False


async def _get_or_init_db_instance(user_id: int) -> KanbanDB:
    """
    Get or create a KanbanDB instance for a user.

    Uses caching with health checks to avoid returning stale instances.
    """
    cache_key = f"kanban::{user_id}"

    # Check cache first
    with _kanban_db_lock:
        db_instance = _kanban_db_instances.get(cache_key)
        last_health_check = _kanban_db_health_checks.get(cache_key)
        should_check_health = (
            last_health_check is None
            or (time.monotonic() - last_health_check) >= _KANBAN_INSTANCE_HEALTHCHECK_TTL_SECS
        )

    if db_instance:
        if not should_check_health:
            return db_instance
        if await _is_instance_healthy(db_instance):
            with _kanban_db_lock:
                _kanban_db_health_checks[cache_key] = time.monotonic()
            return db_instance
        logger.warning(f"Kanban cached instance unhealthy for user {user_id}; evicting and rebuilding.")
        with _kanban_db_lock:
            if _kanban_db_instances.get(cache_key) is db_instance:
                _kanban_db_instances.pop(cache_key, None)
                _kanban_db_health_checks.pop(cache_key, None)

    # Create new instance
    loop = asyncio.get_running_loop()
    start = time.perf_counter()
    try:
        db_instance = await asyncio.wait_for(
            loop.run_in_executor(_get_kanban_executor(), _create_kanban_db, user_id),
            timeout=_KANBAN_INIT_TIMEOUT_SECS,
        )
        duration_ms = (time.perf_counter() - start) * 1000
        _record_init(duration_ms, True)
        logger.debug(f"KanbanDB initialized for user {user_id} in {duration_ms:.2f}ms")
    except asyncio.TimeoutError as e:
        _record_init(_KANBAN_INIT_TIMEOUT_SECS * 1000, False, e)
        logger.error(f"Kanban DB initialization timed out for user {user_id}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Kanban database initialization timed out",
        ) from e
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        _record_init(duration_ms, False, e)
        logger.error(f"Kanban DB initialization failed for user {user_id}")
        if isinstance(e, (InputError, ConflictError)) or (
            isinstance(e, KanbanDBError) and not isinstance(e, NotFoundError)
        ):
            raise _map_kanban_init_db_error(e) from e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not initialize Kanban database for user",
        ) from e

    # Cache the instance
    with _kanban_db_lock:
        _kanban_db_instances[cache_key] = db_instance
        _kanban_db_health_checks[cache_key] = time.monotonic()
        _KANBAN_HEALTH["cached_instances"] = len(_kanban_db_instances)

    return db_instance


async def get_kanban_db_for_user(current_user: User = Depends(get_request_user)) -> KanbanDB:
    """
    FastAPI dependency to get the KanbanDB instance for the authenticated user.

    Handles caching and health checks; initialization runs in a dedicated executor.

    Args:
        current_user: The authenticated user from the request.

    Returns:
        KanbanDB instance for the user.

    Raises:
        HTTPException: If user identification fails or DB initialization fails.
    """
    if not current_user or not isinstance(current_user.id, int):
        logger.error("get_kanban_db_for_user called without a valid User object or user.id is not int.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User identification failed for Kanban DB."
        )

    user_id = current_user.id
    return await _get_or_init_db_instance(user_id)


def close_all_kanban_db_instances() -> None:
    """
    Close all cached KanbanDB instances.

    Should be called during application shutdown.
    """
    with _kanban_db_lock:
        cached_items = list(_kanban_db_instances.items())
        logger.info(f"Closing all cached KanbanDB instances ({len(cached_items)})...")
        _kanban_db_instances.clear()
        _kanban_db_health_checks.clear()

    for cache_key, db_instance in cached_items:
        close_method = getattr(db_instance, "close", None)
        if callable(close_method):
            try:
                close_method()
            except Exception:
                logger.debug(f"Error closing KanbanDB instance {cache_key}")

    logger.info("All KanbanDB instances closed and cache cleared.")


def shutdown_kanban_executor(wait: bool = False) -> None:
    """
    Shut down the Kanban executor to avoid lingering threads on shutdown.

    Args:
        wait: If True, block until all pending futures complete.
    """
    global _KANBAN_EXECUTOR, _KANBAN_EXECUTOR_SHUTDOWN
    with _KANBAN_EXECUTOR_LOCK:
        executor = _KANBAN_EXECUTOR
        _KANBAN_EXECUTOR = None
        _KANBAN_EXECUTOR_SHUTDOWN = True
    if executor is None:
        return
    try:
        executor.shutdown(wait=wait, cancel_futures=True)
        logger.info("Kanban executor shut down successfully.")
    except Exception:
        logger.debug("Kanban executor shutdown error")


# --- Exception Handlers (for use in endpoints) ---

def handle_kanban_db_error(e: Exception) -> HTTPException:
    """
    Convert KanbanDB exceptions to appropriate HTTP responses.

    Args:
        e: The exception to handle.

    Returns:
        HTTPException with appropriate status code.
    """
    if isinstance(e, (NotFoundError, InputError, ConflictError, KanbanDBError)):
        return map_db_error_to_http(e, default_detail="Kanban operation failed")

    logger.error("Unexpected error in Kanban operation")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An unexpected error occurred"
    )


# =============================================================================
# Rate Limiting Infrastructure (Phase 7)
# =============================================================================

# Rate limiting configuration - default limits per minute
KANBAN_RATE_LIMITS: dict[str, int] = {
    # Board operations
    "kanban.boards.create": int(os.getenv("KANBAN_RATE_LIMIT_BOARDS_CREATE", "30")),
    "kanban.boards.list": int(os.getenv("KANBAN_RATE_LIMIT_BOARDS_LIST", "120")),
    "kanban.boards.get": int(os.getenv("KANBAN_RATE_LIMIT_BOARDS_GET", "120")),
    "kanban.boards.update": int(os.getenv("KANBAN_RATE_LIMIT_BOARDS_UPDATE", "60")),
    "kanban.boards.delete": int(os.getenv("KANBAN_RATE_LIMIT_BOARDS_DELETE", "30")),
    "kanban.boards.archive": int(os.getenv("KANBAN_RATE_LIMIT_BOARDS_ARCHIVE", "60")),
    "kanban.boards.export": int(os.getenv("KANBAN_RATE_LIMIT_BOARDS_EXPORT", "10")),
    "kanban.boards.import": int(os.getenv("KANBAN_RATE_LIMIT_BOARDS_IMPORT", "10")),

    # List operations
    "kanban.lists.create": int(os.getenv("KANBAN_RATE_LIMIT_LISTS_CREATE", "60")),
    "kanban.lists.list": int(os.getenv("KANBAN_RATE_LIMIT_LISTS_LIST", "120")),
    "kanban.lists.get": int(os.getenv("KANBAN_RATE_LIMIT_LISTS_GET", "120")),
    "kanban.lists.update": int(os.getenv("KANBAN_RATE_LIMIT_LISTS_UPDATE", "60")),
    "kanban.lists.delete": int(os.getenv("KANBAN_RATE_LIMIT_LISTS_DELETE", "60")),
    "kanban.lists.reorder": int(os.getenv("KANBAN_RATE_LIMIT_LISTS_REORDER", "60")),

    # Card operations
    "kanban.cards.create": int(os.getenv("KANBAN_RATE_LIMIT_CARDS_CREATE", "120")),
    "kanban.cards.list": int(os.getenv("KANBAN_RATE_LIMIT_CARDS_LIST", "200")),
    "kanban.cards.get": int(os.getenv("KANBAN_RATE_LIMIT_CARDS_GET", "200")),
    "kanban.cards.update": int(os.getenv("KANBAN_RATE_LIMIT_CARDS_UPDATE", "120")),
    "kanban.cards.delete": int(os.getenv("KANBAN_RATE_LIMIT_CARDS_DELETE", "60")),
    "kanban.cards.move": int(os.getenv("KANBAN_RATE_LIMIT_CARDS_MOVE", "120")),
    "kanban.cards.copy": int(os.getenv("KANBAN_RATE_LIMIT_CARDS_COPY", "30")),
    "kanban.cards.reorder": int(os.getenv("KANBAN_RATE_LIMIT_CARDS_REORDER", "120")),
    "kanban.cards.bulk": int(os.getenv("KANBAN_RATE_LIMIT_CARDS_BULK", "30")),
    "kanban.cards.filter": int(os.getenv("KANBAN_RATE_LIMIT_CARDS_FILTER", "60")),

    # Search operations
    "kanban.search": int(os.getenv("KANBAN_RATE_LIMIT_SEARCH", "60")),

    # Card links operations
    "kanban.links.create": int(os.getenv("KANBAN_RATE_LIMIT_LINKS_CREATE", "120")),
    "kanban.links.list": int(os.getenv("KANBAN_RATE_LIMIT_LINKS_LIST", "200")),
    "kanban.links.delete": int(os.getenv("KANBAN_RATE_LIMIT_LINKS_DELETE", "60")),
    "kanban.links.bulk": int(os.getenv("KANBAN_RATE_LIMIT_LINKS_BULK", "30")),
    "kanban.links.lookup": int(os.getenv("KANBAN_RATE_LIMIT_LINKS_LOOKUP", "120")),

    # Comment operations
    "kanban.comments.create": int(os.getenv("KANBAN_RATE_LIMIT_COMMENTS_CREATE", "120")),
    "kanban.comments.list": int(os.getenv("KANBAN_RATE_LIMIT_COMMENTS_LIST", "200")),
    "kanban.comments.get": int(os.getenv("KANBAN_RATE_LIMIT_COMMENTS_GET", "200")),
    "kanban.comments.update": int(os.getenv("KANBAN_RATE_LIMIT_COMMENTS_UPDATE", "120")),
    "kanban.comments.delete": int(os.getenv("KANBAN_RATE_LIMIT_COMMENTS_DELETE", "60")),

    # Label operations
    "kanban.labels.create": int(os.getenv("KANBAN_RATE_LIMIT_LABELS_CREATE", "60")),
    "kanban.labels.list": int(os.getenv("KANBAN_RATE_LIMIT_LABELS_LIST", "200")),
    "kanban.labels.get": int(os.getenv("KANBAN_RATE_LIMIT_LABELS_GET", "200")),
    "kanban.labels.update": int(os.getenv("KANBAN_RATE_LIMIT_LABELS_UPDATE", "60")),
    "kanban.labels.delete": int(os.getenv("KANBAN_RATE_LIMIT_LABELS_DELETE", "60")),
    "kanban.labels.assign": int(os.getenv("KANBAN_RATE_LIMIT_LABELS_ASSIGN", "120")),
    "kanban.labels.remove": int(os.getenv("KANBAN_RATE_LIMIT_LABELS_REMOVE", "120")),

    # Checklist operations
    "kanban.checklists.create": int(os.getenv("KANBAN_RATE_LIMIT_CHECKLISTS_CREATE", "120")),
    "kanban.checklists.list": int(os.getenv("KANBAN_RATE_LIMIT_CHECKLISTS_LIST", "200")),
    "kanban.checklists.get": int(os.getenv("KANBAN_RATE_LIMIT_CHECKLISTS_GET", "200")),
    "kanban.checklists.update": int(os.getenv("KANBAN_RATE_LIMIT_CHECKLISTS_UPDATE", "120")),
    "kanban.checklists.delete": int(os.getenv("KANBAN_RATE_LIMIT_CHECKLISTS_DELETE", "60")),
    "kanban.checklists.reorder": int(os.getenv("KANBAN_RATE_LIMIT_CHECKLISTS_REORDER", "120")),
    "kanban.checklist_items.create": int(os.getenv("KANBAN_RATE_LIMIT_CHECKLIST_ITEMS_CREATE", "200")),
    "kanban.checklist_items.list": int(os.getenv("KANBAN_RATE_LIMIT_CHECKLIST_ITEMS_LIST", "200")),
    "kanban.checklist_items.get": int(os.getenv("KANBAN_RATE_LIMIT_CHECKLIST_ITEMS_GET", "200")),
    "kanban.checklist_items.update": int(os.getenv("KANBAN_RATE_LIMIT_CHECKLIST_ITEMS_UPDATE", "200")),
    "kanban.checklist_items.delete": int(os.getenv("KANBAN_RATE_LIMIT_CHECKLIST_ITEMS_DELETE", "120")),
    "kanban.checklist_items.reorder": int(os.getenv("KANBAN_RATE_LIMIT_CHECKLIST_ITEMS_REORDER", "120")),
    "kanban.checklist_items.toggle_all": int(os.getenv("KANBAN_RATE_LIMIT_CHECKLIST_ITEMS_TOGGLE_ALL", "120")),
}

# In-memory rate limit tracking (per-user, per-action)
# NOTE: This is process-local and will not work as a global rate limit across
# multiple server workers/instances. For distributed deployments, use a shared
# store (e.g., Redis) and/or an ingress rate limiter.
_rate_limit_lock = threading.Lock()
_rate_limit_windows: dict[str, deque] = {}
RATE_LIMIT_WINDOW_SECONDS = 60
_RATE_LIMIT_CLEANUP_INTERVAL_SECONDS = max(
    1.0, float(os.getenv("KANBAN_RATE_LIMIT_CLEANUP_INTERVAL_SECONDS", "300"))
)
_rate_limit_last_cleanup_ts: float = 0.0


def _maybe_cleanup_rate_limit_windows(now: float) -> None:
    """Remove stale rate limit windows to avoid unbounded growth.

    Must be called with `_rate_limit_lock` held.
    """
    global _rate_limit_last_cleanup_ts

    now_monotonic = time.monotonic()
    if (now_monotonic - _rate_limit_last_cleanup_ts) < _RATE_LIMIT_CLEANUP_INTERVAL_SECONDS:
        return

    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    for key, window in list(_rate_limit_windows.items()):
        while window and window[0] < window_start:
            window.popleft()
        if not window:
            _rate_limit_windows.pop(key, None)

    _rate_limit_last_cleanup_ts = now_monotonic


def check_kanban_rate_limit(user_id: int, action: str) -> tuple[bool, dict[str, Any]]:
    """
    Check if a user has exceeded the rate limit for an action.

    Args:
        user_id: The user ID to check.
        action: The action being performed (e.g., "kanban.boards.create").

    Returns:
        Tuple of (allowed: bool, info: dict with limit details).
    """
    limit = KANBAN_RATE_LIMITS.get(action, 60)  # Default 60/min
    key = f"{user_id}:{action}"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    with _rate_limit_lock:
        _maybe_cleanup_rate_limit_windows(now)
        window = _rate_limit_windows.setdefault(key, deque())

        # Remove expired entries
        while window and window[0] < window_start:
            window.popleft()

        current_count = len(window)
        remaining = max(0, limit - current_count)
        reset_time = int(window[0] + RATE_LIMIT_WINDOW_SECONDS) if window else int(now + RATE_LIMIT_WINDOW_SECONDS)

        info = {
            "limit": limit,
            "remaining": remaining,
            "reset": reset_time,
            "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
        }

        if current_count >= limit:
            return False, info

        # Record this request
        window.append(now)
        info["remaining"] = remaining - 1
        return True, info


def kanban_rate_limit(action: str) -> Callable:
    """
    FastAPI dependency factory for rate limiting Kanban operations.

    Usage:
        @router.post("/boards", dependencies=[Depends(kanban_rate_limit("kanban.boards.create"))])
        async def create_board(...):

    Args:
        action: The action identifier for rate limit lookup.

    Returns:
        A FastAPI dependency function.
    """
    async def rate_limit_dependency(current_user: User = Depends(get_request_user)) -> None:
        # Match Auth dependencies: bypass in test contexts for deterministic tests.
        if _is_explicit_pytest_runtime() or _is_test_mode():
            return

        user_id = current_user.id if current_user else 0
        allowed, info = check_kanban_rate_limit(user_id, action)

        if not allowed:
            logger.warning(
                f"Rate limit exceeded for user {user_id} on action {action}: "
                f"{info['limit']}/{info['window_seconds']}s"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {info['reset'] - int(time.time())} seconds.",
                headers={
                    "X-RateLimit-Limit": str(info["limit"]),
                    "X-RateLimit-Remaining": str(info["remaining"]),
                    "X-RateLimit-Reset": str(info["reset"]),
                    "Retry-After": str(info["reset"] - int(time.time())),
                },
            )

    return rate_limit_dependency


def get_rate_limit_status(user_id: int, action: str) -> dict[str, Any]:
    """
    Get current rate limit status for a user/action without consuming a request.

    Args:
        user_id: The user ID.
        action: The action identifier.

    Returns:
        Dict with limit, remaining, and reset information.
    """
    limit = KANBAN_RATE_LIMITS.get(action, 60)
    key = f"{user_id}:{action}"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    with _rate_limit_lock:
        _maybe_cleanup_rate_limit_windows(now)
        window = _rate_limit_windows.get(key, deque())

        # Count valid entries (don't modify the window)
        valid_count = sum(1 for ts in window if ts >= window_start)
        remaining = max(0, limit - valid_count)
        reset_time = int(now + RATE_LIMIT_WINDOW_SECONDS)

        return {
            "limit": limit,
            "remaining": remaining,
            "reset": reset_time,
            "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
        }
