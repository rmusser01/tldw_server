"""ChaChaNotes runtime: the executor, per-user DB cache, init state, health counters and shutdown.

One ``ChaChaRuntimeManager`` lives in the app-service container (``get_chacha_runtime()``).
It raises core errors: ``core/Character_Chat/chacha_db_provider.py`` opens per-user databases through it,
and the API layer maps its errors to HTTP.
"""

from __future__ import annotations

import asyncio
import faulthandler
import os
import sqlite3
import sys
import threading
import time
from collections.abc import Callable, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from cachetools import LRUCache
from loguru import logger

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
from tldw_Server_API.app.core.DB_Management.chacha.health import probe_chacha_connection
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import (
    chacha_operation,
    current_operation_holds_connection,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_Server_API.app.core.exceptions import BuiltinCharacterSeedError
from tldw_Server_API.app.core.Utils.app_services import app_services

_CHACHA_SHUTDOWN_INIT_ERROR_DETAIL = "ChaChaNotes shutdown in progress"
_CHACHA_INIT_TIMEOUT_DETAIL = "ChaChaNotes initialization timed out"
_CHACHA_CORRUPTION_ERROR_DETAIL = "ChaChaNotes DB corruption detected; repair or restore required"
_CHACHA_RECOVERY_DOC = "Docs/Operations/ChaChaNotes_DB_Recovery.md"
_CHACHA_RECOVERY_MESSAGE = "Restore the affected ChaChaNotes DB from backup or move it aside before retrying."
_SQLITE_CORRUPTION_SIGNATURES = (
    "database disk image is malformed",
    "malformed database schema",
    "file is not a database",
)


class ChaChaRuntimeUnavailableError(RuntimeError):
    """Raised when ChaChaNotes resources are unavailable for new work."""


class ChaChaInitAbortedError(ChaChaRuntimeUnavailableError):
    """Shutdown abandoned the initialization a waiter was blocked on."""


class ChaChaInitTimeoutError(RuntimeError):
    """Initialization, or the wait for another caller's initialization, timed out."""

    def __init__(self, message: str = _CHACHA_INIT_TIMEOUT_DETAIL) -> None:
        super().__init__(message)


class ChaChaInitFailedError(RuntimeError):
    """The DB factory failed. ``__cause__`` is its error, or None when it is unknown."""


class ChaChaDatabaseCorruptionError(CharactersRAGDBError):
    """Raised when an existing ChaChaNotes SQLite DB fails integrity preflight."""

    def __init__(
        self,
        detail: str = _CHACHA_CORRUPTION_ERROR_DETAIL,
        *,
        user_id: int | None = None,
        db_path: Path | None = None,
        reason_code: str = "sqlite_corruption",
    ) -> None:
        super().__init__(detail)
        self.user_id = user_id
        self.db_path = db_path
        self.reason_code = reason_code


def _is_sqlite_corruption_error(error: Exception | None) -> bool:
    if isinstance(error, ChaChaDatabaseCorruptionError):
        return True
    text = str(error or "").strip().lower()
    if not text:
        return False
    return any(signature in text for signature in _SQLITE_CORRUPTION_SIGNATURES)


def _safe_chacha_health_error(error: Exception | None) -> str:
    if _is_sqlite_corruption_error(error):
        return "sqlite_corruption"
    return type(error).__name__ if error else "unknown error"


def _sanitize_chacha_db_identifier(user_id: int | None, db_path: Path | None = None) -> str:
    """Return a public-safe DB label for ChaChaNotes health payloads.

    Aggregate health is unauthenticated, so the default label must not expose
    numeric user IDs, absolute paths, or nonstandard filenames. The context
    arguments are accepted only so callers can pass corruption context
    consistently; they are intentionally not encoded in the returned value.

    Args:
        user_id: Optional owner id from a corruption exception. Ignored to avoid
            leaking per-user identifiers in public health responses.
        db_path: Optional filesystem path from a corruption exception. Ignored
            to avoid leaking path or filename details.

    Returns:
        The constant public label ``"ChaChaNotes.db"``. For example, both
        ``user_id=42`` and ``db_path=/private/user/42/ChaChaNotes.db`` return
        ``"ChaChaNotes.db"``.
    """
    return "ChaChaNotes.db"


def _build_chacha_failure_details(error: Exception | None) -> dict[str, Any] | None:
    """Build sanitized recovery metadata for current ChaChaNotes failures.

    Args:
        error: Exception raised while opening a ChaChaNotes database. Only
            exceptions recognized by ``_is_sqlite_corruption_error`` are exposed.

    Returns:
        ``None`` for non-corruption failures so generic runtime errors are not
        copied into health responses. For corruption failures, returns a dict
        with ``reason_code`` from the exception or ``"sqlite_corruption"``,
        ``affected_db`` from ``_sanitize_chacha_db_identifier``, and ``recovery``
        containing ``automatic_repair=False``, ``documentation`` from
        ``_CHACHA_RECOVERY_DOC``, and ``message`` from
        ``_CHACHA_RECOVERY_MESSAGE``. The payload avoids raw exception text,
        filesystem paths, and user identifiers.
    """
    if not _is_sqlite_corruption_error(error):
        return None

    user_id = getattr(error, "user_id", None)
    db_path = getattr(error, "db_path", None)
    reason_code = getattr(error, "reason_code", "sqlite_corruption")
    return {
        "reason_code": str(reason_code or "sqlite_corruption"),
        "affected_db": _sanitize_chacha_db_identifier(user_id, db_path),
        "recovery": {
            "automatic_repair": False,
            "documentation": _CHACHA_RECOVERY_DOC,
            "message": _CHACHA_RECOVERY_MESSAGE,
        },
    }


def _copy_chacha_failure_details(failure: Any) -> dict[str, Any] | None:
    """Copy stored failure metadata while preserving public redaction rules.

    Args:
        failure: Candidate failure dict from shared health state.

    Returns:
        ``None`` when the input is not a dict. Otherwise returns a new dict with
        a string ``reason_code``, a re-sanitized ``affected_db`` fallback of
        ``"ChaChaNotes.db"``, and a shallow copy of the ``recovery`` dict. The
        copy prevents callers from mutating shared health state and also
        normalizes dictionaries seeded directly by tests or older in-process
        state.
    """
    if not isinstance(failure, dict):
        return None

    recovery = failure.get("recovery")
    recovery_details = dict(recovery) if isinstance(recovery, dict) else {}
    affected_db = failure.get("affected_db")
    safe_db = _sanitize_chacha_db_identifier(None, Path(str(affected_db))) if affected_db else "ChaChaNotes.db"
    return {
        "reason_code": str(failure.get("reason_code") or "unknown"),
        "affected_db": safe_db,
        "recovery": recovery_details,
    }


def _on_closed_loop(awaitable: asyncio.Future) -> bool:
    """True when the task/future's event loop is closed, so it can never finish.

    Waiting on one only burns the full drain timeout (and cancelling a task on a
    closed loop raises), on every shutdown for the rest of the process.
    """
    try:
        return awaitable.get_loop().is_closed()
    except (AttributeError, RuntimeError):
        return False


class ChaChaRuntimeManager:
    """Owns the ChaChaNotes executor, per-user DB cache, init state, health counters and lifecycle."""

    def __init__(
        self,
        *,
        max_cached: int = 20,
        max_workers: int = 4,
        init_timeout_secs: float = 30.0,
        watchdog_secs: float = 5.0,
    ) -> None:
        self._lock = threading.RLock()
        self._shutting_down = False
        self.max_workers = max_workers
        # Read on every get_db, so a test can change them on the instance.
        self.init_timeout_secs = init_timeout_secs
        self.watchdog_secs = watchdog_secs
        self._executor: ThreadPoolExecutor | None = None
        self._executor_shutdown = False
        self._executor_lock = threading.Lock()
        self._instances: MutableMapping[str, CharactersRAGDB] = LRUCache(maxsize=max_cached)
        self._cache_lock = threading.Lock()
        self._init_events: dict[str, threading.Event] = {}
        self._init_errors: dict[str, Exception] = {}
        self._health_lock = threading.Lock()
        self._health: dict[str, Any] = {
            "init_attempts": 0,
            "init_failures": 0,
            "last_init_ms": None,
            "last_error": None,
            "last_init_success": None,
            "last_warn_dump": None,
            "cached_instances": 0,
            "consecutive_failures": 0,
            "default_char_ensures": 0,
            "default_char_failures": 0,
            "warm_startups": 0,
            "last_failure": None,
        }
        self._tasks: set[asyncio.Task[Any]] = set()
        self._futures: set[asyncio.Future[Any]] = set()
        self._futures_lock = threading.Lock()

    def shutdown(self) -> None:
        """Stop accepting new ChaChaNotes initialization work."""
        with self._lock:
            self._shutting_down = True

    def resume(self) -> None:
        """Accept new work again, as after a lifespan restart in the same process."""
        with self._lock:
            self._shutting_down = False

    @property
    def shutting_down(self) -> bool:
        with self._lock:
            return self._shutting_down

    def executor(self) -> ThreadPoolExecutor:
        """
        Return a live executor for ChaChaNotes DB work.

        The main FastAPI app shuts down this executor during application shutdown.
        In test suites, startup/shutdown can happen repeatedly within the same
        Python process; recreating the executor on-demand avoids order-dependent
        failures like "cannot schedule new futures after shutdown".
        """
        with self._executor_lock:
            if self._executor is None or self._executor_shutdown:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix="chacha-db",
                )
                self._executor_shutdown = False
            return self._executor

    def shutdown_executor(self, wait: bool = False) -> None:
        """
        Shut down the ChaChaNotes executor to avoid lingering threads on shutdown.

        Captures the current executor under lock, clears the reference,
        marks shutdown, then releases the lock before shutting down the captured
        executor. This pattern avoids deadlock and allows executor()
        to safely create a new executor if called again (e.g., in test scenarios).

        Args:
            wait: If True, block until all pending futures complete. If False,
                return immediately after cancelling pending futures.

        Note:
            Uses cancel_futures=True to aggressively terminate pending work.
            Safe to call multiple times; subsequent calls are no-ops.
        """
        with self._executor_lock:
            executor = self._executor
            self._executor = None
            self._executor_shutdown = True
        if executor is None:
            return
        try:
            executor.shutdown(wait=wait, cancel_futures=True)
        except (RuntimeError, OSError, ValueError) as e:
            logger.debug("ChaChaNotes executor shutdown error ({})", type(e).__name__)

    def cached_instances(self) -> tuple[CharactersRAGDB, ...]:
        """Return a lock-protected snapshot of the cached DB instances."""
        with self._cache_lock:
            return tuple(self._instances.values())

    def close_all(self) -> None:
        """Closes all cached ChaChaNotesDB connections. Useful for application shutdown."""
        pending_init_events: list[threading.Event] = []
        pending_shutdown_errors: dict[str, Exception] = {}
        with self._cache_lock:
            logger.info(f"Closing all cached ChaChaNotesDB instances ({len(self._instances)})...")
            for user_id, db_instance in list(self._instances.items()):
                try:
                    db_instance.close_all_connections()
                    logger.info(f"Closed ChaChaNotesDB instance for user {user_id}.")
                except (CharactersRAGDBError, OSError, RuntimeError, ValueError, TypeError) as e:
                    logger.error("Error closing ChaChaNotesDB instance ({})", type(e).__name__)
            pending_init_events = list(self._init_events.values())
            pending_shutdown_errors = {
                cache_key: ChaChaInitAbortedError(_CHACHA_SHUTDOWN_INIT_ERROR_DETAIL)
                for cache_key in self._init_events
            }
            self._instances.clear()
            self._init_events.clear()
            self._init_errors.clear()
            self._init_errors.update(pending_shutdown_errors)
            logger.info("All ChaChaNotesDB instances closed and cache cleared.")
        for init_event in pending_init_events:
            init_event.set()

    def track(self, pending: asyncio.Task[Any] | asyncio.Future[Any]) -> None:
        """Track default-character work so shutdown can drain it."""
        if isinstance(pending, asyncio.Task):
            self._tasks.add(pending)
            pending.add_done_callback(self._tasks.discard)
            return

        def _cleanup(future: asyncio.Future[Any]) -> None:
            with self._futures_lock:
                self._futures.discard(future)

        with self._futures_lock:
            self._futures.add(pending)
        pending.add_done_callback(_cleanup)

    def count(self, counter: Literal["warm_startups", "default_char_ensures", "default_char_failures"]) -> None:
        """Increment one health counter."""
        with self._health_lock:
            self._health[counter] += 1

    def _record_init(self, duration_ms: float, success: bool, error: Exception | None = None) -> None:
        """Record one ChaChaNotes initialization attempt and current health state."""
        cached_count = len(self.cached_instances())
        with self._health_lock:
            self._health["init_attempts"] += 1
            self._health["last_init_ms"] = duration_ms
            self._health["cached_instances"] = cached_count
            self._health["last_init_success"] = success
            if success:
                self._health["last_error"] = None
                self._health["last_failure"] = None
                self._health["consecutive_failures"] = 0
            else:
                self._health["init_failures"] += 1
                self._health["consecutive_failures"] = int(self._health.get("consecutive_failures") or 0) + 1
                self._health["last_error"] = _safe_chacha_health_error(error)
                self._health["last_failure"] = _build_chacha_failure_details(error)

    def _maybe_dump_traceback(self, reason: str) -> None:
        now = time.time()
        last_dump = self._health.get("last_warn_dump")
        # Rate limit dumps to avoid log spam
        if last_dump and now - float(last_dump) < 300:
            return
        with self._health_lock:
            self._health["last_warn_dump"] = now
        try:
            logger.warning(f"ChaChaNotes watchdog dump triggered: {reason}")
            faulthandler.dump_traceback(file=sys.stderr)
        except (OSError, RuntimeError, ValueError) as dump_err:
            logger.debug("Faulthandler dump failed ({})", type(dump_err).__name__)

    def health_snapshot(self) -> dict[str, Any]:
        """Return current ChaChaNotes health plus lifetime counters.

        `status` reflects the latest/current initialization state, not the lifetime
        failure counter. `init_failures` remains a monotonic diagnostic counter while
        `consecutive_failures` and `last_init_success` drive recovery back to healthy
        after an operator fixes the database and a later init succeeds.
        """
        with self._health_lock:
            init_attempts = self._health.get("init_attempts")
            init_failures = self._health.get("init_failures")
            last_init_ms = self._health.get("last_init_ms")
            last_error = self._health.get("last_error")
            last_init_success = self._health.get("last_init_success")
            consecutive_failures = int(self._health.get("consecutive_failures") or 0)
            default_char_ensures = self._health.get("default_char_ensures")
            default_char_failures = self._health.get("default_char_failures")
            warm_startups = self._health.get("warm_startups")
            last_failure = _copy_chacha_failure_details(self._health.get("last_failure"))
        cached_instances = len(self.cached_instances())

        status = "degraded" if consecutive_failures else "healthy"
        return {
            "status": status,
            "init_attempts": init_attempts,
            "init_failures": init_failures,
            "last_init_ms": last_init_ms,
            "last_error": last_error,
            "last_init_success": last_init_success,
            "consecutive_failures": consecutive_failures,
            "last_failure": last_failure,
            "cached_instances": cached_instances,
            "default_char_ensures": default_char_ensures,
            "default_char_failures": default_char_failures,
            "warm_startups": warm_startups,
        }

    @chacha_operation(independent=True)
    def _probe(self, db_instance: CharactersRAGDB) -> bool:
        try:
            probe_chacha_connection(db_instance)
            return True
        except (BackendDatabaseError, CharactersRAGDBError, sqlite3.Error, OSError, RuntimeError, ValueError) as e:
            logger.warning("ChaChaNotes health probe failed ({})", type(e).__name__)
            return False

    async def _is_healthy(self, db_instance: CharactersRAGDB) -> bool:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._probe, db_instance),
                timeout=1.0,
            )
            return bool(result)
        except (asyncio.TimeoutError, OSError, RuntimeError):
            return False

    async def get_db(self, key: str, factory: Callable[[], CharactersRAGDB], *, user_id: int) -> CharactersRAGDB:
        """Return a cached owner database or serialize and publish its initialization."""
        if self.shutting_down:
            raise ChaChaRuntimeUnavailableError(_CHACHA_SHUTDOWN_INIT_ERROR_DETAIL)
        with self._cache_lock:
            db_instance = self._instances.get(key)
        if db_instance:
            # A connection this operation already holds proves the instance usable.
            # Re-probing could evict it mid-request (the probe has a 1s deadline)
            # and hand a nested accessor a different instance, splitting one request
            # across two connections that cannot see each other's pending writes.
            if current_operation_holds_connection(db_instance) or await self._is_healthy(db_instance):
                return db_instance
            logger.warning(f"ChaChaNotes cached instance unhealthy for user {user_id}; evicting and rebuilding.")
            with self._cache_lock:
                if self._instances.get(key) is db_instance:
                    self._instances.pop(key, None)

        wait_for_existing_init = False
        with self._cache_lock:
            cached_instance = self._instances.get(key)
            cached_count = len(self._instances)
            if cached_instance is None:
                init_event = self._init_events.get(key)
                if init_event is None:
                    init_event = threading.Event()
                    self._init_events[key] = init_event
                else:
                    wait_for_existing_init = True
        if cached_instance is not None:
            with self._health_lock:
                self._health["cached_instances"] = cached_count
            return cached_instance

        if wait_for_existing_init:
            wait_timeout = self._init_timeout()
            try:
                completed = await asyncio.wait_for(
                    asyncio.to_thread(init_event.wait),
                    timeout=wait_timeout,
                )
            except asyncio.TimeoutError as e:
                self._record_init(wait_timeout * 1000, False, e)
                raise ChaChaInitTimeoutError() from e
            if not completed:
                self._record_init(wait_timeout * 1000, False)
                raise ChaChaInitTimeoutError()
            with self._cache_lock:
                cached_instance = self._instances.get(key)
                init_error = self._init_errors.get(key)
            if cached_instance is not None:
                return cached_instance
            if isinstance(init_error, ChaChaInitAbortedError):
                raise ChaChaInitAbortedError(_CHACHA_SHUTDOWN_INIT_ERROR_DETAIL) from init_error
            raise ChaChaInitFailedError() from init_error

        loop = asyncio.get_running_loop()
        start = time.perf_counter()
        init_timeout = self._init_timeout()
        try:
            db_instance = await asyncio.wait_for(
                loop.run_in_executor(self.executor(), factory),
                timeout=init_timeout,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            self._record_init(duration_ms, True)
            if duration_ms / 1000 > self.watchdog_secs:
                self._maybe_dump_traceback(f"ChaChaNotes init exceeded {self.watchdog_secs}s for user {user_id}")
        except asyncio.TimeoutError as e:
            self._record_init(init_timeout * 1000, False, e)
            self._maybe_dump_traceback(f"ChaChaNotes init timed out for user {user_id}")
            raise ChaChaInitTimeoutError() from e
        except (
            BuiltinCharacterSeedError,
            CharactersRAGDBError,
            sqlite3.Error,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as e:
            duration_ms = (time.perf_counter() - start) * 1000
            self._record_init(duration_ms, False, e)
            with self._cache_lock:
                self._init_errors[key] = e
            raise ChaChaInitFailedError() from e
        else:
            with self._cache_lock:
                self._instances[key] = db_instance
                self._init_errors.pop(key, None)
                cached_count = len(self._instances)
            with self._health_lock:
                self._health["cached_instances"] = cached_count
            return db_instance
        finally:
            with self._cache_lock:
                init_event = self._init_events.pop(key, None)
            if init_event is not None:
                init_event.set()

    def _init_timeout(self) -> float:
        """Return the hard ChaChaNotes initialization timeout."""
        return max(self.init_timeout_secs, self.watchdog_secs * 3, 5)

    async def _drain_tasks(self, timeout: float = 5.0) -> None:
        stale = [task for task in list(self._tasks) if _on_closed_loop(task)]
        self._tasks.difference_update(stale)
        tasks = [task for task in list(self._tasks) if not task.done()]
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        if pending:
            logger.warning(
                "ChaChaNotes shutdown: {} default-character tasks still running; cancelling.",
                len(pending),
            )
            for task in pending:
                task.cancel()
            await asyncio.wait(pending, timeout=1.0)
        self._tasks.difference_update(done)
        self._tasks.difference_update(pending)

    async def _drain_futures(self, timeout: float = 5.0) -> None:
        with self._futures_lock:
            stale = [future for future in list(self._futures) if _on_closed_loop(future)]
            self._futures.difference_update(stale)
            futures = [future for future in list(self._futures) if not future.done()]
        if not futures:
            return
        done, pending = await asyncio.wait(futures, timeout=timeout)
        if pending:
            logger.warning(
                "ChaChaNotes shutdown: {} default-character futures still running; waiting on executor shutdown.",
                len(pending),
            )
        with self._futures_lock:
            for future in done:
                self._futures.discard(future)

    async def aclose(self, wait_timeout: float = 5.0) -> None:
        """Drain ChaChaNotes tasks and close resources without racing active threads."""
        self.shutdown()
        await self._drain_tasks(timeout=wait_timeout)
        await self._drain_futures(timeout=wait_timeout)
        # Block the shutdown path until worker threads complete to avoid closing
        # SQLite connections mid-query during test teardown.
        self.shutdown_executor(wait=True)
        self.close_all()


# Serializes first-time creation of the container's ``chacha_runtime`` slot.
_runtime_lock = threading.Lock()


def get_chacha_runtime() -> ChaChaRuntimeManager:
    """Get the ChaChaNotes runtime from the app-service container, creating it on first use.

    Fallback for code with no request or constructor argument; request handlers reach it
    through ``Depends(get_chacha_db_for_user)``.
    """
    services = app_services()
    if services.chacha_runtime is None:
        with _runtime_lock:
            if services.chacha_runtime is None:
                max_cached = int(settings.get("MAX_CACHED_CHACHA_DB_INSTANCES", "20"))
                services.chacha_runtime = ChaChaRuntimeManager(
                    max_cached=max_cached,
                    max_workers=max(1, int(os.getenv("CHACHA_EXECUTOR_MAX_WORKERS", "4"))),
                    init_timeout_secs=float(os.getenv("CHACHA_INIT_TIMEOUT_SECS", "30")),
                    watchdog_secs=float(os.getenv("CHACHA_INIT_WATCHDOG_SECS", "5")),
                )
                logger.debug("Using LRUCache for ChaChaNotes DB instances (maxsize={}).", max_cached)
    return services.chacha_runtime
