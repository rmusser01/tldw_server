# tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py
import asyncio
import faulthandler
import inspect
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from cachetools import LRUCache
from fastapi import Depends, HTTPException, status
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

# Local Imports
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
    SchemaError,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management import sqlite_policy

#
#######################################################################################################################


# --- Configuration ---
_CHACHA_EXECUTOR: ThreadPoolExecutor | None = None
_CHACHA_EXECUTOR_SHUTDOWN: bool = False
_CHACHA_EXECUTOR_LOCK = threading.Lock()
_CHACHA_EXECUTOR_MAX_WORKERS = max(1, int(os.getenv("CHACHA_EXECUTOR_MAX_WORKERS", "4")))
_CHACHA_WATCHDOG_SECS = float(os.getenv("CHACHA_INIT_WATCHDOG_SECS", "5"))
_CHACHA_SHUTDOWN_INIT_ERROR_DETAIL = "ChaChaNotes shutdown in progress"
_CHACHA_HEALTH_LOCK = threading.Lock()
_CHACHA_HEALTH: dict[str, Any] = {
    "init_attempts": 0,
    "init_failures": 0,
    "last_init_ms": None,
    "last_error": None,
    "last_warn_dump": None,
    "cached_instances": 0,
    "default_char_ensures": 0,
    "default_char_failures": 0,
    "warm_startups": 0,
}
_CHACHA_SHUTTING_DOWN = False
_CHACHA_SHUTDOWN_LOCK = threading.Lock()


def _set_chacha_shutting_down(value: bool) -> None:
    global _CHACHA_SHUTTING_DOWN
    with _CHACHA_SHUTDOWN_LOCK:
        _CHACHA_SHUTTING_DOWN = value


def _is_chacha_shutting_down() -> bool:
    with _CHACHA_SHUTDOWN_LOCK:
        return _CHACHA_SHUTTING_DOWN


def reset_chacha_shutdown_state() -> None:
    _set_chacha_shutting_down(False)


def _get_chacha_executor() -> ThreadPoolExecutor:
    """
    Return a live executor for ChaChaNotes DB work.

    The main FastAPI app shuts down this executor during application shutdown.
    In test suites, startup/shutdown can happen repeatedly within the same
    Python process; recreating the executor on-demand avoids order-dependent
    failures like "cannot schedule new futures after shutdown".
    """
    global _CHACHA_EXECUTOR, _CHACHA_EXECUTOR_SHUTDOWN
    with _CHACHA_EXECUTOR_LOCK:
        if _CHACHA_EXECUTOR is None or _CHACHA_EXECUTOR_SHUTDOWN:
            _CHACHA_EXECUTOR = ThreadPoolExecutor(
                max_workers=_CHACHA_EXECUTOR_MAX_WORKERS,
                thread_name_prefix="chacha-db",
            )
            _CHACHA_EXECUTOR_SHUTDOWN = False
        return _CHACHA_EXECUTOR


def _record_init(duration_ms: float, success: bool, error: Exception | None = None) -> None:
    with _CHACHA_HEALTH_LOCK:
        _CHACHA_HEALTH["init_attempts"] += 1
        _CHACHA_HEALTH["last_init_ms"] = duration_ms
        _CHACHA_HEALTH["cached_instances"] = len(_chacha_db_instances)
        if success:
            _CHACHA_HEALTH["last_error"] = None
        else:
            _CHACHA_HEALTH["init_failures"] += 1
            _CHACHA_HEALTH["last_error"] = str(error) if error else "unknown error"


def _record_default_character(success: bool) -> None:
    with _CHACHA_HEALTH_LOCK:
        if success:
            _CHACHA_HEALTH["default_char_ensures"] += 1
        else:
            _CHACHA_HEALTH["default_char_failures"] += 1


def _maybe_dump_traceback(reason: str) -> None:
    now = time.time()
    last_dump = _CHACHA_HEALTH.get("last_warn_dump")
    # Rate limit dumps to avoid log spam
    if last_dump and now - float(last_dump) < 300:
        return
    with _CHACHA_HEALTH_LOCK:
        _CHACHA_HEALTH["last_warn_dump"] = now
    try:
        logger.warning(f"ChaChaNotes watchdog dump triggered: {reason}")
        faulthandler.dump_traceback(file=sys.stderr)
    except (OSError, RuntimeError, ValueError) as dump_err:
        logger.debug(f"Faulthandler dump failed: {dump_err}")


def _track_default_character_future(future: asyncio.Future) -> None:
    def _cleanup(_future: asyncio.Future) -> None:
        with _chacha_default_char_futures_lock:
            _chacha_default_char_futures.discard(_future)

    with _chacha_default_char_futures_lock:
        _chacha_default_char_futures.add(future)
    future.add_done_callback(_cleanup)


def get_chacha_health_snapshot() -> dict[str, Any]:
    status = "healthy"
    if _CHACHA_HEALTH.get("init_failures"):
        status = "degraded"
    return {
        "status": status,
        "init_attempts": _CHACHA_HEALTH.get("init_attempts"),
        "init_failures": _CHACHA_HEALTH.get("init_failures"),
        "last_init_ms": _CHACHA_HEALTH.get("last_init_ms"),
        "last_error": _CHACHA_HEALTH.get("last_error"),
        "cached_instances": len(_chacha_db_instances),
        "default_char_ensures": _CHACHA_HEALTH.get("default_char_ensures"),
        "default_char_failures": _CHACHA_HEALTH.get("default_char_failures"),
    }


def resolve_chacha_user_base_dir() -> Path:
    """Public helper to expose the resolved user database base directory."""
    return DatabasePaths.get_user_db_base_dir()


SERVER_CLIENT_ID = settings.get("SERVER_CLIENT_ID")
if not SERVER_CLIENT_ID:
    logger.error("CRITICAL: SERVER_CLIENT_ID is not configured in settings.")
    SERVER_CLIENT_ID = "default_server_client_id"
    logger.warning(f"SERVER_CLIENT_ID not set, using placeholder: {SERVER_CLIENT_ID}")

# Global directory creation for a *common* ChaChaNotes base is removed
# as each user gets their DB under their own USER_DB_BASE_DIR/user_id/

# +++ Default Character Configuration +++
DEFAULT_CHARACTER_NAME = "Helpful AI Assistant"
DEFAULT_CHARACTER_DESCRIPTION = "A default, friendly assistant created automatically by the system."

# --- Global Cache for ChaChaNotes DB Instances ---
MAX_CACHED_CHACHA_DB_INSTANCES = int(settings.get("MAX_CACHED_CHACHA_DB_INSTANCES", "20"))

_chacha_db_instances: LRUCache = LRUCache(maxsize=MAX_CACHED_CHACHA_DB_INSTANCES)
logger.info(f"Using LRUCache for ChaChaNotes DB instances (maxsize={MAX_CACHED_CHACHA_DB_INSTANCES}).")

_chacha_db_lock = threading.Lock()
_chacha_db_init_events: dict[str, threading.Event] = {}
_chacha_db_init_errors: dict[str, Exception] = {}
_chacha_default_char_tasks: set[asyncio.Task] = set()
_chacha_default_char_futures: set[asyncio.Future] = set()
_chacha_default_char_futures_lock = threading.Lock()


#######################################################################################################################

# --- Helper Functions ---


class _ChaChaInitializationAborted(RuntimeError):
    """Internal sentinel used to wake blocked waiters during shutdown."""


def _get_chacha_db_path_for_user(user_id: int) -> Path:
    """
    Resolve the per-user ChaChaNotes DB path under the configured base.

    Policy: store each user's notes/chats DB at
      USER_DB_BASE_DIR / <user_id> / "ChaChaNotes.db"

    Notes:
    - USER_DB_BASE_DIR is read from global settings and can be overridden by env.
    - This per-user layout is intentional to isolate data and simplify backups.
    - Falls back to the default `Databases/user_databases` under repo root when unset.
    """
    db_file = DatabasePaths.get_chacha_db_path(user_id)
    logger.info(f"Ensured ChaChaNotes DB directory for user {user_id}: {db_file.parent}")
    return db_file


def _apply_sqlite_tuning(db_instance: CharactersRAGDB) -> None:
    if db_instance.backend_type != BackendType.SQLITE:
        return
    try:
        conn = db_instance.get_connection()
        sqlite_policy.configure_sqlite_connection(
            conn,
            use_wal=True,
            synchronous="NORMAL",
            foreign_keys=True,
            busy_timeout_ms=10000,
            temp_store=None,
        )
    except (CharactersRAGDBError, sqlite3.Error, OSError, RuntimeError, ValueError) as e:
        logger.debug(f"ChaChaNotes tuning skipped: {e}")


def _health_check_instance(db_instance: CharactersRAGDB) -> bool:
    try:
        conn = db_instance.get_connection()
        sqlite_policy.configure_sqlite_connection(
            conn,
            use_wal=False,
            synchronous=None,
            foreign_keys=True,
            busy_timeout_ms=1000,
            temp_store=None,
        )
        conn.execute("SELECT 1")
        return True
    except (CharactersRAGDBError, sqlite3.Error, OSError, RuntimeError, ValueError) as e:
        logger.warning(f"ChaChaNotes health probe failed: {e}")
        return False


def _create_and_prepare_db(user_id: int, client_id: str) -> CharactersRAGDB:
    db_path: Optional[Path] = None
    db_path = _get_chacha_db_path_for_user(user_id)
    try:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    except OSError as _mk2:
        logger.debug(f"Secondary ensure for ChaChaNotes parent failed softly: {_mk2}")
    logger.info(f"Initializing CharactersRAGDB instance for user {user_id} at path: {db_path}")
    db_instance = CharactersRAGDB(db_path=str(db_path), client_id=str(client_id))
    _apply_sqlite_tuning(db_instance)
    return db_instance


async def _ensure_default_character_async(db_instance: CharactersRAGDB, user_id: int) -> None:
    loop = asyncio.get_running_loop()
    try:
        future = loop.run_in_executor(_get_chacha_executor(), _ensure_default_character, db_instance)
        _track_default_character_future(future)
        await asyncio.wait_for(
            asyncio.shield(future),
            timeout=5,
        )
        _record_default_character(True)
    except asyncio.TimeoutError:
        _record_default_character(False)
        logger.warning(f"Timed out ensuring default character for user {user_id}; will retry on next access.")
    except (
        CharactersRAGDBError,
        SchemaError,
        InputError,
        ConflictError,
        sqlite3.Error,
        OSError,
        RuntimeError,
        ValueError,
    ) as e:
        _record_default_character(False)
        logger.warning(
            f"Error ensuring default character for user {user_id}: {e}. Continuing; will retry on next access.",
            exc_info=True,
        )


def _ensure_default_character(db_instance: CharactersRAGDB) -> Optional[int]:
    """
    Checks if the default character exists in the DB, creates it if not.
    Returns the character_id of the default character.
    """
    try:
        db_instance.ensure_character_tables_ready()
        default_char = db_instance.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
        if default_char:
            logger.debug(f"Default character '{DEFAULT_CHARACTER_NAME}' already exists with ID: {default_char['id']}.")
            return default_char["id"]
        else:
            logger.info(f"Default character '{DEFAULT_CHARACTER_NAME}' not found. Creating now...")
            card_data = {
                "name": DEFAULT_CHARACTER_NAME,
                "description": DEFAULT_CHARACTER_DESCRIPTION,
                # All other fields will be None or default in the DB
                "personality": "Supportive, patient, and concise.",
                "scenario": "General assistance",
                "system_prompt": "You are a helpful AI assistant.",
                "image": None,
                "post_history_instructions": None,
                "first_message": "Hello! I'm your Helpful AI Assistant. How can I support you today?",
                "message_example": None,
                "creator_notes": "This character is automatically generated to provide a reliable default assistant persona.",
                "alternate_greetings": None,
                "tags": json.dumps(["default", "neutral", "assistant"]),  # Store as JSON string
                "creator": "System",
                "character_version": "1.0",
                "extensions": None,
                "client_id": db_instance.client_id,  # Ensure client_id is set
            }
            # The add_character_card in CharactersRAGDB handles versioning and timestamps.
            char_id = db_instance.add_character_card(card_data)
            if char_id:
                logger.info(f"Successfully created default character '{DEFAULT_CHARACTER_NAME}' with ID: {char_id}.")
                return char_id
            else:
                # This should ideally not happen if add_character_card raises on failure
                logger.error(
                    f"Failed to create default character '{DEFAULT_CHARACTER_NAME}'. add_character_card returned None."
                )
                return None
    except ConflictError as e:  # Should only happen if get_character_card_by_name had an issue or race condition
        logger.warning(f"Conflict error while ensuring default character (likely race condition, re-fetching): {e}")
        # Re-fetch, as it might have been created by another thread.
        refetched_char = db_instance.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
        if refetched_char:
            return refetched_char["id"]
        logger.error(f"Still could not get/create default character after conflict: {e}")
        return None
    except (CharactersRAGDBError, SchemaError, InputError) as e:
        logger.error(f"Database error while ensuring default character '{DEFAULT_CHARACTER_NAME}': {e}", exc_info=True)
        return None  # Indicate failure
    except (AttributeError, KeyError, TypeError, ValueError, OSError, RuntimeError) as e_gen:
        logger.error(
            f"Unexpected error while ensuring default character '{DEFAULT_CHARACTER_NAME}': {e_gen}", exc_info=True
        )
        return None


async def _is_instance_healthy(db_instance: CharactersRAGDB) -> bool:
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_health_check_instance, db_instance),
            timeout=1.0,
        )
        return bool(result)
    except (asyncio.TimeoutError, OSError, RuntimeError):
        return False


async def _get_or_init_db_instance(user_id: int, client_id: str) -> CharactersRAGDB:
    user_dir = DatabasePaths.get_user_base_directory(user_id)
    cache_key = str(user_dir)
    with _chacha_db_lock:
        db_instance = _chacha_db_instances.get(cache_key)
    if db_instance:
        if await _is_instance_healthy(db_instance):
            return db_instance
        logger.warning(f"ChaChaNotes cached instance unhealthy for user {user_id}; evicting and rebuilding.")
        with _chacha_db_lock:
            if _chacha_db_instances.get(cache_key) is db_instance:
                _chacha_db_instances.pop(cache_key, None)

    wait_for_existing_init = False
    with _chacha_db_lock:
        cached_instance = _chacha_db_instances.get(cache_key)
        if cached_instance is not None:
            _CHACHA_HEALTH["cached_instances"] = len(_chacha_db_instances)
            return cached_instance
        init_event = _chacha_db_init_events.get(cache_key)
        if init_event is None:
            init_event = threading.Event()
            _chacha_db_init_events[cache_key] = init_event
        else:
            wait_for_existing_init = True

    if wait_for_existing_init:
        wait_timeout = max(_CHACHA_WATCHDOG_SECS * 3, 5)
        try:
            completed = await asyncio.wait_for(
                asyncio.to_thread(init_event.wait),
                timeout=wait_timeout,
            )
        except asyncio.TimeoutError as e:
            _record_init(wait_timeout * 1000, False, e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ChaChaNotes initialization timed out",
            ) from e
        if not completed:
            _record_init(wait_timeout * 1000, False)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ChaChaNotes initialization timed out",
            )
        with _chacha_db_lock:
            cached_instance = _chacha_db_instances.get(cache_key)
            init_error = _chacha_db_init_errors.get(cache_key)
        if cached_instance is not None:
            return cached_instance
        if init_error is not None:
            if isinstance(init_error, _ChaChaInitializationAborted):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=str(init_error),
                ) from init_error
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not initialize character & notes database for user: {init_error}",
            ) from init_error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not initialize character & notes database for user: unknown error",
        )

    loop = asyncio.get_running_loop()
    start = time.perf_counter()
    try:
        db_instance = await asyncio.wait_for(
            loop.run_in_executor(_get_chacha_executor(), _create_and_prepare_db, user_id, client_id),
            timeout=max(_CHACHA_WATCHDOG_SECS * 3, 5),
        )
        duration_ms = (time.perf_counter() - start) * 1000
        _record_init(duration_ms, True)
        if duration_ms / 1000 > _CHACHA_WATCHDOG_SECS:
            _maybe_dump_traceback(f"ChaChaNotes init exceeded {_CHACHA_WATCHDOG_SECS}s for user {user_id}")
    except asyncio.TimeoutError as e:
        _record_init(_CHACHA_WATCHDOG_SECS * 1000, False, e)
        _maybe_dump_traceback(f"ChaChaNotes init timed out for user {user_id}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ChaChaNotes initialization timed out",
        ) from e
    except (CharactersRAGDBError, sqlite3.Error, OSError, RuntimeError, ValueError, TypeError) as e:
        duration_ms = (time.perf_counter() - start) * 1000
        _record_init(duration_ms, False, e)
        with _chacha_db_lock:
            _chacha_db_init_errors[cache_key] = e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not initialize character & notes database for user: {e}",
        ) from e
    else:
        with _chacha_db_lock:
            _chacha_db_instances[cache_key] = db_instance
            _chacha_db_init_errors.pop(cache_key, None)
            _CHACHA_HEALTH["cached_instances"] = len(_chacha_db_instances)
        return db_instance
    finally:
        with _chacha_db_lock:
            init_event = _chacha_db_init_events.pop(cache_key, None)
        if init_event is not None:
            init_event.set()


async def warm_chacha_db_for_user(user_id: int, client_id: str | None = None) -> None:
    if _is_chacha_shutting_down():
        logger.debug("ChaChaNotes shutdown in progress; skipping warmup for user {}", user_id)
        return
    try:
        db_instance = await _get_or_init_db_instance(user_id, client_id or str(user_id))
        _CHACHA_HEALTH["warm_startups"] += 1
        task = asyncio.create_task(_ensure_default_character_async(db_instance, user_id))
        _chacha_default_char_tasks.add(task)
        task.add_done_callback(_chacha_default_char_tasks.discard)
    except (HTTPException, OSError, RuntimeError, ValueError, TypeError) as e:
        logger.warning(f"Warm-up for ChaChaNotes user {user_id} failed: {e}")


async def get_chacha_db_for_user_id(user_id: int, client_id: str | None = None) -> CharactersRAGDB:
    """
    Fetch a CharactersRAGDB instance for a specific user ID.

    This helper mirrors get_chacha_db_for_user but is intended for non-request contexts
    (e.g., WebSocket handlers) where we already know the user id.
    """
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid owner_user_id.",
        )

    db_instance = await _get_or_init_db_instance(user_id, client_id or str(user_id))
    if not _is_chacha_shutting_down():
        task = asyncio.create_task(_ensure_default_character_async(db_instance, user_id))
        _chacha_default_char_tasks.add(task)
        task.add_done_callback(_chacha_default_char_tasks.discard)
    return db_instance


# --- Main Dependency Function ---


async def get_chacha_db_for_user(current_user: User = Depends(get_request_user)) -> CharactersRAGDB:
    """
    FastAPI dependency to get the CharactersRAGDB instance for the identified user.
    Handles caching and health checks; heavy initialization runs in a dedicated executor.
    """
    # Respect FastAPI dependency overrides explicitly if they exist.
    # Some test environments reset overrides aggressively; checking here ensures
    # we still honor an override bound to this callable.
    try:
        from tldw_Server_API.app.main import app as _app  # Local import to avoid import cycles at module load

        override_fn = _app.dependency_overrides.get(get_chacha_db_for_user)
        if override_fn is not None:
            try:
                result = override_fn()
                if inspect.isawaitable(result):
                    result = await result  # type: ignore[func-returns-value]
                if isinstance(result, CharactersRAGDB):
                    return result
            except (HTTPException, TypeError, ValueError, RuntimeError, AttributeError):
                # Fall back to standard resolution on any override execution issue
                pass
    except (ImportError, AttributeError, RuntimeError):
        # If importing app or inspecting overrides fails, proceed normally
        pass

    logger.info("<<<<< ACTUAL get_chacha_db_for_user CALLED >>>>>")
    if not current_user or not isinstance(current_user.id, int):  # Ensure user_id is an int
        logger.error("get_chacha_db_for_user called without a valid User object or user.id is not int.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="User identification failed for ChaChaNotes DB."
        )

    user_id = current_user.id
    db_instance = await _get_or_init_db_instance(user_id, str(current_user.id))
    if not _is_chacha_shutting_down():
        task = asyncio.create_task(_ensure_default_character_async(db_instance, user_id))
        _chacha_default_char_tasks.add(task)
        task.add_done_callback(_chacha_default_char_tasks.discard)
    return db_instance


async def get_chacha_db_for_owner(owner_user_id: int) -> CharactersRAGDB:
    """
    Return a CharactersRAGDB for an arbitrary user (the workspace owner).

    Used by the sharing module to read sources from the owner's DB
    while the accessor chats with their own DB.
    """
    if not isinstance(owner_user_id, int):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid owner_user_id.",
        )
    return await _get_or_init_db_instance(owner_user_id, str(owner_user_id))


def close_all_chacha_db_instances():
    """Closes all cached ChaChaNotesDB connections. Useful for application shutdown."""
    pending_init_events: list[threading.Event] = []
    pending_shutdown_errors: dict[str, Exception] = {}
    with _chacha_db_lock:
        logger.info(f"Closing all cached ChaChaNotesDB instances ({len(_chacha_db_instances)})...")
        for user_id, db_instance in list(_chacha_db_instances.items()):
            try:
                db_instance.close_all_connections()
                logger.info(f"Closed ChaChaNotesDB instance for user {user_id}.")
            except (CharactersRAGDBError, OSError, RuntimeError, ValueError, TypeError) as e:
                logger.error(f"Error closing ChaChaNotesDB instance for user {user_id}: {e}", exc_info=True)
        pending_init_events = list(_chacha_db_init_events.values())
        pending_shutdown_errors = {
            cache_key: _ChaChaInitializationAborted(_CHACHA_SHUTDOWN_INIT_ERROR_DETAIL)
            for cache_key in _chacha_db_init_events
        }
        _chacha_db_instances.clear()
        _chacha_db_init_events.clear()
        _chacha_db_init_errors.clear()
        _chacha_db_init_errors.update(pending_shutdown_errors)
        logger.info("All ChaChaNotesDB instances closed and cache cleared.")
    for init_event in pending_init_events:
        init_event.set()


async def _drain_default_character_tasks(timeout: float = 5.0) -> None:
    tasks = [task for task in list(_chacha_default_char_tasks) if not task.done()]
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
    _chacha_default_char_tasks.difference_update(done)
    _chacha_default_char_tasks.difference_update(pending)


async def _drain_default_character_futures(timeout: float = 5.0) -> None:
    with _chacha_default_char_futures_lock:
        futures = [future for future in list(_chacha_default_char_futures) if not future.done()]
    if not futures:
        return
    done, pending = await asyncio.wait(futures, timeout=timeout)
    if pending:
        logger.warning(
            "ChaChaNotes shutdown: {} default-character futures still running; waiting on executor shutdown.",
            len(pending),
        )
    with _chacha_default_char_futures_lock:
        for future in done:
            _chacha_default_char_futures.discard(future)


async def shutdown_chacha_resources(wait_timeout: float = 5.0) -> None:
    """Drain ChaChaNotes tasks and close resources without racing active threads."""
    _set_chacha_shutting_down(True)
    await _drain_default_character_tasks(timeout=wait_timeout)
    await _drain_default_character_futures(timeout=wait_timeout)
    # Block the shutdown path until worker threads complete to avoid closing
    # SQLite connections mid-query during test teardown.
    shutdown_chacha_executor(wait=True)
    close_all_chacha_db_instances()


def shutdown_chacha_executor(wait: bool = False) -> None:
    """
    Shut down the ChaChaNotes executor to avoid lingering threads on shutdown.

    Captures the current executor under lock, clears the global reference,
    marks shutdown, then releases the lock before shutting down the captured
    executor. This pattern avoids deadlock and allows _get_chacha_executor()
    to safely create a new executor if called again (e.g., in test scenarios).

    Args:
        wait: If True, block until all pending futures complete. If False,
            return immediately after cancelling pending futures.

    Note:
        Uses cancel_futures=True to aggressively terminate pending work.
        Safe to call multiple times; subsequent calls are no-ops.
    """
    global _CHACHA_EXECUTOR, _CHACHA_EXECUTOR_SHUTDOWN
    with _CHACHA_EXECUTOR_LOCK:
        executor = _CHACHA_EXECUTOR
        _CHACHA_EXECUTOR = None
        _CHACHA_EXECUTOR_SHUTDOWN = True
    if executor is None:
        return
    try:
        executor.shutdown(wait=wait, cancel_futures=True)
    except (RuntimeError, OSError, ValueError) as e:
        logger.debug(f"ChaChaNotes executor shutdown error: {e}")


# Example of how to register for shutdown event in FastAPI:
# from fastapi import FastAPI
# app = FastAPI()
# @app.on_event("shutdown")
# async def shutdown_event():
#     close_all_chacha_db_instances()
#     # also close other DB instances if you have similar managers
