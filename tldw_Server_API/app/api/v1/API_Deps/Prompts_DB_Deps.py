# tldw_Server_API/app/api/v1/API_Deps/Prompts_DB_Deps.py
#
# Imports
import asyncio
import uuid as _uuid
from pathlib import Path
from typing import Optional

from cachetools import LRUCache  # Assuming cachetools is available

#
# Third-party imports
from fastapi import Depends, HTTPException, Request, status
from loguru import logger

from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Prompt_Management.Prompts_Interop import (
    ConflictError,
    DatabaseError,
    InputError,
    PromptsDatabase,
    SchemaError,
)
from tldw_Server_API.app.core.testing import is_test_mode

#
# Local Imports

#
########################################################################################################################
#
# Functions:

# Back-compat for tests that patch this module-level path.
try:
    MAIN_USER_DATA_BASE_DIR = DatabasePaths.get_user_db_base_dir()
except (OSError, ValueError, RuntimeError):
    MAIN_USER_DATA_BASE_DIR = None

# --- Configuration ---
SERVER_CLIENT_ID = settings.get("SERVER_CLIENT_ID")
if not SERVER_CLIENT_ID:
    SERVER_CLIENT_ID = "default_server_client_id_prompts" # Unique default
    logger.warning(f"SERVER_CLIENT_ID not set for prompts, using placeholder: {SERVER_CLIENT_ID}")

# --- Global Cache for Prompts DB Instances (managed by prompts_interop, but we track paths) ---
MAX_CACHED_PROMPTS_DB_INSTANCES = settings.get("MAX_CACHED_PROMPTS_DB_INSTANCES", 20)
_prompts_cache_lock = asyncio.Lock()
# INVARIANT: All mutations to _user_db_instances and _user_db_locks MUST be
# performed while holding _prompts_cache_lock. The eviction callbacks
# cross-reference each other's caches and assume this lock is held.
_pending_close_queue: asyncio.Queue[tuple[tuple[int, str], PromptsDatabase, str]] | None = None
_pending_close_loop: asyncio.AbstractEventLoop | None = None
_pending_close_task: asyncio.Task | None = None

# --- Helper Functions ---

def _get_prompts_db_path_for_user(user_id: int, salt: Optional[str] = None) -> Path:
    """
    Determines the Prompts database file path for a given user ID.
    Ensures the user's specific directory exists.
    Path: USER_DB_BASE_DIR / <user_id> / prompts_user_dbs / user_prompts_v2.sqlite
    When salt is provided: USER_DB_BASE_DIR / <user_id> / prompts_user_dbs / user_prompts_v2_{salt}.sqlite
    """
    db_file = DatabasePaths.get_prompts_db_path(user_id, salt=salt)
    logger.info(f"Ensured Prompts DB directory for user {user_id}: {db_file.parent}")
    return db_file

# --- Main Dependency Function ---

_user_db_instances: LRUCache | None = None
_user_db_locks: LRUCache | None = None


def _close_prompts_db_instance_sync(
    cache_key: tuple[int, str],
    db_instance: PromptsDatabase,
    *,
    reason: str,
) -> None:
    try:
        db_instance.close_connection()
        logger.info(
            'Closed PromptsDatabase instance for cache_key={} (reason={}).',
            cache_key,
            reason,
        )
    except (DatabaseError, OSError, RuntimeError, ValueError, TypeError) as exc:
        logger.error(
            "Failed to close PromptsDatabase instance",
            error_type=type(exc).__name__,
        )


def _enqueue_prompts_db_close(
    cache_key: tuple[int, str],
    db_instance: PromptsDatabase,
    *,
    reason: str,
) -> None:
    if not _ensure_pending_close_worker():
        _close_prompts_db_instance_sync(cache_key, db_instance, reason=reason)
        return
    try:
        _pending_close_queue.put_nowait((cache_key, db_instance, reason))
    except (asyncio.QueueFull, RuntimeError) as exc:
        logger.error(
            "Failed to enqueue PromptsDatabase close",
            error_type=type(exc).__name__,
        )


def _close_prompts_db_instance(
    cache_key: tuple[int, str],
    db_instance: PromptsDatabase,
    *,
    reason: str,
    run_sync: bool = False,
) -> None:
    if run_sync:
        _close_prompts_db_instance_sync(cache_key, db_instance, reason=reason)
        return
    _enqueue_prompts_db_close(cache_key, db_instance, reason=reason)


async def _process_pending_closes() -> None:
    global _pending_close_queue
    queue = _pending_close_queue
    if queue is None:
        return
    while True:
        try:
            cache_key, db_instance, reason = await queue.get()
        except asyncio.CancelledError:
            return
        try:
            await asyncio.to_thread(
                _close_prompts_db_instance_sync,
                cache_key,
                db_instance,
                reason=reason,
            )
        except asyncio.CancelledError:
            raise
        except (DatabaseError, OSError, RuntimeError, ValueError, TypeError) as exc:
            logger.error(
                "Failed to process pending PromptsDatabase close",
                error_type=type(exc).__name__,
            )
        finally:
            queue.task_done()


def _ensure_pending_close_worker() -> bool:
    global _pending_close_loop, _pending_close_queue, _pending_close_task
    if _pending_close_task is not None and not _pending_close_task.done():
        return True
    if _pending_close_task is not None and _pending_close_task.done():
        _pending_close_task = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop yet; worker will be started on first async entry.
        return False
    if _pending_close_queue is None or _pending_close_loop is not loop:
        _pending_close_queue = asyncio.Queue()
        _pending_close_loop = loop
    _pending_close_task = loop.create_task(_process_pending_closes())
    return True


def start_prompts_pending_close_worker() -> bool:
    """Start the async close worker when called from an active event loop."""
    started = _ensure_pending_close_worker()
    if not started:
        logger.debug("Prompts close worker startup deferred: no running event loop.")
    return started


async def stop_prompts_pending_close_worker() -> None:
    """Cancel the async close worker and clear the task reference."""
    global _pending_close_loop, _pending_close_queue, _pending_close_task
    task = _pending_close_task
    if task is None:
        _pending_close_queue = None
        _pending_close_loop = None
        return
    _pending_close_task = None
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    finally:
        _pending_close_queue = None
        _pending_close_loop = None


def _on_prompts_db_eviction(cache_key: tuple[int, str], db_instance: PromptsDatabase) -> None:
    if _user_db_locks is not None:
        # NOTE: LRUCache.pop() does not trigger eviction callbacks; safe while holding _prompts_cache_lock.
        _user_db_locks.pop(cache_key, None)
    _enqueue_prompts_db_close(cache_key, db_instance, reason="evicted")


def _on_prompts_lock_eviction(cache_key: tuple[int, str], _lock: asyncio.Lock) -> None:
    if _user_db_instances is None:
        return
    # NOTE: LRUCache.pop() does not trigger eviction callbacks; safe while holding _prompts_cache_lock.
    db_instance = _user_db_instances.pop(cache_key, None)
    if db_instance:
        _enqueue_prompts_db_close(cache_key, db_instance, reason="lock-evicted")


class _EvictingLRUCache(LRUCache):
    def __init__(self, *args, on_evict=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_evict = on_evict

    def popitem(self):
        cache_key, value = super().popitem()
        if self._on_evict is not None:
            try:
                self._on_evict(cache_key, value)
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
                logger.error(
                    "Prompts DB cache eviction callback failed",
                    error_type=type(exc).__name__,
                )
        return cache_key, value


_user_db_instances = _EvictingLRUCache(
    maxsize=MAX_CACHED_PROMPTS_DB_INSTANCES,
    on_evict=_on_prompts_db_eviction,
)
_user_db_locks = _EvictingLRUCache(
    maxsize=MAX_CACHED_PROMPTS_DB_INSTANCES,
    on_evict=_on_prompts_lock_eviction,
)


def _is_db_instance_alive(db_instance: PromptsDatabase) -> bool:
    try:
        conn = db_instance.get_connection()
        conn.execute("SELECT 1")
        return True
    except (DatabaseError, OSError, RuntimeError, ValueError):
        return False


def _create_prompts_db_instance(
    user_id: int,
    salt: Optional[str],
    client_id: str,
) -> tuple[PromptsDatabase, Path]:
    # Pass salt positionally so older test monkeypatches that still accept
    # (user_id, db_version) remain compatible.
    db_path = _get_prompts_db_path_for_user(user_id, salt)
    db_instance = PromptsDatabase(db_path=str(db_path), client_id=client_id)
    return db_instance, db_path


async def get_prompts_db_for_user(
        request: Request,
        current_user: User = Depends(get_request_user),
) -> PromptsDatabase:
    """
    FastAPI dependency to get the PromptsDatabase instance for the identified user,
    managed via the prompts_interop layer.
    """
    start_prompts_pending_close_worker()
    if _user_db_instances is None or _user_db_locks is None:
        logger.error("Prompts DB caches are not initialized.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prompts DB cache unavailable.",
        )
    # More robust check for User object and its id
    if not isinstance(current_user, User) or not hasattr(current_user, 'id') or not isinstance(current_user.id, int):
        logger.error(
            f"get_prompts_db_for_user called with an invalid User object. "
            f"Expected User model with int id. Got type: {type(current_user)}, value: {current_user}"
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="User identification failed for Prompts DB (Invalid User object).")

    user_id = current_user.id

    # In test mode, isolate DB per app instance to avoid cross-test conflicts
    salt = ""
    try:
        if is_test_mode():
            if not hasattr(request.app.state, "prompts_db_salt"):
                request.app.state.prompts_db_salt = _uuid.uuid4().hex
            salt = str(request.app.state.prompts_db_salt)
    except (AttributeError, RuntimeError, ValueError):
        salt = ""

    cache_key = (user_id, salt)

    # Get or create a lock for this specific user_id, then verify we hold
    # the current lock stored in the cache before proceeding.
    while True:
        async with _prompts_cache_lock:
            user_specific_lock = _user_db_locks.get(cache_key)
            if user_specific_lock is None:
                user_specific_lock = asyncio.Lock()
                _user_db_locks[cache_key] = user_specific_lock

        await user_specific_lock.acquire()
        async with _prompts_cache_lock:
            if _user_db_locks.get(cache_key) is user_specific_lock:
                break
        user_specific_lock.release()

    try:
        # Check if an instance for this user_id already exists (cached for this request/app lifetime if persistent)
        # The prompts_interop itself doesn't manage multiple DBs, it manages ONE.
        # So, we need a way to tell prompts_interop WHICH db to use, or manage instances here.
        # Let's manage instances here and pass the db_path to prompts_interop.
        # This means prompts_interop's global _db_instance is less useful in a multi-user, multi-db context.
        #
        # REVISED APPROACH:
        # The `prompts_interop` as written manages a SINGLE global instance.
        # For a multi-user system where each user has their OWN DB, we cannot use
        # the interop's global instance directly.
        #
        # Option 1: Modify `prompts_interop` to NOT use a global singleton, but return instances. (More work)
        # Option 2: Instantiate `PromptsDatabase` directly here, and wrap its calls.
        #           This bypasses the benefit of the interop being a single point of call.
        # Option 3: (Chosen for simplicity given current interop)
        #           The interop layer is more of a "wrapper" for the DB methods.
        #           We will instantiate PromptsDatabase directly here, per user.
        #           The `prompts_interop.py` utility functions that take `db_instance` can still be used.
        #           The interop's own instance-based methods will be bypassed.

        async with _prompts_cache_lock:
            try:
                db_instance = _user_db_instances[cache_key]
            except KeyError:
                db_instance = None
        if db_instance:
            is_alive = await asyncio.to_thread(_is_db_instance_alive, db_instance)
            if is_alive:
                logger.debug(f"Using cached PromptsDatabase instance for user_id: {user_id}")
                return db_instance
            logger.warning(f"Cached PromptsDatabase for user {user_id} inactive. Re-creating.")
            async with _prompts_cache_lock:
                _user_db_instances.pop(cache_key, None)
            _close_prompts_db_instance(cache_key, db_instance, reason="inactive")

        # If not cached or cache was bad, create a new one
        db_path: Optional[Path] = None
        try:
            # Call with positional args to be compatible with test monkeypatches
            db_instance, db_path = await asyncio.to_thread(
                _create_prompts_db_instance,
                user_id,
                salt or None,
                SERVER_CLIENT_ID,
            )
            logger.info(f"Initializing PromptsDatabase instance for user {user_id} at path: {db_path}")

            # Instantiate PromptsDatabase directly
            # The client_id for the PromptsDatabase should be the SERVER_CLIENT_ID,
            # as it's the server application making changes on behalf of the user.
            # If you need to track the specific end-user initiating the change,
            # that would be a different field, or SERVER_CLIENT_ID could be user-specific.
            # For now, using a global server client ID.
            async with _prompts_cache_lock:
                _user_db_instances[cache_key] = db_instance # Cache it for this user/app salt
            logger.info(f"PromptsDatabase instance created and cached for user {user_id} (salt={salt or 'none'}) at {db_path}")
            return db_instance

        except (DatabaseError, SchemaError, InputError, ConflictError) as e:
            logger.error(
                "Failed to initialize PromptsDatabase",
                user_id=user_id,
                error_type=type(e).__name__,
            )
            raise map_db_error_to_http(
                e,
                default_detail="Prompts DB unavailable",
                input_status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                conflict_status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                conflict_detail="Prompts DB unavailable",
            ) from e
        except OSError as e:
            logger.error(
                "Failed to get PromptsDatabase path",
                user_id=user_id,
                error_type=type(e).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prompts DB unavailable",
            ) from e
        except Exception as e:
            logger.error(
                "Unexpected error initializing PromptsDatabase",
                user_id=user_id,
                error_type=type(e).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during prompts database setup."
            ) from e
    finally:
        user_specific_lock.release()


async def close_all_cached_prompts_db_instances() -> None:
    """Closes all cached PromptsDatabase connections. Useful for application shutdown."""
    if _user_db_instances is None or _user_db_locks is None:
        return
    async with _prompts_cache_lock:
        logger.info(f"Closing all cached PromptsDatabase instances ({len(_user_db_instances)})...")
        for cache_key, db_instance in list(_user_db_instances.items()):
            try:
                await asyncio.to_thread(db_instance.close_connection)
                logger.info(f"Closed PromptsDatabase connection for cache_key {cache_key}.")
            except (DatabaseError, OSError, RuntimeError, ValueError, TypeError) as e:
                logger.error(
                    "Failed to close cached PromptsDatabase instance",
                    error_type=type(e).__name__,
                )
        _user_db_instances.clear()
        _user_db_locks.clear()  # Clear user-specific locks as well
        logger.info("All PromptsDatabase instances cleared from cache and locks removed.")

# Register for shutdown in your main FastAPI app:
# @app.on_event("shutdown")
# async def shutdown_event():
#     await close_all_cached_prompts_db_instances()

#
# End of Prompts_DB_Deps.py
########################################################################################################################
