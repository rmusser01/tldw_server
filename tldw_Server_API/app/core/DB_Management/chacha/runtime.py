from __future__ import annotations

import asyncio
import faulthandler
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from cachetools import LRUCache
from loguru import logger

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
from tldw_Server_API.app.core.config import settings

DEFAULT_CHARACTER_NAME = "Helpful AI Assistant"
DEFAULT_CHARACTER_DESCRIPTION = "A default, friendly assistant created automatically by the system."
MAX_CACHED_CHACHA_DB_INSTANCES = int(settings.get("MAX_CACHED_CHACHA_DB_INSTANCES", "20"))
_CHACHA_EXECUTOR_MAX_WORKERS = max(1, int(os.getenv("CHACHA_EXECUTOR_MAX_WORKERS", "4")))
_CHACHA_WATCHDOG_SECS = float(os.getenv("CHACHA_INIT_WATCHDOG_SECS", "5"))


class ChaChaRuntimeUnavailableError(RuntimeError):
    """Transport-neutral runtime failure for callers that translate elsewhere."""


class ChaChaRuntimeInitError(RuntimeError):
    """Transport-neutral runtime failure for initialization errors."""


class _ChaChaRuntimeState:
    def __init__(self) -> None:
        self.executor: ThreadPoolExecutor | None = None
        self.executor_shutdown = False
        self.executor_lock = threading.Lock()
        self.health_lock = threading.Lock()
        self.health: dict[str, Any] = {
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
        self.shutting_down = False
        self.shutdown_lock = threading.Lock()
        self.cache: LRUCache[str, CharactersRAGDB] = LRUCache(maxsize=MAX_CACHED_CHACHA_DB_INSTANCES)
        self.db_lock = threading.Lock()
        self.init_events: dict[str, threading.Event] = {}
        self.init_errors: dict[str, Exception] = {}
        self.default_char_tasks: set[asyncio.Task[Any]] = set()
        self.default_char_futures: set[asyncio.Future[Any]] = set()
        self.default_char_futures_lock = threading.Lock()


_STATE = _ChaChaRuntimeState()


def _set_shutting_down(value: bool) -> None:
    with _STATE.shutdown_lock:
        _STATE.shutting_down = value


def _is_shutting_down() -> bool:
    with _STATE.shutdown_lock:
        return _STATE.shutting_down


def _get_executor() -> ThreadPoolExecutor:
    with _STATE.executor_lock:
        if _STATE.executor is None or _STATE.executor_shutdown:
            _STATE.executor = ThreadPoolExecutor(
                max_workers=_CHACHA_EXECUTOR_MAX_WORKERS,
                thread_name_prefix="chacha-db",
            )
            _STATE.executor_shutdown = False
        return _STATE.executor


def _record_init(duration_ms: float, success: bool, error: Exception | None = None) -> None:
    with _STATE.health_lock:
        _STATE.health["init_attempts"] += 1
        _STATE.health["last_init_ms"] = duration_ms
        _STATE.health["cached_instances"] = len(_STATE.cache)
        if success:
            _STATE.health["last_error"] = None
        else:
            _STATE.health["init_failures"] += 1
            _STATE.health["last_error"] = str(error) if error else "unknown error"


def _record_default_character(success: bool) -> None:
    with _STATE.health_lock:
        if success:
            _STATE.health["default_char_ensures"] += 1
        else:
            _STATE.health["default_char_failures"] += 1


def _maybe_dump_traceback(reason: str) -> None:
    now = time.time()
    last_dump = _STATE.health.get("last_warn_dump")
    if last_dump and now - float(last_dump) < 300:
        return
    with _STATE.health_lock:
        _STATE.health["last_warn_dump"] = now
    try:
        logger.warning("ChaChaNotes watchdog dump triggered: {}", reason)
        faulthandler.dump_traceback(file=sys.stderr)
    except (OSError, RuntimeError, ValueError) as dump_err:
        logger.debug("Faulthandler dump failed: {}", dump_err)


def _track_default_character_future(future: asyncio.Future[Any]) -> None:
    def _cleanup(_future: asyncio.Future[Any]) -> None:
        with _STATE.default_char_futures_lock:
            _STATE.default_char_futures.discard(_future)

    with _STATE.default_char_futures_lock:
        _STATE.default_char_futures.add(future)
    future.add_done_callback(_cleanup)


def _get_chacha_db_path_for_user(user_id: int) -> Path:
    db_file = DatabasePaths.get_chacha_db_path(user_id)
    logger.info("Ensured ChaChaNotes DB directory for user {}: {}", user_id, db_file.parent)
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
        logger.debug("ChaChaNotes tuning skipped: {}", e)


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
        logger.warning("ChaChaNotes health probe failed: {}", e)
        return False


def _create_and_prepare_db(user_id: int, client_id: str) -> CharactersRAGDB:
    db_path = _get_chacha_db_path_for_user(user_id)
    try:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    except OSError as mk_err:
        logger.debug("Secondary ensure for ChaChaNotes parent failed softly: {}", mk_err)
    logger.info("Initializing CharactersRAGDB instance for user {} at path: {}", user_id, db_path)
    db_instance = CharactersRAGDB(db_path=str(db_path), client_id=str(client_id))
    _apply_sqlite_tuning(db_instance)
    return db_instance


def _ensure_default_character(db_instance: CharactersRAGDB) -> int | None:
    try:
        db_instance.ensure_character_tables_ready()
        default_char = db_instance.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
        if default_char:
            logger.debug(
                "Default character '{}' already exists with ID: {}.",
                DEFAULT_CHARACTER_NAME,
                default_char["id"],
            )
            return default_char["id"]
        logger.info("Default character '{}' not found. Creating now...", DEFAULT_CHARACTER_NAME)
        card_data = {
            "name": DEFAULT_CHARACTER_NAME,
            "description": DEFAULT_CHARACTER_DESCRIPTION,
            "personality": "Supportive, patient, and concise.",
            "scenario": "General assistance",
            "system_prompt": "You are a helpful AI assistant.",
            "image": None,
            "post_history_instructions": None,
            "first_message": "Hello! I'm your Helpful AI Assistant. How can I support you today?",
            "message_example": None,
            "creator_notes": "This character is automatically generated to provide a reliable default assistant persona.",
            "alternate_greetings": None,
            "tags": json.dumps(["default", "neutral", "assistant"]),
            "creator": "System",
            "character_version": "1.0",
            "extensions": None,
            "client_id": db_instance.client_id,
        }
        char_id = db_instance.add_character_card(card_data)
        if char_id:
            logger.info("Successfully created default character '{}' with ID: {}.", DEFAULT_CHARACTER_NAME, char_id)
            return char_id
        logger.error("Failed to create default character '{}'. add_character_card returned None.", DEFAULT_CHARACTER_NAME)
        return None
    except ConflictError as e:
        logger.warning("Conflict error while ensuring default character (likely race condition, re-fetching): {}", e)
        refetched_char = db_instance.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
        if refetched_char:
            return refetched_char["id"]
        logger.error("Still could not get/create default character after conflict: {}", e)
        return None
    except (CharactersRAGDBError, SchemaError, InputError) as e:
        logger.error("Database error while ensuring default character '{}': {}", DEFAULT_CHARACTER_NAME, e, exc_info=True)
        return None
    except (AttributeError, KeyError, TypeError, ValueError, OSError, RuntimeError) as e_gen:
        logger.error(
            "Unexpected error while ensuring default character '{}': {}",
            DEFAULT_CHARACTER_NAME,
            e_gen,
            exc_info=True,
        )
        return None


async def _ensure_default_character_async(db_instance: CharactersRAGDB, user_id: int) -> None:
    loop = asyncio.get_running_loop()
    try:
        future = loop.run_in_executor(_get_executor(), _ensure_default_character, db_instance)
        _track_default_character_future(future)
        await asyncio.wait_for(asyncio.shield(future), timeout=5)
        _record_default_character(True)
    except asyncio.TimeoutError:
        _record_default_character(False)
        logger.warning("Timed out ensuring default character for user {}; will retry on next access.", user_id)
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
            "Error ensuring default character for user {}: {}. Continuing; will retry on next access.",
            user_id,
            e,
            exc_info=True,
        )


async def _is_instance_healthy(db_instance: CharactersRAGDB) -> bool:
    try:
        result = await asyncio.wait_for(asyncio.to_thread(_health_check_instance, db_instance), timeout=1.0)
        return bool(result)
    except (asyncio.TimeoutError, OSError, RuntimeError):
        return False


async def _get_or_init_db_instance(user_id: int, client_id: str) -> CharactersRAGDB:
    if _is_shutting_down():
        raise ChaChaRuntimeUnavailableError("ChaChaNotes shutdown in progress")
    user_dir = DatabasePaths.get_user_base_directory(user_id)
    cache_key = str(user_dir)
    with _STATE.db_lock:
        db_instance = _STATE.cache.get(cache_key)
    if db_instance:
        if await _is_instance_healthy(db_instance):
            return db_instance
        logger.warning("ChaChaNotes cached instance unhealthy for user {}; evicting and rebuilding.", user_id)
        with _STATE.db_lock:
            if _STATE.cache.get(cache_key) is db_instance:
                _STATE.cache.pop(cache_key, None)

    wait_for_existing_init = False
    with _STATE.db_lock:
        cached_instance = _STATE.cache.get(cache_key)
        if cached_instance is not None:
            _STATE.health["cached_instances"] = len(_STATE.cache)
            return cached_instance
        init_event = _STATE.init_events.get(cache_key)
        if init_event is None:
            init_event = threading.Event()
            _STATE.init_events[cache_key] = init_event
        else:
            wait_for_existing_init = True

    if wait_for_existing_init:
        wait_timeout = max(_CHACHA_WATCHDOG_SECS * 3, 5)
        try:
            completed = await asyncio.wait_for(asyncio.to_thread(init_event.wait), timeout=wait_timeout)
        except asyncio.TimeoutError as e:
            _record_init(wait_timeout * 1000, False, e)
            raise ChaChaRuntimeUnavailableError("ChaChaNotes initialization timed out") from e
        if not completed:
            _record_init(wait_timeout * 1000, False)
            raise ChaChaRuntimeUnavailableError("ChaChaNotes initialization timed out")
        with _STATE.db_lock:
            cached_instance = _STATE.cache.get(cache_key)
            init_error = _STATE.init_errors.get(cache_key)
        if cached_instance is not None:
            return cached_instance
        if init_error is not None:
            raise ChaChaRuntimeInitError(
                f"Could not initialize character & notes database for user: {init_error}"
            ) from init_error
        raise ChaChaRuntimeInitError("Could not initialize character & notes database for user: unknown error")

    loop = asyncio.get_running_loop()
    start = time.perf_counter()
    try:
        db_instance = await asyncio.wait_for(
            loop.run_in_executor(_get_executor(), _create_and_prepare_db, user_id, client_id),
            timeout=max(_CHACHA_WATCHDOG_SECS * 3, 5),
        )
        duration_ms = (time.perf_counter() - start) * 1000
        _record_init(duration_ms, True)
        if duration_ms / 1000 > _CHACHA_WATCHDOG_SECS:
            _maybe_dump_traceback(f"ChaChaNotes init exceeded {_CHACHA_WATCHDOG_SECS}s for user {user_id}")
    except asyncio.TimeoutError as e:
        _record_init(_CHACHA_WATCHDOG_SECS * 1000, False, e)
        _maybe_dump_traceback(f"ChaChaNotes init timed out for user {user_id}")
        raise ChaChaRuntimeUnavailableError("ChaChaNotes initialization timed out") from e
    except (CharactersRAGDBError, sqlite3.Error, OSError, RuntimeError, ValueError, TypeError) as e:
        duration_ms = (time.perf_counter() - start) * 1000
        _record_init(duration_ms, False, e)
        with _STATE.db_lock:
            _STATE.init_errors[cache_key] = e
        raise ChaChaRuntimeInitError(f"Could not initialize character & notes database for user: {e}") from e
    else:
        with _STATE.db_lock:
            _STATE.cache[cache_key] = db_instance
            _STATE.init_errors.pop(cache_key, None)
            _STATE.health["cached_instances"] = len(_STATE.cache)
        return db_instance
    finally:
        with _STATE.db_lock:
            init_event = _STATE.init_events.pop(cache_key, None)
        if init_event is not None:
            init_event.set()


async def _warm_chacha_db_for_user(user_id: int, client_id: str | None = None) -> None:
    if _is_shutting_down():
        logger.debug("ChaChaNotes shutdown in progress; skipping warmup for user {}", user_id)
        return
    try:
        db_instance = await _get_or_init_db_instance(user_id, client_id or str(user_id))
        _STATE.health["warm_startups"] += 1
        schedule_default_character_ensure(db_instance, user_id)
    except (ChaChaRuntimeUnavailableError, OSError, RuntimeError, ValueError, TypeError) as e:
        logger.warning("Warm-up for ChaChaNotes user {} failed: {}", user_id, e)


def schedule_default_character_ensure(db_instance: CharactersRAGDB, user_id: int) -> asyncio.Task[Any] | None:
    if _is_shutting_down():
        logger.debug("ChaChaNotes shutdown in progress; skipping default-character ensure for user {}", user_id)
        return None
    task = asyncio.create_task(_ensure_default_character_async(db_instance, user_id))
    _STATE.default_char_tasks.add(task)
    task.add_done_callback(_STATE.default_char_tasks.discard)
    return task


def _close_all_instances() -> None:
    with _STATE.db_lock:
        logger.info("Closing all cached ChaChaNotesDB instances ({})...", len(_STATE.cache))
        for user_id, db_instance in list(_STATE.cache.items()):
            try:
                db_instance.close_all_connections()
                logger.info("Closed ChaChaNotesDB instance for user {}.", user_id)
            except (CharactersRAGDBError, OSError, RuntimeError, ValueError, TypeError) as e:
                logger.error("Error closing ChaChaNotesDB instance for user {}: {}", user_id, e, exc_info=True)
        _STATE.cache.clear()
        _STATE.init_errors.clear()
        logger.info("All ChaChaNotesDB instances closed and cache cleared.")


async def _drain_default_character_tasks(timeout: float = 5.0) -> None:
    tasks = [task for task in list(_STATE.default_char_tasks) if not task.done()]
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
    _STATE.default_char_tasks.difference_update(done)
    _STATE.default_char_tasks.difference_update(pending)


async def _drain_default_character_futures(timeout: float = 5.0) -> None:
    with _STATE.default_char_futures_lock:
        futures = [future for future in list(_STATE.default_char_futures) if not future.done()]
    if not futures:
        return
    done, pending = await asyncio.wait(futures, timeout=timeout)
    if pending:
        logger.warning(
            "ChaChaNotes shutdown: {} default-character futures still running; waiting on executor shutdown.",
            len(pending),
        )
    with _STATE.default_char_futures_lock:
        for future in done:
            _STATE.default_char_futures.discard(future)


def _shutdown_executor(wait: bool = False) -> None:
    with _STATE.executor_lock:
        executor = _STATE.executor
        _STATE.executor = None
        _STATE.executor_shutdown = True
    if executor is None:
        return
    try:
        executor.shutdown(wait=wait, cancel_futures=True)
    except (RuntimeError, OSError, ValueError) as e:
        logger.debug("ChaChaNotes executor shutdown error: {}", e)


async def _shutdown(wait_timeout: float = 5.0) -> None:
    _set_shutting_down(True)
    await _drain_default_character_tasks(timeout=wait_timeout)
    await _drain_default_character_futures(timeout=wait_timeout)
    _shutdown_executor(wait=True)
    _close_all_instances()


def _snapshot() -> dict[str, Any]:
    with _STATE.health_lock:
        status = "healthy"
        if _STATE.health.get("init_failures"):
            status = "degraded"
        return {
            "status": status,
            "init_attempts": _STATE.health.get("init_attempts"),
            "init_failures": _STATE.health.get("init_failures"),
            "last_init_ms": _STATE.health.get("last_init_ms"),
            "last_error": _STATE.health.get("last_error"),
            "cached_instances": len(_STATE.cache),
            "default_char_ensures": _STATE.health.get("default_char_ensures"),
            "default_char_failures": _STATE.health.get("default_char_failures"),
            "warm_startups": _STATE.health.get("warm_startups"),
            "shutting_down": _is_shutting_down(),
            "pending_init_events": len(_STATE.init_events),
            "default_char_tasks": len(_STATE.default_char_tasks),
            "default_char_futures": len(_STATE.default_char_futures),
            "executor_shutdown": _STATE.executor_shutdown,
        }


def _reset_for_tests() -> None:
    _set_shutting_down(False)
    _shutdown_executor(wait=False)
    _close_all_instances()
    with _STATE.db_lock:
        _STATE.init_events.clear()
    with _STATE.default_char_futures_lock:
        _STATE.default_char_futures.clear()
    _STATE.default_char_tasks.clear()
    with _STATE.health_lock:
        _STATE.health.update(
            {
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
        )


class ChaChaRuntimeManager:
    async def get_or_create(self, user_id: int, client_id: str | None) -> CharactersRAGDB:
        return await _get_or_init_db_instance(user_id, client_id or str(user_id))

    async def warm_for_user(self, user_id: int, client_id: str | None = None) -> None:
        await _warm_chacha_db_for_user(user_id, client_id)

    def schedule_default_character_ensure(
        self, db_instance: CharactersRAGDB, user_id: int
    ) -> asyncio.Task[Any] | None:
        return schedule_default_character_ensure(db_instance, user_id)

    def is_shutting_down(self) -> bool:
        return _is_shutting_down()

    async def shutdown(self, wait_timeout: float = 5.0) -> None:
        await _shutdown(wait_timeout=wait_timeout)

    def shutdown_executor(self, wait: bool = False) -> None:
        _shutdown_executor(wait=wait)

    def snapshot(self) -> dict[str, Any]:
        return _snapshot()

    def reset_for_tests(self) -> None:
        _reset_for_tests()

    def close_all_instances(self) -> None:
        _close_all_instances()


__all__ = [
    "ChaChaRuntimeManager",
    "ChaChaRuntimeInitError",
    "ChaChaRuntimeUnavailableError",
    "DEFAULT_CHARACTER_DESCRIPTION",
    "DEFAULT_CHARACTER_NAME",
    "MAX_CACHED_CHACHA_DB_INSTANCES",
    "_apply_sqlite_tuning",
    "_health_check_instance",
    "schedule_default_character_ensure",
]
