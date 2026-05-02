"""
Cleanup-worker startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.services.worker_registry import (
    ManagedWorker,
    ShutdownPhase,
    start_stop_event_worker,
)

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_TRUTHY_VALUES = {"true", "1", "yes", "y", "on"}


@dataclass
class CleanupWorkerHandles:
    """Startup-owned cleanup worker resources that still require shutdown handling."""

    cleanup_task: Any | None = None
    chatbooks_cleanup_task: Any | None = None
    chatbooks_cleanup_stop_event: Any | None = None
    storage_cleanup_service: Any | None = None


async def start_cleanup_workers(
    app_settings: Mapping[str, Any],
    *,
    test_mode: bool,
    worker_inventory: Any | None = None,
) -> CleanupWorkerHandles:
    """Start the small cleanup-worker startup slice and return explicit handles."""
    cleanup_task = await _start_ephemeral_cleanup_worker(
        app_settings,
        worker_inventory=worker_inventory,
    )
    secondary = await _start_secondary_cleanup_workers(
        test_mode=test_mode,
        worker_inventory=worker_inventory,
    )
    return CleanupWorkerHandles(
        cleanup_task=cleanup_task,
        chatbooks_cleanup_task=secondary.chatbooks_cleanup_task,
        chatbooks_cleanup_stop_event=secondary.chatbooks_cleanup_stop_event,
        storage_cleanup_service=secondary.storage_cleanup_service,
    )


async def _start_secondary_cleanup_workers(
    *,
    test_mode: bool,
    worker_inventory: Any | None = None,
) -> CleanupWorkerHandles:
    """Start cleanup workers whose enablement is driven by process env."""
    chatbooks_cleanup_task, chatbooks_cleanup_stop_event = await _start_chatbooks_cleanup_worker(
        worker_inventory=worker_inventory,
    )
    storage_cleanup_service = await _start_storage_cleanup_worker(
        test_mode=test_mode,
        worker_inventory=worker_inventory,
    )
    return CleanupWorkerHandles(
        chatbooks_cleanup_task=chatbooks_cleanup_task,
        chatbooks_cleanup_stop_event=chatbooks_cleanup_stop_event,
        storage_cleanup_service=storage_cleanup_service,
    )


async def _start_ephemeral_cleanup_worker(
    app_settings: Mapping[str, Any],
    *,
    worker_inventory: Any | None = None,
) -> Any | None:
    """Start the ephemeral collections cleanup loop when enabled."""
    try:
        single_uid, db_path, interval_sec = _resolve_ephemeral_cleanup_config(app_settings)
        enabled = _settings_truthy(app_settings.get("EPHEMERAL_CLEANUP_ENABLED", True))

        if enabled:
            if worker_inventory is not None:
                task, _stop_event = await start_stop_event_worker(
                    worker_inventory,
                    name="ephemeral_cleanup_task",
                    task_name="ephemeral_cleanup_task",
                    coroutine_factory=lambda stop_event: _run_ephemeral_cleanup_loop(
                        app_settings,
                        single_uid=single_uid,
                        db_path=db_path,
                        interval_sec=interval_sec,
                        stop_event=stop_event,
                    ),
                    category="cleanup",
                    shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
                )
                return task
            return asyncio.create_task(
                _run_ephemeral_cleanup_loop(
                    app_settings,
                    single_uid=single_uid,
                    db_path=db_path,
                    interval_sec=interval_sec,
                ),
                name="ephemeral_cleanup_task",
            )
        logger.info("Ephemeral cleanup worker disabled by settings")
        return None
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start ephemeral cleanup worker: {exc}")
        return None


async def _run_ephemeral_cleanup_loop(
    app_settings: Mapping[str, Any],
    *,
    single_uid: int | None = None,
    db_path: str | None = None,
    interval_sec: int | None = None,
    stop_event: Any | None = None,
) -> None:
    """Run ephemeral collection cleanup until cancelled or the stop event is set."""
    if _stop_requested(stop_event):
        return

    if single_uid is None or db_path is None or interval_sec is None:
        single_uid, db_path, interval_sec = _resolve_ephemeral_cleanup_config(app_settings)
    user_id = str(single_uid)
    logger.info(f"Starting ephemeral collections cleanup worker (every {interval_sec}s)")
    db = _create_evaluations_db(db_path)
    adapter = _create_vector_store_adapter(app_settings, user_id)
    await _maybe_await(getattr(adapter, "initialize", lambda: None)())
    while not _stop_requested(stop_event):
        sleep_for = interval_sec
        try:
            enabled_dyn = _settings_truthy(app_settings.get("EPHEMERAL_CLEANUP_ENABLED", True))
            interval_dyn = int(app_settings.get("EPHEMERAL_CLEANUP_INTERVAL_SEC", interval_sec))
            sleep_for = interval_dyn if enabled_dyn else interval_sec
            if enabled_dyn:
                expired = db.list_expired_ephemeral_collections()
                if expired:
                    deleted = 0
                    for collection_name in expired:
                        if _stop_requested(stop_event):
                            logger.info("Ephemeral cleanup: stop requested; exiting delete batch early")
                            break
                        try:
                            await _maybe_await(adapter.delete_collection(collection_name))
                            db.mark_ephemeral_deleted(collection_name)
                            deleted += 1
                        except _STARTUP_GUARD_EXCEPTIONS as exc:
                            logger.warning(f"Ephemeral cleanup: failed to delete {collection_name}: {exc}")
                    if deleted:
                        logger.info(f"Ephemeral cleanup: deleted {deleted}/{len(expired)} expired collections")
        except _STARTUP_GUARD_EXCEPTIONS as exc:
            logger.warning(f"Ephemeral cleanup loop error: {exc}")
        await _sleep_or_stop(stop_event, sleep_for)


async def _start_chatbooks_cleanup_worker(
    *,
    worker_inventory: Any | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the scheduled chatbooks cleanup worker when enabled."""
    try:
        interval_sec = int(os.getenv("CHATBOOKS_CLEANUP_INTERVAL_SEC", "0") or "0")
        if interval_sec > 0:
            if worker_inventory is not None:
                task, stop_event = await start_stop_event_worker(
                    worker_inventory,
                    name="chatbooks_cleanup",
                    task_name="chatbooks_cleanup_task",
                    coroutine_factory=_run_chatbooks_cleanup_loop,
                    category="cleanup",
                    shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
                )
            else:
                stop_event = asyncio.Event()
                task = asyncio.create_task(
                    _run_chatbooks_cleanup_loop(stop_event),
                    name="chatbooks_cleanup_task",
                )
            logger.info("Chatbooks cleanup worker started")
            return task, stop_event
        logger.info("Chatbooks cleanup worker disabled by settings")
        return None, None
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start chatbooks cleanup worker: {exc}")
        return None, None


async def _start_storage_cleanup_worker(
    *,
    test_mode: bool,
    worker_inventory: Any | None = None,
) -> Any | None:
    """Start the storage cleanup worker when enabled by the current mode/env."""
    try:
        storage_cleanup_default = "false" if test_mode else "true"
        storage_cleanup_enabled = os.getenv("STORAGE_CLEANUP_ENABLED", storage_cleanup_default).lower() in {
            "true",
            "1",
            "yes",
            "y",
            "on",
        }
        if storage_cleanup_enabled:
            storage_cleanup_service = _get_storage_cleanup_service()
            await storage_cleanup_service.start()
            if worker_inventory is not None:
                await _register_storage_cleanup_service(
                    worker_inventory=worker_inventory,
                    storage_cleanup_service=storage_cleanup_service,
                )
            logger.info("Storage cleanup worker started")
            return storage_cleanup_service
        logger.info("Storage cleanup worker disabled by settings")
        return None
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start storage cleanup worker: {exc}")
        return None


async def _register_storage_cleanup_service(
    *,
    worker_inventory: Any,
    storage_cleanup_service: Any,
) -> None:
    """Register a started storage cleanup service with lifecycle inventory."""
    task = _get_storage_cleanup_task(storage_cleanup_service)
    if task is None:
        logger.warning("Storage cleanup worker started without a task handle; lifecycle inventory skipped")
        return
    try:
        worker_inventory.register(
            ManagedWorker(
                name="storage_cleanup_service",
                task=task,
                stop_event=None,
                shutdown_callback=storage_cleanup_service.stop,
                category="cleanup",
                shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            )
        )
    except _STARTUP_GUARD_EXCEPTIONS:
        await _stop_started_storage_cleanup_service(storage_cleanup_service)
        raise


def _get_storage_cleanup_task(storage_cleanup_service: Any) -> Any | None:
    """Return the public or legacy background task handle from the cleanup service."""
    task = getattr(storage_cleanup_service, "task", None)
    if task is not None:
        return task
    return getattr(storage_cleanup_service, "_task", None)


async def _stop_started_storage_cleanup_service(storage_cleanup_service: Any) -> None:
    """Best-effort rollback for a service that started but failed inventory registration."""
    try:
        await storage_cleanup_service.stop()
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.debug(f"Storage cleanup startup rollback stop failed: {exc}")


async def _maybe_await(value: Any) -> Any:
    """Await values that are awaitable and return plain values unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value


def _stop_requested(stop_event: Any | None) -> bool:
    """Return whether a lifecycle stop event has been signaled."""
    if stop_event is None:
        return False
    is_set = getattr(stop_event, "is_set", None)
    if not callable(is_set):
        return False
    return bool(is_set())


async def _sleep_or_stop(stop_event: Any | None, delay: int) -> None:
    """Sleep for delay seconds, returning early when a stop event is signaled."""
    if stop_event is None:
        await asyncio.sleep(delay)
        return
    wait = getattr(stop_event, "wait", None)
    if not callable(wait):
        await asyncio.sleep(delay)
        return
    try:
        await asyncio.wait_for(wait(), timeout=delay)
    except asyncio.TimeoutError:
        pass


def _settings_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY_VALUES
    return bool(value)


def _resolve_ephemeral_cleanup_config(app_settings: Mapping[str, Any]) -> tuple[int, str, int]:
    """Resolve and validate the ephemeral cleanup user, DB path, and interval."""
    single_uid = int(app_settings.get("SINGLE_USER_FIXED_ID", "1"))
    db_path = str(_get_evaluations_db_path(single_uid))
    interval_sec = int(app_settings.get("EPHEMERAL_CLEANUP_INTERVAL_SEC", 1800))
    return single_uid, db_path, interval_sec


def _create_evaluations_db(db_path: str) -> Any:
    from tldw_Server_API.app.core.DB_Management.DB_Manager import (
        create_evaluations_database as _create_evals_db,
    )

    return _create_evals_db(db_path=db_path)


def _get_evaluations_db_path(user_id: int) -> Any:
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths as _DBP

    return _DBP.get_evaluations_db_path(user_id)


def _create_vector_store_adapter(app_settings: Mapping[str, Any], user_id: str) -> Any:
    from tldw_Server_API.app.core.RAG.rag_service.vector_stores import (
        create_from_settings_for_user as _create_vs_from_settings,
    )

    return _create_vs_from_settings(app_settings, user_id)


async def _run_chatbooks_cleanup_loop(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.chatbooks_cleanup_service import (
        run_chatbooks_cleanup_loop as _run_chatbooks_cleanup,
    )

    return await _run_chatbooks_cleanup(stop_event)


def _get_storage_cleanup_service() -> Any:
    from tldw_Server_API.app.services.storage_cleanup_service import (
        get_cleanup_service as _get_storage_cleanup,
    )

    return _get_storage_cleanup()
