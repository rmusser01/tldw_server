"""
Cleanup-worker startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from dataclasses import dataclass
from typing import Any, Mapping

from loguru import logger

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


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
) -> CleanupWorkerHandles:
    """Start the small cleanup-worker startup slice and return explicit handles."""
    cleanup_task = await _start_ephemeral_cleanup_worker(app_settings)
    secondary = await _start_secondary_cleanup_workers(test_mode=test_mode)
    return CleanupWorkerHandles(
        cleanup_task=cleanup_task,
        chatbooks_cleanup_task=secondary.chatbooks_cleanup_task,
        chatbooks_cleanup_stop_event=secondary.chatbooks_cleanup_stop_event,
        storage_cleanup_service=secondary.storage_cleanup_service,
    )


async def _start_secondary_cleanup_workers(*, test_mode: bool) -> CleanupWorkerHandles:
    """Start cleanup workers whose enablement is driven by process env."""
    chatbooks_cleanup_task, chatbooks_cleanup_stop_event = await _start_chatbooks_cleanup_worker()
    storage_cleanup_service = await _start_storage_cleanup_worker(test_mode=test_mode)
    return CleanupWorkerHandles(
        chatbooks_cleanup_task=chatbooks_cleanup_task,
        chatbooks_cleanup_stop_event=chatbooks_cleanup_stop_event,
        storage_cleanup_service=storage_cleanup_service,
    )


async def _start_ephemeral_cleanup_worker(app_settings: Mapping[str, Any]) -> Any | None:
    """Start the ephemeral collections cleanup loop when enabled."""
    try:
        single_uid = int(app_settings.get("SINGLE_USER_FIXED_ID", "1"))
        db_path = str(_get_evaluations_db_path(single_uid))
        enabled = bool(app_settings.get("EPHEMERAL_CLEANUP_ENABLED", True))
        interval_sec = int(app_settings.get("EPHEMERAL_CLEANUP_INTERVAL_SEC", 1800))

        async def _ephemeral_cleanup_loop() -> None:
            logger.info(f"Starting ephemeral collections cleanup worker (every {interval_sec}s)")
            db = _create_evaluations_db(db_path)
            adapter = _create_vector_store_adapter(app_settings, str(app_settings.get("SINGLE_USER_FIXED_ID", "1")))
            await _maybe_await(getattr(adapter, "initialize", lambda: None)())
            while True:
                try:
                    enabled_dyn = bool(app_settings.get("EPHEMERAL_CLEANUP_ENABLED", True))
                    interval_dyn = int(app_settings.get("EPHEMERAL_CLEANUP_INTERVAL_SEC", interval_sec))
                    if not enabled_dyn:
                        await asyncio.sleep(interval_sec)
                        continue
                    expired = db.list_expired_ephemeral_collections()
                    if expired:
                        deleted = 0
                        for collection_name in expired:
                            try:
                                await _maybe_await(adapter.delete_collection(collection_name))
                                db.mark_ephemeral_deleted(collection_name)
                                deleted += 1
                            except _STARTUP_GUARD_EXCEPTIONS as exc:
                                logger.warning(f"Ephemeral cleanup: failed to delete {collection_name}: {exc}")
                        if deleted:
                            logger.info(
                                f"Ephemeral cleanup: deleted {deleted}/{len(expired)} expired collections"
                            )
                except _STARTUP_GUARD_EXCEPTIONS as exc:
                    logger.warning(f"Ephemeral cleanup loop error: {exc}")
                await asyncio.sleep(interval_dyn)

        if enabled:
            return asyncio.create_task(_ephemeral_cleanup_loop())
        logger.info("Ephemeral cleanup worker disabled by settings")
        return None
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start ephemeral cleanup worker: {exc}")
        return None


async def _start_chatbooks_cleanup_worker() -> tuple[Any | None, Any | None]:
    """Start the scheduled chatbooks cleanup worker when enabled."""
    try:
        interval_sec = int(os.getenv("CHATBOOKS_CLEANUP_INTERVAL_SEC", "0") or "0")
        if interval_sec > 0:
            stop_event = asyncio.Event()
            task = asyncio.create_task(_run_chatbooks_cleanup_loop(stop_event))
            logger.info("Chatbooks cleanup worker started")
            return task, stop_event
        logger.info("Chatbooks cleanup worker disabled by settings")
        return None, None
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start chatbooks cleanup worker: {exc}")
        return None, None


async def _start_storage_cleanup_worker(*, test_mode: bool) -> Any | None:
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
            logger.info("Storage cleanup worker started")
            return storage_cleanup_service
        logger.info("Storage cleanup worker disabled by settings")
        return None
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start storage cleanup worker: {exc}")
        return None


async def _maybe_await(value: Any) -> Any:
    """Await values that are awaitable and return plain values unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value


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
