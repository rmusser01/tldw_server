"""
Pre-worker cleanup shutdown helper extracted from the application lifespan.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class PreWorkerCleanupHandles:
    """Updated cleanup-related handles after pre-worker shutdown processing."""

    cleanup_task: Any | None = None
    chatbooks_cleanup_task: Any | None = None
    chatbooks_cleanup_stop_event: Any | None = None
    storage_cleanup_service: Any | None = None


async def shutdown_pre_worker_cleanup(
    *,
    app: Any,
    cleanup_task: Any | None,
    chatbooks_cleanup_task: Any | None,
    chatbooks_cleanup_stop_event: Any | None,
    storage_cleanup_service: Any | None,
    coordinated_legacy_component_names: set[str],
    guard_exceptions: tuple[type[BaseException], ...],
    stopped_background_worker_names: set[str] | None = None,
) -> PreWorkerCleanupHandles:
    """Run the cleanup/reset shutdown slice that precedes the worker helpers."""
    await _shutdown_pre_worker_cleanup(
        app=app,
        cleanup_task=cleanup_task,
        chatbooks_cleanup_task=chatbooks_cleanup_task,
        chatbooks_cleanup_stop_event=chatbooks_cleanup_stop_event,
        storage_cleanup_service=storage_cleanup_service,
        coordinated_legacy_component_names=coordinated_legacy_component_names,
        guard_exceptions=guard_exceptions,
        stopped_background_worker_names=stopped_background_worker_names,
    )
    return PreWorkerCleanupHandles(
        cleanup_task=cleanup_task,
        chatbooks_cleanup_task=chatbooks_cleanup_task,
        chatbooks_cleanup_stop_event=chatbooks_cleanup_stop_event,
        storage_cleanup_service=storage_cleanup_service,
    )


async def run_shutdown_pre_worker_cleanup(
    *,
    app: Any,
    cleanup_task: Any | None,
    chatbooks_cleanup_task: Any | None,
    chatbooks_cleanup_stop_event: Any | None,
    storage_cleanup_service: Any | None,
    coordinated_legacy_component_names: set[str],
    guard_exceptions: tuple[type[BaseException], ...],
    stopped_background_worker_names: set[str] | None = None,
) -> PreWorkerCleanupHandles:
    """Run pre-worker cleanup with main-lifespan fallback behavior."""
    try:
        return await shutdown_pre_worker_cleanup(
            app=app,
            cleanup_task=cleanup_task,
            chatbooks_cleanup_task=chatbooks_cleanup_task,
            chatbooks_cleanup_stop_event=chatbooks_cleanup_stop_event,
            storage_cleanup_service=storage_cleanup_service,
            coordinated_legacy_component_names=coordinated_legacy_component_names,
            guard_exceptions=guard_exceptions,
            stopped_background_worker_names=stopped_background_worker_names,
        )
    except guard_exceptions as exc:
        logger.debug(f"Pre-worker cleanup skipped: {exc}")
        return PreWorkerCleanupHandles(
            cleanup_task=cleanup_task,
            chatbooks_cleanup_task=chatbooks_cleanup_task,
            chatbooks_cleanup_stop_event=chatbooks_cleanup_stop_event,
            storage_cleanup_service=storage_cleanup_service,
        )


async def _shutdown_pre_worker_cleanup(
    *,
    app: Any,
    cleanup_task: Any | None,
    chatbooks_cleanup_task: Any | None,
    chatbooks_cleanup_stop_event: Any | None,
    storage_cleanup_service: Any | None,
    coordinated_legacy_component_names: set[str],
    guard_exceptions: tuple[type[BaseException], ...],
    stopped_background_worker_names: set[str] | None = None,
) -> None:
    stopped_background_worker_names = stopped_background_worker_names or set()
    await _cancel_deferred_startup_task(
        app=app,
        guard_exceptions=guard_exceptions,
    )
    if cleanup_task:
        cleanup_task.cancel()
    if (
        "chatbooks_cleanup" not in coordinated_legacy_component_names
        and "chatbooks_cleanup" not in stopped_background_worker_names
    ):
        if chatbooks_cleanup_stop_event:
            chatbooks_cleanup_stop_event.set()
        if chatbooks_cleanup_task:
            chatbooks_cleanup_task.cancel()
    if storage_cleanup_service and "storage_cleanup_service" not in coordinated_legacy_component_names:
        try:
            await storage_cleanup_service.stop()
            logger.info("Storage cleanup worker stopped")
        except guard_exceptions:
            pass
    if "storage_cleanup_service" not in coordinated_legacy_component_names:
        await _reset_storage_service_singletons(
            guard_exceptions=guard_exceptions,
        )
    await _reset_authnz_rate_limiter_singleton(
        guard_exceptions=guard_exceptions,
    )


async def _cancel_deferred_startup_task(
    *,
    app: Any,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        state = getattr(app, "state", None)
        bg = getattr(state, "bg_tasks", None)
        if isinstance(bg, dict):
            task = bg.get("deferred_startup")
            if task:
                with suppress(guard_exceptions):
                    task.cancel()
    except guard_exceptions:
        pass


async def _reset_storage_service_singletons(
    *,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        await _reset_cleanup_service()
        await _reset_storage_service()
        logger.info("Storage service singletons reset")
    except guard_exceptions:
        pass


async def _reset_authnz_rate_limiter_singleton(
    *,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        await _reset_authnz_rate_limiter()
        logger.info("AuthNZ limiter singletons reset")
    except guard_exceptions:
        pass


async def _reset_cleanup_service() -> None:
    from tldw_Server_API.app.services.storage_cleanup_service import (
        reset_cleanup_service as _reset_cleanup_service_impl,
    )

    await _reset_cleanup_service_impl()


async def _reset_storage_service() -> None:
    from tldw_Server_API.app.services.storage_quota_service import (
        reset_storage_service as _reset_storage_service_impl,
    )

    await _reset_storage_service_impl()


async def _reset_authnz_rate_limiter() -> None:
    from tldw_Server_API.app.core.AuthNZ.rate_limiter import (
        reset_rate_limiter as _reset_rate_limiter_impl,
    )

    await _reset_rate_limiter_impl()
