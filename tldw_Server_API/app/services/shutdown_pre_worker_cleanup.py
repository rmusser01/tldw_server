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
    """Pre-worker cleanup results after finalizers have run."""


async def shutdown_pre_worker_cleanup(
    *,
    app: Any,
    guard_exceptions: tuple[type[BaseException], ...],
) -> PreWorkerCleanupHandles:
    """Run the cleanup/reset shutdown slice that precedes the worker helpers."""
    await _shutdown_pre_worker_cleanup(
        app=app,
        guard_exceptions=guard_exceptions,
    )
    return PreWorkerCleanupHandles()


async def run_shutdown_pre_worker_cleanup(
    *,
    app: Any,
    guard_exceptions: tuple[type[BaseException], ...],
) -> PreWorkerCleanupHandles:
    """Run pre-worker cleanup with main-lifespan fallback behavior."""
    try:
        return await shutdown_pre_worker_cleanup(
            app=app,
            guard_exceptions=guard_exceptions,
        )
    except guard_exceptions as exc:
        logger.debug(f"Pre-worker cleanup skipped: {exc}")
        return PreWorkerCleanupHandles()


async def _shutdown_pre_worker_cleanup(
    *,
    app: Any,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    await _cancel_deferred_startup_task(
        app=app,
        guard_exceptions=guard_exceptions,
    )
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
