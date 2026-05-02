"""
Embeddings compactor and WebSub startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

_TRUTHY_ENV_VALUES = {"true", "1", "yes", "y", "on"}


@dataclass
class CompactorWebsubStartupHandles:
    """Startup handles for the embeddings compactor and WebSub worker."""

    embeddings_compactor_stop_event: Any | None = None
    embeddings_compactor_task: Any | None = None
    websub_renewal_task: Any | None = None


async def start_compactor_websub_workers(
    *,
    should_start_worker: Callable[..., bool],
    worker_inventory: Any | None = None,
) -> CompactorWebsubStartupHandles:
    """Start the embeddings compactor and WebSub renewal worker."""

    embeddings_compactor_stop_event, embeddings_compactor_task = await _start_embeddings_vector_compactor(
        worker_inventory=worker_inventory,
    )
    websub_renewal_task = await _start_websub_renewal_worker(
        should_start_worker=should_start_worker,
        worker_inventory=worker_inventory,
    )
    return CompactorWebsubStartupHandles(
        embeddings_compactor_stop_event=embeddings_compactor_stop_event,
        embeddings_compactor_task=embeddings_compactor_task,
        websub_renewal_task=websub_renewal_task,
    )


def _make_event() -> Any:
    return asyncio.Event()


def _create_task(awaitable: Any, *, name: str | None = None) -> Any:
    """Create a task with an optional asyncio task name for diagnostics."""
    return asyncio.create_task(awaitable, name=name)


async def _start_embeddings_vector_compactor(
    *,
    worker_inventory: Any | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the embeddings vector compactor when enabled.

    When a worker inventory is available, register the compactor as a
    background-shutdown worker and return ``(stop_event, task)`` for legacy
    state compatibility. Without an inventory, create the legacy stop
    event/task pair directly.
    """
    try:
        enabled = os.getenv("EMBEDDINGS_COMPACTOR_ENABLED", "false").lower() in _TRUTHY_ENV_VALUES
        if not enabled:
            logger.info("Embeddings Vector Compactor disabled by flag (EMBEDDINGS_COMPACTOR_ENABLED)")
            return None, None

        if worker_inventory is not None:
            from tldw_Server_API.app.services.lifecycle_workers import (
                ShutdownPhase,
                start_stop_event_worker,
            )

            task, stop_event = await start_stop_event_worker(
                worker_inventory,
                name="embeddings_compactor_task",
                task_name="embeddings_compactor_task",
                coroutine_factory=_run_embeddings_vector_compactor_service,
                category="embeddings",
                shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            )
            logger.info("Embeddings Vector Compactor started with explicit stop_event signal")
            return stop_event, task

        stop_event = _make_event()
        task = _create_task(_run_embeddings_vector_compactor_service(stop_event))
        logger.info("Embeddings Vector Compactor started with explicit stop_event signal")
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Embeddings Vector Compactor: {exc}")
        return None, None


async def _start_websub_renewal_worker(
    *,
    should_start_worker: Callable[..., bool],
    worker_inventory: Any | None = None,
) -> Any | None:
    try:
        callback_base_url = os.getenv("WEBSUB_CALLBACK_BASE_URL", "").strip()
        enabled = bool(callback_base_url) and should_start_worker(
            "WEBSUB_RENEWAL_WORKER_ENABLED",
            "collections-websub",
        )
        if not enabled:
            logger.info("WebSub renewal worker disabled (no WEBSUB_CALLBACK_BASE_URL or flag off)")
            return None

        task = _create_task(_run_websub_renewal_loop(), name="websub_renewal_task")
        if worker_inventory is not None:
            from tldw_Server_API.app.services.lifecycle_workers import (
                ManagedWorker,
                ShutdownPhase,
            )

            try:
                worker_inventory.register(
                    ManagedWorker(
                        name="websub_renewal_task",
                        task=task,
                        stop_event=None,
                        category="collections-websub",
                        shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
                    )
                )
            except Exception:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    logger.debug(f"WebSub renewal task raised during startup rollback: {exc}")
                raise
        logger.info("WebSub lease renewal worker started")
        return task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start WebSub renewal worker: {exc}")
        return None


def _run_embeddings_vector_compactor_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.core.Embeddings.services.vector_compactor import (
        run as _run_embeddings_vector_compactor,
    )

    return _run_embeddings_vector_compactor(stop_event)


def _run_websub_renewal_loop() -> Any:
    from tldw_Server_API.app.core.Watchlists.websub import websub_renewal_loop

    return websub_renewal_loop()
