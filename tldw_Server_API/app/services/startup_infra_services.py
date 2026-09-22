"""
Infrastructure startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.core.testing import env_flag_enabled as _env_flag_enabled
from tldw_Server_API.app.services.lifecycle_worker_specs import (
    WorkerLifecycleContext,
    WorkerSpec,
    stop_event_worker_spec,
)
from tldw_Server_API.app.services.lifecycle_worker_startup_adapters import (
    run_started_task_until_stop,
)
from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass
class InfraStartupHandles:
    """Startup-owned infrastructure handles that should stay referenced in lifespan."""

    tts_history_cleanup_task: Any | None = None
    tts_history_cleanup_stop_event: Any | None = None


@dataclass
class ConnectorsStartupHandles:
    """Startup-owned connectors worker handles that should stay referenced in lifespan."""

    connectors_jobs_task: Any | None = None
    connectors_jobs_stop_event: Any | None = None


def provide_infra_worker_specs(
    _context: WorkerLifecycleContext | None = None,
) -> tuple[WorkerSpec, ...]:
    """Return declarative specs for infrastructure service workers."""

    return (
        stop_event_worker_spec(
            name="tts_history_cleanup_task",
            worker_service=_run_tts_history_cleanup_loop,
            category="maintenance",
            phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        ),
        WorkerSpec(
            name="connectors_jobs_task",
            task_name="connectors_jobs_task",
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=_connectors_worker_enabled,
            factory=lambda _context, stop_event: run_started_task_until_stop(
                stop_event,
                starter=lambda: _start_connectors_worker_service(
                    stop_event=stop_event
                ),
            ),
        ),
    )


def _connectors_worker_enabled(_context: WorkerLifecycleContext) -> bool:
    return _env_flag_enabled("CONNECTORS_WORKER_ENABLED")


async def start_infra_services(
    *,
    run_pg_rls_auto_ensure: Callable[[Any], Any],
    worker_inventory: Any | None = None,
) -> InfraStartupHandles:
    """Start the small infrastructure startup slice and return explicit handles."""
    await _maybe_ensure_pg_rls(run_pg_rls_auto_ensure)
    tts_history_cleanup_task, tts_history_cleanup_stop_event = await _start_tts_history_cleanup_worker(
        worker_inventory=worker_inventory,
    )
    return InfraStartupHandles(
        tts_history_cleanup_task=tts_history_cleanup_task,
        tts_history_cleanup_stop_event=tts_history_cleanup_stop_event,
    )


async def run_startup_infra_non_worker_setup(
    *,
    run_pg_rls_auto_ensure: Callable[[Any], Any],
) -> None:
    """Run infrastructure startup setup that is not lifecycle-worker owned."""

    await _maybe_ensure_pg_rls(run_pg_rls_auto_ensure)


async def start_connectors_startup(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller,
) -> ConnectorsStartupHandles:
    """Start the connectors worker slice and return explicit handles."""
    connectors_jobs_task, connectors_jobs_stop_event = await _start_connectors_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
    )
    return ConnectorsStartupHandles(
        connectors_jobs_task=connectors_jobs_task,
        connectors_jobs_stop_event=connectors_jobs_stop_event,
    )


def _postgres_content_mode_active() -> bool:
    """True when user content shares one PostgreSQL database across accounts."""
    try:
        from tldw_Server_API.app.core.DB_Management.media_db.runtime.defaults import (
            build_media_runtime_config,
        )

        return bool(build_media_runtime_config().postgres_content_mode)
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.debug("Could not determine content backend mode: {}", exc)
        return False


async def _maybe_ensure_pg_rls(run_pg_rls_auto_ensure: Callable[[Any], Any]) -> None:
    """Apply PostgreSQL RLS policies.

    In PostgreSQL content mode every account's rows share the same tables, so
    these policies are the only database-level thing keeping one account out of
    another's data. They are therefore applied unconditionally and a failure
    aborts startup: a server that cannot isolate accounts should not serve them.

    On SQLite, per-user database files provide the boundary and there is nothing
    to apply, so this stays opt-in via RAG_ENSURE_PG_RLS for operators who want
    the policies installed ahead of a migration.
    """
    required = _postgres_content_mode_active()

    if not required and not _env_flag_enabled("RAG_ENSURE_PG_RLS"):
        logger.info(
            "PG RLS auto-ensure skipped: content backend is not PostgreSQL "
            "(set RAG_ENSURE_PG_RLS=true to install policies anyway)"
        )
        return

    from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
    from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory

    try:
        config = DatabaseConfig.from_env()
        backend = DatabaseBackendFactory.create_backend(config)
        run_pg_rls_auto_ensure(backend)
    except Exception as exc:
        if required:
            raise RuntimeError(
                "Failed to apply PostgreSQL RLS policies, and the content backend "
                "is PostgreSQL, where every account shares the same tables. "
                "Refusing to start without tenant isolation policies in place. "
                f"Cause: {exc}"
            ) from exc
        logger.warning(f"Failed to apply PG RLS policies automatically: {exc}")


async def _start_tts_history_cleanup_worker(
    *,
    worker_inventory: Any | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the TTS history cleanup worker and return task/stop handles."""
    try:
        if worker_inventory is not None:
            task, stop_event = await worker_inventory.register_custom(
                name="tts_history_cleanup_task",
                task_name="tts_history_cleanup_task",
                coroutine_factory=_run_tts_history_cleanup_loop,
                timeout_sec=5.0,
                category="maintenance",
                shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            )
            logger.info("TTS history cleanup worker started")
            return task, stop_event

        stop_event = asyncio.Event()
        task = asyncio.create_task(
            _run_tts_history_cleanup_loop(stop_event),
            name="tts_history_cleanup_task",
        )
        logger.info("TTS history cleanup worker started")
        return task, stop_event
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start TTS history cleanup worker: {exc}")
        return None, None


async def _start_connectors_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller,
) -> tuple[Any | None, Any | None]:
    """Start the connectors worker and register it as a managed poller when active."""
    task = None
    try:
        stop_event = asyncio.Event()
        task = await _start_connectors_worker_service(stop_event=stop_event)
        if task:
            logger.info("Connectors worker started")
            try:
                register_owned_job_poller(
                    app,
                    owned_job_pollers,
                    name="connectors_jobs_task",
                    task=task,
                    stop_event=stop_event,
                )
            except _STARTUP_GUARD_EXCEPTIONS:
                _safe_cancel_task(task)
                raise
            return task, stop_event
        logger.info("Connectors worker disabled (CONNECTORS_WORKER_ENABLED != true)")
        return None, None
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Connectors worker: {exc}")
        return None, None


async def _run_tts_history_cleanup_loop(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.tts_history_cleanup_service import run_tts_history_cleanup_loop

    return await run_tts_history_cleanup_loop(stop_event)


def _safe_cancel_task(task: Any | None) -> None:
    if task is None:
        return
    try:
        task.cancel()
    except _STARTUP_GUARD_EXCEPTIONS:
        pass


async def _start_connectors_worker_service(*, stop_event: Any) -> Any:
    from tldw_Server_API.app.services.connectors_worker import start_connectors_worker

    return await start_connectors_worker(stop_event=stop_event)
