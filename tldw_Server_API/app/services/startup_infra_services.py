"""
Infrastructure startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.testing import env_flag_enabled as _env_flag_enabled

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


async def start_infra_services(
    *,
    run_pg_rls_auto_ensure,
) -> InfraStartupHandles:
    """Start the small infrastructure startup slice and return explicit handles."""
    await _maybe_ensure_pg_rls(run_pg_rls_auto_ensure)
    tts_history_cleanup_task, tts_history_cleanup_stop_event = await _start_tts_history_cleanup_worker()
    return InfraStartupHandles(
        tts_history_cleanup_task=tts_history_cleanup_task,
        tts_history_cleanup_stop_event=tts_history_cleanup_stop_event,
    )


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


async def _maybe_ensure_pg_rls(run_pg_rls_auto_ensure) -> None:
    """Apply optional PostgreSQL RLS policies when enabled by env."""
    try:
        if not _env_flag_enabled("RAG_ENSURE_PG_RLS"):
            logger.info("PG RLS auto-ensure disabled (set RAG_ENSURE_PG_RLS=true to enable)")
            return

        from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig, DatabaseError
        from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory

        config = DatabaseConfig.from_env()
        backend = DatabaseBackendFactory.create_backend(config)
        try:
            run_pg_rls_auto_ensure(backend)
        except DatabaseError as exc:
            logger.warning(f"Failed to apply PG RLS policies automatically: {exc}")
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to apply PG RLS policies automatically: {exc}")


async def _start_tts_history_cleanup_worker() -> tuple[Any | None, Any | None]:
    """Start the TTS history cleanup worker and return task/stop handles."""
    try:
        stop_event = asyncio.Event()
        task = asyncio.create_task(_run_tts_history_cleanup_loop(stop_event))
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
    try:
        stop_event = asyncio.Event()
        task = await _start_connectors_worker_service(stop_event=stop_event)
        if task:
            logger.info("Connectors worker started")
            register_owned_job_poller(
                app,
                owned_job_pollers,
                name="connectors_jobs_task",
                task=task,
                stop_event=stop_event,
            )
            return task, stop_event
        logger.info("Connectors worker disabled (CONNECTORS_WORKER_ENABLED != true)")
        return None, None
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Connectors worker: {exc}")
        return None, None


async def _run_tts_history_cleanup_loop(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.tts_history_cleanup_service import run_tts_history_cleanup_loop

    return await run_tts_history_cleanup_loop(stop_event)


async def _start_connectors_worker_service(*, stop_event: Any) -> Any:
    from tldw_Server_API.app.services.connectors_worker import start_connectors_worker

    return await start_connectors_worker(stop_event=stop_event)
