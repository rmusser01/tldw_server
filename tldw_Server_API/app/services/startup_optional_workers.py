"""
Optional worker startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.testing import env_flag_enabled as _env_flag_enabled

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

OptionalWorkerCoroutineFactory = Callable[[asyncio.Event], Awaitable[Any]]


@dataclass
class OptionalWorkerStartupHandles:
    """Startup-owned optional worker handles that remain part of shutdown flow."""

    jobs_metrics_reconcile_stop: Any | None = None
    jobs_metrics_reconcile_task: Any | None = None
    jobs_crypto_rotate_stop_event: Any | None = None
    jobs_crypto_rotate_task: Any | None = None
    jobs_webhooks_stop_event: Any | None = None
    jobs_webhooks_task: Any | None = None
    meetings_webhook_dlq_stop_event: Any | None = None
    meetings_webhook_dlq_task: Any | None = None
    workflows_dlq_stop_event: Any | None = None
    workflows_dlq_task: Any | None = None
    workflows_gc_stop_event: Any | None = None
    workflows_gc_task: Any | None = None
    workflows_maint_stop_event: Any | None = None
    workflows_maint_task: Any | None = None
    jobs_integrity_stop_event: Any | None = None
    jobs_integrity_task: Any | None = None
    persona_visual_generation_stop_event: Any | None = None
    persona_visual_generation_task: Any | None = None
    persona_visual_portability_stop_event: Any | None = None
    persona_visual_portability_task: Any | None = None


async def start_optional_workers(
    *,
    worker_inventory: Any | None = None,
) -> OptionalWorkerStartupHandles:
    """Start optional stop-event workers and return the task/stop handles."""
    jobs_metrics_reconcile_stop, jobs_metrics_reconcile_task = await _start_jobs_metrics_reconcile_worker(
        worker_inventory=worker_inventory,
    )
    jobs_crypto_rotate_stop_event, jobs_crypto_rotate_task = await _start_jobs_crypto_rotate_worker(
        worker_inventory=worker_inventory,
    )
    jobs_webhooks_stop_event, jobs_webhooks_task = await _start_jobs_webhooks_worker(
        worker_inventory=worker_inventory,
    )
    meetings_webhook_dlq_stop_event, meetings_webhook_dlq_task = await _start_meetings_webhook_dlq_worker(
        worker_inventory=worker_inventory,
    )
    workflows_dlq_stop_event, workflows_dlq_task = await _start_workflows_webhook_dlq_worker(
        worker_inventory=worker_inventory,
    )
    workflows_gc_stop_event, workflows_gc_task = await _start_workflows_artifact_gc_worker(
        worker_inventory=worker_inventory,
    )
    workflows_maint_stop_event, workflows_maint_task = await _start_workflows_db_maintenance_worker(
        worker_inventory=worker_inventory,
    )
    jobs_integrity_stop_event, jobs_integrity_task = await _start_jobs_integrity_sweeper(
        worker_inventory=worker_inventory,
    )
    persona_visual_generation_stop_event, persona_visual_generation_task = await _start_persona_visual_generation_worker(
        worker_inventory=worker_inventory,
    )
    persona_visual_portability_stop_event, persona_visual_portability_task = await _start_persona_visual_portability_worker(
        worker_inventory=worker_inventory,
    )
    return OptionalWorkerStartupHandles(
        jobs_metrics_reconcile_stop=jobs_metrics_reconcile_stop,
        jobs_metrics_reconcile_task=jobs_metrics_reconcile_task,
        jobs_crypto_rotate_stop_event=jobs_crypto_rotate_stop_event,
        jobs_crypto_rotate_task=jobs_crypto_rotate_task,
        jobs_webhooks_stop_event=jobs_webhooks_stop_event,
        jobs_webhooks_task=jobs_webhooks_task,
        meetings_webhook_dlq_stop_event=meetings_webhook_dlq_stop_event,
        meetings_webhook_dlq_task=meetings_webhook_dlq_task,
        workflows_dlq_stop_event=workflows_dlq_stop_event,
        workflows_dlq_task=workflows_dlq_task,
        workflows_gc_stop_event=workflows_gc_stop_event,
        workflows_gc_task=workflows_gc_task,
        workflows_maint_stop_event=workflows_maint_stop_event,
        workflows_maint_task=workflows_maint_task,
        jobs_integrity_stop_event=jobs_integrity_stop_event,
        jobs_integrity_task=jobs_integrity_task,
        persona_visual_generation_stop_event=persona_visual_generation_stop_event,
        persona_visual_generation_task=persona_visual_generation_task,
        persona_visual_portability_stop_event=persona_visual_portability_stop_event,
        persona_visual_portability_task=persona_visual_portability_task,
    )


def _make_event() -> Any:
    return asyncio.Event()


def _create_task(awaitable: Any) -> Any:
    return asyncio.create_task(awaitable)


async def _start_jobs_metrics_reconcile_worker(
    *,
    worker_inventory: Any | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the jobs metrics reconcile worker through inventory or legacy handles."""

    try:
        if not _env_flag_enabled("JOBS_METRICS_RECONCILE_ENABLE"):
            logger.info("Jobs metrics reconcile worker disabled by flag (JOBS_METRICS_RECONCILE_ENABLE)")
            return None, None
        if worker_inventory is not None:
            from tldw_Server_API.app.services.lifecycle_workers import (
                ShutdownPhase,
                start_stop_event_worker,
            )

            task, stop_event = await start_stop_event_worker(
                worker_inventory,
                name="jobs_metrics_reconcile_task",
                task_name="jobs_metrics_reconcile_task",
                coroutine_factory=_run_jobs_metrics_reconcile_service,
                category="jobs",
                shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            )
            logger.info("Jobs metrics reconcile worker started with explicit stop_event signal")
            return stop_event, task
        stop_event = _make_event()
        task = _create_task(_run_jobs_metrics_reconcile_service(stop_event))
        logger.info("Jobs metrics reconcile worker started with explicit stop_event signal")
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Jobs metrics reconcile worker: {exc}")
        return None, None


async def _start_jobs_crypto_rotate_worker(
    *,
    worker_inventory: Any | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the jobs crypto rotation worker through inventory or legacy handles."""

    try:
        if not _env_flag_enabled("JOBS_CRYPTO_ROTATE_SERVICE_ENABLED"):
            logger.info("Jobs crypto rotate worker disabled by flag")
            return None, None
        if worker_inventory is not None:
            from tldw_Server_API.app.services.lifecycle_workers import (
                ShutdownPhase,
                start_stop_event_worker,
            )

            task, stop_event = await start_stop_event_worker(
                worker_inventory,
                name="jobs_crypto_rotate_task",
                task_name="jobs_crypto_rotate_task",
                coroutine_factory=_run_jobs_crypto_rotate_service,
                category="jobs",
                shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            )
            logger.info("Jobs crypto rotate worker started with explicit stop_event signal")
            return stop_event, task
        stop_event = _make_event()
        task = _create_task(_run_jobs_crypto_rotate_service(stop_event))
        logger.info("Jobs crypto rotate worker started with explicit stop_event signal")
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Jobs crypto rotate worker: {exc}")
        return None, None


async def _start_jobs_webhooks_worker(
    *,
    worker_inventory: Any | None = None,
) -> tuple[Any | None, Any | None]:
    """Start Jobs webhook delivery when enabled and configured.

    Requires both JOBS_WEBHOOKS_ENABLED and JOBS_WEBHOOKS_URL. With a worker
    inventory, the worker is registered for background-worker lifecycle
    shutdown; without one, it uses the legacy stop-event task path.
    """
    try:
        if not _env_flag_enabled("JOBS_WEBHOOKS_ENABLED") or not os.getenv("JOBS_WEBHOOKS_URL"):
            logger.info("Jobs webhooks worker disabled by flag or missing URL")
            return None, None
        if worker_inventory is not None:
            from tldw_Server_API.app.services.lifecycle_workers import (
                ShutdownPhase,
                start_stop_event_worker,
            )

            task, stop_event = await start_stop_event_worker(
                worker_inventory,
                name="jobs_webhooks_task",
                task_name="jobs_webhooks_task",
                coroutine_factory=_run_jobs_webhooks_worker_service,
                category="jobs",
                shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            )
            logger.info("Jobs webhooks worker started with explicit stop_event signal")
            return stop_event, task
        stop_event = _make_event()
        task = _create_task(_run_jobs_webhooks_worker_service(stop_event))
        logger.info("Jobs webhooks worker started with explicit stop_event signal")
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Jobs webhooks worker: {exc}")
        return None, None


async def _start_meetings_webhook_dlq_worker(
    *,
    worker_inventory: Any | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the meetings webhook DLQ worker through inventory or legacy handles."""

    try:
        if not _env_flag_enabled("MEETINGS_WEBHOOK_DLQ_ENABLED"):
            logger.info("Meetings webhook DLQ worker disabled by flag")
            return None, None
        if worker_inventory is not None:
            stop_event, task = await _start_registered_optional_worker(
                worker_inventory=worker_inventory,
                name="meetings_webhook_dlq_task",
                coroutine_factory=_run_meetings_webhook_dlq_worker_service,
                category="meetings",
            )
            logger.info("Meetings webhook DLQ worker started with explicit stop_event signal")
            return stop_event, task
        stop_event = _make_event()
        task = _create_task(_run_meetings_webhook_dlq_worker_service(stop_event))
        logger.info("Meetings webhook DLQ worker started with explicit stop_event signal")
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Meetings webhook DLQ worker: {exc}")
        return None, None


async def _start_workflows_webhook_dlq_worker(
    *,
    worker_inventory: Any | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the workflows webhook DLQ worker through inventory or legacy handles."""

    try:
        if not _env_flag_enabled("WORKFLOWS_WEBHOOK_DLQ_ENABLED"):
            logger.info("Workflows webhook DLQ worker disabled by flag")
            return None, None
        if worker_inventory is not None:
            stop_event, task = await _start_registered_optional_worker(
                worker_inventory=worker_inventory,
                name="workflows_dlq_task",
                coroutine_factory=_run_workflows_webhook_dlq_worker_service,
                category="workflows",
            )
            logger.info("Workflows webhook DLQ worker started with explicit stop_event signal")
            return stop_event, task
        stop_event = _make_event()
        task = _create_task(_run_workflows_webhook_dlq_worker_service(stop_event))
        logger.info("Workflows webhook DLQ worker started with explicit stop_event signal")
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Workflows webhook DLQ worker: {exc}")
        return None, None


async def _start_workflows_artifact_gc_worker(
    *,
    worker_inventory: Any | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the workflows artifact GC worker through inventory or legacy handles."""

    try:
        if not _env_flag_enabled("WORKFLOWS_ARTIFACT_GC_ENABLED"):
            logger.info("Workflows artifact GC worker disabled by flag")
            return None, None
        if worker_inventory is not None:
            stop_event, task = await _start_registered_optional_worker(
                worker_inventory=worker_inventory,
                name="workflows_gc_task",
                coroutine_factory=_run_workflows_artifact_gc_worker_service,
                category="workflows",
            )
            logger.info("Workflows artifact GC worker started with explicit stop_event signal")
            return stop_event, task
        stop_event = _make_event()
        task = _create_task(_run_workflows_artifact_gc_worker_service(stop_event))
        logger.info("Workflows artifact GC worker started with explicit stop_event signal")
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Workflows artifact GC worker: {exc}")
        return None, None


async def _start_workflows_db_maintenance_worker(
    *,
    worker_inventory: Any | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the workflows DB maintenance worker through inventory or legacy handles."""

    try:
        if not _env_flag_enabled("WORKFLOWS_DB_MAINTENANCE_ENABLED"):
            logger.info("Workflows DB maintenance worker disabled by flag")
            return None, None
        if worker_inventory is not None:
            stop_event, task = await _start_registered_optional_worker(
                worker_inventory=worker_inventory,
                name="workflows_maint_task",
                coroutine_factory=_run_workflows_db_maintenance_worker_service,
                category="workflows",
            )
            logger.info("Workflows DB maintenance worker started with explicit stop_event signal")
            return stop_event, task
        stop_event = _make_event()
        task = _create_task(_run_workflows_db_maintenance_worker_service(stop_event))
        logger.info("Workflows DB maintenance worker started with explicit stop_event signal")
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Workflows DB maintenance worker: {exc}")
        return None, None


async def _start_jobs_integrity_sweeper(
    *,
    worker_inventory: Any | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the jobs integrity sweeper through inventory or legacy handles."""

    try:
        if not _env_flag_enabled("JOBS_INTEGRITY_SWEEP_ENABLED"):
            logger.info("Jobs integrity sweeper disabled by flag")
            return None, None
        if worker_inventory is not None:
            from tldw_Server_API.app.services.lifecycle_workers import (
                ShutdownPhase,
                start_stop_event_worker,
            )

            task, stop_event = await start_stop_event_worker(
                worker_inventory,
                name="jobs_integrity_task",
                task_name="jobs_integrity_task",
                coroutine_factory=_run_jobs_integrity_sweeper_service,
                category="jobs",
                shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            )
            logger.info("Jobs integrity sweeper started with explicit stop_event signal")
            return stop_event, task
        stop_event = _make_event()
        task = _create_task(_run_jobs_integrity_sweeper_service(stop_event))
        logger.info("Jobs integrity sweeper started with explicit stop_event signal")
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Jobs integrity sweeper: {exc}")
        return None, None


async def _start_persona_visual_generation_worker(
    *,
    worker_inventory: Any | None = None,
) -> tuple[Any | None, Any | None]:
    """Start persona visual generation worker only when explicitly enabled."""

    try:
        if not _env_flag_enabled("PERSONA_VISUAL_GENERATION_WORKER_ENABLED"):
            logger.info("Persona visual generation worker disabled by flag")
            return None, None
        if worker_inventory is not None:
            stop_event, task = await _start_registered_optional_worker(
                worker_inventory=worker_inventory,
                name="persona_visual_generation_task",
                coroutine_factory=_run_persona_visual_generation_worker_service,
                category="persona",
            )
            logger.info("Persona visual generation worker started with explicit stop_event signal")
            return stop_event, task
        stop_event = _make_event()
        task = _create_task(_run_persona_visual_generation_worker_service(stop_event))
        logger.info("Persona visual generation worker started with explicit stop_event signal")
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start persona visual generation worker: {exc}")
        return None, None


async def _start_persona_visual_portability_worker(
    *,
    worker_inventory: Any | None = None,
) -> tuple[Any | None, Any | None]:
    """Start persona visual portability worker only when explicitly enabled."""

    try:
        if not _env_flag_enabled("PERSONA_VISUAL_PORTABILITY_WORKER_ENABLED"):
            logger.info("Persona visual portability worker disabled by flag")
            return None, None
        if worker_inventory is not None:
            stop_event, task = await _start_registered_optional_worker(
                worker_inventory=worker_inventory,
                name="persona_visual_portability_task",
                coroutine_factory=_run_persona_visual_portability_worker_service,
                category="persona",
            )
            logger.info("Persona visual portability worker started with explicit stop_event signal")
            return stop_event, task
        stop_event = _make_event()
        task = _create_task(_run_persona_visual_portability_worker_service(stop_event))
        logger.info("Persona visual portability worker started with explicit stop_event signal")
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start persona visual portability worker: {exc}")
        return None, None


async def _start_registered_optional_worker(
    *,
    worker_inventory: Any,
    name: str,
    coroutine_factory: OptionalWorkerCoroutineFactory,
    category: str,
) -> tuple[Any, Any]:
    """Register an optional stop-event worker with the lifecycle inventory."""

    from tldw_Server_API.app.services.lifecycle_workers import (
        ShutdownPhase,
        start_stop_event_worker,
    )

    task, stop_event = await start_stop_event_worker(
        worker_inventory,
        name=name,
        task_name=name,
        coroutine_factory=coroutine_factory,
        category=category,
        shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
    )
    return stop_event, task


def _run_jobs_metrics_reconcile_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.jobs_metrics_service import (
        run_jobs_metrics_reconcile as _run_jobs_reconcile,
    )

    return _run_jobs_reconcile(stop_event)


def _run_jobs_crypto_rotate_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.jobs_crypto_rotate_service import (
        run_jobs_crypto_rotate as _run_jobs_crypto,
    )

    return _run_jobs_crypto(stop_event)


def _run_jobs_webhooks_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.jobs_webhooks_service import (
        run_jobs_webhooks_worker as _run_jobs_webhooks,
    )

    return _run_jobs_webhooks(stop_event)


def _run_meetings_webhook_dlq_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.meetings_webhook_dlq_service import (
        run_meetings_webhook_dlq_worker as _run_meetings_dlq,
    )

    return _run_meetings_dlq(stop_event)


def _run_workflows_webhook_dlq_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.workflows_webhook_dlq_service import (
        run_workflows_webhook_dlq_worker as _run_wf_dlq,
    )

    return _run_wf_dlq(stop_event)


def _run_workflows_artifact_gc_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.workflows_artifact_gc_service import (
        run_workflows_artifact_gc_worker as _run_wf_gc,
    )

    return _run_wf_gc(stop_event)


def _run_workflows_db_maintenance_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.workflows_db_maintenance import (
        run_workflows_db_maintenance as _run_wf_maint,
    )

    return _run_wf_maint(stop_event)


def _run_jobs_integrity_sweeper_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.jobs_integrity_service import (
        run_jobs_integrity_sweeper as _run_jobs_integrity,
    )

    return _run_jobs_integrity(stop_event)


def _run_persona_visual_generation_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.core.Persona.visual_jobs_worker import (
        run_persona_visual_generation_worker as _run_persona_visuals,
    )

    return _run_persona_visuals(stop_event)


def _run_persona_visual_portability_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.core.Persona.visual_jobs_worker import (
        run_persona_visual_portability_worker as _run_persona_visual_portability,
    )

    return _run_persona_visual_portability(stop_event)
