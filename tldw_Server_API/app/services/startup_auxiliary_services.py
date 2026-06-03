"""
Auxiliary startup-service helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.config import legacy_get as _legacy_get
from tldw_Server_API.app.core.testing import env_flag_enabled as _env_flag_enabled
from tldw_Server_API.app.services.lifecycle_worker_specs import (
    WorkerLifecycleContext,
    WorkerSpec,
)
from tldw_Server_API.app.services.lifecycle_worker_startup_adapters import (
    run_started_task_until_stop,
)
from tldw_Server_API.app.services.lifecycle_workers import ManagedWorker, ShutdownPhase

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass
class AuxiliaryStartupHandles:
    """Startup-owned auxiliary service handles that should stay referenced in lifespan."""

    claims_alerts_task: Any | None = None
    claims_review_metrics_task: Any | None = None
    usage_task: Any | None = None
    llm_usage_task: Any | None = None


def provide_auxiliary_worker_specs(
    _context: WorkerLifecycleContext | None = None,
) -> tuple[WorkerSpec, ...]:
    """Return declarative specs for auxiliary scheduler workers."""

    return (
        _auxiliary_scheduler_spec(
            name="claims_alerts_task",
            task_name="claims_alerts_scheduler",
            enabled=_claims_alerts_scheduler_enabled,
            starter=_start_claims_alerts_scheduler_service,
        ),
        _auxiliary_scheduler_spec(
            name="claims_review_metrics_task",
            task_name="claims_review_metrics_scheduler",
            enabled=_claims_review_metrics_scheduler_enabled,
            starter=_start_claims_review_metrics_scheduler_service,
        ),
    )


def _auxiliary_scheduler_spec(
    *,
    name: str,
    task_name: str,
    enabled,
    starter,
) -> WorkerSpec:
    return WorkerSpec(
        name=name,
        task_name=task_name,
        category="auxiliary",
        phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        enabled=enabled,
        factory=lambda _context, stop_event: run_started_task_until_stop(
            stop_event,
            starter=starter,
        ),
    )


def _claims_alerts_scheduler_enabled(context: WorkerLifecycleContext) -> bool:
    return _env_flag_enabled("CLAIMS_ALERTS_SCHEDULER_ENABLED") or bool(
        _legacy_get(
            "CLAIMS_ALERTS_SCHEDULER_ENABLED",
            context.settings.get("CLAIMS_ALERTS_SCHEDULER_ENABLED", False),
        )
    )


def _claims_review_metrics_scheduler_enabled(
    context: WorkerLifecycleContext,
) -> bool:
    return _env_flag_enabled("CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED") or bool(
        _legacy_get(
            "CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED",
            context.settings.get("CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED", False),
        )
    )


async def start_auxiliary_services(
    app_settings: Mapping[str, Any],
    *,
    worker_inventory: Any | None = None,
) -> AuxiliaryStartupHandles:
    """Start auxiliary services and return explicit task handles.

    When ``worker_inventory`` is provided, claims schedulers and usage
    aggregators are registered with lifecycle worker management. Passing
    ``None`` preserves the legacy direct-task path in downstream startup
    helpers.
    """
    claims_alerts_task = await _start_claims_alerts_scheduler(
        worker_inventory=worker_inventory,
    )
    claims_review_metrics_task = await _start_claims_review_metrics_scheduler(
        worker_inventory=worker_inventory,
    )
    usage_task = await _start_usage_aggregator(worker_inventory=worker_inventory)
    llm_usage_task = await _start_llm_usage_aggregator(worker_inventory=worker_inventory)
    await _start_personalization_consolidation(app_settings)
    return AuxiliaryStartupHandles(
        claims_alerts_task=claims_alerts_task,
        claims_review_metrics_task=claims_review_metrics_task,
        usage_task=usage_task,
        llm_usage_task=llm_usage_task,
    )


async def _start_claims_alerts_scheduler(
    *,
    worker_inventory: Any | None = None,
) -> Any | None:
    try:
        task = await _start_claims_alerts_scheduler_service()
        if task:
            await _register_auxiliary_task(
                worker_inventory=worker_inventory,
                task=task,
                worker_name="claims_alerts_task",
            )
            logger.info("Claims alerts scheduler started")
        return task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start claims alerts scheduler: {exc}")
        return None


async def _start_claims_review_metrics_scheduler(
    *,
    worker_inventory: Any | None = None,
) -> Any | None:
    try:
        task = await _start_claims_review_metrics_scheduler_service()
        if task:
            await _register_auxiliary_task(
                worker_inventory=worker_inventory,
                task=task,
                worker_name="claims_review_metrics_task",
            )
            logger.info("Claims review metrics scheduler started")
        return task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start claims review metrics scheduler: {exc}")
        return None


async def _start_usage_aggregator(
    *,
    worker_inventory: Any | None = None,
) -> Any | None:
    """Start usage aggregation through lifecycle inventory or legacy task mode."""
    try:
        if _env_flag_enabled("DISABLE_USAGE_AGGREGATOR"):
            logger.info("Usage aggregator disabled via DISABLE_USAGE_AGGREGATOR")
            return None
        task = await _start_usage_aggregator_service(worker_inventory=worker_inventory)
        if task:
            logger.info("Usage aggregator started")
        return task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start usage aggregator: {exc}")
        return None


async def _start_llm_usage_aggregator(
    *,
    worker_inventory: Any | None = None,
) -> Any | None:
    """Start LLM usage aggregation through lifecycle inventory or legacy task mode."""
    try:
        if _env_flag_enabled("DISABLE_LLM_USAGE_AGGREGATOR"):
            logger.info("LLM usage aggregator disabled via DISABLE_LLM_USAGE_AGGREGATOR")
            return None
        task = await _start_llm_usage_aggregator_service(worker_inventory=worker_inventory)
        if task:
            logger.info("LLM usage aggregator started")
        return task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start LLM usage aggregator: {exc}")
        return None


async def _register_auxiliary_task(
    *,
    worker_inventory: Any | None,
    task: Any,
    worker_name: str,
) -> None:
    """Register a scheduler task for lifecycle shutdown management.

    If inventory registration fails after the task has been created, the
    task is cancelled as rollback cleanup. Rollback errors are logged without
    replacing the original registration failure.
    """
    if worker_inventory is None:
        return
    inventory_handles = getattr(worker_inventory, "handles", None)
    initial_handle_count = len(inventory_handles) if isinstance(inventory_handles, list) else None
    try:
        worker_inventory.register(
            ManagedWorker(
                name=worker_name,
                task=task,
                stop_event=None,
                category="auxiliary",
                shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            )
        )
    except Exception:  # noqa: BLE001 - rollback must preserve the original registration failure.
        if isinstance(inventory_handles, list) and initial_handle_count is not None:
            del inventory_handles[initial_handle_count:]
            publish = getattr(worker_inventory, "publish", None)
            if callable(publish):
                try:
                    publish()
                except Exception as exc:  # noqa: BLE001 - best-effort rollback state republish.
                    logger.debug(f"Auxiliary scheduler inventory rollback publish failed: {exc}")
        try:
            await _cancel_unregistered_task(task)
        except Exception as exc:  # noqa: BLE001 - cleanup must not shadow registration failure.
            logger.debug(f"Auxiliary scheduler startup rollback failed: {exc}")
        raise


async def _cancel_unregistered_task(task: Any, *, timeout: float = 1.0) -> None:
    """Cancel a scheduler task that could not be registered with inventory.

    The wait is bounded so startup rollback cannot hang indefinitely when a
    task ignores cancellation.
    """
    try:
        task.cancel()
    except Exception as exc:  # noqa: BLE001 - rollback cleanup is best effort.
        logger.debug(f"Auxiliary scheduler startup rollback cancel failed: {exc}")
        return
    try:
        await asyncio.wait_for(task, timeout=timeout)
    except asyncio.CancelledError:
        pass
    except asyncio.TimeoutError:
        logger.warning(
            "Auxiliary scheduler did not cancel within {}s during startup rollback",
            timeout,
        )
    except Exception as exc:  # noqa: BLE001 - task exceptions during rollback are logged only.
        logger.debug(f"Auxiliary scheduler raised during startup rollback: {exc}")


async def _start_personalization_consolidation(app_settings: Mapping[str, Any]) -> None:
    try:
        personalization_enabled = bool(
            _legacy_get("PERSONALIZATION_ENABLED", app_settings.get("PERSONALIZATION_ENABLED", True))
        )
        skip_consolidation = _env_flag_enabled("DISABLE_PERSONALIZATION_CONSOLIDATION")
        if not personalization_enabled or skip_consolidation:
            logger.info("Personalization consolidation disabled (flag or env)")
            return
        consolidation_service = _get_consolidation_service()
        await consolidation_service.start()
        logger.info("Personalization consolidation service started")
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start personalization consolidation: {exc}")


async def _start_claims_alerts_scheduler_service() -> Any | None:
    from tldw_Server_API.app.services.claims_alerts_scheduler import start_claims_alerts_scheduler

    return await start_claims_alerts_scheduler()


async def _start_claims_review_metrics_scheduler_service() -> Any | None:
    from tldw_Server_API.app.services.claims_review_metrics_scheduler import (
        start_claims_review_metrics_scheduler,
    )

    return await start_claims_review_metrics_scheduler()


async def _start_usage_aggregator_service(**kwargs: Any) -> Any | None:
    from tldw_Server_API.app.services.usage_aggregator import start_usage_aggregator

    return await start_usage_aggregator(**kwargs)


async def _start_llm_usage_aggregator_service(**kwargs: Any) -> Any | None:
    from tldw_Server_API.app.services.llm_usage_aggregator import start_llm_usage_aggregator

    return await start_llm_usage_aggregator(**kwargs)


def _get_consolidation_service() -> Any:
    from tldw_Server_API.app.services.personalization_consolidation import get_consolidation_service

    return get_consolidation_service()
