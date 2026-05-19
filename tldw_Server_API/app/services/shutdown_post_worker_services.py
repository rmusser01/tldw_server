"""
Post-worker shutdown tail helper extracted from the application lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class PostWorkerShutdownHandles:
    """Updated handles produced by the post-worker shutdown tail."""

    jobs_notifications_bridge_task: Any | None = None
    jobs_metrics_task: Any | None = None
    loop_lag_task: Any | None = None
    jobs_metrics_reconcile_task: Any | None = None
    jobs_metrics_reconcile_stop: Any | None = None
    jobs_crypto_rotate_task: Any | None = None
    jobs_integrity_task: Any | None = None
    jobs_webhooks_task: Any | None = None
    meetings_webhook_dlq_task: Any | None = None
    workflows_dlq_task: Any | None = None
    workflows_gc_task: Any | None = None
    workflows_maint_task: Any | None = None


async def shutdown_post_worker_services(
    *,
    jobs_notifications_bridge_task: Any | None,
    jobs_metrics_task: Any | None,
    jobs_metrics_stop_event: Any | None,
    loop_lag_task: Any | None,
    loop_lag_stop_event: Any | None,
    jobs_metrics_reconcile_task: Any | None,
    jobs_metrics_reconcile_stop: Any | None,
    jobs_crypto_rotate_task: Any | None,
    jobs_crypto_rotate_stop_event: Any | None,
    jobs_integrity_task: Any | None,
    jobs_integrity_stop_event: Any | None,
    jobs_webhooks_task: Any | None,
    jobs_webhooks_stop_event: Any | None,
    meetings_webhook_dlq_task: Any | None,
    meetings_webhook_dlq_stop_event: Any | None,
    workflows_dlq_task: Any | None,
    workflows_dlq_stop_event: Any | None,
    workflows_gc_task: Any | None,
    workflows_gc_stop_event: Any | None,
    workflows_maint_task: Any | None,
    workflows_maint_stop_event: Any | None,
    guard_exceptions: tuple[type[BaseException], ...],
    stopped_background_worker_names: set[str] | None = None,
) -> PostWorkerShutdownHandles:
    """Run the post-worker shutdown tail in the legacy order."""
    stopped_background_worker_names = stopped_background_worker_names or set()

    def _task_if_not_stopped(name: str, task: Any | None) -> Any | None:
        if name in stopped_background_worker_names:
            return None
        return task

    def _stop_event_if_not_stopped(name: str, stop_event: Any | None) -> Any | None:
        if name in stopped_background_worker_names:
            return None
        return stop_event

    notifications_shutdown_handles = await _shutdown_notifications_compactor_websub_workers(
        jobs_notifications_bridge_task=_task_if_not_stopped(
            "jobs_notifications_bridge_task",
            jobs_notifications_bridge_task,
        ),
        guard_exceptions=guard_exceptions,
    )
    runtime_monitor_shutdown_handles = await _shutdown_runtime_monitors(
        jobs_metrics_task=_task_if_not_stopped("jobs_metrics_task", jobs_metrics_task),
        jobs_metrics_stop_event=_stop_event_if_not_stopped(
            "jobs_metrics_task",
            jobs_metrics_stop_event,
        ),
        loop_lag_task=_task_if_not_stopped("loop_lag_task", loop_lag_task),
        loop_lag_stop_event=_stop_event_if_not_stopped(
            "loop_lag_task",
            loop_lag_stop_event,
        ),
        guard_exceptions=guard_exceptions,
    )
    jobs_metrics_reconcile_shutdown_handles = await _shutdown_jobs_metrics_reconcile(
        jobs_metrics_reconcile_task=_task_if_not_stopped(
            "jobs_metrics_reconcile_task",
            jobs_metrics_reconcile_task,
        ),
        jobs_metrics_reconcile_stop=_stop_event_if_not_stopped(
            "jobs_metrics_reconcile_task",
            jobs_metrics_reconcile_stop,
        ),
        guard_exceptions=guard_exceptions,
    )
    await _shutdown_personalization_consolidation(
        guard_exceptions=guard_exceptions,
    )
    optional_worker_shutdown_handles = await _shutdown_optional_workers(
        jobs_crypto_rotate_task=_task_if_not_stopped(
            "jobs_crypto_rotate_task",
            jobs_crypto_rotate_task,
        ),
        jobs_crypto_rotate_stop_event=_stop_event_if_not_stopped(
            "jobs_crypto_rotate_task",
            jobs_crypto_rotate_stop_event,
        ),
        jobs_integrity_task=_task_if_not_stopped("jobs_integrity_task", jobs_integrity_task),
        jobs_integrity_stop_event=_stop_event_if_not_stopped(
            "jobs_integrity_task",
            jobs_integrity_stop_event,
        ),
        jobs_webhooks_task=_task_if_not_stopped("jobs_webhooks_task", jobs_webhooks_task),
        jobs_webhooks_stop_event=_stop_event_if_not_stopped(
            "jobs_webhooks_task",
            jobs_webhooks_stop_event,
        ),
        meetings_webhook_dlq_task=_task_if_not_stopped(
            "meetings_webhook_dlq_task",
            meetings_webhook_dlq_task,
        ),
        meetings_webhook_dlq_stop_event=_stop_event_if_not_stopped(
            "meetings_webhook_dlq_task",
            meetings_webhook_dlq_stop_event,
        ),
        workflows_dlq_task=_task_if_not_stopped("workflows_dlq_task", workflows_dlq_task),
        workflows_dlq_stop_event=_stop_event_if_not_stopped(
            "workflows_dlq_task",
            workflows_dlq_stop_event,
        ),
        workflows_gc_task=_task_if_not_stopped("workflows_gc_task", workflows_gc_task),
        workflows_gc_stop_event=_stop_event_if_not_stopped(
            "workflows_gc_task",
            workflows_gc_stop_event,
        ),
        workflows_maint_task=_task_if_not_stopped("workflows_maint_task", workflows_maint_task),
        workflows_maint_stop_event=_stop_event_if_not_stopped(
            "workflows_maint_task",
            workflows_maint_stop_event,
        ),
        guard_exceptions=guard_exceptions,
    )
    return PostWorkerShutdownHandles(
        jobs_notifications_bridge_task=notifications_shutdown_handles.jobs_notifications_bridge_task,
        jobs_metrics_task=runtime_monitor_shutdown_handles.jobs_metrics_task,
        loop_lag_task=runtime_monitor_shutdown_handles.loop_lag_task,
        jobs_metrics_reconcile_task=jobs_metrics_reconcile_shutdown_handles.jobs_metrics_reconcile_task,
        jobs_metrics_reconcile_stop=jobs_metrics_reconcile_shutdown_handles.jobs_metrics_reconcile_stop,
        jobs_crypto_rotate_task=optional_worker_shutdown_handles.jobs_crypto_rotate_task,
        jobs_integrity_task=optional_worker_shutdown_handles.jobs_integrity_task,
        jobs_webhooks_task=optional_worker_shutdown_handles.jobs_webhooks_task,
        meetings_webhook_dlq_task=optional_worker_shutdown_handles.meetings_webhook_dlq_task,
        workflows_dlq_task=optional_worker_shutdown_handles.workflows_dlq_task,
        workflows_gc_task=optional_worker_shutdown_handles.workflows_gc_task,
        workflows_maint_task=optional_worker_shutdown_handles.workflows_maint_task,
    )


async def run_shutdown_post_worker_services(
    *,
    jobs_notifications_bridge_task: Any | None,
    jobs_metrics_task: Any | None,
    jobs_metrics_stop_event: Any | None,
    loop_lag_task: Any | None,
    loop_lag_stop_event: Any | None,
    jobs_metrics_reconcile_task: Any | None,
    jobs_metrics_reconcile_stop: Any | None,
    jobs_crypto_rotate_task: Any | None,
    jobs_crypto_rotate_stop_event: Any | None,
    jobs_integrity_task: Any | None,
    jobs_integrity_stop_event: Any | None,
    jobs_webhooks_task: Any | None,
    jobs_webhooks_stop_event: Any | None,
    meetings_webhook_dlq_task: Any | None,
    meetings_webhook_dlq_stop_event: Any | None,
    workflows_dlq_task: Any | None,
    workflows_dlq_stop_event: Any | None,
    workflows_gc_task: Any | None,
    workflows_gc_stop_event: Any | None,
    workflows_maint_task: Any | None,
    workflows_maint_stop_event: Any | None,
    guard_exceptions: tuple[type[BaseException], ...],
    stopped_background_worker_names: set[str] | None = None,
) -> PostWorkerShutdownHandles:
    """Run post-worker shutdown with main-lifespan fallback behavior."""
    stopped_background_worker_names = stopped_background_worker_names or set()

    def _fallback_if_not_stopped(name: str, value: Any | None) -> Any | None:
        if name in stopped_background_worker_names:
            return None
        return value

    try:
        return await shutdown_post_worker_services(
            jobs_notifications_bridge_task=jobs_notifications_bridge_task,
            jobs_metrics_task=jobs_metrics_task,
            jobs_metrics_stop_event=jobs_metrics_stop_event,
            loop_lag_task=loop_lag_task,
            loop_lag_stop_event=loop_lag_stop_event,
            jobs_metrics_reconcile_task=jobs_metrics_reconcile_task,
            jobs_metrics_reconcile_stop=jobs_metrics_reconcile_stop,
            jobs_crypto_rotate_task=jobs_crypto_rotate_task,
            jobs_crypto_rotate_stop_event=jobs_crypto_rotate_stop_event,
            jobs_integrity_task=jobs_integrity_task,
            jobs_integrity_stop_event=jobs_integrity_stop_event,
            jobs_webhooks_task=jobs_webhooks_task,
            jobs_webhooks_stop_event=jobs_webhooks_stop_event,
            meetings_webhook_dlq_task=meetings_webhook_dlq_task,
            meetings_webhook_dlq_stop_event=meetings_webhook_dlq_stop_event,
            workflows_dlq_task=workflows_dlq_task,
            workflows_dlq_stop_event=workflows_dlq_stop_event,
            workflows_gc_task=workflows_gc_task,
            workflows_gc_stop_event=workflows_gc_stop_event,
            workflows_maint_task=workflows_maint_task,
            workflows_maint_stop_event=workflows_maint_stop_event,
            stopped_background_worker_names=stopped_background_worker_names,
            guard_exceptions=guard_exceptions,
        )
    except guard_exceptions as exc:
        logger.debug(f"Post-worker services skipped: {exc}")
        return PostWorkerShutdownHandles(
            jobs_notifications_bridge_task=_fallback_if_not_stopped(
                "jobs_notifications_bridge_task",
                jobs_notifications_bridge_task,
            ),
            jobs_metrics_task=_fallback_if_not_stopped("jobs_metrics_task", jobs_metrics_task),
            loop_lag_task=_fallback_if_not_stopped("loop_lag_task", loop_lag_task),
            jobs_metrics_reconcile_task=_fallback_if_not_stopped(
                "jobs_metrics_reconcile_task",
                jobs_metrics_reconcile_task,
            ),
            jobs_metrics_reconcile_stop=_fallback_if_not_stopped(
                "jobs_metrics_reconcile_task",
                jobs_metrics_reconcile_stop,
            ),
            jobs_crypto_rotate_task=_fallback_if_not_stopped(
                "jobs_crypto_rotate_task",
                jobs_crypto_rotate_task,
            ),
            jobs_integrity_task=_fallback_if_not_stopped(
                "jobs_integrity_task",
                jobs_integrity_task,
            ),
            jobs_webhooks_task=_fallback_if_not_stopped("jobs_webhooks_task", jobs_webhooks_task),
            meetings_webhook_dlq_task=_fallback_if_not_stopped(
                "meetings_webhook_dlq_task",
                meetings_webhook_dlq_task,
            ),
            workflows_dlq_task=_fallback_if_not_stopped("workflows_dlq_task", workflows_dlq_task),
            workflows_gc_task=_fallback_if_not_stopped("workflows_gc_task", workflows_gc_task),
            workflows_maint_task=_fallback_if_not_stopped(
                "workflows_maint_task",
                workflows_maint_task,
            ),
        )


async def _shutdown_notifications_compactor_websub_workers(**kwargs):
    from tldw_Server_API.app.services.shutdown_notifications_compactor_websub_workers import (
        shutdown_notifications_compactor_websub_workers,
    )

    return await shutdown_notifications_compactor_websub_workers(**kwargs)


async def _shutdown_runtime_monitors(**kwargs):
    from tldw_Server_API.app.services.shutdown_runtime_monitors import shutdown_runtime_monitors

    return await shutdown_runtime_monitors(**kwargs)


async def _shutdown_jobs_metrics_reconcile(**kwargs):
    from tldw_Server_API.app.services.shutdown_jobs_metrics_reconcile import (
        shutdown_jobs_metrics_reconcile,
    )

    return await shutdown_jobs_metrics_reconcile(**kwargs)


async def _shutdown_personalization_consolidation(**kwargs):
    from tldw_Server_API.app.services.shutdown_personalization_consolidation import (
        shutdown_personalization_consolidation,
    )

    await shutdown_personalization_consolidation(**kwargs)


async def _shutdown_optional_workers(**kwargs):
    from tldw_Server_API.app.services.shutdown_optional_workers import shutdown_optional_workers

    return await shutdown_optional_workers(**kwargs)
