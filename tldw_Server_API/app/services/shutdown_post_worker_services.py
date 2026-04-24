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

    claims_task: Any | None = None
    jobs_prune_task: Any | None = None
    files_export_gc_task: Any | None = None
    notifications_prune_task: Any | None = None
    jobs_notifications_bridge_task: Any | None = None
    embeddings_compactor_task: Any | None = None
    embeddings_compactor_stop_event: Any | None = None
    websub_renewal_task: Any | None = None
    usage_task: Any | None = None
    llm_usage_task: Any | None = None
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
    claims_task: Any | None,
    jobs_prune_task: Any | None,
    files_export_gc_task: Any | None,
    notifications_prune_task: Any | None,
    jobs_notifications_bridge_task: Any | None,
    embeddings_compactor_task: Any | None,
    embeddings_compactor_stop_event: Any | None,
    websub_renewal_task: Any | None,
    coordinated_legacy_component_names: set[str],
    usage_task: Any | None,
    llm_usage_task: Any | None,
    workflows_sched_task: Any | None,
    reading_digest_sched_task: Any | None,
    admin_backup_sched_task: Any | None,
    companion_reflection_sched_task: Any | None,
    reminders_sched_task: Any | None,
    connectors_sync_sched_task: Any | None,
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
) -> PostWorkerShutdownHandles:
    """Run the post-worker shutdown tail in the legacy order."""
    claims_shutdown_handles = await _shutdown_claims_maintenance_tasks(
        claims_task=claims_task,
        jobs_prune_task=jobs_prune_task,
        files_export_gc_task=files_export_gc_task,
        notifications_prune_task=notifications_prune_task,
    )
    notifications_shutdown_handles = await _shutdown_notifications_compactor_websub_workers(
        jobs_notifications_bridge_task=jobs_notifications_bridge_task,
        embeddings_compactor_task=embeddings_compactor_task,
        embeddings_compactor_stop_event=embeddings_compactor_stop_event,
        websub_renewal_task=websub_renewal_task,
        guard_exceptions=guard_exceptions,
    )
    usage_shutdown_handles = await _stop_usage_aggregators(
        coordinated_legacy_component_names=coordinated_legacy_component_names,
        usage_task=usage_task,
        llm_usage_task=llm_usage_task,
        guard_exceptions=guard_exceptions,
    )
    await _stop_recurring_schedulers(
        workflows_sched_task=workflows_sched_task,
        reading_digest_sched_task=reading_digest_sched_task,
        admin_backup_sched_task=admin_backup_sched_task,
        companion_reflection_sched_task=companion_reflection_sched_task,
        reminders_sched_task=reminders_sched_task,
        connectors_sync_sched_task=connectors_sync_sched_task,
    )
    runtime_monitor_shutdown_handles = await _shutdown_runtime_monitors(
        jobs_metrics_task=jobs_metrics_task,
        jobs_metrics_stop_event=jobs_metrics_stop_event,
        loop_lag_task=loop_lag_task,
        loop_lag_stop_event=loop_lag_stop_event,
        guard_exceptions=guard_exceptions,
    )
    jobs_metrics_reconcile_shutdown_handles = await _shutdown_jobs_metrics_reconcile(
        jobs_metrics_reconcile_task=jobs_metrics_reconcile_task,
        jobs_metrics_reconcile_stop=jobs_metrics_reconcile_stop,
        guard_exceptions=guard_exceptions,
    )
    await _shutdown_personalization_consolidation(
        guard_exceptions=guard_exceptions,
    )
    optional_worker_shutdown_handles = await _shutdown_optional_workers(
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
        guard_exceptions=guard_exceptions,
    )
    return PostWorkerShutdownHandles(
        claims_task=claims_shutdown_handles.claims_task,
        jobs_prune_task=claims_shutdown_handles.jobs_prune_task,
        files_export_gc_task=claims_shutdown_handles.files_export_gc_task,
        notifications_prune_task=claims_shutdown_handles.notifications_prune_task,
        jobs_notifications_bridge_task=notifications_shutdown_handles.jobs_notifications_bridge_task,
        embeddings_compactor_task=notifications_shutdown_handles.embeddings_compactor_task,
        embeddings_compactor_stop_event=notifications_shutdown_handles.embeddings_compactor_stop_event,
        websub_renewal_task=notifications_shutdown_handles.websub_renewal_task,
        usage_task=usage_shutdown_handles.usage_task,
        llm_usage_task=usage_shutdown_handles.llm_usage_task,
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
    claims_task: Any | None,
    jobs_prune_task: Any | None,
    files_export_gc_task: Any | None,
    notifications_prune_task: Any | None,
    jobs_notifications_bridge_task: Any | None,
    embeddings_compactor_task: Any | None,
    embeddings_compactor_stop_event: Any | None,
    websub_renewal_task: Any | None,
    coordinated_legacy_component_names: set[str],
    usage_task: Any | None,
    llm_usage_task: Any | None,
    workflows_sched_task: Any | None,
    reading_digest_sched_task: Any | None,
    admin_backup_sched_task: Any | None,
    companion_reflection_sched_task: Any | None,
    reminders_sched_task: Any | None,
    connectors_sync_sched_task: Any | None,
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
) -> PostWorkerShutdownHandles:
    """Run post-worker shutdown with main-lifespan fallback behavior."""
    try:
        return await shutdown_post_worker_services(
            claims_task=claims_task,
            jobs_prune_task=jobs_prune_task,
            files_export_gc_task=files_export_gc_task,
            notifications_prune_task=notifications_prune_task,
            jobs_notifications_bridge_task=jobs_notifications_bridge_task,
            embeddings_compactor_task=embeddings_compactor_task,
            embeddings_compactor_stop_event=embeddings_compactor_stop_event,
            websub_renewal_task=websub_renewal_task,
            coordinated_legacy_component_names=coordinated_legacy_component_names,
            usage_task=usage_task,
            llm_usage_task=llm_usage_task,
            workflows_sched_task=workflows_sched_task,
            reading_digest_sched_task=reading_digest_sched_task,
            admin_backup_sched_task=admin_backup_sched_task,
            companion_reflection_sched_task=companion_reflection_sched_task,
            reminders_sched_task=reminders_sched_task,
            connectors_sync_sched_task=connectors_sync_sched_task,
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
            guard_exceptions=guard_exceptions,
        )
    except guard_exceptions as exc:
        logger.debug(f"Post-worker services skipped: {exc}")
        return PostWorkerShutdownHandles(
            claims_task=claims_task,
            jobs_prune_task=jobs_prune_task,
            files_export_gc_task=files_export_gc_task,
            notifications_prune_task=notifications_prune_task,
            jobs_notifications_bridge_task=jobs_notifications_bridge_task,
            embeddings_compactor_task=embeddings_compactor_task,
            embeddings_compactor_stop_event=embeddings_compactor_stop_event,
            websub_renewal_task=websub_renewal_task,
            usage_task=usage_task,
            llm_usage_task=llm_usage_task,
            jobs_metrics_task=jobs_metrics_task,
            loop_lag_task=loop_lag_task,
            jobs_metrics_reconcile_task=jobs_metrics_reconcile_task,
            jobs_metrics_reconcile_stop=jobs_metrics_reconcile_stop,
            jobs_crypto_rotate_task=jobs_crypto_rotate_task,
            jobs_integrity_task=jobs_integrity_task,
            jobs_webhooks_task=jobs_webhooks_task,
            meetings_webhook_dlq_task=meetings_webhook_dlq_task,
            workflows_dlq_task=workflows_dlq_task,
            workflows_gc_task=workflows_gc_task,
            workflows_maint_task=workflows_maint_task,
        )


async def _shutdown_claims_maintenance_tasks(**kwargs):
    from tldw_Server_API.app.services.shutdown_claims_maintenance_tasks import (
        shutdown_claims_maintenance_tasks,
    )

    return await shutdown_claims_maintenance_tasks(**kwargs)


async def _shutdown_notifications_compactor_websub_workers(**kwargs):
    from tldw_Server_API.app.services.shutdown_notifications_compactor_websub_workers import (
        shutdown_notifications_compactor_websub_workers,
    )

    return await shutdown_notifications_compactor_websub_workers(**kwargs)


async def _stop_usage_aggregators(**kwargs):
    from tldw_Server_API.app.services.shutdown_usage_aggregators import stop_usage_aggregators

    return await stop_usage_aggregators(**kwargs)


async def _stop_recurring_schedulers(**kwargs):
    from tldw_Server_API.app.services.shutdown_recurring_schedulers import (
        stop_recurring_schedulers,
    )

    await stop_recurring_schedulers(**kwargs)


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
