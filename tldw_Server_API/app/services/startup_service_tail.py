"""
Startup service-tail helper extracted from the application lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class StartupServiceTailHandles:
    """Flattened handles returned from the startup service-tail block."""

    jobs_metrics_stop_event: Any | None = None
    jobs_metrics_task: Any | None = None
    loop_lag_stop_event: Any | None = None
    loop_lag_task: Any | None = None
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
    claims_task: Any | None = None
    claims_alerts_task: Any | None = None
    claims_review_metrics_task: Any | None = None
    usage_task: Any | None = None
    llm_usage_task: Any | None = None
    tts_history_cleanup_task: Any | None = None
    tts_history_cleanup_stop_event: Any | None = None
    quality_eval_task: Any | None = None
    outputs_purge_task: Any | None = None
    kanban_activity_cleanup_task: Any | None = None
    ingestion_sources_cleanup_task: Any | None = None
    kanban_purge_task: Any | None = None
    files_export_gc_task: Any | None = None
    notifications_prune_task: Any | None = None
    jobs_prune_task: Any | None = None
    connectors_jobs_task: Any | None = None
    connectors_jobs_stop_event: Any | None = None
    authnz_scheduler_started: bool = False
    workflows_sched_task: Any | None = None
    reading_digest_sched_task: Any | None = None
    admin_backup_sched_task: Any | None = None
    companion_reflection_sched_task: Any | None = None
    reminders_sched_task: Any | None = None
    connectors_sync_sched_task: Any | None = None


async def initialize_startup_service_tail(
    *,
    app: Any,
    app_settings: Any,
    run_pg_rls_auto_ensure: Callable[[], Any],
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    startup_worker_group_handles: Any,
    replace_owned_job_poller_inventory: Callable[..., None],
    test_mode: bool,
    logger: Any,
    startup_api_key_log_value: Any,
    shared_is_truthy: Callable[..., bool],
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
    worker_inventory: Any | None = None,
) -> StartupServiceTailHandles:
    """Run the startup service tail in the legacy order."""
    startup_service_group_handles = await _start_service_groups(
        app=app,
        app_settings=app_settings,
        run_pg_rls_auto_ensure=run_pg_rls_auto_ensure,
        owned_job_pollers=owned_job_pollers,
        worker_inventory=worker_inventory,
        register_owned_job_poller=register_owned_job_poller,
    )
    recurring_scheduler_handles = await _finalize_startup_tail(
        app=app,
        owned_job_pollers=owned_job_pollers,
        startup_worker_group_handles=startup_worker_group_handles,
        startup_service_group_handles=startup_service_group_handles,
        replace_owned_job_poller_inventory=replace_owned_job_poller_inventory,
        test_mode=test_mode,
    )
    await _report_startup_environment(
        app=app,
        logger=logger,
        startup_api_key_log_value=startup_api_key_log_value,
        shared_is_truthy=shared_is_truthy,
        startup_guard_exceptions=startup_guard_exceptions,
        import_exceptions=import_exceptions,
    )
    return StartupServiceTailHandles(
        jobs_metrics_stop_event=startup_service_group_handles.jobs_metrics_stop_event,
        jobs_metrics_task=startup_service_group_handles.jobs_metrics_task,
        loop_lag_stop_event=startup_service_group_handles.loop_lag_stop_event,
        loop_lag_task=startup_service_group_handles.loop_lag_task,
        jobs_metrics_reconcile_stop=startup_service_group_handles.jobs_metrics_reconcile_stop,
        jobs_metrics_reconcile_task=startup_service_group_handles.jobs_metrics_reconcile_task,
        jobs_crypto_rotate_stop_event=startup_service_group_handles.jobs_crypto_rotate_stop_event,
        jobs_crypto_rotate_task=startup_service_group_handles.jobs_crypto_rotate_task,
        jobs_webhooks_stop_event=startup_service_group_handles.jobs_webhooks_stop_event,
        jobs_webhooks_task=startup_service_group_handles.jobs_webhooks_task,
        meetings_webhook_dlq_stop_event=startup_service_group_handles.meetings_webhook_dlq_stop_event,
        meetings_webhook_dlq_task=startup_service_group_handles.meetings_webhook_dlq_task,
        workflows_dlq_stop_event=startup_service_group_handles.workflows_dlq_stop_event,
        workflows_dlq_task=startup_service_group_handles.workflows_dlq_task,
        workflows_gc_stop_event=startup_service_group_handles.workflows_gc_stop_event,
        workflows_gc_task=startup_service_group_handles.workflows_gc_task,
        workflows_maint_stop_event=startup_service_group_handles.workflows_maint_stop_event,
        workflows_maint_task=startup_service_group_handles.workflows_maint_task,
        jobs_integrity_stop_event=startup_service_group_handles.jobs_integrity_stop_event,
        jobs_integrity_task=startup_service_group_handles.jobs_integrity_task,
        claims_task=startup_service_group_handles.claims_task,
        claims_alerts_task=startup_service_group_handles.claims_alerts_task,
        claims_review_metrics_task=startup_service_group_handles.claims_review_metrics_task,
        usage_task=startup_service_group_handles.usage_task,
        llm_usage_task=startup_service_group_handles.llm_usage_task,
        tts_history_cleanup_task=startup_service_group_handles.tts_history_cleanup_task,
        tts_history_cleanup_stop_event=startup_service_group_handles.tts_history_cleanup_stop_event,
        quality_eval_task=startup_service_group_handles.quality_eval_task,
        outputs_purge_task=startup_service_group_handles.outputs_purge_task,
        kanban_activity_cleanup_task=startup_service_group_handles.kanban_activity_cleanup_task,
        ingestion_sources_cleanup_task=startup_service_group_handles.ingestion_sources_cleanup_task,
        kanban_purge_task=startup_service_group_handles.kanban_purge_task,
        files_export_gc_task=startup_service_group_handles.files_export_gc_task,
        notifications_prune_task=startup_service_group_handles.notifications_prune_task,
        jobs_prune_task=startup_service_group_handles.jobs_prune_task,
        connectors_jobs_task=startup_service_group_handles.connectors_jobs_task,
        connectors_jobs_stop_event=startup_service_group_handles.connectors_jobs_stop_event,
        authnz_scheduler_started=recurring_scheduler_handles.authnz_scheduler_started,
        workflows_sched_task=recurring_scheduler_handles.workflows_sched_task,
        reading_digest_sched_task=recurring_scheduler_handles.reading_digest_sched_task,
        admin_backup_sched_task=recurring_scheduler_handles.admin_backup_sched_task,
        companion_reflection_sched_task=recurring_scheduler_handles.companion_reflection_sched_task,
        reminders_sched_task=recurring_scheduler_handles.reminders_sched_task,
        connectors_sync_sched_task=recurring_scheduler_handles.connectors_sync_sched_task,
    )


async def _start_service_groups(**kwargs):
    from tldw_Server_API.app.services.startup_service_groups import start_service_groups

    return await start_service_groups(**kwargs)


async def _finalize_startup_tail(**kwargs):
    from tldw_Server_API.app.services.startup_tail_finalization import (
        finalize_startup_tail,
    )

    return await finalize_startup_tail(**kwargs)


async def _report_startup_environment(**kwargs) -> None:
    from tldw_Server_API.app.services.startup_environment_reporting import (
        report_startup_environment,
    )

    await report_startup_environment(**kwargs)
