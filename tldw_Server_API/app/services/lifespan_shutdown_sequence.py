"""
Shutdown sequence orchestration extracted from the application lifespan.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI

from tldw_Server_API.app.services.lifecycle_workers import (
    ShutdownPhase,
    stop_registered_workers,
)
from tldw_Server_API.app.services.lifespan_worker_runtime_state import (
    LifespanWorkerRuntimeState,
)


async def run_lifespan_shutdown_sequence(
    *,
    app: FastAPI,
    worker_runtime: LifespanWorkerRuntimeState,
    readiness_state: dict[str, Any],
    db_pool: Any,
    session_manager: Any,
    heavy_startup_handles: Any,
    build_legacy_shutdown_context: Any,
    apply_shutdown_transition_gate: Any,
    quiesce_owned_job_pollers_for_shutdown: Any,
    run_coordinated_shutdown: Any,
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
    in_pytest_runtime: bool,
    test_db_instance_ref: Any,
    timed_shutdown_segment: Any,
    record_shutdown_timing_total: Any,
    monotonic: Any = time.monotonic,
) -> None:
    """Run the post-yield lifespan shutdown sequence in the existing order."""
    shutdown_started = monotonic()
    try:
        app.state._tldw_shutdown_timing_segments = []
    except startup_guard_exceptions:
        pass

    legacy_shutdown_plan: list[Any] = []
    with timed_shutdown_segment(app, "transition_handoff"):
        from tldw_Server_API.app.services.shutdown_transition_handoff import (
            shutdown_transition_handoff,
        )

        transition_handoff_handles = await shutdown_transition_handoff(
            app=app,
            readiness_state=readiness_state,
            build_legacy_shutdown_context=build_legacy_shutdown_context,
            apply_shutdown_transition_gate=apply_shutdown_transition_gate,
            startup_guard_exceptions=startup_guard_exceptions,
            import_exceptions=import_exceptions,
        )
        legacy_shutdown_plan = transition_handoff_handles.legacy_shutdown_plan

    from tldw_Server_API.app.services.shutdown_job_poller_handoff import (
        run_shutdown_job_poller_handoff,
    )

    job_poller_handoff_handles = await run_shutdown_job_poller_handoff(
        app=app,
        owned_job_pollers=worker_runtime.owned_job_pollers,
        quiesce_owned_job_pollers_for_shutdown=quiesce_owned_job_pollers_for_shutdown,
        startup_guard_exceptions=startup_guard_exceptions,
        import_exceptions=import_exceptions,
    )
    should_run_late_stop = job_poller_handoff_handles.should_run_late_stop

    with timed_shutdown_segment(app, "background_worker_shutdown"):
        await stop_registered_workers(
            app,
            _handles_for_shutdown_phase(
                worker_runtime,
                ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            ),
            stopped_names_attr="_tldw_shutdown_stopped_background_worker_names",
            log_label="background worker",
        )
        stopped_background_worker_names = set(getattr(app.state, "_tldw_shutdown_stopped_background_worker_names", []))

    from tldw_Server_API.app.services.shutdown_coordinated_legacy_components import (
        run_shutdown_coordinated_legacy_components,
    )

    await run_shutdown_coordinated_legacy_components(
        app=app,
        legacy_shutdown_plan=legacy_shutdown_plan,
        run_coordinated_shutdown=run_coordinated_shutdown,
        startup_guard_exceptions=startup_guard_exceptions,
        import_exceptions=import_exceptions,
        stopped_background_worker_names=stopped_background_worker_names,
    )

    from tldw_Server_API.app.services.shutdown_pre_worker_cleanup import (
        run_shutdown_pre_worker_cleanup,
    )

    pre_worker_cleanup_handles = await run_shutdown_pre_worker_cleanup(
        app=app,
        guard_exceptions=startup_guard_exceptions,
    )
    worker_runtime.apply_pre_worker_cleanup_handles(pre_worker_cleanup_handles)

    from tldw_Server_API.app.services.shutdown_primary_late_stop_workers import (
        run_shutdown_primary_late_stop_workers,
    )

    primary_late_stop_worker_handles = await run_shutdown_primary_late_stop_workers(
        core_jobs_task=worker_runtime.core_jobs_task,
        core_jobs_stop_event=worker_runtime.core_jobs_stop_event,
        files_jobs_task=worker_runtime.files_jobs_task,
        files_jobs_stop_event=worker_runtime.files_jobs_stop_event,
        data_tables_jobs_task=worker_runtime.data_tables_jobs_task,
        data_tables_jobs_stop_event=worker_runtime.data_tables_jobs_stop_event,
        prompt_studio_jobs_task=worker_runtime.prompt_studio_jobs_task,
        prompt_studio_jobs_stop_event=worker_runtime.prompt_studio_jobs_stop_event,
        vllm_management_task=worker_runtime.vllm_management_task,
        vllm_management_stop_event=worker_runtime.vllm_management_stop_event,
        privilege_snapshot_task=worker_runtime.privilege_snapshot_task,
        privilege_snapshot_stop_event=worker_runtime.privilege_snapshot_stop_event,
        audio_jobs_task=worker_runtime.audio_jobs_task,
        audio_jobs_stop_event=worker_runtime.audio_jobs_stop_event,
        presentation_render_jobs_task=worker_runtime.presentation_render_jobs_task,
        presentation_render_jobs_stop_event=worker_runtime.presentation_render_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=startup_guard_exceptions,
    )
    worker_runtime.apply_primary_late_stop_worker_handles(
        primary_late_stop_worker_handles,
    )

    from tldw_Server_API.app.services.shutdown_grouped_late_stop_workers import (
        run_shutdown_grouped_late_stop_workers,
    )

    grouped_late_stop_worker_handles = await run_shutdown_grouped_late_stop_workers(
        media_ingest_jobs_task=worker_runtime.media_ingest_jobs_task,
        media_ingest_jobs_stop_event=worker_runtime.media_ingest_jobs_stop_event,
        media_ingest_heavy_jobs_task=worker_runtime.media_ingest_heavy_jobs_task,
        media_ingest_heavy_jobs_stop_event=worker_runtime.media_ingest_heavy_jobs_stop_event,
        reading_digest_jobs_task=worker_runtime.reading_digest_jobs_task,
        reading_digest_jobs_stop_event=worker_runtime.reading_digest_jobs_stop_event,
        study_pack_jobs_task=worker_runtime.study_pack_jobs_task,
        study_pack_jobs_stop_event=worker_runtime.study_pack_jobs_stop_event,
        study_suggestions_jobs_task=worker_runtime.study_suggestions_jobs_task,
        study_suggestions_jobs_stop_event=worker_runtime.study_suggestions_jobs_stop_event,
        companion_reflection_jobs_task=worker_runtime.companion_reflection_jobs_task,
        companion_reflection_jobs_stop_event=worker_runtime.companion_reflection_jobs_stop_event,
        reminder_jobs_task=worker_runtime.reminder_jobs_task,
        admin_backup_jobs_task=worker_runtime.admin_backup_jobs_task,
        admin_maintenance_rotation_jobs_task=worker_runtime.admin_maintenance_rotation_jobs_task,
        admin_maintenance_rotation_jobs_stop_event=worker_runtime.admin_maintenance_rotation_jobs_stop_event,
        recipe_run_jobs_task=worker_runtime.recipe_run_jobs_task,
        recipe_run_jobs_stop_event=worker_runtime.recipe_run_jobs_stop_event,
        evals_abtest_jobs_task=worker_runtime.evals_abtest_jobs_task,
        evals_abtest_jobs_stop_event=worker_runtime.evals_abtest_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=startup_guard_exceptions,
    )
    worker_runtime.apply_grouped_late_stop_worker_handles(
        grouped_late_stop_worker_handles,
    )

    from tldw_Server_API.app.services.shutdown_post_worker_services import (
        run_shutdown_post_worker_services,
    )

    post_worker_shutdown_handles = await run_shutdown_post_worker_services(
        jobs_notifications_bridge_task=worker_runtime.jobs_notifications_bridge_task,
        jobs_metrics_task=worker_runtime.jobs_metrics_task,
        jobs_metrics_stop_event=worker_runtime.jobs_metrics_stop_event,
        loop_lag_task=worker_runtime.loop_lag_task,
        loop_lag_stop_event=worker_runtime.loop_lag_stop_event,
        jobs_metrics_reconcile_task=worker_runtime.jobs_metrics_reconcile_task,
        jobs_metrics_reconcile_stop=worker_runtime.jobs_metrics_reconcile_stop,
        jobs_crypto_rotate_task=worker_runtime.jobs_crypto_rotate_task,
        jobs_crypto_rotate_stop_event=worker_runtime.jobs_crypto_rotate_stop_event,
        jobs_integrity_task=worker_runtime.jobs_integrity_task,
        jobs_integrity_stop_event=worker_runtime.jobs_integrity_stop_event,
        jobs_webhooks_task=worker_runtime.jobs_webhooks_task,
        jobs_webhooks_stop_event=worker_runtime.jobs_webhooks_stop_event,
        meetings_webhook_dlq_task=worker_runtime.meetings_webhook_dlq_task,
        meetings_webhook_dlq_stop_event=worker_runtime.meetings_webhook_dlq_stop_event,
        workflows_dlq_task=worker_runtime.workflows_dlq_task,
        workflows_dlq_stop_event=worker_runtime.workflows_dlq_stop_event,
        workflows_gc_task=worker_runtime.workflows_gc_task,
        workflows_gc_stop_event=worker_runtime.workflows_gc_stop_event,
        workflows_maint_task=worker_runtime.workflows_maint_task,
        workflows_maint_stop_event=worker_runtime.workflows_maint_stop_event,
        stopped_background_worker_names=stopped_background_worker_names,
        guard_exceptions=startup_guard_exceptions,
    )
    worker_runtime.apply_post_worker_shutdown_handles(
        post_worker_shutdown_handles,
    )

    from tldw_Server_API.app.services.shutdown_final_cleanup_tail import (
        shutdown_final_cleanup_tail,
    )

    cleanup_timed_shutdown_handles = await shutdown_final_cleanup_tail(
        app=app,
        db_pool=db_pool,
        session_manager=session_manager,
        heavy_startup_handles=heavy_startup_handles,
        in_pytest_for_db_pool_shutdown=in_pytest_runtime,
        in_pytest_for_tts_shutdown=in_pytest_runtime,
        import_exceptions=import_exceptions,
        startup_guard_exceptions=startup_guard_exceptions,
        test_db_instance_ref=test_db_instance_ref,
        timed_shutdown_segment=timed_shutdown_segment,
    )
    worker_runtime.apply_final_cleanup_handles(cleanup_timed_shutdown_handles)

    record_shutdown_timing_total(
        app,
        int((monotonic() - shutdown_started) * 1000),
    )


def _handles_for_shutdown_phase(
    worker_runtime: LifespanWorkerRuntimeState,
    shutdown_phase: ShutdownPhase,
) -> list[Any]:
    worker_inventory = getattr(worker_runtime, "worker_inventory", None)
    handles_for_phase = getattr(worker_inventory, "handles_for_phase", None)
    if callable(handles_for_phase):
        return list(handles_for_phase(shutdown_phase))

    target_phase_values = {shutdown_phase, shutdown_phase.value}
    return [
        handle
        for handle in getattr(worker_runtime, "owned_job_pollers", [])
        if getattr(handle, "shutdown_phase", None) in target_phase_values
    ]
