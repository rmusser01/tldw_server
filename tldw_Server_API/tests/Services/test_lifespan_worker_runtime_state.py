from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_apply_startup_worker_bootstrap_handles_copies_known_fields() -> None:
    from tldw_Server_API.app.services.lifespan_worker_runtime_state import (
        LifespanWorkerRuntimeState,
    )
    from tldw_Server_API.app.services.startup_service_tail import StartupServiceTailHandles
    from tldw_Server_API.app.services.startup_worker_bootstrap import (
        StartupWorkerBootstrapHandles,
    )
    from tldw_Server_API.app.services.startup_worker_groups import StartupWorkerGroupHandles

    runtime = LifespanWorkerRuntimeState()
    startup_handles = StartupWorkerBootstrapHandles(
        app_settings="settings",
        owned_job_pollers=["poller-a", "poller-b"],
        startup_worker_group_handles=StartupWorkerGroupHandles(
            core_jobs_task="core-jobs-task",
            audio_jobs_stop_event="audio-stop",
        ),
        startup_service_tail_handles=StartupServiceTailHandles(
            jobs_metrics_task="jobs-metrics-task",
        ),
    )

    runtime.apply_startup_worker_bootstrap_handles(startup_handles)

    assert runtime.owned_job_pollers == ["poller-a", "poller-b"]
    assert not hasattr(runtime, "cleanup_task")
    assert runtime.core_jobs_task == "core-jobs-task"
    assert runtime.audio_jobs_stop_event == "audio-stop"
    assert runtime.jobs_metrics_task == "jobs-metrics-task"
    assert not hasattr(runtime, "claims_task")
    assert not hasattr(runtime, "authnz_scheduler_started")


def test_shutdown_apply_methods_copy_known_fields() -> None:
    from tldw_Server_API.app.services.lifespan_worker_runtime_state import (
        LifespanWorkerRuntimeState,
    )
    from tldw_Server_API.app.services.shutdown_cleanup_timed_segments import (
        CleanupTimedShutdownHandles,
    )
    from tldw_Server_API.app.services.shutdown_grouped_late_stop_workers import (
        GroupedLateStopWorkerHandles,
    )
    from tldw_Server_API.app.services.shutdown_post_worker_services import (
        PostWorkerShutdownHandles,
    )
    from tldw_Server_API.app.services.shutdown_pre_worker_cleanup import (
        PreWorkerCleanupHandles,
    )
    from tldw_Server_API.app.services.shutdown_primary_late_stop_workers import (
        PrimaryLateStopWorkerHandles,
    )

    runtime = LifespanWorkerRuntimeState()

    runtime.apply_pre_worker_cleanup_handles(PreWorkerCleanupHandles())
    runtime.apply_primary_late_stop_worker_handles(
        PrimaryLateStopWorkerHandles(
            core_jobs_task="core-jobs-task",
            prompt_studio_jobs_stop_event="prompt-stop",
        )
    )
    runtime.apply_grouped_late_stop_worker_handles(
        GroupedLateStopWorkerHandles(
            media_ingest_jobs_task="media-task",
            reminder_jobs_task="reminder-task",
        )
    )
    runtime.apply_post_worker_shutdown_handles(
        PostWorkerShutdownHandles(
            jobs_notifications_bridge_task="notifications-bridge-task",
            workflows_gc_task="workflows-gc-task",
        )
    )
    runtime.apply_final_cleanup_handles(CleanupTimedShutdownHandles())

    assert not hasattr(runtime, "cleanup_task")
    assert not hasattr(runtime, "chatbooks_cleanup_task")
    assert not hasattr(runtime, "storage_cleanup_service")
    assert runtime.core_jobs_task == "core-jobs-task"
    assert runtime.prompt_studio_jobs_stop_event == "prompt-stop"
    assert runtime.media_ingest_jobs_task == "media-task"
    assert runtime.reminder_jobs_task == "reminder-task"
    assert runtime.jobs_notifications_bridge_task == "notifications-bridge-task"
    assert runtime.workflows_gc_task == "workflows-gc-task"
    assert not hasattr(runtime, "authnz_scheduler_started")


def test_runtime_state_omits_registry_owned_scheduler_and_maintenance_handles() -> None:
    from tldw_Server_API.app.services.lifespan_worker_runtime_state import (
        LifespanWorkerRuntimeState,
    )

    runtime = LifespanWorkerRuntimeState()
    removed_fields = {
        "authnz_scheduler_started",
        "workflows_sched_task",
        "reading_digest_sched_task",
        "admin_backup_sched_task",
        "companion_reflection_sched_task",
        "reminders_sched_task",
        "connectors_sync_sched_task",
        "quality_eval_task",
        "outputs_purge_task",
        "kanban_activity_cleanup_task",
        "ingestion_sources_cleanup_task",
        "kanban_purge_task",
        "files_export_gc_task",
        "notifications_prune_task",
        "jobs_prune_task",
    }

    assert all(not hasattr(runtime, field_name) for field_name in removed_fields)
