from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_run_lifespan_shutdown_sequence_runs_wrappers_in_order_and_updates_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import (
        lifespan_shutdown_sequence,
        shutdown_coordinated_legacy_components,
        shutdown_final_cleanup_tail,
        shutdown_grouped_late_stop_workers,
        shutdown_job_poller_handoff,
        shutdown_post_worker_services,
        shutdown_pre_worker_cleanup,
        shutdown_primary_late_stop_workers,
        shutdown_transition_handoff,
    )
    from tldw_Server_API.app.services.lifespan_worker_runtime_state import (
        LifespanWorkerRuntimeState,
    )

    app = FastAPI()
    worker_runtime = LifespanWorkerRuntimeState(
        owned_job_pollers=["poller-a"],
        core_jobs_task="core-start",
        core_jobs_stop_event="core-stop-start",
        prompt_studio_jobs_task="prompt-start",
        prompt_studio_jobs_stop_event="prompt-stop-start",
        media_ingest_jobs_task="media-start",
        media_ingest_jobs_stop_event="media-stop-start",
        media_ingest_heavy_jobs_task="media-heavy-start",
        media_ingest_heavy_jobs_stop_event="media-heavy-stop-start",
        reminder_jobs_task="reminder-start",
        admin_maintenance_rotation_jobs_task="admin-rotation-start",
        admin_maintenance_rotation_jobs_stop_event="admin-rotation-stop-start",
        recipe_run_jobs_task="recipe-start",
        recipe_run_jobs_stop_event="recipe-stop-start",
        evals_abtest_jobs_task="abtest-start",
        evals_abtest_jobs_stop_event="abtest-stop-start",
        jobs_notifications_bridge_task="bridge-start",
        jobs_metrics_task="metrics-start",
        jobs_metrics_stop_event="metrics-stop-start",
        loop_lag_task="loop-lag-start",
        loop_lag_stop_event="loop-lag-stop-start",
        jobs_metrics_reconcile_task="reconcile-start",
        jobs_metrics_reconcile_stop="reconcile-stop-start",
        jobs_crypto_rotate_task="crypto-start",
        jobs_crypto_rotate_stop_event="crypto-stop-start",
        jobs_integrity_task="integrity-start",
        jobs_integrity_stop_event="integrity-stop-start",
        jobs_webhooks_task="webhooks-start",
        jobs_webhooks_stop_event="webhooks-stop-start",
        meetings_webhook_dlq_task="meetings-start",
        meetings_webhook_dlq_stop_event="meetings-stop-start",
        workflows_dlq_task="workflows-dlq-start",
        workflows_dlq_stop_event="workflows-dlq-stop-start",
        workflows_gc_task="workflows-gc-start",
        workflows_gc_stop_event="workflows-gc-stop-start",
        workflows_maint_task="workflows-maint-start",
        workflows_maint_stop_event="workflows-maint-stop-start",
    )
    calls: list[tuple[str, dict[str, object]]] = []
    recorded_totals: list[int] = []
    monotonic_values = iter((10.0, 10.25))

    @contextmanager
    def _timed_shutdown_segment(_app: FastAPI, segment_name: str):
        calls.append(("segment", {"segment_name": segment_name}))
        yield

    async def _fake_transition_handoff(**kwargs):
        calls.append(("transition", kwargs))
        return SimpleNamespace(legacy_shutdown_plan=["transition-plan"])

    async def _fake_job_poller_handoff(**kwargs):
        calls.append(("pollers", kwargs))
        return SimpleNamespace(should_run_late_stop=True)

    async def _fake_stop_registered_workers(
        app: FastAPI,
        handles: list[object],
        *,
        stopped_names_attr: str,
        log_label: str,
    ) -> None:
        calls.append(
            (
                "stop-background",
                {
                    "handles": handles,
                    "stopped_names_attr": stopped_names_attr,
                    "log_label": log_label,
                },
            )
        )
        setattr(
            app.state,
            stopped_names_attr,
            ["authnz_scheduler", "chatbooks_cleanup", "ephemeral_cleanup_task"],
        )

    async def _fake_coordinated_shutdown(**kwargs):
        calls.append(("coordinated", kwargs))
        return SimpleNamespace(
            coordinated_legacy_component_names={"usage_aggregator"},
        )

    async def _fake_pre_worker_cleanup(**kwargs):
        calls.append(("pre", kwargs))
        return SimpleNamespace()

    async def _fake_primary_late_stop(**kwargs):
        calls.append(("primary", kwargs))
        return SimpleNamespace(
            core_jobs_task="core-after",
            prompt_studio_jobs_stop_event="prompt-stop-after",
        )

    async def _fake_grouped_late_stop(**kwargs):
        calls.append(("grouped", kwargs))
        return SimpleNamespace(
            media_ingest_jobs_task="media-after",
            reminder_jobs_task="reminder-after",
        )

    async def _fake_post_worker_services(**kwargs):
        calls.append(("post", kwargs))
        return SimpleNamespace(
            jobs_metrics_task="metrics-after",
        )

    async def _fake_final_cleanup(**kwargs):
        calls.append(("final", kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(
        shutdown_transition_handoff,
        "shutdown_transition_handoff",
        _fake_transition_handoff,
    )
    monkeypatch.setattr(
        shutdown_job_poller_handoff,
        "run_shutdown_job_poller_handoff",
        _fake_job_poller_handoff,
    )
    monkeypatch.setattr(
        lifespan_shutdown_sequence,
        "stop_registered_workers",
        _fake_stop_registered_workers,
    )
    monkeypatch.setattr(
        shutdown_coordinated_legacy_components,
        "run_shutdown_coordinated_legacy_components",
        _fake_coordinated_shutdown,
    )
    monkeypatch.setattr(
        shutdown_pre_worker_cleanup,
        "run_shutdown_pre_worker_cleanup",
        _fake_pre_worker_cleanup,
    )
    monkeypatch.setattr(
        shutdown_primary_late_stop_workers,
        "run_shutdown_primary_late_stop_workers",
        _fake_primary_late_stop,
    )
    monkeypatch.setattr(
        shutdown_grouped_late_stop_workers,
        "run_shutdown_grouped_late_stop_workers",
        _fake_grouped_late_stop,
    )
    monkeypatch.setattr(
        shutdown_post_worker_services,
        "run_shutdown_post_worker_services",
        _fake_post_worker_services,
    )
    monkeypatch.setattr(
        shutdown_final_cleanup_tail,
        "shutdown_final_cleanup_tail",
        _fake_final_cleanup,
    )
    await lifespan_shutdown_sequence.run_lifespan_shutdown_sequence(
        app=app,
        worker_runtime=worker_runtime,
        readiness_state={"ready": True},
        db_pool="db-pool",
        session_manager="session-manager",
        heavy_startup_handles="heavy-handles",
        build_legacy_shutdown_context=lambda **kwargs: kwargs,
        apply_shutdown_transition_gate=lambda *_args, **_kwargs: None,
        quiesce_owned_job_pollers_for_shutdown=lambda **_kwargs: None,
        run_coordinated_shutdown=lambda **_kwargs: None,
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
        in_pytest_runtime=True,
        test_db_instance_ref="test-db-ref",
        timed_shutdown_segment=_timed_shutdown_segment,
        record_shutdown_timing_total=lambda _app, total_ms: recorded_totals.append(total_ms),
        monotonic=lambda: next(monotonic_values),
    )

    assert [name for name, _ in calls] == [
        "segment",
        "transition",
        "pollers",
        "segment",
        "stop-background",
        "coordinated",
        "pre",
        "primary",
        "grouped",
        "post",
        "final",
    ]
    assert calls[0][1]["segment_name"] == "transition_handoff"
    assert "usage_task" not in calls[1][1]
    assert "llm_usage_task" not in calls[1][1]
    assert "chatbooks_cleanup_task" not in calls[1][1]
    assert "chatbooks_cleanup_stop_event" not in calls[1][1]
    assert "storage_cleanup_service" not in calls[1][1]
    assert calls[2][1]["owned_job_pollers"] == ["poller-a"]
    assert calls[3][1]["segment_name"] == "background_worker_shutdown"
    assert calls[4][1]["stopped_names_attr"] == "_tldw_shutdown_stopped_background_worker_names"
    assert calls[5][1]["legacy_shutdown_plan"] == ["transition-plan"]
    assert "coordinated_legacy_component_names" not in calls[6][1]
    assert "cleanup_task" not in calls[6][1]
    assert "chatbooks_cleanup_task" not in calls[6][1]
    assert "chatbooks_cleanup_stop_event" not in calls[6][1]
    assert "storage_cleanup_service" not in calls[6][1]
    assert calls[7][1]["should_run_late_stop"] is True
    assert calls[8][1]["should_run_late_stop"] is True
    assert "claims_task" not in calls[9][1]
    assert "embeddings_compactor_task" not in calls[9][1]
    assert "embeddings_compactor_stop_event" not in calls[9][1]
    assert "websub_renewal_task" not in calls[9][1]
    assert "usage_task" not in calls[9][1]
    assert "llm_usage_task" not in calls[9][1]
    assert "jobs_prune_task" not in calls[9][1]
    assert "files_export_gc_task" not in calls[9][1]
    assert "notifications_prune_task" not in calls[9][1]
    assert "workflows_sched_task" not in calls[9][1]
    assert "reading_digest_sched_task" not in calls[9][1]
    assert "admin_backup_sched_task" not in calls[9][1]
    assert "companion_reflection_sched_task" not in calls[9][1]
    assert "reminders_sched_task" not in calls[9][1]
    assert "connectors_sync_sched_task" not in calls[9][1]
    assert "authnz_scheduler_started" not in calls[10][1]
    assert "stopped_background_worker_names" not in calls[10][1]
    assert calls[10][1]["db_pool"] == "db-pool"
    assert calls[10][1]["session_manager"] == "session-manager"
    assert calls[10][1]["heavy_startup_handles"] == "heavy-handles"
    assert calls[10][1]["in_pytest_for_db_pool_shutdown"] is True
    assert calls[10][1]["in_pytest_for_tts_shutdown"] is True
    assert not hasattr(worker_runtime, "cleanup_task")
    assert not hasattr(worker_runtime, "chatbooks_cleanup_task")
    assert not hasattr(worker_runtime, "storage_cleanup_service")
    assert not hasattr(worker_runtime, "claims_task")
    assert not hasattr(worker_runtime, "embeddings_compactor_task")
    assert not hasattr(worker_runtime, "embeddings_compactor_stop_event")
    assert not hasattr(worker_runtime, "websub_renewal_task")
    assert not hasattr(worker_runtime, "usage_task")
    assert not hasattr(worker_runtime, "llm_usage_task")
    assert not hasattr(worker_runtime, "authnz_scheduler_started")
    assert not hasattr(worker_runtime, "files_export_gc_task")
    assert not hasattr(worker_runtime, "notifications_prune_task")
    assert worker_runtime.core_jobs_task == "core-after"
    assert worker_runtime.prompt_studio_jobs_stop_event == "prompt-stop-after"
    assert worker_runtime.media_ingest_jobs_task == "media-after"
    assert worker_runtime.reminder_jobs_task == "reminder-after"
    assert worker_runtime.jobs_metrics_task == "metrics-after"
    assert app.state._tldw_shutdown_timing_segments == []
    assert recorded_totals == [250]
