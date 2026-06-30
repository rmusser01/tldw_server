from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_initialize_startup_service_tail_runs_helpers_in_order_and_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import startup_service_tail as startup_tail

    calls: list[tuple[str, dict[str, object]]] = []

    async def _record_start_service_groups(**kwargs):
        calls.append(("service_groups", kwargs))
        return SimpleNamespace(
            jobs_metrics_stop_event="jobs-metrics-stop",
            jobs_metrics_task="jobs-metrics-task",
            loop_lag_stop_event="loop-lag-stop",
            loop_lag_task="loop-lag-task",
            jobs_metrics_reconcile_stop="reconcile-stop",
            jobs_metrics_reconcile_task="reconcile-task",
            jobs_crypto_rotate_stop_event="crypto-stop",
            jobs_crypto_rotate_task="crypto-task",
            jobs_webhooks_stop_event="webhooks-stop",
            jobs_webhooks_task="webhooks-task",
            meetings_webhook_dlq_stop_event="meetings-stop",
            meetings_webhook_dlq_task="meetings-task",
            workflows_dlq_stop_event="workflows-dlq-stop",
            workflows_dlq_task="workflows-dlq-task",
            workflows_gc_stop_event="workflows-gc-stop",
            workflows_gc_task="workflows-gc-task",
            workflows_maint_stop_event="workflows-maint-stop",
            workflows_maint_task="workflows-maint-task",
            jobs_integrity_stop_event="integrity-stop",
            jobs_integrity_task="integrity-task",
            claims_alerts_task="claims-alerts-task",
            claims_review_metrics_task="claims-review-task",
            tts_history_cleanup_task="tts-history-task",
            tts_history_cleanup_stop_event="tts-history-stop",
            connectors_jobs_task="connectors-task",
            connectors_jobs_stop_event="connectors-stop",
        )

    async def _record_finalize_startup_tail(**kwargs):
        calls.append(("finalize", kwargs))
        return SimpleNamespace()

    async def _record_report_startup_environment(**kwargs):
        calls.append(("report", kwargs))

    monkeypatch.setattr(startup_tail, "_start_service_groups", _record_start_service_groups)
    monkeypatch.setattr(startup_tail, "_finalize_startup_tail", _record_finalize_startup_tail)
    monkeypatch.setattr(
        startup_tail,
        "_report_startup_environment",
        _record_report_startup_environment,
    )

    handles = await startup_tail.initialize_startup_service_tail(
        app="app",
        app_settings="settings",
        run_pg_rls_auto_ensure="pg-ensure",
        owned_job_pollers=["poller"],
        register_owned_job_poller="register-poller",
        startup_worker_group_handles="worker-handles",
        replace_owned_job_poller_inventory="replace-inventory",
        test_mode=True,
        logger="logger",
        startup_api_key_log_value="api-key",
        shared_is_truthy="truthy",
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert [name for name, _ in calls] == ["service_groups", "finalize", "report"]
    assert calls[0][1]["app"] == "app"
    assert calls[0][1]["app_settings"] == "settings"
    assert calls[0][1]["run_pg_rls_auto_ensure"] == "pg-ensure"
    assert calls[0][1]["owned_job_pollers"] == ["poller"]
    assert calls[1][1]["app"] == "app"
    assert calls[1][1]["startup_worker_group_handles"] == "worker-handles"
    assert calls[1][1]["replace_owned_job_poller_inventory"] == "replace-inventory"
    assert calls[1][1]["test_mode"] is True
    assert calls[2][1]["app"] == "app"
    assert calls[2][1]["logger"] == "logger"
    assert calls[2][1]["startup_api_key_log_value"] == "api-key"
    assert calls[2][1]["shared_is_truthy"] == "truthy"
    assert calls[2][1]["startup_guard_exceptions"] == (RuntimeError,)
    assert calls[2][1]["import_exceptions"] == (ImportError,)
    assert handles.jobs_metrics_task == "jobs-metrics-task"
    assert not hasattr(handles, "claims_task")
    assert handles.connectors_jobs_task == "connectors-task"
    assert not hasattr(handles, "authnz_scheduler_started")
    assert not hasattr(handles, "workflows_sched_task")
    assert not hasattr(handles, "connectors_sync_sched_task")
