from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def _import_startup_service_groups():
    sys.modules.pop("tldw_Server_API.app.services.startup_service_groups", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_service_groups")


@pytest.mark.asyncio
async def test_start_service_groups_runs_helpers_in_order_and_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_groups = _import_startup_service_groups()
    calls: list[str] = []
    app = object()
    app_settings = {"SINGLE_USER_FIXED_ID": "9"}
    owned_job_pollers: list[object] = []
    register_owned_job_poller = object()
    run_pg_rls_auto_ensure = object()
    worker_inventory = object()
    worker_inventory_ref = worker_inventory

    async def _record_runtime_monitors(*, worker_inventory):
        assert worker_inventory is worker_inventory_ref
        calls.append("runtime")
        return SimpleNamespace(
            jobs_metrics_stop_event="jobs-metrics-stop",
            jobs_metrics_task="jobs-metrics-task",
            loop_lag_stop_event="loop-lag-stop",
            loop_lag_task="loop-lag-task",
        )

    async def _record_optional_workers(*, worker_inventory):
        assert worker_inventory is worker_inventory_ref
        calls.append("optional")
        return SimpleNamespace(
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
        )

    async def _record_claims_rebuild_worker(seen_app_settings):
        assert seen_app_settings is app_settings
        calls.append("claims")
        return "claims-task"

    async def _record_auxiliary_services(seen_app_settings):
        assert seen_app_settings is app_settings
        calls.append("auxiliary")
        return SimpleNamespace(
            claims_alerts_task="claims-alerts-task",
            claims_review_metrics_task="claims-review-task",
            usage_task="usage-task",
            llm_usage_task="llm-usage-task",
        )

    async def _record_infra_services(*, run_pg_rls_auto_ensure, worker_inventory):
        assert run_pg_rls_auto_ensure is run_pg_rls_auto_ensure_ref
        assert worker_inventory is worker_inventory_ref
        calls.append("infra")
        return SimpleNamespace(
            tts_history_cleanup_task="tts-history-task",
            tts_history_cleanup_stop_event="tts-history-stop",
        )

    async def _record_maintenance_schedulers(*, worker_inventory=None):
        assert worker_inventory is worker_inventory_ref
        calls.append("maintenance")
        return SimpleNamespace(
            quality_eval_task="quality-task",
            outputs_purge_task="purge-task",
            kanban_activity_cleanup_task="kanban-cleanup-task",
            ingestion_sources_cleanup_task="ingestion-task",
            kanban_purge_task="kanban-purge-task",
            files_export_gc_task="files-gc-task",
            notifications_prune_task="notifications-prune-task",
            jobs_prune_task="jobs-prune-task",
        )

    async def _record_connectors_startup(
        *,
        app,
        owned_job_pollers,
        register_owned_job_poller,
    ):
        assert app is app_ref
        assert owned_job_pollers is owned_job_pollers_ref
        assert register_owned_job_poller is register_owned_job_poller_ref
        calls.append("connectors")
        return SimpleNamespace(
            connectors_jobs_task="connectors-task",
            connectors_jobs_stop_event="connectors-stop",
        )

    app_ref = app
    owned_job_pollers_ref = owned_job_pollers
    register_owned_job_poller_ref = register_owned_job_poller
    run_pg_rls_auto_ensure_ref = run_pg_rls_auto_ensure
    worker_inventory_ref = worker_inventory

    monkeypatch.setattr(startup_groups, "_start_runtime_monitors", _record_runtime_monitors)
    monkeypatch.setattr(startup_groups, "_start_optional_workers", _record_optional_workers)
    monkeypatch.setattr(startup_groups, "_start_claims_rebuild_worker", _record_claims_rebuild_worker)
    monkeypatch.setattr(startup_groups, "_start_auxiliary_services", _record_auxiliary_services)
    monkeypatch.setattr(startup_groups, "_start_infra_services", _record_infra_services)
    monkeypatch.setattr(
        startup_groups,
        "_start_maintenance_schedulers",
        _record_maintenance_schedulers,
    )
    monkeypatch.setattr(startup_groups, "_start_connectors_startup", _record_connectors_startup)

    handles = await startup_groups.start_service_groups(
        app=app,
        app_settings=app_settings,
        run_pg_rls_auto_ensure=run_pg_rls_auto_ensure,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        worker_inventory=worker_inventory_ref,
    )

    assert calls == [
        "runtime",
        "optional",
        "claims",
        "auxiliary",
        "infra",
        "maintenance",
        "connectors",
    ]
    assert handles.jobs_metrics_task == "jobs-metrics-task"
    assert handles.jobs_integrity_task == "integrity-task"
    assert handles.claims_task == "claims-task"
    assert handles.usage_task == "usage-task"
    assert handles.tts_history_cleanup_task == "tts-history-task"
    assert handles.jobs_prune_task == "jobs-prune-task"
    assert handles.connectors_jobs_task == "connectors-task"


@pytest.mark.asyncio
async def test_start_service_groups_keeps_no_arg_maintenance_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_groups = _import_startup_service_groups()
    calls: list[str] = []

    async def _record_runtime_monitors():
        calls.append("runtime")
        return SimpleNamespace(
            jobs_metrics_stop_event=None,
            jobs_metrics_task=None,
            loop_lag_stop_event=None,
            loop_lag_task=None,
        )

    async def _record_optional_workers():
        calls.append("optional")
        return SimpleNamespace(
            jobs_metrics_reconcile_stop=None,
            jobs_metrics_reconcile_task=None,
            jobs_crypto_rotate_stop_event=None,
            jobs_crypto_rotate_task=None,
            jobs_webhooks_stop_event=None,
            jobs_webhooks_task=None,
            meetings_webhook_dlq_stop_event=None,
            meetings_webhook_dlq_task=None,
            workflows_dlq_stop_event=None,
            workflows_dlq_task=None,
            workflows_gc_stop_event=None,
            workflows_gc_task=None,
            workflows_maint_stop_event=None,
            workflows_maint_task=None,
            jobs_integrity_stop_event=None,
            jobs_integrity_task=None,
        )

    async def _record_claims_rebuild_worker(app_settings):
        calls.append("claims")
        return None

    async def _record_auxiliary_services(app_settings):
        calls.append("auxiliary")
        return SimpleNamespace(
            claims_alerts_task=None,
            claims_review_metrics_task=None,
            usage_task=None,
            llm_usage_task=None,
        )

    async def _record_infra_services(*, run_pg_rls_auto_ensure, worker_inventory=None):
        assert worker_inventory is None
        calls.append("infra")
        return SimpleNamespace(
            tts_history_cleanup_task=None,
            tts_history_cleanup_stop_event=None,
        )

    async def _record_maintenance_schedulers():
        calls.append("maintenance")
        return SimpleNamespace(
            quality_eval_task=None,
            outputs_purge_task=None,
            kanban_activity_cleanup_task=None,
            ingestion_sources_cleanup_task=None,
            kanban_purge_task=None,
            files_export_gc_task=None,
            notifications_prune_task=None,
            jobs_prune_task=None,
        )

    async def _record_connectors_startup(
        *,
        app,
        owned_job_pollers,
        register_owned_job_poller,
    ):
        calls.append("connectors")
        return SimpleNamespace(
            connectors_jobs_task=None,
            connectors_jobs_stop_event=None,
        )

    monkeypatch.setattr(startup_groups, "_start_runtime_monitors", _record_runtime_monitors)
    monkeypatch.setattr(startup_groups, "_start_optional_workers", _record_optional_workers)
    monkeypatch.setattr(startup_groups, "_start_claims_rebuild_worker", _record_claims_rebuild_worker)
    monkeypatch.setattr(startup_groups, "_start_auxiliary_services", _record_auxiliary_services)
    monkeypatch.setattr(startup_groups, "_start_infra_services", _record_infra_services)
    monkeypatch.setattr(
        startup_groups,
        "_start_maintenance_schedulers",
        _record_maintenance_schedulers,
    )
    monkeypatch.setattr(startup_groups, "_start_connectors_startup", _record_connectors_startup)

    await startup_groups.start_service_groups(
        app=object(),
        app_settings={},
        run_pg_rls_auto_ensure=object(),
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
    )

    assert calls == [
        "runtime",
        "optional",
        "claims",
        "auxiliary",
        "infra",
        "maintenance",
        "connectors",
    ]


@pytest.mark.asyncio
async def test_start_maintenance_schedulers_wrapper_supports_no_arg_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_groups = _import_startup_service_groups()
    startup_maintenance = importlib.import_module(
        "tldw_Server_API.app.services.startup_maintenance_schedulers"
    )
    calls: list[str] = []

    async def _fake_start_maintenance_schedulers():
        calls.append("maintenance")
        return "maintenance-handles"

    monkeypatch.setattr(
        startup_maintenance,
        "start_maintenance_schedulers",
        _fake_start_maintenance_schedulers,
    )

    handles = await startup_groups._start_maintenance_schedulers(
        worker_inventory=object(),
    )

    assert handles == "maintenance-handles"
    assert calls == ["maintenance"]
