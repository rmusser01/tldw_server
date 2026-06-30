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
async def test_start_service_groups_requires_worker_inventory_before_starting_registry_owned_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_groups = _import_startup_service_groups()
    calls: list[str] = []

    async def _record_runtime_monitors() -> SimpleNamespace:
        calls.append("runtime")
        return SimpleNamespace(
            jobs_metrics_stop_event=None,
            jobs_metrics_task=None,
            loop_lag_stop_event=None,
            loop_lag_task=None,
        )

    async def _record_optional_workers() -> SimpleNamespace:
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

    async def _record_claims_rebuild_worker(_app_settings: object) -> None:
        calls.append("claims")

    async def _record_auxiliary_services(_app_settings: object) -> SimpleNamespace:
        calls.append("auxiliary")
        return SimpleNamespace(
            claims_alerts_task=None,
            claims_review_metrics_task=None,
        )

    async def _record_infra_services(**_kwargs: object) -> SimpleNamespace:
        calls.append("infra")
        return SimpleNamespace(
            tts_history_cleanup_task=None,
            tts_history_cleanup_stop_event=None,
        )

    async def _record_maintenance_schedulers(
        *,
        worker_inventory: object | None = None,
    ) -> SimpleNamespace:
        assert worker_inventory is None
        calls.append("maintenance")
        return SimpleNamespace()

    async def _record_connectors_startup(**_kwargs: object) -> SimpleNamespace:
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

    with pytest.raises(RuntimeError, match="worker_inventory is required"):
        await startup_groups.start_service_groups(
            app=object(),
            app_settings={},
            run_pg_rls_auto_ensure=object(),
            owned_job_pollers=[],
            register_owned_job_poller=object(),
            worker_inventory=None,
        )

    assert calls == []


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

    async def _record_runtime_monitors(*, worker_inventory: object) -> SimpleNamespace:
        assert worker_inventory is worker_inventory_ref
        calls.append("runtime")
        return SimpleNamespace(
            jobs_metrics_stop_event="jobs-metrics-stop",
            jobs_metrics_task="jobs-metrics-task",
            loop_lag_stop_event="loop-lag-stop",
            loop_lag_task="loop-lag-task",
        )

    async def _record_optional_workers(*, worker_inventory: object) -> SimpleNamespace:
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

    async def _record_claims_rebuild_worker(
        seen_app_settings: object,
        **kwargs: object,
    ) -> str:
        assert seen_app_settings is app_settings
        assert kwargs == {"worker_inventory": worker_inventory_ref}
        calls.append("claims")
        return "claims-task"

    async def _record_auxiliary_services(
        seen_app_settings: object,
        **kwargs: object,
    ) -> SimpleNamespace:
        assert seen_app_settings is app_settings
        assert kwargs == {"worker_inventory": worker_inventory_ref}
        calls.append("auxiliary")
        return SimpleNamespace(
            claims_alerts_task="claims-alerts-task",
            claims_review_metrics_task="claims-review-task",
        )

    async def _record_infra_services(
        *,
        run_pg_rls_auto_ensure: object,
        worker_inventory: object,
    ) -> SimpleNamespace:
        assert run_pg_rls_auto_ensure is run_pg_rls_auto_ensure_ref
        assert worker_inventory is worker_inventory_ref
        calls.append("infra")
        return SimpleNamespace(
            tts_history_cleanup_task="tts-history-task",
            tts_history_cleanup_stop_event="tts-history-stop",
        )

    async def _record_maintenance_schedulers(
        *,
        worker_inventory: object | None = None,
    ) -> SimpleNamespace:
        assert worker_inventory is worker_inventory_ref
        calls.append("maintenance")
        return SimpleNamespace()

    async def _record_connectors_startup(
        *,
        app: object,
        owned_job_pollers: list[object],
        register_owned_job_poller: object,
    ) -> SimpleNamespace:
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
    assert not hasattr(handles, "claims_task")
    assert not hasattr(handles, "usage_task")
    assert handles.tts_history_cleanup_task == "tts-history-task"
    assert not hasattr(handles, "jobs_prune_task")
    assert handles.connectors_jobs_task == "connectors-task"


@pytest.mark.asyncio
async def test_start_service_groups_passes_worker_inventory_to_maintenance_schedulers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_groups = _import_startup_service_groups()
    worker_inventory = object()
    maintenance_kwargs: dict[str, object] = {}

    async def _record_runtime_monitors(*, worker_inventory: object) -> SimpleNamespace:
        return SimpleNamespace(
            jobs_metrics_stop_event=None,
            jobs_metrics_task=None,
            loop_lag_stop_event=None,
            loop_lag_task=None,
        )

    async def _record_optional_workers(*, worker_inventory: object) -> SimpleNamespace:
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

    async def _record_claims_rebuild_worker(
        _app_settings: object,
        **_kwargs: object,
    ) -> None:
        return None

    async def _record_auxiliary_services(
        _app_settings: object,
        **_kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            claims_alerts_task=None,
            claims_review_metrics_task=None,
        )

    async def _record_infra_services(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            tts_history_cleanup_task=None,
            tts_history_cleanup_stop_event=None,
        )

    async def _record_maintenance_schedulers(**kwargs: object) -> SimpleNamespace:
        maintenance_kwargs.update(kwargs)
        return SimpleNamespace()

    async def _record_connectors_startup(**_kwargs: object) -> SimpleNamespace:
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
        register_owned_job_poller=object(),
        worker_inventory=worker_inventory,
    )

    assert maintenance_kwargs == {"worker_inventory": worker_inventory}


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
