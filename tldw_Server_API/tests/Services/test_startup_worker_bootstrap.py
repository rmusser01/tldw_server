from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_initialize_startup_worker_bootstrap_runs_helpers_in_order_and_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import startup_worker_bootstrap as startup_bootstrap

    calls: list[tuple[str, dict[str, object]]] = []

    def _record_load_app_settings() -> str:
        calls.append(("app_settings", {}))
        return "settings"

    async def _record_start_worker_groups(**kwargs):
        calls.append(("worker_groups", kwargs))
        return SimpleNamespace(cleanup_task="cleanup-task")

    async def _record_initialize_startup_service_tail(**kwargs):
        calls.append(("service_tail", kwargs))
        return SimpleNamespace(jobs_metrics_task="jobs-metrics-task")

    monkeypatch.setattr(startup_bootstrap, "_load_app_settings", _record_load_app_settings)
    monkeypatch.setattr(startup_bootstrap, "_start_worker_groups", _record_start_worker_groups)
    monkeypatch.setattr(
        startup_bootstrap,
        "_initialize_startup_service_tail",
        _record_initialize_startup_service_tail,
    )

    handles = await startup_bootstrap.initialize_startup_worker_bootstrap(
        app="app",
        test_mode=True,
        route_enabled="route-enabled",
        run_pg_rls_auto_ensure="run-pg-ensure",
        register_owned_job_poller="register-poller",
        replace_owned_job_poller_inventory="replace-inventory",
        logger="logger",
        startup_api_key_log_value="api-key",
        shared_is_truthy="truthy",
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert [name for name, _ in calls] == [
        "app_settings",
        "worker_groups",
        "service_tail",
    ]
    assert calls[1][1]["app"] == "app"
    assert calls[1][1]["app_settings"] == "settings"
    assert calls[1][1]["test_mode"] is True
    assert calls[1][1]["route_enabled"] == "route-enabled"
    assert calls[1][1]["owned_job_pollers"] == []
    assert calls[1][1]["worker_inventory"].handles is calls[1][1]["owned_job_pollers"]
    assert calls[1][1]["register_owned_job_poller"] == "register-poller"
    assert calls[1][1]["startup_guard_exceptions"] == (RuntimeError,)
    assert calls[1][1]["worker_inventory"].handles is calls[1][1]["owned_job_pollers"]
    assert calls[2][1]["app"] == "app"
    assert calls[2][1]["app_settings"] == "settings"
    assert calls[2][1]["run_pg_rls_auto_ensure"] == "run-pg-ensure"
    assert calls[2][1]["owned_job_pollers"] == []
    assert calls[2][1]["worker_inventory"].handles is calls[2][1]["owned_job_pollers"]
    assert calls[2][1]["register_owned_job_poller"] == "register-poller"
    assert calls[2][1]["startup_worker_group_handles"].cleanup_task == "cleanup-task"
    assert calls[2][1]["replace_owned_job_poller_inventory"] == "replace-inventory"
    assert calls[2][1]["test_mode"] is True
    assert calls[2][1]["logger"] == "logger"
    assert calls[2][1]["startup_api_key_log_value"] == "api-key"
    assert calls[2][1]["shared_is_truthy"] == "truthy"
    assert calls[2][1]["startup_guard_exceptions"] == (RuntimeError,)
    assert calls[2][1]["import_exceptions"] == (ImportError,)
    assert handles.app_settings == "settings"
    assert handles.owned_job_pollers == []
    assert handles.worker_inventory is calls[2][1]["worker_inventory"]
    assert handles.startup_worker_group_handles.cleanup_task == "cleanup-task"
    assert handles.startup_service_tail_handles.jobs_metrics_task == "jobs-metrics-task"


def test_startup_worker_bootstrap_no_longer_exposes_dead_poller_hook() -> None:
    from tldw_Server_API.app.services import startup_worker_bootstrap as startup_bootstrap

    signature = inspect.signature(startup_bootstrap.initialize_startup_worker_bootstrap)

    assert "publish_shutdown_job_poller_inventory" not in signature.parameters
    assert not hasattr(startup_bootstrap, "_prepare_startup_owned_job_pollers")
