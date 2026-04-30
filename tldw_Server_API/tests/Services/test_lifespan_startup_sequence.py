from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_run_lifespan_startup_sequence_runs_helpers_in_order_and_updates_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import startup_core_initialization
    from tldw_Server_API.app.services import startup_pre_core
    from tldw_Server_API.app.services import startup_worker_bootstrap
    from tldw_Server_API.app.services.lifespan_startup_sequence import (
        run_lifespan_startup_sequence,
    )
    from tldw_Server_API.app.services.lifespan_worker_runtime_state import (
        LifespanWorkerRuntimeState,
    )

    app = FastAPI()
    worker_runtime = LifespanWorkerRuntimeState()
    calls: list[tuple[str, dict[str, object]]] = []

    async def _fake_prepare_startup_pre_core(**kwargs):
        calls.append(("pre_core", kwargs))
        return True

    async def _fake_initialize_startup_core_components(**kwargs):
        calls.append(("core", kwargs))
        return SimpleNamespace(
            db_pool="db-pool",
            session_manager="session-manager",
            heavy_startup_handles="heavy-handles",
        )

    async def _fake_initialize_startup_worker_bootstrap(**kwargs):
        calls.append(("bootstrap", kwargs))
        return startup_worker_bootstrap.StartupWorkerBootstrapHandles(
            app_settings="app-settings",
            owned_job_pollers=["poller-a"],
            startup_worker_group_handles=SimpleNamespace(
                cleanup_task="cleanup-task",
                core_jobs_task="core-task",
                audio_jobs_stop_event="audio-stop",
            ),
            startup_service_tail_handles=SimpleNamespace(
                jobs_metrics_task="jobs-metrics-task",
                claims_task="claims-task",
                authnz_scheduler_started=True,
            ),
        )

    monkeypatch.setattr(
        startup_pre_core,
        "prepare_startup_pre_core",
        _fake_prepare_startup_pre_core,
    )
    monkeypatch.setattr(
        startup_core_initialization,
        "initialize_startup_core_components",
        _fake_initialize_startup_core_components,
    )
    monkeypatch.setattr(
        startup_worker_bootstrap,
        "initialize_startup_worker_bootstrap",
        _fake_initialize_startup_worker_bootstrap,
    )

    handles = await run_lifespan_startup_sequence(
        app=app,
        worker_runtime=worker_runtime,
        module_file="module-file.py",
        logger="logger-sentinel",
        readiness_state={"ready": False},
        shared_is_truthy=lambda *_args, **_kwargs: True,
        route_enabled=lambda *_args, **_kwargs: True,
        get_mcp_config=lambda: {"enabled": True},
        validate_mcp_config=lambda *_args, **_kwargs: None,
        test_mode=True,
        run_pg_rls_auto_ensure=lambda *_args, **_kwargs: None,
        register_owned_job_poller=lambda *_args, **_kwargs: None,
        replace_owned_job_poller_inventory=lambda *_args, **_kwargs: None,
        startup_api_key_log_value="api-key-log-value",
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert [name for name, _ in calls] == ["pre_core", "core", "bootstrap"]
    assert calls[0][1]["app"] is app
    assert calls[0][1]["readiness_state"] == {"ready": False}
    assert calls[0][1]["test_mode"] is True
    assert calls[1][1]["app"] is app
    assert calls[1][1]["module_file"] == "module-file.py"
    assert calls[1][1]["defer_heavy"] is True
    assert calls[2][1]["app"] is app
    assert calls[2][1]["test_mode"] is True
    assert "publish_shutdown_job_poller_inventory" not in calls[2][1]
    assert calls[2][1]["startup_api_key_log_value"] == "api-key-log-value"
    assert handles.db_pool == "db-pool"
    assert handles.session_manager == "session-manager"
    assert handles.heavy_startup_handles == "heavy-handles"
    assert worker_runtime.owned_job_pollers == ["poller-a"]
    assert worker_runtime.cleanup_task == "cleanup-task"
    assert worker_runtime.core_jobs_task == "core-task"
    assert worker_runtime.audio_jobs_stop_event == "audio-stop"
    assert worker_runtime.jobs_metrics_task == "jobs-metrics-task"
    assert worker_runtime.claims_task == "claims-task"
    assert worker_runtime.authnz_scheduler_started is True
