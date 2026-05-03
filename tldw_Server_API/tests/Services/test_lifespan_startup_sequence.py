from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_run_lifespan_startup_sequence_runs_helpers_in_order_and_updates_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import startup_core_initialization, startup_pre_core, startup_worker_bootstrap
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
    assert not hasattr(worker_runtime, "cleanup_task")
    assert worker_runtime.core_jobs_task == "core-task"
    assert worker_runtime.audio_jobs_stop_event == "audio-stop"
    assert worker_runtime.jobs_metrics_task == "jobs-metrics-task"
    assert not hasattr(worker_runtime, "claims_task")
    assert not hasattr(worker_runtime, "authnz_scheduler_started")


@pytest.mark.asyncio
async def test_startup_initializes_registry_and_runs_sandbox_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import (
        startup_core_initialization,
        startup_pre_core,
        startup_warning_sandbox,
        startup_worker_bootstrap,
    )
    from tldw_Server_API.app.services.lifespan_startup_sequence import (
        run_lifespan_startup_sequence,
    )
    from tldw_Server_API.app.services.lifespan_worker_runtime_state import (
        LifespanWorkerRuntimeState,
    )

    app = FastAPI()
    worker_runtime = LifespanWorkerRuntimeState()
    producer_calls: list[dict[str, object]] = []

    async def _fake_prepare_startup_pre_core(**kwargs):
        return True

    async def _fake_initialize_startup_core_components(**kwargs):
        return SimpleNamespace(
            db_pool="db-pool",
            session_manager="session-manager",
            heavy_startup_handles="heavy-handles",
            startup_sandbox_orchestrator="orch-sentinel",
        )

    async def _fake_initialize_startup_worker_bootstrap(**kwargs):
        return startup_worker_bootstrap.StartupWorkerBootstrapHandles(
            app_settings="app-settings",
            owned_job_pollers=[],
            startup_worker_group_handles=SimpleNamespace(),
            startup_service_tail_handles=SimpleNamespace(),
        )

    def _fake_produce_sandbox_startup_warnings(**kwargs):
        producer_calls.append(kwargs)
        return []

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
    monkeypatch.setattr(
        startup_warning_sandbox,
        "produce_sandbox_startup_warnings",
        _fake_produce_sandbox_startup_warnings,
    )

    await run_lifespan_startup_sequence(
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

    registry = app.state.startup_warning_registry
    assert registry.summary()["total"] == 0
    assert producer_calls == [
        {
            "orchestrator": "orch-sentinel",
            "registry": registry,
        }
    ]


@pytest.mark.asyncio
async def test_startup_blocks_on_protocol_mismatch_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import (
        startup_core_initialization,
        startup_pre_core,
        startup_warning_sandbox,
        startup_worker_bootstrap,
    )
    from tldw_Server_API.app.services.lifespan_startup_sequence import (
        run_lifespan_startup_sequence,
    )
    from tldw_Server_API.app.services.lifespan_worker_runtime_state import (
        LifespanWorkerRuntimeState,
    )
    from tldw_Server_API.app.services.startup_warning_models import (
        StartupWarningRecord,
    )

    app = FastAPI()
    worker_runtime = LifespanWorkerRuntimeState()

    async def _fake_prepare_startup_pre_core(**kwargs):
        return True

    async def _fake_initialize_startup_core_components(**kwargs):
        return SimpleNamespace(
            db_pool="db-pool",
            session_manager="session-manager",
            heavy_startup_handles="heavy-handles",
            startup_sandbox_orchestrator="orch-sentinel",
        )

    async def _fake_initialize_startup_worker_bootstrap(**kwargs):
        return startup_worker_bootstrap.StartupWorkerBootstrapHandles(
            app_settings="app-settings",
            owned_job_pollers=[],
            startup_worker_group_handles=SimpleNamespace(),
            startup_service_tail_handles=SimpleNamespace(),
        )

    def _fake_produce_sandbox_startup_warnings(**kwargs):
        kwargs["registry"].add_warning(
            StartupWarningRecord(
                component="sandbox.vz_linux",
                severity="error",
                startup_action="block_startup",
                code="vz_helper_protocol_mismatch",
                summary="protocol mismatch",
                remediation="restart the helper",
                details={"reason": "protocol_mismatch"},
            )
        )
        return []

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
    monkeypatch.setattr(
        startup_warning_sandbox,
        "produce_sandbox_startup_warnings",
        _fake_produce_sandbox_startup_warnings,
    )

    with pytest.raises(RuntimeError, match="vz_helper_protocol_mismatch"):
        await run_lifespan_startup_sequence(
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


@pytest.mark.asyncio
async def test_startup_warning_registry_is_available_on_app_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import (
        startup_core_initialization,
        startup_pre_core,
        startup_warning_sandbox,
        startup_worker_bootstrap,
    )
    from tldw_Server_API.app.services.lifespan_startup_sequence import (
        run_lifespan_startup_sequence,
    )
    from tldw_Server_API.app.services.lifespan_worker_runtime_state import (
        LifespanWorkerRuntimeState,
    )
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )

    app = FastAPI()
    worker_runtime = LifespanWorkerRuntimeState()

    async def _fake_prepare_startup_pre_core(**kwargs):
        return True

    async def _fake_initialize_startup_core_components(**kwargs):
        return SimpleNamespace(
            db_pool="db-pool",
            session_manager="session-manager",
            heavy_startup_handles="heavy-handles",
            startup_sandbox_orchestrator="orch-sentinel",
        )

    async def _fake_initialize_startup_worker_bootstrap(**kwargs):
        return startup_worker_bootstrap.StartupWorkerBootstrapHandles(
            app_settings="app-settings",
            owned_job_pollers=[],
            startup_worker_group_handles=SimpleNamespace(),
            startup_service_tail_handles=SimpleNamespace(),
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
    monkeypatch.setattr(
        startup_warning_sandbox,
        "produce_sandbox_startup_warnings",
        lambda **kwargs: [],
    )

    await run_lifespan_startup_sequence(
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

    assert isinstance(app.state.startup_warning_registry, StartupWarningRegistry)
