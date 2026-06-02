from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.services.lifecycle_worker_specs import (
    WorkerLifecycleContext,
    WorkerSpec,
)
from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase

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

    specs = (object(),)
    lifecycle_session = SimpleNamespace(handles_by_name={"worker": "handle"})

    def _record_collect_startup_worker_specs(context: WorkerLifecycleContext):
        calls.append(
            (
                "collect_specs",
                {
                    "app": context.app,
                    "settings": context.settings,
                    "test_mode": context.test_mode,
                    "route_enabled": context.route_enabled,
                    "logger": context.logger,
                    "startup_guard_exceptions": context.startup_guard_exceptions,
                    "import_exceptions": context.import_exceptions,
                },
            )
        )
        return specs

    async def _record_start_lifecycle_workers(context, collected_specs):
        calls.append(
            (
                "lifecycle",
                {
                    "context": context,
                    "specs": collected_specs,
                },
            )
        )
        return lifecycle_session

    async def _record_run_startup_non_worker_tail(**kwargs):
        calls.append(("non_worker_tail", kwargs))

    monkeypatch.setattr(startup_bootstrap, "_load_app_settings", _record_load_app_settings)
    monkeypatch.setattr(
        startup_bootstrap,
        "_collect_startup_worker_specs",
        _record_collect_startup_worker_specs,
    )
    monkeypatch.setattr(
        startup_bootstrap,
        "_start_lifecycle_workers",
        _record_start_lifecycle_workers,
    )
    monkeypatch.setattr(
        startup_bootstrap,
        "_run_startup_non_worker_tail",
        _record_run_startup_non_worker_tail,
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
        "collect_specs",
        "lifecycle",
        "non_worker_tail",
    ]
    assert calls[1][1]["app"] == "app"
    assert calls[1][1]["settings"] == "settings"
    assert calls[1][1]["test_mode"] is True
    assert calls[1][1]["route_enabled"] == "route-enabled"
    assert calls[1][1]["logger"] == "logger"
    assert calls[1][1]["startup_guard_exceptions"] == (RuntimeError,)
    assert calls[1][1]["import_exceptions"] == (ImportError,)
    assert calls[2][1]["context"].settings == "settings"
    assert calls[2][1]["specs"] is specs
    assert calls[3][1]["app"] == "app"
    assert calls[3][1]["app_settings"] == "settings"
    assert calls[3][1]["run_pg_rls_auto_ensure"] == "run-pg-ensure"
    assert calls[3][1]["logger"] == "logger"
    assert calls[3][1]["startup_api_key_log_value"] == "api-key"
    assert calls[3][1]["shared_is_truthy"] == "truthy"
    assert calls[3][1]["startup_guard_exceptions"] == (RuntimeError,)
    assert calls[3][1]["import_exceptions"] == (ImportError,)
    assert handles.worker_lifecycle_session is lifecycle_session
    assert not hasattr(handles, "app_settings")
    assert not hasattr(handles, "owned_job_pollers")
    assert not hasattr(handles, "worker_inventory")
    assert not hasattr(handles, "startup_worker_group_handles")
    assert not hasattr(handles, "startup_service_tail_handles")


def test_startup_worker_bootstrap_no_longer_exposes_dead_poller_hook() -> None:
    from tldw_Server_API.app.services import startup_worker_bootstrap as startup_bootstrap

    signature = inspect.signature(startup_bootstrap.initialize_startup_worker_bootstrap)

    assert "publish_shutdown_job_poller_inventory" not in signature.parameters
    assert not hasattr(startup_bootstrap, "_prepare_startup_owned_job_pollers")


@pytest.mark.asyncio
async def test_initialize_startup_worker_bootstrap_starts_lifecycle_session_from_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import startup_worker_bootstrap as startup_bootstrap
    from tldw_Server_API.app.services.lifecycle_worker_engine import LifecycleWorkerEngine
    from tldw_Server_API.app.services.lifecycle_worker_session import WorkerLifecycleSession

    worker_started = False
    calls: list[str] = []

    async def _worker(stop_event):
        nonlocal worker_started
        worker_started = True
        await stop_event.wait()

    def _record_load_app_settings() -> dict[str, object]:
        calls.append("settings")
        return {"SETTING": "value"}

    def _collect_specs(context: WorkerLifecycleContext) -> tuple[WorkerSpec, ...]:
        calls.append("collect")
        assert context.app == "app"
        assert context.settings == {"SETTING": "value"}
        assert context.test_mode is True
        return (
            WorkerSpec(
                name="bootstrap_worker",
                task_name="bootstrap_worker_task",
                category="test",
                phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
                factory=lambda _context, stop_event: _worker(stop_event),
            ),
        )

    async def _record_non_worker_tail(**kwargs):
        calls.append("non_worker_tail")
        assert kwargs["app"] == "app"
        assert kwargs["app_settings"] == {"SETTING": "value"}

    monkeypatch.setattr(startup_bootstrap, "_load_app_settings", _record_load_app_settings)
    monkeypatch.setattr(startup_bootstrap, "_collect_startup_worker_specs", _collect_specs)
    monkeypatch.setattr(startup_bootstrap, "_run_startup_non_worker_tail", _record_non_worker_tail)

    handles = await startup_bootstrap.initialize_startup_worker_bootstrap(
        app="app",
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        run_pg_rls_auto_ensure="run-pg-ensure",
        register_owned_job_poller="register-poller",
        replace_owned_job_poller_inventory="replace-inventory",
        logger="logger",
        startup_api_key_log_value="api-key",
        shared_is_truthy="truthy",
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert calls == ["settings", "collect", "non_worker_tail"]
    assert isinstance(handles.worker_lifecycle_session, WorkerLifecycleSession)
    assert set(handles.worker_lifecycle_session.handles_by_name) == {"bootstrap_worker"}
    assert worker_started is True

    await LifecycleWorkerEngine().stop_phase(
        handles.worker_lifecycle_session,
        ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
    )


@pytest.mark.asyncio
async def test_initialize_startup_worker_bootstrap_records_disabled_worker_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import startup_worker_bootstrap as startup_bootstrap

    monkeypatch.setattr(startup_bootstrap, "_load_app_settings", lambda: {})
    monkeypatch.setattr(
        startup_bootstrap,
        "_collect_startup_worker_specs",
        lambda _context: (
            WorkerSpec(
                name="disabled_worker",
                task_name="disabled_worker_task",
                category="test",
                phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
                enabled=lambda _context: False,
                factory=lambda _context, _stop_event: None,
            ),
        ),
    )
    async def _skip_non_worker_tail(**_kwargs):
        return None

    monkeypatch.setattr(
        startup_bootstrap,
        "_run_startup_non_worker_tail",
        _skip_non_worker_tail,
    )

    handles = await startup_bootstrap.initialize_startup_worker_bootstrap(
        app="app",
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        run_pg_rls_auto_ensure="run-pg-ensure",
        register_owned_job_poller="register-poller",
        replace_owned_job_poller_inventory="replace-inventory",
        logger="logger",
        startup_api_key_log_value="api-key",
        shared_is_truthy="truthy",
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert handles.worker_lifecycle_session.disabled_names == {"disabled_worker"}
    assert handles.worker_lifecycle_session.handles_by_name == {}
