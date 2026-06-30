from __future__ import annotations

import importlib
import inspect
import sys
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.services.lifecycle_worker_specs import (
    ShutdownPhase,
    WorkerLifecycleContext,
    WorkerStrategy,
)

pytestmark = pytest.mark.unit


def _import_startup_infra_services():
    sys.modules.pop("tldw_Server_API.app.services.startup_infra_services", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_infra_services")


def _context() -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app=object(),
        settings={},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )


def _specs_by_name(startup_infra):
    return {
        spec.name: spec
        for spec in startup_infra.provide_infra_worker_specs()
    }


def test_infra_worker_specs_match_legacy_worker_contract() -> None:
    startup_infra = _import_startup_infra_services()

    specs = _specs_by_name(startup_infra)

    assert set(specs) == {"tts_history_cleanup_task", "connectors_jobs_task"}

    tts = specs["tts_history_cleanup_task"]
    assert tts.task_name == "tts_history_cleanup_task"
    assert tts.category == "maintenance"
    assert tts.phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
    assert tts.timeout_sec == 5.0
    assert tts.strategy is WorkerStrategy.STOP_EVENT_TASK
    assert tts.factory is not None

    connectors = specs["connectors_jobs_task"]
    assert connectors.task_name == "connectors_jobs_task"
    assert connectors.category == "jobs"
    assert connectors.phase is ShutdownPhase.JOB_POLLER_QUIESCE
    assert connectors.timeout_sec == 5.0
    assert connectors.strategy is WorkerStrategy.STOP_EVENT_TASK
    assert connectors.factory is not None


def test_infra_worker_spec_factories_delegate_to_existing_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_infra = _import_startup_infra_services()
    calls: list[tuple[str, object]] = []

    async def _fake_connectors(*, stop_event: object) -> str:
        calls.append(("connectors", stop_event))
        return "connectors-task"

    monkeypatch.setattr(
        startup_infra,
        "_run_tts_history_cleanup_loop",
        lambda stop_event: calls.append(("tts", stop_event)) or "tts-coro",
    )
    monkeypatch.setattr(startup_infra, "_start_connectors_worker_service", _fake_connectors)

    specs = _specs_by_name(startup_infra)

    assert specs["tts_history_cleanup_task"].factory(_context(), "tts-stop") == "tts-coro"
    connectors_result = specs["connectors_jobs_task"].factory(_context(), "connectors-stop")
    assert inspect.isawaitable(connectors_result)
    connectors_result.close()
    assert calls == [("tts", "tts-stop")]


@pytest.mark.asyncio
async def test_start_infra_services_runs_pg_rls_then_tts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_infra = _import_startup_infra_services()
    calls: list[str] = []

    async def _fake_pg_rls(run_pg_rls_auto_ensure):
        calls.append("pg-rls")
        assert run_pg_rls_auto_ensure is not None

    async def _fake_tts(*, worker_inventory):
        assert worker_inventory is None
        calls.append("tts")
        return ("tts-task", "tts-stop")

    monkeypatch.setattr(startup_infra, "_maybe_ensure_pg_rls", _fake_pg_rls)
    monkeypatch.setattr(startup_infra, "_start_tts_history_cleanup_worker", _fake_tts)

    handles = await startup_infra.start_infra_services(
        run_pg_rls_auto_ensure=object(),
    )

    assert calls == ["pg-rls", "tts"]
    assert handles.tts_history_cleanup_task == "tts-task"
    assert handles.tts_history_cleanup_stop_event == "tts-stop"


@pytest.mark.asyncio
async def test_maybe_ensure_pg_rls_skips_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_infra = _import_startup_infra_services()
    invoked = []

    monkeypatch.setattr(startup_infra, "_env_flag_enabled", lambda key: False)

    def _fake_run(backend):
        invoked.append(backend)

    await startup_infra._maybe_ensure_pg_rls(_fake_run)

    assert invoked == []


@pytest.mark.asyncio
async def test_start_tts_history_cleanup_worker_creates_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_infra = _import_startup_infra_services()
    created_tasks = []

    async def _fake_runner(stop_event):
        return stop_event

    def _record_create_task(coro, *, name=None):
        assert name == "tts_history_cleanup_task"
        task = SimpleNamespace(coro=coro, cancel=lambda: None)
        created_tasks.append(task)
        coro.close()
        return task

    monkeypatch.setattr(startup_infra.asyncio, "create_task", _record_create_task)
    monkeypatch.setattr(startup_infra, "_run_tts_history_cleanup_loop", _fake_runner)

    task, stop_event = await startup_infra._start_tts_history_cleanup_worker()

    assert task is created_tasks[0]
    assert stop_event is not None


@pytest.mark.asyncio
async def test_start_tts_history_cleanup_worker_registers_background_inventory() -> None:
    startup_infra = _import_startup_infra_services()
    registrations: list[dict[str, object]] = []

    class _FakeInventory:
        async def register_custom(self, **kwargs):
            registrations.append(kwargs)
            coroutine = kwargs["coroutine_factory"](startup_infra.asyncio.Event())
            assert inspect.isawaitable(coroutine)
            coroutine.close()
            return "tts-task", "tts-stop"

    task, stop_event = await startup_infra._start_tts_history_cleanup_worker(
        worker_inventory=_FakeInventory(),
    )

    assert task == "tts-task"
    assert stop_event == "tts-stop"
    assert registrations == [
        {
            "name": "tts_history_cleanup_task",
            "task_name": "tts_history_cleanup_task",
            "coroutine_factory": startup_infra._run_tts_history_cleanup_loop,
            "timeout_sec": 5.0,
            "category": "maintenance",
            "shutdown_phase": startup_infra.ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        }
    ]


@pytest.mark.asyncio
async def test_start_connectors_startup_registers_owned_poller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_infra = _import_startup_infra_services()
    registrations = []

    async def _fake_start(*, stop_event):
        assert stop_event is not None
        return "connectors-task"

    def _fake_register(app, owned_job_pollers, *, name, task, stop_event):
        registrations.append((app, tuple(owned_job_pollers), name, task, stop_event))

    monkeypatch.setattr(startup_infra, "_start_connectors_worker_service", _fake_start)

    handles = await startup_infra.start_connectors_startup(
        app="app",
        owned_job_pollers=["poller"],
        register_owned_job_poller=_fake_register,
    )

    assert handles.connectors_jobs_task == "connectors-task"
    assert handles.connectors_jobs_stop_event is not None
    assert registrations == [
        (
            "app",
            ("poller",),
            "connectors_jobs_task",
            "connectors-task",
            handles.connectors_jobs_stop_event,
        ),
    ]


@pytest.mark.asyncio
async def test_start_connectors_worker_cancels_task_when_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_infra = _import_startup_infra_services()

    class _FakeTask:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    task = _FakeTask()

    async def _fake_start(*, stop_event):
        assert stop_event is not None
        return task

    def _failing_register(*args, **kwargs):
        raise RuntimeError("registration boom")

    monkeypatch.setattr(startup_infra, "_start_connectors_worker_service", _fake_start)

    returned_task, stop_event = await startup_infra._start_connectors_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_failing_register,
    )

    assert returned_task is None
    assert stop_event is None
    assert task.cancelled is True
