from __future__ import annotations

import importlib
import inspect
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def _import_startup_infra_services():
    sys.modules.pop("tldw_Server_API.app.services.startup_infra_services", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_infra_services")


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
