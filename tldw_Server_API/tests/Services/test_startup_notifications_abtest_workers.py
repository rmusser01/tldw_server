from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_startup_notifications_abtest_workers():
    sys.modules.pop("tldw_Server_API.app.services.startup_notifications_abtest_workers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_notifications_abtest_workers")


@pytest.mark.asyncio
async def test_start_notifications_abtest_workers_combines_handles_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_notifications_abtest_workers()
    calls: list[str] = []

    async def _record_notifications(**kwargs):
        del kwargs
        calls.append("notifications")
        return "bridge-task"

    async def _record_abtest(**kwargs):
        del kwargs
        calls.append("abtest")
        return ("abtest-stop", "abtest-task")

    monkeypatch.setattr(startup_workers, "_start_jobs_notifications_bridge_worker", _record_notifications)
    monkeypatch.setattr(startup_workers, "_start_evals_abtest_jobs_worker", _record_abtest)

    handles = await startup_workers.start_notifications_abtest_workers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        sidecar_mode=False,
    )

    assert calls == ["notifications", "abtest"]
    assert handles.jobs_notifications_bridge_task == "bridge-task"
    assert handles.evals_abtest_jobs_stop_event == "abtest-stop"
    assert handles.evals_abtest_jobs_task == "abtest-task"


@pytest.mark.asyncio
async def test_start_jobs_notifications_bridge_worker_skips_in_sidecar_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_notifications_abtest_workers()
    monkeypatch.setattr(
        startup_workers,
        "_start_jobs_notifications_service",
        lambda: (_ for _ in ()).throw(AssertionError("no start")),
    )

    task = await startup_workers._start_jobs_notifications_bridge_worker(sidecar_mode=True)

    assert task is None


@pytest.mark.asyncio
async def test_start_jobs_notifications_bridge_worker_returns_task_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_notifications_abtest_workers()
    monkeypatch.setattr(
        startup_workers,
        "_start_jobs_notifications_service",
        lambda: "bridge-task",
    )

    task = await startup_workers._start_jobs_notifications_bridge_worker(sidecar_mode=False)

    assert task == "bridge-task"


@pytest.mark.asyncio
async def test_start_evals_abtest_jobs_worker_registers_owned_poller_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_notifications_abtest_workers()
    captured_stop_events: list[object] = []
    created_coroutines: list[object] = []
    registrations: list[dict[str, object]] = []
    env_reads = iter(["true"])

    monkeypatch.setattr(startup_workers, "_make_event", lambda: "abtest-stop")
    monkeypatch.setattr(
        startup_workers,
        "_create_task",
        lambda coro: created_coroutines.append(coro) or "abtest-task",
    )
    monkeypatch.setattr(
        startup_workers,
        "_run_embeddings_abtest_jobs_worker_service",
        lambda stop_event: captured_stop_events.append(stop_event) or "abtest-coro",
    )
    monkeypatch.setattr(
        startup_workers.os,
        "getenv",
        lambda key, default=None: next(env_reads) if key == "EVALUATIONS_ABTEST_JOBS_WORKER_ENABLED" else default,
    )

    def _register_owned_job_poller(app, owned_job_pollers, *, name, task, stop_event):
        registrations.append(
            {
                "app": app,
                "owned_job_pollers": owned_job_pollers,
                "name": name,
                "task": task,
                "stop_event": stop_event,
            }
        )

    owned_job_pollers: list[object] = []
    stop_event, task = await startup_workers._start_evals_abtest_jobs_worker(
        app="app",
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=_register_owned_job_poller,
        sidecar_mode=False,
    )

    assert stop_event == "abtest-stop"
    assert task == "abtest-task"
    assert captured_stop_events == ["abtest-stop"]
    assert created_coroutines == ["abtest-coro"]
    assert registrations == [
        {
            "app": "app",
            "owned_job_pollers": owned_job_pollers,
            "name": "evals_abtest_jobs_task",
            "task": "abtest-task",
            "stop_event": "abtest-stop",
        }
    ]


@pytest.mark.asyncio
async def test_start_evals_abtest_jobs_worker_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_notifications_abtest_workers()

    monkeypatch.setattr(
        startup_workers.os,
        "getenv",
        lambda key, default=None: "true" if key == "EVALUATIONS_ABTEST_JOBS_WORKER_ENABLED" else default,
    )
    monkeypatch.setattr(startup_workers, "_make_event", lambda: "abtest-stop")

    def _failing_create_task(coro):
        del coro
        raise RuntimeError("boom")

    monkeypatch.setattr(startup_workers, "_create_task", _failing_create_task)
    monkeypatch.setattr(
        startup_workers,
        "_run_embeddings_abtest_jobs_worker_service",
        lambda stop_event: stop_event,
    )

    stop_event, task = await startup_workers._start_evals_abtest_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        sidecar_mode=False,
    )

    assert stop_event is None
    assert task is None


@pytest.mark.asyncio
async def test_start_evals_abtest_jobs_worker_cancels_task_when_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_notifications_abtest_workers()

    class _FakeTask:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    task = _FakeTask()

    monkeypatch.setattr(
        startup_workers.os,
        "getenv",
        lambda key, default=None: "true" if key == "EVALUATIONS_ABTEST_JOBS_WORKER_ENABLED" else default,
    )
    monkeypatch.setattr(startup_workers, "_make_event", lambda: "abtest-stop")
    monkeypatch.setattr(startup_workers, "_create_task", lambda coro: task)
    monkeypatch.setattr(
        startup_workers,
        "_run_embeddings_abtest_jobs_worker_service",
        lambda stop_event: stop_event,
    )

    def _failing_register(*args, **kwargs):
        raise RuntimeError("registration boom")

    stop_event, returned_task = await startup_workers._start_evals_abtest_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_failing_register,
        sidecar_mode=False,
    )

    assert stop_event is None
    assert returned_task is None
    assert task.cancelled is True
