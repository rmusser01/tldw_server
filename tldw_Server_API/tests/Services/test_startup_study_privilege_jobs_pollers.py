from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_startup_study_privilege_jobs_pollers():
    sys.modules.pop("tldw_Server_API.app.services.startup_study_privilege_jobs_pollers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_study_privilege_jobs_pollers")


@pytest.mark.asyncio
async def test_start_study_privilege_jobs_pollers_combines_handles_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_study_privilege_jobs_pollers()
    calls: list[str] = []

    async def _record_study_pack(**kwargs):
        del kwargs
        calls.append("study-pack")
        return ("study-pack-stop", "study-pack-task")

    async def _record_study_suggestions(**kwargs):
        del kwargs
        calls.append("study-suggestions")
        return ("study-suggestions-stop", "study-suggestions-task")

    async def _record_privilege_snapshot(**kwargs):
        del kwargs
        calls.append("privilege-snapshot")
        return ("privilege-stop", "privilege-task")

    monkeypatch.setattr(startup_pollers, "_start_study_pack_jobs_worker", _record_study_pack)
    monkeypatch.setattr(startup_pollers, "_start_study_suggestions_jobs_worker", _record_study_suggestions)
    monkeypatch.setattr(startup_pollers, "_start_privilege_snapshot_worker", _record_privilege_snapshot)

    handles = await startup_pollers.start_study_privilege_jobs_pollers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        should_start_worker=lambda *args, **kwargs: False,
    )

    assert calls == ["study-pack", "study-suggestions", "privilege-snapshot"]
    assert handles.study_pack_jobs_stop_event == "study-pack-stop"
    assert handles.study_pack_jobs_task == "study-pack-task"
    assert handles.study_suggestions_jobs_stop_event == "study-suggestions-stop"
    assert handles.study_suggestions_jobs_task == "study-suggestions-task"
    assert handles.privilege_snapshot_stop_event == "privilege-stop"
    assert handles.privilege_snapshot_task == "privilege-task"


@pytest.mark.asyncio
async def test_start_study_pack_jobs_worker_registers_owned_poller_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_study_privilege_jobs_pollers()
    captured_stop_events: list[object] = []
    created_coroutines: list[object] = []
    registrations: list[dict[str, object]] = []

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "study-pack-stop")
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: created_coroutines.append(coro) or "study-pack-task",
    )
    monkeypatch.setattr(
        startup_pollers,
        "_run_study_pack_jobs_worker_service",
        lambda stop_event: captured_stop_events.append(stop_event) or "study-pack-coro",
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
    stop_event, task = await startup_pollers._start_study_pack_jobs_worker(
        app="app",
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda flag, route, **kwargs: (flag, route, kwargs) == (
            "STUDY_PACK_JOBS_WORKER_ENABLED",
            "flashcards",
            {},
        ),
    )

    assert stop_event == "study-pack-stop"
    assert task == "study-pack-task"
    assert captured_stop_events == ["study-pack-stop"]
    assert created_coroutines == ["study-pack-coro"]
    assert registrations == [
        {
            "app": "app",
            "owned_job_pollers": owned_job_pollers,
            "name": "study_pack_jobs_task",
            "task": "study-pack-task",
            "stop_event": "study-pack-stop",
        }
    ]


@pytest.mark.asyncio
async def test_start_study_suggestions_jobs_worker_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_study_privilege_jobs_pollers()

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "study-suggestions-stop")

    def _failing_create_task(coro):
        del coro
        raise RuntimeError("boom")

    monkeypatch.setattr(startup_pollers, "_create_task", _failing_create_task)
    monkeypatch.setattr(
        startup_pollers,
        "_run_study_suggestions_jobs_worker_service",
        lambda stop_event: stop_event,
    )

    stop_event, task = await startup_pollers._start_study_suggestions_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        should_start_worker=lambda flag, route, **kwargs: (flag, route, kwargs) == (
            "STUDY_SUGGESTIONS_JOBS_WORKER_ENABLED",
            "study-suggestions",
            {},
        ),
    )

    assert stop_event is None
    assert task is None


@pytest.mark.asyncio
async def test_start_privilege_snapshot_worker_registers_owned_poller_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_study_privilege_jobs_pollers()
    captured_stop_events: list[object] = []
    created_coroutines: list[object] = []
    registrations: list[dict[str, object]] = []

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "privilege-stop")
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: created_coroutines.append(coro) or "privilege-task",
    )
    monkeypatch.setattr(
        startup_pollers,
        "_run_privilege_snapshot_worker_service",
        lambda stop_event: captured_stop_events.append(stop_event) or "privilege-coro",
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
    stop_event, task = await startup_pollers._start_privilege_snapshot_worker(
        app="app",
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda flag, route, **kwargs: (flag, route, kwargs) == (
            "PRIVILEGE_SNAPSHOT_WORKER_ENABLED",
            "privileges",
            {},
        ),
    )

    assert stop_event == "privilege-stop"
    assert task == "privilege-task"
    assert captured_stop_events == ["privilege-stop"]
    assert created_coroutines == ["privilege-coro"]
    assert registrations == [
        {
            "app": "app",
            "owned_job_pollers": owned_job_pollers,
            "name": "privilege_snapshot_task",
            "task": "privilege-task",
            "stop_event": "privilege-stop",
        }
    ]
