from __future__ import annotations

import importlib
import sys
from collections.abc import Callable

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

    async def _record_study_pack(**kwargs: object) -> tuple[str, str]:
        """Record that the Study Pack worker starter ran."""

        del kwargs
        calls.append("study-pack")
        return ("study-pack-stop", "study-pack-task")

    async def _record_study_suggestions(**kwargs: object) -> tuple[str, str]:
        """Record that the Study Suggestions worker starter ran."""

        del kwargs
        calls.append("study-suggestions")
        return ("study-suggestions-stop", "study-suggestions-task")

    async def _record_privilege_snapshot(**kwargs: object) -> tuple[str, str]:
        """Record that the privilege snapshot worker starter ran."""

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
async def test_start_study_privilege_jobs_pollers_passes_inventory_to_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_study_privilege_jobs_pollers()
    worker_inventory = object()
    captured_kwargs_by_worker: dict[str, dict[str, object]] = {}

    def _record_worker(label: str) -> Callable[..., object]:
        """Build a starter stub that captures kwargs for one worker label."""

        async def _record(**kwargs: object) -> tuple[str, str]:
            """Capture worker startup kwargs and return deterministic handles."""

            captured_kwargs_by_worker[label] = kwargs
            return (f"{label}-stop", f"{label}-task")

        return _record

    monkeypatch.setattr(startup_pollers, "_start_study_pack_jobs_worker", _record_worker("study-pack"))
    monkeypatch.setattr(
        startup_pollers,
        "_start_study_suggestions_jobs_worker",
        _record_worker("study-suggestions"),
    )
    monkeypatch.setattr(startup_pollers, "_start_privilege_snapshot_worker", _record_worker("privilege"))

    await startup_pollers.start_study_privilege_jobs_pollers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        should_start_worker=lambda *args, **kwargs: False,
        worker_inventory=worker_inventory,
    )

    assert {
        worker: kwargs["worker_inventory"]
        for worker, kwargs in captured_kwargs_by_worker.items()
    } == {
        "study-pack": worker_inventory,
        "study-suggestions": worker_inventory,
        "privilege": worker_inventory,
    }


@pytest.mark.parametrize(
    (
        "starter_name",
        "flag_name",
        "route_name",
        "registered_name",
        "factory_name",
    ),
    [
        (
            "_start_study_pack_jobs_worker",
            "STUDY_PACK_JOBS_WORKER_ENABLED",
            "flashcards",
            "study_pack_jobs_task",
            "_run_study_pack_jobs_worker_service",
        ),
        (
            "_start_study_suggestions_jobs_worker",
            "STUDY_SUGGESTIONS_JOBS_WORKER_ENABLED",
            "study-suggestions",
            "study_suggestions_jobs_task",
            "_run_study_suggestions_jobs_worker_service",
        ),
        (
            "_start_privilege_snapshot_worker",
            "PRIVILEGE_SNAPSHOT_WORKER_ENABLED",
            "privileges",
            "privilege_snapshot_task",
            "_run_privilege_snapshot_worker_service",
        ),
    ],
)
@pytest.mark.asyncio
async def test_study_privilege_jobs_worker_registers_with_worker_inventory_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    starter_name: str,
    flag_name: str,
    route_name: str,
    registered_name: str,
    factory_name: str,
) -> None:
    startup_pollers = _import_startup_study_privilege_jobs_pollers()
    registrations: list[dict[str, object]] = []

    class _FakeWorkerInventory:
        """Test double that records custom worker registration calls."""

        async def register_custom(self, **kwargs: object) -> tuple[str, str]:
            """Capture registration kwargs and return deterministic handles."""

            registrations.append(kwargs)
            return f"{registered_name}-task", f"{registered_name}-stop"

    monkeypatch.setattr(
        startup_pollers,
        "_make_event",
        lambda: (_ for _ in ()).throw(AssertionError("legacy event path should not run")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: (_ for _ in ()).throw(AssertionError("legacy task path should not run")),
    )

    def _register_owned_job_poller(*args: object, **kwargs: object) -> None:
        """Fail when the registry path falls back to legacy registration."""

        raise AssertionError("legacy poller registration should not run")

    stop_event, task = await getattr(startup_pollers, starter_name)(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda flag, route, **kwargs: (flag, route, kwargs) == (
            flag_name,
            route_name,
            {},
        ),
        worker_inventory=_FakeWorkerInventory(),
    )

    assert stop_event == f"{registered_name}-stop"
    assert task == f"{registered_name}-task"
    assert registrations == [
        {
            "name": registered_name,
            "task_name": registered_name,
            "coroutine_factory": getattr(startup_pollers, factory_name),
            "timeout_sec": 5.0,
            "category": "jobs",
            "shutdown_phase": startup_pollers.ShutdownPhase.JOB_POLLER_QUIESCE,
        }
    ]


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
