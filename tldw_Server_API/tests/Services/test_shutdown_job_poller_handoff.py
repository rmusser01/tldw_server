from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI


def _make_worker(name: str, *, phase: Any | None = None, task: object | None = object()):
    from tldw_Server_API.app.services.lifecycle_workers import ManagedWorker, ShutdownPhase

    return ManagedWorker(
        name=name,
        task=task,
        stop_event=None,
        shutdown_callback=None,
        shutdown_phase=phase or ShutdownPhase.JOB_POLLER_QUIESCE,
    )


def _make_session(app: FastAPI, workers: list[Any]):
    from tldw_Server_API.app.services.lifecycle_worker_session import WorkerLifecycleSession
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        WorkerSpec,
        validate_worker_spec_graph,
    )

    specs = [
        WorkerSpec(
            name=worker.name,
            task_name=f"{worker.name}-task",
            category="jobs",
            phase=worker.shutdown_phase,
            factory=lambda _context, stop_event: stop_event.wait(),
        )
        for worker in workers
    ]
    graph = validate_worker_spec_graph(specs)
    session = WorkerLifecycleSession(app=app, graph=graph)
    for spec, worker in zip(specs, workers):
        session.register_handle(spec, worker)
    return session


@pytest.mark.unit
def test_filter_job_poller_quiesce_handles_accepts_string_phase() -> None:
    from tldw_Server_API.app.services import shutdown_job_poller_handoff as handoff_module

    poller = SimpleNamespace(task=object(), shutdown_phase="job_poller_quiesce")
    background = SimpleNamespace(task=object(), shutdown_phase="background_worker_shutdown")

    assert handoff_module._filter_job_poller_quiesce_handles([poller, background]) == [poller]


class _FakeLifecycleWorkerEngine:
    def __init__(self) -> None:
        self.stop_phase_calls: list[tuple[object, object]] = []

    async def stop_phase(self, session: Any, phase: Any) -> None:
        self.stop_phase_calls.append((session, phase))
        for handle in list(session.handles_for_phase(phase)):
            session.mark_stopped(handle.name, phase)
        session.publish_stopped_names(phase)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_job_poller_handoff_uses_session_engine_env_and_job_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_job_poller_handoff as handoff_module
    from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase

    app = FastAPI()
    poller = _make_worker("core_jobs_task")
    session = _make_session(app, [poller])
    engine = _FakeLifecycleWorkerEngine()
    recorded: dict[str, Any] = {}

    class _FakeJobManager:
        def count_active_processing(self) -> int:
            return 7

    async def _fake_quiesce(
        current_app: FastAPI,
        poller_handles: list[object],
        *,
        wait_for_leases_sec: int,
        count_active_processing,
        stop_registered_job_pollers,
    ) -> None:
        recorded["app"] = current_app
        recorded["poller_handles"] = poller_handles
        recorded["wait_for_leases_sec"] = wait_for_leases_sec
        recorded["active_processing"] = count_active_processing()
        await stop_registered_job_pollers(current_app, poller_handles)

    monkeypatch.setenv("JOBS_SHUTDOWN_WAIT_FOR_LEASES_SEC", "12")
    monkeypatch.setattr(handoff_module, "_load_shutdown_job_manager", lambda: _FakeJobManager)

    handles = await handoff_module.shutdown_job_poller_handoff(
        app=app,
        worker_lifecycle_session=session,
        lifecycle_worker_engine=engine,
        quiesce_owned_job_pollers_for_shutdown=_fake_quiesce,
        startup_guard_exceptions=(ValueError,),
        import_exceptions=(ImportError,),
    )

    assert recorded["app"] is app
    assert recorded["poller_handles"] == [poller]
    assert recorded["wait_for_leases_sec"] == 12
    assert recorded["active_processing"] == 7
    assert engine.stop_phase_calls == [(session, ShutdownPhase.JOB_POLLER_QUIESCE)]
    assert handles.early_quiesced_job_poller_names == {"core_jobs_task"}
    assert handles.should_run_late_stop("core_jobs_task", object()) is False
    assert handles.should_run_late_stop("media_ingest_jobs_task", object()) is True
    assert handles.should_run_late_stop("media_ingest_jobs_task", None) is False
    assert app.state._tldw_shutdown_quiesced_job_poller_names == ["core_jobs_task"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_job_poller_handoff_filters_to_tasked_job_poller_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_job_poller_handoff as handoff_module
    from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase

    app = FastAPI()
    poller = _make_worker("content_jobs_task")
    callback_only_poller = _make_worker("callback_only_poller", task=None)
    background_worker = _make_worker(
        "authnz_scheduler",
        phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
    )
    session = _make_session(app, [callback_only_poller, poller, background_worker])
    engine = _FakeLifecycleWorkerEngine()
    recorded: dict[str, Any] = {}

    async def _fake_quiesce(
        current_app: FastAPI,
        poller_handles: list[object],
        *,
        wait_for_leases_sec: int,
        count_active_processing,
        stop_registered_job_pollers,
    ) -> None:
        del wait_for_leases_sec, count_active_processing
        recorded["app"] = current_app
        recorded["poller_handles"] = poller_handles
        await stop_registered_job_pollers(current_app, poller_handles)

    monkeypatch.setenv("JOBS_SHUTDOWN_WAIT_FOR_LEASES_SEC", "0")

    handles = await handoff_module.shutdown_job_poller_handoff(
        app=app,
        worker_lifecycle_session=session,
        lifecycle_worker_engine=engine,
        quiesce_owned_job_pollers_for_shutdown=_fake_quiesce,
        startup_guard_exceptions=(ValueError,),
        import_exceptions=(ImportError,),
    )

    assert recorded["app"] is app
    assert recorded["poller_handles"] == [poller]
    assert handles.early_quiesced_job_poller_names == {
        "callback_only_poller",
        "content_jobs_task",
    }
    assert engine.stop_phase_calls == [(session, ShutdownPhase.JOB_POLLER_QUIESCE)]
    assert "authnz_scheduler" not in session.stopped_or_quiesced_names


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_job_poller_handoff_preserves_already_quiesced_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_job_poller_handoff as handoff_module
    from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase

    app = FastAPI()
    already_quiesced = _make_worker("files_jobs_task")
    active_poller = _make_worker("audio_jobs_task")
    session = _make_session(app, [already_quiesced, active_poller])
    session.mark_stopped("files_jobs_task", ShutdownPhase.JOB_POLLER_QUIESCE)
    engine = _FakeLifecycleWorkerEngine()
    recorded: dict[str, Any] = {}

    async def _fake_quiesce(
        current_app: FastAPI,
        poller_handles: list[object],
        *,
        wait_for_leases_sec: int,
        count_active_processing,
        stop_registered_job_pollers,
    ) -> None:
        recorded["poller_handles"] = poller_handles
        recorded["wait_for_leases_sec"] = wait_for_leases_sec
        recorded["active_processing"] = count_active_processing()
        await stop_registered_job_pollers(current_app, poller_handles)

    def _raise_import_error() -> type[object]:
        raise ImportError("jobs manager unavailable")

    monkeypatch.setenv("JOBS_SHUTDOWN_WAIT_FOR_LEASES_SEC", "not-an-int")
    monkeypatch.setattr(handoff_module, "_load_shutdown_job_manager", _raise_import_error)

    handles = await handoff_module.shutdown_job_poller_handoff(
        app=app,
        worker_lifecycle_session=session,
        lifecycle_worker_engine=engine,
        quiesce_owned_job_pollers_for_shutdown=_fake_quiesce,
        startup_guard_exceptions=(ValueError,),
        import_exceptions=(ImportError,),
    )

    assert recorded["poller_handles"] == [active_poller]
    assert recorded["wait_for_leases_sec"] == 0
    assert recorded["active_processing"] == 0
    assert handles.early_quiesced_job_poller_names == {
        "audio_jobs_task",
        "files_jobs_task",
    }
    assert handles.should_run_late_stop("files_jobs_task", object()) is False
    assert handles.should_run_late_stop("audio_jobs_task", object()) is False
    assert handles.should_run_late_stop("core_jobs_task", object()) is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_job_poller_handoff_returns_default_handles_without_session() -> None:
    from tldw_Server_API.app.services import shutdown_job_poller_handoff as handoff_module

    async def _raise_if_called(*_args, **_kwargs) -> None:
        raise AssertionError("quiesce should not run without a worker lifecycle session")

    handles = await handoff_module.shutdown_job_poller_handoff(
        app=FastAPI(),
        worker_lifecycle_session=None,
        lifecycle_worker_engine=_FakeLifecycleWorkerEngine(),
        quiesce_owned_job_pollers_for_shutdown=_raise_if_called,
        startup_guard_exceptions=(ValueError,),
        import_exceptions=(ImportError,),
    )

    assert handles.early_quiesced_job_poller_names == set()
    assert handles.should_run_late_stop("core_jobs_task", object()) is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_shutdown_job_poller_handoff_delegates_and_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_job_poller_handoff as handoff_module

    app = FastAPI()
    session = object()
    engine = object()
    recorded_calls: list[dict[str, Any]] = []
    expected_handles = handoff_module.JobPollerShutdownHandoffHandles(
        early_quiesced_job_poller_names={"core_jobs_task"},
        should_run_late_stop=lambda task_name, task: bool(task) and task_name != "core_jobs_task",
    )

    async def _fake_shutdown_job_poller_handoff(**kwargs):
        recorded_calls.append(kwargs)
        return expected_handles

    monkeypatch.setattr(
        handoff_module,
        "shutdown_job_poller_handoff",
        _fake_shutdown_job_poller_handoff,
    )

    handles = await handoff_module.run_shutdown_job_poller_handoff(
        app=app,
        worker_lifecycle_session=session,
        lifecycle_worker_engine=engine,
        quiesce_owned_job_pollers_for_shutdown=object(),
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert handles is expected_handles
    assert len(recorded_calls) == 1
    assert recorded_calls[0]["app"] is app
    assert recorded_calls[0]["worker_lifecycle_session"] is session
    assert recorded_calls[0]["lifecycle_worker_engine"] is engine


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_shutdown_job_poller_handoff_logs_and_returns_default_handles_on_guard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_job_poller_handoff as handoff_module

    debug_messages: list[str] = []

    async def _raise_guard_failure(**_kwargs):
        raise RuntimeError("handoff unavailable")

    monkeypatch.setattr(
        handoff_module,
        "shutdown_job_poller_handoff",
        _raise_guard_failure,
    )
    monkeypatch.setattr(
        handoff_module.logger,
        "debug",
        lambda message, *args, **kwargs: debug_messages.append(str(message)),
    )

    handles = await handoff_module.run_shutdown_job_poller_handoff(
        app=FastAPI(),
        worker_lifecycle_session=object(),
        lifecycle_worker_engine=object(),
        quiesce_owned_job_pollers_for_shutdown=object(),
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert handles.early_quiesced_job_poller_names == set()
    assert handles.should_run_late_stop("core_jobs_task", object()) is False
    assert any("Job-poller shutdown handoff skipped" in message for message in debug_messages)
