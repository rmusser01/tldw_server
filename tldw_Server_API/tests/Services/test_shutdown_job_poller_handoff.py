from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_job_poller_handoff_uses_env_and_job_manager_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_job_poller_handoff as handoff_module

    app = FastAPI()
    recorded: dict[str, Any] = {}
    owned_job_pollers = [object()]

    class _FakeJobManager:
        def count_active_processing(self) -> int:
            return 7

    async def _fake_quiesce(
        current_app: FastAPI,
        poller_handles: list[object],
        *,
        wait_for_leases_sec: int,
        count_active_processing,
    ) -> None:
        recorded["app"] = current_app
        recorded["poller_handles"] = poller_handles
        recorded["wait_for_leases_sec"] = wait_for_leases_sec
        recorded["active_processing"] = count_active_processing()

    monkeypatch.setattr(handoff_module._env_os, "getenv", lambda *_args, **_kwargs: "12")
    monkeypatch.setattr(handoff_module, "_load_shutdown_job_manager", lambda: _FakeJobManager)

    handles = await handoff_module.shutdown_job_poller_handoff(
        app=app,
        owned_job_pollers=owned_job_pollers,
        quiesce_owned_job_pollers_for_shutdown=_fake_quiesce,
        startup_guard_exceptions=(ValueError,),
        import_exceptions=(ImportError,),
    )

    assert recorded["app"] is app
    assert recorded["poller_handles"] is owned_job_pollers
    assert recorded["wait_for_leases_sec"] == 12
    assert recorded["active_processing"] == 7
    assert handles.early_quiesced_job_poller_names == set()
    assert handles.should_run_late_stop("media_ingest_jobs_task", object()) is True
    assert handles.should_run_late_stop("media_ingest_jobs_task", None) is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_job_poller_handoff_falls_back_on_invalid_env_and_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_job_poller_handoff as handoff_module

    app = FastAPI()
    recorded: dict[str, Any] = {}
    owned_job_pollers = [object(), object()]

    async def _fake_quiesce(
        current_app: FastAPI,
        poller_handles: list[object],
        *,
        wait_for_leases_sec: int,
        count_active_processing,
    ) -> None:
        current_app.state._tldw_shutdown_quiesced_job_poller_names = [
            "files_jobs_task",
            "audio_jobs_task",
        ]
        recorded["app"] = current_app
        recorded["poller_handles"] = poller_handles
        recorded["wait_for_leases_sec"] = wait_for_leases_sec
        recorded["active_processing"] = count_active_processing()

    def _raise_import_error() -> type[object]:
        raise ImportError("jobs manager unavailable")

    monkeypatch.setattr(handoff_module._env_os, "getenv", lambda *_args, **_kwargs: "not-an-int")
    monkeypatch.setattr(handoff_module, "_load_shutdown_job_manager", _raise_import_error)

    handles = await handoff_module.shutdown_job_poller_handoff(
        app=app,
        owned_job_pollers=owned_job_pollers,
        quiesce_owned_job_pollers_for_shutdown=_fake_quiesce,
        startup_guard_exceptions=(ValueError,),
        import_exceptions=(ImportError,),
    )

    assert recorded["app"] is app
    assert recorded["poller_handles"] is owned_job_pollers
    assert recorded["wait_for_leases_sec"] == 0
    assert recorded["active_processing"] == 0
    assert handles.early_quiesced_job_poller_names == {
        "files_jobs_task",
        "audio_jobs_task",
    }
    assert handles.should_run_late_stop("files_jobs_task", object()) is False
    assert handles.should_run_late_stop("audio_jobs_task", object()) is False
    assert handles.should_run_late_stop("core_jobs_task", object()) is True
    assert handles.should_run_late_stop("core_jobs_task", None) is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_shutdown_job_poller_handoff_delegates_and_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_job_poller_handoff as handoff_module

    app = FastAPI()
    owned_job_pollers = [object()]
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
        owned_job_pollers=owned_job_pollers,
        quiesce_owned_job_pollers_for_shutdown=object(),
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert handles is expected_handles
    assert len(recorded_calls) == 1
    assert recorded_calls[0]["app"] is app
    assert recorded_calls[0]["owned_job_pollers"] is owned_job_pollers


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
        owned_job_pollers=[],
        quiesce_owned_job_pollers_for_shutdown=object(),
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert handles.early_quiesced_job_poller_names == set()
    assert handles.should_run_late_stop("core_jobs_task", object()) is False
    assert any("Job-poller shutdown handoff skipped" in message for message in debug_messages)
