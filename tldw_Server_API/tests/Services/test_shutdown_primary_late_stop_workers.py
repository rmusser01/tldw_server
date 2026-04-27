from __future__ import annotations

from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_shutdown_primary_late_stop_workers_runs_helpers_in_order_and_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_primary_late_stop_workers as shutdown_workers

    calls: list[tuple[str, dict[str, object]]] = []
    should_run_late_stop = lambda *args, **kwargs: True
    guard_exceptions = (RuntimeError,)

    async def _record_core(**kwargs):
        calls.append(("core", kwargs))
        return SimpleNamespace(
            core_jobs_task=kwargs["core_jobs_task"],
            core_jobs_stop_event=kwargs["core_jobs_stop_event"],
        )

    async def _record_files(**kwargs):
        calls.append(("files", kwargs))
        return SimpleNamespace(
            files_jobs_task=kwargs["files_jobs_task"],
            files_jobs_stop_event=kwargs["files_jobs_stop_event"],
        )

    async def _record_data_tables(**kwargs):
        calls.append(("data_tables", kwargs))
        return SimpleNamespace(
            data_tables_jobs_task=kwargs["data_tables_jobs_task"],
            data_tables_jobs_stop_event=kwargs["data_tables_jobs_stop_event"],
        )

    async def _record_prompt_studio(**kwargs):
        calls.append(("prompt_studio", kwargs))
        return SimpleNamespace(
            prompt_studio_jobs_task=kwargs["prompt_studio_jobs_task"],
            prompt_studio_jobs_stop_event=kwargs["prompt_studio_jobs_stop_event"],
        )

    async def _record_privilege_snapshot(**kwargs):
        calls.append(("privilege_snapshot", kwargs))
        return SimpleNamespace(
            privilege_snapshot_task=kwargs["privilege_snapshot_task"],
            privilege_snapshot_stop_event=kwargs["privilege_snapshot_stop_event"],
        )

    async def _record_audio(**kwargs):
        calls.append(("audio", kwargs))
        return SimpleNamespace(
            audio_jobs_task=kwargs["audio_jobs_task"],
            audio_jobs_stop_event=kwargs["audio_jobs_stop_event"],
        )

    async def _record_presentation_render(**kwargs):
        calls.append(("presentation_render", kwargs))
        return SimpleNamespace(
            presentation_render_jobs_task=kwargs["presentation_render_jobs_task"],
            presentation_render_jobs_stop_event=kwargs["presentation_render_jobs_stop_event"],
        )

    monkeypatch.setattr(shutdown_workers, "_shutdown_core_jobs_worker", _record_core)
    monkeypatch.setattr(shutdown_workers, "_shutdown_files_jobs_worker", _record_files)
    monkeypatch.setattr(shutdown_workers, "_shutdown_data_tables_jobs_worker", _record_data_tables)
    monkeypatch.setattr(shutdown_workers, "_shutdown_prompt_studio_jobs_worker", _record_prompt_studio)
    monkeypatch.setattr(shutdown_workers, "_shutdown_privilege_snapshot_worker", _record_privilege_snapshot)
    monkeypatch.setattr(shutdown_workers, "_shutdown_audio_jobs_worker", _record_audio)
    monkeypatch.setattr(
        shutdown_workers,
        "_shutdown_presentation_render_jobs_worker",
        _record_presentation_render,
    )

    handles = await shutdown_workers.shutdown_primary_late_stop_workers(
        core_jobs_task="core-task",
        core_jobs_stop_event="core-stop",
        files_jobs_task="files-task",
        files_jobs_stop_event="files-stop",
        data_tables_jobs_task="data-task",
        data_tables_jobs_stop_event="data-stop",
        prompt_studio_jobs_task="prompt-task",
        prompt_studio_jobs_stop_event="prompt-stop",
        privilege_snapshot_task="priv-task",
        privilege_snapshot_stop_event="priv-stop",
        audio_jobs_task="audio-task",
        audio_jobs_stop_event="audio-stop",
        presentation_render_jobs_task="render-task",
        presentation_render_jobs_stop_event="render-stop",
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )

    assert [name for name, _ in calls] == [
        "core",
        "files",
        "data_tables",
        "prompt_studio",
        "privilege_snapshot",
        "audio",
        "presentation_render",
    ]
    assert all(call_kwargs["should_run_late_stop"] is should_run_late_stop for _, call_kwargs in calls)
    assert all(call_kwargs["guard_exceptions"] == guard_exceptions for _, call_kwargs in calls)
    assert handles.core_jobs_task == "core-task"
    assert handles.core_jobs_stop_event == "core-stop"
    assert handles.files_jobs_task == "files-task"
    assert handles.files_jobs_stop_event == "files-stop"
    assert handles.data_tables_jobs_task == "data-task"
    assert handles.data_tables_jobs_stop_event == "data-stop"
    assert handles.prompt_studio_jobs_task == "prompt-task"
    assert handles.prompt_studio_jobs_stop_event == "prompt-stop"
    assert handles.privilege_snapshot_task == "priv-task"
    assert handles.privilege_snapshot_stop_event == "priv-stop"
    assert handles.audio_jobs_task == "audio-task"
    assert handles.audio_jobs_stop_event == "audio-stop"
    assert handles.presentation_render_jobs_task == "render-task"
    assert handles.presentation_render_jobs_stop_event == "render-stop"


@pytest.mark.asyncio
async def test_run_shutdown_primary_late_stop_workers_delegates_and_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_primary_late_stop_workers as shutdown_workers

    should_run_late_stop = lambda *args, **kwargs: True
    recorded_calls: list[dict[str, object]] = []
    expected_handles = shutdown_workers.PrimaryLateStopWorkerHandles(
        core_jobs_task="core-task",
        core_jobs_stop_event="core-stop",
        files_jobs_task="files-task",
        files_jobs_stop_event="files-stop",
        data_tables_jobs_task="data-task",
        data_tables_jobs_stop_event="data-stop",
        prompt_studio_jobs_task="prompt-task",
        prompt_studio_jobs_stop_event="prompt-stop",
        privilege_snapshot_task="priv-task",
        privilege_snapshot_stop_event="priv-stop",
        audio_jobs_task="audio-task",
        audio_jobs_stop_event="audio-stop",
        presentation_render_jobs_task="render-task",
        presentation_render_jobs_stop_event="render-stop",
    )

    async def _fake_shutdown_primary_late_stop_workers(**kwargs):
        recorded_calls.append(kwargs)
        return expected_handles

    monkeypatch.setattr(
        shutdown_workers,
        "shutdown_primary_late_stop_workers",
        _fake_shutdown_primary_late_stop_workers,
    )

    handles = await shutdown_workers.run_shutdown_primary_late_stop_workers(
        core_jobs_task="core-task",
        core_jobs_stop_event="core-stop",
        files_jobs_task="files-task",
        files_jobs_stop_event="files-stop",
        data_tables_jobs_task="data-task",
        data_tables_jobs_stop_event="data-stop",
        prompt_studio_jobs_task="prompt-task",
        prompt_studio_jobs_stop_event="prompt-stop",
        privilege_snapshot_task="priv-task",
        privilege_snapshot_stop_event="priv-stop",
        audio_jobs_task="audio-task",
        audio_jobs_stop_event="audio-stop",
        presentation_render_jobs_task="render-task",
        presentation_render_jobs_stop_event="render-stop",
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=(RuntimeError,),
    )

    assert handles is expected_handles
    assert len(recorded_calls) == 1
    assert recorded_calls[0]["should_run_late_stop"] is should_run_late_stop
    assert recorded_calls[0]["core_jobs_task"] == "core-task"
    assert recorded_calls[0]["presentation_render_jobs_stop_event"] == "render-stop"


@pytest.mark.asyncio
async def test_run_shutdown_primary_late_stop_workers_logs_and_returns_original_handles_on_guard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_primary_late_stop_workers as shutdown_workers

    debug_messages: list[str] = []

    async def _raise_guard_failure(**_kwargs):
        raise RuntimeError("primary late-stop unavailable")

    monkeypatch.setattr(
        shutdown_workers,
        "shutdown_primary_late_stop_workers",
        _raise_guard_failure,
    )
    monkeypatch.setattr(
        shutdown_workers.logger,
        "debug",
        lambda message, *args, **kwargs: debug_messages.append(str(message)),
    )

    handles = await shutdown_workers.run_shutdown_primary_late_stop_workers(
        core_jobs_task="core-task",
        core_jobs_stop_event="core-stop",
        files_jobs_task="files-task",
        files_jobs_stop_event="files-stop",
        data_tables_jobs_task="data-task",
        data_tables_jobs_stop_event="data-stop",
        prompt_studio_jobs_task="prompt-task",
        prompt_studio_jobs_stop_event="prompt-stop",
        privilege_snapshot_task="priv-task",
        privilege_snapshot_stop_event="priv-stop",
        audio_jobs_task="audio-task",
        audio_jobs_stop_event="audio-stop",
        presentation_render_jobs_task="render-task",
        presentation_render_jobs_stop_event="render-stop",
        should_run_late_stop=lambda *args, **kwargs: True,
        guard_exceptions=(RuntimeError,),
    )

    assert handles.core_jobs_task == "core-task"
    assert handles.files_jobs_task == "files-task"
    assert handles.presentation_render_jobs_stop_event == "render-stop"
    assert any("Primary late-stop workers skipped" in message for message in debug_messages)
