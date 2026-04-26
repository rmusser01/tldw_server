from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_startup_content_jobs_pollers():
    sys.modules.pop("tldw_Server_API.app.services.startup_content_jobs_pollers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_content_jobs_pollers")


@pytest.mark.asyncio
async def test_start_content_jobs_pollers_combines_handles_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    calls: list[str] = []

    async def _record_audio(**kwargs):
        del kwargs
        calls.append("audio")
        return ("audio-stop", "audio-task")

    async def _record_audiobook(**kwargs):
        del kwargs
        calls.append("audiobook")
        return ("audiobook-stop", "audiobook-task")

    async def _record_presentation(**kwargs):
        del kwargs
        calls.append("presentation")
        return ("presentation-stop", "presentation-task")

    async def _record_media_ingest(**kwargs):
        del kwargs
        calls.append("media-ingest")
        return ("media-stop", "media-task", "media-heavy-stop", "media-heavy-task")

    async def _record_reading_digest(**kwargs):
        del kwargs
        calls.append("reading-digest")
        return ("reading-stop", "reading-task")

    async def _record_companion(**kwargs):
        del kwargs
        calls.append("companion")
        return ("companion-stop", "companion-task")

    monkeypatch.setattr(startup_pollers, "_start_audio_jobs_worker", _record_audio)
    monkeypatch.setattr(startup_pollers, "_start_audiobook_jobs_worker", _record_audiobook)
    monkeypatch.setattr(startup_pollers, "_start_presentation_render_jobs_worker", _record_presentation)
    monkeypatch.setattr(startup_pollers, "_start_media_ingest_jobs_workers", _record_media_ingest)
    monkeypatch.setattr(startup_pollers, "_start_reading_digest_jobs_worker", _record_reading_digest)
    monkeypatch.setattr(startup_pollers, "_start_companion_reflection_jobs_worker", _record_companion)

    handles = await startup_pollers.start_content_jobs_pollers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        should_start_worker=lambda *args, **kwargs: False,
    )

    assert calls == ["audio", "audiobook", "presentation", "media-ingest", "reading-digest", "companion"]
    assert handles.audio_jobs_stop_event == "audio-stop"
    assert handles.audio_jobs_task == "audio-task"
    assert handles.audiobook_jobs_stop_event == "audiobook-stop"
    assert handles.audiobook_jobs_task == "audiobook-task"
    assert handles.presentation_render_jobs_stop_event == "presentation-stop"
    assert handles.presentation_render_jobs_task == "presentation-task"
    assert handles.media_ingest_jobs_stop_event == "media-stop"
    assert handles.media_ingest_jobs_task == "media-task"
    assert handles.media_ingest_heavy_jobs_stop_event == "media-heavy-stop"
    assert handles.media_ingest_heavy_jobs_task == "media-heavy-task"
    assert handles.reading_digest_jobs_stop_event == "reading-stop"
    assert handles.reading_digest_jobs_task == "reading-task"
    assert handles.companion_reflection_jobs_stop_event == "companion-stop"
    assert handles.companion_reflection_jobs_task == "companion-task"


@pytest.mark.asyncio
async def test_start_audio_jobs_worker_registers_owned_poller_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    captured_stop_events: list[object] = []
    created_coroutines: list[object] = []
    registrations: list[dict[str, object]] = []

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "audio-stop")
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: created_coroutines.append(coro) or "audio-task",
    )
    monkeypatch.setattr(
        startup_pollers,
        "_run_audio_jobs_worker_service",
        lambda stop_event: captured_stop_events.append(stop_event) or "audio-coro",
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
    stop_event, task = await startup_pollers._start_audio_jobs_worker(
        app="app",
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda flag, route, **kwargs: (flag, route, kwargs) == (
            "AUDIO_JOBS_WORKER_ENABLED",
            "audio-jobs",
            {},
        ),
    )

    assert stop_event == "audio-stop"
    assert task == "audio-task"
    assert captured_stop_events == ["audio-stop"]
    assert created_coroutines == ["audio-coro"]
    assert registrations == [
        {
            "app": "app",
            "owned_job_pollers": owned_job_pollers,
            "name": "audio_jobs_task",
            "task": "audio-task",
            "stop_event": "audio-stop",
        }
    ]


@pytest.mark.asyncio
async def test_start_media_ingest_jobs_workers_respects_heavy_default_stable_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    created_coroutines: list[object] = []
    registrations: list[dict[str, object]] = []
    calls: list[tuple[str, str, dict[str, object]]] = []
    stop_events = iter(["media-stop", "media-heavy-stop"])

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: next(stop_events))
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: created_coroutines.append(coro) or f"task-{len(created_coroutines)}",
    )
    monkeypatch.setattr(
        startup_pollers,
        "_run_media_ingest_jobs_worker_service",
        lambda stop_event: f"media-coro-{stop_event}",
    )
    monkeypatch.setattr(
        startup_pollers,
        "_run_media_ingest_heavy_jobs_worker_service",
        lambda stop_event: f"media-heavy-coro-{stop_event}",
    )

    def _register_owned_job_poller(app, owned_job_pollers, *, name, task, stop_event):
        del app, owned_job_pollers
        registrations.append({"name": name, "task": task, "stop_event": stop_event})

    def _should_start_worker(flag, route, **kwargs):
        calls.append((flag, route, kwargs))
        return True

    handles = await startup_pollers._start_media_ingest_jobs_workers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=_should_start_worker,
    )

    assert handles == ("media-stop", "task-1", "media-heavy-stop", "task-2")
    assert created_coroutines == ["media-coro-media-stop", "media-heavy-coro-media-heavy-stop"]
    assert registrations == [
        {"name": "media_ingest_jobs_task", "task": "task-1", "stop_event": "media-stop"},
        {
            "name": "media_ingest_heavy_jobs_task",
            "task": "task-2",
            "stop_event": "media-heavy-stop",
        },
    ]
    assert calls == [
        ("MEDIA_INGEST_JOBS_WORKER_ENABLED", "media", {}),
        (
            "MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED",
            "media-ingest-heavy-jobs",
            {"default_stable": False},
        ),
    ]


@pytest.mark.asyncio
async def test_start_media_ingest_jobs_workers_preserves_light_handles_when_heavy_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    stop_events = iter(["media-stop", "media-heavy-stop"])
    registrations: list[dict[str, object]] = []

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: next(stop_events))
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: "media-task" if coro == "media-coro" else (_ for _ in ()).throw(RuntimeError("heavy boom")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_run_media_ingest_jobs_worker_service",
        lambda stop_event: "media-coro",
    )
    monkeypatch.setattr(
        startup_pollers,
        "_run_media_ingest_heavy_jobs_worker_service",
        lambda stop_event: "media-heavy-coro",
    )

    def _register_owned_job_poller(app, owned_job_pollers, *, name, task, stop_event):
        del app, owned_job_pollers
        registrations.append({"name": name, "task": task, "stop_event": stop_event})

    handles = await startup_pollers._start_media_ingest_jobs_workers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda *args, **kwargs: True,
    )

    assert handles == ("media-stop", "media-task", None, None)
    assert registrations == [
        {"name": "media_ingest_jobs_task", "task": "media-task", "stop_event": "media-stop"}
    ]


@pytest.mark.asyncio
async def test_start_companion_reflection_jobs_worker_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "companion-stop")

    def _failing_create_task(coro):
        del coro
        raise RuntimeError("boom")

    monkeypatch.setattr(startup_pollers, "_create_task", _failing_create_task)
    monkeypatch.setattr(
        startup_pollers,
        "_run_companion_reflection_jobs_worker_service",
        lambda stop_event: stop_event,
    )

    stop_event, task = await startup_pollers._start_companion_reflection_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        should_start_worker=lambda flag, route, **kwargs: (flag, route, kwargs) == (
            "COMPANION_REFLECTION_JOBS_WORKER_ENABLED",
            "companion",
            {},
        ),
    )

    assert stop_event is None
    assert task is None
