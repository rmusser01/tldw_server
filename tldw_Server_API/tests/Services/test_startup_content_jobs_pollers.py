from __future__ import annotations

import importlib
import sys
from collections.abc import Callable

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

    async def _record_llamacpp_acquisition(**kwargs):
        del kwargs
        calls.append("llamacpp-acquisition")
        return ("llamacpp-stop", "llamacpp-task")

    async def _record_vn_asset(**kwargs: object) -> tuple[str, str, str, str]:
        """Record that the VN asset worker starter ran."""

        del kwargs
        calls.append("vn-asset")
        return ("vn-asset-stop", "vn-asset-task", "vn-generation-stop", "vn-generation-task")

    async def _record_companion(**kwargs):
        del kwargs
        calls.append("companion")
        return ("companion-stop", "companion-task")

    monkeypatch.setattr(startup_pollers, "_start_audio_jobs_worker", _record_audio)
    monkeypatch.setattr(startup_pollers, "_start_audiobook_jobs_worker", _record_audiobook)
    monkeypatch.setattr(startup_pollers, "_start_presentation_render_jobs_worker", _record_presentation)
    monkeypatch.setattr(startup_pollers, "_start_media_ingest_jobs_workers", _record_media_ingest)
    monkeypatch.setattr(startup_pollers, "_start_reading_digest_jobs_worker", _record_reading_digest)
    monkeypatch.setattr(
        startup_pollers,
        "_start_llamacpp_acquisition_jobs_worker",
        _record_llamacpp_acquisition,
    )
    monkeypatch.setattr(startup_pollers, "_start_vn_asset_jobs_workers", _record_vn_asset)
    monkeypatch.setattr(startup_pollers, "_start_companion_reflection_jobs_worker", _record_companion)

    handles = await startup_pollers.start_content_jobs_pollers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        should_start_worker=lambda *args, **kwargs: False,
    )

    assert calls == [
        "audio",
        "audiobook",
        "presentation",
        "media-ingest",
        "reading-digest",
        "llamacpp-acquisition",
        "vn-asset",
        "companion",
    ]
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
    assert handles.llamacpp_acquisition_jobs_stop_event == "llamacpp-stop"
    assert handles.llamacpp_acquisition_jobs_task == "llamacpp-task"
    assert handles.vn_asset_jobs_stop_event == "vn-asset-stop"
    assert handles.vn_asset_jobs_task == "vn-asset-task"
    assert handles.vn_asset_generation_jobs_stop_event == "vn-generation-stop"
    assert handles.vn_asset_generation_jobs_task == "vn-generation-task"
    assert handles.companion_reflection_jobs_stop_event == "companion-stop"
    assert handles.companion_reflection_jobs_task == "companion-task"


@pytest.mark.asyncio
async def test_start_content_jobs_pollers_passes_inventory_to_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    worker_inventory = object()
    captured_kwargs_by_worker: dict[str, dict[str, object]] = {}

    def _record_worker(label: str, handles: tuple[object, ...]) -> Callable[..., object]:
        """Build a starter stub that captures kwargs for one content worker group."""

        async def _record(**kwargs: object) -> tuple[object, ...]:
            """Capture worker startup kwargs and return deterministic handles."""

            captured_kwargs_by_worker[label] = kwargs
            return handles

        return _record

    monkeypatch.setattr(
        startup_pollers,
        "_start_audio_jobs_worker",
        _record_worker("audio", ("audio-stop", "audio-task")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_start_audiobook_jobs_worker",
        _record_worker("audiobook", ("audiobook-stop", "audiobook-task")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_start_presentation_render_jobs_worker",
        _record_worker("presentation", ("presentation-stop", "presentation-task")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_start_media_ingest_jobs_workers",
        _record_worker("media-ingest", ("media-stop", "media-task", "media-heavy-stop", "media-heavy-task")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_start_reading_digest_jobs_worker",
        _record_worker("reading-digest", ("reading-stop", "reading-task")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_start_llamacpp_acquisition_jobs_worker",
        _record_worker("llamacpp-acquisition", ("llamacpp-stop", "llamacpp-task")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_start_vn_asset_jobs_workers",
        _record_worker("vn-asset", ("vn-asset-stop", "vn-asset-task", "vn-generation-stop", "vn-generation-task")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_start_companion_reflection_jobs_worker",
        _record_worker("companion", ("companion-stop", "companion-task")),
    )

    await startup_pollers.start_content_jobs_pollers(
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
        "audio": worker_inventory,
        "audiobook": worker_inventory,
        "presentation": worker_inventory,
        "media-ingest": worker_inventory,
        "reading-digest": worker_inventory,
        "llamacpp-acquisition": worker_inventory,
        "vn-asset": worker_inventory,
        "companion": worker_inventory,
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
            "_start_audio_jobs_worker",
            "AUDIO_JOBS_WORKER_ENABLED",
            "audio-jobs",
            "audio_jobs_task",
            "_run_audio_jobs_worker_service",
        ),
        (
            "_start_audiobook_jobs_worker",
            "AUDIOBOOK_JOBS_WORKER_ENABLED",
            "audiobooks",
            "audiobook_jobs_task",
            "_run_audiobook_jobs_worker_service",
        ),
        (
            "_start_presentation_render_jobs_worker",
            "PRESENTATION_RENDER_JOBS_WORKER_ENABLED",
            "slides",
            "presentation_render_jobs_task",
            "_run_presentation_render_jobs_worker_service",
        ),
        (
            "_start_reading_digest_jobs_worker",
            "READING_DIGEST_JOBS_WORKER_ENABLED",
            "reading",
            "reading_digest_jobs_task",
            "_run_reading_digest_jobs_worker_service",
        ),
        (
            "_start_llamacpp_acquisition_jobs_worker",
            "LLAMACPP_ACQUISITION_JOBS_WORKER_ENABLED",
            "llamacpp-acquisition",
            "llamacpp_acquisition_jobs_task",
            "_run_llamacpp_acquisition_jobs_worker_service",
        ),
        (
            "_start_companion_reflection_jobs_worker",
            "COMPANION_REFLECTION_JOBS_WORKER_ENABLED",
            "companion",
            "companion_reflection_jobs_task",
            "_run_companion_reflection_jobs_worker_service",
        ),
    ],
)
@pytest.mark.asyncio
async def test_content_jobs_worker_registers_with_worker_inventory_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    starter_name: str,
    flag_name: str,
    route_name: str,
    registered_name: str,
    factory_name: str,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
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
        raise AssertionError("legacy poller registration should not run")

    stop_event, task = await getattr(startup_pollers, starter_name)(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda flag, route, **kwargs: (flag, route, kwargs) == (flag_name, route_name, {}),
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
async def test_media_ingest_jobs_workers_register_with_worker_inventory_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    registrations: list[dict[str, object]] = []

    class _FakeWorkerInventory:
        """Test double that records custom worker registration calls."""

        async def register_custom(self, **kwargs: object) -> tuple[str, str]:
            """Capture registration kwargs and return deterministic handles."""

            registrations.append(kwargs)
            name = str(kwargs["name"])
            return f"{name}-task", f"{name}-stop"

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
        raise AssertionError("legacy poller registration should not run")

    handles = await startup_pollers._start_media_ingest_jobs_workers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda *args, **kwargs: True,
        worker_inventory=_FakeWorkerInventory(),
    )

    assert handles == (
        "media_ingest_jobs_task-stop",
        "media_ingest_jobs_task-task",
        "media_ingest_heavy_jobs_task-stop",
        "media_ingest_heavy_jobs_task-task",
    )
    assert registrations == [
        {
            "name": "media_ingest_jobs_task",
            "task_name": "media_ingest_jobs_task",
            "coroutine_factory": startup_pollers._run_media_ingest_jobs_worker_service,
            "timeout_sec": 5.0,
            "category": "jobs",
            "shutdown_phase": startup_pollers.ShutdownPhase.JOB_POLLER_QUIESCE,
        },
        {
            "name": "media_ingest_heavy_jobs_task",
            "task_name": "media_ingest_heavy_jobs_task",
            "coroutine_factory": startup_pollers._run_media_ingest_heavy_jobs_worker_service,
            "timeout_sec": 5.0,
            "category": "jobs",
            "shutdown_phase": startup_pollers.ShutdownPhase.JOB_POLLER_QUIESCE,
        },
    ]


@pytest.mark.asyncio
async def test_vn_asset_jobs_workers_register_with_worker_inventory_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    registrations: list[dict[str, object]] = []

    class _FakeWorkerInventory:
        """Test double that records custom worker registration calls."""

        async def register_custom(self, **kwargs: object) -> tuple[str, str]:
            """Capture registration kwargs and return deterministic handles."""

            registrations.append(kwargs)
            name = str(kwargs["name"])
            return f"{name}-task", f"{name}-stop"

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
        raise AssertionError("legacy poller registration should not run")

    handles = await startup_pollers._start_vn_asset_jobs_workers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda *args, **kwargs: True,
        worker_inventory=_FakeWorkerInventory(),
    )

    assert handles == (
        "vn_asset_jobs_task-stop",
        "vn_asset_jobs_task-task",
        "vn_asset_generation_jobs_task-stop",
        "vn_asset_generation_jobs_task-task",
    )
    assert registrations == [
        {
            "name": "vn_asset_jobs_task",
            "task_name": "vn_asset_jobs_task",
            "coroutine_factory": startup_pollers._run_vn_asset_jobs_worker_service,
            "timeout_sec": 5.0,
            "category": "jobs",
            "shutdown_phase": startup_pollers.ShutdownPhase.JOB_POLLER_QUIESCE,
        },
        {
            "name": "vn_asset_generation_jobs_task",
            "task_name": "vn_asset_generation_jobs_task",
            "coroutine_factory": startup_pollers._run_vn_asset_generation_jobs_worker_service,
            "timeout_sec": 5.0,
            "category": "jobs",
            "shutdown_phase": startup_pollers.ShutdownPhase.JOB_POLLER_QUIESCE,
        },
    ]


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
async def test_start_llamacpp_acquisition_jobs_worker_registers_owned_poller_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    captured_stop_events: list[object] = []
    created_coroutines: list[object] = []
    registrations: list[dict[str, object]] = []

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "llamacpp-stop")
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: created_coroutines.append(coro) or "llamacpp-task",
    )
    monkeypatch.setattr(
        startup_pollers,
        "_run_llamacpp_acquisition_jobs_worker_service",
        lambda stop_event: captured_stop_events.append(stop_event) or "llamacpp-coro",
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
    stop_event, task = await startup_pollers._start_llamacpp_acquisition_jobs_worker(
        app="app",
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda flag, route, **kwargs: (flag, route, kwargs) == (
            "LLAMACPP_ACQUISITION_JOBS_WORKER_ENABLED",
            "llamacpp-acquisition",
            {},
        ),
    )

    assert stop_event == "llamacpp-stop"
    assert task == "llamacpp-task"
    assert captured_stop_events == ["llamacpp-stop"]
    assert created_coroutines == ["llamacpp-coro"]
    assert registrations == [
        {
            "app": "app",
            "owned_job_pollers": owned_job_pollers,
            "name": "llamacpp_acquisition_jobs_task",
            "task": "llamacpp-task",
            "stop_event": "llamacpp-stop",
        }
    ]


@pytest.mark.asyncio
async def test_start_llamacpp_acquisition_jobs_worker_cancels_task_when_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    created_coroutines: list[object] = []

    class _FakeTask:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    task = _FakeTask()
    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "llamacpp-stop")
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: created_coroutines.append(coro) or task,
    )
    monkeypatch.setattr(
        startup_pollers,
        "_run_llamacpp_acquisition_jobs_worker_service",
        lambda stop_event: f"llamacpp-coro:{stop_event}",
    )

    def _register_owned_job_poller(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("registration failed")

    stop_event, returned_task = await startup_pollers._start_llamacpp_acquisition_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda flag, route, **kwargs: (flag, route, kwargs) == (
            "LLAMACPP_ACQUISITION_JOBS_WORKER_ENABLED",
            "llamacpp-acquisition",
            {},
        ),
    )

    assert stop_event is None
    assert returned_task is None
    assert created_coroutines == ["llamacpp-coro:llamacpp-stop"]
    assert task.cancelled is True


@pytest.mark.asyncio
async def test_start_vn_asset_jobs_workers_use_stable_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    stop_events = iter(["vn-stop", "vn-generation-stop"])
    calls: list[tuple[str, str, dict[str, object]]] = []

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: next(stop_events))
    monkeypatch.setattr(startup_pollers, "_create_task", lambda coro: f"task:{coro}")
    monkeypatch.setattr(
        startup_pollers,
        "_run_vn_asset_jobs_worker_service",
        lambda stop_event: f"vn-coro:{stop_event}",
    )
    monkeypatch.setattr(
        startup_pollers,
        "_run_vn_asset_generation_jobs_worker_service",
        lambda stop_event: f"vn-generation-coro:{stop_event}",
    )

    def _should_start_worker(flag_key: str, route_key: str, **kwargs: object) -> bool:
        calls.append((flag_key, route_key, kwargs))
        return False

    handles = await startup_pollers._start_vn_asset_jobs_workers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        should_start_worker=_should_start_worker,
    )

    assert handles == (None, None, None, None)
    assert calls == [
        ("VN_ASSET_JOBS_WORKER_ENABLED", "vn-assets", {"default_stable": True}),
        (
            "VN_ASSET_GENERATION_JOBS_WORKER_ENABLED",
            "vn-assets-generation",
            {"default_stable": True},
        ),
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
