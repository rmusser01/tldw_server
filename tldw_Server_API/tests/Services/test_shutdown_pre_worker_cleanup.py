from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_pre_worker_cleanup() -> Any:
    sys.modules.pop("tldw_Server_API.app.services.shutdown_pre_worker_cleanup", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_pre_worker_cleanup")


class _FakeTask:
    def __init__(self, *, exc: BaseException | None = None) -> None:
        self.cancelled = False
        self._exc = exc

    def cancel(self) -> None:
        if self._exc is not None:
            raise self._exc
        self.cancelled = True


class _FakeStopEvent:
    def __init__(self, *, exc: BaseException | None = None) -> None:
        self.is_set = False
        self._exc = exc

    def set(self) -> None:
        if self._exc is not None:
            raise self._exc
        self.is_set = True


class _FakeStorageCleanupService:
    def __init__(self, *, exc: BaseException | None = None) -> None:
        self.stopped = False
        self._exc = exc

    async def stop(self) -> None:
        if self._exc is not None:
            raise self._exc
        self.stopped = True


@pytest.mark.asyncio
async def test_shutdown_pre_worker_cleanup_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_cleanup = _import_shutdown_pre_worker_cleanup()
    calls: list[dict[str, object]] = []

    async def _record_shutdown(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(shutdown_cleanup, "_shutdown_pre_worker_cleanup", _record_shutdown)

    handles = await shutdown_cleanup.shutdown_pre_worker_cleanup(
        app="app",
        cleanup_task="cleanup-task",
        chatbooks_cleanup_task="chatbooks-task",
        chatbooks_cleanup_stop_event="chatbooks-stop",
        storage_cleanup_service="storage-service",
        coordinated_legacy_component_names={"chatbooks_cleanup"},
        guard_exceptions=(RuntimeError,),
    )

    assert len(calls) == 1
    assert calls[0]["app"] == "app"
    assert calls[0]["cleanup_task"] == "cleanup-task"
    assert calls[0]["chatbooks_cleanup_task"] == "chatbooks-task"
    assert calls[0]["chatbooks_cleanup_stop_event"] == "chatbooks-stop"
    assert calls[0]["storage_cleanup_service"] == "storage-service"
    assert calls[0]["coordinated_legacy_component_names"] == {"chatbooks_cleanup"}
    assert calls[0]["guard_exceptions"] == (RuntimeError,)
    assert handles.cleanup_task == "cleanup-task"
    assert handles.chatbooks_cleanup_task == "chatbooks-task"
    assert handles.chatbooks_cleanup_stop_event == "chatbooks-stop"
    assert handles.storage_cleanup_service == "storage-service"


@pytest.mark.asyncio
async def test_shutdown_pre_worker_cleanup_cancels_deferred_and_cleanup_tasks_and_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_cleanup = _import_shutdown_pre_worker_cleanup()
    deferred_task = _FakeTask()
    cleanup_task = _FakeTask()
    app = SimpleNamespace(state=SimpleNamespace(bg_tasks={"deferred_startup": deferred_task}))
    reset_calls: list[str] = []

    async def _reset_cleanup_service() -> None:
        reset_calls.append("cleanup")

    async def _reset_storage_service() -> None:
        reset_calls.append("storage")

    async def _reset_authnz_rate_limiter() -> None:
        reset_calls.append("auth")

    monkeypatch.setattr(shutdown_cleanup, "_reset_cleanup_service", _reset_cleanup_service)
    monkeypatch.setattr(shutdown_cleanup, "_reset_storage_service", _reset_storage_service)
    monkeypatch.setattr(shutdown_cleanup, "_reset_authnz_rate_limiter", _reset_authnz_rate_limiter)

    await shutdown_cleanup._shutdown_pre_worker_cleanup(
        app=app,
        cleanup_task=cleanup_task,
        chatbooks_cleanup_task=None,
        chatbooks_cleanup_stop_event=None,
        storage_cleanup_service=None,
        coordinated_legacy_component_names=set(),
        guard_exceptions=(RuntimeError,),
    )

    assert deferred_task.cancelled is True
    assert cleanup_task.cancelled is True
    assert reset_calls == ["cleanup", "storage", "auth"]


@pytest.mark.asyncio
async def test_shutdown_pre_worker_cleanup_skips_background_stopped_ephemeral_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_cleanup = _import_shutdown_pre_worker_cleanup()
    cleanup_task = _FakeTask()
    app = SimpleNamespace(state=SimpleNamespace(bg_tasks={}))

    async def _noop() -> None:
        return None

    monkeypatch.setattr(shutdown_cleanup, "_reset_cleanup_service", _noop)
    monkeypatch.setattr(shutdown_cleanup, "_reset_storage_service", _noop)
    monkeypatch.setattr(shutdown_cleanup, "_reset_authnz_rate_limiter", _noop)

    handles = await shutdown_cleanup.shutdown_pre_worker_cleanup(
        app=app,
        cleanup_task=cleanup_task,
        chatbooks_cleanup_task=None,
        chatbooks_cleanup_stop_event=None,
        storage_cleanup_service=None,
        coordinated_legacy_component_names=set(),
        guard_exceptions=(RuntimeError,),
        stopped_background_worker_names={"ephemeral_cleanup_task"},
    )

    assert cleanup_task.cancelled is False
    assert handles.cleanup_task is None


@pytest.mark.asyncio
async def test_shutdown_pre_worker_cleanup_stops_chatbooks_and_storage_when_not_coordinated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_cleanup = _import_shutdown_pre_worker_cleanup()
    stop_event = _FakeStopEvent()
    chatbooks_task = _FakeTask()
    storage_cleanup_service = _FakeStorageCleanupService()

    async def _noop() -> None:
        return None

    monkeypatch.setattr(shutdown_cleanup, "_reset_cleanup_service", _noop)
    monkeypatch.setattr(shutdown_cleanup, "_reset_storage_service", _noop)
    monkeypatch.setattr(shutdown_cleanup, "_reset_authnz_rate_limiter", _noop)

    await shutdown_cleanup._shutdown_pre_worker_cleanup(
        app=SimpleNamespace(state=SimpleNamespace()),
        cleanup_task=None,
        chatbooks_cleanup_task=chatbooks_task,
        chatbooks_cleanup_stop_event=stop_event,
        storage_cleanup_service=storage_cleanup_service,
        coordinated_legacy_component_names=set(),
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert chatbooks_task.cancelled is True
    assert storage_cleanup_service.stopped is True


@pytest.mark.asyncio
async def test_shutdown_pre_worker_cleanup_skips_coordinated_chatbooks_and_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_cleanup = _import_shutdown_pre_worker_cleanup()
    stop_event = _FakeStopEvent()
    chatbooks_task = _FakeTask()
    storage_cleanup_service = _FakeStorageCleanupService()
    reset_calls: list[str] = []

    async def _record_cleanup() -> None:
        reset_calls.append("cleanup")

    async def _record_storage() -> None:
        reset_calls.append("storage")

    async def _record_auth() -> None:
        reset_calls.append("auth")

    monkeypatch.setattr(shutdown_cleanup, "_reset_cleanup_service", _record_cleanup)
    monkeypatch.setattr(shutdown_cleanup, "_reset_storage_service", _record_storage)
    monkeypatch.setattr(shutdown_cleanup, "_reset_authnz_rate_limiter", _record_auth)

    await shutdown_cleanup._shutdown_pre_worker_cleanup(
        app=SimpleNamespace(state=SimpleNamespace()),
        cleanup_task=None,
        chatbooks_cleanup_task=chatbooks_task,
        chatbooks_cleanup_stop_event=stop_event,
        storage_cleanup_service=storage_cleanup_service,
        coordinated_legacy_component_names={"chatbooks_cleanup", "storage_cleanup_service"},
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is False
    assert chatbooks_task.cancelled is False
    assert storage_cleanup_service.stopped is False
    assert reset_calls == ["auth"]


@pytest.mark.asyncio
async def test_shutdown_pre_worker_cleanup_skips_stopped_background_chatbooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_cleanup = _import_shutdown_pre_worker_cleanup()
    stop_event = _FakeStopEvent()
    chatbooks_task = _FakeTask()

    async def _noop() -> None:
        return None

    monkeypatch.setattr(shutdown_cleanup, "_reset_cleanup_service", _noop)
    monkeypatch.setattr(shutdown_cleanup, "_reset_storage_service", _noop)
    monkeypatch.setattr(shutdown_cleanup, "_reset_authnz_rate_limiter", _noop)

    await shutdown_cleanup._shutdown_pre_worker_cleanup(
        app=SimpleNamespace(state=SimpleNamespace()),
        cleanup_task=None,
        chatbooks_cleanup_task=chatbooks_task,
        chatbooks_cleanup_stop_event=stop_event,
        storage_cleanup_service=None,
        coordinated_legacy_component_names=set(),
        stopped_background_worker_names={"chatbooks_cleanup"},
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is False
    assert chatbooks_task.cancelled is False


@pytest.mark.asyncio
async def test_shutdown_pre_worker_cleanup_swallows_guard_exceptions_from_local_guarded_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_cleanup = _import_shutdown_pre_worker_cleanup()
    deferred_task = _FakeTask(exc=RuntimeError("deferred boom"))
    storage_cleanup_service = _FakeStorageCleanupService(exc=RuntimeError("storage boom"))
    app = SimpleNamespace(state=SimpleNamespace(bg_tasks={"deferred_startup": deferred_task}))
    reset_calls: list[str] = []

    async def _failing_reset_cleanup_service() -> None:
        reset_calls.append("cleanup")
        raise RuntimeError("cleanup boom")

    async def _record_storage_reset() -> None:
        reset_calls.append("storage")

    async def _record_auth_reset() -> None:
        reset_calls.append("auth")

    monkeypatch.setattr(shutdown_cleanup, "_reset_cleanup_service", _failing_reset_cleanup_service)
    monkeypatch.setattr(shutdown_cleanup, "_reset_storage_service", _record_storage_reset)
    monkeypatch.setattr(shutdown_cleanup, "_reset_authnz_rate_limiter", _record_auth_reset)

    await shutdown_cleanup._shutdown_pre_worker_cleanup(
        app=app,
        cleanup_task=None,
        chatbooks_cleanup_task=None,
        chatbooks_cleanup_stop_event=None,
        storage_cleanup_service=storage_cleanup_service,
        coordinated_legacy_component_names=set(),
        guard_exceptions=(RuntimeError,),
    )

    assert deferred_task.cancelled is False
    assert storage_cleanup_service.stopped is False
    assert reset_calls == ["cleanup", "auth"]


@pytest.mark.asyncio
async def test_shutdown_pre_worker_cleanup_propagates_cleanup_task_cancel_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_cleanup = _import_shutdown_pre_worker_cleanup()
    cleanup_task = _FakeTask(exc=RuntimeError("cleanup boom"))

    async def _noop() -> None:
        return None

    monkeypatch.setattr(shutdown_cleanup, "_reset_cleanup_service", _noop)
    monkeypatch.setattr(shutdown_cleanup, "_reset_storage_service", _noop)
    monkeypatch.setattr(shutdown_cleanup, "_reset_authnz_rate_limiter", _noop)

    with pytest.raises(RuntimeError, match="cleanup boom"):
        await shutdown_cleanup._shutdown_pre_worker_cleanup(
            app=SimpleNamespace(state=SimpleNamespace()),
            cleanup_task=cleanup_task,
            chatbooks_cleanup_task=None,
            chatbooks_cleanup_stop_event=None,
            storage_cleanup_service=None,
            coordinated_legacy_component_names=set(),
            guard_exceptions=(RuntimeError,),
        )


@pytest.mark.asyncio
async def test_run_shutdown_pre_worker_cleanup_delegates_and_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_cleanup = _import_shutdown_pre_worker_cleanup()
    app = SimpleNamespace()
    cleanup_task = object()
    chatbooks_cleanup_task = object()
    chatbooks_cleanup_stop_event = object()
    storage_cleanup_service = object()
    recorded_calls: list[dict[str, object]] = []
    expected_handles = shutdown_cleanup.PreWorkerCleanupHandles(
        cleanup_task=cleanup_task,
        chatbooks_cleanup_task=chatbooks_cleanup_task,
        chatbooks_cleanup_stop_event=chatbooks_cleanup_stop_event,
        storage_cleanup_service=storage_cleanup_service,
    )

    async def _fake_shutdown_pre_worker_cleanup(**kwargs: object) -> Any:
        recorded_calls.append(kwargs)
        return expected_handles

    monkeypatch.setattr(
        shutdown_cleanup,
        "shutdown_pre_worker_cleanup",
        _fake_shutdown_pre_worker_cleanup,
    )

    handles = await shutdown_cleanup.run_shutdown_pre_worker_cleanup(
        app=app,
        cleanup_task=cleanup_task,
        chatbooks_cleanup_task=chatbooks_cleanup_task,
        chatbooks_cleanup_stop_event=chatbooks_cleanup_stop_event,
        storage_cleanup_service=storage_cleanup_service,
        coordinated_legacy_component_names={"usage_aggregator"},
        stopped_background_worker_names={"ephemeral_cleanup_task"},
        guard_exceptions=(RuntimeError,),
    )

    assert handles is expected_handles
    assert len(recorded_calls) == 1
    assert recorded_calls[0]["app"] is app
    assert recorded_calls[0]["cleanup_task"] is cleanup_task
    assert recorded_calls[0]["chatbooks_cleanup_task"] is chatbooks_cleanup_task
    assert recorded_calls[0]["chatbooks_cleanup_stop_event"] is chatbooks_cleanup_stop_event
    assert recorded_calls[0]["storage_cleanup_service"] is storage_cleanup_service
    assert recorded_calls[0]["coordinated_legacy_component_names"] == {"usage_aggregator"}
    assert recorded_calls[0]["stopped_background_worker_names"] == {"ephemeral_cleanup_task"}


@pytest.mark.asyncio
async def test_run_shutdown_pre_worker_cleanup_logs_and_returns_original_handles_on_guard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_cleanup = _import_shutdown_pre_worker_cleanup()
    cleanup_task = object()
    chatbooks_cleanup_task = object()
    chatbooks_cleanup_stop_event = object()
    storage_cleanup_service = object()
    debug_messages: list[str] = []

    async def _raise_guard_failure(**_kwargs: object) -> None:
        raise RuntimeError("pre-worker unavailable")

    monkeypatch.setattr(
        shutdown_cleanup,
        "shutdown_pre_worker_cleanup",
        _raise_guard_failure,
    )
    monkeypatch.setattr(
        shutdown_cleanup.logger,
        "debug",
        lambda message, *args, **kwargs: debug_messages.append(str(message)),
    )

    handles = await shutdown_cleanup.run_shutdown_pre_worker_cleanup(
        app=SimpleNamespace(),
        cleanup_task=cleanup_task,
        chatbooks_cleanup_task=chatbooks_cleanup_task,
        chatbooks_cleanup_stop_event=chatbooks_cleanup_stop_event,
        storage_cleanup_service=storage_cleanup_service,
        coordinated_legacy_component_names=set(),
        stopped_background_worker_names={"ephemeral_cleanup_task"},
        guard_exceptions=(RuntimeError,),
    )

    assert handles.cleanup_task is None
    assert handles.chatbooks_cleanup_task is chatbooks_cleanup_task
    assert handles.chatbooks_cleanup_stop_event is chatbooks_cleanup_stop_event
    assert handles.storage_cleanup_service is storage_cleanup_service
    assert any("Pre-worker cleanup skipped" in message for message in debug_messages)
