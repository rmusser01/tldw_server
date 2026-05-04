from __future__ import annotations

import importlib
import inspect
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


@pytest.mark.asyncio
async def test_shutdown_pre_worker_cleanup_returns_empty_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_cleanup = _import_shutdown_pre_worker_cleanup()
    calls: list[dict[str, object]] = []

    async def _record_shutdown(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(shutdown_cleanup, "_shutdown_pre_worker_cleanup", _record_shutdown)

    handles = await shutdown_cleanup.shutdown_pre_worker_cleanup(
        app="app",
        guard_exceptions=(RuntimeError,),
    )

    assert len(calls) == 1
    assert calls[0] == {
        "app": "app",
        "guard_exceptions": (RuntimeError,),
    }
    assert vars(handles) == {}


def test_shutdown_pre_worker_cleanup_no_longer_accepts_registry_owned_worker_handles() -> None:
    shutdown_cleanup = _import_shutdown_pre_worker_cleanup()

    obsolete_parameters = {
        "cleanup_task",
        "chatbooks_cleanup_task",
        "chatbooks_cleanup_stop_event",
        "storage_cleanup_service",
        "coordinated_legacy_component_names",
        "stopped_background_worker_names",
    }

    for helper_name in (
        "shutdown_pre_worker_cleanup",
        "run_shutdown_pre_worker_cleanup",
        "_shutdown_pre_worker_cleanup",
    ):
        parameters = set(inspect.signature(getattr(shutdown_cleanup, helper_name)).parameters)
        assert parameters.isdisjoint(obsolete_parameters)


@pytest.mark.asyncio
async def test_shutdown_pre_worker_cleanup_cancels_deferred_startup_and_runs_finalizers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_cleanup = _import_shutdown_pre_worker_cleanup()
    deferred_task = _FakeTask()
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
        guard_exceptions=(RuntimeError,),
    )

    assert deferred_task.cancelled is True
    assert reset_calls == ["cleanup", "storage", "auth"]


@pytest.mark.asyncio
async def test_shutdown_pre_worker_cleanup_does_not_direct_stop_registry_owned_cleanup_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_cleanup = _import_shutdown_pre_worker_cleanup()
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
        guard_exceptions=(RuntimeError,),
    )

    assert reset_calls == ["cleanup", "storage", "auth"]


@pytest.mark.asyncio
async def test_shutdown_pre_worker_cleanup_swallows_guard_exceptions_from_local_guarded_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_cleanup = _import_shutdown_pre_worker_cleanup()
    deferred_task = _FakeTask(exc=RuntimeError("deferred boom"))
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
        guard_exceptions=(RuntimeError,),
    )

    assert deferred_task.cancelled is False
    assert reset_calls == ["cleanup", "auth"]


@pytest.mark.asyncio
async def test_run_shutdown_pre_worker_cleanup_delegates_and_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_cleanup = _import_shutdown_pre_worker_cleanup()
    app = SimpleNamespace()
    recorded_calls: list[dict[str, object]] = []
    expected_handles = shutdown_cleanup.PreWorkerCleanupHandles()

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
        guard_exceptions=(RuntimeError,),
    )

    assert handles is expected_handles
    assert len(recorded_calls) == 1
    assert recorded_calls[0] == {
        "app": app,
        "guard_exceptions": (RuntimeError,),
    }


@pytest.mark.asyncio
async def test_run_shutdown_pre_worker_cleanup_logs_and_returns_empty_handles_on_guard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_cleanup = _import_shutdown_pre_worker_cleanup()
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
        guard_exceptions=(RuntimeError,),
    )

    assert vars(handles) == {}
    assert any("Pre-worker cleanup skipped" in message for message in debug_messages)
