from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import suppress
import importlib
import sys

from fastapi import FastAPI
import pytest


pytestmark = pytest.mark.unit


def _import_startup_compactor_websub_workers():
    sys.modules.pop("tldw_Server_API.app.services.startup_compactor_websub_workers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_compactor_websub_workers")


@pytest.mark.asyncio
async def test_start_compactor_websub_workers_combines_handles_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_compactor_websub_workers()
    calls: list[str] = []

    async def _record_compactor(**kwargs):
        assert kwargs == {"worker_inventory": None}
        calls.append("compactor")
        return ("compactor-stop", "compactor-task")

    async def _record_websub(**kwargs):
        del kwargs
        calls.append("websub")
        return "websub-task"

    monkeypatch.setattr(startup_workers, "_start_embeddings_vector_compactor", _record_compactor)
    monkeypatch.setattr(startup_workers, "_start_websub_renewal_worker", _record_websub)

    handles = await startup_workers.start_compactor_websub_workers(
        should_start_worker=lambda *args, **kwargs: False,
    )

    assert calls == ["compactor", "websub"]
    assert handles.embeddings_compactor_stop_event == "compactor-stop"
    assert handles.embeddings_compactor_task == "compactor-task"
    assert handles.websub_renewal_task == "websub-task"


@pytest.mark.asyncio
async def test_start_embeddings_vector_compactor_starts_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_compactor_websub_workers()
    captured_stop_events: list[object] = []
    created_coroutines: list[object] = []

    monkeypatch.setattr(
        startup_workers.os,
        "getenv",
        lambda key, default=None: "true" if key == "EMBEDDINGS_COMPACTOR_ENABLED" else default,
    )
    monkeypatch.setattr(startup_workers, "_make_event", lambda: "compactor-stop")
    monkeypatch.setattr(
        startup_workers,
        "_create_task",
        lambda coro: created_coroutines.append(coro) or "compactor-task",
    )
    monkeypatch.setattr(
        startup_workers,
        "_run_embeddings_vector_compactor_service",
        lambda stop_event: captured_stop_events.append(stop_event) or "compactor-coro",
    )

    stop_event, task = await startup_workers._start_embeddings_vector_compactor()

    assert stop_event == "compactor-stop"
    assert task == "compactor-task"
    assert captured_stop_events == ["compactor-stop"]
    assert created_coroutines == ["compactor-coro"]


@pytest.mark.asyncio
async def test_start_embeddings_vector_compactor_registers_background_inventory_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_compactor_websub_workers()
    from tldw_Server_API.app.services.lifecycle_workers import (
        ShutdownPhase,
        WorkerRegistry,
    )

    app = FastAPI()
    worker_inventory = WorkerRegistry(app)
    observed_stop_events: list[asyncio.Event] = []

    async def _fake_compactor(stop_event: asyncio.Event) -> None:
        observed_stop_events.append(stop_event)
        await stop_event.wait()

    monkeypatch.setattr(
        startup_workers.os,
        "getenv",
        lambda key, default=None: "true" if key == "EMBEDDINGS_COMPACTOR_ENABLED" else default,
    )
    monkeypatch.setattr(
        startup_workers,
        "_run_embeddings_vector_compactor_service",
        _fake_compactor,
    )

    stop_event = None
    task = None
    try:
        stop_event, task = await startup_workers._start_embeddings_vector_compactor(
            worker_inventory=worker_inventory,
        )
        await asyncio.sleep(0)

        assert stop_event is not None
        assert task is not None
        assert task.get_name() == "embeddings_compactor_task"
        assert observed_stop_events == [stop_event]
        assert len(worker_inventory.handles) == 1
        handle = worker_inventory.handles[0]
        assert handle.name == "embeddings_compactor_task"
        assert handle.task is task
        assert handle.stop_event is stop_event
        assert handle.category == "embeddings"
        assert handle.shutdown_phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
        assert app.state._tldw_shutdown_worker_inventory == [
            {
                "name": "embeddings_compactor_task",
                "task_name": "embeddings_compactor_task",
                "has_stop_event": True,
                "timeout_sec": 5.0,
                "category": "embeddings",
                "shutdown_phase": "background_worker_shutdown",
            }
        ]
        assert app.state._tldw_shutdown_job_poller_inventory == []
    finally:
        if stop_event is not None:
            stop_event.set()
        if task is not None:
            await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_start_embeddings_vector_compactor_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_compactor_websub_workers()

    monkeypatch.setattr(
        startup_workers.os,
        "getenv",
        lambda key, default=None: "true" if key == "EMBEDDINGS_COMPACTOR_ENABLED" else default,
    )
    monkeypatch.setattr(startup_workers, "_make_event", lambda: "compactor-stop")

    def _failing_create_task(coro):
        del coro
        raise RuntimeError("boom")

    monkeypatch.setattr(startup_workers, "_create_task", _failing_create_task)
    monkeypatch.setattr(
        startup_workers,
        "_run_embeddings_vector_compactor_service",
        lambda stop_event: stop_event,
    )

    stop_event, task = await startup_workers._start_embeddings_vector_compactor()

    assert stop_event is None
    assert task is None


@pytest.mark.asyncio
async def test_start_websub_renewal_worker_starts_when_callback_and_worker_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_compactor_websub_workers()
    created_tasks: list[tuple[object, str | None]] = []

    def _fake_create_task(coro: object, *, name: str | None = None) -> str:
        created_tasks.append((coro, name))
        return "websub-task"

    monkeypatch.setattr(
        startup_workers.os,
        "getenv",
        lambda key, default=None: "http://callback.example" if key == "WEBSUB_CALLBACK_BASE_URL" else default,
    )
    monkeypatch.setattr(startup_workers, "_create_task", _fake_create_task)
    monkeypatch.setattr(startup_workers, "_run_websub_renewal_loop", lambda: "websub-coro")

    task = await startup_workers._start_websub_renewal_worker(
        should_start_worker=lambda flag, route, **kwargs: (flag, route, kwargs) == (
            "WEBSUB_RENEWAL_WORKER_ENABLED",
            "collections-websub",
            {},
        ),
    )

    assert task == "websub-task"
    assert created_tasks == [("websub-coro", "websub_renewal_task")]


@pytest.mark.asyncio
async def test_start_websub_renewal_worker_registers_background_inventory_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_compactor_websub_workers()
    from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase, WorkerRegistry

    app = FastAPI()
    worker_inventory = WorkerRegistry(app)
    started = asyncio.Event()

    async def _fake_websub_loop() -> None:
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        startup_workers.os,
        "getenv",
        lambda key, default=None: "http://callback.example" if key == "WEBSUB_CALLBACK_BASE_URL" else default,
    )
    monkeypatch.setattr(startup_workers, "_run_websub_renewal_loop", _fake_websub_loop)

    task = await startup_workers._start_websub_renewal_worker(
        should_start_worker=lambda flag, route, **kwargs: (flag, route, kwargs) == (
            "WEBSUB_RENEWAL_WORKER_ENABLED",
            "collections-websub",
            {},
        ),
        worker_inventory=worker_inventory,
    )

    try:
        await asyncio.wait_for(started.wait(), timeout=1)

        assert task is not None
        assert task.get_name() == "websub_renewal_task"
        assert len(worker_inventory.handles) == 1
        handle = worker_inventory.handles[0]
        assert handle.name == "websub_renewal_task"
        assert handle.task is task
        assert handle.stop_event is None
        assert handle.category == "collections-websub"
        assert handle.shutdown_phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
        assert app.state._tldw_shutdown_worker_inventory == [
            {
                "name": "websub_renewal_task",
                "task_name": "websub_renewal_task",
                "has_stop_event": False,
                "timeout_sec": 5.0,
                "category": "collections-websub",
                "shutdown_phase": "background_worker_shutdown",
            }
        ]
        assert app.state._tldw_shutdown_job_poller_inventory == []
    finally:
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_start_websub_renewal_worker_cancels_task_when_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_compactor_websub_workers()
    created_task_names: list[str | None] = []
    registered_workers: list[object] = []

    class FakeTask:
        def __init__(self) -> None:
            self.cancelled = False
            self.awaited = False

        def cancel(self) -> None:
            self.cancelled = True

        def __await__(self) -> Generator[object, None, None]:
            self.awaited = True

            async def _cancelled() -> None:
                raise asyncio.CancelledError

            return _cancelled().__await__()

    class FailingInventory:
        def register(self, worker: object) -> object:
            registered_workers.append(worker)
            raise RuntimeError("inventory failed")

    fake_task = FakeTask()

    def _fake_create_task(coro: object, *, name: str | None = None) -> FakeTask:
        del coro
        created_task_names.append(name)
        return fake_task

    monkeypatch.setattr(
        startup_workers.os,
        "getenv",
        lambda key, default=None: "http://callback.example" if key == "WEBSUB_CALLBACK_BASE_URL" else default,
    )
    monkeypatch.setattr(startup_workers, "_create_task", _fake_create_task)
    monkeypatch.setattr(startup_workers, "_run_websub_renewal_loop", lambda: "websub-coro")

    task = await startup_workers._start_websub_renewal_worker(
        should_start_worker=lambda *args, **kwargs: True,
        worker_inventory=FailingInventory(),
    )

    assert task is None
    assert created_task_names == ["websub_renewal_task"]
    assert fake_task.cancelled is True
    assert fake_task.awaited is True
    assert len(registered_workers) == 1


@pytest.mark.asyncio
async def test_start_websub_renewal_worker_skips_without_callback_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_compactor_websub_workers()
    monkeypatch.setattr(
        startup_workers.os,
        "getenv",
        lambda key, default=None: "" if key == "WEBSUB_CALLBACK_BASE_URL" else default,
    )
    monkeypatch.setattr(
        startup_workers,
        "_create_task",
        lambda coro: (_ for _ in ()).throw(AssertionError("no task")),
    )

    task = await startup_workers._start_websub_renewal_worker(
        should_start_worker=lambda *args, **kwargs: True,
    )

    assert task is None
