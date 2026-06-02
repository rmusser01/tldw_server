from __future__ import annotations

import asyncio
import importlib
import sys
from collections.abc import Generator
from contextlib import suppress
from typing import Any

import pytest
from fastapi import FastAPI

from tldw_Server_API.app.services.lifecycle_worker_specs import (
    ShutdownPhase,
    WorkerLifecycleContext,
    WorkerStrategy,
)

pytestmark = pytest.mark.unit


def _import_startup_compactor_websub_workers():
    sys.modules.pop("tldw_Server_API.app.services.startup_compactor_websub_workers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_compactor_websub_workers")


def _context(
    *,
    route_enabled=lambda *_args, **_kwargs: True,
) -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app=FastAPI(),
        settings={},
        test_mode=True,
        route_enabled=route_enabled,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )


def _specs_by_name(startup_workers: Any) -> dict[str, Any]:
    return {
        spec.name: spec
        for spec in startup_workers.provide_compactor_websub_worker_specs()
    }


def test_compactor_websub_worker_specs_match_legacy_worker_contract() -> None:
    startup_workers = _import_startup_compactor_websub_workers()

    specs = _specs_by_name(startup_workers)

    compactor = specs["embeddings_compactor_task"]
    assert compactor.task_name == "embeddings_compactor_task"
    assert compactor.category == "embeddings"
    assert compactor.phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
    assert compactor.timeout_sec == 5.0
    assert compactor.strategy is WorkerStrategy.STOP_EVENT_TASK
    assert compactor.factory is not None

    websub = specs["websub_renewal_task"]
    assert websub.task_name == "websub_renewal_task"
    assert websub.category == "collections-websub"
    assert websub.phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
    assert websub.timeout_sec == 5.0
    assert websub.strategy is WorkerStrategy.STOP_EVENT_TASK
    assert websub.factory is not None
    assert websub.shutdown_callback_factory is None


def test_compactor_websub_worker_specs_use_expected_names() -> None:
    startup_workers = _import_startup_compactor_websub_workers()

    assert [spec.name for spec in startup_workers.provide_compactor_websub_worker_specs()] == [
        "embeddings_compactor_task",
        "websub_renewal_task",
    ]


def test_embeddings_compactor_worker_spec_factory_delegates_to_existing_worker_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_compactor_websub_workers()
    calls: list[object] = []
    monkeypatch.setattr(
        startup_workers,
        "_run_embeddings_vector_compactor_service",
        lambda stop_event: calls.append(stop_event) or "compactor-awaitable",
    )

    spec = _specs_by_name(startup_workers)["embeddings_compactor_task"]

    assert spec.factory is not None
    assert spec.factory(_context(), "compactor-stop") == "compactor-awaitable"
    assert calls == ["compactor-stop"]


def test_websub_worker_spec_predicate_uses_callback_url_and_route_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_compactor_websub_workers()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        startup_workers.os,
        "getenv",
        lambda key, default=None: {
            "WEBSUB_CALLBACK_BASE_URL": "https://callback.test",
            "WEBSUB_RENEWAL_WORKER_ENABLED": "true",
        }.get(key, default),
    )

    def _route_enabled(*args: object, **kwargs: object) -> bool:
        calls.append((args, kwargs))
        return False

    spec = _specs_by_name(startup_workers)["websub_renewal_task"]

    assert spec.enabled(_context(route_enabled=_route_enabled)) is False
    assert calls == [(("collections-websub",), {})]


@pytest.mark.asyncio
async def test_websub_worker_spec_factory_starts_and_cancels_legacy_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_compactor_websub_workers()
    websub_started = asyncio.Event()
    cancelled: list[str] = []

    async def _websub_loop() -> None:
        websub_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append("websub")
            raise

    monkeypatch.setattr(startup_workers, "_run_websub_renewal_loop", _websub_loop)

    spec = _specs_by_name(startup_workers)["websub_renewal_task"]
    stop_event = asyncio.Event()
    assert spec.factory is not None
    lifecycle_task = asyncio.create_task(spec.factory(_context(), stop_event))

    await asyncio.wait_for(websub_started.wait(), timeout=1)
    assert lifecycle_task.done() is False

    stop_event.set()
    await asyncio.wait_for(lifecycle_task, timeout=1)

    assert cancelled == ["websub"]


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
