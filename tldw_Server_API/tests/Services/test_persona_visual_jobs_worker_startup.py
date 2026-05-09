from __future__ import annotations

import asyncio
import importlib
import sys

import pytest
from fastapi import FastAPI


pytestmark = pytest.mark.unit


def _import_startup_optional_workers():
    sys.modules.pop("tldw_Server_API.app.services.startup_optional_workers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_optional_workers")


@pytest.mark.asyncio
async def test_persona_visual_generation_worker_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_optional_workers()

    monkeypatch.delenv("PERSONA_VISUAL_GENERATION_WORKER_ENABLED", raising=False)
    monkeypatch.setattr(
        startup_workers,
        "_make_event",
        lambda: (_ for _ in ()).throw(AssertionError("no event")),
    )

    stop_event, task = await startup_workers._start_persona_visual_generation_worker()

    assert stop_event is None
    assert task is None


@pytest.mark.asyncio
async def test_persona_visual_generation_worker_registers_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_optional_workers()
    from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase, WorkerRegistry

    app = FastAPI()
    worker_inventory = WorkerRegistry(app)
    observed_stop_events: list[asyncio.Event] = []

    async def _fake_worker(stop_event: asyncio.Event) -> None:
        observed_stop_events.append(stop_event)
        await stop_event.wait()

    monkeypatch.setenv("PERSONA_VISUAL_GENERATION_WORKER_ENABLED", "1")
    monkeypatch.setattr(
        startup_workers,
        "_run_persona_visual_generation_worker_service",
        _fake_worker,
    )

    stop_event = None
    task = None
    try:
        stop_event, task = await startup_workers._start_persona_visual_generation_worker(
            worker_inventory=worker_inventory,
        )
        await asyncio.sleep(0)

        assert stop_event is not None
        assert task is not None
        assert task.get_name() == "persona_visual_generation_task"
        assert observed_stop_events == [stop_event]
        assert len(worker_inventory.handles) == 1
        handle = worker_inventory.handles[0]
        assert handle.name == "persona_visual_generation_task"
        assert handle.task is task
        assert handle.stop_event is stop_event
        assert handle.category == "persona"
        assert handle.shutdown_phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
    finally:
        if stop_event is not None:
            stop_event.set()
        if task is not None:
            await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_persona_visual_portability_worker_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_optional_workers()

    monkeypatch.delenv("PERSONA_VISUAL_PORTABILITY_WORKER_ENABLED", raising=False)
    monkeypatch.setattr(
        startup_workers,
        "_make_event",
        lambda: (_ for _ in ()).throw(AssertionError("no event")),
    )

    stop_event, task = await startup_workers._start_persona_visual_portability_worker()

    assert stop_event is None
    assert task is None


@pytest.mark.asyncio
async def test_persona_visual_portability_worker_registers_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_optional_workers()
    from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase, WorkerRegistry

    app = FastAPI()
    worker_inventory = WorkerRegistry(app)
    observed_stop_events: list[asyncio.Event] = []

    async def _fake_worker(stop_event: asyncio.Event) -> None:
        observed_stop_events.append(stop_event)
        await stop_event.wait()

    monkeypatch.setenv("PERSONA_VISUAL_PORTABILITY_WORKER_ENABLED", "1")
    monkeypatch.setattr(
        startup_workers,
        "_run_persona_visual_portability_worker_service",
        _fake_worker,
    )

    stop_event = None
    task = None
    try:
        stop_event, task = await startup_workers._start_persona_visual_portability_worker(
            worker_inventory=worker_inventory,
        )
        await asyncio.sleep(0)

        assert stop_event is not None
        assert task is not None
        assert task.get_name() == "persona_visual_portability_task"
        assert observed_stop_events == [stop_event]
        assert len(worker_inventory.handles) == 1
        handle = worker_inventory.handles[0]
        assert handle.name == "persona_visual_portability_task"
        assert handle.task is task
        assert handle.stop_event is stop_event
        assert handle.category == "persona"
        assert handle.shutdown_phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
    finally:
        if stop_event is not None:
            stop_event.set()
        if task is not None:
            await asyncio.wait_for(task, timeout=1)
