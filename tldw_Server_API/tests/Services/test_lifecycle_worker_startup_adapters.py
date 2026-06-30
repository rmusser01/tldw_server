from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_run_started_task_until_stop_bounds_hung_stopper_and_cancels_task() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_startup_adapters import (
        run_started_task_until_stop,
    )

    stop_event = asyncio.Event()
    stop_event.set()
    worker_never_stops = asyncio.Event()
    task = asyncio.create_task(worker_never_stops.wait())
    stopper_started = asyncio.Event()

    async def _starter():
        return task

    async def _hung_stopper(_task):
        stopper_started.set()
        await asyncio.Event().wait()

    await run_started_task_until_stop(
        stop_event,
        starter=_starter,
        stopper=_hung_stopper,
        timeout_sec=0.01,
    )

    assert stopper_started.is_set()
    assert task.done()


@pytest.mark.asyncio
async def test_run_start_stop_service_until_stop_bounds_hung_stopper() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_startup_adapters import (
        run_start_stop_service_until_stop,
    )

    stop_event = asyncio.Event()
    stop_event.set()
    calls: list[str] = []

    async def _starter() -> None:
        calls.append("start")

    async def _hung_stopper() -> None:
        calls.append("stop")
        await asyncio.Event().wait()

    await run_start_stop_service_until_stop(
        stop_event,
        starter=_starter,
        stopper=_hung_stopper,
        timeout_sec=0.01,
    )

    assert calls == ["start", "stop"]
