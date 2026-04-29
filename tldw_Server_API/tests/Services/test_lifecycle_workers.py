import asyncio

import pytest
from fastapi import FastAPI


async def _wait_for_stop(stop_event: asyncio.Event) -> None:
    await stop_event.wait()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_worker_inventory_publishes_full_and_filtered_views() -> None:
    from tldw_Server_API.app.services.lifecycle_workers import (
        ManagedWorker,
        ShutdownPhase,
        publish_worker_inventory,
    )

    app = FastAPI()
    job_stop_event = asyncio.Event()
    background_stop_event = asyncio.Event()
    job_task = asyncio.create_task(_wait_for_stop(job_stop_event), name="job-task")
    background_task = asyncio.create_task(
        _wait_for_stop(background_stop_event),
        name="background-task",
    )

    try:
        publish_worker_inventory(
            app,
            [
                ManagedWorker(
                    name="job_worker",
                    task=job_task,
                    stop_event=job_stop_event,
                    timeout_sec=5.0,
                    category="jobs",
                    shutdown_phase=ShutdownPhase.JOB_POLLER_QUIESCE,
                ),
                ManagedWorker(
                    name="background_worker",
                    task=background_task,
                    stop_event=background_stop_event,
                    timeout_sec=2.0,
                    category="jobs",
                    shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
                ),
            ],
        )

        assert app.state._tldw_shutdown_worker_inventory == [
            {
                "name": "job_worker",
                "task_name": "job-task",
                "has_stop_event": True,
                "timeout_sec": 5.0,
                "category": "jobs",
                "shutdown_phase": "job_poller_quiesce",
            },
            {
                "name": "background_worker",
                "task_name": "background-task",
                "has_stop_event": True,
                "timeout_sec": 2.0,
                "category": "jobs",
                "shutdown_phase": "background_worker_shutdown",
            },
        ]
        assert app.state._tldw_shutdown_job_poller_inventory == [
            {
                "name": "job_worker",
                "task_name": "job-task",
                "has_stop_event": True,
                "timeout_sec": 5.0,
            }
        ]
    finally:
        job_stop_event.set()
        background_stop_event.set()
        await asyncio.wait_for(asyncio.gather(job_task, background_task), timeout=1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_stop_event_worker_registers_named_task_and_stop_event() -> None:
    from tldw_Server_API.app.services.lifecycle_workers import (
        ShutdownPhase,
        WorkerInventory,
        start_stop_event_worker,
    )

    app = FastAPI()
    inventory = WorkerInventory(app)
    observed_stop_event: asyncio.Event | None = None

    async def _worker(stop_event: asyncio.Event) -> None:
        nonlocal observed_stop_event
        observed_stop_event = stop_event
        await stop_event.wait()

    task, stop_event = await start_stop_event_worker(
        inventory,
        name="background_worker",
        task_name="stable-background-task",
        coroutine_factory=_worker,
        timeout_sec=2.5,
        category="jobs",
        shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
    )

    try:
        await asyncio.sleep(0)

        assert isinstance(stop_event, asyncio.Event)
        assert observed_stop_event is stop_event
        assert task.get_name() == "stable-background-task"
        assert len(inventory.handles) == 1

        handle = inventory.handles[0]
        assert handle.name == "background_worker"
        assert handle.task is task
        assert handle.stop_event is stop_event
        assert handle.timeout_sec == 2.5
        assert handle.category == "jobs"
        assert handle.shutdown_phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
        assert app.state._tldw_shutdown_worker_inventory == [
            {
                "name": "background_worker",
                "task_name": "stable-background-task",
                "has_stop_event": True,
                "timeout_sec": 2.5,
                "category": "jobs",
                "shutdown_phase": "background_worker_shutdown",
            }
        ]
    finally:
        stop_event.set()
        await asyncio.wait_for(task, timeout=1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_registered_workers_sets_events_and_waits_concurrently() -> None:
    from tldw_Server_API.app.services.lifecycle_workers import (
        ManagedWorker,
        stop_registered_workers,
    )

    app = FastAPI()

    async def _delayed_shutdown(stop_event: asyncio.Event) -> None:
        await stop_event.wait()
        await asyncio.sleep(0.12)

    stop_a = asyncio.Event()
    stop_b = asyncio.Event()
    task_a = asyncio.create_task(_delayed_shutdown(stop_a), name="worker-a-task")
    task_b = asyncio.create_task(_delayed_shutdown(stop_b), name="worker-b-task")

    started = asyncio.get_running_loop().time()
    await stop_registered_workers(
        app,
        [
            ManagedWorker(
                name="worker_a",
                task=task_a,
                stop_event=stop_a,
                timeout_sec=1.0,
            ),
            ManagedWorker(
                name="worker_b",
                task=task_b,
                stop_event=stop_b,
                timeout_sec=1.0,
            ),
        ],
        stopped_names_attr="_tldw_stopped_worker_names",
        log_label="test worker",
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert stop_a.is_set() is True
    assert stop_b.is_set() is True
    assert elapsed < 0.2
    assert app.state._tldw_stopped_worker_names == ["worker_a", "worker_b"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_registered_workers_cancels_timeout_without_blocking_cooperative_worker() -> None:
    from tldw_Server_API.app.services.lifecycle_workers import (
        ManagedWorker,
        stop_registered_workers,
    )

    app = FastAPI()
    cooperative_stop_event = asyncio.Event()
    stubborn_stop_event = asyncio.Event()

    async def _cooperative_worker(stop_event: asyncio.Event) -> None:
        await stop_event.wait()

    async def _stubborn_worker(stop_event: asyncio.Event) -> None:
        await stop_event.wait()
        await asyncio.Future()

    cooperative_task = asyncio.create_task(
        _cooperative_worker(cooperative_stop_event),
        name="cooperative-task",
    )
    stubborn_task = asyncio.create_task(
        _stubborn_worker(stubborn_stop_event),
        name="stubborn-task",
    )
    await asyncio.sleep(0)

    await stop_registered_workers(
        app,
        [
            ManagedWorker(
                name="cooperative_worker",
                task=cooperative_task,
                stop_event=cooperative_stop_event,
                timeout_sec=1.0,
            ),
            ManagedWorker(
                name="stubborn_worker",
                task=stubborn_task,
                stop_event=stubborn_stop_event,
                timeout_sec=0.01,
            ),
        ],
        stopped_names_attr="_tldw_stopped_worker_names",
        log_label="test worker",
    )

    assert cooperative_stop_event.is_set() is True
    assert stubborn_stop_event.is_set() is True
    assert cooperative_task.done() is True
    assert stubborn_task.cancelled() is True
    assert app.state._tldw_stopped_worker_names == [
        "cooperative_worker",
        "stubborn_worker",
    ]
