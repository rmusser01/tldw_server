from __future__ import annotations

import asyncio
import importlib
import sys
from contextlib import contextmanager

import pytest
from fastapi import FastAPI

pytestmark = pytest.mark.unit


def _import_shutdown_owned_job_pollers():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_owned_job_pollers", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_owned_job_pollers")


@pytest.mark.asyncio
async def test_register_owned_job_poller_publishes_lifecycle_worker_inventory() -> None:
    shutdown_pollers = _import_shutdown_owned_job_pollers()
    from tldw_Server_API.app.services.lifecycle_workers import (
        ManagedWorker,
        ShutdownPhase,
    )

    app = FastAPI()
    stop_event = asyncio.Event()
    task = asyncio.create_task(stop_event.wait(), name="core-jobs-task")
    handles: list[object] = []

    try:
        shutdown_pollers.register_owned_job_poller(
            app,
            handles,
            name="core_jobs_task",
            task=task,
            stop_event=stop_event,
            timeout_sec=3.0,
        )

        assert len(handles) == 1
        handle = handles[0]
        assert isinstance(handle, ManagedWorker)
        assert handle.shutdown_phase is ShutdownPhase.JOB_POLLER_QUIESCE
        assert app.state._tldw_shutdown_worker_inventory == [
            {
                "name": "core_jobs_task",
                "task_name": "core-jobs-task",
                "has_stop_event": True,
                "timeout_sec": 3.0,
                "category": None,
                "shutdown_phase": "job_poller_quiesce",
            }
        ]
        assert app.state._tldw_shutdown_job_poller_inventory == [
            {
                "name": "core_jobs_task",
                "task_name": "core-jobs-task",
                "has_stop_event": True,
                "timeout_sec": 3.0,
            }
        ]
    finally:
        stop_event.set()
        await asyncio.wait_for(task, timeout=1)


def test_publish_shutdown_job_poller_inventory_surfaces_invalid_handle_shape() -> None:
    shutdown_pollers = _import_shutdown_owned_job_pollers()
    app = FastAPI()

    with pytest.raises(AttributeError):
        shutdown_pollers.publish_shutdown_job_poller_inventory(app, [object()])


@pytest.mark.asyncio
async def test_replace_owned_job_poller_inventory_preserves_background_workers() -> None:
    shutdown_pollers = _import_shutdown_owned_job_pollers()
    from tldw_Server_API.app.services.lifecycle_workers import (
        ManagedWorker,
        ShutdownPhase,
    )

    app = FastAPI()
    background_stop_event = asyncio.Event()
    poller_stop_event = asyncio.Event()
    background_task = asyncio.create_task(
        background_stop_event.wait(),
        name="background-worker",
    )
    poller_task = asyncio.create_task(poller_stop_event.wait(), name="poller-worker")
    handles: list[object] = [
        ManagedWorker(
            name="background_worker",
            task=background_task,
            stop_event=background_stop_event,
            shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        )
    ]

    try:
        shutdown_pollers.replace_owned_job_poller_inventory(
            app,
            handles,
            registrations=[
                ("poller_worker", poller_task, poller_stop_event, 3.0),
            ],
        )

        assert {handle.name for handle in handles} == {
            "background_worker",
            "poller_worker",
        }
        assert {entry["name"] for entry in app.state._tldw_shutdown_worker_inventory} == {
            "background_worker",
            "poller_worker",
        }
        assert app.state._tldw_shutdown_job_poller_inventory == [
            {
                "name": "poller_worker",
                "task_name": "poller-worker",
                "has_stop_event": True,
                "timeout_sec": 3.0,
            }
        ]
    finally:
        background_stop_event.set()
        poller_stop_event.set()
        await asyncio.wait_for(asyncio.gather(background_task, poller_task), timeout=1)


def test_register_and_replace_owned_job_poller_inventory_refreshes_app_state() -> None:
    shutdown_pollers = _import_shutdown_owned_job_pollers()
    app = FastAPI()
    loop = asyncio.new_event_loop()
    task_one = None
    task_two = None
    try:
        asyncio.set_event_loop(loop)
        task_one = loop.create_task(asyncio.sleep(0), name="first-task")
        task_two = loop.create_task(asyncio.sleep(0), name="second-task")

        handles: list[object] = []
        shutdown_pollers.register_owned_job_poller(
            app,
            handles,
            name="first",
            task=task_one,
            stop_event=None,
            timeout_sec=1.5,
        )
        shutdown_pollers.replace_owned_job_poller_inventory(
            app,
            handles,
            registrations=[
                ("second", task_two, None, 2.5),
            ],
        )

        assert [handle.name for handle in handles] == ["second"]
        assert app.state._tldw_shutdown_job_poller_inventory == [
            {
                "name": "second",
                "task_name": "second-task",
                "has_stop_event": False,
                "timeout_sec": 2.5,
            }
        ]
    finally:
        if task_one is not None or task_two is not None:
            loop.run_until_complete(asyncio.gather(*(task for task in (task_one, task_two) if task is not None)))
        asyncio.set_event_loop(None)
        loop.close()


@pytest.mark.asyncio
async def test_quiesce_owned_job_pollers_waits_then_calls_stop_callback() -> None:
    shutdown_pollers = _import_shutdown_owned_job_pollers()
    app = FastAPI()
    counts = iter([2, 1, 0])
    observed_sleeps: list[float] = []
    stop_calls: list[str] = []

    async def _fake_sleep(delay: float) -> None:
        observed_sleeps.append(delay)

    async def _fake_stop_registered_job_pollers(_app: FastAPI, _handles: list[object]) -> None:
        stop_calls.append("stop")

    @contextmanager
    def _fake_timed_shutdown_segment(_app: FastAPI, segment_name: str, **extra: object):
        shutdown_pollers.record_shutdown_timing_segment(app, f"enter:{segment_name}", 0, **extra)
        yield

    await shutdown_pollers.quiesce_owned_job_pollers_for_shutdown(
        app,
        [
            shutdown_pollers.ManagedJobPoller(
                name="poller",
                task=asyncio.get_running_loop().create_future(),
                stop_event=asyncio.Event(),
            )
        ],
        wait_for_leases_sec=5,
        count_active_processing=lambda: next(counts),
        stop_registered_job_pollers=_fake_stop_registered_job_pollers,
        timed_shutdown_segment=_fake_timed_shutdown_segment,
        asyncio_module=type("_AsyncioProxy", (), {"sleep": staticmethod(_fake_sleep)})(),
    )

    assert observed_sleeps == [0.5, 0.5]
    assert stop_calls == ["stop"]
    segments = app.state._tldw_shutdown_timing_segments
    assert segments[0]["segment"] == "optional_lease_wait"
    assert segments[0]["skipped"] is False
    assert segments[0]["initial_active"] == 2
    assert segments[1]["segment"] == "enter:job_poller_quiesce"


@pytest.mark.asyncio
async def test_stop_registered_job_pollers_logs_task_cancel_failure_at_warning() -> None:
    shutdown_pollers = _import_shutdown_owned_job_pollers()
    app = FastAPI()
    debug_messages: list[str] = []
    warning_messages: list[str] = []

    class _FakeTask:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

        def done(self) -> bool:
            return False

        def get_name(self) -> str:
            return "fake-poller"

    class _AsyncioProxy:
        CancelledError = asyncio.CancelledError
        TimeoutError = asyncio.TimeoutError

        @staticmethod
        def shield(task):
            return task

        @staticmethod
        async def wait_for(awaitable, *, timeout):
            del awaitable
            if timeout == 0.01:
                raise asyncio.TimeoutError()
            raise RuntimeError("cancel guard")

        @staticmethod
        async def gather(*awaitables, return_exceptions):
            assert return_exceptions is False
            return [await awaitable for awaitable in awaitables]

    logger_obj = type(
        "_Logger",
        (),
        {
            "debug": lambda _self, message, *args: debug_messages.append(str(message)),
            "warning": lambda _self, message, *args: warning_messages.append(str(message)),
        },
    )()
    task = _FakeTask()

    await shutdown_pollers.stop_registered_job_pollers(
        app,
        [
            shutdown_pollers.ManagedJobPoller(
                name="poller",
                task=task,
                stop_event=None,
                timeout_sec=0.01,
            )
        ],
        logger_obj=logger_obj,
        guard_exceptions=(RuntimeError,),
        asyncio_module=_AsyncioProxy(),
    )

    assert task.cancelled is True
    assert not any("Job poller cancel guard triggered" in message for message in debug_messages)
    assert any("raised after cancellation" in message for message in warning_messages)


@pytest.mark.asyncio
async def test_stop_registered_job_pollers_handles_taskless_worker_without_crashing() -> None:
    shutdown_pollers = _import_shutdown_owned_job_pollers()
    app = FastAPI()

    await shutdown_pollers.stop_registered_job_pollers(
        app,
        [
            shutdown_pollers.ManagedJobPoller(
                name="callback_only_worker",
                task=None,
                stop_event=None,
                timeout_sec=0.01,
            )
        ],
    )

    assert app.state._tldw_shutdown_quiesced_job_poller_names == [
        "callback_only_worker"
    ]
