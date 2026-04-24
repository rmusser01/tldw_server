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
        assert getattr(app.state, "_tldw_shutdown_job_poller_inventory") == [
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
    segments = getattr(app.state, "_tldw_shutdown_timing_segments")
    assert segments[0]["segment"] == "optional_lease_wait"
    assert segments[0]["skipped"] is False
    assert segments[0]["initial_active"] == 2
    assert segments[1]["segment"] == "enter:job_poller_quiesce"
