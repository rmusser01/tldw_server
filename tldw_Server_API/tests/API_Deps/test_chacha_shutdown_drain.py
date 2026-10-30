"""ChaCha shutdown must not wait on default-character work stranded on a closed loop."""

from __future__ import annotations

import asyncio
import time

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.runtime import ChaChaRuntimeManager, get_chacha_runtime

pytestmark = pytest.mark.unit


def _strand(monkeypatch: pytest.MonkeyPatch, runtime: ChaChaRuntimeManager, kind: str) -> None:
    """Register an unfinished task/future on a loop that is then closed."""
    monkeypatch.setattr(runtime, "_tasks", set())
    monkeypatch.setattr(runtime, "_futures", set())

    async def register() -> None:
        if kind == "task":
            runtime.track(asyncio.create_task(asyncio.sleep(3600)))
        else:
            runtime.track(asyncio.get_running_loop().create_future())

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(register())
    finally:
        loop.close()


@pytest.mark.parametrize("kind", ["task", "future"])
def test_drain_drops_work_stranded_on_a_closed_loop(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    """Such work can never finish; waiting cost the full timeout on every shutdown.

    The test conftest drains after every test, so one stranded future made each
    later test wait 3 x 5s -- enough to push the db-privileges CI shard past its
    60-minute job timeout.
    """
    runtime = get_chacha_runtime()
    _strand(monkeypatch, runtime, kind)

    async def drain() -> None:
        await runtime._drain_tasks(timeout=2.0)
        await runtime._drain_futures(timeout=2.0)

    started = time.monotonic()
    asyncio.run(drain())

    assert time.monotonic() - started < 1.0, "the drain waited on work that can never complete"
    assert not runtime._tasks
    assert not runtime._futures
