"""ChaCha shutdown must not wait on default-character work stranded on a closed loop."""

from __future__ import annotations

import asyncio
import time

import pytest

from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as deps

pytestmark = pytest.mark.unit


def _strand(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    """Register an unfinished task/future on a loop that is then closed."""
    monkeypatch.setattr(deps, "_chacha_default_char_tasks", set())
    monkeypatch.setattr(deps, "_chacha_default_char_futures", set())

    async def register() -> None:
        if kind == "task":
            task = asyncio.create_task(asyncio.sleep(3600))
            deps._chacha_default_char_tasks.add(task)
            task.add_done_callback(deps._chacha_default_char_tasks.discard)
        else:
            deps._track_default_character_future(asyncio.get_running_loop().create_future())

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
    _strand(monkeypatch, kind)

    async def drain() -> None:
        await deps._drain_default_character_tasks(timeout=2.0)
        await deps._drain_default_character_futures(timeout=2.0)

    started = time.monotonic()
    asyncio.run(drain())

    assert time.monotonic() - started < 1.0, "the drain waited on work that can never complete"
    assert not deps._chacha_default_char_tasks
    assert not deps._chacha_default_char_futures
