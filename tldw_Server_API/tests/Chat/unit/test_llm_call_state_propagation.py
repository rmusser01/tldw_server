"""The LLM call state reaches pool threads and queued processors (LLM usage accounting spec §1)."""

import asyncio
from typing import Any

import pytest

from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool, await_bounded_sync_call
from tldw_Server_API.app.core.Chat.request_queue import RequestPriority, RequestQueue
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.Usage.llm_call_context import (
    current_llm_call_context,
    current_llm_user_id,
    llm_call_context,
)

pytestmark = pytest.mark.unit


def _probe() -> tuple[int | None, bool, bool]:
    """What the current thread sees."""
    context = current_llm_call_context()
    return current_llm_user_id(), context.records_itself, context.exempt


async def test_a_pool_thread_sees_the_starter_s_call_state() -> None:
    """A bounded sync call sees the scope user and the flags of the code that started it."""
    pool = BoundedDaemonPool(capacity=2)
    with scoped_context(user_id=6), llm_call_context(records_itself=True):
        marked = await await_bounded_sync_call(_probe, pool=pool, exhaustion_message="full")
    bare = await await_bounded_sync_call(_probe, pool=pool, exhaustion_message="full")
    assert marked == (6, True, False)
    assert bare == (None, False, False)


async def test_a_queued_processor_runs_with_the_enqueuer_s_call_state() -> None:
    """The queue's workers start with no call state; the processor still sees the enqueuer's."""
    queue = RequestQueue(max_queue_size=4, max_concurrent=1)
    await queue.start(num_workers=1)
    try:
        with llm_call_context(user_id=4, records_itself=True):
            marked: asyncio.Future[Any] = await queue.enqueue(
                "req-marked", {}, "client", RequestPriority.NORMAL, processor=_probe
            )
        bare: asyncio.Future[Any] = await queue.enqueue(
            "req-bare", {}, "client", RequestPriority.NORMAL, processor=_probe
        )
        assert await asyncio.wait_for(marked, timeout=5) == (4, True, False)
        assert await asyncio.wait_for(bare, timeout=5) == (None, False, False)
    finally:
        await queue.stop()


async def test_a_queued_stream_factory_runs_with_the_enqueuer_s_call_state() -> None:
    """A streaming request's factory also runs with the enqueuer's state."""
    queue = RequestQueue(max_queue_size=4, max_concurrent=1)
    await queue.start(num_workers=1)
    seen: list[tuple[int | None, bool, bool]] = []

    def _factory() -> Any:
        """Record the state, then return an empty stream."""
        seen.append(_probe())
        return iter(())

    channel: asyncio.Queue[Any] = asyncio.Queue()
    bare_channel: asyncio.Queue[Any] = asyncio.Queue()
    try:
        with llm_call_context(user_id=4, records_itself=True):
            future = await queue.enqueue(
                "req-stream", {}, "client", RequestPriority.NORMAL,
                processor=_factory, streaming=True, stream_channel=channel,
            )
        bare_future = await queue.enqueue(
            "req-stream-bare", {}, "client", RequestPriority.NORMAL,
            processor=_factory, streaming=True, stream_channel=bare_channel,
        )
        await asyncio.wait_for(asyncio.shield(future), timeout=5)
        await asyncio.wait_for(asyncio.shield(bare_future), timeout=5)
    finally:
        await queue.stop()
    assert seen == [(4, True, False), (None, False, False)]
