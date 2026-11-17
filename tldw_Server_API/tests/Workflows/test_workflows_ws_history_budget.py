"""Cooperative history work shares a budget across senders on the same loop."""

import asyncio
import gc
import weakref
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints import workflows


@pytest.mark.asyncio
async def test_history_budget_bounds_combined_work_and_rechecks_after_waking():
    loop = asyncio.get_running_loop()
    clock = SimpleNamespace(now=0.0)
    checkpoints = []

    def schedule(delay, callback):
        assert delay == 0

        def resume():
            checkpoints.append(clock.now)
            callback()

        loop.call_soon(resume)

    budget = workflows._WorkflowHistoryBudget(
        SimpleNamespace(time=lambda: clock.now, create_future=loop.create_future, call_later=schedule)
    )
    delivered = [[] for _ in range(16)]

    async def sender(index):
        for sequence in range(20):
            await budget.checkpoint()
            delivered[index].append(sequence)
            clock.now += 0.0004

    await asyncio.gather(*(sender(index) for index in range(16)))

    assert all(sequences == list(range(20)) for sequences in delivered)
    assert len(checkpoints) > 50
    assert max(after - before for before, after in zip(checkpoints, checkpoints[1:])) <= 0.001401


@pytest.mark.asyncio
async def test_history_budget_cancelled_waiter_does_not_cancel_other_senders():
    loop = asyncio.get_running_loop()
    callbacks = []
    budget = workflows._WorkflowHistoryBudget(
        SimpleNamespace(
            time=lambda: 0.0, create_future=loop.create_future, call_later=lambda delay, cb: callbacks.append(cb)
        )
    )
    first = asyncio.create_task(budget.checkpoint())
    second = asyncio.create_task(budget.checkpoint())
    await asyncio.sleep(0)
    try:
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert len(callbacks) == 1
        assert not second.done()
        callbacks.pop()()
        await asyncio.wait_for(second, 1)
    finally:
        for task in (first, second):
            task.cancel()
        await asyncio.gather(first, second, return_exceptions=True)


def test_history_budgets_are_shared_per_loop_and_release_unused_state():
    retained = []

    async def connection_pair():
        first = workflows._workflow_history_budget()
        assert workflows._workflow_history_budget() is first
        retained.append(first)

    asyncio.run(connection_pair())
    asyncio.run(connection_pair())
    assert retained[0] is not retained[1]
    refs = [weakref.ref(budget) for budget in retained]
    retained.clear()
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert not workflows._WORKFLOW_HISTORY_BUDGETS


@pytest.mark.asyncio
async def test_history_checkpoint_resumes_senders_without_an_extra_ready_generation():
    loop = asyncio.get_running_loop()
    callbacks, resumed = [], []
    budget = workflows._WorkflowHistoryBudget(
        SimpleNamespace(
            time=lambda: 0.0, create_future=loop.create_future, call_later=lambda delay, cb: callbacks.append(cb)
        )
    )
    sender = asyncio.create_task(budget.checkpoint())
    try:
        await asyncio.sleep(0)
        callbacks.pop()()
        loop.call_soon(lambda: resumed.append(sender.done()))
        await sender
        assert resumed == [True], "An extra callback generation delayed the sender after the checkpoint"
    finally:
        sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)
