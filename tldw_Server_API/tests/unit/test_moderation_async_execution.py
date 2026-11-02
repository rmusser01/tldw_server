"""Transport-only offload contract, independent of future incident reporting."""

import asyncio
import contextvars
import threading
import time

import pytest

from tldw_Server_API.app.core.Moderation.execution_models import ExecutionScope

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def scope():
    return ExecutionScope(time.monotonic() + 2, threading.Event(), "async")


async def test_callable_timeout_error_is_not_misreported_as_transport_deadline():
    from tldw_Server_API.app.core.Moderation.async_execution import _run_moderation_offload

    shared = scope()

    def call():
        raise TimeoutError("trusted override timeout")

    with pytest.raises(TimeoutError, match="trusted override timeout"):
        await _run_moderation_offload(call, scope=shared)
    assert not shared.cancelled.is_set()


async def test_async_heartbeat_runs_while_worker_stalls():
    from tldw_Server_API.app.core.Moderation.async_execution import _run_moderation_offload

    release = threading.Event()
    task = asyncio.create_task(_run_moderation_offload(lambda: release.wait(1), scope=scope()))
    try:
        for _ in range(5):
            await asyncio.sleep(0.01)
        assert not task.done()
    finally:
        release.set()
        assert await task


async def test_cancelled_offload_signals_owner():
    from tldw_Server_API.app.core.Moderation.async_execution import _run_moderation_offload

    shared, entered, finished = scope(), threading.Event(), threading.Event()

    def call():
        entered.set()
        shared.cancelled.wait(2)
        finished.set()

    task = asyncio.create_task(_run_moderation_offload(call, scope=shared))
    while not entered.is_set():
        await asyncio.sleep(0.001)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert shared.cancelled.is_set()
    assert await asyncio.to_thread(finished.wait, 1)


async def test_context_does_not_leak_between_concurrent_calls():
    from tldw_Server_API.app.core.Moderation.async_execution import _run_moderation_offload

    current = contextvars.ContextVar("independent")

    async def one(value):
        token = current.set(value)
        try:
            return await _run_moderation_offload(current.get, scope=scope())
        finally:
            current.reset(token)

    assert await asyncio.gather(one("first"), one("second")) == ["first", "second"]


async def test_cancelled_offloads_retain_four_thread_slots_until_finished():
    from tldw_Server_API.app.core.exception_types import ModerationExecutionError
    from tldw_Server_API.app.core.Moderation.async_execution import _run_moderation_offload

    release = threading.Event()
    entered = [threading.Event() for _ in range(4)]

    def call(index):
        entered[index].set()
        release.wait(2)

    tasks = [asyncio.create_task(_run_moderation_offload(lambda i=i: call(i), scope=scope())) for i in range(4)]
    try:
        while not all(event.is_set() for event in entered):
            await asyncio.sleep(0.001)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        with pytest.raises(ModerationExecutionError, match="capacity_exhausted"):
            await _run_moderation_offload(lambda: None, scope=scope())
    finally:
        release.set()
        await asyncio.sleep(0.05)


async def test_quarantined_injected_runner_keeps_offload_admission_owned():
    from tldw_Server_API.app.core.exception_types import ModerationExecutionError
    from tldw_Server_API.app.core.Moderation.async_execution import _run_moderation_offload
    from tldw_Server_API.tests.unit.test_moderation_execution_runner import Child, request, runner

    children = [Child(stalled=True, unreapable=True) for _ in range(4)]
    owners = [runner(lambda *a, child=child, **k: child) for child in children]
    scopes = [scope() for _ in range(4)]
    for shared in scopes:
        shared.deadline = time.monotonic() + 0.05
    tasks = [
        asyncio.create_task(
            _run_moderation_offload(
                lambda owner=owner, shared=shared: owner.execute(request(), shared),
                scope=shared,
            )
        )
        for owner, shared in zip(owners, scopes, strict=True)
    ]
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(0.65)
        with pytest.raises(ModerationExecutionError, match="capacity_exhausted"):
            await _run_moderation_offload(lambda: None, scope=scope())
    finally:
        for child in children:
            child.returncode = -9
        for owner in owners:
            owner.close()
        await asyncio.sleep(0.1)


@pytest.mark.timeout(30)
async def test_real_worker_heartbeat_in_independent_containment(tmp_path):
    import sys
    from pathlib import Path

    from tldw_Server_API.tests.helpers.moderation_process_harness import run_isolated_probe
    from tldw_Server_API.tests.unit.test_moderation_process_harness import (
        test_outer_harness_timeout_kills_probe_and_grandchild,
        test_probe_hard_exit_and_invoker_disconnect_still_clean_tree,
    )

    test_outer_harness_timeout_kills_probe_and_grandchild(tmp_path)
    test_probe_hard_exit_and_invoker_disconnect_still_clean_tree(tmp_path)
    probe = Path(__file__).resolve().parents[1] / "helpers" / "moderation_supervisor_probe.py"
    result = run_isolated_probe([sys.executable, "-I", "-S", str(probe), "heartbeat"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith(b"OK heartbeat")
