import base64

import pytest

from tldw_Server_API.app.core.Utils import cpu_bound_handler as cbh
from tldw_Server_API.app.core.Utils.cpu_bound_handler import (
    CPUBoundBatcher,
    _process_pool_disabled,
    decode_large_base64_async,
    json_encode_heavy,
    process_large_json_async,
    run_cpu_bound_thread,
)


@pytest.mark.asyncio
async def test_process_large_json_async_consistent_encoding():
    small = {"text": "café"}
    large = ["café"] * 200

    assert await process_large_json_async(small) == json_encode_heavy(small)
    assert await process_large_json_async(large) == json_encode_heavy(large)


@pytest.mark.asyncio
async def test_decode_large_base64_async_strips_whitespace():
    data = b"hello world"
    b64 = base64.b64encode(data).decode("ascii")
    spaced = f"{b64[:4]} \n{b64[4:]}"

    decoded = await decode_large_base64_async(spaced)

    assert decoded == data


@pytest.mark.asyncio
async def test_run_cpu_bound_thread_accepts_kwargs():
    def combine(prefix: str, *, suffix: str) -> str:
        return f"{prefix}-{suffix}"

    result = await run_cpu_bound_thread(combine, "left", suffix="right")

    assert result == "left-right"


def test_process_pool_disabled_accepts_testing_y(monkeypatch):
    monkeypatch.delenv("TLDR_DISABLE_CPU_PROCPOOL", raising=False)
    monkeypatch.setenv("TESTING", "y")
    assert _process_pool_disabled() is True


@pytest.mark.asyncio
async def test_cpu_bound_batcher_reschedules_operations_added_while_batch_is_draining(monkeypatch):
    started = cbh.asyncio.Event()
    release = cbh.asyncio.Event()

    async def fake_run_cpu_bound_thread(func, *args, **kwargs):
        if args and args[0] == "first":
            started.set()
            await release.wait()
        return func(*args, **kwargs)

    monkeypatch.setattr(cbh, "run_cpu_bound_thread", fake_run_cpu_bound_thread)

    batcher = CPUBoundBatcher(batch_size=10, timeout=0.01)

    def identity(value: str) -> str:
        return value

    first = cbh.asyncio.create_task(batcher.add_operation(identity, "first"))
    await cbh.asyncio.wait_for(started.wait(), timeout=0.5)

    second = cbh.asyncio.create_task(batcher.add_operation(identity, "second"))
    await cbh.asyncio.sleep(0)
    release.set()

    results = await cbh.asyncio.wait_for(cbh.asyncio.gather(first, second), timeout=0.5)

    assert results == ["first", "second"]


@pytest.mark.asyncio
async def test_cpu_bound_batcher_keeps_rescheduled_task_reference_after_full_batch(monkeypatch):
    started = cbh.asyncio.Event()
    release = cbh.asyncio.Event()

    async def fake_run_cpu_bound_thread(func, *args, **kwargs):
        if args and args[0] == "first":
            started.set()
            await release.wait()
        return func(*args, **kwargs)

    monkeypatch.setattr(cbh, "run_cpu_bound_thread", fake_run_cpu_bound_thread)

    batcher = CPUBoundBatcher(batch_size=2, timeout=60)

    def identity(value: str) -> str:
        return value

    first = cbh.asyncio.create_task(batcher.add_operation(identity, "first"))
    await cbh.asyncio.sleep(0)
    second = cbh.asyncio.create_task(batcher.add_operation(identity, "second"))
    await cbh.asyncio.wait_for(started.wait(), timeout=0.5)
    third = cbh.asyncio.create_task(batcher.add_operation(identity, "third"))
    await cbh.asyncio.sleep(0)

    release.set()
    await cbh.asyncio.wait_for(cbh.asyncio.gather(first, second), timeout=0.5)

    try:
        assert not third.done()
        assert batcher._batch_task is not None
        assert not batcher._batch_task.done()
    finally:
        if batcher._batch_task is not None and not batcher._batch_task.done():
            batcher._batch_task.cancel()
            try:
                await batcher._batch_task
            except cbh.asyncio.CancelledError:
                pass
            batcher._batch_task = None
        if not third.done():
            await batcher._process_batch(delay=False)

    assert await cbh.asyncio.wait_for(third, timeout=0.5) == "third"
