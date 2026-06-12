import asyncio
import threading

import pytest

from tldw_Server_API.app.core.Chat.request_queue import RequestQueue, RequestPriority


@pytest.mark.asyncio
async def test_concurrent_processing_non_streaming():
    q = RequestQueue(max_queue_size=10, max_concurrent=2, timeout=5)
    await q.start(num_workers=2)

    both_started = threading.Event()
    release_processors = threading.Event()
    lock = threading.Lock()
    active_processors = 0
    peak_active_processors = 0
    started_processors = 0

    async def submit_job(idx: int):
        def proc():
            nonlocal active_processors, peak_active_processors, started_processors

            with lock:
                active_processors += 1
                peak_active_processors = max(peak_active_processors, active_processors)
                started_processors += 1
                if started_processors == 2:
                    both_started.set()

            release_processors.wait(timeout=2)

            with lock:
                active_processors -= 1
            return {"idx": idx}

        fut = await q.enqueue(
            request_id=f"r{idx}",
            request_data={"test": True},
            client_id="u1",
            priority=RequestPriority.NORMAL,
            estimated_tokens=1,
            processor=proc,
            streaming=False,
        )
        return await fut

    tasks = [asyncio.create_task(submit_job(1)), asyncio.create_task(submit_job(2))]

    try:
        started_in_time = await asyncio.to_thread(both_started.wait, 2)
        release_processors.set()
        res = await asyncio.gather(*tasks)

        assert started_in_time, "Queue workers did not start both processors concurrently"
        assert peak_active_processors == 2
        assert all("idx" in r for r in res)
    finally:
        release_processors.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await q.stop()


@pytest.mark.asyncio
async def test_streaming_job_pumps_and_done():
    q = RequestQueue(max_queue_size=10, max_concurrent=1, timeout=5)
    await q.start(num_workers=1)

    def streaming_proc():

        def gen():
            yield "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n"
            yield "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n"
            yield "data: [DONE]\n\n"
        return gen()

    ch: asyncio.Queue = asyncio.Queue(maxsize=10)
    fut = await q.enqueue(
        request_id="s1",
        request_data={"stream": True},
        client_id="u1",
        priority=RequestPriority.HIGH,
        estimated_tokens=1,
        processor=streaming_proc,
        streaming=True,
        stream_channel=ch,
    )

    # Consume channel until sentinel None
    received = []
    while True:
        item = await ch.get()
        if item is None:
            break
        received.append(item)

    # Future result should resolve after pumping is complete
    result = await fut
    assert result.get("status") == "stream_completed"
    assert any("data: [DONE]" in x for x in received)
    assert len(received) >= 3

    await q.stop()


@pytest.mark.asyncio
async def test_streaming_processor_error_emits_error_and_done():
    q = RequestQueue(max_queue_size=10, max_concurrent=1, timeout=5)
    await q.start(num_workers=1)

    def failing_proc():

        raise RuntimeError("boom")

    ch: asyncio.Queue = asyncio.Queue(maxsize=10)
    fut = await q.enqueue(
        request_id="s2",
        request_data={"stream": True},
        client_id="u1",
        priority=RequestPriority.HIGH,
        estimated_tokens=1,
        processor=failing_proc,
        streaming=True,
        stream_channel=ch,
    )

    # Expect error frames then done; future should raise
    items = []
    while True:
        item = await ch.get()
        items.append(item)
        if item is None:
            break

    assert any((isinstance(x, str) and "error" in x) for x in items if x is not None)
    assert any((isinstance(x, str) and "data: [DONE]" in x) for x in items if x is not None)

    with pytest.raises(Exception):
        await fut

    await q.stop()


@pytest.mark.asyncio
async def test_priority_preempts_backlog():
    q = RequestQueue(max_queue_size=10, max_concurrent=1, timeout=5)
    await q.start(num_workers=1)

    async def submit_low(idx: int):
        def proc():
            time.sleep(0.2)
            return {"id": f"L{idx}", "t": time.time()}

        fut = await q.enqueue(
            request_id=f"low{idx}",
            request_data={"t": idx},
            client_id="c1",
            priority=RequestPriority.LOW,
            estimated_tokens=1,
            processor=proc,
            streaming=False,
        )
        return await fut

    async def submit_high():
        def proc():
            time.sleep(0.01)
            return {"id": "H", "t": time.time()}

        fut = await q.enqueue(
            request_id="high",
            request_data={"t": 0},
            client_id="c2",
            priority=RequestPriority.HIGH,
            estimated_tokens=1,
            processor=proc,
            streaming=False,
        )
        return await fut

    low1 = asyncio.create_task(submit_low(1))
    low2 = asyncio.create_task(submit_low(2))
    low3 = asyncio.create_task(submit_low(3))
    # Let first low start
    await asyncio.sleep(0.05)
    high = await submit_high()
    l1 = await low1
    l2 = await low2
    l3 = await low3

    # High should finish before the tail of the low backlog
    assert high["t"] < l3["t"], f"High priority job did not preempt backlog: high={high['t']}, l3={l3['t']}"

    await q.stop()


@pytest.mark.asyncio
async def test_streaming_sequence_preserved():
    q = RequestQueue(max_queue_size=10, max_concurrent=1, timeout=5)
    await q.start(num_workers=1)

    def gen():

        yield ": heartbeat 1\n\n"
        yield "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n"
        yield ": heartbeat 2\n\n"
        yield "data: [DONE]\n\n"
    def streaming_proc():
        return gen()

    ch: asyncio.Queue = asyncio.Queue(maxsize=10)
    await q.enqueue(
        request_id="s3",
        request_data={},
        client_id="u",
        priority=RequestPriority.NORMAL,
        estimated_tokens=1,
        processor=streaming_proc,
        streaming=True,
        stream_channel=ch,
    )

    items = []
    while True:
        item = await ch.get()
        if item is None:
            break
        items.append(item)

    assert items[0].startswith(": heartbeat")
    assert "data: [DONE]" in items[-1]

    await q.stop()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_multi_client_alternating_no_starvation():
    q = RequestQueue(max_queue_size=10, max_concurrent=2, timeout=5)
    await q.start(num_workers=2)

    completed = []

    async def submit(client: str, idx: int):
        def proc():
            time.sleep(0.05)
            return {"client": client, "idx": idx, "t": time.time()}

        fut = await q.enqueue(
            request_id=f"{client}-{idx}",
            request_data={"i": idx},
            client_id=client,
            priority=RequestPriority.NORMAL,
            estimated_tokens=1,
            processor=proc,
            streaming=False,
        )
        res = await fut
        completed.append(res)

    # Enqueue alternating clients
    await asyncio.gather(
        submit("c1", 1),
        submit("c2", 1),
        submit("c1", 2),
        submit("c2", 2),
    )

    # Should contain both clients and no starvation
    assert any(r["client"] == "c1" for r in completed)
    assert any(r["client"] == "c2" for r in completed)
    # Ordering across equal-priority clients is not guaranteed; ensure both got service
    counts = {c: sum(1 for r in completed if r["client"] == c) for c in ("c1", "c2")}
    assert counts["c1"] > 0 and counts["c2"] > 0

    await q.stop()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_two_streams_interleaved_heartbeats_arrive():
    q = RequestQueue(max_queue_size=10, max_concurrent=2, timeout=5)
    await q.start(num_workers=2)

    # Fast heartbeat streams
    def stream_gen(tag):
        def gen():
            yield f": hb {tag}1\n\n"
            time.sleep(0.02)
            yield f"data: {{\"choices\":[{{\"delta\":{{\"content\":\"{tag}x\"}}}}]}}\n\n"
            time.sleep(0.02)
            yield "data: [DONE]\n\n"
        return gen()

    ch1: asyncio.Queue = asyncio.Queue(maxsize=10)
    ch2: asyncio.Queue = asyncio.Queue(maxsize=10)

    await q.enqueue(
        request_id="sA",
        request_data={},
        client_id="c1",
        priority=RequestPriority.NORMAL,
        estimated_tokens=1,
        processor=lambda: stream_gen("A"),
        streaming=True,
        stream_channel=ch1,
    )
    await q.enqueue(
        request_id="sB",
        request_data={},
        client_id="c2",
        priority=RequestPriority.NORMAL,
        estimated_tokens=1,
        processor=lambda: stream_gen("B"),
        streaming=True,
        stream_channel=ch2,
    )

    async def next_non_none(qch):
        while True:
            it = await qch.get()
            if it is None:
                return None
            return it

    # Expect first chunk (heartbeat) from both streams in a small window
    t0 = time.time()
    f1 = await next_non_none(ch1)
    f2 = await next_non_none(ch2)
    elapsed = time.time() - t0
    assert f1 is not None and f1.startswith(": hb")
    assert f2 is not None and f2.startswith(": hb")
    assert elapsed < 0.2

    await q.stop()
