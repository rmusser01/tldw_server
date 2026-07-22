import asyncio
import threading
import time

import pytest
from loguru import logger

from tldw_Server_API.app.core.Chat import bounded_daemon as bounded_daemon_module
from tldw_Server_API.app.core.Chat import request_queue as request_queue_module
from tldw_Server_API.app.core.Chat.bounded_daemon import (
    BoundedDaemonPool,
    DaemonCapacityError,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import SanitizedProviderStreamError
from tldw_Server_API.app.core.Chat.request_queue import (
    QueuedRequest,
    RateLimitedQueue,
    RequestPriority,
    RequestQueue,
)


@pytest.mark.asyncio
async def test_request_queue_rate_limit_rollback_on_admission_failure():
    queue = RateLimitedQueue(
        max_queue_size=0,
        max_concurrent=1,
        global_rate_limit=5,
        per_client_rate_limit=5,
    )

    with pytest.raises(ValueError):
        await queue.enqueue(
            request_id="req-1",
            request_data={"endpoint": "/api/v1/chat/completions"},
            client_id="client-1",
            priority=RequestPriority.NORMAL,
        )

    assert queue.global_request_times == []
    assert queue.client_request_times.get("client-1") in (None, [])


@pytest.mark.asyncio
async def test_zero_budget_queue_operation_never_schedules_coroutine_body():
    body_ran = False

    async def body():
        nonlocal body_ran
        body_ran = True

    with pytest.raises(asyncio.TimeoutError):
        await request_queue_module._await_queue_stream_operation(body(), timeout=0)
    await asyncio.sleep(0)

    assert body_ran is False


class _CountingAsyncIterator:
    def __init__(self):
        self.count = 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.count += 1
        await asyncio.sleep(0)
        return f"chunk-{self.count}"

    async def aclose(self):
        self.closed = True


class _UnboundedAsyncIterator:
    def __init__(self):
        self.closed = False
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self._i += 1
        await asyncio.sleep(0)
        return f"chunk-{self._i}"

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_request_queue_stream_cancel_stops_async_iterator():
    queue = RequestQueue(max_queue_size=10, max_concurrent=1)
    # _process_request checks the running flag; set it explicitly for this unit test
    queue._running = True
    stream_channel: asyncio.Queue = asyncio.Queue(maxsize=10)
    async_iter = _CountingAsyncIterator()

    def processor():
        return async_iter

    future: asyncio.Future = asyncio.Future()
    request = QueuedRequest(
        priority=RequestPriority.HIGH.value,
        timestamp=0.0,
        request_id="stream-1",
        request_data={},
        future=future,
        client_id="client-1",
        estimated_tokens=0,
        processor=processor,
        processor_args=(),
        processor_kwargs={},
        streaming=True,
        stream_channel=stream_channel,
    )

    task = asyncio.create_task(queue._process_request(request))

    first = await stream_channel.get()
    assert str(first).startswith("chunk-")

    future.cancel()

    # Drain until sentinel
    while True:
        item = await stream_channel.get()
        if item is None:
            break

    await task

    assert async_iter.closed is True
    assert async_iter.count <= 2

@pytest.mark.asyncio
async def test_request_queue_stream_cancel_with_full_channel_exits():
    queue = RequestQueue(max_queue_size=10, max_concurrent=1)
    queue._running = True
    stream_channel: asyncio.Queue = asyncio.Queue(maxsize=1)
    await stream_channel.put("prefill")
    async_iter = _UnboundedAsyncIterator()

    def processor():
        return async_iter

    future: asyncio.Future = asyncio.Future()
    request = QueuedRequest(
        priority=RequestPriority.HIGH.value,
        timestamp=0.0,
        request_id="stream-full-1",
        request_data={},
        future=future,
        client_id="client-1",
        estimated_tokens=0,
        processor=processor,
        processor_args=(),
        processor_kwargs={},
        streaming=True,
        stream_channel=stream_channel,
    )

    task = asyncio.create_task(queue._process_request(request))

    await asyncio.sleep(0.05)
    future.cancel()
    while True:
        try:
            stream_channel.get_nowait()
        except asyncio.QueueEmpty:
            break

    await asyncio.wait_for(task, timeout=1.5)
    assert async_iter.closed is True

@pytest.mark.asyncio
async def test_request_queue_noncooperative_async_stream_cleanup_is_bounded_and_isolated(
    monkeypatch,
):
    release = asyncio.Event()
    blocked = asyncio.Event()
    healthy_closed = asyncio.Event()
    sentinel = "queued-cleanup-secret-/srv/provider"

    class ResistantIterator:
        def __init__(self):
            self.first = True

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.first:
                self.first = False
                return "queued output"
            blocked.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
                raise StopAsyncIteration from None

        async def aclose(self):
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            raise RuntimeError(sentinel)

    class HealthyIterator:
        def __init__(self):
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return "healthy queued output"

        async def aclose(self):
            healthy_closed.set()

    def queued_request(request_id, iterator):
        return QueuedRequest(
            priority=RequestPriority.HIGH.value,
            timestamp=0.0,
            request_id=request_id,
            request_data={},
            future=asyncio.Future(),
            client_id=request_id,
            estimated_tokens=0,
            processor=lambda: iterator,
            processor_args=(),
            processor_kwargs={},
            streaming=True,
            stream_channel=asyncio.Queue(maxsize=10),
        )

    monkeypatch.setattr(
        request_queue_module,
        "REQUEST_QUEUE_STREAM_CANCEL_DRAIN_SECONDS",
        0.01,
        raising=False,
    )
    monkeypatch.setattr(
        request_queue_module,
        "REQUEST_QUEUE_STREAM_CLEANUP_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )
    queue = RequestQueue(max_queue_size=10, max_concurrent=2)
    queue._running = True
    resistant_request = queued_request("resistant", ResistantIterator())
    healthy_request = queued_request("healthy", HealthyIterator())
    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    resistant_task = asyncio.create_task(queue._process_request(resistant_request))
    assert await resistant_request.stream_channel.get() == "queued output"
    await asyncio.wait_for(blocked.wait(), 1.0)
    healthy_task = asyncio.create_task(queue._process_request(healthy_request))
    resistant_request.future.cancel()

    try:
        healthy_result = await asyncio.wait_for(
            asyncio.shield(healthy_task),
            timeout=1.0,
        )
        resistant_result = await asyncio.wait_for(
            asyncio.shield(resistant_task),
            timeout=1.0,
        )
        assert not release.is_set()
    finally:
        release.set()
        logger.remove(sink_id)

    assert resistant_result["status"] == "stream_completed"
    assert healthy_result["status"] == "stream_completed"
    assert await healthy_request.stream_channel.get() == "healthy queued output"
    assert healthy_closed.is_set()
    assert sentinel not in "".join(logs)


@pytest.mark.asyncio
async def test_raw_sync_stream_block_after_output_does_not_pin_queue_or_shutdown():
    next_blocked = threading.Event()
    release_next = threading.Event()
    blocked_closed = threading.Event()
    sentinel = "raw-sync-queue-close-secret-/srv/provider"
    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")

    class BlockingAfterFirstChunk:
        def __init__(self):
            self.first = True

        def __iter__(self):
            return self

        def __next__(self):
            if self.first:
                self.first = False
                return "first raw sync output"
            next_blocked.set()
            release_next.wait()
            raise StopIteration

        def close(self):
            blocked_closed.set()
            raise RuntimeError(sentinel)

    class HealthySyncIterator:
        def __init__(self):
            self.sent = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.sent:
                raise StopIteration
            self.sent = True
            return "healthy raw sync output"

    queue = RequestQueue(max_queue_size=10, max_concurrent=1, timeout=1.0)
    await queue.start(num_workers=2)
    blocked_channel: asyncio.Queue = asyncio.Queue(maxsize=10)
    healthy_channel: asyncio.Queue = asyncio.Queue(maxsize=10)
    stop_task = None
    try:
        blocked_future = await queue.enqueue(
            request_id="raw-sync-blocked",
            request_data={},
            client_id="blocked",
            processor=BlockingAfterFirstChunk,
            streaming=True,
            stream_channel=blocked_channel,
        )
        assert await asyncio.wait_for(blocked_channel.get(), 1.0) == "first raw sync output"
        assert await asyncio.to_thread(next_blocked.wait, 1.0)
        blocked_future.cancel()

        healthy_future = await queue.enqueue(
            request_id="raw-sync-healthy",
            request_data={},
            client_id="healthy",
            processor=HealthySyncIterator,
            streaming=True,
            stream_channel=healthy_channel,
        )
        assert await asyncio.wait_for(healthy_channel.get(), 1.0) == "healthy raw sync output"
        assert await asyncio.wait_for(healthy_channel.get(), 1.0) is None
        assert (await asyncio.wait_for(asyncio.shield(healthy_future), 1.0))["status"] == (
            "stream_completed"
        )

        assert not release_next.is_set()
        stop_task = asyncio.create_task(queue.stop())
        await asyncio.wait_for(asyncio.shield(stop_task), timeout=1.0)
        assert not release_next.is_set()
    finally:
        release_next.set()
        if stop_task is not None and not stop_task.done():
            await asyncio.wait_for(stop_task, timeout=1.0)
        elif not queue._stopped:
            await asyncio.wait_for(queue.stop(), timeout=1.0)
        assert await asyncio.to_thread(blocked_closed.wait, 1.0)
        logger.remove(sink_id)

    assert sentinel not in "".join(logs)


@pytest.mark.asyncio
async def test_sync_daemon_capacity_rejects_extra_work_and_keeps_executor_healthy(
    monkeypatch,
):
    release = threading.Event()
    blocked_started = threading.Event()
    blocked_finished = threading.Event()
    extra_started = threading.Event()
    normal_executor_started = threading.Event()

    def blocked_factory():
        blocked_started.set()
        release.wait()
        blocked_finished.set()
        return iter(())

    def extra_factory():
        extra_started.set()
        return iter(("capacity bypass output",))

    def normal_processor():
        normal_executor_started.set()
        return "normal executor healthy"

    pool = BoundedDaemonPool(capacity=1)
    monkeypatch.setattr(bounded_daemon_module, "STREAM_DAEMON_POOL", pool)
    queue = RequestQueue(max_queue_size=10, max_concurrent=1, timeout=1.0)
    await queue.start(num_workers=2)
    blocked_channel: asyncio.Queue = asyncio.Queue(maxsize=10)
    rejected_channel: asyncio.Queue = asyncio.Queue(maxsize=10)
    stop_task = None
    try:
        blocked_future = await queue.enqueue(
            request_id="daemon-capacity-blocked",
            request_data={},
            client_id="blocked",
            processor=blocked_factory,
            streaming=True,
            stream_channel=blocked_channel,
        )
        assert await asyncio.to_thread(blocked_started.wait, 1.0)
        blocked_future.cancel()

        rejected_future = await queue.enqueue(
            request_id="daemon-capacity-rejected",
            request_data={},
            client_id="rejected",
            processor=extra_factory,
            streaming=True,
            stream_channel=rejected_channel,
        )
        rejected_terminal = await asyncio.wait_for(
            rejected_channel.get(),
            timeout=1.0,
        )
        with pytest.raises(DaemonCapacityError):
            await asyncio.wait_for(asyncio.shield(rejected_future), timeout=1.0)

        normal_future = await queue.enqueue(
            request_id="daemon-capacity-normal",
            request_data={},
            client_id="normal",
            processor=normal_processor,
            streaming=False,
        )
        assert await asyncio.wait_for(normal_future, timeout=1.0) == "normal executor healthy"
        assert normal_executor_started.is_set()
        assert pool.active_count == 1
        assert not extra_started.is_set()
        assert not release.is_set()
        assert isinstance(
            rejected_terminal,
            request_queue_module.QueueStreamTerminalError,
        )
        assert rejected_terminal.code == "provider_unavailable"
        assert "capacity bypass output" not in repr(rejected_terminal)

        stop_task = asyncio.create_task(queue.stop())
        await asyncio.wait_for(asyncio.shield(stop_task), timeout=1.0)
        assert not release.is_set()
    finally:
        release.set()
        assert await asyncio.to_thread(blocked_finished.wait, 1.0)
        if stop_task is not None and not stop_task.done():
            await asyncio.wait_for(stop_task, timeout=1.0)
        elif not queue._stopped:
            await asyncio.wait_for(queue.stop(), timeout=1.0)


@pytest.mark.asyncio
async def test_abandoned_late_close_stays_off_loop_and_holds_daemon_capacity(
    monkeypatch,
):
    factory_started = threading.Event()
    release_factory = threading.Event()
    abandon = threading.Event()
    close_started = threading.Event()
    release_close = threading.Event()
    close_finished = threading.Event()
    heartbeat_ran = threading.Event()
    capacity_checked = threading.Event()
    watchdog_released = threading.Event()
    extra_started = threading.Event()

    class BlockingCloseStream:
        def close(self):
            close_started.set()
            release_close.wait()
            close_finished.set()

    def delayed_factory():
        factory_started.set()
        release_factory.wait()
        return BlockingCloseStream()

    def extra_work():
        extra_started.set()
        return None

    pool = BoundedDaemonPool(capacity=1)
    monkeypatch.setattr(bounded_daemon_module, "STREAM_DAEMON_POOL", pool)
    loop = asyncio.get_running_loop()

    def observe_loop_and_release_close():
        if not close_started.wait(1.0):
            watchdog_released.set()
            release_close.set()
            return
        loop.call_soon_threadsafe(heartbeat_ran.set)
        if not heartbeat_ran.wait(1.0):
            watchdog_released.set()
            release_close.set()
            return
        if not capacity_checked.wait(1.0):
            watchdog_released.set()
        release_close.set()

    observer = threading.Thread(target=observe_loop_and_release_close, daemon=True)
    observer.start()
    abandoned_task = asyncio.create_task(
        request_queue_module._run_queue_stream_sync_daemon(
            delayed_factory,
            should_abandon=abandon.is_set,
            close_late_result=True,
        )
    )
    try:
        assert await asyncio.to_thread(factory_started.wait, 1.0)
        abandon.set()
        assert (
            await asyncio.wait_for(abandoned_task, timeout=1.0)
            is request_queue_module._QUEUE_STREAM_ABANDONED
        )
        release_factory.set()
        assert await asyncio.to_thread(close_started.wait, 1.0)
        assert await asyncio.to_thread(heartbeat_ran.wait, 1.0)

        with pytest.raises(DaemonCapacityError):
            await request_queue_module._run_queue_stream_sync_daemon(
                extra_work,
                should_abandon=lambda: False,
            )
        assert not extra_started.is_set()
        assert pool.active_count == 1
        assert not watchdog_released.is_set()
        capacity_checked.set()
        assert await asyncio.to_thread(close_finished.wait, 1.0)
    finally:
        capacity_checked.set()
        release_factory.set()
        release_close.set()
        if not abandoned_task.done():
            abandoned_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await abandoned_task


@pytest.mark.asyncio
async def test_capacity_one_factory_handoff_to_sync_pump_succeeds(monkeypatch):
    pool = BoundedDaemonPool(capacity=1)
    monkeypatch.setattr(bounded_daemon_module, "STREAM_DAEMON_POOL", pool)
    closed = threading.Event()

    class FiniteStream:
        def __init__(self):
            self._sent = False

        def __iter__(self):
            return self

        def __next__(self):
            if self._sent:
                raise StopIteration
            self._sent = True
            return "capacity-one-output"

        def close(self):
            closed.set()

    queue = RequestQueue(max_queue_size=2, max_concurrent=1)
    queue._running = True
    channel: asyncio.Queue = asyncio.Queue(maxsize=4)
    request = QueuedRequest(
        priority=RequestPriority.NORMAL,
        timestamp=time.time(),
        request_id="capacity-one-handoff",
        request_data={},
        future=asyncio.get_running_loop().create_future(),
        client_id="handoff",
        processor=FiniteStream,
        streaming=True,
        stream_channel=channel,
    )
    try:
        result = await asyncio.wait_for(queue._process_request(request), timeout=1.0)
        assert result["status"] == "stream_completed"
        assert await asyncio.wait_for(channel.get(), timeout=1.0) == "capacity-one-output"
        assert await asyncio.wait_for(channel.get(), timeout=1.0) is None
        assert await asyncio.to_thread(closed.wait, 1.0)
        assert pool.active_count == 0
    finally:
        queue._running = False


@pytest.mark.asyncio
async def test_factory_handoff_abandonment_has_one_after_release_close_owner(
    monkeypatch,
):
    target_finished = threading.Event()
    allow_release = threading.Event()
    abandon = threading.Event()
    closed = threading.Event()
    close_calls = 0

    class GatedReleasePool(BoundedDaemonPool):
        def start(self, target, *, name, released_event=None):
            def gated_target():
                target()
                target_finished.set()
                allow_release.wait()

            return super().start(
                gated_target,
                name=name,
                released_event=released_event,
            )

    class CloseCountingStream:
        def close(self):
            nonlocal close_calls
            close_calls += 1
            closed.set()

    pool = GatedReleasePool(capacity=1)
    monkeypatch.setattr(bounded_daemon_module, "STREAM_DAEMON_POOL", pool)
    task = asyncio.create_task(
        request_queue_module._run_queue_stream_sync_daemon(
            CloseCountingStream,
            should_abandon=abandon.is_set,
            close_late_result=True,
            wait_for_worker_release=True,
        )
    )
    try:
        assert await asyncio.to_thread(target_finished.wait, 1.0)
        abandon.set()
        assert (
            await asyncio.wait_for(task, timeout=1.0)
            is request_queue_module._QUEUE_STREAM_ABANDONED
        )
        assert close_calls == 0
        allow_release.set()
        assert await asyncio.to_thread(closed.wait, 1.0)
    finally:
        allow_release.set()
        if not task.done():
            await asyncio.wait_for(task, timeout=1.0)

    assert close_calls == 1
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_cancelled_factory_handoff_uses_reserved_cleanup_after_normal_reacquire(
    monkeypatch,
):
    normal_reacquired = threading.Event()
    release_normal = threading.Event()
    release_cleanup_handoff = threading.Event()
    closed = threading.Event()
    close_calls = 0

    class ReacquiringPool(BoundedDaemonPool):
        def start(self, target, *, name, released_event=None):
            if name != "queued-stream-sync-work":
                return super().start(
                    target,
                    name=name,
                    released_event=released_event,
                )

            worker_released = threading.Event()
            thread = super().start(
                target,
                name=name,
                released_event=worker_released,
            )

            def reacquire_before_handoff():
                worker_released.wait()
                super(ReacquiringPool, self).start(
                    release_normal.wait,
                    name="normal-capacity-reacquired",
                )
                normal_reacquired.set()
                release_cleanup_handoff.wait()
                if released_event is not None:
                    released_event.set()

            threading.Thread(
                target=reacquire_before_handoff,
                name="queued-handoff-controller",
                daemon=True,
            ).start()
            return thread

    class CloseCountingStream:
        def close(self):
            nonlocal close_calls
            close_calls += 1
            closed.set()

    normal_pool = ReacquiringPool(capacity=1)
    cleanup_pool = BoundedDaemonPool(capacity=1)
    monkeypatch.setattr(bounded_daemon_module, "STREAM_DAEMON_POOL", normal_pool)
    monkeypatch.setattr(
        bounded_daemon_module,
        "STREAM_CLEANUP_DAEMON_POOL",
        cleanup_pool,
    )
    task = asyncio.create_task(
        request_queue_module._run_queue_stream_sync_daemon(
            CloseCountingStream,
            should_abandon=lambda: False,
            close_late_result=True,
            wait_for_worker_release=True,
        )
    )
    try:
        assert await asyncio.to_thread(normal_reacquired.wait, 1.0)
        assert normal_pool.active_count == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release_cleanup_handoff.set()
        assert await asyncio.to_thread(closed.wait, 1.0)
    finally:
        release_cleanup_handoff.set()
        release_normal.set()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    assert close_calls == 1


@pytest.mark.asyncio
async def test_failing_factory_handoff_releases_capacity_before_healthy_followup(
    monkeypatch,
):
    target_finished = threading.Event()
    allow_release = threading.Event()
    healthy_called = threading.Event()

    class GatedReleasePool(BoundedDaemonPool):
        def start(self, target, *, name, released_event=None):
            def gated_target():
                target()
                target_finished.set()
                allow_release.wait()

            return super().start(
                gated_target,
                name=name,
                released_event=released_event,
            )

    pool = GatedReleasePool(capacity=1)
    monkeypatch.setattr(bounded_daemon_module, "STREAM_DAEMON_POOL", pool)
    monkeypatch.setattr(
        request_queue_module,
        "REQUEST_QUEUE_STREAM_HANDOFF_CLEANUP_SECONDS",
        1.0,
    )

    def failing_factory():
        raise RuntimeError("private queued factory failure")

    failing_task = asyncio.create_task(
        request_queue_module._run_queue_stream_sync_daemon(
            failing_factory,
            should_abandon=lambda: False,
            close_late_result=True,
            wait_for_worker_release=True,
        )
    )
    try:
        assert await asyncio.to_thread(target_finished.wait, 1.0)
        await asyncio.sleep(0)
        assert not failing_task.done()
        assert pool.active_count == 1
        allow_release.set()
        with pytest.raises(RuntimeError, match="private queued factory failure"):
            await asyncio.wait_for(failing_task, timeout=1.0)

        result = await request_queue_module._run_queue_stream_sync_daemon(
            lambda: (healthy_called.set(), "healthy")[1],
            should_abandon=lambda: False,
            wait_for_worker_release=True,
        )
    finally:
        allow_release.set()
        if not failing_task.done():
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(failing_task, timeout=1.0)

    assert result == "healthy"
    assert healthy_called.is_set()
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_failing_factory_handoff_abandonment_bounds_release_wait(
    monkeypatch,
):
    target_finished = threading.Event()
    allow_release = threading.Event()
    abandon = threading.Event()
    threads: list[threading.Thread] = []

    class GatedReleasePool(BoundedDaemonPool):
        def start(self, target, *, name, released_event=None):
            def gated_target():
                target()
                target_finished.set()
                allow_release.wait()

            thread = super().start(
                gated_target,
                name=name,
                released_event=released_event,
            )
            threads.append(thread)
            return thread

    pool = GatedReleasePool(capacity=1)
    monkeypatch.setattr(bounded_daemon_module, "STREAM_DAEMON_POOL", pool)
    monkeypatch.setattr(
        request_queue_module,
        "REQUEST_QUEUE_STREAM_HANDOFF_CLEANUP_SECONDS",
        1.0,
    )

    def failing_factory():
        raise RuntimeError("private queued factory failure")

    failing_task = asyncio.create_task(
        request_queue_module._run_queue_stream_sync_daemon(
            failing_factory,
            should_abandon=abandon.is_set,
            close_late_result=True,
            wait_for_worker_release=True,
        )
    )
    try:
        assert await asyncio.to_thread(target_finished.wait, 1.0)
        assert not failing_task.done()
        abandon.set()
        with pytest.raises(RuntimeError, match="private queued factory failure"):
            await asyncio.wait_for(failing_task, timeout=1.0)
        assert pool.active_count == 1
    finally:
        allow_release.set()
        for thread in threads:
            await asyncio.to_thread(thread.join, 1.0)
        if not failing_task.done():
            failing_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await failing_task

    healthy_result = await asyncio.wait_for(
        request_queue_module._run_queue_stream_sync_daemon(
            lambda: "healthy",
            should_abandon=lambda: False,
            wait_for_worker_release=True,
        ),
        timeout=1.0,
    )

    assert healthy_result == "healthy"
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_loop_closed_delivery_failure_still_runs_sync_cleanup(monkeypatch):
    real_loop = asyncio.get_running_loop()
    cleaned = threading.Event()

    class RejectingDeliveryLoop:
        def create_future(self):
            return real_loop.create_future()

        def call_soon_threadsafe(self, *_args, **_kwargs):
            raise RuntimeError("loop closed")

        def is_closed(self):
            return True

    monkeypatch.setattr(
        request_queue_module.asyncio,
        "get_running_loop",
        lambda: RejectingDeliveryLoop(),
    )
    result = await asyncio.wait_for(
        request_queue_module._run_queue_stream_sync_daemon(
            lambda: (_ for _ in ()).throw(RuntimeError("private failure")),
            should_abandon=cleaned.is_set,
            cleanup_after_delivery=cleaned.set,
        ),
        timeout=1.0,
    )

    assert result is request_queue_module._QUEUE_STREAM_ABANDONED
    assert cleaned.is_set()


def test_bounded_daemon_constructor_failure_releases_admission(monkeypatch):
    pool = BoundedDaemonPool(capacity=1)
    real_thread = threading.Thread

    with monkeypatch.context() as scoped:
        scoped.setattr(
            bounded_daemon_module.threading,
            "Thread",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("constructor failed")),
        )
        with pytest.raises(RuntimeError, match="constructor failed"):
            pool.start(lambda: None, name="constructor-failure")

    completed = threading.Event()
    thread = pool.start(completed.set, name="constructor-recovery")
    thread.join(timeout=1.0)

    assert real_thread is threading.Thread
    assert completed.is_set()
    assert pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_kind", ["sync", "async"])
async def test_stream_error_frames_precede_single_terminal_sentinel(stream_kind):
    sentinel = f"queued-{stream_kind}-error-secret-/srv/provider"

    def sync_processor():
        def stream():
            raise RuntimeError(sentinel)
            yield  # pragma: no cover

        return stream()

    def async_processor():
        async def stream():
            raise RuntimeError(sentinel)
            yield  # pragma: no cover

        return stream()

    queue = RequestQueue(max_queue_size=10, max_concurrent=1, timeout=1.0)
    await queue.start(num_workers=1)
    channel: asyncio.Queue = asyncio.Queue(maxsize=10)
    try:
        future = await queue.enqueue(
            request_id=f"ordered-error-{stream_kind}",
            request_data={},
            client_id=stream_kind,
            processor=sync_processor if stream_kind == "sync" else async_processor,
            streaming=True,
            stream_channel=channel,
        )
        terminal = await asyncio.wait_for(channel.get(), timeout=1.0)
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(asyncio.shield(future), timeout=1.0)
    finally:
        await asyncio.wait_for(queue.stop(), timeout=1.0)

    assert isinstance(terminal, request_queue_module.QueueStreamTerminalError)
    assert terminal.code == "provider_unavailable"
    assert sentinel not in repr(terminal)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_kind", ["sync", "async"])
async def test_late_stream_failure_preserves_buffered_output_before_terminal_error(
    stream_kind,
):
    sentinel = f"queued-{stream_kind}-late-secret-/srv/provider"
    producer_failed = threading.Event()

    def sync_processor():
        def stream():
            yield f"buffered-{stream_kind}-output"
            producer_failed.set()
            raise RuntimeError(sentinel)

        return stream()

    def async_processor():
        async def stream():
            yield f"buffered-{stream_kind}-output"
            producer_failed.set()
            raise RuntimeError(sentinel)

        return stream()

    queue = RequestQueue(max_queue_size=4, max_concurrent=1, timeout=1.0)
    await queue.start(num_workers=1)
    channel: asyncio.Queue = request_queue_module.QueueStreamChannel(maxsize=1)
    try:
        future = await queue.enqueue(
            request_id=f"buffered-late-error-{stream_kind}",
            request_data={},
            client_id=stream_kind,
            processor=sync_processor if stream_kind == "sync" else async_processor,
            streaming=True,
            stream_channel=channel,
        )
        assert await asyncio.to_thread(producer_failed.wait, 1.0)
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(asyncio.shield(future), timeout=1.0)
        first = await asyncio.wait_for(channel.get(), timeout=1.0)
        terminal = await asyncio.wait_for(channel.get(), timeout=1.0)
    finally:
        await asyncio.wait_for(queue.stop(), timeout=1.0)

    assert first == f"buffered-{stream_kind}-output"
    assert isinstance(terminal, request_queue_module.QueueStreamTerminalError)
    assert terminal.code == "provider_unavailable"
    assert sentinel not in repr(terminal)


@pytest.mark.asyncio
async def test_successful_stream_reserves_terminal_behind_full_data_queue():
    """Completion must not wait for or lose its sole terminal sentinel."""

    async def processor():
        yield "accepted-output"

    queue = RequestQueue(max_queue_size=2, max_concurrent=1, timeout=1.0)
    await queue.start(num_workers=1)
    channel = request_queue_module.QueueStreamChannel(maxsize=1)
    try:
        future = await queue.enqueue(
            request_id="delayed-success-terminal",
            request_data={},
            client_id="delayed-consumer",
            processor=processor,
            streaming=True,
            stream_channel=channel,
        )
        result = await asyncio.wait_for(asyncio.shield(future), timeout=1.0)

        assert result["status"] == "stream_completed"
        assert await asyncio.wait_for(channel.get(), timeout=0.1) == "accepted-output"
        assert await asyncio.wait_for(channel.get(), timeout=0.1) is None
        assert channel.empty()
    finally:
        await asyncio.wait_for(queue.stop(), timeout=1.0)


def test_queue_stream_channel_reserves_exactly_one_none_terminal():
    channel = request_queue_module.QueueStreamChannel(maxsize=1)

    assert channel.put_terminal_nowait(None) is True
    assert (
        channel.put_terminal_nowait(
            request_queue_module.QueueStreamTerminalError(
                code="provider_unavailable"
            )
        )
        is False
    )
    assert channel.get_nowait() is None
    with pytest.raises(asyncio.QueueEmpty):
        channel.get_nowait()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["factory", "iterator"])
async def test_sync_provider_cancellation_is_terminal_and_worker_survives(
    failure_phase,
):
    """A sync provider's cancellation must not cancel the queue worker."""

    sentinel = f"sync-{failure_phase}-cancel-secret-/srv/provider"

    def cancelled_processor():
        if failure_phase == "factory":
            raise asyncio.CancelledError(sentinel)

        def cancelled_stream():
            raise asyncio.CancelledError(sentinel)
            yield  # pragma: no cover

        return cancelled_stream()

    def healthy_processor():
        return iter(("healthy-sync-output",))

    queue = RequestQueue(max_queue_size=4, max_concurrent=1, timeout=1.0)
    await queue.start(num_workers=1)
    cancelled_channel = request_queue_module.QueueStreamChannel(maxsize=1)
    healthy_channel = request_queue_module.QueueStreamChannel(maxsize=1)
    cancelled_future = None
    healthy_future = None
    try:
        cancelled_future = await queue.enqueue(
            request_id=f"sync-provider-{failure_phase}-cancelled",
            request_data={},
            client_id="cancelled-sync-provider",
            processor=cancelled_processor,
            streaming=True,
            stream_channel=cancelled_channel,
        )
        terminal = await asyncio.wait_for(cancelled_channel.get(), timeout=1.0)
        assert isinstance(terminal, request_queue_module.QueueStreamTerminalError)
        assert terminal.code == "provider_unavailable"
        with pytest.raises(SanitizedProviderStreamError) as exc_info:
            await asyncio.wait_for(asyncio.shield(cancelled_future), timeout=1.0)
        assert sentinel not in repr(exc_info.value)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None

        healthy_future = await queue.enqueue(
            request_id=f"healthy-after-sync-{failure_phase}-cancel",
            request_data={},
            client_id="healthy-sync-provider",
            processor=healthy_processor,
            streaming=True,
            stream_channel=healthy_channel,
        )
        assert await asyncio.wait_for(healthy_channel.get(), timeout=1.0) == (
            "healthy-sync-output"
        )
        assert await asyncio.wait_for(healthy_channel.get(), timeout=1.0) is None
        healthy_result = await asyncio.wait_for(healthy_future, timeout=1.0)
        assert healthy_result["status"] == "stream_completed"
    finally:
        for future in (cancelled_future, healthy_future):
            if future is not None and not future.done():
                future.cancel()
        await asyncio.wait_for(queue.stop(), timeout=1.0)


@pytest.mark.asyncio
async def test_provider_child_cancellation_is_terminal_and_worker_survives():
    """A provider-cancelled child task must not cancel the sole queue worker."""

    sentinel = "provider-child-cancel-secret-/srv/provider"

    async def cancelled_processor():
        raise asyncio.CancelledError(sentinel)
        yield  # pragma: no cover

    async def healthy_processor():
        yield "healthy-output"

    queue = RequestQueue(max_queue_size=4, max_concurrent=1, timeout=1.0)
    await queue.start(num_workers=1)
    cancelled_channel = request_queue_module.QueueStreamChannel(maxsize=1)
    healthy_channel = request_queue_module.QueueStreamChannel(maxsize=1)
    try:
        cancelled_future = await queue.enqueue(
            request_id="provider-child-cancelled",
            request_data={},
            client_id="cancelled-provider",
            processor=cancelled_processor,
            streaming=True,
            stream_channel=cancelled_channel,
        )
        terminal = await asyncio.wait_for(cancelled_channel.get(), timeout=1.0)
        assert isinstance(terminal, request_queue_module.QueueStreamTerminalError)
        assert terminal.code == "provider_unavailable"
        with pytest.raises(SanitizedProviderStreamError) as exc_info:
            await asyncio.wait_for(asyncio.shield(cancelled_future), timeout=1.0)
        assert sentinel not in repr(exc_info.value)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None

        healthy_future = await queue.enqueue(
            request_id="healthy-after-provider-cancel",
            request_data={},
            client_id="healthy-provider",
            processor=healthy_processor,
            streaming=True,
            stream_channel=healthy_channel,
        )
        assert await asyncio.wait_for(healthy_channel.get(), timeout=1.0) == "healthy-output"
        assert await asyncio.wait_for(healthy_channel.get(), timeout=1.0) is None
        healthy_result = await asyncio.wait_for(healthy_future, timeout=1.0)
        assert healthy_result["status"] == "stream_completed"
    finally:
        await asyncio.wait_for(queue.stop(), timeout=1.0)


@pytest.mark.asyncio
async def test_stop_terminalizes_active_stream_future_and_channel():
    yielded = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    def processor():
        async def stream():
            try:
                yielded.set()
                yield "active output"
                await release.wait()
            finally:
                closed.set()

        return stream()

    queue = RequestQueue(max_queue_size=10, max_concurrent=1, timeout=1.0)
    await queue.start(num_workers=1)
    channel: asyncio.Queue = asyncio.Queue(maxsize=10)
    future = await queue.enqueue(
        request_id="stop-active-stream",
        request_data={},
        client_id="stop",
        processor=processor,
        streaming=True,
        stream_channel=channel,
    )
    try:
        assert await asyncio.wait_for(channel.get(), timeout=1.0) == "active output"
        await asyncio.wait_for(yielded.wait(), timeout=1.0)
        await asyncio.wait_for(queue.stop(), timeout=1.0)

        assert future.done()
        assert future.cancelled()
        assert isinstance(
            await asyncio.wait_for(channel.get(), timeout=1.0),
            request_queue_module.QueueStreamTerminalError,
        )
        assert await asyncio.wait_for(closed.wait(), timeout=1.0)
    finally:
        release.set()
        if not queue._stopped:
            await asyncio.wait_for(queue.stop(), timeout=1.0)


@pytest.mark.asyncio
async def test_regular_def_async_iterator_aclose_never_blocks_queue_loop(monkeypatch):
    monkeypatch.setattr(
        request_queue_module,
        "REQUEST_QUEUE_STREAM_CLEANUP_TIMEOUT_SECONDS",
        0.01,
    )
    close_started = threading.Event()
    release_close = threading.Event()
    close_finished = threading.Event()
    heartbeat = asyncio.Event()

    class RegularBlockingCloseAsyncIterator:
        def __init__(self):
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return "queue regular-close output"

        def aclose(self):
            close_started.set()
            release_close.wait()
            close_finished.set()

    queue = RequestQueue(max_queue_size=2, max_concurrent=1)
    queue._running = True
    request = QueuedRequest(
        priority=RequestPriority.NORMAL,
        timestamp=time.time(),
        request_id="regular-def-aclose",
        request_data={},
        future=asyncio.get_running_loop().create_future(),
        client_id="regular-close",
        processor=RegularBlockingCloseAsyncIterator,
        streaming=True,
        stream_channel=asyncio.Queue(maxsize=4),
    )
    task = asyncio.create_task(queue._process_request(request))
    try:
        assert await asyncio.wait_for(request.stream_channel.get(), timeout=1.0) == (
            "queue regular-close output"
        )
        assert await asyncio.to_thread(close_started.wait, 1.0)
        asyncio.get_running_loop().call_soon(heartbeat.set)
        await asyncio.wait_for(heartbeat.wait(), timeout=1.0)
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result["status"] == "stream_completed"
        assert not release_close.is_set()
    finally:
        release_close.set()
        assert await asyncio.to_thread(close_finished.wait, 1.0)
        if not task.done():
            await asyncio.wait_for(task, timeout=1.0)
        queue._running = False


@pytest.mark.asyncio
async def test_stop_terminalizes_active_blocking_sync_stream_without_releasing_capacity(
    monkeypatch,
):
    pool = BoundedDaemonPool(capacity=1)
    monkeypatch.setattr(bounded_daemon_module, "STREAM_DAEMON_POOL", pool)
    next_blocked = threading.Event()
    release_next = threading.Event()
    closed = threading.Event()
    extra_started = threading.Event()

    class BlockingSyncStream:
        def __init__(self):
            self._first = True

        def __iter__(self):
            return self

        def __next__(self):
            if self._first:
                self._first = False
                return "active sync output"
            next_blocked.set()
            release_next.wait()
            raise StopIteration

        def close(self):
            closed.set()

    queue = RequestQueue(max_queue_size=10, max_concurrent=1, timeout=1.0)
    await queue.start(num_workers=1)
    channel: asyncio.Queue = asyncio.Queue(maxsize=10)
    future = await queue.enqueue(
        request_id="stop-active-sync-stream",
        request_data={},
        client_id="stop-sync",
        processor=BlockingSyncStream,
        streaming=True,
        stream_channel=channel,
    )
    try:
        assert await asyncio.wait_for(channel.get(), timeout=1.0) == "active sync output"
        assert await asyncio.to_thread(next_blocked.wait, 1.0)

        heartbeat = asyncio.Event()
        asyncio.get_running_loop().call_soon(heartbeat.set)
        await asyncio.wait_for(heartbeat.wait(), timeout=1.0)
        await asyncio.wait_for(queue.stop(), timeout=1.0)

        assert future.done()
        assert future.cancelled()
        assert isinstance(
            await asyncio.wait_for(channel.get(), timeout=1.0),
            request_queue_module.QueueStreamTerminalError,
        )
        assert pool.active_count == 1
        assert not release_next.is_set()
        with pytest.raises(DaemonCapacityError):
            pool.start(extra_started.set, name="must-not-start")
        assert not extra_started.is_set()
    finally:
        release_next.set()
        assert await asyncio.to_thread(closed.wait, 1.0)
        if not queue._stopped:
            await asyncio.wait_for(queue.stop(), timeout=1.0)


@pytest.mark.asyncio
async def test_clear_pending_stream_fails_closed_and_queue_remains_usable():
    queue = RequestQueue(max_queue_size=4, max_concurrent=1, timeout=1.0)
    channel: asyncio.Queue = asyncio.Queue(maxsize=1)
    channel.put_nowait("stale pending output")
    adapter_started = threading.Event()
    pending_future = await queue.enqueue(
        request_id="clear-pending-stream",
        request_data={},
        client_id="pending",
        processor=lambda: adapter_started.set(),
        streaming=True,
        stream_channel=channel,
    )

    await queue.clear_queue()
    terminal = await asyncio.wait_for(channel.get(), timeout=1.0)
    assert pending_future.cancelled()
    assert not adapter_started.is_set()
    assert isinstance(terminal, request_queue_module.QueueStreamTerminalError)

    await queue.start(num_workers=1)
    try:
        healthy = await queue.enqueue(
            request_id="healthy-after-clear",
            request_data={},
            client_id="healthy",
            processor=lambda: "healthy",
            streaming=False,
        )
        assert await asyncio.wait_for(healthy, timeout=1.0) == "healthy"
    finally:
        await asyncio.wait_for(queue.stop(), timeout=1.0)


@pytest.mark.asyncio
async def test_stream_expiring_before_dispatch_emits_terminal_error_without_adapter_start():
    queue = RequestQueue(max_queue_size=4, max_concurrent=1, timeout=0.01)
    channel: asyncio.Queue = asyncio.Queue(maxsize=1)
    channel.put_nowait("stale timed-out output")
    adapter_started = threading.Event()
    future = await queue.enqueue(
        request_id="expired-before-dispatch",
        request_data={},
        client_id="expired",
        processor=lambda: adapter_started.set(),
        streaming=True,
        stream_channel=channel,
    )
    await asyncio.sleep(0.02)
    await queue.start(num_workers=1)
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(future), timeout=1.0)
        terminal = await asyncio.wait_for(channel.get(), timeout=1.0)
    finally:
        await asyncio.wait_for(queue.stop(), timeout=1.0)

    assert not adapter_started.is_set()
    assert isinstance(terminal, request_queue_module.QueueStreamTerminalError)


@pytest.mark.asyncio
async def test_request_queue_is_one_shot_after_stop():
    queue = RequestQueue(max_queue_size=10, max_concurrent=1)
    await queue.start(num_workers=1)
    await queue.stop()

    with pytest.raises(RuntimeError, match="cannot be restarted"):
        await queue.start(num_workers=1)
