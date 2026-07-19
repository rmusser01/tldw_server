"""Concurrency regressions for non-cooperative async provider streams."""

from __future__ import annotations

import asyncio
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.core.Chat import bounded_daemon, streaming_utils
from tldw_Server_API.app.core.Chat import request_queue as request_queue_module
from tldw_Server_API.app.core.Chat.bounded_daemon import (
    BoundedDaemonPool,
    DaemonCapacityError,
)
from tldw_Server_API.app.core.Chat.request_queue import (
    QueuedRequest,
    RequestPriority,
    RequestQueue,
)


class _ResistantAsyncIterator:
    """Keep one ``__anext__`` live until the test releases it."""

    def __init__(
        self,
        *,
        release: asyncio.Event,
        started: list[asyncio.Event],
        live: set[int],
        identifier: int,
    ) -> None:
        self._release = release
        self._started = started
        self._live = live
        self._identifier = identifier
        self.closed = False

    def __aiter__(self) -> _ResistantAsyncIterator:
        return self

    async def __anext__(self) -> str:
        self._live.add(self._identifier)
        self._started[self._identifier].set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await self._release.wait()
            raise StopAsyncIteration from None
        finally:
            self._live.discard(self._identifier)
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


class _HealthyAsyncIterator:
    def __init__(self, content: str) -> None:
        self._content = content
        self._sent = False
        self.started = False
        self.closed = False

    def __aiter__(self) -> _HealthyAsyncIterator:
        return self

    async def __anext__(self) -> str:
        self.started = True
        if self._sent:
            raise StopAsyncIteration
        self._sent = True
        return self._content

    async def aclose(self) -> None:
        self.closed = True


class _TwoChunkAsyncIterator:
    def __init__(self) -> None:
        self._chunks = iter(("first output", "second output"))
        self.closed = False

    def __aiter__(self) -> _TwoChunkAsyncIterator:
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.closed = True


async def _wait_for_live_count(live: set[int], expected: int) -> None:
    async def poll() -> None:
        while len(live) != expected:
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=1.0)


async def _wait_for_started_count(
    started: list[asyncio.Event],
    expected: int,
) -> None:
    async def poll() -> None:
        while sum(event.is_set() for event in started) < expected:
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=1.0)


async def _wait_for_done_count(
    tasks: list[asyncio.Task[Any]],
    expected: int,
) -> None:
    async def poll() -> None:
        while sum(task.done() for task in tasks) < expected:
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=1.0)


async def _wait_for_pool_count(pool: BoundedDaemonPool, expected: int) -> None:
    async def poll() -> None:
        while pool.active_count != expected:
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=1.0)


async def _wait_for_stream_task_count(expected: int) -> None:
    async def poll() -> None:
        while True:
            with streaming_utils._STREAM_TASK_CAPACITY_LOCK:
                active = streaming_utils._STREAM_TASK_ACTIVE_COUNT
            if active == expected:
                return
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=1.0)


def _queued_request(request_id: str, iterator: Any) -> QueuedRequest:
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


@pytest.mark.asyncio
async def test_direct_noncooperative_provider_tasks_are_bounded_and_capacity_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capacity = 2
    attempts = 6
    release = asyncio.Event()
    started = [asyncio.Event() for _ in range(attempts)]
    live: set[int] = set()
    streams: list[_ResistantAsyncIterator] = []
    generators = []
    next_tasks: list[asyncio.Task[str]] = []

    monkeypatch.setattr(streaming_utils, "STREAM_TASK_MAX_ACTIVE", capacity, raising=False)
    monkeypatch.setattr(streaming_utils, "STREAM_TASK_CANCEL_DRAIN_SECONDS", 0.01)
    monkeypatch.setattr(streaming_utils, "STREAM_CLEANUP_TIMEOUT_SECONDS", 0.01)

    for identifier in range(attempts):
        stream = _ResistantAsyncIterator(
            release=release,
            started=started,
            live=live,
            identifier=identifier,
        )
        streams.append(stream)
        generator = streaming_utils.create_streaming_response_with_timeout(
            stream,
            f"direct-{identifier}",
            "model",
            heartbeat_interval=0,
        )
        assert "event: stream_start" in await asyncio.wait_for(
            generator.__anext__(),
            timeout=1.0,
        )
        generators.append(generator)

    try:
        next_tasks = [asyncio.create_task(generator.__anext__()) for generator in generators]
        await _wait_for_started_count(started, capacity)
        await _wait_for_done_count(next_tasks, attempts - capacity)

        for task in next_tasks:
            if not task.done():
                task.cancel()
        outcomes = await asyncio.wait_for(
            asyncio.gather(*next_tasks, return_exceptions=True),
            timeout=1.0,
        )

        assert len(live) == capacity
        rejected = [
            (index, outcome)
            for index, outcome in enumerate(outcomes)
            if isinstance(outcome, str) and '"code": "provider_unavailable"' in outcome
        ]
        assert len(rejected) == attempts - capacity
        for index, _ in rejected:
            assert await generators[index].__anext__() == "data: [DONE]\n\n"
            with pytest.raises(StopAsyncIteration):
                await generators[index].__anext__()
            assert streams[index].closed is True
    finally:
        release.set()
        for task in next_tasks:
            if not task.done():
                task.cancel()
        if next_tasks:
            await asyncio.gather(*next_tasks, return_exceptions=True)
        await _wait_for_live_count(live, 0)
        for generator in generators:
            await generator.aclose()
        await _wait_for_stream_task_count(0)

    healthy = _HealthyAsyncIterator("healthy direct output")
    wire = "".join(
        [
            chunk
            async for chunk in streaming_utils.create_streaming_response_with_timeout(
                healthy,
                "direct-healthy",
                "model",
                heartbeat_interval=0,
            )
        ]
    )
    assert "healthy direct output" in wire
    assert wire.count("data: [DONE]\n\n") == 1
    assert healthy.closed is True


@pytest.mark.asyncio
async def test_queued_noncooperative_provider_tasks_are_bounded_and_capacity_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capacity = 2
    attempts = 6
    release = asyncio.Event()
    started = [asyncio.Event() for _ in range(attempts)]
    live: set[int] = set()
    queue = RequestQueue(max_queue_size=10, max_concurrent=attempts)
    queue._running = True
    streams = [
        _ResistantAsyncIterator(
            release=release,
            started=started,
            live=live,
            identifier=identifier,
        )
        for identifier in range(attempts)
    ]
    requests = [
        _queued_request(f"queued-{identifier}", stream)
        for identifier, stream in enumerate(streams)
    ]
    tasks: list[asyncio.Task[dict[str, Any]]] = []

    monkeypatch.setattr(streaming_utils, "STREAM_TASK_MAX_ACTIVE", capacity, raising=False)
    monkeypatch.setattr(request_queue_module, "REQUEST_QUEUE_STREAM_CANCEL_DRAIN_SECONDS", 0.01)
    monkeypatch.setattr(request_queue_module, "REQUEST_QUEUE_STREAM_CLEANUP_TIMEOUT_SECONDS", 0.01)

    try:
        tasks = [
            asyncio.create_task(queue._process_request(request))
            for request in requests
        ]
        await _wait_for_started_count(started, capacity)
        await _wait_for_done_count(tasks, attempts - capacity)
        for request in requests:
            request.future.cancel()
        outcomes = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=1.0,
        )

        assert len(live) == capacity
        rejected = [
            index for index, outcome in enumerate(outcomes)
            if isinstance(outcome, RuntimeError)
        ]
        assert len(rejected) == attempts - capacity
        assert all(streams[index].closed for index in rejected)
    finally:
        release.set()
        for request in requests:
            request.future.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await _wait_for_live_count(live, 0)

    healthy = _HealthyAsyncIterator("healthy queued output")
    healthy_request = _queued_request("queued-healthy", healthy)
    healthy_result = await asyncio.wait_for(
        queue._process_request(healthy_request),
        timeout=1.0,
    )

    assert healthy_result["status"] == "stream_completed"
    assert await healthy_request.stream_channel.get() == "healthy queued output"
    assert healthy.closed is True


@pytest.mark.asyncio
async def test_endpoint_prime_capacity_failure_closes_unstarted_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capacity = 2
    release = asyncio.Event()
    started = [asyncio.Event() for _ in range(capacity)]

    async def resistant_work(identifier: int) -> None:
        started[identifier].set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()

    monkeypatch.setattr(streaming_utils, "STREAM_TASK_MAX_ACTIVE", capacity, raising=False)
    monkeypatch.setattr(chat_endpoint, "PROVIDER_STREAM_PRIME_CLEANUP_TIMEOUT_SECONDS", 0.1)
    leases = [
        streaming_utils.create_bounded_stream_task(resistant_work(identifier))
        for identifier in range(capacity)
    ]
    await asyncio.gather(*(asyncio.wait_for(event.wait(), 1.0) for event in started))

    adapter = _HealthyAsyncIterator("must not dispatch")

    class Response:
        body_iterator = adapter

    try:
        primed = await chat_endpoint._prime_provider_stream_response(Response(), {})
        assert primed == ((), "provider_unavailable", False, False)
        assert adapter.started is False
        assert adapter.closed is True
    finally:
        await streaming_utils.cancel_stream_tasks_bounded(leases, timeout=0.01)
        release.set()
        await asyncio.gather(*leases, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_stream_tasks_observes_late_failure_when_caller_is_cancelled() -> None:
    """A second cancellation must not orphan a resistant provider task."""

    class CancellationResistantFuture(asyncio.Future[None]):
        def __init__(self) -> None:
            super().__init__()
            self.observer_attached = False

        def cancel(self, msg=None) -> bool:
            return False

        def add_done_callback(self, fn, *, context=None) -> None:
            if fn is streaming_utils._observe_stream_task:
                self.observer_attached = True
            super().add_done_callback(fn, context=context)

    provider_task = CancellationResistantFuture()
    cleanup = asyncio.create_task(
        streaming_utils.cancel_stream_tasks_bounded(
            [provider_task],
            timeout=10.0,
        )
    )
    await asyncio.sleep(0)
    cleanup.cancel()

    with pytest.raises(asyncio.CancelledError):
        await cleanup

    assert provider_task.observer_attached is True
    provider_task.set_exception(RuntimeError("late provider failure"))
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_noncooperative_cleanup_tasks_have_a_separate_bound_and_recover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capacity = 2
    attempts = 6
    release = asyncio.Event()
    live: set[int] = set()

    async def resistant_cleanup(identifier: int) -> None:
        live.add(identifier)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
        finally:
            live.discard(identifier)

    monkeypatch.setattr(
        streaming_utils,
        "STREAM_CLEANUP_TASK_MAX_ACTIVE",
        capacity,
        raising=False,
    )
    calls = [
        asyncio.create_task(
            streaming_utils.await_stream_operation_bounded(
                resistant_cleanup(identifier),
                timeout=0.01,
                cleanup=True,
            )
        )
        for identifier in range(attempts)
    ]
    try:
        outcomes = await asyncio.wait_for(
            asyncio.gather(*calls, return_exceptions=True),
            timeout=1.0,
        )
        assert len(live) == capacity
        assert sum(isinstance(item, asyncio.TimeoutError) for item in outcomes) == capacity
        assert sum(
            isinstance(item, streaming_utils.StreamTaskCapacityError)
            for item in outcomes
        ) == attempts - capacity
    finally:
        release.set()
        await asyncio.gather(*calls, return_exceptions=True)
        await _wait_for_live_count(live, 0)
        await asyncio.sleep(0)

    closed = False

    async def healthy_cleanup() -> None:
        nonlocal closed
        closed = True

    await streaming_utils.await_stream_operation_bounded(
        healthy_cleanup(),
        timeout=0.1,
        cleanup=True,
    )
    assert closed is True


@pytest.mark.asyncio
async def test_direct_initial_capacity_failure_closes_adapter_and_finalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = asyncio.Event()
    started = asyncio.Event()

    async def resistant_work() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()

    monkeypatch.setattr(streaming_utils, "STREAM_TASK_MAX_ACTIVE", 1, raising=False)
    lease = streaming_utils.create_bounded_stream_task(resistant_work())
    await asyncio.wait_for(started.wait(), timeout=1.0)
    adapter = _HealthyAsyncIterator("must not dispatch")
    finalized: list[dict[str, bool]] = []

    async def finalize(**outcome: bool) -> None:
        finalized.append(outcome)

    try:
        wire = "".join(
            [
                chunk
                async for chunk in streaming_utils.create_streaming_response_with_timeout(
                    adapter,
                    "initial-capacity-rejection",
                    "model",
                    finalize_callback=finalize,
                    heartbeat_interval=0,
                )
            ]
        )
        assert '"code": "provider_unavailable"' in wire
        assert wire.count("data: [DONE]\n\n") == 1
        assert adapter.started is False
        assert adapter.closed is True
        assert finalized == [
            {"success": False, "cancelled": False, "error": True}
        ]
    finally:
        await streaming_utils.cancel_stream_tasks_bounded([lease], timeout=0.01)
        release.set()
        await asyncio.gather(lease, return_exceptions=True)


@pytest.mark.asyncio
async def test_midstream_capacity_failure_finalizes_as_provider_error_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capacity = 2
    release = asyncio.Event()
    started = [asyncio.Event() for _ in range(capacity)]

    async def resistant_work(identifier: int) -> None:
        started[identifier].set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()

    monkeypatch.setattr(streaming_utils, "STREAM_TASK_MAX_ACTIVE", capacity, raising=False)
    stream = _TwoChunkAsyncIterator()
    finalized: list[dict[str, bool]] = []

    async def finalize(**outcome: bool) -> None:
        finalized.append(outcome)

    generator = streaming_utils.create_streaming_response_with_timeout(
        stream,
        "midstream-capacity-rejection",
        "model",
        finalize_callback=finalize,
        heartbeat_interval=0,
    )
    wire = ""
    while "first output" not in wire:
        wire += await asyncio.wait_for(generator.__anext__(), timeout=1.0)

    leases: list[asyncio.Future[Any]] = []
    try:
        await _wait_for_stream_task_count(0)
        for identifier in range(capacity):
            leases.append(
                streaming_utils.create_bounded_stream_task(
                    resistant_work(identifier)
                )
            )
        await _wait_for_started_count(started, capacity)
        wire += await asyncio.wait_for(generator.__anext__(), timeout=1.0)
        wire += await asyncio.wait_for(generator.__anext__(), timeout=1.0)
        with pytest.raises(StopAsyncIteration):
            await generator.__anext__()

        assert '"code": "provider_unavailable"' in wire
        assert "second output" not in wire
        assert wire.count("data: [DONE]\n\n") == 1
        assert finalized == [
            {"success": False, "cancelled": False, "error": True}
        ]
    finally:
        release.set()
        for lease in leases:
            lease.cancel()
        await asyncio.gather(*leases, return_exceptions=True)
        await generator.aclose()

    assert stream.closed is True


@pytest.mark.asyncio
async def test_sync_adapter_cleanup_uses_bounded_reserved_daemon_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regular_release = threading.Event()
    regular_started = threading.Event()
    cleanup_release = threading.Event()
    cleanup_started = threading.Event()
    cleanup_finished = threading.Event()
    rejected_started = threading.Event()
    recovered_started = threading.Event()
    regular_pool = BoundedDaemonPool(capacity=1)
    cleanup_pool = BoundedDaemonPool(capacity=1)

    def regular_work() -> None:
        regular_started.set()
        regular_release.wait()

    def resistant_close() -> None:
        cleanup_started.set()
        cleanup_release.wait()
        cleanup_finished.set()

    def rejected_close() -> None:
        rejected_started.set()

    def recovered_close() -> None:
        recovered_started.set()

    monkeypatch.setattr(bounded_daemon, "STREAM_DAEMON_POOL", regular_pool)
    monkeypatch.setattr(bounded_daemon, "STREAM_CLEANUP_DAEMON_POOL", cleanup_pool)
    bounded_daemon.start_bounded_stream_daemon(
        regular_work,
        name="test-regular-capacity-holder",
    )
    assert await asyncio.to_thread(regular_started.wait, 1.0)

    try:
        first_close = asyncio.create_task(
            streaming_utils.invoke_stream_close_bounded(resistant_close, timeout=0.02)
        )
        assert await asyncio.to_thread(cleanup_started.wait, 1.0)
        with pytest.raises(asyncio.TimeoutError):
            await first_close
        assert regular_pool.active_count == 1
        assert cleanup_pool.active_count == 1

        with pytest.raises(DaemonCapacityError):
            await streaming_utils.invoke_stream_close_bounded(rejected_close, timeout=0.1)
        assert rejected_started.is_set() is False

        cleanup_release.set()
        assert await asyncio.to_thread(cleanup_finished.wait, 1.0)
        await _wait_for_pool_count(cleanup_pool, 0)

        await streaming_utils.invoke_stream_close_bounded(recovered_close, timeout=0.1)
        assert recovered_started.is_set() is True
        assert regular_pool.active_count == 1
    finally:
        cleanup_release.set()
        regular_release.set()
        await _wait_for_pool_count(cleanup_pool, 0)
        await _wait_for_pool_count(regular_pool, 0)


@pytest.mark.asyncio
async def test_owned_operation_timeout_retains_task_and_cleanup_capacity_until_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deadline must hand late work to bounded cleanup without cancelling it."""
    release = asyncio.Event()
    started = asyncio.Event()
    cleanup_finished = asyncio.Event()
    cleanup_claimed = threading.Event()
    cleanup_calls = 0
    rejected_started = False

    monkeypatch.setattr(streaming_utils, "STREAM_TASK_MAX_ACTIVE", 1)
    monkeypatch.setattr(streaming_utils, "STREAM_CLEANUP_TASK_MAX_ACTIVE", 1)

    async def blocked_operation() -> str:
        started.set()
        await release.wait()
        return "late-result"

    async def cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        cleanup_finished.set()

    with pytest.raises(TimeoutError, match="owned operation timed out"):
        await streaming_utils.await_bounded_owned_operation(
            blocked_operation(),
            timeout_seconds=0.01,
            timeout_message="owned operation timed out",
            on_abandoned=cleanup,
            cleanup_claimed=cleanup_claimed,
        )

    assert started.is_set()
    assert cleanup_claimed.is_set()
    assert cleanup_finished.is_set() is False
    assert cleanup_calls == 0
    with streaming_utils._STREAM_TASK_CAPACITY_LOCK:
        assert streaming_utils._STREAM_TASK_ACTIVE_COUNT == 1
        assert streaming_utils._STREAM_CLEANUP_TASK_ACTIVE_COUNT == 1

    async def rejected_operation() -> None:
        nonlocal rejected_started
        rejected_started = True

    with pytest.raises(streaming_utils.StreamTaskCapacityError):
        await streaming_utils.await_bounded_owned_operation(
            rejected_operation(),
            timeout_seconds=0.1,
            timeout_message="must not start",
            on_abandoned=cleanup,
        )
    assert rejected_started is False

    release.set()
    await asyncio.wait_for(cleanup_finished.wait(), timeout=1.0)
    await _wait_for_stream_task_count(0)
    async def wait_for_cleanup_capacity() -> None:
        while True:
            with streaming_utils._STREAM_TASK_CAPACITY_LOCK:
                if streaming_utils._STREAM_CLEANUP_TASK_ACTIVE_COUNT == 0:
                    return
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_cleanup_capacity(), timeout=1.0)
    assert cleanup_calls == 1

    assert await streaming_utils.await_bounded_owned_operation(
        asyncio.sleep(0, result="healthy"),
        timeout_seconds=0.1,
        timeout_message="healthy operation timed out",
        on_abandoned=cleanup,
    ) == "healthy"
    assert cleanup_calls == 1


@pytest.mark.asyncio
async def test_owned_operation_cancellation_is_prompt_and_cleans_after_true_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller cancellation leaves resistant work owned until its natural exit."""
    release = asyncio.Event()
    started = asyncio.Event()
    cleanup_finished = asyncio.Event()
    cleanup_calls = 0

    monkeypatch.setattr(streaming_utils, "STREAM_TASK_MAX_ACTIVE", 1)
    monkeypatch.setattr(streaming_utils, "STREAM_CLEANUP_TASK_MAX_ACTIVE", 1)

    async def blocked_operation() -> None:
        started.set()
        await release.wait()

    async def cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        cleanup_finished.set()

    caller = asyncio.create_task(
        streaming_utils.await_bounded_owned_operation(
            blocked_operation(),
            timeout_seconds=10.0,
            timeout_message="unexpected timeout",
            on_abandoned=cleanup,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    caller.cancel()
    done, _pending = await asyncio.wait({caller}, timeout=0.1)
    assert caller in done
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert cleanup_finished.is_set() is False

    release.set()
    await asyncio.wait_for(cleanup_finished.wait(), timeout=1.0)
    assert cleanup_calls == 1


@pytest.mark.asyncio
async def test_owned_operation_self_cancellation_releases_cleanup_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An awaitable that cancels itself must not strand its cleanup decision."""
    cleanup_calls = 0
    worker_released = threading.Event()

    monkeypatch.setattr(streaming_utils, "STREAM_TASK_MAX_ACTIVE", 1)
    monkeypatch.setattr(streaming_utils, "STREAM_CLEANUP_TASK_MAX_ACTIVE", 1)

    async def self_cancel() -> None:
        raise asyncio.CancelledError

    async def cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    with pytest.raises(asyncio.CancelledError):
        await streaming_utils.await_bounded_owned_operation(
            self_cancel(),
            timeout_seconds=0.1,
            timeout_message="unexpected timeout",
            on_abandoned=cleanup,
            released_event=worker_released,
        )

    async def wait_for_capacity_release() -> None:
        while True:
            with streaming_utils._STREAM_TASK_CAPACITY_LOCK:
                if (
                    streaming_utils._STREAM_TASK_ACTIVE_COUNT == 0
                    and streaming_utils._STREAM_CLEANUP_TASK_ACTIVE_COUNT == 0
                ):
                    return
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_capacity_release(), timeout=1.0)
    assert cleanup_calls == 0
    assert await streaming_utils.await_bounded_owned_operation(
        asyncio.sleep(0, result="recovered"),
        timeout_seconds=0.1,
        timeout_message="unexpected timeout",
        on_abandoned=cleanup,
    ) == "recovered"


@pytest.mark.asyncio
async def test_owned_operation_cleanup_survives_cancel_success_callback_failure() -> None:
    """Best-effort usage marking cannot prevent mandatory abandoned cleanup."""
    release = asyncio.Event()
    started = asyncio.Event()
    cleanup_done = asyncio.Event()

    async def operation() -> None:
        started.set()
        await release.wait()

    async def failing_success_callback() -> None:
        raise RuntimeError("private credential marker")

    async def cleanup() -> None:
        cleanup_done.set()

    caller = asyncio.create_task(
        streaming_utils.await_bounded_owned_operation(
            operation(),
            timeout_seconds=10.0,
            timeout_message="unexpected timeout",
            on_abandoned=cleanup,
            on_cancel_success=failing_success_callback,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    release.set()
    await asyncio.wait_for(cleanup_done.wait(), timeout=1.0)


@pytest.mark.asyncio
async def test_daemon_timeout_exposes_true_worker_release_event() -> None:
    """Timeout callers can retain external resources through real worker exit."""
    pool = BoundedDaemonPool(capacity=1)
    worker_started = threading.Event()
    release_worker = threading.Event()
    worker_released = threading.Event()

    def blocked_call() -> None:
        worker_started.set()
        release_worker.wait(timeout=2.0)

    with pytest.raises(TimeoutError, match="daemon timed out"):
        await asyncio.to_thread(
            bounded_daemon.run_bounded_daemon_with_timeout,
            blocked_call,
            pool=pool,
            name="released-event-regression",
            timeout_seconds=0.01,
            timeout_message="daemon timed out",
            released_event=worker_released,
        )

    assert worker_started.is_set()
    assert worker_released.is_set() is False
    assert pool.active_count == 1
    release_worker.set()
    assert await asyncio.to_thread(worker_released.wait, 1.0)
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_daemon_drain_after_timeout_rejects_until_exit_then_raises() -> None:
    """The explicit owned-timeout mode drains release but never returns late data."""
    lifecycle: list[str] = []
    worker_started = threading.Event()
    release_worker = threading.Event()
    call_count = 0

    class TrackingPool(BoundedDaemonPool):
        def _release_capacity(self) -> None:
            lifecycle.append("capacity-release")
            super()._release_capacity()

    def blocked_call() -> str:
        nonlocal call_count
        call_count += 1
        lifecycle.append("provider-start")
        worker_started.set()
        release_worker.wait(timeout=2.0)
        lifecycle.append("provider-exit")
        return "late result must not escape"

    pool = TrackingPool(capacity=1)
    task = asyncio.create_task(
        bounded_daemon.await_bounded_daemon_with_timeout(
            blocked_call,
            pool=pool,
            name="drain-after-timeout",
            timeout_seconds=0.01,
            timeout_message="owned provider timed out",
            drain_after_timeout=True,
        )
    )
    try:
        assert await asyncio.to_thread(worker_started.wait, 1.0)
        await asyncio.sleep(0.03)
        assert task.done() is False
        assert pool.active_count == 1

        with pytest.raises(DaemonCapacityError):
            await bounded_daemon.await_bounded_daemon_with_timeout(
                blocked_call,
                pool=pool,
                name="capacity-rejected",
                timeout_seconds=0.01,
                timeout_message="must reject before dispatch",
                drain_after_timeout=True,
            )
        assert call_count == 1

        release_worker.set()
        with pytest.raises(TimeoutError, match="owned provider timed out"):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release_worker.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert pool.active_count == 0
    assert lifecycle == [
        "provider-start",
        "provider-exit",
        "capacity-release",
    ]


@pytest.mark.asyncio
async def test_daemon_rejects_conflicting_late_result_modes_before_admission() -> None:
    """Late-result modes are exclusive and invalid input never starts a worker."""
    call_count = 0

    def target() -> str:
        nonlocal call_count
        call_count += 1
        return "must not run"

    pool = BoundedDaemonPool(capacity=1)
    with pytest.raises(ValueError, match="cannot both retain and discard"):
        await bounded_daemon.await_bounded_daemon_with_timeout(
            target,
            pool=pool,
            name="conflicting-late-result-modes",
            timeout_seconds=0.01,
            timeout_message="must not time out",
            retain_result_after_timeout=True,
            drain_after_timeout=True,
        )

    assert call_count == 0
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_owned_sync_close_bypasses_saturated_default_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An owned close starts directly instead of waiting in the default executor."""

    loop = asyncio.get_running_loop()
    default_executor = ThreadPoolExecutor(max_workers=1)
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    close_calls: list[None] = []
    cleanup_pool = BoundedDaemonPool(capacity=1)
    monkeypatch.setattr(
        bounded_daemon,
        "STREAM_CLEANUP_DAEMON_POOL",
        cleanup_pool,
    )

    def block_default_executor() -> None:
        blocker_started.set()
        release_blocker.wait(timeout=2.0)

    def close() -> str:
        close_calls.append(None)
        return "closed"

    loop.set_default_executor(default_executor)
    blocker = loop.run_in_executor(None, block_default_executor)
    while not blocker_started.is_set():
        await asyncio.sleep(0)

    outcome: str
    try:
        outcome = await asyncio.wait_for(
            streaming_utils.invoke_owned_stream_close(close, timeout=0.1),
            timeout=0.5,
        )
    except asyncio.TimeoutError:
        outcome = "timed-out"
    finally:
        release_blocker.set()
        await blocker
        default_executor.shutdown(wait=True, cancel_futures=True)

    assert outcome == "closed"
    assert close_calls == [None]
    assert cleanup_pool.active_count == 0


@pytest.mark.asyncio
async def test_async_daemon_bridge_consumes_failure_when_cancelled_during_release_wait() -> None:
    """A completed failure stays private while the daemon lease is releasing."""

    sentinel = "async-daemon-release-secret-/srv/provider"
    target_finished = threading.Event()
    allow_release = threading.Event()
    release_wait_polled = threading.Event()
    loop_errors: list[dict[str, Any]] = []

    class SecretFailure(Exception):
        pass

    class TrackingReleaseEvent(threading.Event):
        def is_set(self) -> bool:
            release_wait_polled.set()
            return super().is_set()

    class GatedReleasePool(BoundedDaemonPool):
        def start(self, target, *, name, released_event=None):
            def gated_target() -> None:
                target()
                target_finished.set()
                allow_release.wait(timeout=2.0)

            return super().start(
                gated_target,
                name=name,
                released_event=released_event,
            )

    def fail() -> None:
        raise SecretFailure(sentinel)

    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    pool = GatedReleasePool(capacity=1)
    worker_released = TrackingReleaseEvent()
    caller = asyncio.create_task(
        bounded_daemon.await_bounded_daemon_with_timeout(
            fail,
            pool=pool,
            name="cancelled-release-wait",
            timeout_seconds=1.0,
            timeout_message="unexpected timeout",
            released_event=worker_released,
        )
    )
    try:
        while not release_wait_polled.is_set():
            await asyncio.sleep(0)
        assert target_finished.is_set()
        assert worker_released.is_set() is False
        assert caller.done() is False
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller

        allow_release.set()
        await _wait_for_pool_count(pool, 0)
        del caller
        for _ in range(3):
            gc.collect()
            await asyncio.sleep(0)
    finally:
        allow_release.set()
        loop.set_exception_handler(previous_handler)

    assert loop_errors == []
