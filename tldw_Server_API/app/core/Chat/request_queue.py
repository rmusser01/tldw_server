# request_queue.py
# Description: Request queuing system with backpressure and priority management
#
# Imports
import asyncio
import contextlib
import inspect
import threading
import time
from collections import deque
from concurrent.futures import TimeoutError
from dataclasses import dataclass, field
from enum import IntEnum
from functools import partial
from heapq import heappop, heappush
from typing import Any, Callable, Optional

from loguru import logger

from tldw_Server_API.app.core.Chat.bounded_daemon import (
    SYNC_ADAPTER_CALL_POOL,
    await_bounded_sync_call,
    start_bounded_stream_daemon,
)
from tldw_Server_API.app.core.Chat.streaming_utils import (
    cancel_stream_tasks_bounded,
    create_bounded_stream_task,
    invoke_stream_close_bounded,
    provider_stream_error_payload,
    sanitized_provider_stream_exception,
)
from tldw_Server_API.app.core.exceptions import raise_detached_error

REQUEST_QUEUE_STREAM_CANCEL_DRAIN_SECONDS = 0.05
REQUEST_QUEUE_STREAM_CLEANUP_TIMEOUT_SECONDS = 0.05
REQUEST_QUEUE_STREAM_DAEMON_POLL_SECONDS = 0.05
REQUEST_QUEUE_STREAM_HANDOFF_CLEANUP_SECONDS = 0.1
REQUEST_QUEUE_STREAM_TERMINAL_PUT_SECONDS = 0.1
_QUEUE_STREAM_ABANDONED = object()
_QUEUE_STREAM_TERMINAL_ABSENT = object()


@dataclass(frozen=True, slots=True)
class QueueStreamTerminalError:
    """One atomic terminal signal for a queued provider failure."""

    code: str


class QueueStreamChannel(asyncio.Queue[Any]):
    """Bound data chunks while reserving one out-of-band terminal slot."""

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)
        self._terminal_item: Any = _QUEUE_STREAM_TERMINAL_ABSENT
        self._terminal_ready = asyncio.Event()

    def put_terminal_nowait(self, item: QueueStreamTerminalError | None) -> bool:
        """Append exactly one terminal item outside bounded data capacity."""

        if self._terminal_item is not _QUEUE_STREAM_TERMINAL_ABSENT:
            return False
        self._terminal_item = item
        self._terminal_ready.set()
        return True

    def get_nowait(self) -> Any:
        """Return buffered data before the reserved terminal error."""

        if not super().empty():
            return super().get_nowait()
        if self._terminal_item is _QUEUE_STREAM_TERMINAL_ABSENT:
            raise asyncio.QueueEmpty
        terminal = self._terminal_item
        self._terminal_item = _QUEUE_STREAM_TERMINAL_ABSENT
        self._terminal_ready.clear()
        return terminal

    async def get(self) -> Any:
        """Wait for either buffered data or the reserved terminal error."""

        try:
            return self.get_nowait()
        except asyncio.QueueEmpty:
            pass

        data_task = asyncio.create_task(super().get())
        terminal_task = asyncio.create_task(self._terminal_ready.wait())
        try:
            await asyncio.wait(
                {data_task, terminal_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if data_task.done() and not data_task.cancelled():
                return data_task.result()
            data_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                return await data_task
            return self.get_nowait()
        finally:
            for task in (data_task, terminal_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(data_task, terminal_task, return_exceptions=True)

    def empty(self) -> bool:
        """Return whether neither data nor a terminal error is buffered."""

        return (
            super().empty()
            and self._terminal_item is _QUEUE_STREAM_TERMINAL_ABSENT
        )


_REQUEST_QUEUE_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


async def _await_queue_stream_operation(awaitable: Any, timeout: float) -> Any:
    """Await queue stream work with a hard bound and detached cancellation."""

    if timeout <= 0:
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        elif isinstance(awaitable, asyncio.Future):
            awaitable.cancel()
        raise asyncio.TimeoutError
    task = create_bounded_stream_task(awaitable)
    try:
        done, _ = await asyncio.wait({task}, timeout=max(0.0, timeout))
    except asyncio.CancelledError:
        await cancel_stream_tasks_bounded(
            [task],
            REQUEST_QUEUE_STREAM_CANCEL_DRAIN_SECONDS,
        )
        raise
    if task not in done:
        await cancel_stream_tasks_bounded(
            [task],
            REQUEST_QUEUE_STREAM_CANCEL_DRAIN_SECONDS,
        )
        raise asyncio.TimeoutError
    return task.result()


def _close_late_queue_stream(stream: Any) -> None:
    """Best-effort close a stream returned after its queued request was abandoned."""

    close = getattr(stream, "aclose", None)
    if not callable(close):
        close = getattr(stream, "close", None)
    if not callable(close):
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            asyncio.run(result)
    except BaseException as close_error:
        if isinstance(close_error, (KeyboardInterrupt, SystemExit)):
            raise
        logger.debug(
            "Late queued stream close failed error_type={}",
            type(close_error).__name__,
        )


def _observe_queue_cleanup_task(task: asyncio.Task[Any]) -> None:
    """Consume completion from a detached queue-result cleanup task."""

    with contextlib.suppress(asyncio.CancelledError, Exception):
        task.result()


async def _close_accepted_queue_result_after_release(
    stream: Any,
    worker_released: threading.Event,
) -> None:
    """Close a caller-owned factory result after its daemon lease is released."""

    deadline = time.monotonic() + REQUEST_QUEUE_STREAM_HANDOFF_CLEANUP_SECONDS
    while not worker_released.is_set():
        if time.monotonic() >= deadline:
            logger.warning("Queued stream cleanup missed worker handoff deadline")
            return
        await asyncio.sleep(0)
    close = getattr(stream, "aclose", None)
    if not callable(close):
        close = getattr(stream, "close", None)
    if not callable(close):
        return
    try:
        await invoke_stream_close_bounded(
            close,
            max(0.0, deadline - time.monotonic()),
        )
    except asyncio.TimeoutError:
        logger.warning("Queued stream handoff cleanup exceeded bounded timeout")
    except _REQUEST_QUEUE_NONCRITICAL_EXCEPTIONS as close_error:
        logger.debug(
            "Queued stream handoff cleanup failed error_type={}",
            type(close_error).__name__,
        )


async def _run_queue_stream_sync_daemon(
    fn: Callable[..., Any],
    *args: Any,
    should_abandon: Callable[[], bool],
    close_late_result: bool = False,
    cleanup_after_delivery: Callable[[], Any] | None = None,
    wait_for_worker_release: bool = False,
) -> Any:
    """Run sync streaming work in a disposable daemon, polling request liveness."""

    if should_abandon():
        return _QUEUE_STREAM_ABANDONED

    loop = asyncio.get_running_loop()
    result_future: asyncio.Future[Any] = loop.create_future()
    abandoned = threading.Event()
    accepted = threading.Event()
    decision_made = threading.Event()
    delivery_observed = threading.Event()
    close_late = threading.Event()
    worker_released = threading.Event()

    def deliver(value: Any = None, error: BaseException | None = None) -> None:
        try:
            if abandoned.is_set() or result_future.done():
                if close_late_result and error is None and value is not None:
                    close_late.set()
                return
            if error is not None:
                if isinstance(error, asyncio.CancelledError):
                    error = sanitized_provider_stream_exception(error)
                result_future.set_exception(error)
            else:
                result_future.set_result(value)
        finally:
            delivery_observed.set()

    def finish_delivery(value: Any = None) -> None:
        while not delivery_observed.wait(REQUEST_QUEUE_STREAM_DAEMON_POLL_SECONDS):
            if loop.is_closed():
                if close_late_result and value is not None:
                    _close_late_queue_stream(value)
                if cleanup_after_delivery is not None:
                    cleanup_after_delivery()
                return
        if wait_for_worker_release:
            while not decision_made.wait(REQUEST_QUEUE_STREAM_DAEMON_POLL_SECONDS):
                if loop.is_closed():
                    abandoned.set()
                    decision_made.set()
                    break
        if (close_late.is_set() or (abandoned.is_set() and not accepted.is_set())) and value is not None:
            _close_late_queue_stream(value)
        if cleanup_after_delivery is not None:
            cleanup_after_delivery()

    def worker() -> None:
        try:
            value = fn(*args)
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            try:
                loop.call_soon_threadsafe(deliver, None, error)
            except RuntimeError:
                if cleanup_after_delivery is not None:
                    cleanup_after_delivery()
                return
            finish_delivery()
        else:
            try:
                loop.call_soon_threadsafe(deliver, value, None)
            except RuntimeError:
                if close_late_result:
                    _close_late_queue_stream(value)
                if cleanup_after_delivery is not None:
                    cleanup_after_delivery()
                return
            finish_delivery(value)

    start_bounded_stream_daemon(
        worker,
        name="queued-stream-sync-work",
        released_event=worker_released,
    )
    try:
        while True:
            done, _ = await asyncio.wait(
                {result_future},
                timeout=REQUEST_QUEUE_STREAM_DAEMON_POLL_SECONDS,
            )
            if result_future in done:
                result = result_future.result()
                if wait_for_worker_release:
                    if should_abandon():
                        abandoned.set()
                        decision_made.set()
                        return _QUEUE_STREAM_ABANDONED
                    accepted.set()
                    decision_made.set()
                    while not worker_released.is_set():
                        if should_abandon():
                            cleanup_task = asyncio.create_task(
                                _close_accepted_queue_result_after_release(
                                    result,
                                    worker_released,
                                )
                            )
                            cleanup_task.add_done_callback(_observe_queue_cleanup_task)
                            return _QUEUE_STREAM_ABANDONED
                        await asyncio.sleep(0)
                return result
            if should_abandon():
                abandoned.set()
                decision_made.set()
                return _QUEUE_STREAM_ABANDONED
    except asyncio.CancelledError:
        if accepted.is_set() and "result" in locals():
            cleanup_task = asyncio.create_task(
                _close_accepted_queue_result_after_release(result, worker_released)
            )
            cleanup_task.add_done_callback(_observe_queue_cleanup_task)
        else:
            abandoned.set()
            decision_made.set()
        raise
    except BaseException:
        abandoned.set()
        decision_made.set()
        if wait_for_worker_release:
            release_deadline = (
                time.monotonic() + REQUEST_QUEUE_STREAM_HANDOFF_CLEANUP_SECONDS
            )
            while not worker_released.is_set():
                if should_abandon() or time.monotonic() >= release_deadline:
                    logger.warning(
                        "Queued stream worker release exceeded bounded error handoff"
                    )
                    break
                await asyncio.sleep(0)
        raise

#######################################################################################################################
#
# Types:

class RequestPriority(IntEnum):
    """Request priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

@dataclass(order=True)
class QueuedRequest:
    """Represents a queued request with priority."""
    priority: int
    timestamp: float = field(compare=False)
    request_id: str = field(compare=False)
    request_data: Any = field(compare=False)
    future: asyncio.Future = field(compare=False)
    client_id: str = field(compare=False)
    estimated_tokens: int = field(compare=False, default=0)
    # Optional processor for actual work execution
    processor: Optional[Callable[..., Any]] = field(compare=False, default=None)
    processor_args: tuple[Any, ...] = field(compare=False, default_factory=tuple)
    processor_kwargs: dict[str, Any] = field(compare=False, default_factory=dict)
    streaming: bool = field(compare=False, default=False)
    # For streaming jobs, a channel to emit provider chunks (bytes or str). Sentinel None indicates end.
    stream_channel: Optional[asyncio.Queue] = field(compare=False, default=None)
    stream_factory_timeout: float | None = field(compare=False, default=None)
    stream_terminal_error_emitted: bool = field(compare=False, default=False)


async def _emit_stream_terminal_error(
    request: QueuedRequest,
    value: Any = "provider_unavailable",
    *,
    preserve_buffered: bool = False,
) -> None:
    """Emit one fail-closed terminal signal, optionally after buffered output."""

    channel = request.stream_channel
    if (
        not request.streaming
        or channel is None
        or request.stream_terminal_error_emitted
    ):
        return
    payload = provider_stream_error_payload(value)
    code = payload["error"]["code"]
    terminal = QueueStreamTerminalError(code=code)
    request.stream_terminal_error_emitted = True
    if not preserve_buffered:
        while True:
            try:
                channel.get_nowait()
            except asyncio.QueueEmpty:
                break
    put_terminal_nowait = getattr(channel, "put_terminal_nowait", None)
    if callable(put_terminal_nowait):
        put_terminal_nowait(terminal)
        return
    if preserve_buffered:
        try:
            await asyncio.wait_for(
                channel.put(terminal),
                timeout=REQUEST_QUEUE_STREAM_TERMINAL_PUT_SECONDS,
            )
        except (asyncio.TimeoutError, RuntimeError):
            logger.warning("Queued terminal error append exceeded bounded timeout")
        return
    with contextlib.suppress(asyncio.QueueFull, RuntimeError):
        channel.put_nowait(terminal)


async def _terminalize_cancelled_request(request: QueuedRequest) -> None:
    """Cancel a queued future and fail closed for any streaming consumer."""

    if not request.future.done():
        request.future.cancel()
    await _emit_stream_terminal_error(request)

#######################################################################################################################
#
# Classes:

class RequestQueue:
    """
    Priority-based request queue with backpressure management.
    """

    def __init__(
        self,
        max_queue_size: int = 100,
        max_concurrent: int = 10,
        timeout: float = 300.0
    ):
        """
        Initialize the request queue.

        Args:
            max_queue_size: Maximum number of queued requests
            max_concurrent: Maximum concurrent processing
            timeout: Request timeout in seconds
        """
        self.max_queue_size = max_queue_size
        self.max_concurrent = max_concurrent
        self.timeout = timeout

        self.queue = []  # Priority queue
        self.processing_count = 0
        self.total_processed = 0
        self.total_rejected = 0
        # Track active request IDs to prevent duplicates
        self._active_request_ids: set = set()

        self._lock = asyncio.Lock()
        self._processing_semaphore = asyncio.Semaphore(max_concurrent)
        self._workers = []
        self._running = False
        # Lifecycle contract: queue instances are one-shot.
        # After stop(), create a fresh instance via initialize_request_queue().
        self._stopped = False
        # Rolling recent activity (last N jobs)
        self._recent_activity = deque(maxlen=200)
        # Event to wake workers when new items arrive (avoids polling delay)
        self._has_items = asyncio.Event()
    async def start(self, num_workers: int = 4):
        """
        Start the queue workers.

        Args:
            num_workers: Number of worker tasks
        """
        if self._running:
            return
        if self._stopped:
            raise RuntimeError(
                "RequestQueue cannot be restarted after stop(); "
                "initialize a new queue instance instead."
            )

        self._running = True
        # Ensure event starts cleared
        self._has_items.clear()
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)

        logger.info("Started {} queue workers", num_workers)

    async def stop(self):
        """Stop the queue workers."""
        if self._stopped:
            return
        self._running = False

        async with self._lock:
            pending_requests = list(self.queue)
            self.queue.clear()
            self._has_items.clear()
            for request in pending_requests:
                self._active_request_ids.discard(request.request_id)
        for request in pending_requests:
            await _terminalize_cancelled_request(request)

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._stopped = True

        logger.info("Stopped queue workers")

    def is_running(self) -> bool:
        """Return True if the queue has active worker tasks processing items."""
        if not self._running:
            return False
        alive = False
        for worker in list(self._workers):
            try:
                if not worker.done():
                    alive = True
                    break
            except _REQUEST_QUEUE_NONCRITICAL_EXCEPTIONS:
                alive = True
                break
        if not alive:
            self._running = False
        return alive

    async def _worker(self, worker_id: str):
        """
        Worker task that processes queued requests.

        Args:
            worker_id: Worker identifier
        """
        logger.debug("Worker {} started", worker_id)

        # Timeout for waiting on empty queue (60 seconds) - prevents indefinite waits
        wait_timeout = 60.0

        while self._running:
            request: QueuedRequest | None = None
            try:
                # Get next request from queue
                request = await self._get_next_request()
                if not request:
                    # No requests; wait until enqueued instead of polling, with timeout
                    try:
                        await asyncio.wait_for(self._has_items.wait(), timeout=wait_timeout)
                    except asyncio.TimeoutError:
                        # Timeout is expected when queue is idle; just continue loop
                        continue
                    # Loop will attempt to fetch again
                    continue

                # Check if request has timed out
                if time.time() - request.timestamp > self.timeout:
                    logger.warning(f"Request {request.request_id} timed out in queue")
                    try:
                        request.future.set_exception(
                            TimeoutError(f"Request timed out after {self.timeout}s in queue")
                        )
                    except asyncio.InvalidStateError:
                        logger.debug(f"Future already resolved for timed-out request {request.request_id}")
                    await _emit_stream_terminal_error(request)
                    # Clean up request ID for timed-out requests
                    async with self._lock:
                        self._active_request_ids.discard(request.request_id)
                    continue

                # Process request
                async with self._processing_semaphore:
                    self.processing_count += 1
                    process_succeeded = False
                    try:
                        # Check if the request was cancelled before starting
                        if request.future.cancelled():
                            logger.info(f"Request {request.request_id} was cancelled before processing")
                            process_succeeded = True  # Count as processed (client initiated cancel)
                            continue

                        # Execute the actual request processing
                        result = await self._process_request(request)

                        # Check if cancelled during processing
                        if request.future.cancelled():
                            logger.info(f"Request {request.request_id} was cancelled during processing")
                            process_succeeded = True
                            continue

                        try:
                            request.future.set_result(result)
                            process_succeeded = True
                        except asyncio.InvalidStateError:
                            # Future was already resolved (e.g., cancelled by client)
                            logger.debug(f"Future already resolved for request {request.request_id}")
                            process_succeeded = True  # Still count as processed
                    except asyncio.CancelledError:
                        # Request was cancelled - propagate but count as handled
                        logger.info(f"Request {request.request_id} processing was cancelled")
                        process_succeeded = True
                        raise
                    except BaseException as e:
                        if isinstance(e, (KeyboardInterrupt, SystemExit)):
                            raise
                        logger.error(
                            "Error processing request {} error_type={}",
                            request.request_id,
                            type(e).__name__,
                        )
                        try:
                            if not request.future.cancelled():
                                request.future.set_exception(e)
                        except asyncio.InvalidStateError:
                            logger.debug(f"Future already resolved for failed request {request.request_id}")
                    finally:
                        self.processing_count -= 1
                        # Update total_processed only when process completed (success or client cancelled)
                        if process_succeeded:
                            self.total_processed += 1
                        # Remove request ID from active tracking
                        async with self._lock:
                            self._active_request_ids.discard(request.request_id)

            except asyncio.CancelledError:
                if request is not None:
                    await _terminalize_cancelled_request(request)
                    self._active_request_ids.discard(request.request_id)
                break
            except BaseException as e:
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                logger.error(
                    "Worker {} error_type={}",
                    worker_id,
                    type(e).__name__,
                )
                await asyncio.sleep(1)

        logger.debug("Worker {} stopped", worker_id)

    async def _get_next_request(self) -> Optional[QueuedRequest]:
        """Get the next request from the priority queue."""
        async with self._lock:
            if self.queue:
                item = heappop(self.queue)
                # If queue becomes empty after pop, clear the wake event
                if not self.queue:
                    self._has_items.clear()
                return item
        return None

    async def _process_request(self, request: QueuedRequest) -> Any:
        """
        Process a request (placeholder for actual processing).

        Args:
            request: The request to process

        Returns:
            Processing result
        """
        # If a processor is provided, execute it; otherwise perform placeholder work
        start_ts = time.time()
        if request.processor is None:
            logger.debug(
                "Processing request {} (no processor; admission-only)",
                request.request_id,
            )
            duration = time.time() - start_ts
            # record activity
            self._recent_activity.append({
                "request_id": request.request_id,
                "client_id": request.client_id,
                "priority": request.priority,
                "streaming": request.streaming,
                "duration": duration,
                "result": "completed",
                "ts": time.time(),
            })
            return {"status": "completed", "request_id": request.request_id}

        logger.debug(
            "Processing request {} with processor; streaming={}",
            request.request_id,
            request.streaming,
        )
        loop = asyncio.get_running_loop()

        # Non-streaming provider work shares the process-wide adapter capacity cap.
        if not request.streaming:
            try:
                fn = partial(
                    request.processor,
                    *request.processor_args,
                    **request.processor_kwargs,
                )

                def _invoke_sync_processor() -> Any:
                    try:
                        return fn()
                    except asyncio.CancelledError as provider_cancel:
                        raise_detached_error(
                            sanitized_provider_stream_exception(provider_cancel)
                        )

                result = await await_bounded_sync_call(
                    _invoke_sync_processor,
                    pool=SYNC_ADAPTER_CALL_POOL,
                    exhaustion_message="Provider adapter capacity is exhausted",
                )
                duration = time.time() - start_ts
                self._recent_activity.append({
                    "request_id": request.request_id,
                    "client_id": request.client_id,
                    "priority": request.priority,
                    "streaming": False,
                    "duration": duration,
                    "result": "completed",
                    "ts": time.time(),
                })
                return result
            except Exception as e:
                logger.error(
                    "Processor error for request {} error_type={}",
                    request.request_id,
                    type(e).__name__,
                )
                self._recent_activity.append({
                    "request_id": request.request_id,
                    "client_id": request.client_id,
                    "priority": request.priority,
                    "streaming": False,
                    "duration": time.time() - start_ts,
                    "result": "error",
                    "error_type": type(e).__name__,
                    "ts": time.time(),
                })
                raise

        # Streaming path: processor should return an iterator (sync or async) that yields chunks
        if request.stream_channel is None:
            logger.error(f"Streaming job {request.request_id} missing stream_channel")
            raise RuntimeError("Streaming channel not provided for streaming job")

        async def _put_async_with_backpressure(item: Any, *, terminal: bool = False) -> bool:
            """Put an item onto the stream channel with cancellation-aware backpressure.

            QueueStreamChannel reserves one terminal slot, so completion never
            waits behind accepted data or loses its sole terminal sentinel.
            """
            if terminal:
                put_terminal_nowait = getattr(
                    request.stream_channel,
                    "put_terminal_nowait",
                    None,
                )
                if callable(put_terminal_nowait):
                    return bool(put_terminal_nowait(item))
            while True:
                if not terminal and (request.future.cancelled() or loop.is_closed() or not self._running):
                    return False
                if terminal and (loop.is_closed() or not self._running):
                    return False
                try:
                    timeout_s = 0.25 if terminal else 1.0
                    await asyncio.wait_for(request.stream_channel.put(item), timeout=timeout_s)
                    return True
                except asyncio.TimeoutError:
                    if terminal:
                        return False
                    # Keep waiting while consumer is alive; cancellation checks at loop top.
                    continue
                except _REQUEST_QUEUE_NONCRITICAL_EXCEPTIONS as ch_e:
                    logger.warning(
                        "Failed to enqueue stream chunk for {} error_type={}",
                        request.request_id,
                        type(ch_e).__name__,
                    )
                    return False

        async def _pump_async_iterator(async_iter):
            aiter = async_iter.__aiter__() if hasattr(async_iter, "__aiter__") else async_iter
            next_task: asyncio.Future[Any] | None = None
            try:
                while True:
                    if request.future.cancelled() or loop.is_closed() or not self._running:
                        break
                    try:
                        next_task = create_bounded_stream_task(aiter.__anext__())
                        while True:
                            done, _ = await asyncio.wait(
                                {next_task},
                                timeout=REQUEST_QUEUE_STREAM_CANCEL_DRAIN_SECONDS,
                            )
                            if next_task in done:
                                try:
                                    chunk = next_task.result()
                                except asyncio.CancelledError as provider_cancel:
                                    worker_task = asyncio.current_task()
                                    if (
                                        worker_task is not None
                                        and worker_task.cancelling()
                                    ):
                                        raise
                                    raise_detached_error(
                                        sanitized_provider_stream_exception(
                                            provider_cancel
                                        )
                                    )
                                break
                            if (
                                request.future.cancelled()
                                or loop.is_closed()
                                or not self._running
                            ):
                                await cancel_stream_tasks_bounded(
                                    [next_task],
                                    REQUEST_QUEUE_STREAM_CANCEL_DRAIN_SECONDS,
                                )
                                next_task = None
                                return
                    except StopAsyncIteration:
                        break
                    if request.future.cancelled() or loop.is_closed() or not self._running:
                        break
                    if not await _put_async_with_backpressure(chunk):
                        break
            finally:
                # Ensure async iterators are closed on cancellation or early exit
                if next_task is not None and not next_task.done():
                    await cancel_stream_tasks_bounded(
                        [next_task],
                        REQUEST_QUEUE_STREAM_CANCEL_DRAIN_SECONDS,
                    )
                try:
                    aclose = getattr(aiter, "aclose", None)
                    if callable(aclose):
                        await invoke_stream_close_bounded(
                            aclose,
                            REQUEST_QUEUE_STREAM_CLEANUP_TIMEOUT_SECONDS,
                        )
                except asyncio.TimeoutError:
                    logger.debug(
                        "Queued async stream close exceeded bounded timeout request={}",
                        request.request_id,
                    )
                except _REQUEST_QUEUE_NONCRITICAL_EXCEPTIONS as close_error:
                    logger.debug(
                        "Queued async stream close failed request={} error_type={}",
                        request.request_id,
                        type(close_error).__name__,
                    )

        def _pump_sync_iterator(sync_iter):
            def _put_with_backpressure(item: Any) -> bool:
                try:
                    fut = asyncio.run_coroutine_threadsafe(request.stream_channel.put(item), loop)
                except _REQUEST_QUEUE_NONCRITICAL_EXCEPTIONS as ch_e:
                    logger.warning(
                        "Failed to enqueue stream chunk (sync) for {} error_type={}",
                        request.request_id,
                        type(ch_e).__name__,
                    )
                    return False
                while True:
                    try:
                        fut.result(timeout=1.0)
                        return True
                    except TimeoutError:
                        if request.future.cancelled() or loop.is_closed() or not self._running:
                            with contextlib.suppress(_REQUEST_QUEUE_NONCRITICAL_EXCEPTIONS):
                                fut.cancel()
                            return False
                    except _REQUEST_QUEUE_NONCRITICAL_EXCEPTIONS as ch_e:
                        logger.warning(
                            "Failed to enqueue stream chunk (sync) for {} error_type={}",
                            request.request_id,
                            type(ch_e).__name__,
                        )
                        return False

            for chunk in sync_iter:
                if not _put_with_backpressure(chunk):
                    return None
            return None

        # Run the processor to obtain the stream (potentially blocking)
        try:
            fn = partial(
                request.processor,
                *request.processor_args,
                **request.processor_kwargs,
            )
            stream_factory_deadline: float | None = None

            def dispatch_stream_factory() -> Any:
                nonlocal stream_factory_deadline
                if request.stream_factory_timeout is not None:
                    stream_factory_deadline = time.monotonic() + max(
                        0.0,
                        request.stream_factory_timeout,
                    )
                return fn()

            stream = await _run_queue_stream_sync_daemon(
                dispatch_stream_factory,
                should_abandon=lambda: (
                    request.future.cancelled()
                    or loop.is_closed()
                    or not self._running
                    or (
                        stream_factory_deadline is not None
                        and time.monotonic() >= stream_factory_deadline
                    )
                ),
                close_late_result=True,
                wait_for_worker_release=True,
            )
            if stream is _QUEUE_STREAM_ABANDONED:
                if (
                    stream_factory_deadline is not None
                    and time.monotonic() >= stream_factory_deadline
                    and not request.future.cancelled()
                    and self._running
                ):
                    raise TimeoutError("Provider stream factory timed out")
                with contextlib.suppress(_REQUEST_QUEUE_NONCRITICAL_EXCEPTIONS):
                    await _put_async_with_backpressure(None, terminal=True)
                return {
                    "status": "stream_cancelled",
                    "request_id": request.request_id,
                }
        except Exception as e:
            await _emit_stream_terminal_error(request, e)
            logger.error(
                "Processor error starting stream for {} error_type={}",
                request.request_id,
                type(e).__name__,
            )
            self._recent_activity.append({
                "request_id": request.request_id,
                "client_id": request.client_id,
                "priority": request.priority,
                "streaming": True,
                "duration": time.time() - start_ts,
                "result": "error",
                "error_type": type(e).__name__,
                "ts": time.time(),
            })
            raise

        # Pump stream depending on iterator type
        try:
            if hasattr(stream, "__aiter__"):
                await _pump_async_iterator(stream)
            else:
                # Sync iterator; run pumping in thread
                pump_result = await _run_queue_stream_sync_daemon(
                    _pump_sync_iterator,
                    stream,
                    should_abandon=lambda: (
                        request.future.cancelled()
                        or loop.is_closed()
                        or not self._running
                    ),
                    cleanup_after_delivery=lambda: _close_late_queue_stream(stream),
                )
                if pump_result is _QUEUE_STREAM_ABANDONED:
                    with contextlib.suppress(_REQUEST_QUEUE_NONCRITICAL_EXCEPTIONS):
                        await _put_async_with_backpressure(None, terminal=True)
                    return {
                        "status": "stream_cancelled",
                        "request_id": request.request_id,
                    }
            with contextlib.suppress(_REQUEST_QUEUE_NONCRITICAL_EXCEPTIONS):
                await _put_async_with_backpressure(None, terminal=True)
            # For streaming jobs, return a simple status when pumping completes
            duration = time.time() - start_ts
            self._recent_activity.append({
                "request_id": request.request_id,
                "client_id": request.client_id,
                "priority": request.priority,
                "streaming": True,
                "duration": duration,
                "result": "stream_completed",
                "ts": time.time(),
            })
            return {"status": "stream_completed", "request_id": request.request_id}
        except Exception as e:
            await _emit_stream_terminal_error(request, e, preserve_buffered=True)
            logger.error(
                "Streaming processor error for {} error_type={}",
                request.request_id,
                type(e).__name__,
            )
            self._recent_activity.append({
                "request_id": request.request_id,
                "client_id": request.client_id,
                "priority": request.priority,
                "streaming": True,
                "duration": time.time() - start_ts,
                "result": "error",
                "error_type": type(e).__name__,
                "ts": time.time(),
            })
            raise

    async def enqueue(
        self,
        request_id: str,
        request_data: Any,
        client_id: str,
        priority: RequestPriority = RequestPriority.NORMAL,
        estimated_tokens: int = 0,
        *,
        processor: Optional[Callable[..., Any]] = None,
        processor_args: tuple[Any, ...] = (),
        processor_kwargs: Optional[dict[str, Any]] = None,
        streaming: bool = False,
        stream_channel: Optional[asyncio.Queue] = None,
        stream_factory_timeout: float | None = None,
    ) -> asyncio.Future:
        """
        Add a request to the queue.

        Args:
            request_id: Unique request identifier
            request_data: The request data
            client_id: Client identifier
            priority: Request priority
            estimated_tokens: Estimated token count for the request

        Returns:
            Future that will contain the result

        Raises:
            ValueError: If queue is full or request ID is duplicate
        """
        async with self._lock:
            # Check for duplicate request ID
            if request_id in self._active_request_ids:
                raise ValueError(f"Duplicate request ID: {request_id}")

            # Check queue size (backpressure)
            if len(self.queue) >= self.max_queue_size:
                self.total_rejected += 1
                raise ValueError(f"Queue full: {len(self.queue)} requests pending")

            # Track the request ID
            self._active_request_ids.add(request_id)

            # Create queued request
            future = asyncio.Future()
            if processor_kwargs is None:
                processor_kwargs = {}
            request = QueuedRequest(
                priority=priority.value,
                timestamp=time.time(),
                request_id=request_id,
                request_data=request_data,
                future=future,
                client_id=client_id,
                estimated_tokens=estimated_tokens,
                processor=processor,
                processor_args=processor_args,
                processor_kwargs=processor_kwargs,
                streaming=streaming,
                stream_channel=stream_channel,
                stream_factory_timeout=(
                    max(0.0, float(stream_factory_timeout))
                    if streaming and stream_factory_timeout is not None
                    else None
                ),
            )

            # Add to priority queue
            heappush(self.queue, request)
            # Signal workers that items are available
            self._has_items.set()

            logger.debug(
                "Enqueued request {} with priority {}",
                request_id,
                priority.name,
            )

        return future

    def get_queue_status(self) -> dict[str, Any]:
        """
        Get current queue status.

        Returns:
            Dictionary with queue statistics
        """
        return {
            "queue_size": len(self.queue),
            "processing_count": self.processing_count,
            "max_queue_size": self.max_queue_size,
            "max_concurrent": self.max_concurrent,
            "total_processed": self.total_processed,
            "total_rejected": self.total_rejected,
            "is_running": self._running
        }

    def get_recent_activity(self, limit: Optional[int] = None) -> Any:
        """Return recent processed job summaries (most recent last)."""
        items = list(self._recent_activity)
        if limit is not None:
            items = items[-int(limit):]
        return items

    async def clear_queue(self):
        """Clear all pending requests."""
        async with self._lock:
            # Cancel all pending requests
            pending = list(self.queue)
            for request in pending:
                self._active_request_ids.discard(request.request_id)
            self.queue.clear()
            self._has_items.clear()
            logger.info("Cleared request queue")
        for request in pending:
            await _terminalize_cancelled_request(request)


class RateLimitedQueue(RequestQueue):
    """
    Request queue with rate limiting per client and globally.
    """

    def __init__(
        self,
        max_queue_size: int = 100,
        max_concurrent: int = 10,
        timeout: float = 300.0,
        global_rate_limit: int = 60,  # requests per minute
        per_client_rate_limit: int = 20  # requests per minute per client
    ):
        """
        Initialize rate-limited queue.

        Args:
            max_queue_size: Maximum queue size
            max_concurrent: Maximum concurrent processing
            timeout: Request timeout
            global_rate_limit: Global requests per minute
            per_client_rate_limit: Per-client requests per minute
        """
        super().__init__(max_queue_size, max_concurrent, timeout)

        self.global_rate_limit = global_rate_limit
        self.per_client_rate_limit = per_client_rate_limit

        # Track request times for rate limiting
        self.global_request_times = []
        self.client_request_times = {}
        # Lock for thread-safe rate limit state modifications
        self._rate_limit_lock = asyncio.Lock()

    async def _reserve_rate_limit(self, client_id: str) -> Optional[float]:
        """
        Reserve a rate limit slot for this client.

        This method is thread-safe and uses locking to prevent race conditions.
        It mutates the rate-limit state on success and returns the reservation
        timestamp so callers can roll back if downstream admission fails.

        Args:
            client_id: Client identifier

        Returns:
            Reservation timestamp if within limits, None otherwise
        """
        current_time = time.time()
        minute_ago = current_time - 60

        async with self._rate_limit_lock:
            # Clean old entries
            self.global_request_times = [
                t for t in self.global_request_times if t > minute_ago
            ]

            if client_id in self.client_request_times:
                self.client_request_times[client_id] = [
                    t for t in self.client_request_times[client_id] if t > minute_ago
                ]

            # Check global rate limit
            if len(self.global_request_times) >= self.global_rate_limit:
                return None

            # Check per-client rate limit
            client_requests = self.client_request_times.get(client_id, [])
            if len(client_requests) >= self.per_client_rate_limit:
                return None

            # Record request time
            self.global_request_times.append(current_time)
            if client_id not in self.client_request_times:
                self.client_request_times[client_id] = []
            self.client_request_times[client_id].append(current_time)

            return current_time

    async def _rollback_rate_limit(self, client_id: str, reservation_ts: float) -> None:
        """Rollback a previously reserved rate-limit slot."""
        async with self._rate_limit_lock:
            with contextlib.suppress(ValueError):
                self.global_request_times.remove(reservation_ts)
            client_times = self.client_request_times.get(client_id, [])
            with contextlib.suppress(ValueError):
                client_times.remove(reservation_ts)
            if client_times:
                self.client_request_times[client_id] = client_times
            else:
                self.client_request_times.pop(client_id, None)

    async def enqueue(
        self,
        request_id: str,
        request_data: Any,
        client_id: str,
        priority: RequestPriority = RequestPriority.NORMAL,
        estimated_tokens: int = 0,
        *,
        processor: Optional[Callable[..., Any]] = None,
        processor_args: tuple[Any, ...] = (),
        processor_kwargs: Optional[dict[str, Any]] = None,
        streaming: bool = False,
        stream_channel: Optional[asyncio.Queue] = None,
        stream_factory_timeout: float | None = None,
    ) -> asyncio.Future:
        """
        Add a request to the queue with rate limiting.

        Args:
            request_id: Unique request identifier
            request_data: The request data
            client_id: Client identifier
            priority: Request priority
            estimated_tokens: Estimated token count
            processor: Optional callable executed when the request is serviced
            processor_args: Positional args for the processor
            processor_kwargs: Keyword args for the processor
            streaming: Whether the request expects streaming output
            stream_channel: Channel used to emit streaming chunks

        Returns:
            Future that will contain the result

        Raises:
            ValueError: If queue is full or rate limit exceeded
        """
        # Reserve rate limit capacity (async with locking)
        reservation = await self._reserve_rate_limit(client_id)
        if reservation is None:
            raise ValueError(f"Rate limit exceeded for client {client_id}")

        if processor_kwargs is None:
            processor_kwargs = {}

        # Proceed with normal enqueue; roll back reservation if admission fails
        try:
            return await super().enqueue(
                request_id,
                request_data,
                client_id,
                priority,
                estimated_tokens,
                processor=processor,
                processor_args=processor_args,
                processor_kwargs=processor_kwargs,
                streaming=streaming,
                stream_channel=stream_channel,
                stream_factory_timeout=stream_factory_timeout,
            )
        except Exception:
            await self._rollback_rate_limit(client_id, reservation)
            raise


# Global queue instance
_request_queue: Optional[RateLimitedQueue] = None

def get_request_queue() -> Optional[RateLimitedQueue]:
    """Get the global request queue instance."""
    return _request_queue

def initialize_request_queue(
    max_queue_size: int = 100,
    max_concurrent: int = 10,
    global_rate_limit: int = 60,
    per_client_rate_limit: int = 20
) -> RateLimitedQueue:
    """
    Initialize the global request queue.

    Args:
        max_queue_size: Maximum queue size
        max_concurrent: Maximum concurrent processing
        global_rate_limit: Global rate limit
        per_client_rate_limit: Per-client rate limit

    Returns:
        The initialized queue
    """
    global _request_queue
    _request_queue = RateLimitedQueue(
        max_queue_size=max_queue_size,
        max_concurrent=max_concurrent,
        global_rate_limit=global_rate_limit,
        per_client_rate_limit=per_client_rate_limit
    )
    return _request_queue
