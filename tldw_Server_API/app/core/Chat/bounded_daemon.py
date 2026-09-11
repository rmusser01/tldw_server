"""Capacity-bounded daemon workers for non-cooperative streaming operations."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import math
import os
import threading
from collections.abc import Awaitable, Callable, Iterator
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any


class DaemonCapacityError(RuntimeError):
    """Raised when detached streaming work cannot be admitted safely."""


async def _drain_owned_task(task: asyncio.Future[Any]) -> tuple[bool, Any]:
    """Wait through repeated caller cancellation and consume *task*'s outcome."""

    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:  # noqa: BLE001 - arbitrary owned-worker failures are consumed
            break

    if task.cancelled():
        return False, None
    if task.exception() is not None:
        return False, None
    return True, task.result()


async def await_owned_worker(
    awaitable: Awaitable[Any],
    *,
    on_cancel_success: Callable[[], Awaitable[None] | None] | None = None,
    on_cancel_result: Callable[[Any], Awaitable[None] | None] | None = None,
) -> Any:
    """Keep owned work alive through caller cancellation, then re-raise it.

    This is for work whose surrounding async scope owns resources used by a
    non-cooperative worker.  Shielding prevents cancellation from abandoning
    that worker; draining keeps the resource scope open until it really exits.
    """

    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is None or current.cancelling() == 0:
            # The owned task itself was cancelled rather than its caller.
            raise

        succeeded, result = await _drain_owned_task(task)
        if succeeded and on_cancel_result is not None:
            # Late-result cleanup is deliberately best effort during cancellation.
            with contextlib.suppress(Exception):
                callback_result = on_cancel_result(result)
                if inspect.isawaitable(callback_result):
                    await _drain_owned_task(asyncio.ensure_future(callback_result))
        if succeeded and on_cancel_success is not None:
            # Credential usage marking is deliberately best effort during cancellation.
            with contextlib.suppress(Exception):
                callback_result = on_cancel_success()
                if inspect.isawaitable(callback_result):
                    await _drain_owned_task(asyncio.ensure_future(callback_result))
        raise


class BoundedDaemonPool:
    """Admit owned work only while fixed process-local capacity is available."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or int(capacity) < 1:
            raise ValueError("Daemon worker capacity must be a positive integer")
        self.capacity = int(capacity)
        self._semaphore = threading.BoundedSemaphore(self.capacity)
        self._state_lock = threading.Lock()
        self._active_count = 0

    @property
    def active_count(self) -> int:
        """Return the number of admitted operations that have not actually exited."""

        with self._state_lock:
            return self._active_count

    def _acquire_capacity(self, exhaustion_message: str) -> None:
        """Acquire one capacity slot or fail before the operation is dispatched."""

        if not self._semaphore.acquire(blocking=False):
            raise DaemonCapacityError(exhaustion_message)

        with self._state_lock:
            self._active_count += 1

    def _release_capacity(self) -> None:
        """Release one capacity slot after its admitted operation really exits."""

        with self._state_lock:
            self._active_count -= 1
        self._semaphore.release()

    @contextlib.contextmanager
    def lease(self, *, exhaustion_message: str) -> Iterator[None]:
        """Hold one admission slot for the complete lifetime of owned work."""

        self._acquire_capacity(exhaustion_message)
        try:
            yield
        finally:
            self._release_capacity()

    def start(
        self,
        target: Callable[[], Any],
        *,
        name: str,
        released_event: threading.Event | None = None,
        exhaustion_message: str = "Streaming worker capacity is exhausted",
    ) -> threading.Thread:
        """Start *target* as a daemon or fail closed without invoking it."""

        self._acquire_capacity(exhaustion_message)

        def guarded_target() -> None:
            try:
                target()
            finally:
                self._release_capacity()
                if released_event is not None:
                    released_event.set()

        try:
            thread = threading.Thread(target=guarded_target, name=name, daemon=True)
            thread.start()
        except BaseException:
            self._release_capacity()
            if released_event is not None:
                released_event.set()
            raise
        return thread


async def await_bounded_sync_call(
    call: Callable[[], Any],
    *,
    pool: BoundedDaemonPool,
    exhaustion_message: str,
    on_cancel_result: Callable[[Any], Awaitable[None] | None] | None = None,
) -> Any:
    """Dispatch one bounded sync call directly and drain it on cancellation."""

    result: Future[Any] = Future()
    released = threading.Event()

    def invoke() -> None:
        try:
            value = call()
        except BaseException as exc:  # noqa: BLE001 - every worker outcome resolves the future
            result.set_exception(exc)
        else:
            result.set_result(value)

    pool.start(
        invoke,
        name="provider-sync-adapter",
        released_event=released,
        exhaustion_message=exhaustion_message,
    )

    async def await_result_and_release() -> Any:
        loop_result = asyncio.wrap_future(result)
        try:
            return await loop_result
        finally:
            # The result is published inside the target, before start()'s
            # guarded cleanup releases capacity.  Preserve that ownership
            # boundary for successful, exceptional, and cancelled callers.
            while not released.is_set():
                await asyncio.sleep(0.001)

    return await await_owned_worker(
        await_result_and_release(),
        on_cancel_result=on_cancel_result,
    )


def run_bounded_daemon_with_timeout(
    call: Callable[[], Any],
    *,
    pool: BoundedDaemonPool,
    name: str,
    timeout_seconds: float,
    timeout_message: str,
    released_event: threading.Event | None = None,
) -> Any:
    """Run one sync call with a deadline while retaining capacity until real exit."""

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("Daemon call timeout must be a positive finite number")

    result: Future[Any] = Future()

    def invoke() -> None:
        try:
            value = call()
        except BaseException as exc:  # noqa: BLE001 - every worker outcome must resolve the future
            result.set_exception(exc)
        else:
            result.set_result(value)

    released = released_event or threading.Event()
    try:
        pool.start(invoke, name=name, released_event=released)
    except BaseException:
        # Capacity rejection happens before BoundedDaemonPool owns the signal.
        # Let callers know no worker remains that could retain their resources.
        released.set()
        raise
    try:
        value = result.result(timeout=float(timeout_seconds))
    except FutureTimeoutError:
        if result.done():
            released.wait()
            return result.result()
    except BaseException:  # noqa: BLE001 - wait for real worker exit before re-raising
        released.wait()
        raise
    else:
        released.wait()
        return value

    raise TimeoutError(timeout_message)


async def await_bounded_daemon_with_timeout(
    call: Callable[[], Any],
    *,
    pool: BoundedDaemonPool,
    name: str,
    timeout_seconds: float,
    timeout_message: str,
    released_event: threading.Event | None = None,
    retain_result_after_timeout: bool = False,
    drain_after_timeout: bool = False,
) -> Any:
    """Run one sync call directly from the loop without executor queueing.

    Admission and thread creation happen before the first suspension point, so
    a saturated default executor cannot delay a timed-out call until later.
    Owned cleanup may retain a late result when it must finish a returned
    awaitable before releasing its surrounding resources.  Callers that must
    preserve a resource lifetime but still honor the deadline can drain the
    admitted worker and then raise the configured timeout.
    """

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("Daemon call timeout must be a positive finite number")
    if retain_result_after_timeout and drain_after_timeout:
        raise ValueError(
            "Daemon calls cannot both retain and discard a late result"
        )

    result: Future[Any] = Future()

    def invoke() -> None:
        try:
            value = call()
        except BaseException as exc:  # noqa: BLE001 - every worker outcome resolves the future
            result.set_exception(exc)
        else:
            result.set_result(value)

    released = released_event or threading.Event()
    try:
        pool.start(invoke, name=name, released_event=released)
    except BaseException:
        released.set()
        raise

    loop_result = asyncio.wrap_future(result)

    def consume_late_result(done: asyncio.Future[Any]) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            done.exception()

    async def await_release() -> None:
        while not released.is_set():
            await asyncio.sleep(0.001)

    try:
        done, _pending = await asyncio.wait(
            {loop_result},
            timeout=float(timeout_seconds),
        )
        deadline_expired = loop_result not in done
        if deadline_expired:
            if not retain_result_after_timeout and not drain_after_timeout:
                loop_result.add_done_callback(consume_late_result)
                raise TimeoutError(timeout_message)
            await asyncio.wait({loop_result})

        await await_release()
        if deadline_expired and drain_after_timeout:
            consume_late_result(loop_result)
            raise TimeoutError(timeout_message)
        return loop_result.result()
    except asyncio.CancelledError:
        if drain_after_timeout:
            with contextlib.suppress(BaseException):
                await _drain_owned_task(loop_result)
            release_waiter = asyncio.create_task(await_release())
            with contextlib.suppress(BaseException):
                await _drain_owned_task(release_waiter)
            raise
        if loop_result.done():
            consume_late_result(loop_result)
        else:
            loop_result.add_done_callback(consume_late_result)
        raise


def daemon_capacity_from_env(name: str, *, default: int = 32) -> int:
    """Read a positive, bounded daemon capacity without trusting invalid input."""

    try:
        value = int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return default
    return value if 1 <= value <= 256 else default


STREAM_DAEMON_POOL = BoundedDaemonPool(
    daemon_capacity_from_env("CHAT_STREAM_DAEMON_MAX_WORKERS")
)
STREAM_CLEANUP_DAEMON_POOL = BoundedDaemonPool(
    daemon_capacity_from_env("CHAT_STREAM_CLEANUP_DAEMON_MAX_WORKERS", default=4)
)
SYNC_ADAPTER_CALL_POOL = BoundedDaemonPool(
    daemon_capacity_from_env("CHAT_SYNC_ADAPTER_MAX_WORKERS")
)


def start_bounded_stream_daemon(
    target: Callable[[], Any],
    *,
    name: str,
    released_event: threading.Event | None = None,
) -> threading.Thread:
    """Start one process-wide capacity-accounted streaming daemon."""

    return STREAM_DAEMON_POOL.start(
        target,
        name=name,
        released_event=released_event,
    )


def start_bounded_stream_cleanup_daemon(
    target: Callable[[], Any],
    *,
    name: str,
    released_event: threading.Event | None = None,
) -> threading.Thread:
    """Start cleanup using capacity reserved from regular streaming work."""

    return STREAM_CLEANUP_DAEMON_POOL.start(
        target,
        name=name,
        released_event=released_event,
    )


__all__ = [
    "await_bounded_daemon_with_timeout",
    "await_bounded_sync_call",
    "await_owned_worker",
    "BoundedDaemonPool",
    "DaemonCapacityError",
    "STREAM_CLEANUP_DAEMON_POOL",
    "STREAM_DAEMON_POOL",
    "SYNC_ADAPTER_CALL_POOL",
    "daemon_capacity_from_env",
    "run_bounded_daemon_with_timeout",
    "start_bounded_stream_cleanup_daemon",
    "start_bounded_stream_daemon",
]
