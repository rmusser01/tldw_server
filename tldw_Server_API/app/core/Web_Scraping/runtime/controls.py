"""Lower-level parent controls; no crawl engine or orchestration dependencies."""

from __future__ import annotations

import asyncio
import contextvars
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Generic, Literal, Protocol, TypeVar

Counter = Literal[
    "page_attempts", "network_requests", "browser_attempts", "active_probes", "transfer_bytes", "discovered_candidates"
]
COUNTERS: tuple[Counter, ...] = (
    "page_attempts",
    "network_requests",
    "browser_attempts",
    "active_probes",
    "transfer_bytes",
    "discovered_candidates",
)
T = TypeVar("T")


def positive_integer(value: int, name: str) -> None:
    """Reject non-integral and non-positive trusted safety allowances."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def positive_duration(value: float, name: str) -> None:
    """Reject booleans, non-finite numbers, and disabled time bounds."""
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and positive")


class ParentStop(Exception):
    """Non-retryable parent deadline or resource exhaustion (not cancellation)."""

    def __init__(self, reason: str) -> None:
        if reason not in {"deadline_exhausted", "budget_exhausted"}:
            raise ValueError("invalid parent stop reason")
        self.reason = reason
        super().__init__(reason)


class Reservation:
    """One cumulative reservation, charged at admission and refundable until start.

    The callbacks run under the owner's lock. Commit must check parent stop state
    before marking physical work started; close refunds unused admission once.
    This is not an active-capacity lease: committed work is never refunded.
    """

    def __init__(self, lock: Lock, check: Callable[[], None], rollback: Callable[[], None]) -> None:
        self._lock = lock
        self._check = check
        self._rollback = rollback
        self._state = "reserved"

    def commit(self) -> None:
        """Atomically mark work started unless the parent stopped first."""
        with self._lock:
            if self._state == "closed":
                raise RuntimeError("reservation already closed")
            if self._state == "reserved":
                self._check()
                self._state = "committed"

    def close(self) -> None:
        """Refund an unused reservation once; started work stays charged."""
        with self._lock:
            if self._state == "reserved":
                self._rollback()
                self._state = "closed"


class ParentControls(Protocol):
    """Controls carried downward into governed acquisition and preflight."""

    def check(self) -> None: ...
    def reserve(self, counter: Counter, amount: int = 1) -> Reservation: ...
    def charge_bytes(self, amount: int) -> None: ...
    def cap_timeout(self, requested: float | None) -> float: ...


@dataclass(frozen=True)
class WorkHandle(Generic[T]):
    """Owned logical completion and independent physical drain notification.

    The owner supplies a thread-safe cancel callback. Consumers must never cancel
    the shared futures to claim that the underlying physical work has drained.
    """

    result: asyncio.Future[T]
    drained: asyncio.Future[None]
    cancel: Callable[[], None]


class PhysicalWork:
    """Per-operation ownership, independent of logical coroutine cancellation.

    Registration happens before dispatch. Thread callbacks only schedule releases
    onto the owning loop; unrelated operations never contribute to this drain.
    """

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.drained: asyncio.Future[None] = self.loop.create_future()
        self._count = 1

    def acquire(self) -> Callable[[], None]:
        """Retain physical ownership until the returned idempotent release."""
        self._count += 1
        lock = Lock()
        released = False

        def release() -> None:
            nonlocal released
            with lock:
                if released:
                    return
                released = True
            self.loop.call_soon_threadsafe(self.finish)

        return release

    def finish(self) -> None:
        """Release one owner on the event loop."""
        self._count -= 1
        if self._count == 0 and not self.drained.done():
            self.drained.set_result(None)

    def observe(self, future: object) -> None:
        """Observe an actual concurrent future or retained cleanup task."""
        release = self.acquire()
        future.add_done_callback(lambda _done: release())


physical_work: contextvars.ContextVar[PhysicalWork | None] = contextvars.ContextVar(
    "scrape_physical_work", default=None
)


def observe_physical(future: object) -> None:
    """Attach existing lifecycle work to the current operation, when composed."""
    owner = physical_work.get()
    if owner is not None:
        owner.observe(future)


async def owned_to_thread(func: Callable[..., T], *args: object, **kwargs: object) -> T:
    """Run existing default-pool work without confusing awaiter cancel with exit."""
    owner = physical_work.get()
    if owner is None:
        return await asyncio.to_thread(func, *args, **kwargs)
    release = owner.acquire()
    context = contextvars.copy_context()
    dispatched = False

    def call() -> T:
        try:
            return func(*args, **kwargs)
        finally:
            release()

    async def execute() -> T:
        nonlocal dispatched
        # Use the same default pool as to_thread, but release from the physical
        # callable, not the cancelable asyncio proxy around its future.
        future = owner.loop.run_in_executor(None, context.run, call)
        dispatched = True
        future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return await asyncio.shield(future)

    def completed(done: asyncio.Task[T]) -> None:
        if not dispatched:
            release()
        if not done.cancelled():
            done.exception()

    operation = execute()
    try:
        task = asyncio.create_task(operation)
    except BaseException:
        operation.close()
        release()
        raise
    task.add_done_callback(completed)
    return await asyncio.shield(task)


class PageControls:
    """Page-local cancellation and first-stop latch around the shared parent."""

    def __init__(self, parent: ParentControls) -> None:
        self.parent = parent
        self._cancelled = threading.Event()
        self._stop: BaseException | None = None
        self._lock = Lock()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled.set()

    def _call(self, func: Callable[..., T], *args: object) -> T:
        with self._lock:
            return self._call_locked(func, *args)

    def _call_locked(self, func: Callable[..., T], *args: object) -> T:
        if self._stop is not None:
            raise self._stop
        if self._cancelled.is_set():
            raise asyncio.CancelledError
        try:
            return func(*args)
        except (ParentStop, asyncio.CancelledError) as exc:
            self._stop = exc
            raise

    def check(self) -> None:
        self._call(self.parent.check)

    def reserve(self, counter: Counter, amount: int = 1) -> Reservation:
        reservation = self._call(self.parent.reserve, counter, amount)
        # Page cancellation and parent commit share the page admission lock.
        return Reservation(self._lock, lambda: self._call_locked(reservation.commit), reservation.close)

    def charge_bytes(self, amount: int) -> None:
        if amount:
            self._call(self.parent.charge_bytes, amount)
        else:
            self.check()

    def cap_timeout(self, requested: float | None) -> float:
        return self._call(self.parent.cap_timeout, requested)


@dataclass(frozen=True)
class BudgetSnapshot:
    """Known admitted usage (including uncommitted reservations), not wire/RSS use."""

    page_attempts: int
    network_requests: int
    browser_attempts: int
    active_probes: int
    transfer_bytes: int
    discovered_candidates: int
    remaining_seconds: float
    accounting_complete: bool = True
