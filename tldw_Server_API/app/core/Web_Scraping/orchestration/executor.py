"""Bounded, cancellation-aware execution for synchronous article extraction."""

from __future__ import annotations

import asyncio
import atexit
import contextvars
import math
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, TypeVar, cast

from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import (
    cancellation_checkpoint_scope,
)

from .article_models import ArticleFailure

DEFAULT_EXTRACTOR_MAX_WORKERS = 4
MAX_EXTRACTOR_WORKERS = 64
DEFAULT_EXTRACTOR_ADMISSION_TIMEOUT_SECONDS = 30.0
_INITIAL_ADMISSION_DELAY_SECONDS = 0.01
_MAX_ADMISSION_DELAY_SECONDS = 0.1

_T = TypeVar("_T")


class ManagerState(str, Enum):
    """Lifecycle states for one process-local executor manager."""

    RUNNING = "running"
    RELOADING = "reloading"
    SHUTDOWN = "shutdown"


@dataclass(slots=True)
class ExecutorGeneration:
    """One immutable-capacity executor generation owned by a single process."""

    pid: int
    generation_id: int
    worker_count: int
    executor: Any
    permits: threading.BoundedSemaphore
    closed: bool = False
    outstanding: dict[Future[Any], _PermitLease] = field(
        default_factory=dict,
        repr=False,
    )

    @classmethod
    def create_for_tests(cls, worker_count: int) -> ExecutorGeneration:
        """Create a standalone generation for deterministic contract tests."""

        normalized = _normalize_worker_count(worker_count)
        return cls(
            pid=os.getpid(),
            generation_id=1,
            worker_count=normalized,
            executor=ThreadPoolExecutor(
                max_workers=normalized,
                thread_name_prefix="web-extraction-test",
            ),
            permits=threading.BoundedSemaphore(normalized),
        )


@dataclass(slots=True)
class _PermitLease:
    """Release one generation permit at most once."""

    permits: threading.BoundedSemaphore
    _released: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self.permits.release()


@dataclass(slots=True)
class _Transition:
    """Loop-neutral completion signal for a reload or shutdown drain."""

    kind: ManagerState
    complete: threading.Event = field(default_factory=threading.Event)
    error: ArticleFailure | None = None


def _normalize_worker_count(value: Any) -> int:
    """Accept positive, non-boolean worker counts up to the server ceiling."""

    if type(value) is int and 0 < value <= MAX_EXTRACTOR_WORKERS:
        return value
    if type(value) is str:
        normalized = value.strip()
        if normalized.isascii() and normalized.isdecimal() and len(normalized) <= len(str(MAX_EXTRACTOR_WORKERS)):
            try:
                parsed = int(normalized)
            except (ValueError, OverflowError):
                return DEFAULT_EXTRACTOR_MAX_WORKERS
            if 0 < parsed <= MAX_EXTRACTOR_WORKERS:
                return parsed
    return DEFAULT_EXTRACTOR_MAX_WORKERS


def _load_worker_count() -> Any:
    return os.environ.get("EXTRACTOR_MAX_WORKERS")


def _normalize_admission_timeout(value: Any) -> float:
    if isinstance(value, bool):
        return DEFAULT_EXTRACTOR_ADMISSION_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_EXTRACTOR_ADMISSION_TIMEOUT_SECONDS
    if not math.isfinite(timeout) or timeout <= 0:
        return DEFAULT_EXTRACTOR_ADMISSION_TIMEOUT_SECONDS
    return timeout


def _load_admission_timeout() -> Any:
    return os.environ.get("EXTRACTOR_ADMISSION_TIMEOUT_SECONDS")


def _default_lifecycle_observer(outcome: str) -> None:
    from tldw_Server_API.app.core.Web_Scraping.extraction.metrics import emit_global_counter

    emit_global_counter("extraction_executor_total", labels={"outcome": outcome})


def _default_executor_factory(worker_count: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="web-extraction",
    )


def _raise_if_cancelled(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise asyncio.CancelledError


class ExtractionExecutorManager:
    """Own bounded extraction capacity across event loops in one process."""

    def __init__(
        self,
        *,
        worker_count_loader: Callable[[], Any] = _load_worker_count,
        admission_timeout_loader: Callable[[], Any] = _load_admission_timeout,
        executor_factory: Callable[[int], Any] = _default_executor_factory,
        pid_getter: Callable[[], int] = os.getpid,
        lifecycle_observer: Callable[[str], None] | None = _default_lifecycle_observer,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._worker_count_loader = worker_count_loader
        self._admission_timeout_loader = admission_timeout_loader
        self._executor_factory = executor_factory
        self._pid_getter = pid_getter
        self._lifecycle_observer = lifecycle_observer
        self._clock = clock
        self._lock = threading.RLock()
        self._pid = pid_getter()
        self._state = ManagerState.RUNNING
        self._generation: ExecutorGeneration | None = None
        self._generation_id = 0
        self._transition: _Transition | None = None
        self._failed_generations: list[ExecutorGeneration] = []

    @property
    def state(self) -> ManagerState:
        with self._lock:
            self._reconcile_pid_locked()
            return self._state

    @property
    def current_generation(self) -> ExecutorGeneration | None:
        with self._lock:
            self._reconcile_pid_locked()
            return self._generation

    async def run(
        self,
        func: Callable[..., _T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        """Run one synchronous extraction without using the executor queue as capacity."""

        cancelled = threading.Event()
        context = contextvars.copy_context()
        delay = _INITIAL_ADMISSION_DELAY_SECONDS
        future: Future[_T] | None = None
        emitted: set[str] = set()
        try:
            configured_timeout = self._admission_timeout_loader()
        except Exception:  # noqa: BLE001 - config failures use the safe default
            configured_timeout = None
        admission_deadline = self._clock() + _normalize_admission_timeout(configured_timeout)

        def observe_once(outcome: str) -> None:
            if outcome in emitted:
                return
            emitted.add(outcome)
            observer = self._lifecycle_observer
            if observer is None:
                return
            try:
                observer(outcome)
            except Exception:  # noqa: BLE001 - observability must not replace extraction
                return

        try:
            while True:
                transition: _Transition | None = None
                with self._lock:
                    self._reconcile_pid_locked()
                    if self._state is ManagerState.SHUTDOWN:
                        raise ArticleFailure("extraction_error", "shutdown")
                    if self._state is ManagerState.RELOADING:
                        transition = self._transition
                        generation = None
                    else:
                        generation = self._ensure_generation_locked()
                        self._reap_completed_locked(generation)

                if transition is not None:
                    observe_once("queued")
                    try:
                        await self._wait_for_transition(
                            transition,
                            deadline=admission_deadline,
                            clock=self._clock,
                        )
                    except ArticleFailure:
                        observe_once("saturated")
                        raise
                    delay = _INITIAL_ADMISSION_DELAY_SECONDS
                    continue
                if generation is None:
                    observe_once("queued")
                    observe_once("saturated")
                    await self._sleep_for_admission(admission_deadline, delay)
                    delay = min(delay * 2, _MAX_ADMISSION_DELAY_SECONDS)
                    continue
                if not generation.permits.acquire(blocking=False):
                    observe_once("queued")
                    observe_once("saturated")
                    await self._sleep_for_admission(admission_deadline, delay)
                    delay = min(delay * 2, _MAX_ADMISSION_DELAY_SECONDS)
                    continue

                lease = _PermitLease(generation.permits)
                with self._lock:
                    self._reconcile_pid_locked()
                    stale = (
                        self._state is not ManagerState.RUNNING
                        or self._generation is not generation
                        or generation.closed
                        or generation.pid != self._pid
                    )
                    if not stale:
                        worker_call = partial(
                            self._worker_call,
                            context,
                            cancelled,
                            func,
                            args,
                            kwargs,
                        )
                        try:
                            future = cast(
                                Future[_T],
                                generation.executor.submit(worker_call),
                            )
                        except Exception:  # noqa: BLE001 - sanitize executor boundary
                            lease.release()
                            raise ArticleFailure("extraction_error", "submit") from None
                        generation.outstanding[cast(Future[Any], future)] = lease
                        future.add_done_callback(partial(self._release_completed, generation, lease))

                if future is None:
                    lease.release()
                    delay = _INITIAL_ADMISSION_DELAY_SECONDS
                    await self._sleep_for_admission(admission_deadline, 0)
                    continue

                observe_once("running")
                return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            observe_once("cancelled")
            if future is not None:
                cancelled.set()
                if not future.cancel():
                    observe_once("discarded")
            raise

    async def reload(self) -> None:
        """Drain the current generation and install one fresh config snapshot."""

        owns_transition = False
        generation: ExecutorGeneration | None = None
        with self._lock:
            self._reconcile_pid_locked()
            if self._state is ManagerState.SHUTDOWN:
                raise ArticleFailure("extraction_error", "shutdown")
            if self._state is ManagerState.RELOADING:
                transition = self._transition
            else:
                self._state = ManagerState.RELOADING
                generation = self._detach_generation_locked()
                transition = _Transition(ManagerState.RELOADING)
                self._transition = transition
                owns_transition = True

        if transition is None:
            raise ArticleFailure("extraction_error", "reload")
        if owns_transition:
            try:
                self._start_transition_thread(
                    target=self._complete_reload,
                    args=(transition, generation),
                    name="web-extraction-reload",
                )
            except Exception:  # noqa: BLE001 - finish failed transition safely
                self._complete_reload(
                    transition,
                    generation,
                    startup_failed=True,
                )

        await self._wait_for_transition(transition)

    async def shutdown(self) -> None:
        """Drain the current generation and permanently close admission."""

        owns_transition = False
        generation: ExecutorGeneration | None = None
        with self._lock:
            self._reconcile_pid_locked()
            if self._state is ManagerState.SHUTDOWN:
                transition = self._transition
            elif self._state is ManagerState.RELOADING:
                self._state = ManagerState.SHUTDOWN
                transition = self._transition
            else:
                self._state = ManagerState.SHUTDOWN
                generation = self._detach_generation_locked()
                transition = _Transition(ManagerState.SHUTDOWN)
                self._transition = transition
                owns_transition = True

        if transition is None:
            return
        if owns_transition:
            try:
                self._start_transition_thread(
                    target=self._complete_shutdown,
                    args=(transition, generation),
                    name="web-extraction-shutdown",
                )
            except Exception:  # noqa: BLE001 - finish failed transition safely
                self._complete_shutdown(
                    transition,
                    generation,
                    startup_failed=True,
                )

        await self._wait_for_transition(transition)

    def reset_for_tests(self) -> None:
        """Synchronously discard all owned state and restore initial admission."""

        with self._lock:
            transition = self._transition
        if transition is not None:
            transition.complete.wait()

        with self._lock:
            generation = self._detach_generation_locked()
            failed_generations = tuple(self._failed_generations)
            self._failed_generations.clear()
            self._transition = None
            self._state = ManagerState.SHUTDOWN
        self._shutdown_generation(generation)
        for failed_generation in failed_generations:
            self._shutdown_generation(failed_generation)
        with self._lock:
            self._pid = self._pid_getter()
            self._generation_id = 0
            self._state = ManagerState.RUNNING

    def close_at_exit(self) -> bool:
        """Drain process-owned workers and report whether cleanup succeeded."""

        try:
            with self._lock:
                transition = self._transition
                self._state = ManagerState.SHUTDOWN
                generation = self._detach_generation_locked()
            if transition is not None:
                transition.complete.wait()
            with self._lock:
                late_generation = self._detach_generation_locked()
                failed_generations = tuple(self._failed_generations)
                self._failed_generations.clear()

            generations: list[ExecutorGeneration] = []
            for candidate in (generation, late_generation, *failed_generations):
                if candidate is not None and all(candidate is not owned for owned in generations):
                    generations.append(candidate)

            failed_again: list[ExecutorGeneration] = []
            for owned_generation in generations:
                drained = self._shutdown_generation(owned_generation)
                if not drained:
                    drained = self._shutdown_generation(owned_generation)
                if not drained:
                    failed_again.append(owned_generation)
            if failed_again:
                with self._lock:
                    self._failed_generations.extend(failed_again)
                return False
            return True
        except Exception:  # noqa: BLE001 - interpreter cleanup is best effort
            return False

    def _worker_call(
        self,
        context: contextvars.Context,
        cancelled: threading.Event,
        func: Callable[..., _T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> _T:
        checkpoint = partial(_raise_if_cancelled, cancelled)

        def invoke() -> _T:
            with cancellation_checkpoint_scope(checkpoint):
                return func(*args, **kwargs)

        return context.run(invoke)

    def _ensure_generation_locked(self) -> ExecutorGeneration:
        generation = self._generation
        if generation is not None and not generation.closed:
            return generation

        try:
            configured_workers = self._worker_count_loader()
        except Exception:  # noqa: BLE001 - config failures use the safe default
            configured_workers = None
        worker_count = _normalize_worker_count(configured_workers)
        try:
            executor = self._executor_factory(worker_count)
        except Exception:  # noqa: BLE001 - sanitize injected factory failures
            raise ArticleFailure("extraction_error", "startup") from None
        self._generation_id += 1
        generation = ExecutorGeneration(
            pid=self._pid,
            generation_id=self._generation_id,
            worker_count=worker_count,
            executor=executor,
            permits=threading.BoundedSemaphore(worker_count),
        )
        self._generation = generation
        return generation

    def _detach_generation_locked(self) -> ExecutorGeneration | None:
        generation = self._generation
        self._generation = None
        if generation is not None:
            generation.closed = True
        return generation

    def _release_completed(
        self,
        generation: ExecutorGeneration,
        lease: _PermitLease,
        future: Future[Any],
    ) -> None:
        with self._lock:
            generation.outstanding.pop(future, None)
        lease.release()

    def _reap_completed_locked(self, generation: ExecutorGeneration) -> None:
        completed = [(future, lease) for future, lease in generation.outstanding.items() if future.done()]
        for future, lease in completed:
            generation.outstanding.pop(future, None)
            lease.release()

    def _reconcile_pid_locked(self) -> None:
        current_pid = self._pid_getter()
        if current_pid == self._pid:
            return
        generation = self._generation
        if generation is not None:
            generation.closed = True
        self._pid = current_pid
        self._state = ManagerState.RUNNING
        self._generation = None
        self._generation_id = 0
        self._transition = None
        self._failed_generations.clear()

    def _after_fork_child(self) -> None:
        """Replace inherited synchronization without touching parent-owned threads."""

        generation = self._generation
        if generation is not None:
            generation.closed = True
        self._lock = threading.RLock()
        self._pid = self._pid_getter()
        self._state = ManagerState.RUNNING
        self._generation = None
        self._generation_id = 0
        self._transition = None
        self._failed_generations.clear()

    def _retain_failed_generation_locked(
        self,
        generation: ExecutorGeneration | None,
    ) -> None:
        if generation is not None and all(generation is not owned for owned in self._failed_generations):
            self._failed_generations.append(generation)

    def _start_transition_thread(
        self,
        *,
        target: Callable[..., None],
        args: tuple[Any, ...],
        name: str,
    ) -> None:
        threading.Thread(target=target, args=args, name=name, daemon=True).start()

    def _complete_reload(
        self,
        transition: _Transition,
        generation: ExecutorGeneration | None,
        *,
        startup_failed: bool = False,
    ) -> None:
        error: ArticleFailure | None = None
        drained = False
        try:
            drained = self._shutdown_generation(generation)
            with self._lock:
                owns_transition = self._transition is transition
                if owns_transition and startup_failed:
                    stage = "shutdown" if self._state is ManagerState.SHUTDOWN else "reload"
                    error = ArticleFailure("extraction_error", stage)
                if owns_transition and not drained:
                    stage = "shutdown" if self._state is ManagerState.SHUTDOWN else "reload"
                    error = ArticleFailure("extraction_error", stage)
                    self._retain_failed_generation_locked(generation)
                    self._state = ManagerState.SHUTDOWN
                elif owns_transition and self._state is ManagerState.RELOADING:
                    try:
                        self._ensure_generation_locked()
                    except ArticleFailure:
                        error = ArticleFailure("extraction_error", "reload")
                    self._state = ManagerState.RUNNING
        except Exception:  # noqa: BLE001 - transitions must always finish
            with self._lock:
                if self._transition is transition:
                    stage = "shutdown" if self._state is ManagerState.SHUTDOWN else "reload"
                    error = ArticleFailure("extraction_error", stage)
                    self._state = ManagerState.SHUTDOWN
        finally:
            with self._lock:
                if not drained:
                    self._retain_failed_generation_locked(generation)
                if self._transition is transition:
                    self._transition = None
                transition.error = error
                transition.complete.set()

    def _complete_shutdown(
        self,
        transition: _Transition,
        generation: ExecutorGeneration | None,
        *,
        startup_failed: bool = False,
    ) -> None:
        error: ArticleFailure | None = None
        drained = False
        try:
            drained = self._shutdown_generation(generation)
            if startup_failed or not drained:
                error = ArticleFailure("extraction_error", "shutdown")
        except Exception:  # noqa: BLE001 - transitions must always finish
            error = ArticleFailure("extraction_error", "shutdown")
        finally:
            with self._lock:
                if not drained:
                    self._retain_failed_generation_locked(generation)
                if self._transition is transition:
                    self._transition = None
                transition.error = error
                transition.complete.set()

    @staticmethod
    def _shutdown_generation(generation: ExecutorGeneration | None) -> bool:
        if generation is None:
            return True
        try:
            generation.executor.shutdown(wait=True, cancel_futures=False)
        except Exception:  # noqa: BLE001 - report lifecycle failure safely
            return False
        return True

    async def _sleep_for_admission(self, deadline: float, delay: float) -> None:
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise ArticleFailure("extraction_error", "capacity")
        await asyncio.sleep(min(delay, remaining))

    @staticmethod
    async def _wait_for_transition(
        transition: _Transition,
        *,
        deadline: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        while not transition.complete.is_set():
            delay = _INITIAL_ADMISSION_DELAY_SECONDS
            if deadline is not None:
                remaining = deadline - clock()
                if remaining <= 0:
                    raise ArticleFailure("extraction_error", "capacity")
                delay = min(delay, remaining)
            await asyncio.sleep(delay)
        if transition.error is not None:
            raise transition.error


DEFAULT_EXTRACTION_EXECUTOR = ExtractionExecutorManager()


async def run_extraction(
    func: Callable[..., _T],
    /,
    *args: Any,
    **kwargs: Any,
) -> _T:
    """Run one extraction through the process-default manager."""

    return await DEFAULT_EXTRACTION_EXECUTOR.run(func, *args, **kwargs)


async def reload_extraction_executor() -> None:
    """Reload the process-default extraction executor generation."""

    await DEFAULT_EXTRACTION_EXECUTOR.reload()


async def shutdown_extraction_executor() -> None:
    """Shut down the process-default extraction executor permanently."""

    await DEFAULT_EXTRACTION_EXECUTOR.shutdown()


def reset_extraction_executor_for_tests() -> None:
    """Reset the process-default manager for isolated tests."""

    DEFAULT_EXTRACTION_EXECUTOR.reset_for_tests()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=DEFAULT_EXTRACTION_EXECUTOR._after_fork_child)
atexit.register(DEFAULT_EXTRACTION_EXECUTOR.close_at_exit)


__all__ = [
    "DEFAULT_EXTRACTION_EXECUTOR",
    "DEFAULT_EXTRACTOR_ADMISSION_TIMEOUT_SECONDS",
    "DEFAULT_EXTRACTOR_MAX_WORKERS",
    "ExecutorGeneration",
    "ExtractionExecutorManager",
    "ManagerState",
    "reload_extraction_executor",
    "reset_extraction_executor_for_tests",
    "run_extraction",
    "shutdown_extraction_executor",
]
