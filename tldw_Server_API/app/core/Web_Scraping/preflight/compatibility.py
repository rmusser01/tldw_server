"""Synchronous compatibility bridge for legacy preflight analyzer callers."""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import os
import threading
from collections.abc import Coroutine
from time import monotonic
from typing import Any, Generic, TypeVar

T = TypeVar("T")

_CLEANUP_TIMEOUT_S = 2.0
_CANCELLATION_GRACE_S = 1.0
_THREAD_JOIN_RESERVE_S = 0.25
_REAL_GETPID = os.getpid


class _Submission(Generic[T]):
    """Transfer one coroutine from caller ownership to the loop exactly once."""

    def __init__(self, coroutine: Coroutine[Any, Any, T]) -> None:
        self._lock = threading.Lock()
        self._coroutine: Coroutine[Any, Any, T] | None = coroutine
        self.started = threading.Event()
        self.completed = threading.Event()

    def close_if_unstarted(self) -> bool:
        with self._lock:
            if self.started.is_set() or self._coroutine is None:
                return False
            coroutine = self._coroutine
            self._coroutine = None
        coroutine.close()
        self.completed.set()
        return True

    async def run(self) -> T:
        with self._lock:
            coroutine = self._coroutine
            self._coroutine = None
            if coroutine is None:
                self.completed.set()
                raise asyncio.CancelledError
            self.started.set()
        try:
            return await coroutine
        finally:
            self.completed.set()


class _LoopShutdownState:
    """Share one retirement deadline with the loop-owning thread."""

    def __init__(self) -> None:
        self.deadline: float | None = None


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - monotonic())


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        pass


async def _cancel_pending_tasks(*, deadline: float) -> None:
    current = asyncio.current_task()
    pending = [task for task in asyncio.all_tasks() if task is not current and not task.done()]
    for task in pending:
        task.cancel()
    if not pending:
        return

    grace_deadline = min(deadline, monotonic() + _CANCELLATION_GRACE_S)
    done, still_pending = await asyncio.wait(pending, timeout=_remaining(grace_deadline))
    for task in done:
        _consume_task_result(task)

    for task in still_pending:
        task.cancel()
    if still_pending and _remaining(deadline) > 0:
        done, still_pending = await asyncio.wait(still_pending, timeout=_remaining(deadline))
        for task in done:
            _consume_task_result(task)

    for task in still_pending:
        task.add_done_callback(_consume_task_result)


def _close_loop(loop: asyncio.AbstractEventLoop, shutdown_state: _LoopShutdownState) -> None:
    """Close within the retirement deadline, even for cancellation-resistant tasks.

    Python cannot forcibly terminate a coroutine that suppresses cancellation
    forever. Such a coroutine may remain pending when its loop is closed, but it
    cannot make bridge shutdown or process exit wait without a bound.
    """
    deadline = shutdown_state.deadline
    if deadline is None:
        deadline = monotonic() + _CLEANUP_TIMEOUT_S
    pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
    for task in pending:
        task.cancel()
    if pending and _remaining(deadline) > 0:
        done, pending = loop.run_until_complete(asyncio.wait(pending, timeout=_remaining(deadline)))
        for task in done:
            _consume_task_result(task)
    for task in pending:
        task.add_done_callback(_consume_task_result)
    loop.close()


def _retire_loop(
    loop: asyncio.AbstractEventLoop | None,
    thread: threading.Thread | None,
    shutdown_state: _LoopShutdownState | None,
    *,
    deadline: float,
) -> None:
    if shutdown_state is not None:
        shutdown_state.deadline = deadline
    if thread is None:
        return
    if not thread.is_alive():
        if loop is not None and not loop.is_closed() and not loop.is_running():
            _close_loop(loop, shutdown_state or _LoopShutdownState())
        return
    if loop is None:
        if threading.current_thread() is not thread:
            thread.join(timeout=_remaining(deadline))
        return
    cleanup_deadline = max(monotonic(), deadline - _THREAD_JOIN_RESERVE_S)
    if threading.current_thread() is thread:
        cleanup = loop.create_task(_cancel_pending_tasks(deadline=cleanup_deadline))
        cleanup.add_done_callback(lambda _task: loop.stop())
        return

    cleanup_coroutine = _cancel_pending_tasks(deadline=cleanup_deadline)
    try:
        cleanup = asyncio.run_coroutine_threadsafe(cleanup_coroutine, loop)
    except RuntimeError:
        cleanup_coroutine.close()
    else:
        try:
            cleanup.result(timeout=_remaining(cleanup_deadline))
        except (concurrent.futures.CancelledError, concurrent.futures.TimeoutError):
            cleanup.cancel()
        except RuntimeError:
            cleanup.cancel()
    try:
        # This is valid before run_forever starts and prevents a transitioning
        # loop from becoming an unretirable live thread.
        loop.call_soon_threadsafe(loop.stop)
    except RuntimeError:
        pass
    thread.join(timeout=_remaining(deadline))


class _BackgroundLoopBridge:
    """Own one lazily started event loop thread for a single process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._real_pid = _REAL_GETPID()
        self._owner_pid: int | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown_state: _LoopShutdownState | None = None
        self._generation = 0
        self._shut_down = False

    def _replace_inherited_state(self, pid: int, real_pid: int) -> None:
        # Do not inspect, stop, or close inherited loop/thread objects here.
        # The inherited lock may be held by a vanished parent thread.
        self._lock = threading.Lock()
        self._real_pid = real_pid
        self._owner_pid = pid
        self._thread = None
        self._loop = None
        self._shutdown_state = None
        self._generation += 1
        self._shut_down = False

    def _reset_for_pid_change(self, pid: int, *, deadline: float | None = None) -> None:
        while True:
            real_pid = _REAL_GETPID()
            if self._real_pid != real_pid:
                self._replace_inherited_state(pid, real_pid)
                return

            lock = self._lock
            with lock:
                if self._real_pid != _REAL_GETPID():
                    continue
                owner_pid = self._owner_pid
                if owner_pid is None or owner_pid == pid:
                    return
                old_loop = self._loop
                old_thread = self._thread
                old_shutdown_state = self._shutdown_state
                self._owner_pid = pid
                self._thread = None
                self._loop = None
                self._shutdown_state = None
                self._generation += 1
                self._shut_down = False
                _retire_loop(
                    old_loop,
                    old_thread,
                    old_shutdown_state,
                    deadline=(deadline if deadline is not None else monotonic() + _CLEANUP_TIMEOUT_S),
                )
                return

    def _start_locked(self, pid: int) -> None:
        ready = threading.Event()
        abandoned = threading.Event()
        startup_lock = threading.Lock()
        startup_errors: list[BaseException] = []
        shutdown_state = _LoopShutdownState()
        generation = self._generation

        def run_loop() -> None:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            except Exception as exc:  # noqa: BLE001 - report startup failure to the caller
                with startup_lock:
                    startup_errors.append(exc)
                    ready.set()
                return

            with startup_lock:
                if abandoned.is_set() or self._generation != generation:
                    should_run = False
                else:
                    self._loop = loop
                    loop.call_soon(ready.set)
                    should_run = True
            if not should_run:
                loop.close()
                return
            try:
                loop.run_forever()
            finally:
                _close_loop(loop, shutdown_state)

        thread = threading.Thread(
            target=run_loop,
            name=f"preflight-compat-loop-{pid}",
            daemon=True,
        )
        self._thread = thread
        self._shutdown_state = shutdown_state
        thread.start()
        if not ready.wait(timeout=_CLEANUP_TIMEOUT_S):
            with startup_lock:
                if not ready.is_set():
                    abandoned.set()
                    self._shut_down = True
                    loop = self._loop
                    if loop is not None:
                        try:
                            loop.call_soon_threadsafe(loop.stop)
                        except RuntimeError:
                            pass
                    raise RuntimeError("Legacy analyzer bridge failed to start.")
        if startup_errors:
            self._thread = None
            raise startup_errors[0]
        if self._loop is None:
            self._thread = None
            raise RuntimeError("Legacy analyzer bridge failed to start.")

    def _ensure_started_for_pid(self, pid: int) -> None:
        while True:
            self._reset_for_pid_change(pid)
            lock = self._lock
            with lock:
                if self._owner_pid not in (None, pid):
                    continue
                if self._owner_pid is None:
                    self._owner_pid = pid
                if self._shut_down:
                    raise RuntimeError("Legacy analyzer bridge has been shut down.")
                if self._thread is None or not self._thread.is_alive():
                    self._start_locked(pid)
                return

    def submit(
        self,
        coroutine: Coroutine[Any, Any, T],
        *,
        timeout_s: float | None = None,
    ) -> T:
        """Run an owned coroutine on the bridge and wait synchronously."""
        submission = _Submission(coroutine)
        scheduled = False
        wrapper: Coroutine[Any, Any, T] | None = None
        try:
            pid = os.getpid()
            self._ensure_started_for_pid(pid)
            wrapper = submission.run()
            with self._lock:
                if self._shut_down or self._owner_pid != pid:
                    raise RuntimeError("Legacy analyzer bridge has been shut down.")
                loop = self._loop
                if loop is None or not loop.is_running():
                    raise RuntimeError("Legacy analyzer bridge is not running.")
                future = asyncio.run_coroutine_threadsafe(wrapper, loop)
                scheduled = True
        except BaseException:
            if not scheduled:
                if wrapper is not None:
                    wrapper.close()
                submission.close_if_unstarted()
            raise

        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError as exc:
            if future.done():
                return future.result()
            future.cancel()
            submission.close_if_unstarted()
            submission.completed.wait(timeout=_CLEANUP_TIMEOUT_S)
            raise TimeoutError("Legacy analyzer timed out.") from exc

    def shutdown(self) -> None:
        """Stop this process owner's loop once and reject further work."""
        deadline = monotonic() + _CLEANUP_TIMEOUT_S
        pid = os.getpid()
        while True:
            self._reset_for_pid_change(pid, deadline=deadline)
            lock = self._lock
            with lock:
                if self._owner_pid not in (None, pid):
                    continue
                if self._owner_pid is None:
                    self._owner_pid = pid
                if not self._shut_down:
                    self._shut_down = True
                loop = self._loop
                thread = self._thread
                shutdown_state = self._shutdown_state
                break
        _retire_loop(loop, thread, shutdown_state, deadline=deadline)


_PROCESS_BRIDGE = _BackgroundLoopBridge()


def _run_sync_compat(
    coroutine: Coroutine[Any, Any, T],
    *,
    timeout_s: float | None = None,
) -> T:
    """Run one legacy coroutine without depending on the caller's event loop."""
    return _PROCESS_BRIDGE.submit(coroutine, timeout_s=timeout_s)


def _shutdown_process_bridge() -> None:
    _PROCESS_BRIDGE.shutdown()


atexit.register(_shutdown_process_bridge)
