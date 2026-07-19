"""Synchronous compatibility bridge for legacy preflight analyzer callers."""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import os
import threading
from collections.abc import Coroutine
from typing import Any, Generic, TypeVar

T = TypeVar("T")

_CLEANUP_TIMEOUT_S = 2.0
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


async def _cancel_pending_tasks() -> None:
    current = asyncio.current_task()
    pending = [task for task in asyncio.all_tasks() if task is not current and not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def _close_loop(loop: asyncio.AbstractEventLoop) -> None:
    pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    loop.close()


def _retire_loop(
    loop: asyncio.AbstractEventLoop | None,
    thread: threading.Thread | None,
) -> None:
    if thread is None or not thread.is_alive():
        return
    if loop is None:
        if threading.current_thread() is not thread:
            thread.join(timeout=_CLEANUP_TIMEOUT_S)
        return
    if threading.current_thread() is thread:
        cleanup = loop.create_task(_cancel_pending_tasks())
        cleanup.add_done_callback(lambda _task: loop.stop())
        return

    if loop.is_running():
        cleanup_coroutine = _cancel_pending_tasks()
        try:
            cleanup = asyncio.run_coroutine_threadsafe(cleanup_coroutine, loop)
        except RuntimeError:
            cleanup_coroutine.close()
        else:
            try:
                cleanup.result(timeout=_CLEANUP_TIMEOUT_S)
            except concurrent.futures.CancelledError:
                pass
            except concurrent.futures.TimeoutError:
                cleanup.cancel()
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass
    thread.join(timeout=_CLEANUP_TIMEOUT_S)


class _BackgroundLoopBridge:
    """Own one lazily started event loop thread for a single process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner_pid: int | None = None
        self._owner_real_pid: int | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shut_down = False

    def _reset_for_pid_change(self, pid: int) -> None:
        owner_pid = self._owner_pid
        if owner_pid is None or owner_pid == pid:
            return

        old_loop = self._loop
        old_thread = self._thread
        if self._owner_real_pid == _REAL_GETPID():
            _retire_loop(old_loop, old_thread)

        # A fork may inherit a locked mutex. Replace all process-bound state
        # before any lock acquisition and never touch the parent's loop.
        self._lock = threading.Lock()
        self._owner_pid = pid
        self._owner_real_pid = _REAL_GETPID()
        self._thread = None
        self._loop = None
        self._shut_down = False

    def _start_locked(self, pid: int) -> None:
        ready = threading.Event()
        abandoned = threading.Event()
        startup_lock = threading.Lock()
        startup_errors: list[BaseException] = []

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
                if abandoned.is_set():
                    should_run = False
                else:
                    self._loop = loop
                    ready.set()
                    should_run = True
            if not should_run:
                loop.close()
                return
            try:
                loop.run_forever()
            finally:
                _close_loop(loop)

        thread = threading.Thread(
            target=run_loop,
            name=f"preflight-compat-loop-{pid}",
            daemon=True,
        )
        self._thread = thread
        thread.start()
        if not ready.wait(timeout=_CLEANUP_TIMEOUT_S):
            with startup_lock:
                if not ready.is_set():
                    abandoned.set()
                    self._shut_down = True
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
                    self._owner_real_pid = _REAL_GETPID()
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
        pid = os.getpid()
        while True:
            self._reset_for_pid_change(pid)
            lock = self._lock
            with lock:
                if self._owner_pid not in (None, pid):
                    continue
                if self._owner_pid is None:
                    self._owner_pid = pid
                    self._owner_real_pid = _REAL_GETPID()
                if not self._shut_down:
                    self._shut_down = True
                loop = self._loop
                thread = self._thread
                break
        _retire_loop(loop, thread)


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
