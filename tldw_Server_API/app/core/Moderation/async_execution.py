"""Bounded cancellable transport with a mandatory metadata reporting boundary."""

from __future__ import annotations

import asyncio
import contextvars
import os
import threading
import time
from collections.abc import Callable
from typing import TypeVar

from tldw_Server_API.app.core.exception_types import ModerationExecutionError
from tldw_Server_API.app.core.Moderation.execution_models import ExecutionScope, FailureMetadata, execution_scope
from tldw_Server_API.app.core.Moderation.execution_runner import _scope_is_owned

T = TypeVar("T")
_lock = threading.Lock()
_active = 0
_closed = False


async def run_moderation_async(call: Callable[[], T], *, metadata: FailureMetadata) -> T:
    """Establish the shared reporting boundary before size/offload admission."""
    from tldw_Server_API.app.core.Moderation.execution_incidents import report_execution_failure_async

    if type(metadata) is not FailureMetadata:
        raise ValueError("invalid moderation failure metadata")
    metadata = FailureMetadata(metadata.operation, metadata.phase, metadata.input_chars)
    with execution_scope() as scope:
        try:
            if metadata.input_chars > 1_000_000:
                raise ModerationExecutionError("size_exceeded")
            return await _run_moderation_offload(call, scope=scope)
        except ModerationExecutionError as error:
            await report_execution_failure_async(error, metadata=metadata, scope=scope)
            raise


def _after_fork() -> None:
    global _lock, _active, _closed
    _lock, _active, _closed = threading.Lock(), 0, False


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def close_moderation_offloads() -> None:
    """Stop new orchestration admission without forgetting running owners."""
    global _closed
    with _lock:
        _closed = True


async def _run_moderation_offload(call: Callable[[], T], *, scope: ExecutionScope) -> T:
    """Copy context into one admitted thread; cancellation signals its scope."""
    global _active
    if scope.cancelled.is_set():
        raise asyncio.CancelledError
    if time.monotonic() >= scope.deadline:
        raise ModerationExecutionError("deadline_exceeded")
    with _lock:
        if _closed:
            raise ModerationExecutionError("worker_unavailable")
        if _active >= 4:
            raise ModerationExecutionError("capacity_exhausted")
        _active += 1
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    context = contextvars.copy_context()

    def deliver(value: T | None, error: BaseException | None) -> None:
        if not future.done():
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(value)

    def run() -> None:
        global _active
        value, error = None, None
        try:
            if scope.cancelled.is_set():
                raise asyncio.CancelledError
            if time.monotonic() >= scope.deadline:
                raise ModerationExecutionError("deadline_exceeded")
            value = context.run(call)
        except BaseException as caught:  # noqa: BLE001 - transport preserves cancellation too.
            error = caught
        finally:
            # A failed execution can have returned with quarantined pipe/PID
            # ownership. Its orchestration slot cannot be recycled yet.
            while _scope_is_owned(scope):
                time.sleep(0.01)
            with _lock:
                _active -= 1
            try:
                loop.call_soon_threadsafe(deliver, value, error)
            except RuntimeError:
                # The event loop is gone, but resource ownership was completed.
                pass

    try:
        thread = threading.Thread(target=run, name="moderation-offload", daemon=False)
        thread.start()
    except (RuntimeError, OSError):
        with _lock:
            _active -= 1
        raise ModerationExecutionError("worker_unavailable") from None
    try:
        done, _ = await asyncio.wait({future}, timeout=max(0, scope.deadline - time.monotonic()))
        if done:
            return future.result()
        scope.cancelled.set()
        future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        raise ModerationExecutionError("deadline_exceeded")
    except asyncio.CancelledError:
        scope.cancelled.set()
        # Consume eventual exceptions even after the awaiting client disappears.
        future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        raise
