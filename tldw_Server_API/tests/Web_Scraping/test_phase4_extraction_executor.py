"""Deterministic contracts for bounded extraction executor generations."""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import (
    build_default_dependencies,
)
from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import (
    ArticleFailure,
)
from tldw_Server_API.app.core.Web_Scraping.orchestration.executor import (
    ExecutorGeneration,
    ExtractionExecutorManager,
    ManagerState,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _manager(*, workers: Any = 1, pid: list[int] | None = None) -> ExtractionExecutorManager:
    current_pid = pid or [os.getpid()]
    return ExtractionExecutorManager(
        worker_count_loader=lambda: workers,
        pid_getter=lambda: current_pid[0],
    )


@pytest.fixture
def managers() -> list[ExtractionExecutorManager]:
    owned: list[ExtractionExecutorManager] = []
    yield owned
    for manager in owned:
        manager.reset_for_tests()


def _own(
    managers: list[ExtractionExecutorManager],
    manager: ExtractionExecutorManager,
) -> ExtractionExecutorManager:
    managers.append(manager)
    return manager


def test_public_contract_is_importable_without_starting_threads() -> None:
    script = """
import json
import threading

threads_before = {thread.ident for thread in threading.enumerate()}
from tldw_Server_API.app.core.Web_Scraping.orchestration.executor import (
    DEFAULT_EXTRACTION_EXECUTOR,
    ExtractionExecutorManager,
    reload_extraction_executor,
    reset_extraction_executor_for_tests,
    shutdown_extraction_executor,
)
threads_after = {thread.ident for thread in threading.enumerate()}
print(
    "EXECUTOR_IMPORT="
    + json.dumps(
        {
            "is_manager": isinstance(DEFAULT_EXTRACTION_EXECUTOR, ExtractionExecutorManager),
            "generation_is_none": DEFAULT_EXTRACTION_EXECUTOR.current_generation is None,
            "reload_is_callable": callable(reload_extraction_executor),
            "reset_is_callable": callable(reset_extraction_executor_for_tests),
            "shutdown_is_callable": callable(shutdown_extraction_executor),
            "new_threads": sorted(threads_after - threads_before),
        },
        sort_keys=True,
    )
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    marker = next(line for line in completed.stdout.splitlines() if line.startswith("EXECUTOR_IMPORT="))

    assert json.loads(marker.removeprefix("EXECUTOR_IMPORT=")) == {
        "generation_is_none": True,
        "is_manager": True,
        "new_threads": [],
        "reload_is_callable": True,
        "reset_is_callable": True,
        "shutdown_is_callable": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured",
    [
        None,
        "",
        True,
        False,
        0,
        -1,
        "bad",
        pytest.param("9" * 5_000, id="oversized-numeric"),
    ],
)
async def test_invalid_worker_settings_use_four(
    managers: list[ExtractionExecutorManager],
    configured: Any,
) -> None:
    manager = _own(managers, _manager(workers=configured))

    assert await manager.run(lambda: "ok") == "ok"

    generation = manager.current_generation
    assert generation is not None
    assert generation.worker_count == 4


@pytest.mark.asyncio
async def test_worker_setting_read_failure_uses_four(
    managers: list[ExtractionExecutorManager],
) -> None:
    def fail_read() -> int:
        raise RuntimeError("raw config detail")

    manager = _own(
        managers,
        ExtractionExecutorManager(worker_count_loader=fail_read),
    )

    assert await manager.run(lambda: "ok") == "ok"

    generation = manager.current_generation
    assert generation is not None
    assert generation.worker_count == 4


@pytest.mark.asyncio
async def test_worker_setting_is_snapshotted_until_reload(
    managers: list[ExtractionExecutorManager],
) -> None:
    configured = [2]
    manager = _own(
        managers,
        ExtractionExecutorManager(worker_count_loader=lambda: configured[0]),
    )

    await manager.run(lambda: None)
    first = manager.current_generation
    assert first is not None and first.worker_count == 2
    configured[0] = 3
    await manager.run(lambda: None)
    assert manager.current_generation is first

    await manager.reload()

    second = manager.current_generation
    assert second is not None and second is not first
    assert second.worker_count == 3
    assert first.closed is True


@pytest.mark.asyncio
async def test_positive_environment_worker_setting_is_captured(
    managers: list[ExtractionExecutorManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTRACTOR_MAX_WORKERS", "3")
    manager = _own(managers, ExtractionExecutorManager())

    await manager.run(lambda: None)

    generation = manager.current_generation
    assert generation is not None
    assert generation.worker_count == 3


@pytest.mark.asyncio
async def test_run_propagates_context_and_cooperative_checkpoint(
    managers: list[ExtractionExecutorManager],
) -> None:
    marker: contextvars.ContextVar[str] = contextvars.ContextVar("executor-marker")
    token = marker.set("request-context")
    manager = _own(managers, _manager())

    try:
        result = await manager.run(lambda: (marker.get(), build_default_dependencies().cancellation_checkpoint()))
    finally:
        marker.reset(token)

    assert result == ("request-context", None)


@pytest.mark.asyncio
async def test_saturation_waits_outside_executor_queue_and_is_cancellable(
    managers: list[ExtractionExecutorManager],
) -> None:
    manager = _own(managers, _manager())
    started = threading.Event()
    release = threading.Event()
    first = asyncio.create_task(manager.run(lambda: (started.set(), release.wait())))
    queued: asyncio.Task[str] | None = None
    try:
        assert await asyncio.to_thread(started.wait, 1.0) is True
        queued = asyncio.create_task(manager.run(lambda: "must-not-submit"))
        await asyncio.sleep(0.03)
        generation = manager.current_generation
        assert generation is not None
        assert generation.executor._work_queue.qsize() == 0
        assert queued.done() is False

        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
    finally:
        release.set()
        if queued is not None and not queued.done():
            queued.cancel()
        pending = [first] if queued is None else [first, queued]
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_saturated_admission_expires_with_bounded_lifecycle_events(
    managers: list[ExtractionExecutorManager],
) -> None:
    events: list[str] = []
    manager = _own(
        managers,
        ExtractionExecutorManager(
            worker_count_loader=lambda: 1,
            admission_timeout_loader=lambda: 0.03,
            lifecycle_observer=events.append,
        ),
    )
    started = threading.Event()
    release = threading.Event()
    first = asyncio.create_task(manager.run(lambda: (started.set(), release.wait())))
    try:
        assert await asyncio.to_thread(started.wait, 1.0) is True

        with pytest.raises(ArticleFailure) as raised:
            await asyncio.wait_for(manager.run(lambda: "must-not-submit"), timeout=0.5)

        assert raised.value.code == "extraction_error"
        assert raised.value.stage == "capacity"
        assert events == ["running", "queued", "saturated"]
        generation = manager.current_generation
        assert generation is not None
        assert generation.executor._work_queue.qsize() == 0
    finally:
        release.set()
        await asyncio.gather(first, return_exceptions=True)


@pytest.mark.asyncio
async def test_running_cancellation_emits_cancelled_and_discarded_once(
    managers: list[ExtractionExecutorManager],
) -> None:
    events: list[str] = []
    manager = _own(
        managers,
        ExtractionExecutorManager(
            worker_count_loader=lambda: 1,
            lifecycle_observer=events.append,
        ),
    )
    started = threading.Event()
    release = threading.Event()
    running = asyncio.create_task(manager.run(lambda: (started.set(), release.wait())))
    try:
        assert await asyncio.to_thread(started.wait, 1.0) is True
        running.cancel()

        with pytest.raises(asyncio.CancelledError):
            await running

        assert events == ["running", "cancelled", "discarded"]
    finally:
        release.set()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
async def test_running_cancellation_returns_immediately_and_holds_slot_until_exit(
    managers: list[ExtractionExecutorManager],
) -> None:
    manager = _own(managers, _manager())
    started = threading.Event()
    release = threading.Event()
    checkpoint_seen = threading.Event()

    def resistant() -> None:
        dependencies = build_default_dependencies()
        started.set()
        release.wait()
        try:
            dependencies.cancellation_checkpoint()
        except asyncio.CancelledError:
            checkpoint_seen.set()
            raise

    running = asyncio.create_task(manager.run(resistant))
    later: asyncio.Task[str] | None = None
    try:
        assert await asyncio.to_thread(started.wait, 1.0) is True
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, timeout=0.2)

        later = asyncio.create_task(manager.run(lambda: "later"))
        await asyncio.sleep(0.03)
        assert later.done() is False
        release.set()
        assert await asyncio.wait_for(later, timeout=1.0) == "later"
        assert checkpoint_seen.wait(1.0)
    finally:
        release.set()
        pending = [running] if later is None else [running, later]
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_caller_discards_successful_late_result(
    managers: list[ExtractionExecutorManager],
) -> None:
    manager = _own(managers, _manager())
    started = threading.Event()
    release = threading.Event()
    worker_results: list[str] = []

    def ignore_cancellation() -> str:
        started.set()
        release.wait()
        worker_results.append("late")
        return "late"

    running = asyncio.create_task(manager.run(ignore_cancellation))
    later: asyncio.Task[str] | None = None
    try:
        assert await asyncio.to_thread(started.wait, 1.0) is True
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

        later = asyncio.create_task(manager.run(lambda: "next"))
        await asyncio.sleep(0.03)
        assert later.done() is False
        release.set()
        assert await asyncio.wait_for(later, timeout=1.0) == "next"
        assert worker_results == ["late"]
        assert running.cancelled() is True
    finally:
        release.set()
        pending = [running] if later is None else [running, later]
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_worker_exception_propagates_to_orchestration_boundary(
    managers: list[ExtractionExecutorManager],
) -> None:
    manager = _own(managers, _manager())

    def fail() -> None:
        raise LookupError("worker detail")

    with pytest.raises(LookupError, match="worker detail"):
        await manager.run(fail)


class _SubmitFailExecutor:
    def submit(self, *_args: Any, **_kwargs: Any) -> Future[Any]:
        raise RuntimeError("raw submit detail")

    def shutdown(self, **_kwargs: Any) -> None:
        return None


class _DoubleCallbackFuture(Future[Any]):
    def __init__(self) -> None:
        super().__init__()
        self._callback_registrations = 0

    def add_done_callback(self, fn: Any) -> None:
        double_callback = self._callback_registrations == 0
        self._callback_registrations += 1
        super().add_done_callback(fn)
        if double_callback:
            fn(self)


class _DoubleCallbackExecutor:
    def submit(self, func: Any) -> Future[Any]:
        future = _DoubleCallbackFuture()
        future.set_result(func())
        return future

    def shutdown(self, **_kwargs: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_submit_failure_is_sanitized_and_permit_recovers(
    managers: list[ExtractionExecutorManager],
) -> None:
    manager = _own(
        managers,
        ExtractionExecutorManager(
            worker_count_loader=lambda: 1,
            executor_factory=lambda _workers: _SubmitFailExecutor(),
        ),
    )

    with pytest.raises(ArticleFailure) as raised:
        await manager.run(lambda: None)

    assert raised.value.code == "extraction_error"
    assert raised.value.stage == "submit"
    generation = manager.current_generation
    assert generation is not None
    assert generation.permits.acquire(blocking=False) is True
    generation.permits.release()


@pytest.mark.asyncio
async def test_duplicate_completion_callback_releases_permit_once(
    managers: list[ExtractionExecutorManager],
) -> None:
    manager = _own(
        managers,
        ExtractionExecutorManager(
            worker_count_loader=lambda: 1,
            executor_factory=lambda _workers: _DoubleCallbackExecutor(),
        ),
    )

    assert await manager.run(lambda: "done") == "done"

    generation = manager.current_generation
    assert generation is not None
    assert generation.permits.acquire(blocking=False) is True
    assert generation.permits.acquire(blocking=False) is False
    generation.permits.release()


class _GatedPermit:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.acquired = threading.Event()
        self.proceed = threading.Event()

    def acquire(self, *, blocking: bool = True) -> bool:
        acquired = self._delegate.acquire(blocking=blocking)
        if acquired:
            self.acquired.set()
            self.proceed.wait(1.0)
        return acquired

    def release(self) -> None:
        self._delegate.release()


@pytest.mark.asyncio
async def test_stale_admission_retries_on_replacement_generation(
    managers: list[ExtractionExecutorManager],
) -> None:
    manager = _own(managers, _manager())
    await manager.run(lambda: None)
    old_generation = manager.current_generation
    assert old_generation is not None
    gated_permit = _GatedPermit(old_generation.permits)
    old_generation.permits = gated_permit  # type: ignore[assignment]
    results: list[str] = []
    errors: list[BaseException] = []

    def runner() -> None:
        try:
            results.append(asyncio.run(manager.run(lambda: "replacement")))
        except BaseException as exc:  # noqa: BLE001  # pragma: no cover
            errors.append(exc)

    thread = threading.Thread(target=runner)
    thread.start()
    try:
        assert await asyncio.to_thread(gated_permit.acquired.wait, 1.0) is True
        await manager.reload()
        replacement = manager.current_generation
        gated_permit.proceed.set()
        await asyncio.to_thread(thread.join, 2.0)

        assert errors == []
        assert results == ["replacement"]
        assert replacement is not None and replacement is not old_generation
        assert thread.is_alive() is False
    finally:
        gated_permit.proceed.set()
        await asyncio.to_thread(thread.join, 2.0)


@pytest.mark.asyncio
async def test_same_manager_runs_from_two_event_loops(
    managers: list[ExtractionExecutorManager],
) -> None:
    manager = _own(managers, _manager(workers=2))
    barrier = threading.Barrier(2)
    results: list[int] = []
    errors: list[BaseException] = []

    def runner(value: int) -> None:
        try:
            result = asyncio.run(manager.run(lambda: (barrier.wait(timeout=1.0), value)[1]))
            results.append(result)
        except BaseException as exc:  # noqa: BLE001  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=runner, args=(value,)) for value in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert errors == []
    assert sorted(results) == [1, 2]
    assert all(not thread.is_alive() for thread in threads)


@pytest.mark.asyncio
async def test_concurrent_reload_installs_one_new_generation(
    managers: list[ExtractionExecutorManager],
) -> None:
    configured = [1]
    manager = _own(
        managers,
        ExtractionExecutorManager(worker_count_loader=lambda: configured[0]),
    )
    await manager.run(lambda: None)
    first = manager.current_generation
    assert first is not None
    configured[0] = 2

    await asyncio.gather(manager.reload(), manager.reload(), manager.reload())

    second = manager.current_generation
    assert second is not None
    assert second.generation_id == first.generation_id + 1
    assert second.worker_count == 2


@pytest.mark.asyncio
async def test_reload_closes_admission_but_waiters_resume_on_replacement(
    managers: list[ExtractionExecutorManager],
) -> None:
    manager = _own(managers, _manager())
    started = threading.Event()
    release = threading.Event()
    running = asyncio.create_task(manager.run(lambda: (started.set(), release.wait())))
    reload_task: asyncio.Task[None] | None = None
    waiter: asyncio.Task[str] | None = None
    try:
        assert await asyncio.to_thread(started.wait, 1.0) is True
        reload_task = asyncio.create_task(manager.reload())
        await asyncio.sleep(0.03)
        waiter = asyncio.create_task(manager.run(lambda: "replacement"))
        await asyncio.sleep(0.03)
        assert reload_task.done() is False
        assert waiter.done() is False

        release.set()
        await running
        await reload_task
        assert await waiter == "replacement"
    finally:
        release.set()
        pending = [running]
        if reload_task is not None:
            pending.append(reload_task)
        if waiter is not None:
            pending.append(waiter)
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_reload_does_not_abandon_transition(
    managers: list[ExtractionExecutorManager],
) -> None:
    manager = _own(managers, _manager())
    started = threading.Event()
    release = threading.Event()
    running = asyncio.create_task(manager.run(lambda: (started.set(), release.wait())))
    reload_task: asyncio.Task[None] | None = None
    try:
        assert await asyncio.to_thread(started.wait, 1.0) is True
        reload_task = asyncio.create_task(manager.reload())
        await asyncio.sleep(0.02)
        reload_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reload_task

        release.set()
        await running
        for _ in range(100):
            if manager.state is ManagerState.RUNNING:
                break
            await asyncio.sleep(0.01)
        assert manager.state is ManagerState.RUNNING
        assert await manager.run(lambda: "healthy") == "healthy"
    finally:
        release.set()
        pending = [running] if reload_task is None else [running, reload_task]
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_shutdown_is_terminal_for_waiters_and_later_calls(
    managers: list[ExtractionExecutorManager],
) -> None:
    manager = _own(managers, _manager())
    started = threading.Event()
    release = threading.Event()
    running = asyncio.create_task(manager.run(lambda: (started.set(), release.wait())))
    waiter: asyncio.Task[str] | None = None
    shutdown: asyncio.Task[None] | None = None
    try:
        assert await asyncio.to_thread(started.wait, 1.0) is True
        waiter = asyncio.create_task(manager.run(lambda: "never"))
        shutdown = asyncio.create_task(manager.shutdown())
        await asyncio.sleep(0.03)

        with pytest.raises(ArticleFailure) as waiting_failure:
            await waiter
        assert waiting_failure.value.stage == "shutdown"
        release.set()
        await running
        await shutdown
        assert manager.state is ManagerState.SHUTDOWN

        with pytest.raises(ArticleFailure) as later_failure:
            await manager.run(lambda: None)
        assert later_failure.value.code == "extraction_error"
        assert later_failure.value.stage == "shutdown"
    finally:
        release.set()
        pending = [running]
        if waiter is not None:
            pending.append(waiter)
        if shutdown is not None:
            pending.append(shutdown)
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_process_cleanup_during_reload_cannot_install_a_replacement(
    managers: list[ExtractionExecutorManager],
) -> None:
    manager = _own(managers, _manager())
    started = threading.Event()
    release = threading.Event()
    running = asyncio.create_task(manager.run(lambda: (started.set(), release.wait())))
    reload_task: asyncio.Task[None] | None = None
    cleanup_task: asyncio.Task[None] | None = None
    try:
        assert await asyncio.to_thread(started.wait, 1.0) is True
        reload_task = asyncio.create_task(manager.reload())
        for _ in range(100):
            if manager.state is ManagerState.RELOADING:
                break
            await asyncio.sleep(0.01)
        assert manager.state is ManagerState.RELOADING

        cleanup_task = asyncio.create_task(asyncio.to_thread(manager.close_at_exit))
        for _ in range(100):
            if manager.state is ManagerState.SHUTDOWN:
                break
            await asyncio.sleep(0.01)
        assert manager.state is ManagerState.SHUTDOWN
        release.set()
        await running
        await reload_task
        await cleanup_task

        assert manager.state is ManagerState.SHUTDOWN
        assert manager.current_generation is None
        with pytest.raises(ArticleFailure):
            await manager.run(lambda: "must-not-run")
    finally:
        release.set()
        pending = [running]
        if reload_task is not None:
            pending.append(reload_task)
        if cleanup_task is not None:
            pending.append(cleanup_task)
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_reset_is_the_only_way_to_leave_terminal_shutdown(
    managers: list[ExtractionExecutorManager],
) -> None:
    manager = _own(managers, _manager())
    await manager.shutdown()
    with pytest.raises(ArticleFailure):
        await manager.reload()

    manager.reset_for_tests()

    assert manager.state is ManagerState.RUNNING
    assert await manager.run(lambda: "reset") == "reset"


class _FakeExecutor:
    def __init__(self) -> None:
        self.shutdown_calls: list[dict[str, Any]] = []

    def submit(self, func: Any) -> Future[Any]:
        future: Future[Any] = Future()
        try:
            future.set_result(func())
        except BaseException as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future

    def shutdown(self, **kwargs: Any) -> None:
        self.shutdown_calls.append(dict(kwargs))


class _ShutdownFailExecutor(_FakeExecutor):
    def shutdown(self, **kwargs: Any) -> None:
        super().shutdown(**kwargs)
        raise RuntimeError("raw shutdown detail")


class _RecordingThreadExecutor:
    def __init__(self, worker_count: int) -> None:
        self._delegate = ThreadPoolExecutor(max_workers=worker_count)
        self.shutdown_calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def submit(self, func: Any) -> Future[Any]:
        return self._delegate.submit(func)

    def shutdown(self, **kwargs: Any) -> None:
        with self._lock:
            self.shutdown_calls.append(dict(kwargs))
        self._delegate.shutdown(**kwargs)


@pytest.mark.asyncio
async def test_concurrent_shutdown_callers_share_one_drain(
    managers: list[ExtractionExecutorManager],
) -> None:
    executors: list[_RecordingThreadExecutor] = []

    def factory(worker_count: int) -> _RecordingThreadExecutor:
        executor = _RecordingThreadExecutor(worker_count)
        executors.append(executor)
        return executor

    manager = _own(
        managers,
        ExtractionExecutorManager(
            worker_count_loader=lambda: 1,
            executor_factory=factory,
        ),
    )
    started = threading.Event()
    release = threading.Event()
    running = asyncio.create_task(manager.run(lambda: (started.set(), release.wait())))
    shutdowns: list[asyncio.Task[None]] = []
    try:
        assert await asyncio.to_thread(started.wait, 1.0) is True
        shutdowns = [asyncio.create_task(manager.shutdown()) for _ in range(3)]
        await asyncio.sleep(0.03)
        assert all(task.done() is False for task in shutdowns)
        release.set()
        await running
        await asyncio.gather(*shutdowns)
        assert manager.state is ManagerState.SHUTDOWN
        assert executors[0].shutdown_calls == [{"wait": True, "cancel_futures": False}]
    finally:
        release.set()
        await asyncio.gather(running, *shutdowns, return_exceptions=True)


def _fail_transition_thread_start(**_kwargs: Any) -> None:
    raise RuntimeError("raw thread detail")


@pytest.mark.asyncio
async def test_reload_thread_start_failure_drains_and_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executors: list[_FakeExecutor] = []

    def factory(_workers: int) -> _FakeExecutor:
        executor = _FakeExecutor()
        executors.append(executor)
        return executor

    manager = ExtractionExecutorManager(
        worker_count_loader=lambda: 1,
        executor_factory=factory,
    )
    await manager.run(lambda: None)
    first = manager.current_generation
    monkeypatch.setattr(
        manager,
        "_start_transition_thread",
        _fail_transition_thread_start,
    )

    try:
        with pytest.raises(ArticleFailure) as raised:
            await manager.reload()

        assert raised.value.stage == "reload"
        assert manager.state is ManagerState.RUNNING
        assert manager.current_generation is not first
        assert executors[0].shutdown_calls == [{"wait": True, "cancel_futures": False}]
    finally:
        if manager.state is ManagerState.RELOADING:
            with manager._lock:
                manager._transition = None
                manager._state = ManagerState.RUNNING
                manager._generation = first
                if first is not None:
                    first.closed = False
        manager.reset_for_tests()


@pytest.mark.asyncio
async def test_shutdown_thread_start_failure_drains_and_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _FakeExecutor()
    manager = ExtractionExecutorManager(
        worker_count_loader=lambda: 1,
        executor_factory=lambda _workers: executor,
    )
    await manager.run(lambda: None)
    first = manager.current_generation
    monkeypatch.setattr(
        manager,
        "_start_transition_thread",
        _fail_transition_thread_start,
    )

    try:
        with pytest.raises(ArticleFailure) as raised:
            await manager.shutdown()

        assert raised.value.stage == "shutdown"
        assert manager.state is ManagerState.SHUTDOWN
        assert manager.current_generation is None
        assert executor.shutdown_calls == [{"wait": True, "cancel_futures": False}]
    finally:
        if manager._transition is not None:
            with manager._lock:
                manager._transition = None
                manager._generation = first
        manager.reset_for_tests()


@pytest.mark.asyncio
async def test_shutdown_failure_is_sanitized_and_terminal(
    managers: list[ExtractionExecutorManager],
) -> None:
    executor = _ShutdownFailExecutor()
    manager = _own(
        managers,
        ExtractionExecutorManager(
            worker_count_loader=lambda: 1,
            executor_factory=lambda _workers: executor,
        ),
    )
    await manager.run(lambda: None)

    with pytest.raises(ArticleFailure) as raised:
        await manager.shutdown()

    assert raised.value.code == "extraction_error"
    assert raised.value.stage == "shutdown"
    assert manager.state is ManagerState.SHUTDOWN
    assert manager.current_generation is None
    assert manager.close_at_exit() is False
    assert executor.shutdown_calls == [
        {"wait": True, "cancel_futures": False},
        {"wait": True, "cancel_futures": False},
        {"wait": True, "cancel_futures": False},
    ]


@pytest.mark.asyncio
async def test_pid_mismatch_discards_inherited_generation_without_waiting(
    managers: list[ExtractionExecutorManager],
) -> None:
    pid = [100]
    executors: list[_FakeExecutor] = []

    def factory(_workers: int) -> _FakeExecutor:
        executor = _FakeExecutor()
        executors.append(executor)
        return executor

    manager = _own(
        managers,
        ExtractionExecutorManager(
            worker_count_loader=lambda: 1,
            executor_factory=factory,
            pid_getter=lambda: pid[0],
        ),
    )
    await manager.run(lambda: "parent")
    parent = manager.current_generation
    pid[0] = 200

    assert await manager.run(lambda: "child") == "child"

    child = manager.current_generation
    assert parent is not None and child is not None and child is not parent
    assert parent.executor.shutdown_calls == []
    assert child.pid == 200


@pytest.mark.asyncio
async def test_after_fork_hook_replaces_lock_and_discards_state(
    managers: list[ExtractionExecutorManager],
) -> None:
    manager = _own(managers, _manager())
    await manager.run(lambda: None)
    old_lock = manager._lock
    old_generation = manager.current_generation

    manager._after_fork_child()

    assert manager._lock is not old_lock
    assert manager.current_generation is None
    assert manager.state is ManagerState.RUNNING
    assert old_generation is not None and old_generation.closed is True
    assert await manager.run(lambda: "child") == "child"


def test_generation_contract_uses_thread_neutral_capacity() -> None:
    generation = ExecutorGeneration.create_for_tests(worker_count=1)
    try:
        assert generation.permits.acquire(blocking=False) is True
        assert generation.permits.acquire(blocking=False) is False
        generation.permits.release()
    finally:
        generation.executor.shutdown(wait=True, cancel_futures=False)
