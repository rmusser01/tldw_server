"""Focused tests for the governed legacy-analyzer compatibility boundary."""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import gc
import importlib
import inspect
import threading
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_type_hints

import pytest

from tldw_Server_API.app.core.Web_Scraping.policy.adapters import (
    DefaultProbeEgressGuard,
    DefaultWebOutboundPolicyChecker,
)
from tldw_Server_API.app.core.Web_Scraping.preflight import compatibility, facade
from tldw_Server_API.app.core.Web_Scraping.preflight.adapters.browser import (
    GuardedPlaywrightBrowserProbe,
)
from tldw_Server_API.app.core.Web_Scraping.preflight.adapters.external_tools import (
    GuardedExternalToolProbe,
)
from tldw_Server_API.app.core.Web_Scraping.preflight.adapters.http import (
    GuardedHttpProbe,
)
from tldw_Server_API.app.core.Web_Scraping.preflight.compatibility import (
    _BackgroundLoopBridge,
    _run_sync_compat,
)
from tldw_Server_API.app.core.Web_Scraping.preflight.context import PreflightLimits
from tldw_Server_API.app.core.Web_Scraping.preflight.facade import (
    PreflightAdapterOverrides,
    build_execution_context,
    run_legacy_analyzer,
)
from tldw_Server_API.app.core.Web_Scraping.preflight.options import PreflightOptions
from tldw_Server_API.app.core.Web_Scraping.preflight.probes import (
    BrowserProbe,
    ExternalToolProbe,
    HttpProbe,
)
from tldw_Server_API.app.core.Web_Scraping.preflight.target import PreflightTarget
from tldw_Server_API.app.core.Web_Scraping.preflight.utils.browser_identities import (
    MODERN_BROWSER_IDENTITIES,
)
from tldw_Server_API.app.core.Web_Scraping.runtime import (
    PolicyDecision,
    RuntimeRequestContext,
)
from tldw_Server_API.app.core.Web_Scraping.runtime.policy import (
    OutboundPolicyChecker,
    ProbeEgressGuard,
)
from tldw_Server_API.tests.Web_Scraping.preflight_fakes import (
    FakeBrowserProbe,
    FakeEgressGuard,
    FakeExternalToolProbe,
    FakeHttpProbe,
    FakeIdentitySelector,
    FakePolicyChecker,
)

pytestmark = pytest.mark.unit


class _AnalyzerFailure(Exception):
    pass


class _CountingClock:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return self.value


class _RecordingPolicyChecker:
    def __init__(
        self,
        decision: Any,
        *,
        error: BaseException | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.decision = decision
        self.error = error
        self.events = events
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def decide(self, url: str, **kwargs: Any) -> Any:
        self.calls.append((url, dict(kwargs)))
        if self.events is not None:
            self.events.append("policy")
        if self.error is not None:
            raise self.error
        return self.decision


class _RaisingAllowedDecision:
    @property
    def allowed(self) -> bool:
        raise RuntimeError("secret decision failure")


class _RecordingContext:
    def __init__(
        self,
        *,
        events: list[str] | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.close_error = close_error
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        if self.events is not None:
            self.events.append("close")
        if self.close_error is not None:
            raise self.close_error


def _decision(*, allowed: bool = True, reason: str = "allowed") -> PolicyDecision:
    return PolicyDecision(
        allowed=allowed,
        mode="compat",
        reason=reason,
        stage="preflight",
        source="preflight",
    )


def _target() -> PreflightTarget:
    return PreflightTarget(
        url="https://example.com/path",
        decision=_decision(),
        request_context=RuntimeRequestContext(
            source="test",
            stage="preflight",
            request_id="request-7",
        ),
    )


def _bridge_threads() -> set[threading.Thread]:
    return {thread for thread in threading.enumerate() if thread.name.startswith("preflight-compat-loop")}


def _assert_stopped(
    thread: threading.Thread,
    loop: asyncio.AbstractEventLoop,
) -> None:
    assert not thread.is_alive()
    assert loop.is_closed()
    assert all(task.done() for task in asyncio.all_tasks(loop))


@pytest.fixture
def bridge() -> _BackgroundLoopBridge:
    instance = _BackgroundLoopBridge()
    yield instance
    instance.shutdown()


@pytest.fixture(autouse=True)
def no_leaked_bridge_threads() -> Any:
    existing = _bridge_threads()
    yield
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        leaked = _bridge_threads() - existing
        if not leaked:
            break
        time.sleep(0.01)
    assert not (_bridge_threads() - existing)


def test_bridge_starts_lazily_and_reuses_one_process_thread(
    bridge: _BackgroundLoopBridge,
) -> None:
    assert bridge._thread is None
    assert bridge._loop is None

    assert bridge.submit(asyncio.sleep(0, result="first"), timeout_s=1.0) == "first"
    thread = bridge._thread
    loop = bridge._loop
    assert thread is not None and thread.is_alive()
    assert loop is not None and loop.is_running()

    assert bridge.submit(asyncio.sleep(0, result="second"), timeout_s=1.0) == "second"
    assert bridge._thread is thread
    assert bridge._loop is loop

    bridge.shutdown()
    _assert_stopped(thread, loop)


def test_first_submission_waits_until_run_forever_executes_ready_callback(
    bridge: _BackgroundLoopBridge,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.new_event_loop()
    original_run_forever = loop.run_forever
    run_forever_entered = threading.Event()
    allow_run_forever = threading.Event()
    results: list[str] = []
    errors: list[BaseException] = []

    def gated_run_forever() -> None:
        run_forever_entered.set()
        allow_run_forever.wait(1.0)
        original_run_forever()

    def submit() -> None:
        try:
            results.append(bridge.submit(asyncio.sleep(0, result="ready"), timeout_s=1.0))
        except BaseException as exc:  # noqa: BLE001 - assert exact caller outcome
            errors.append(exc)

    monkeypatch.setattr(compatibility.asyncio, "new_event_loop", lambda: loop)
    monkeypatch.setattr(loop, "run_forever", gated_run_forever)
    caller = threading.Thread(target=submit, name="task-7-startup-barrier-caller")
    caller.start()
    try:
        assert run_forever_entered.wait(1.0)
        time.sleep(0.05)
        assert caller.is_alive()
        assert errors == []
    finally:
        allow_run_forever.set()
        caller.join(2.0)

    assert not caller.is_alive()
    assert errors == []
    assert results == ["ready"]


def test_shutdown_queues_stop_while_initialized_loop_is_transitioning(
    bridge: _BackgroundLoopBridge,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.new_event_loop()
    original_run_forever = loop.run_forever
    run_forever_entered = threading.Event()
    allow_run_forever = threading.Event()
    caller_errors: list[BaseException] = []

    def gated_run_forever() -> None:
        run_forever_entered.set()
        allow_run_forever.wait(1.0)
        original_run_forever()

    async def wait_forever() -> None:
        await asyncio.Event().wait()

    def submit() -> None:
        try:
            bridge.submit(wait_forever())
        except (RuntimeError, concurrent.futures.CancelledError) as exc:
            caller_errors.append(exc)

    monkeypatch.setattr(compatibility.asyncio, "new_event_loop", lambda: loop)
    monkeypatch.setattr(loop, "run_forever", gated_run_forever)
    caller = threading.Thread(target=submit, name="task-7-transition-submit")
    caller.start()
    assert run_forever_entered.wait(1.0)

    shutdown = threading.Thread(target=bridge.shutdown, name="task-7-transition-shutdown")
    shutdown.start()
    try:
        time.sleep(0.05)
        allow_run_forever.set()
        shutdown.join(2.0)
        caller.join(2.0)
        assert not shutdown.is_alive()
        assert not caller.is_alive()
        assert caller_errors
        assert not loop.is_running()
        assert loop.is_closed()
    finally:
        allow_run_forever.set()
        if not loop.is_closed():
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
        shutdown.join(2.0)
        caller.join(2.0)


def test_bridge_preserves_return_values_and_exception_identity(
    bridge: _BackgroundLoopBridge,
) -> None:
    value = object()
    assert bridge.submit(asyncio.sleep(0, result=value), timeout_s=1.0) is value

    failure = _AnalyzerFailure("preserve me")

    async def fail() -> None:
        raise failure

    with pytest.raises(_AnalyzerFailure) as caught:
        bridge.submit(fail(), timeout_s=1.0)
    assert caught.value is failure


@pytest.mark.asyncio
async def test_sync_bridge_can_be_called_from_active_event_loop(
    bridge: _BackgroundLoopBridge,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compatibility, "_PROCESS_BRIDGE", bridge)
    assert _run_sync_compat(asyncio.sleep(0, result="ok"), timeout_s=1.0) == "ok"


def test_bridge_timeout_cancels_once_and_waits_for_awaited_cleanup(
    bridge: _BackgroundLoopBridge,
) -> None:
    cleanup_started = threading.Event()
    cleanup_finished = threading.Event()
    cancellations: list[str] = []

    async def work() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellations.append("cancelled")
            raise
        finally:
            cleanup_started.set()
            await asyncio.sleep(0.02)
            cleanup_finished.set()

    with pytest.raises(TimeoutError, match="^Legacy analyzer timed out\\.$"):
        bridge.submit(work(), timeout_s=0.01)

    assert cleanup_started.is_set()
    assert cleanup_finished.is_set()
    assert cancellations == ["cancelled"]


def test_shutdown_cancels_and_gathers_pending_tasks_before_loop_close(
    bridge: _BackgroundLoopBridge,
) -> None:
    started = threading.Event()
    cleaned = threading.Event()
    caller_errors: list[BaseException] = []

    async def work() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned.set()

    def submit() -> None:
        try:
            bridge.submit(work())
        except concurrent.futures.CancelledError as exc:
            caller_errors.append(exc)

    caller = threading.Thread(target=submit, name="task-7-submit-caller")
    caller.start()
    assert started.wait(1.0)
    bridge_thread = bridge._thread
    loop = bridge._loop
    assert bridge_thread is not None
    assert loop is not None

    bridge.shutdown()
    caller.join(1.0)

    assert not caller.is_alive()
    assert cleaned.is_set()
    assert len(caller_errors) == 1
    assert isinstance(caller_errors[0], concurrent.futures.CancelledError)
    _assert_stopped(bridge_thread, loop)


def test_shutdown_forces_second_cancellation_within_one_bounded_deadline(
    bridge: _BackgroundLoopBridge,
) -> None:
    started = threading.Event()
    first_cancelled = threading.Event()
    second_cancelled = threading.Event()
    cancellation_times: list[float] = []
    caller_errors: list[BaseException] = []
    loop_errors: list[dict[str, Any]] = []

    async def suppress_first_cancellation() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            first_cancelled.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_times.append(time.monotonic())
                second_cancelled.set()
                raise

    def submit() -> None:
        try:
            bridge.submit(suppress_first_cancellation())
        except concurrent.futures.CancelledError as exc:
            caller_errors.append(exc)

    caller = threading.Thread(target=submit, name="task-7-forced-cancel-caller")
    caller.start()
    assert started.wait(1.0)
    loop = bridge._loop
    thread = bridge._thread
    assert loop is not None
    assert thread is not None
    loop.call_soon_threadsafe(loop.set_exception_handler, lambda _loop, context: loop_errors.append(context))

    shutdown_started = time.monotonic()
    bridge.shutdown()
    shutdown_elapsed = time.monotonic() - shutdown_started
    caller.join(1.0)

    assert first_cancelled.is_set()
    assert second_cancelled.is_set()
    assert cancellation_times[0] - shutdown_started < 1.5
    assert shutdown_elapsed < 2.2
    assert not caller.is_alive()
    assert len(caller_errors) == 1
    assert not [context for context in loop_errors if "destroyed" in str(context.get("message", "")).lower()]
    _assert_stopped(thread, loop)


def test_shutdown_is_idempotent_and_rejects_owned_coroutines(
    bridge: _BackgroundLoopBridge,
) -> None:
    bridge.shutdown()
    bridge.shutdown()

    coroutine = asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="bridge has been shut down"):
        bridge.submit(coroutine)
    assert coroutine.cr_frame is None


def test_scheduling_failure_closes_the_unscheduled_coroutine(
    bridge: _BackgroundLoopBridge,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert bridge.submit(asyncio.sleep(0, result="started"), timeout_s=1.0) == "started"

    def fail_schedule(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("schedule failed")

    monkeypatch.setattr(compatibility.asyncio, "run_coroutine_threadsafe", fail_schedule)
    coroutine = asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="schedule failed"):
        bridge.submit(coroutine)
    assert coroutine.cr_frame is None


def test_shutdown_scheduling_failure_closes_cleanup_coroutine_without_warning(
    bridge: _BackgroundLoopBridge,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert bridge.submit(asyncio.sleep(0, result="started"), timeout_s=1.0) == "started"

    def fail_schedule(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("schedule failed")

    monkeypatch.setattr(compatibility.asyncio, "run_coroutine_threadsafe", fail_schedule)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bridge.shutdown()
        gc.collect()

    assert not [warning for warning in caught if "was never awaited" in str(warning.message)]


def test_pre_start_timeout_closes_unclaimed_coroutine_without_warning(
    bridge: _BackgroundLoopBridge,
) -> None:
    assert bridge.submit(asyncio.sleep(0, result="started"), timeout_s=1.0) == "started"
    loop = bridge._loop
    assert loop is not None
    blocking_started = threading.Event()
    release_loop = threading.Event()

    def block_loop() -> None:
        blocking_started.set()
        release_loop.wait(2.0)

    loop.call_soon_threadsafe(block_loop)
    assert blocking_started.wait(1.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coroutine = asyncio.sleep(0)
        with pytest.raises(TimeoutError, match="^Legacy analyzer timed out\\.$"):
            bridge.submit(coroutine, timeout_s=0.01)
        assert coroutine.cr_frame is None
        release_loop.set()
        assert bridge.submit(asyncio.sleep(0, result="flushed"), timeout_s=1.0) == "flushed"
        del coroutine
        gc.collect()

    assert not [warning for warning in caught if "was never awaited" in str(warning.message)]


def test_startup_timeout_is_bounded_and_closes_owned_coroutine(
    bridge: _BackgroundLoopBridge,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_new_event_loop = asyncio.new_event_loop
    startup_entered = threading.Event()
    release_startup = threading.Event()
    startup_resumed = threading.Event()
    allow_loop_creation = threading.Event()

    def stalled_new_event_loop() -> asyncio.AbstractEventLoop:
        startup_entered.set()
        release_startup.wait(3.0)
        startup_resumed.set()
        allow_loop_creation.wait(3.0)
        return original_new_event_loop()

    monkeypatch.setattr(compatibility.asyncio, "new_event_loop", stalled_new_event_loop)
    coroutine = asyncio.sleep(0)
    started_at = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="failed to start"):
            bridge.submit(coroutine)
        startup_elapsed = time.monotonic() - started_at
        assert startup_entered.is_set()
        assert startup_elapsed < 2.5
        assert coroutine.cr_frame is None
        thread = bridge._thread
        assert thread is not None and thread.is_alive()

        release_startup.set()
        assert startup_resumed.wait(1.0)

        def finish_startup() -> None:
            time.sleep(0.05)
            allow_loop_creation.set()

        finisher = threading.Thread(target=finish_startup, name="task-7-startup-finisher")
        finisher.start()
        shutdown_started = time.monotonic()
        bridge.shutdown()
        shutdown_elapsed = time.monotonic() - shutdown_started
        finisher.join(1.0)

        assert 0.04 <= shutdown_elapsed < 2.5
        assert not thread.is_alive()
    finally:
        release_startup.set()
        allow_loop_creation.set()
        thread = bridge._thread
        if thread is not None:
            thread.join(1.0)

    assert thread is not None and not thread.is_alive()


def test_failed_submissions_emit_no_unawaited_coroutine_warning(
    bridge: _BackgroundLoopBridge,
) -> None:
    bridge.shutdown()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coroutine = asyncio.sleep(0)
        with pytest.raises(RuntimeError):
            bridge.submit(coroutine)
        del coroutine
        gc.collect()
    assert not [warning for warning in caught if "was never awaited" in str(warning.message)]


def test_pid_change_retires_a_live_same_process_thread_and_recreates_state(
    bridge: _BackgroundLoopBridge,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert bridge.submit(asyncio.sleep(0, result="parent"), timeout_s=1.0) == "parent"
    old_pid = bridge._owner_pid
    old_lock = bridge._lock
    old_thread = bridge._thread
    old_loop = bridge._loop
    old_generation = bridge._generation
    assert old_pid is not None
    assert old_thread is not None
    assert old_loop is not None

    monkeypatch.setattr(compatibility.os, "getpid", lambda: old_pid + 1000)
    assert bridge.submit(asyncio.sleep(0, result="child"), timeout_s=1.0) == "child"

    assert not old_thread.is_alive()
    assert old_loop.is_closed()
    assert bridge._owner_pid == old_pid + 1000
    assert bridge._lock is old_lock
    assert bridge._generation == old_generation + 1
    assert bridge._thread is not old_thread
    assert bridge._thread is not None and bridge._thread.is_alive()


def test_pid_change_replaces_inherited_mutex_without_touching_parent_loop(
    bridge: _BackgroundLoopBridge,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InheritedLock:
        def __enter__(self) -> None:
            raise AssertionError("inherited mutex was acquired")

        def __exit__(self, *_args: Any) -> None:
            return None

    class ParentLoop:
        def call_soon_threadsafe(self, *_args: Any) -> None:
            raise AssertionError("parent loop was called")

        def close(self) -> None:
            raise AssertionError("parent loop was closed")

    owner_pid = compatibility.os.getpid() - 1
    bridge._owner_pid = owner_pid
    bridge._real_pid -= 1
    bridge._lock = InheritedLock()  # type: ignore[assignment]
    bridge._loop = ParentLoop()  # type: ignore[assignment]
    bridge._thread = threading.Thread(name="inherited-dead-thread")

    assert bridge.submit(asyncio.sleep(0, result="child"), timeout_s=1.0) == "child"
    assert bridge._owner_pid == compatibility.os.getpid()
    assert bridge._thread is not None and bridge._thread.is_alive()


def test_actual_pid_change_resets_ownerless_state_before_inherited_lock() -> None:
    class InheritedLock:
        def __enter__(self) -> None:
            raise AssertionError("inherited mutex was acquired")

        def __exit__(self, *_args: Any) -> None:
            return None

    class ParentLoop:
        def call_soon_threadsafe(self, *_args: Any) -> None:
            raise AssertionError("parent loop was called")

        def close(self) -> None:
            raise AssertionError("parent loop was closed")

    inherited_lock = InheritedLock()
    bridge = _BackgroundLoopBridge()
    bridge._owner_pid = None
    bridge._real_pid -= 1
    bridge._lock = inherited_lock  # type: ignore[assignment]
    bridge._loop = ParentLoop()  # type: ignore[assignment]
    bridge._thread = threading.Thread(name="inherited-ownerless-dead-thread")
    try:
        assert bridge.submit(asyncio.sleep(0, result="child"), timeout_s=1.0) == "child"
        assert bridge._lock is not inherited_lock
        assert bridge._owner_pid == compatibility.os.getpid()
        assert bridge._thread is not None and bridge._thread.is_alive()
    finally:
        bridge.shutdown()


def test_concurrent_simulated_pid_reset_has_one_generation_and_live_loop(
    bridge: _BackgroundLoopBridge,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert bridge.submit(asyncio.sleep(0, result="parent"), timeout_s=1.0) == "parent"
    old_pid = bridge._owner_pid
    old_thread = bridge._thread
    old_loop = bridge._loop
    old_generation = bridge._generation
    assert old_pid is not None
    assert old_thread is not None
    assert old_loop is not None

    monkeypatch.setattr(compatibility.os, "getpid", lambda: old_pid + 1000)
    start = threading.Barrier(3)
    results: list[str] = []
    errors: list[BaseException] = []

    def submit(value: str) -> None:
        start.wait()
        try:
            results.append(bridge.submit(asyncio.sleep(0, result=value), timeout_s=1.0))
        except BaseException as exc:  # noqa: BLE001 - assert concurrent outcome
            errors.append(exc)

    callers = [
        threading.Thread(target=submit, args=(value,), name=f"task-7-pid-reset-{value}") for value in ("one", "two")
    ]
    for caller in callers:
        caller.start()
    start.wait()
    for caller in callers:
        caller.join(3.0)

    assert not any(caller.is_alive() for caller in callers)
    assert errors == []
    assert sorted(results) == ["one", "two"]
    assert bridge._generation == old_generation + 1
    assert not old_thread.is_alive()
    assert old_loop.is_closed()
    assert bridge._thread is not old_thread
    assert bridge._thread is not None and bridge._thread.is_alive()
    assert bridge._loop is not old_loop


def test_process_exit_hook_shuts_down_the_process_singleton(
    bridge: _BackgroundLoopBridge,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compatibility, "_PROCESS_BRIDGE", bridge)
    assert bridge.submit(asyncio.sleep(0, result="ok"), timeout_s=1.0) == "ok"
    thread = bridge._thread
    loop = bridge._loop
    assert thread is not None
    assert loop is not None

    compatibility._shutdown_process_bridge()

    _assert_stopped(thread, loop)


def test_build_execution_context_shares_clock_deadline_guard_and_controls() -> None:
    target = _target()
    checker = FakePolicyChecker()
    guard = FakeEgressGuard()
    clock = _CountingClock(100.0)

    async def sleep(_delay_s: float) -> None:
        return None

    context = build_execution_context(
        target,
        PreflightOptions(timeout_s=5.0, playwright_no_sandbox=True),
        policy_checker=checker,
        injected_adapters=PreflightAdapterOverrides(
            egress_guard=guard,
            clock=clock,
            sleep=sleep,
        ),
    )

    assert clock.calls == 1
    assert context.request_context is target.request_context
    assert context.policy_checker is checker
    assert context.egress_guard is guard
    assert context.controls.deadline == 105.0
    assert context.controls._clock is clock
    assert context.controls._sleep is sleep
    assert context.controls.limits == PreflightLimits()
    assert isinstance(context.http, GuardedHttpProbe)
    assert isinstance(context.browser, GuardedPlaywrightBrowserProbe)
    assert isinstance(context.external_tools, GuardedExternalToolProbe)
    assert context.http._controls is context.controls
    assert context.browser._controls is context.controls
    assert context.external_tools._controls is context.controls
    assert context.http._egress_guard is guard
    assert context.browser._guard is guard
    assert context.external_tools._egress_guard is guard
    assert context.browser._no_sandbox is True


def test_public_facade_type_hints_resolve_runtime_protocols() -> None:
    override_hints = get_type_hints(PreflightAdapterOverrides)
    assert override_hints["http"] == HttpProbe | None
    assert override_hints["browser"] == BrowserProbe | None
    assert override_hints["external_tools"] == ExternalToolProbe | None
    assert override_hints["egress_guard"] == ProbeEgressGuard | None

    evaluate_hints = get_type_hints(facade.evaluate_target)
    build_hints = get_type_hints(build_execution_context)
    helper_hints = get_type_hints(run_legacy_analyzer)
    assert evaluate_hints["policy_checker"] is OutboundPolicyChecker
    assert build_hints["policy_checker"] == OutboundPolicyChecker | None
    assert helper_hints["policy_checker_factory"] == Callable[[], OutboundPolicyChecker]


@pytest.mark.parametrize("timeout_s", [None, 0.0, -1.0])
def test_build_execution_context_does_not_read_clock_without_positive_timeout(
    timeout_s: float | None,
) -> None:
    clock = _CountingClock(20.0)
    context = build_execution_context(
        _target(),
        PreflightOptions(timeout_s=timeout_s),
        injected_adapters=PreflightAdapterOverrides(clock=clock),
    )
    assert clock.calls == 0
    assert context.controls.deadline is None


def test_build_execution_context_defaults_policy_limits_and_egress_guard() -> None:
    context = build_execution_context(_target(), PreflightOptions())
    assert isinstance(context.policy_checker, DefaultWebOutboundPolicyChecker)
    assert isinstance(context.egress_guard, DefaultProbeEgressGuard)
    assert context.controls.limits.requests is None
    assert context.controls.limits.browsers is None
    assert context.controls.limits.active_probes is None


@pytest.mark.parametrize(
    ("slot", "fake", "expected_default"),
    [
        ("http", FakeHttpProbe(), GuardedHttpProbe),
        ("browser", FakeBrowserProbe(), GuardedPlaywrightBrowserProbe),
        ("external_tools", FakeExternalToolProbe(), GuardedExternalToolProbe),
    ],
)
def test_adapter_override_replaces_only_its_own_slot(
    slot: str,
    fake: Any,
    expected_default: type[Any],
) -> None:
    overrides = PreflightAdapterOverrides(**{slot: fake})
    context = build_execution_context(_target(), PreflightOptions(), injected_adapters=overrides)

    assert getattr(context, slot) is fake
    for other_slot, default_type in (
        ("http", GuardedHttpProbe),
        ("browser", GuardedPlaywrightBrowserProbe),
        ("external_tools", GuardedExternalToolProbe),
    ):
        if other_slot != slot:
            assert isinstance(getattr(context, other_slot), default_type)
    assert not isinstance(getattr(context, slot), expected_default)


def test_injected_browser_is_not_modified_by_no_sandbox_option() -> None:
    browser = FakeBrowserProbe()
    context = build_execution_context(
        _target(),
        PreflightOptions(playwright_no_sandbox=True),
        injected_adapters=PreflightAdapterOverrides(browser=browser),
    )
    assert context.browser is browser
    assert not hasattr(browser, "_no_sandbox")


def test_build_execution_context_uses_explicit_limits() -> None:
    limits = PreflightLimits(requests=2, browsers=1, active_probes=3)
    context = build_execution_context(_target(), PreflightOptions(), limits=limits)
    assert context.controls.limits is limits


def test_browser_identity_is_selected_lazily_once_and_defensively_copied() -> None:
    selector = FakeIdentitySelector({"User-Agent": "test-agent", "x-test": "original"})
    context = build_execution_context(
        _target(),
        PreflightOptions(),
        identity_selector=selector,
    )
    assert selector.calls == 0

    first = context.browser_identity()
    first["x-test"] = "mutated"
    selector.identity["User-Agent"] = "changed-after-selection"

    assert context.browser_identity() == {
        "User-Agent": "test-agent",
        "x-test": "original",
    }
    assert selector.calls == 1


def test_default_browser_identity_is_lazy_cached_and_uses_canonical_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = MODERN_BROWSER_IDENTITIES[0]
    expected = dict(selected)
    choice_calls: list[Any] = []

    def choose(identities: Any) -> Any:
        choice_calls.append(identities)
        assert identities is MODERN_BROWSER_IDENTITIES
        return selected

    monkeypatch.setattr(facade.random, "choice", choose)
    context = build_execution_context(_target(), PreflightOptions())
    assert choice_calls == []

    first = context.browser_identity()
    first["User-Agent"] = "caller-mutated"
    monkeypatch.setitem(selected, "User-Agent", "source-mutated")

    assert context.browser_identity() == expected
    assert choice_calls == [MODERN_BROWSER_IDENTITIES]


def test_utility_shims_reexport_canonical_public_surface() -> None:
    from tldw_Server_API.app.core.Web_Scraping.preflight.utils.browser_identities import (
        MODERN_BROWSER_IDENTITIES as CANONICAL_IDENTITIES,
    )
    from tldw_Server_API.app.core.Web_Scraping.preflight.utils.impersonate_target import (
        get_impersonate_target as canonical_impersonate_target,
    )
    from tldw_Server_API.app.core.Web_Scraping.preflight.utils.waf_result_parser import (
        ANSI_RE as CANONICAL_ANSI_RE,
    )
    from tldw_Server_API.app.core.Web_Scraping.preflight.utils.waf_result_parser import (
        GENERIC_PHRASES as CANONICAL_GENERIC_PHRASES,
    )
    from tldw_Server_API.app.core.Web_Scraping.preflight.utils.waf_result_parser import (
        clean_text as canonical_clean_text,
    )
    from tldw_Server_API.app.core.Web_Scraping.preflight.utils.waf_result_parser import (
        parse_wafw00f_output as canonical_waf_parser,
    )
    from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.utils.browser_identities import (
        MODERN_BROWSER_IDENTITIES as LEGACY_IDENTITIES,
    )
    from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.utils.impersonate_target import (
        get_impersonate_target as legacy_impersonate_target,
    )
    from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.utils.waf_result_parser import (
        ANSI_RE as LEGACY_ANSI_RE,
    )
    from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.utils.waf_result_parser import (
        GENERIC_PHRASES as LEGACY_GENERIC_PHRASES,
    )
    from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.utils.waf_result_parser import (
        clean_text as legacy_clean_text,
    )
    from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.utils.waf_result_parser import (
        parse_wafw00f_output as legacy_waf_parser,
    )

    assert LEGACY_IDENTITIES is CANONICAL_IDENTITIES  # nosec B101
    assert legacy_impersonate_target is canonical_impersonate_target  # nosec B101
    assert legacy_waf_parser is canonical_waf_parser  # nosec B101
    assert LEGACY_ANSI_RE is CANONICAL_ANSI_RE  # nosec B101
    assert LEGACY_GENERIC_PHRASES is CANONICAL_GENERIC_PHRASES  # nosec B101
    assert legacy_clean_text is canonical_clean_text  # nosec B101


def test_nonbrowser_analyzer_shims_preserve_identity_signatures_and_classification() -> None:
    module_contracts = {
        "robots_checker": {
            "exports": ["check_robots_txt"],
            "signature": "(url: 'str') -> 'dict[str, Any]'",
            "async": False,
        },
        "tls_analyzer": {
            "exports": ["analyze_tls_fingerprint"],
            "signature": "(url: 'str') -> 'dict[str, Any]'",
            "async": True,
        },
        "rate_limit_profiler": {
            "exports": [
                "GENTLE_PROBE_COUNT",
                "BURST_COUNT",
                "DEFAULT_DELAY",
                "BLOCKING_STATUS_CODES",
                "profile_rate_limits",
            ],
            "signature": (
                "(url: 'str', crawl_delay: 'float | None', impersonate: 'bool' = False) " "-> 'dict[str, Any]'"
            ),
            "async": True,
        },
        "waf_detector": {
            "exports": ["detect_waf"],
            "signature": "(url: 'str', find_all: 'bool' = False) -> 'dict[str, Any]'",
            "async": False,
        },
    }
    public_names = {
        "robots_checker": "check_robots_txt",
        "tls_analyzer": "analyze_tls_fingerprint",
        "rate_limit_profiler": "profile_rate_limits",
        "waf_detector": "detect_waf",
    }

    for module_name, contract in module_contracts.items():
        canonical = importlib.import_module(f"tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.{module_name}")
        legacy = importlib.import_module(
            "tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.analyzers." f"{module_name}"
        )
        public_name = public_names[module_name]
        canonical_callable = getattr(canonical, public_name)
        legacy_callable = getattr(legacy, public_name)

        assert legacy_callable is canonical_callable  # nosec B101
        assert str(inspect.signature(canonical_callable)) == contract["signature"]
        assert inspect.iscoroutinefunction(canonical_callable) is contract["async"]
        assert inspect.iscoroutinefunction(legacy_callable) is contract["async"]
        assert len(legacy.__all__) == len(contract["exports"])
        assert set(legacy.__all__) == set(contract["exports"])
        assert not hasattr(legacy, f"_{public_name}")
        assert not hasattr(canonical, "BROWSER_IDENTITY")
        assert not hasattr(legacy, "BROWSER_IDENTITY")

        source_path = Path(legacy.__file__ or "")
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree)
        )
        assert not any(
            isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names)
            for node in ast.walk(tree)
        )
        assert "warnings" not in {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }

    canonical_rate = importlib.import_module(
        "tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.rate_limit_profiler"
    )
    legacy_rate = importlib.import_module(
        "tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.analyzers.rate_limit_profiler"
    )
    assert canonical_rate.GENTLE_PROBE_COUNT == 4
    assert canonical_rate.BURST_COUNT == 8
    assert canonical_rate.DEFAULT_DELAY == 3.0
    assert {401, 403, 429, 503} == canonical_rate.BLOCKING_STATUS_CODES
    assert legacy_rate.BLOCKING_STATUS_CODES is canonical_rate.BLOCKING_STATUS_CODES


def test_nonbrowser_canonical_analyzers_have_no_concrete_probe_dependencies() -> None:
    forbidden_import_fragments = {
        "http_client",
        "playwright",
        "curl_cffi",
        "subprocess",
        "preflight.adapters",
        "policy.adapters",
    }

    for module_name in (
        "robots_checker",
        "tls_analyzer",
        "rate_limit_profiler",
        "waf_detector",
    ):
        module = importlib.import_module(f"tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.{module_name}")
        tree = ast.parse(Path(module.__file__ or "").read_text(encoding="utf-8"))
        imported_modules = {
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
        } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}

        assert not {
            imported
            for imported in imported_modules
            if any(fragment in imported for fragment in forbidden_import_fragments)
        }


def _context_factory(
    context: _RecordingContext,
    calls: list[tuple[PreflightTarget, PreflightOptions, Any]],
    *,
    events: list[str] | None = None,
) -> Callable[..., _RecordingContext]:
    def build(
        target: PreflightTarget,
        options: PreflightOptions,
        *,
        policy_checker: Any,
    ) -> _RecordingContext:
        calls.append((target, options, policy_checker))
        if events is not None:
            events.append("context")
        return context

    return build


@pytest.mark.asyncio
async def test_run_legacy_analyzer_evaluates_then_forwards_arguments_and_closes() -> None:
    events: list[str] = []
    checker = _RecordingPolicyChecker(_decision(), events=events)
    context = _RecordingContext(events=events)
    context_calls: list[tuple[PreflightTarget, PreflightOptions, Any]] = []
    returned = object()

    async def analyzer(
        url: str,
        selected_context: _RecordingContext,
        count: int,
        *,
        enabled: bool,
    ) -> object:
        events.append("analyzer")
        assert url == "https://example.com/direct"
        assert selected_context is context
        assert count == 7
        assert enabled is True
        return returned

    result = await run_legacy_analyzer(
        "https://example.com/direct",
        analyzer,
        7,
        enabled=True,
        policy_checker_factory=lambda: checker,
        context_factory=_context_factory(context, context_calls, events=events),
    )

    assert result is returned
    assert events == ["policy", "context", "analyzer", "close"]
    assert len(checker.calls) == 1
    checked_url, checked_kwargs = checker.calls[0]
    assert checked_url == "https://example.com/direct"
    assert checked_kwargs["respect_robots"] is False
    assert checked_kwargs["user_agent"] is None
    assert checked_kwargs["config"] is None
    request_context = checked_kwargs["context"]
    assert request_context.source == "preflight"
    assert request_context.stage == "preflight"
    assert len(context_calls) == 1
    target, options, selected_checker = context_calls[0]
    assert target.url == "https://example.com/direct"
    assert target.request_context is request_context
    assert options == PreflightOptions()
    assert selected_checker is checker
    assert context.close_calls == 1


@pytest.mark.asyncio
async def test_run_legacy_analyzer_returns_exact_denial_without_context_or_probe() -> None:
    checker = _RecordingPolicyChecker(_decision(allowed=False, reason="host_denied"))
    context_called = False
    analyzer_called = False

    def context_factory(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal context_called
        context_called = True
        raise AssertionError("context must not be built")

    async def analyzer(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal analyzer_called
        analyzer_called = True
        raise AssertionError("analyzer must not run")

    result = await run_legacy_analyzer(
        "https://example.com/denied",
        analyzer,
        policy_checker_factory=lambda: checker,
        context_factory=context_factory,
    )

    assert result == {
        "status": "error",
        "message": "Probe destination was denied.",
        "error_code": "policy_denied",
    }
    assert context_called is False
    assert analyzer_called is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "checker",
    [
        _RecordingPolicyChecker(
            _decision(),
            error=RuntimeError("https://user:secret@example.com/private"),
        ),
        _RecordingPolicyChecker(_RaisingAllowedDecision()),
    ],
)
async def test_run_legacy_analyzer_sanitizes_checker_and_decision_failures(
    checker: _RecordingPolicyChecker,
) -> None:
    result = await run_legacy_analyzer(
        "https://example.com/failure",
        lambda *_args, **_kwargs: asyncio.sleep(0),
        policy_checker_factory=lambda: checker,
        context_factory=lambda *_args, **_kwargs: pytest.fail("context constructed"),
    )
    assert result == {
        "status": "error",
        "message": "Probe destination was denied.",
        "error_code": "policy_error",
    }
    assert "secret" not in str(result)


@pytest.mark.asyncio
async def test_run_legacy_analyzer_sanitizes_policy_checker_factory_failure() -> None:
    context_called = False
    analyzer_called = False

    def fail_policy_checker_factory() -> Any:
        raise RuntimeError("https://user:secret@example.com/private")

    def context_factory(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal context_called
        context_called = True
        raise AssertionError("context must not be built")

    async def analyzer(*_args: Any, **_kwargs: Any) -> None:
        nonlocal analyzer_called
        analyzer_called = True

    result = await run_legacy_analyzer(
        "https://example.com/factory-failure",
        analyzer,
        policy_checker_factory=fail_policy_checker_factory,
        context_factory=context_factory,
    )

    assert result == {
        "status": "error",
        "message": "Probe destination was denied.",
        "error_code": "policy_error",
    }
    assert "secret" not in str(result)
    assert context_called is False
    assert analyzer_called is False


@pytest.mark.asyncio
async def test_run_legacy_analyzer_propagates_policy_cancellation() -> None:
    checker = _RecordingPolicyChecker(_decision(), error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await run_legacy_analyzer(
            "https://example.com/cancel",
            lambda *_args, **_kwargs: asyncio.sleep(0),
            policy_checker_factory=lambda: checker,
            context_factory=lambda *_args, **_kwargs: pytest.fail("context constructed"),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_analyzer", [False, True])
async def test_run_legacy_analyzer_preserves_analyzer_failures_and_cancellation_after_close(
    cancel_analyzer: bool,
) -> None:
    checker = _RecordingPolicyChecker(_decision())
    context = _RecordingContext()
    context_calls: list[tuple[PreflightTarget, PreflightOptions, Any]] = []
    failure: BaseException = asyncio.CancelledError() if cancel_analyzer else _AnalyzerFailure("exact failure")

    async def analyzer(*_args: Any, **_kwargs: Any) -> None:
        raise failure

    with pytest.raises(type(failure)) as caught:
        await run_legacy_analyzer(
            "https://example.com/analyzer-failure",
            analyzer,
            policy_checker_factory=lambda: checker,
            context_factory=_context_factory(context, context_calls),
        )

    assert caught.value is failure
    assert context.close_calls == 1


@pytest.mark.asyncio
async def test_run_legacy_analyzer_cleanup_failure_does_not_replace_analyzer_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = _RecordingPolicyChecker(_decision())
    context = _RecordingContext(close_error=RuntimeError("sensitive cleanup detail"))
    context_calls: list[tuple[PreflightTarget, PreflightOptions, Any]] = []
    returned = object()
    warnings_logged: list[str] = []

    monkeypatch.setattr(facade.logger, "warning", warnings_logged.append)

    async def analyzer(*_args: Any, **_kwargs: Any) -> object:
        return returned

    result = await run_legacy_analyzer(
        "https://example.com/cleanup",
        analyzer,
        policy_checker_factory=lambda: checker,
        context_factory=_context_factory(context, context_calls),
    )
    assert result is returned
    assert context.close_calls == 1
    assert warnings_logged == ["Legacy analyzer context cleanup failed."]
    assert "sensitive cleanup detail" not in str(warnings_logged)
