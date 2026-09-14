import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import (
    AdmittedModuleOperation,
    BaseModule,
    ModuleCircuitBreakerOpenError,
    ModuleConfig,
)

pytestmark = pytest.mark.unit


class SlowModule(BaseModule):
    def __init__(self, config: ModuleConfig):
        super().__init__(config)
        self.current = 0
        self.max_seen = 0

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict]:
        return []

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context=None):
        return None

    async def _work(self, delay: float):
        self.current += 1
        self.max_seen = max(self.max_seen, self.current)
        try:
            await asyncio.sleep(delay)
        finally:
            self.current -= 1
        return "done"


@pytest.mark.asyncio
async def test_per_module_concurrency_guard_limits_parallelism():
    # Allow only 2 concurrent ops
    mod = SlowModule(ModuleConfig(name="slow", max_concurrent=2))

    async def call_once():
        return await mod.execute_with_circuit_breaker(mod._work, 0.05)

    # Schedule 5 parallel calls
    tasks = [asyncio.create_task(call_once()) for _ in range(5)]
    await asyncio.gather(*tasks)

    # Ensure observed concurrency was limited to 2
    assert mod.max_seen <= 2


@pytest.mark.asyncio
async def test_per_module_concurrency_queue_wait_is_bounded_by_operation_timeout() -> None:
    mod = SlowModule(
        ModuleConfig(name="bounded_queue", max_concurrent=1, timeout_seconds=0.01)
    )
    assert mod._semaphore is not None
    await mod._semaphore.acquire()

    try:
        with pytest.raises(Exception, match="Operation timeout after 0.01s"):
            await asyncio.wait_for(
                mod.execute_with_circuit_breaker(mod._work, 0),
                timeout=0.5,
            )
    finally:
        mod._semaphore.release()

    assert mod.current == 0


@pytest.mark.asyncio
async def test_admission_check_dispatches_before_scheduled_definition_drift() -> None:
    mod = SlowModule(ModuleConfig(name="admission_handoff", max_concurrent=1))
    definition = {"version": "prepared"}
    observed_versions: list[str] = []

    class _MutatingBreaker:
        async def call_async(self, operation: Callable[[], Awaitable[Any]]) -> Any:
            definition["version"] = "changed"
            return await operation()

    async def _admit() -> None:
        if definition["version"] != "prepared":
            raise RuntimeError("stale definition")

    async def _dispatch() -> str:
        observed_versions.append(definition["version"])
        await asyncio.sleep(0)
        return "done"

    mod._circuit_breaker = _MutatingBreaker()
    with pytest.raises(RuntimeError, match="stale definition"):
        await mod.execute_with_circuit_breaker(AdmittedModuleOperation(_admit, _dispatch))

    assert observed_versions == []


@pytest.mark.asyncio
async def test_open_breaker_rejects_before_saturated_concurrency_queue() -> None:
    mod = FlappyModule(
        ModuleConfig(
            name="open_before_queue",
            max_concurrent=1,
            timeout_seconds=0.01,
            circuit_breaker_threshold=1,
            circuit_breaker_timeout=60,
        )
    )
    with pytest.raises(RuntimeError):
        await mod.execute_with_circuit_breaker(mod._always_fail)

    assert mod._semaphore is not None
    await mod._semaphore.acquire()
    try:
        with pytest.raises(ModuleCircuitBreakerOpenError):
            await asyncio.wait_for(
                mod.execute_with_circuit_breaker(mod._always_fail),
                timeout=0.5,
            )
    finally:
        mod._semaphore.release()


class FlappyModule(BaseModule):
    def __init__(self, config: ModuleConfig):
        super().__init__(config)
        self.calls = 0

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict]:
        return []

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context=None):
        return None

    async def _always_fail(self):
        self.calls += 1
        raise RuntimeError("fail")


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_with_backoff_behaves():
    # Threshold=1, initial timeout=0.01s: near-instant half-open; backoff should extend
    mod = FlappyModule(ModuleConfig(
        name="flappy",
        circuit_breaker_threshold=1,
        circuit_breaker_timeout=0.01,
        circuit_breaker_backoff_factor=100.0,
        circuit_breaker_max_timeout=2,
    ))

    # First failure -> open
    with pytest.raises(RuntimeError):
        await mod.execute_with_circuit_breaker(mod._always_fail)

    # Wait for the short initial recovery timeout to expire
    await asyncio.sleep(0.02)
    # Next attempt should enter half-open (is_circuit_breaker_open returns False)
    assert mod.is_circuit_breaker_open() is False

    # Fail again in half-open -> re-open with backoff (longer timeout now)
    with pytest.raises(RuntimeError):
        await mod.execute_with_circuit_breaker(mod._always_fail)

    # Backoff should have increased recovery timeout, so breaker stays OPEN
    assert mod._circuit_breaker._current_recovery_timeout > 0.01
    assert mod.is_circuit_breaker_open() is True


@pytest.mark.asyncio
async def test_fallback_circuit_breaker_recovery_at_uses_epoch_time():
    mod = FlappyModule(ModuleConfig(
        name="fallback_recovery_at",
        circuit_breaker_threshold=1,
        circuit_breaker_timeout=2,
        circuit_breaker_factory=None,
    ))

    with pytest.raises(RuntimeError):
        await mod.execute_with_circuit_breaker(mod._always_fail)

    before_rejection = time.time()
    with pytest.raises(ModuleCircuitBreakerOpenError) as exc_info:
        await mod.execute_with_circuit_breaker(mod._always_fail)
    after_rejection = time.time()

    assert exc_info.value.recovery_at is not None
    assert before_rejection < exc_info.value.recovery_at <= after_rejection + 2


@pytest.mark.asyncio
async def test_fallback_circuit_breaker_limits_half_open_probe_concurrency():
    started = asyncio.Event()
    release = asyncio.Event()
    active_probes = 0

    async def _slow_success() -> str:
        nonlocal active_probes
        active_probes += 1
        started.set()
        try:
            await release.wait()
        finally:
            active_probes -= 1
        return "ok"

    mod = FlappyModule(ModuleConfig(
        name="fallback_half_open_limit",
        circuit_breaker_threshold=1,
        circuit_breaker_timeout=0.01,
        circuit_breaker_factory=None,
    ))

    with pytest.raises(RuntimeError):
        await mod.execute_with_circuit_breaker(mod._always_fail)
    await asyncio.sleep(0.02)

    first_probe = asyncio.create_task(mod.execute_with_circuit_breaker(_slow_success))
    await started.wait()
    assert active_probes == 1

    with pytest.raises(ModuleCircuitBreakerOpenError):
        await mod.execute_with_circuit_breaker(_slow_success)

    release.set()
    assert await first_probe == "ok"


def _breaker_module(
    breaker_kind: str,
    *,
    threshold: int = 3,
    recovery_timeout: float = 0.25,
) -> FlappyModule:
    """Build a module backed by the requested circuit-breaker implementation."""

    circuit_breaker_factory = None
    if breaker_kind == "injected":
        from tldw_Server_API.app.core.MCP_unified.adapters.tldw_runtime import (
            create_tldw_circuit_breaker,
        )

        circuit_breaker_factory = create_tldw_circuit_breaker
    return FlappyModule(
        ModuleConfig(
            name=f"outcomes_{breaker_kind}",
            circuit_breaker_threshold=threshold,
            circuit_breaker_timeout=recovery_timeout,
            circuit_breaker_backoff_factor=2.0,
            circuit_breaker_max_timeout=4,
            circuit_breaker_factory=circuit_breaker_factory,
        )
    )


def _breaker_state_name(module: BaseModule) -> str:
    """Return a normalized state name from either breaker implementation."""

    breaker = module._circuit_breaker
    state = getattr(breaker, "state", None)
    if state is None:
        state = breaker._state
    return state if isinstance(state, str) else state.name.lower()


def _half_open_calls(module: BaseModule) -> int:
    """Return the active half-open probe count from either breaker implementation."""

    breaker = module._circuit_breaker
    calls = getattr(breaker, "half_open_calls", None)
    return calls if calls is not None else breaker._half_open_in_flight


def _current_recovery_timeout(module: BaseModule) -> float:
    """Return the active recovery timeout from either breaker implementation."""

    breaker = module._circuit_breaker
    timeout = getattr(breaker, "current_recovery_timeout", None)
    return timeout if timeout is not None else breaker._current_recovery_timeout


def _force_half_open(module: BaseModule) -> None:
    """Move either breaker implementation into a half-open-ready state."""

    breaker = module._circuit_breaker
    force_half_open = getattr(breaker, "force_half_open", None)
    if callable(force_half_open):
        force_half_open()
        return
    breaker._opened_at = time.time() - breaker._current_recovery_timeout - 1
    assert breaker.can_attempt() is True


async def _raise_runtime_error(message: str = "test failure") -> None:
    """Raise a runtime error for breaker-accounting tests."""

    raise RuntimeError(message)


@pytest.mark.parametrize("breaker_kind", ["fallback", "injected"])
@pytest.mark.asyncio
async def test_ignored_expected_failure_preserves_closed_breaker_counters(
    breaker_kind: str,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.execution_outcomes import (
        ExpectedToolFailure,
        ExpectedToolFailureReason,
    )

    module = _breaker_module(breaker_kind, threshold=3)
    with pytest.raises(RuntimeError):
        await module.execute_with_circuit_breaker(_raise_runtime_error)
    before = (
        module._circuit_breaker.failure_count,
        module._circuit_breaker.success_count,
    )
    failure = ExpectedToolFailure(ExpectedToolFailureReason.IDEMPOTENCY_IN_PROGRESS)

    async def fail_expected() -> None:
        raise failure

    with pytest.raises(ExpectedToolFailure) as exc_info:
        await module.execute_with_circuit_breaker(fail_expected)

    assert exc_info.value is failure
    assert _breaker_state_name(module) == "closed"
    assert (
        module._circuit_breaker.failure_count,
        module._circuit_breaker.success_count,
    ) == before


@pytest.mark.parametrize("breaker_kind", ["fallback", "injected"])
@pytest.mark.asyncio
async def test_ignored_expected_failure_is_neutral_in_half_open_and_releases_probe(
    breaker_kind: str,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.execution_outcomes import (
        ExpectedToolFailure,
        ExpectedToolFailureReason,
    )

    module = _breaker_module(breaker_kind, threshold=1)
    with pytest.raises(RuntimeError):
        await module.execute_with_circuit_breaker(_raise_runtime_error)
    _force_half_open(module)
    before = (
        module._circuit_breaker.failure_count,
        module._circuit_breaker.success_count,
        _current_recovery_timeout(module),
    )

    async def fail_expected() -> None:
        raise ExpectedToolFailure(ExpectedToolFailureReason.STALE_PREPARED_CALL)

    with pytest.raises(ExpectedToolFailure):
        await module.execute_with_circuit_breaker(fail_expected)

    assert _breaker_state_name(module) == "half_open"
    assert _half_open_calls(module) == 0
    assert (
        module._circuit_breaker.failure_count,
        module._circuit_breaker.success_count,
        _current_recovery_timeout(module),
    ) == before

    async def succeed() -> str:
        return "ok"

    assert await module.execute_with_circuit_breaker(succeed) == "ok"
    assert _breaker_state_name(module) == "closed"


@pytest.mark.parametrize("breaker_kind", ["fallback", "injected"])
@pytest.mark.asyncio
async def test_admission_timeout_error_is_neutral_in_half_open(
    breaker_kind: str,
) -> None:
    module = _breaker_module(breaker_kind, threshold=1)
    with pytest.raises(RuntimeError):
        await module.execute_with_circuit_breaker(_raise_runtime_error)
    _force_half_open(module)
    before = (
        module._circuit_breaker.failure_count,
        module._circuit_breaker.success_count,
        _current_recovery_timeout(module),
    )
    admission_error = TimeoutError("admission backend timeout")
    invoked = False

    async def _admit() -> None:
        raise admission_error

    async def _invoke() -> None:
        nonlocal invoked
        invoked = True

    with pytest.raises(TimeoutError) as exc_info:
        await module.execute_with_circuit_breaker(
            AdmittedModuleOperation(_admit, _invoke)
        )

    assert exc_info.value is admission_error
    assert invoked is False
    assert _breaker_state_name(module) == "half_open"
    assert _half_open_calls(module) == 0
    assert (
        module._circuit_breaker.failure_count,
        module._circuit_breaker.success_count,
        _current_recovery_timeout(module),
    ) == before


@pytest.mark.parametrize("breaker_kind", ["fallback", "injected"])
@pytest.mark.asyncio
async def test_counted_expected_failure_reopens_half_open_with_backoff(
    breaker_kind: str,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.execution_outcomes import (
        ExpectedToolFailure,
        ExpectedToolFailureReason,
    )

    module = _breaker_module(breaker_kind, threshold=1)
    with pytest.raises(RuntimeError):
        await module.execute_with_circuit_breaker(_raise_runtime_error)
    _force_half_open(module)
    before_timeout = _current_recovery_timeout(module)
    failure = ExpectedToolFailure(ExpectedToolFailureReason.DEPENDENCY_UNAVAILABLE)

    async def fail_expected() -> None:
        raise failure

    with pytest.raises(ExpectedToolFailure) as exc_info:
        await module.execute_with_circuit_breaker(fail_expected)

    assert exc_info.value is failure
    assert _breaker_state_name(module) == "open"
    assert _half_open_calls(module) == 0
    assert _current_recovery_timeout(module) == min(before_timeout * 2.0, 4.0)


@pytest.mark.parametrize("breaker_kind", ["fallback", "injected"])
@pytest.mark.parametrize(
    "reason_name",
    ["IDEMPOTENCY_IN_PROGRESS", "DEPENDENCY_UNAVAILABLE"],
)
@pytest.mark.asyncio
async def test_expected_failure_preserves_cause_traceback_and_visible_chain(
    breaker_kind: str,
    reason_name: str,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.execution_outcomes import (
        ExpectedToolFailure,
        ExpectedToolFailureReason,
    )

    module = _breaker_module(breaker_kind, threshold=3)
    cause = LookupError("explicit cause")
    reason = ExpectedToolFailureReason[reason_name]
    failure = ExpectedToolFailure(reason)
    failure.__cause__ = cause

    async def fail_with_explicit_cause() -> None:
        raise failure

    with pytest.raises(ExpectedToolFailure) as exc_info:
        await module.execute_with_circuit_breaker(fail_with_explicit_cause)

    caught = exc_info.value
    assert caught is failure
    assert type(caught) is ExpectedToolFailure
    assert caught.__cause__ is cause

    traceback_frames: list[str] = []
    current_traceback = caught.__traceback__
    while current_traceback is not None:
        traceback_frames.append(current_traceback.tb_frame.f_code.co_name)
        current_traceback = current_traceback.tb_next
    assert "fail_with_explicit_cause" in traceback_frames

    visible_chain: list[BaseException] = []
    current: BaseException | None = caught
    while current is not None:
        visible_chain.append(current)
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    assert visible_chain == [failure, cause]
    assert not {
        "_IgnoredModuleOutcome",
        "_CountedModuleOutcome",
    }.intersection(item.__class__.__name__ for item in visible_chain)


class _UnexpectedModuleFailure(Exception):
    """Distinct unexpected failure used to verify exception preservation."""


@pytest.mark.parametrize("breaker_kind", ["fallback", "injected"])
@pytest.mark.asyncio
async def test_unexpected_exception_counts_and_preserves_original_type(
    breaker_kind: str,
) -> None:
    module = _breaker_module(breaker_kind, threshold=3)
    failure = _UnexpectedModuleFailure("do not replace me")

    async def fail_unexpected() -> None:
        raise failure

    with pytest.raises(_UnexpectedModuleFailure) as exc_info:
        await module.execute_with_circuit_breaker(fail_unexpected)

    assert exc_info.value is failure
    assert _breaker_state_name(module) == "closed"
    assert module._circuit_breaker.failure_count == 1


@pytest.mark.parametrize("breaker_kind", ["fallback", "injected"])
@pytest.mark.asyncio
async def test_success_resets_closed_breaker_failure_count(breaker_kind: str) -> None:
    module = _breaker_module(breaker_kind, threshold=3)
    with pytest.raises(RuntimeError):
        await module.execute_with_circuit_breaker(_raise_runtime_error)
    assert module._circuit_breaker.failure_count == 1

    async def succeed() -> str:
        return "ok"

    assert await module.execute_with_circuit_breaker(succeed) == "ok"
    assert _breaker_state_name(module) == "closed"
    assert module._circuit_breaker.failure_count == 0


@pytest.mark.parametrize("breaker_kind", ["fallback", "injected"])
@pytest.mark.asyncio
async def test_cancellation_propagates_without_breaker_accounting(
    breaker_kind: str,
) -> None:
    module = _breaker_module(breaker_kind, threshold=1)

    async def cancel() -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await module.execute_with_circuit_breaker(cancel)

    assert _breaker_state_name(module) == "closed"
    assert module._circuit_breaker.failure_count == 0
    assert module.get_metrics().total_requests == 0
