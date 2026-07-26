import asyncio
import time
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import (
    BaseModule,
    ModuleCircuitBreakerOpenError,
    ModuleConfig,
)


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
    breaker = module._circuit_breaker
    state = getattr(breaker, "state", None)
    if state is None:
        state = breaker._state
    return state if isinstance(state, str) else state.name.lower()


def _half_open_calls(module: BaseModule) -> int:
    breaker = module._circuit_breaker
    calls = getattr(breaker, "half_open_calls", None)
    return calls if calls is not None else breaker._half_open_in_flight


def _current_recovery_timeout(module: BaseModule) -> float:
    breaker = module._circuit_breaker
    timeout = getattr(breaker, "current_recovery_timeout", None)
    return timeout if timeout is not None else breaker._current_recovery_timeout


def _force_half_open(module: BaseModule) -> None:
    breaker = module._circuit_breaker
    force_half_open = getattr(breaker, "force_half_open", None)
    if callable(force_half_open):
        force_half_open()
        return
    breaker._opened_at = time.time() - breaker._current_recovery_timeout - 1
    assert breaker.can_attempt() is True


async def _raise_runtime_error(message: str = "test failure") -> None:
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


class _UnexpectedModuleFailure(Exception):
    pass


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
