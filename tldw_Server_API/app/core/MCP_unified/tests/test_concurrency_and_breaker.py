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
