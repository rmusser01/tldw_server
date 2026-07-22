"""Shared-capacity regressions for response-quality provider calls."""

from __future__ import annotations

import asyncio
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from tldw_Server_API.app.core.Evaluations import response_quality_evaluator
from tldw_Server_API.app.core.Evaluations.response_quality_evaluator import (
    ResponseQualityEvaluator,
)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_all_response_quality_metrics_use_one_shared_sync_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every sync metric reaches the breaker through the shared adapter pool."""
    breaker_functions: list[Any] = []
    bounded_calls: list[dict[str, Any]] = []

    def analyze_response(
        _api_name: str,
        _input_data: Any,
        custom_prompt: str | None = None,
        *_args: Any,
        **_kwargs: Any,
    ) -> str:
        if custom_prompt and "COMPLIANT:" in custom_prompt:
            return "COMPLIANT: yes\nISSUES: none"
        return "4"

    async def call_inline(
        _provider: str,
        function,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        breaker_functions.append(function)
        result = function(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def run_inline(
        call,
        *,
        pool,
        exhaustion_message: str,
        on_cancel_result=None,
    ) -> Any:
        bounded_calls.append(
            {
                "pool": pool,
                "exhaustion_message": exhaustion_message,
                "on_cancel_result": on_cancel_result,
            }
        )
        return call()

    monkeypatch.setattr(response_quality_evaluator, "analyze", analyze_response)
    monkeypatch.setattr(
        response_quality_evaluator.llm_circuit_breaker,
        "call_with_breaker",
        call_inline,
    )
    monkeypatch.setattr(
        response_quality_evaluator,
        "await_bounded_sync_call",
        run_inline,
        raising=False,
    )

    result = await ResponseQualityEvaluator().evaluate(
        prompt="prompt",
        response="response",
        expected_format="markdown",
        custom_criteria={"specificity": "Be specific"},
        api_name="openai",
        api_key="resolved-key",
        model="model-a",
        app_config={"openai_api": {"model": "model-a"}},
        credentials_resolved=True,
    )

    assert set(result["metrics"]) == {
        "relevance",
        "completeness",
        "clarity",
        "accuracy",
        "custom_specificity",
    }
    assert result["format_compliance"] is True
    assert len(breaker_functions) == 6
    assert all(inspect.iscoroutinefunction(function) for function in breaker_functions)
    assert len(bounded_calls) == 6
    assert all(
        call["pool"] is response_quality_evaluator.SYNC_ADAPTER_CALL_POOL
        for call in bounded_calls
    )
    assert {
        call["exhaustion_message"] for call in bounded_calls
    } == {"Response quality provider capacity is exhausted"}
    assert all(call["on_cancel_result"] is None for call in bounded_calls)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_response_quality_public_call_bypasses_saturated_default_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public quality evaluation starts without default-executor queueing."""
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.Evaluations.circuit_breaker import LLMCircuitBreaker

    loop = asyncio.get_running_loop()
    previous_executor = getattr(loop, "_default_executor", None)
    default_executor = ThreadPoolExecutor(max_workers=1)
    default_entered = threading.Event()
    default_release = threading.Event()
    provider_entered = threading.Event()
    pool = BoundedDaemonPool(1)

    def block_default_executor() -> None:
        default_entered.set()
        default_release.wait(timeout=2.0)

    def analyze_score(*_args: Any, **_kwargs: Any) -> str:
        provider_entered.set()
        return "4"

    monkeypatch.setattr(
        response_quality_evaluator,
        "llm_circuit_breaker",
        LLMCircuitBreaker(),
    )
    monkeypatch.setattr(response_quality_evaluator, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    monkeypatch.setattr(response_quality_evaluator, "analyze", analyze_score)
    loop.set_default_executor(default_executor)
    blocker = loop.run_in_executor(None, block_default_executor)
    task: asyncio.Task[tuple] | None = None
    try:
        for _ in range(1000):
            if default_entered.is_set():
                break
            await asyncio.sleep(0.001)
        assert default_entered.is_set()

        task = asyncio.create_task(
            ResponseQualityEvaluator()._evaluate_relevance(
                "prompt",
                "response",
                "openai",
                "resolved-key",
            )
        )
        for _ in range(100):
            if provider_entered.is_set():
                break
            await asyncio.sleep(0.001)
        started_before_default_release = provider_entered.is_set()
    finally:
        default_release.set()
        await asyncio.gather(blocker, return_exceptions=True)
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        loop.set_default_executor(previous_executor or ThreadPoolExecutor())
        default_executor.shutdown(wait=True, cancel_futures=True)

    assert started_before_default_release is True
    assert task is not None
    metric, result = task.result()
    assert metric == "relevance"
    assert result["score"] == pytest.approx(0.8)
    assert pool.active_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_response_quality_breaker_timeout_drains_before_safe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeout returns a fixed failure only after the real provider call exits."""
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.Evaluations.circuit_breaker import (
        CircuitBreakerConfig,
        LLMCircuitBreaker,
    )

    entered = threading.Event()
    release = threading.Event()
    lifecycle: list[str] = []

    class TrackingPool(BoundedDaemonPool):
        def _release_capacity(self) -> None:
            lifecycle.append("capacity-release")
            super()._release_capacity()

    def blocking_analyze(*_args: Any, **_kwargs: Any) -> str:
        lifecycle.append("provider-start")
        entered.set()
        release.wait(timeout=2.0)
        lifecycle.append("provider-exit")
        return "private provider response"

    breaker = LLMCircuitBreaker()
    breaker.provider_configs["openai"] = CircuitBreakerConfig(timeout=0.01)
    pool = TrackingPool(1)
    monkeypatch.setattr(response_quality_evaluator, "llm_circuit_breaker", breaker)
    monkeypatch.setattr(response_quality_evaluator, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    monkeypatch.setattr(response_quality_evaluator, "analyze", blocking_analyze)

    task = asyncio.create_task(
        ResponseQualityEvaluator()._evaluate_relevance(
            "prompt",
            "response",
            "openai",
            "resolved-key",
        )
    )
    try:
        for _ in range(1000):
            if entered.is_set():
                break
            await asyncio.sleep(0.001)
        assert entered.is_set()
        await asyncio.sleep(0.03)

        assert task.done() is False
        assert pool.active_count == 1
        assert lifecycle == ["provider-start"]

        release.set()
        metric, result = await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    provider_breaker = breaker.get_breaker("openai")
    assert metric == "relevance"
    assert result == {
        "name": "relevance",
        "score": 0.0,
        "explanation": "Provider evaluation failed.",
    }
    assert "private provider response" not in repr(result)
    assert provider_breaker.stats.timeouts == 1
    assert provider_breaker.stats.failed_calls == 1
    assert pool.active_count == 0
    assert lifecycle == [
        "provider-start",
        "provider-exit",
        "capacity-release",
    ]
