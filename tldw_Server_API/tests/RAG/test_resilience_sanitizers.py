"""Sanitizer coverage for RAG resilience fallback logs."""

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.RAG.rag_service import resilience
from tldw_Server_API.app.core.RAG.rag_service.resilience import (
    CircuitBreaker,
    ErrorContext,
    ErrorRecoveryCoordinator,
    FallbackChain,
    HealthMonitor,
    HealthStatus,
    RetryConfig,
    RetryPolicy,
)


def _capture_logs(level: str = "DEBUG") -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = resilience.logger.add(lambda message: messages.append(str(message)), level=level)
    return messages, sink_id


def _assert_not_leaked(messages: list[str], *secrets: str) -> None:
    joined = "\n".join(messages)
    for secret in secrets:
        assert secret not in joined


@pytest.mark.parametrize(
    ("state", "callbacks", "expected_message"),
    [
        (resilience._UnifiedState.OPEN, "on_open_callbacks", "Error in open callback"),
        (resilience._UnifiedState.CLOSED, "on_close_callbacks", "Error in close callback"),
        (
            resilience._UnifiedState.HALF_OPEN,
            "on_half_open_callbacks",
            "Error in half-open callback",
        ),
    ],
)
def test_circuit_breaker_callback_logs_do_not_expose_exception_details(
    state,
    callbacks: str,
    expected_message: str,
):
    breaker = CircuitBreaker("vector_store")
    secret = "/private/rag-callback.db?token=secret-callback-token"

    def broken_callback(_breaker):
        raise RuntimeError(f"callback failed at {secret}")

    getattr(breaker, callbacks).append(broken_callback)

    messages, sink_id = _capture_logs(level="ERROR")
    try:
        breaker._dispatch_callback(None, None, state)
    finally:
        resilience.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert expected_message in joined
    _assert_not_leaked(messages, "rag-callback.db", "secret-callback-token")


@pytest.mark.asyncio
async def test_retry_policy_logs_do_not_expose_exception_details():
    policy = RetryPolicy(
        RetryConfig(max_attempts=2, initial_delay=0, jitter=False, retry_on=[RuntimeError])
    )
    secret = "/private/rag-retry.db?token=secret-retry-token"

    async def broken_operation():
        raise RuntimeError(f"retry backend failed at {secret}")

    messages, sink_id = _capture_logs(level="WARNING")
    try:
        with pytest.raises(RuntimeError):
            await policy.execute(broken_operation)
    finally:
        resilience.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Attempt 1 failed" in joined
    assert "Retry failed after 2 attempts" in joined
    _assert_not_leaked(messages, "rag-retry.db", "secret-retry-token")


@pytest.mark.asyncio
async def test_fallback_chain_logs_do_not_expose_exception_details():
    chain = FallbackChain()
    primary_secret = "/private/rag-primary.db?token=secret-primary-token"
    fallback_secret = "/private/rag-fallback.db?token=secret-fallback-token"

    async def primary():
        raise RuntimeError(f"primary failed at {primary_secret}")

    async def fallback():
        raise RuntimeError(f"fallback failed at {fallback_secret}")

    chain.add_strategy(fallback)

    messages, sink_id = _capture_logs(level="WARNING")
    try:
        with pytest.raises(RuntimeError):
            await chain.execute(primary)
    finally:
        resilience.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Primary function failed" in joined
    assert "Fallback failed" in joined
    assert "All fallback strategies failed" in joined
    _assert_not_leaked(
        messages,
        "rag-primary.db",
        "secret-primary-token",
        "rag-fallback.db",
        "secret-fallback-token",
    )


@pytest.mark.asyncio
async def test_health_monitoring_loop_logs_do_not_expose_exception_details(monkeypatch):
    monitor = HealthMonitor()
    secret = "/private/rag-monitor.db?token=secret-monitor-token"

    async def broken_check_all_health():
        raise RuntimeError(f"monitor loop failed at {secret}")

    async def stop_after_log(_delay):
        raise asyncio_cancelled_error()

    def asyncio_cancelled_error():
        return resilience.asyncio.CancelledError()

    monkeypatch.setattr(monitor, "check_all_health", broken_check_all_health)
    monkeypatch.setattr(resilience.asyncio, "sleep", stop_after_log)

    messages, sink_id = _capture_logs(level="ERROR")
    try:
        with pytest.raises(resilience.asyncio.CancelledError):
            await monitor._monitoring_loop()
    finally:
        resilience.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Error in health monitoring" in joined
    _assert_not_leaked(messages, "rag-monitor.db", "secret-monitor-token")


@pytest.mark.asyncio
async def test_component_health_check_logs_do_not_expose_exception_details():
    monitor = HealthMonitor()
    secret = "/private/rag-component.db?token=secret-component-token"

    def broken_health_check():
        raise RuntimeError(f"component failed at {secret}")

    monitor.register_component("vector_store", broken_health_check)

    messages, sink_id = _capture_logs(level="ERROR")
    try:
        result = await monitor.check_all_health()
    finally:
        resilience.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Health check failed for 'vector_store'" in joined
    _assert_not_leaked(messages, "rag-component.db", "secret-component-token")
    assert result["vector_store"] == HealthStatus.UNKNOWN


@pytest.mark.asyncio
async def test_pipeline_health_check_logs_do_not_expose_exception_details():
    context = SimpleNamespace(metadata={})
    secret = "/private/rag-pipeline-health.db?token=secret-pipeline-health-token"

    def broken_health_check():
        raise RuntimeError(f"pipeline health failed at {secret}")

    messages, sink_id = _capture_logs(level="ERROR")
    try:
        result = await resilience.check_component_health(
            context,
            "retriever",
            broken_health_check,
            critical=False,
        )
    finally:
        resilience.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Health check failed for 'retriever'" in joined
    _assert_not_leaked(messages, "rag-pipeline-health.db", "secret-pipeline-health-token")
    assert result.metadata["health_retriever"] == "unknown"


def test_recovery_stats_recent_errors_do_not_expose_exception_details() -> None:
    coordinator = ErrorRecoveryCoordinator()
    coordinator.record_error(
        ErrorContext(
            component="retriever",
            operation="search",
            timestamp=123.0,
            attempt=1,
            error=RuntimeError(
                "retriever failed at /private/rag-recovery.db?token=secret-recovery-token"
            ),
        )
    )

    stats = coordinator.get_recovery_stats()

    assert stats["recent_errors"] == [
        {
            "component": "retriever",
            "operation": "search",
            "timestamp": stats["recent_errors"][0]["timestamp"],
            "error": "Error details unavailable",
        }
    ]
    assert "rag-recovery.db" not in repr(stats)
    assert "secret-recovery-token" not in repr(stats)
