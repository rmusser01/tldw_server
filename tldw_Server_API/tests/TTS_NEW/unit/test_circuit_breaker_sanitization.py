import pytest

from tldw_Server_API.app.core.TTS import circuit_breaker


@pytest.mark.asyncio
@pytest.mark.unit
async def test_perform_health_check_failure_log_sanitizes_exception_text():
    breaker = circuit_breaker.CircuitBreaker(provider_name="qwen3")
    secret_detail = "/Users/example/private/token-sk-tts-health.json"
    logged_messages: list[str] = []

    async def fail_health_check():
        raise RuntimeError(f"health check failed with {secret_detail}")

    sink_id = circuit_breaker.logger.add(
        lambda message: logged_messages.append(message.record["message"]),
        level="ERROR",
    )
    try:
        await breaker._perform_health_check(fail_health_check)
    finally:
        circuit_breaker.logger.remove(sink_id)

    assert breaker._last_health_check is not None
    assert any(
        "Health check failed for qwen3" in message
        for message in logged_messages
    )
    assert all(secret_detail not in message for message in logged_messages)
    assert all("health check failed with" not in message for message in logged_messages)
    assert all("RuntimeError" in message for message in logged_messages)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_health_monitor_loop_failure_log_sanitizes_exception_text(monkeypatch):
    breaker = circuit_breaker.CircuitBreaker(
        provider_name="kokoro",
        failure_threshold=1,
    )
    await breaker.record_manual_failure(RuntimeError("initial failure"))
    assert breaker.state == circuit_breaker.CircuitState.OPEN

    secret_detail = "/Users/example/private/monitor-token-sk-test"
    logged_messages: list[str] = []
    sleep_calls = 0

    async def fake_sleep(_interval):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise circuit_breaker.asyncio.CancelledError

    async def fail_perform_health_check(_health_check_func):
        raise RuntimeError(f"monitor fallback leaked {secret_detail}")

    monkeypatch.setattr(circuit_breaker.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        breaker,
        "_perform_health_check",
        fail_perform_health_check,
    )

    sink_id = circuit_breaker.logger.add(
        lambda message: logged_messages.append(message.record["message"]),
        level="ERROR",
    )
    try:
        await breaker._health_monitor_loop(health_check_func=None)
    finally:
        circuit_breaker.logger.remove(sink_id)

    assert breaker.state == circuit_breaker.CircuitState.OPEN
    assert any(
        "Error in health monitoring for kokoro" in message
        for message in logged_messages
    )
    assert all(secret_detail not in message for message in logged_messages)
    assert all("monitor fallback leaked" not in message for message in logged_messages)
    assert all("RuntimeError" in message for message in logged_messages)
