"""
Unit tests for CircuitBreaker.

Tests fault tolerance and circuit breaker pattern implementation.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

from tldw_Server_API.app.core.Evaluations.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState, CircuitOpenError


class _FakeClock:
    """Deterministic stand-in for ``time.time()`` used to fast-forward the
    circuit breaker's recovery-timeout checks without real wall-clock waits.
    """

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def fake_clock(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr("time.time", clock.time)
    return clock


@pytest.mark.unit
class TestCircuitBreakerInit:
    """Test CircuitBreaker initialization."""

    def test_init_with_defaults(self):

        """Test initialization with default values."""
        cb = CircuitBreaker("test")

        assert cb.config.failure_threshold == 5
        assert cb.config.recovery_timeout == 60
        assert cb.config.expected_exception == Exception
        assert cb.state == CircuitState.CLOSED
        assert cb.stats.consecutive_failures == 0
        assert cb.stats.last_failure_time is None

    def test_init_with_custom_values(self):

        """Test initialization with custom values."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30,
            expected_exception=ValueError
        )
        cb = CircuitBreaker("test", config)

        assert cb.config.failure_threshold == 3
        assert cb.config.recovery_timeout == 30
        assert cb.config.expected_exception == ValueError

    def test_init_with_multiple_exceptions(self):

        """Test initialization with multiple exception types."""
        config = CircuitBreakerConfig(
            expected_exception=(ValueError, TypeError, KeyError)
        )
        cb = CircuitBreaker("test", config)

        assert cb.config.expected_exception == (ValueError, TypeError, KeyError)


@pytest.mark.unit
class TestCircuitBreakerStates:
    """Test circuit breaker state transitions."""

    def test_initial_state_is_closed(self):

        """Test that circuit starts in closed state."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)
        assert cb.state == CircuitState.CLOSED
        # Note: is_closed, is_open, is_half_open properties don't exist in new implementation

    def test_transition_to_open_on_threshold(self):

        """Test transition to open state when failure threshold reached."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)

        # Simulate failures by calling _on_failure directly
        import asyncio
        for i in range(3):
            asyncio.run(cb._on_failure())

        assert cb.state == CircuitState.OPEN
        assert cb.stats.consecutive_failures == 0  # Reset after transition
        assert cb.stats.last_failure_time is not None

    def test_transition_to_half_open_after_timeout(self, fake_clock):

        """Test transition to half-open state after recovery timeout."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1)
        cb = CircuitBreaker("test", config)

        # Open the circuit
        import asyncio
        asyncio.run(cb._on_failure())
        asyncio.run(cb._on_failure())
        assert cb.state == CircuitState.OPEN

        # Fast-forward past the recovery timeout
        fake_clock.advance(1.1)

        # Trigger transition by checking if should attempt reset
        if cb._should_attempt_reset():
            cb._transition_to_half_open()

        assert cb.state == CircuitState.HALF_OPEN

    def test_transition_from_half_open_to_closed_on_success(self):

        """Test transition from half-open to closed on successful call."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config)

        # Set to half-open state manually
        cb.state = CircuitState.HALF_OPEN

        # Record success - need enough successes to close
        import asyncio
        asyncio.run(cb._on_success())
        asyncio.run(cb._on_success())  # Default success_threshold is 2

        assert cb.state == CircuitState.CLOSED
        assert cb.stats.consecutive_failures == 0

    def test_transition_from_half_open_to_open_on_failure(self):

        """Test transition from half-open back to open on failure."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config)

        # Set to half-open state manually
        cb.state = CircuitState.HALF_OPEN

        # Record failure
        import asyncio
        asyncio.run(cb._on_failure())

        assert cb.state == CircuitState.OPEN
        assert cb.stats.consecutive_failures == 0  # Reset after transition


@pytest.mark.unit
class TestCircuitBreakerDecorator:
    """Test circuit breaker as decorator."""

    @pytest.mark.asyncio
    async def test_decorator_on_successful_function(self):
        """Test decorator with function that succeeds."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)

        def successful_function(x):

            return x * 2

        result = await cb.call(successful_function, 5)
        assert result == 10
        assert cb.stats.consecutive_failures == 0
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_decorator_on_failing_function(self):
        """Test decorator with function that fails."""
        config = CircuitBreakerConfig(failure_threshold=3, expected_exception=ValueError)
        cb = CircuitBreaker("test", config)

        def failing_function():

            raise ValueError("Test error")

        # First failures should be allowed
        for i in range(3):
            with pytest.raises(ValueError):
                await cb.call(failing_function)

        assert cb.state == CircuitState.OPEN

        # Next call should raise circuit open exception
        with pytest.raises(CircuitOpenError) as exc_info:
            await cb.call(failing_function)
        assert "Circuit breaker test is OPEN" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_decorator_on_async_function(self):
        """Test decorator with async function."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config)

        async def async_function(x):
            await asyncio.sleep(0.01)
            return x + 1

        result = await cb.call(async_function, 5)
        assert result == 6
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_decorator_on_failing_async_function(self):
        """Test decorator with failing async function."""
        config = CircuitBreakerConfig(failure_threshold=2, expected_exception=TypeError)
        cb = CircuitBreaker("test", config)

        async def failing_async_function():
            await asyncio.sleep(0.01)
            raise TypeError("Async error")

        # Trigger failures
        for i in range(2):
            with pytest.raises(TypeError):
                await cb.call(failing_async_function)

        assert cb.state == CircuitState.OPEN

        # Circuit should be open
        with pytest.raises(CircuitOpenError) as exc_info:
            await cb.call(failing_async_function)
        assert "Circuit breaker test is OPEN" in str(exc_info.value)


@pytest.mark.unit
class TestCircuitBreakerCallMethod:
    """Test circuit breaker call method."""

    @pytest.mark.asyncio
    async def test_call_successful_function(self):
        """Test calling successful function through circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)

        def test_func(a, b):

            return a + b

        result = await cb.call(test_func, 3, 5)
        assert result == 8
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_call_with_expected_exception(self):
        """Test calling function that raises expected exception."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            expected_exception=ValueError
        )
        cb = CircuitBreaker("test", config)

        def test_func():

            raise ValueError("Expected error")

        # Should count as failure
        with pytest.raises(ValueError):
            await cb.call(test_func)

        assert cb.stats.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_call_with_unexpected_exception(self):
        """Test calling function that raises unexpected exception."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            expected_exception=ValueError
        )
        cb = CircuitBreaker("test", config)

        def test_func():

            raise TypeError("Unexpected error")

        # Should not count as circuit breaker failure
        with pytest.raises(TypeError):
            await cb.call(test_func)

        assert cb.stats.consecutive_failures == 0
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_call_when_circuit_open(self):
        """Test calling function when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1, expected_exception=ValueError)
        cb = CircuitBreaker("test", config)

        def test_func():

            raise ValueError("Error")

        # Open the circuit
        with pytest.raises(ValueError):
            await cb.call(test_func)

        assert cb.state == CircuitState.OPEN

        # Should reject without calling function
        with pytest.raises(CircuitOpenError) as exc_info:
            await cb.call(test_func)

        assert "Circuit breaker test is OPEN" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_open_error_preserves_context_fields(self):
        """CircuitOpenError should include structured context from unified breaker."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=12.0,
            expected_exception=ValueError,
        )
        cb = CircuitBreaker("ctx", config)

        def failing_func():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            await cb.call(failing_func)

        with pytest.raises(CircuitOpenError) as exc_info:
            await cb.call(failing_func)

        err = exc_info.value
        assert err.breaker_name == "ctx"
        assert err.category == "evaluations"
        assert err.service  # populated by unified breaker metadata
        assert err.recovery_timeout == 12.0
        assert err.failure_count >= 1
        assert err.recovery_at is not None

    @pytest.mark.asyncio
    async def test_open_rejection_for_sync_func_has_no_unawaited_coroutine_warning(self, recwarn):
        """Rejecting sync functions while OPEN should not leak unawaited coroutines."""
        import gc

        config = CircuitBreakerConfig(failure_threshold=1, expected_exception=ValueError)
        cb = CircuitBreaker("no_leak", config)

        def failing_func():
            raise ValueError("boom")

        def sync_should_not_run():
            return 123

        with pytest.raises(ValueError):
            await cb.call(failing_func)

        with pytest.raises(CircuitOpenError):
            await cb.call(sync_should_not_run)

        gc.collect()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        leaked = [
            warning
            for warning in recwarn
            if issubclass(warning.category, RuntimeWarning)
            and "was never awaited" in str(warning.message)
        ]
        assert leaked == []


@pytest.mark.unit
class TestCircuitBreakerRecovery:
    """Test circuit breaker recovery mechanisms."""

    @pytest.mark.asyncio
    async def test_automatic_recovery_after_timeout(self, fake_clock):
        """Test automatic recovery after timeout period."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.5,  # 0.5 seconds for testing
            expected_exception=ValueError
        )
        cb = CircuitBreaker("test", config)

        def failing_func():

            raise ValueError("Error")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

        # Fast-forward past the recovery timeout
        fake_clock.advance(0.6)

        # Successful call should trigger transition to half-open then closed
        def success_func():
            return "success"

        result = await cb.call(success_func)
        assert result == "success"
        # After 2 successes (default success_threshold), should be closed
        result2 = await cb.call(success_func)
        assert result2 == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.stats.consecutive_failures == 0

    def test_manual_reset(self):

        """Test manual reset of circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config)

        # Open the circuit
        import asyncio
        asyncio.run(cb._on_failure())
        asyncio.run(cb._on_failure())
        assert cb.state == CircuitState.OPEN

        # Manual reset
        cb.reset()

        assert cb.state == CircuitState.CLOSED
        assert cb.stats.consecutive_failures == 0
        assert cb.stats.last_failure_time is None

    @pytest.mark.asyncio
    async def test_half_open_single_test(self, fake_clock):
        """Test that half-open state allows single test call."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1)
        cb = CircuitBreaker("test", config)

        # Open circuit by recording failures
        await cb._on_failure()
        await cb._on_failure()
        assert cb.state == CircuitState.OPEN

        # Fast-forward past the recovery timeout
        fake_clock.advance(0.15)

        call_count = 0

        def test_func():

            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Allow 2 calls for success threshold
                return "success"
            raise ValueError("Should not be called again")

        # First call should transition to half-open then closed
        result = await cb.call(test_func)
        assert result == "success"
        # Need another success to close (default success_threshold=2)
        result2 = await cb.call(test_func)
        assert result2 == "success"
        assert cb.state == CircuitState.CLOSED


@pytest.mark.unit
class TestCircuitBreakerMetrics:
    """Test circuit breaker metrics and monitoring."""

    def test_failure_count_tracking(self):

        """Test accurate failure count tracking."""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker("test", config)

        # Track failures
        import asyncio
        for i in range(3):
            asyncio.run(cb._on_failure())
            assert cb.stats.consecutive_failures == i + 1

        # Reset should clear count
        cb.reset()
        assert cb.stats.consecutive_failures == 0

    def test_last_failure_time_tracking(self):

        """Test tracking of last failure timestamp."""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker("test", config)

        assert cb.stats.last_failure_time is None

        import asyncio
        asyncio.run(cb._on_failure())
        first_failure = cb.stats.last_failure_time
        assert first_failure is not None

        time.sleep(0.1)
        asyncio.run(cb._on_failure())
        second_failure = cb.stats.last_failure_time

        assert second_failure > first_failure

    def test_success_rate_calculation(self):

        """Test calculation of success rate."""
        config = CircuitBreakerConfig(failure_threshold=10)
        cb = CircuitBreaker("test", config)

        # Record mixed results
        import asyncio
        for _ in range(7):
            asyncio.run(cb._on_success())
        for _ in range(3):
            asyncio.run(cb._on_failure())

        state = cb.get_state()
        stats = state["stats"]
        assert stats["total_calls"] == 0  # total_calls is tracked on actual calls, not _on_success/_on_failure
        assert stats["successful_calls"] == 7
        assert stats["failed_calls"] == 3
        # Success rate calculation when total_calls is 0
        assert stats["success_rate"] == 0

    def test_circuit_breaker_statistics(self):

        """Test comprehensive statistics gathering."""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60)
        cb = CircuitBreaker("test", config)

        # Generate some activity
        import asyncio
        asyncio.run(cb._on_success())
        asyncio.run(cb._on_success())
        asyncio.run(cb._on_failure())
        asyncio.run(cb._on_failure())
        asyncio.run(cb._on_failure())  # Opens circuit

        state = cb.get_state()
        stats = state["stats"]
        config_info = state["config"]

        assert state["state"] == "open"
        assert config_info["failure_threshold"] == 3
        assert config_info["recovery_timeout"] == 60
        assert stats["total_calls"] == 0  # total_calls tracks actual call() invocations
        assert stats["successful_calls"] == 2
        assert stats["failed_calls"] == 3


@pytest.mark.unit
class TestCircuitBreakerConcurrency:
    """Test circuit breaker with concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_calls_respect_state(self):
        """Test that concurrent calls respect circuit state."""
        config = CircuitBreakerConfig(failure_threshold=3, expected_exception=ValueError)
        cb = CircuitBreaker("test", config)

        async def failing_task():
            raise ValueError("Concurrent failure")

        # Open the circuit with concurrent failures
        tasks = []
        for _ in range(5):
            tasks.append(cb.call(failing_task))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should be ValueError or circuit open; circuit should end open
        value_errors = sum(1 for r in results if isinstance(r, ValueError))
        circuit_errors = sum(1 for r in results if isinstance(r, CircuitOpenError))

        assert value_errors + circuit_errors == 5
        assert value_errors >= 3  # At least threshold failures needed to trip
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_thread_safe_state_transitions(self):
        """Test thread-safe state transitions."""
        config = CircuitBreakerConfig(failure_threshold=10)
        cb = CircuitBreaker("test", config)

        # Test concurrent failures using asyncio.gather instead of threads
        # This avoids event loop conflicts while still testing concurrency
        async def concurrent_failure():
            await cb._on_failure()

        # Run multiple failures concurrently
        tasks = [concurrent_failure() for _ in range(20)]
        await asyncio.gather(*tasks)

        # Should handle concurrency and transition to open
        assert cb.state == CircuitState.OPEN


@pytest.mark.unit
class TestCircuitBreakerErrorHandling:
    """Test error handling in circuit breaker."""

    @pytest.mark.asyncio
    async def test_handle_none_function(self):
        """Test handling of None function."""
        cb = CircuitBreaker("test")

        with pytest.raises(TypeError):
            await cb.call(None)

    @pytest.mark.asyncio
    async def test_handle_non_callable(self):
        """Test handling of non-callable object."""
        cb = CircuitBreaker("test")

        with pytest.raises(TypeError):
            await cb.call("not a function")

    @pytest.mark.asyncio
    async def test_preserve_original_exception(self):
        """Test that original exceptions are preserved."""
        config = CircuitBreakerConfig(expected_exception=ValueError)
        cb = CircuitBreaker("test", config)

        def failing_func():

            raise ValueError("Original error message")

        with pytest.raises(ValueError) as exc_info:
            await cb.call(failing_func)

        assert "Original error message" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_custom_open_circuit_exception(self):
        """Test that CircuitOpenError is raised when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1, expected_exception=ValueError)
        cb = CircuitBreaker("test", config)

        # Open circuit by calling _on_failure
        await cb._on_failure()

        def test_func():

            return "should not execute"

        with pytest.raises(CircuitOpenError):
            await cb.call(test_func)
