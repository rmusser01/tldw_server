# test_circuit_breaker.py
"""Tests for the unified circuit breaker module."""

import asyncio
import importlib
import threading
import time
from typing import Callable
from unittest.mock import patch

import pytest

from tldw_Server_API.app.core.DB_Management.Circuit_Breaker_Registry_DB import (
    CircuitBreakerOptimisticLockError,
    CircuitBreakerRegistryDB,
    CircuitBreakerStoredState,
)
from tldw_Server_API.app.core.Infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerRegistry,
    CircuitState,
    circuit_breaker,
    registry,
)

_infra_cb_module = importlib.import_module(
    "tldw_Server_API.app.core.Infrastructure.circuit_breaker"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TransientError(Exception):
    """Test exception that should trip the breaker."""


class PermanentError(Exception):
    """Test exception that should NOT trip the breaker."""


def _succeed() -> str:
    return "ok"


def _fail_transient() -> str:
    raise TransientError("boom")


def _fail_permanent() -> str:
    raise PermanentError("fatal")


async def _async_succeed() -> str:
    return "ok"


async def _async_fail() -> str:
    raise TransientError("boom")


@pytest.fixture(autouse=True)
def _isolate_global_registry():
    """Prevent cross-test pollution in the global decorator registry."""
    registry.clear()
    yield
    registry.clear()


# ---------------------------------------------------------------------------
# TestConfig
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        cfg = CircuitBreakerConfig()
        assert cfg.failure_threshold == 5
        assert cfg.success_threshold == 2
        assert cfg.recovery_timeout == 60.0
        assert cfg.half_open_max_calls == 1
        assert cfg.expected_exception is Exception
        assert cfg.window_size == 0
        assert cfg.failure_rate_threshold == 0.5
        assert cfg.backoff_factor == 1.0
        assert cfg.max_recovery_timeout == 300.0

    def test_custom_values(self):
        cfg = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=1,
            recovery_timeout=30.0,
            half_open_max_calls=2,
            expected_exception=TransientError,
            window_size=10,
            failure_rate_threshold=0.3,
            backoff_factor=2.0,
            max_recovery_timeout=600.0,
            category="chat",
            service="svc",
            operation="op",
        )
        assert cfg.failure_threshold == 3
        assert cfg.window_size == 10
        assert cfg.backoff_factor == 2.0


# ---------------------------------------------------------------------------
# TestOpenError
# ---------------------------------------------------------------------------

class TestOpenError:
    def test_attributes(self):
        err = CircuitBreakerOpenError(
            "test msg",
            breaker_name="b1",
            category="cat",
            service="svc",
            recovery_timeout=30.0,
            failure_count=5,
        )
        assert str(err) == "test msg"
        assert err.breaker_name == "b1"
        assert err.category == "cat"
        assert err.service == "svc"
        assert err.recovery_timeout == 30.0
        assert err.failure_count == 5

    def test_default_attributes(self):
        err = CircuitBreakerOpenError("test")
        assert err.breaker_name == ""
        assert err.recovery_timeout == 0.0


# ---------------------------------------------------------------------------
# TestStateTransitions (sync)
# ---------------------------------------------------------------------------

class TestStateTransitionsSync:
    def test_starts_closed(self):
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=2))
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed

    def test_opens_after_threshold(self):
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=2))
        for _ in range(2):
            with pytest.raises(TransientError):
                cb.call(_fail_transient)
        assert cb.is_open

    def test_rejects_when_open(self):
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=1))
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert cb.is_open
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(_succeed)

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.05),
        )
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert cb.is_open
        time.sleep(0.06)
        assert cb.is_half_open

    def test_closes_after_success_threshold(self):
        cb = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout=0.05,
                success_threshold=2,
            ),
        )
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        time.sleep(0.06)
        # Two successes needed
        cb.call(_succeed)
        assert cb.is_half_open  # still half-open after 1
        cb.call(_succeed)
        assert cb.is_closed

    def test_reopens_on_half_open_failure(self):
        cb = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout=0.05,
                success_threshold=2,
            ),
        )
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        time.sleep(0.06)
        assert cb.is_half_open
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert cb.is_open

    def test_success_resets_failure_count_in_closed(self):
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=3))
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        cb.call(_succeed)
        # failure count reset, so 2 more failures should NOT open
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert cb.is_closed
        # 3rd failure (from 0) opens
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert cb.is_open


# ---------------------------------------------------------------------------
# TestStateTransitions (async)
# ---------------------------------------------------------------------------

class TestStateTransitionsAsync:
    @pytest.mark.asyncio
    async def test_async_success(self):
        cb = CircuitBreaker("test-async")
        result = await cb.call_async(_async_succeed)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_async_opens_after_threshold(self):
        cb = CircuitBreaker("test-async", config=CircuitBreakerConfig(failure_threshold=2))
        for _ in range(2):
            with pytest.raises(TransientError):
                await cb.call_async(_async_fail)
        assert cb.is_open

    @pytest.mark.asyncio
    async def test_async_rejects_when_open(self):
        cb = CircuitBreaker("test-async", config=CircuitBreakerConfig(failure_threshold=1))
        with pytest.raises(TransientError):
            await cb.call_async(_async_fail)
        with pytest.raises(CircuitBreakerOpenError):
            await cb.call_async(_async_succeed)

    @pytest.mark.asyncio
    async def test_async_half_open_recovery(self):
        cb = CircuitBreaker(
            "test-async",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout=0.05,
                success_threshold=1,
            ),
        )
        with pytest.raises(TransientError):
            await cb.call_async(_async_fail)
        await asyncio.sleep(0.06)
        result = await cb.call_async(_async_succeed)
        assert result == "ok"
        assert cb.is_closed


# ---------------------------------------------------------------------------
# TestCounterMode
# ---------------------------------------------------------------------------

class TestCounterMode:
    def test_counter_mode_default(self):
        cb = CircuitBreaker(
            "counter",
            config=CircuitBreakerConfig(failure_threshold=3, window_size=0),
        )
        assert cb._window is None
        for _ in range(2):
            with pytest.raises(TransientError):
                cb.call(_fail_transient)
        assert cb.is_closed
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert cb.is_open

    def test_counter_mode_success_resets(self):
        cb = CircuitBreaker(
            "counter",
            config=CircuitBreakerConfig(failure_threshold=3, window_size=0),
        )
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        cb.call(_succeed)
        # Counter reset; need 3 more failures
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert cb.is_closed


# ---------------------------------------------------------------------------
# TestRollingWindowMode
# ---------------------------------------------------------------------------

class TestRollingWindowMode:
    def test_window_mode_trips_on_rate(self):
        cb = CircuitBreaker(
            "window",
            config=CircuitBreakerConfig(
                window_size=4,
                failure_rate_threshold=0.5,
                failure_threshold=999,  # ignored in window mode
            ),
        )
        assert cb._window is not None
        # 2 successes, 2 failures = 50% rate → should trip
        cb.call(_succeed)
        cb.call(_succeed)
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert cb.is_closed  # only 3 calls, window not full
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert cb.is_open  # 4 calls, 50% failure rate

    def test_window_mode_does_not_trip_below_rate(self):
        cb = CircuitBreaker(
            "window",
            config=CircuitBreakerConfig(
                window_size=4,
                failure_rate_threshold=0.5,
            ),
        )
        cb.call(_succeed)
        cb.call(_succeed)
        cb.call(_succeed)
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert cb.is_closed  # 25% rate < 50% threshold

    def test_window_slides(self):
        cb = CircuitBreaker(
            "window",
            config=CircuitBreakerConfig(
                window_size=4,
                failure_rate_threshold=0.5,
                recovery_timeout=0.05,
                success_threshold=1,
            ),
        )
        # Fill window with 4 failures → trips
        for _ in range(4):
            with pytest.raises(TransientError):
                cb.call(_fail_transient)
        assert cb.is_open
        time.sleep(0.06)
        # Window cleared on close, start fresh
        cb.call(_succeed)
        assert cb.is_closed


# ---------------------------------------------------------------------------
# TestHalfOpenSerialProbes
# ---------------------------------------------------------------------------

class TestHalfOpenSerialProbes:
    def test_default_serial_probe(self):
        cb = CircuitBreaker(
            "serial",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout=0.05,
                half_open_max_calls=1,
                success_threshold=1,
            ),
        )
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        time.sleep(0.06)
        assert cb.is_half_open

        # First call takes the slot and succeeds → closes
        cb.call(_succeed)
        assert cb.is_closed

    def test_serial_probe_needs_multiple_successes(self):
        cb = CircuitBreaker(
            "serial",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout=0.05,
                half_open_max_calls=1,
                success_threshold=2,
                max_recovery_timeout=1.0,
            ),
        )
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        time.sleep(0.06)

        # First success doesn't close — still half-open
        cb.call(_succeed)
        assert cb.is_half_open
        # Second success closes
        cb.call(_succeed)
        assert cb.is_closed

    def test_multiple_probe_slots(self):
        cb = CircuitBreaker(
            "multi",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout=0.05,
                half_open_max_calls=3,
                success_threshold=2,
            ),
        )
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        time.sleep(0.06)
        cb.call(_succeed)
        cb.call(_succeed)
        assert cb.is_closed


# ---------------------------------------------------------------------------
# TestExponentialBackoff
# ---------------------------------------------------------------------------

class TestExponentialBackoff:
    def test_backoff_increases_timeout(self):
        cb = CircuitBreaker(
            "backoff",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout=0.05,
                success_threshold=1,
                backoff_factor=2.0,
                max_recovery_timeout=1.0,
            ),
        )
        # First trip
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert cb._current_recovery_timeout == 0.05  # no backoff on first trip from CLOSED
        time.sleep(0.06)
        assert cb.is_half_open
        # Fail again in half-open → re-opens with backoff
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert cb.is_open
        assert cb._current_recovery_timeout == pytest.approx(0.10)

    def test_backoff_caps_at_max(self):
        cb = CircuitBreaker(
            "backoff-cap",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout=0.05,
                success_threshold=1,
                backoff_factor=100.0,
                max_recovery_timeout=0.5,
            ),
        )
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        time.sleep(0.06)
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert cb._current_recovery_timeout == pytest.approx(0.5)

    def test_no_backoff_when_factor_is_one(self):
        cb = CircuitBreaker(
            "no-backoff",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout=0.05,
                success_threshold=1,
                backoff_factor=1.0,
            ),
        )
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        time.sleep(0.06)
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert cb._current_recovery_timeout == pytest.approx(0.05)

    def test_backoff_resets_on_close(self):
        cb = CircuitBreaker(
            "backoff-reset",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout=0.05,
                success_threshold=1,
                backoff_factor=2.0,
            ),
        )
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        time.sleep(0.06)
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert cb._current_recovery_timeout == pytest.approx(0.10)
        time.sleep(0.11)
        cb.call(_succeed)
        assert cb.is_closed
        assert cb._current_recovery_timeout == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# TestExpectedExceptionFiltering
# ---------------------------------------------------------------------------

class TestExpectedExceptionFiltering:
    def test_expected_exception_trips(self):
        cb = CircuitBreaker(
            "filter",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                expected_exception=TransientError,
            ),
        )
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert cb.is_open

    def test_unexpected_exception_does_not_trip(self):
        cb = CircuitBreaker(
            "filter",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                expected_exception=TransientError,
            ),
        )
        with pytest.raises(PermanentError):
            cb.call(_fail_permanent)
        assert cb.is_closed  # PermanentError is not TransientError

    def test_tuple_expected_exceptions(self):
        cb = CircuitBreaker(
            "filter-tuple",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                expected_exception=(TransientError, PermanentError),
            ),
        )
        with pytest.raises(PermanentError):
            cb.call(_fail_permanent)
        assert cb.is_open


# ---------------------------------------------------------------------------
# TestCallbacks
# ---------------------------------------------------------------------------

class TestCallbacks:
    def test_state_change_callback_fires(self):
        log = []

        def on_change(breaker, old, new):
            log.append((old, new))

        cb = CircuitBreaker(
            "cb-callback",
            config=CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.05),
            on_state_change=[on_change],
        )
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert (CircuitState.CLOSED, CircuitState.OPEN) in log

        time.sleep(0.06)
        _ = cb.state  # triggers half-open
        assert (CircuitState.OPEN, CircuitState.HALF_OPEN) in log

    def test_callback_error_does_not_propagate(self):
        def bad_callback(breaker, old, new):
            raise RuntimeError("callback error")

        cb = CircuitBreaker(
            "cb-bad",
            config=CircuitBreakerConfig(failure_threshold=1),
            on_state_change=[bad_callback],
        )
        # Should not raise even though callback throws
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert cb.is_open

    def test_callback_receives_breaker_ref(self):
        received = []

        def on_change(breaker, old, new):
            received.append(breaker.name)

        cb = CircuitBreaker(
            "cb-ref",
            config=CircuitBreakerConfig(failure_threshold=1),
            on_state_change=[on_change],
        )
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert received == ["cb-ref"]


# ---------------------------------------------------------------------------
# TestMetricsEmission
# ---------------------------------------------------------------------------

class TestMetricsEmission:
    @patch.object(_infra_cb_module, "_increment_counter")
    @patch.object(_infra_cb_module, "_set_gauge")
    def test_metrics_on_trip(self, mock_gauge, mock_counter):
        cb = CircuitBreaker(
            "metrics",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                category="chat",
                service="test_svc",
            ),
        )
        with pytest.raises(TransientError):
            cb.call(_fail_transient)

        # Should have emitted failure counter and trip counter
        failure_calls = [
            c for c in mock_counter.call_args_list
            if c[0][0] == "circuit_breaker_failures_total"
        ]
        assert len(failure_calls) >= 1

        trip_calls = [
            c for c in mock_counter.call_args_list
            if c[0][0] == "circuit_breaker_trips_total"
        ]
        assert len(trip_calls) >= 1

        # Should have set state gauge
        gauge_calls = [
            c for c in mock_gauge.call_args_list
            if c[0][0] == "circuit_breaker_state"
        ]
        assert len(gauge_calls) >= 1

    @patch.object(_infra_cb_module, "_increment_counter")
    @patch.object(_infra_cb_module, "_set_gauge")
    def test_metrics_on_success(self, mock_gauge, mock_counter):
        cb = CircuitBreaker("metrics-ok")
        cb.call(_succeed)
        success_calls = [
            c for c in mock_counter.call_args_list
            if c[0][0] == "circuit_breaker_successes_total"
        ]
        assert len(success_calls) == 1

    @patch.object(_infra_cb_module, "_increment_counter")
    @patch.object(_infra_cb_module, "_set_gauge")
    def test_metrics_on_rejection(self, mock_gauge, mock_counter):
        cb = CircuitBreaker("metrics-rej", config=CircuitBreakerConfig(failure_threshold=1))
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(_succeed)
        rejection_calls = [
            c for c in mock_counter.call_args_list
            if c[0][0] == "circuit_breaker_rejections_total"
        ]
        assert len(rejection_calls) >= 1

    @patch.object(_infra_cb_module, "_increment_counter")
    @patch.object(_infra_cb_module, "_set_gauge")
    def test_legacy_aliases_not_emitted_for_standard_metrics(self, mock_gauge, mock_counter):
        cb = CircuitBreaker(
            "metrics-legacy",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                category="chat",
                service="svc",
            ),
        )
        cb.call(_succeed)  # success metric
        with pytest.raises(TransientError):
            cb.call(_fail_transient)  # failure + trip + state_change
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(_succeed)  # rejection

        assert not any(
            c.kwargs.get("labels", {}).get("service") == "chat:svc"
            for c in mock_counter.call_args_list
        )
        assert not any(
            c.kwargs.get("labels", {}).get("service") == "chat:svc"
            for c in mock_gauge.call_args_list
        )


# ---------------------------------------------------------------------------
# TestRegistry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_get_or_create(self):
        reg = CircuitBreakerRegistry()
        cb1 = reg.get_or_create("r1")
        cb2 = reg.get_or_create("r1")
        assert cb1 is cb2

    def test_get_or_create_different_names(self):
        reg = CircuitBreakerRegistry()
        cb1 = reg.get_or_create("r1")
        cb2 = reg.get_or_create("r2")
        assert cb1 is not cb2

    def test_register_and_get(self):
        reg = CircuitBreakerRegistry()
        cb = CircuitBreaker("manual")
        reg.register(cb)
        assert reg.get("manual") is cb

    def test_get_missing_returns_none(self):
        reg = CircuitBreakerRegistry()
        assert reg.get("nonexistent") is None

    def test_get_all_status(self):
        reg = CircuitBreakerRegistry()
        reg.get_or_create("s1")
        reg.get_or_create("s2")
        status = reg.get_all_status()
        assert "s1" in status
        assert "s2" in status
        assert status["s1"]["state"] == "CLOSED"

    def test_reset_all(self):
        reg = CircuitBreakerRegistry()
        cb = reg.get_or_create("reset-all", config=CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure(TransientError("test"))
        assert cb.is_open
        reg.reset_all()
        assert cb.is_closed

    def test_reset_by_name(self):
        reg = CircuitBreakerRegistry()
        cb = reg.get_or_create("reset-one", config=CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure(TransientError("test"))
        assert cb.is_open
        assert reg.reset("reset-one") is True
        assert cb.is_closed
        assert reg.reset("nonexistent") is False

    def test_clear(self):
        reg = CircuitBreakerRegistry()
        reg.get_or_create("c1")
        reg.get_or_create("c2")
        assert reg.get_all_status()
        reg.clear()
        assert reg.get_all_status() == {}
        assert reg.get("c1") is None

    def test_persistence_shares_state_across_registry_instances(self, tmp_path):
        db_path = tmp_path / "cb_registry_shared.db"
        reg_a = CircuitBreakerRegistry(
            persistence_enabled=True,
            db_path=str(db_path),
        )
        reg_b = CircuitBreakerRegistry(
            persistence_enabled=True,
            db_path=str(db_path),
        )

        cfg = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=60.0,
        )
        cb_a = reg_a.get_or_create("shared-breaker", config=cfg)
        cb_b = reg_b.get_or_create("shared-breaker", config=cfg)

        with pytest.raises(TransientError):
            cb_a.call(_fail_transient)
        assert cb_a.is_open

        with pytest.raises(CircuitBreakerOpenError):
            cb_b.call(_succeed)

    def test_registry_persistence_opt_in_enabled(self, tmp_path):
        db_path = tmp_path / "cb_registry_enabled.db"
        reg = CircuitBreakerRegistry(
            persistence_enabled=True,
            db_path=str(db_path),
        )
        assert reg.persistence_enabled is True

    def test_registry_persistence_disabled(self):
        reg = CircuitBreakerRegistry(persistence_enabled=False)
        assert reg.persistence_enabled is False


class TestPersistentStore:
    class _ConflictSimulatingStore:
        def __init__(
            self,
            *,
            name: str,
            state: CircuitBreakerStoredState,
            version: int = 1,
            conflicts: int = 1,
            conflict_injector: Callable[[object], None] | None = None,
        ):
            self._name = name
            self._state = state
            self._version = version
            self._conflicts_remaining = conflicts
            self._conflict_injector = conflict_injector
            self.upsert_calls = 0

        def load(self, name: str):
            if name != self._name:
                return None
            return self._state, self._version

        def upsert(self, name: str, state: CircuitBreakerStoredState, *, expected_version: int):
            self.upsert_calls += 1
            if name != self._name:
                raise CircuitBreakerOptimisticLockError(
                    name,
                    expected_version=expected_version,
                    current_version=None,
                )
            if self._conflicts_remaining > 0:
                self._conflicts_remaining -= 1
                if self._conflict_injector is not None:
                    self._conflict_injector(self)
                raise CircuitBreakerOptimisticLockError(
                    name,
                    expected_version=expected_version,
                    current_version=self._version,
                )
            if expected_version > 0 and expected_version != self._version:
                raise CircuitBreakerOptimisticLockError(
                    name,
                    expected_version=expected_version,
                    current_version=self._version,
                )
            if expected_version <= 0 and self._version != 0:
                raise CircuitBreakerOptimisticLockError(
                    name,
                    expected_version=expected_version,
                    current_version=self._version,
                )
            self._version = self._version + 1 if self._version > 0 else 1
            self._state = state
            return self._version

    @staticmethod
    def _stored(
        *,
        state: CircuitState,
        failure_count: int,
        success_count: int,
        last_failure_time: float | None,
        last_state_change_time: float,
        half_open_calls: int = 0,
        current_recovery_timeout: float = 30.0,
        last_trip_failure_count: int = 0,
    ) -> CircuitBreakerStoredState:
        return CircuitBreakerStoredState(
            state=int(state.value),
            failure_count=failure_count,
            success_count=success_count,
            last_failure_time=last_failure_time,
            last_state_change_time=last_state_change_time,
            half_open_calls=half_open_calls,
            current_recovery_timeout=current_recovery_timeout,
            last_trip_failure_count=last_trip_failure_count,
        )

    def test_optimistic_locking_detects_version_conflicts(self, tmp_path):
        db = CircuitBreakerRegistryDB(tmp_path / "cb_store.db")
        reg = CircuitBreakerRegistry(
            persistence_enabled=True,
            db_path=str(tmp_path / "cb_store.db"),
        )
        cb = reg.get_or_create("lock-test", config=CircuitBreakerConfig())
        cb.call(_succeed)

        row = db.load("lock-test")
        assert row is not None
        stored, version = row
        assert version >= 1

        next_version = db.upsert(
            "lock-test",
            stored,
            expected_version=version,
        )
        assert next_version == version + 1

        with pytest.raises(CircuitBreakerOptimisticLockError):
            db.upsert(
                "lock-test",
                stored,
                expected_version=version,
            )

    def test_conflict_retry_merges_success_and_closes_half_open(self):
        now = time.time()
        breaker_name = "merge-success"
        initial = self._stored(
            state=CircuitState.HALF_OPEN,
            failure_count=0,
            success_count=0,
            last_failure_time=now - 20,
            last_state_change_time=now - 10,
            current_recovery_timeout=30.0,
            last_trip_failure_count=2,
        )

        def _inject_other_success(store: TestPersistentStore._ConflictSimulatingStore) -> None:
            store._version += 1
            store._state = self._stored(
                state=CircuitState.HALF_OPEN,
                failure_count=0,
                success_count=1,
                last_failure_time=now - 5,
                last_state_change_time=now - 4,
                current_recovery_timeout=30.0,
                last_trip_failure_count=2,
            )

        store = self._ConflictSimulatingStore(
            name=breaker_name,
            state=initial,
            version=1,
            conflicts=1,
            conflict_injector=_inject_other_success,
        )
        cb = CircuitBreaker(
            breaker_name,
            config=CircuitBreakerConfig(success_threshold=2),
            state_store=store,
        )
        assert cb.is_half_open

        cb.record_success()

        assert cb.is_closed
        persisted, _ = store.load(breaker_name)
        assert persisted.state == int(CircuitState.CLOSED.value)
        assert persisted.success_count == 0
        assert store.upsert_calls >= 2

    def test_conflict_retry_merges_failure_and_preserves_trip_threshold(self):
        now = time.time()
        breaker_name = "merge-failure"
        initial = self._stored(
            state=CircuitState.CLOSED,
            failure_count=1,
            success_count=0,
            last_failure_time=now - 30,
            last_state_change_time=now - 30,
            current_recovery_timeout=45.0,
        )

        def _inject_other_failure(store: TestPersistentStore._ConflictSimulatingStore) -> None:
            store._version += 1
            store._state = self._stored(
                state=CircuitState.CLOSED,
                failure_count=2,
                success_count=0,
                last_failure_time=now - 2,
                last_state_change_time=now - 2,
                current_recovery_timeout=45.0,
            )

        store = self._ConflictSimulatingStore(
            name=breaker_name,
            state=initial,
            version=1,
            conflicts=1,
            conflict_injector=_inject_other_failure,
        )
        cb = CircuitBreaker(
            breaker_name,
            config=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=45.0,
            ),
            state_store=store,
        )
        assert cb.is_closed

        cb.record_failure(TransientError("boom"))

        assert cb.is_open
        persisted, _ = store.load(breaker_name)
        assert persisted.state == int(CircuitState.OPEN.value)
        assert persisted.last_trip_failure_count >= 3

    def test_conflict_retry_preserves_newer_open_during_transition(self):
        now = time.time()
        breaker_name = "merge-transition-open"
        initial = self._stored(
            state=CircuitState.OPEN,
            failure_count=0,
            success_count=0,
            last_failure_time=now - 10,
            last_state_change_time=now - 10,
            current_recovery_timeout=60.0,
            last_trip_failure_count=4,
        )

        def _inject_newer_open(store: TestPersistentStore._ConflictSimulatingStore) -> None:
            store._version += 1
            store._state = self._stored(
                state=CircuitState.OPEN,
                failure_count=0,
                success_count=0,
                last_failure_time=now - 1,
                last_state_change_time=now + 1,
                current_recovery_timeout=120.0,
                last_trip_failure_count=5,
            )

        store = self._ConflictSimulatingStore(
            name=breaker_name,
            state=initial,
            version=1,
            conflicts=1,
            conflict_injector=_inject_newer_open,
        )
        cb = CircuitBreaker(
            breaker_name,
            config=CircuitBreakerConfig(),
            state_store=store,
        )
        assert cb.is_open

        cb.force_half_open()

        assert cb.is_open
        persisted, _ = store.load(breaker_name)
        assert persisted.state == int(CircuitState.OPEN.value)
        assert persisted.current_recovery_timeout >= 120.0

    def test_conflict_retry_exhaustion_adopts_latest_state_without_raising(self):
        now = time.time()
        breaker_name = "merge-exhaustion"
        latest = self._stored(
            state=CircuitState.CLOSED,
            failure_count=5,
            success_count=0,
            last_failure_time=now - 2,
            last_state_change_time=now - 2,
            current_recovery_timeout=30.0,
        )
        store = self._ConflictSimulatingStore(
            name=breaker_name,
            state=latest,
            version=5,
            conflicts=999,
            conflict_injector=None,
        )
        cb = CircuitBreaker(
            breaker_name,
            config=CircuitBreakerConfig(failure_threshold=100),
            state_store=store,
        )

        with patch.object(_infra_cb_module.logger, "warning") as warning_spy:
            cb.record_failure(TransientError("will-conflict"))

        assert warning_spy.called
        assert cb.failure_count == 5
        assert cb.is_closed
        assert store.upsert_calls == cb._persist_retry_attempts

    def test_distributed_half_open_leases_enforce_global_probe_limit(self, tmp_path):
        db_path = tmp_path / "cb_probe_leases_limit.db"
        name = "lease-limit"
        cfg = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=5.0,
            half_open_max_calls=1,
            success_threshold=1,
        )
        reg_a = CircuitBreakerRegistry(
            persistence_enabled=True,
            db_path=str(db_path),
        )
        reg_b = CircuitBreakerRegistry(
            persistence_enabled=True,
            db_path=str(db_path),
        )
        cb_a = reg_a.get_or_create(name, config=cfg)
        cb_b = reg_b.get_or_create(name, config=cfg)
        db = CircuitBreakerRegistryDB(db_path)

        cb_a.record_failure(TransientError("trip"))
        assert cb_a.is_open
        cb_a.force_half_open()

        release_event = threading.Event()
        result: dict[str, object] = {}

        def _blocking_work():
            release_event.wait(timeout=1.0)
            return "ok"

        def _run_a():
            try:
                result["value"] = cb_a.call(_blocking_work)
            except Exception as exc:  # noqa: BLE001
                result["error"] = exc

        t = threading.Thread(target=_run_a, daemon=True)
        t.start()

        deadline = time.time() + 1.0
        while time.time() < deadline and db.count_active_probe_leases(name) < 1:
            time.sleep(0.01)

        assert db.count_active_probe_leases(name) == 1
        with pytest.raises(CircuitBreakerOpenError):
            cb_b.call(_succeed)

        release_event.set()
        t.join(timeout=1.0)
        assert not t.is_alive()
        assert "error" not in result
        assert result.get("value") == "ok"
        assert db.count_active_probe_leases(name) == 0

    def test_expired_probe_lease_allows_later_acquisition(self, tmp_path):
        db = CircuitBreakerRegistryDB(tmp_path / "cb_probe_leases_expiry.db")
        name = "lease-expiry"

        lease1 = db.acquire_probe_lease(
            name,
            max_calls=1,
            ttl_seconds=0.05,
            owner_id="worker-1",
        )
        assert lease1 is not None
        assert db.count_active_probe_leases(name) == 1

        time.sleep(0.07)
        lease2 = db.acquire_probe_lease(
            name,
            max_calls=1,
            ttl_seconds=1.0,
            owner_id="worker-2",
        )
        assert lease2 is not None
        assert lease2.lease_id != lease1.lease_id
        assert db.count_active_probe_leases(name) == 1
        assert db.release_probe_lease(name, lease2.lease_id) is True
        assert db.count_active_probe_leases(name) == 0

    def test_probe_leases_released_after_success_and_failure_paths(self, tmp_path):
        db_path = tmp_path / "cb_probe_leases_release.db"
        name = "lease-release"
        reg = CircuitBreakerRegistry(
            persistence_enabled=True,
            db_path=str(db_path),
        )
        db = CircuitBreakerRegistryDB(db_path)
        cb = reg.get_or_create(
            name,
            config=CircuitBreakerConfig(
                failure_threshold=1,
                half_open_max_calls=1,
                success_threshold=2,
            ),
        )

        cb.record_failure(TransientError("trip"))
        assert cb.is_open
        cb.force_half_open()

        assert cb.call(_succeed) == "ok"
        assert cb.is_half_open
        assert db.count_active_probe_leases(name) == 0

        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert cb.is_open
        assert db.count_active_probe_leases(name) == 0

    def test_open_and_closed_transitions_clear_probe_leases(self, tmp_path):
        db_path = tmp_path / "cb_probe_leases_transition_cleanup.db"
        name = "lease-transition-cleanup"
        reg = CircuitBreakerRegistry(
            persistence_enabled=True,
            db_path=str(db_path),
        )
        cb = reg.get_or_create(name, config=CircuitBreakerConfig())
        db = CircuitBreakerRegistryDB(db_path)

        lease = db.acquire_probe_lease(
            name,
            max_calls=1,
            ttl_seconds=60.0,
            owner_id="stale-worker",
        )
        assert lease is not None
        assert db.count_active_probe_leases(name) == 1

        cb.force_open()
        assert db.count_active_probe_leases(name) == 0

        lease2 = db.acquire_probe_lease(
            name,
            max_calls=1,
            ttl_seconds=60.0,
            owner_id="stale-worker-2",
        )
        assert lease2 is not None
        assert db.count_active_probe_leases(name) == 1

        cb.reset()
        assert db.count_active_probe_leases(name) == 0


# ---------------------------------------------------------------------------
# TestDecorator
# ---------------------------------------------------------------------------

class TestDecorator:
    def test_sync_decorator(self):
        @circuit_breaker(name="dec-sync-test", failure_threshold=1)
        def my_func():
            return 42

        assert my_func() == 42

    def test_sync_decorator_trips(self):
        @circuit_breaker(name="dec-sync-trip", failure_threshold=1)
        def my_func():
            raise TransientError("boom")

        with pytest.raises(TransientError):
            my_func()
        with pytest.raises(CircuitBreakerOpenError):
            my_func()

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        @circuit_breaker(name="dec-async-test", failure_threshold=1)
        async def my_func():
            return 42

        assert await my_func() == 42

    @pytest.mark.asyncio
    async def test_async_decorator_trips(self):
        @circuit_breaker(name="dec-async-trip", failure_threshold=1)
        async def my_func():
            raise TransientError("boom")

        with pytest.raises(TransientError):
            await my_func()
        with pytest.raises(CircuitBreakerOpenError):
            await my_func()

    def test_decorator_registers_in_global_registry(self):
        @circuit_breaker(name="dec-global-test")
        def my_func():
            return 1

        assert registry.get("dec-global-test") is not None


# ---------------------------------------------------------------------------
# TestCheckPatternAPI
# ---------------------------------------------------------------------------

class TestCheckPatternAPI:
    def test_can_attempt_when_closed(self):
        cb = CircuitBreaker("check")
        assert cb.can_attempt() is True

    def test_can_attempt_when_open(self):
        cb = CircuitBreaker("check", config=CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure(TransientError("test"))
        assert cb.can_attempt() is False

    def test_record_success_closes_from_half_open(self):
        cb = CircuitBreaker(
            "check",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout=0.05,
                success_threshold=1,
            ),
        )
        cb.record_failure(TransientError("test"))
        assert cb.is_open
        time.sleep(0.06)
        assert cb.can_attempt() is True
        cb.record_success()
        assert cb.is_closed

    def test_record_failure_reopens_from_half_open(self):
        cb = CircuitBreaker(
            "check",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout=0.05,
            ),
        )
        cb.record_failure(TransientError("test"))
        time.sleep(0.06)
        cb.record_failure(TransientError("again"))
        assert cb.is_open


# ---------------------------------------------------------------------------
# TestConcurrency
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_thread_safety(self):
        cb = CircuitBreaker(
            "concurrent",
            config=CircuitBreakerConfig(failure_threshold=100),
        )
        errors = []

        def worker():
            try:
                for _ in range(50):
                    try:
                        cb.call(_succeed)
                    except CircuitBreakerOpenError:
                        pass
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert cb.is_closed

    def test_concurrent_failures(self):
        cb = CircuitBreaker(
            "concurrent-fail",
            config=CircuitBreakerConfig(failure_threshold=10),
        )
        errors = []

        def worker():
            try:
                for _ in range(5):
                    try:
                        cb.call(_fail_transient)
                    except (TransientError, CircuitBreakerOpenError):
                        pass
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert cb.is_open


# ---------------------------------------------------------------------------
# TestGetStatus
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_status_fields(self):
        cb = CircuitBreaker("status-test", config=CircuitBreakerConfig(category="cat"))
        status = cb.get_status()
        assert status["name"] == "status-test"
        assert status["state"] == "CLOSED"
        assert status["failure_count"] == 0
        assert status["success_count"] == 0
        assert status["last_failure_time"] is None
        assert "settings" in status

    def test_status_with_window(self):
        cb = CircuitBreaker("status-win", config=CircuitBreakerConfig(window_size=5))
        cb.call(_succeed)
        status = cb.get_status()
        assert "window" in status
        assert "failure_rate" in status
        assert status["window"] == [True]
        assert status["failure_rate"] == 0.0


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_from_open(self):
        cb = CircuitBreaker("reset-test", config=CircuitBreakerConfig(failure_threshold=1))
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert cb.is_open
        cb.reset()
        assert cb.is_closed
        assert cb.failure_count == 0

    def test_reset_clears_window(self):
        cb = CircuitBreaker("reset-win", config=CircuitBreakerConfig(window_size=5))
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        cb.reset()
        assert len(cb._window) == 0

    def test_reset_resets_backoff(self):
        cb = CircuitBreaker(
            "reset-bo",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout=0.05,
                backoff_factor=2.0,
            ),
        )
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        time.sleep(0.06)
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        assert cb._current_recovery_timeout > 0.05
        cb.reset()
        assert cb._current_recovery_timeout == pytest.approx(0.05)

    def test_reset_fires_callback(self):
        log = []
        cb = CircuitBreaker(
            "reset-cb",
            config=CircuitBreakerConfig(failure_threshold=1),
            on_state_change=[lambda b, o, n: log.append((o, n))],
        )
        with pytest.raises(TransientError):
            cb.call(_fail_transient)
        log.clear()
        cb.reset()
        assert (CircuitState.OPEN, CircuitState.CLOSED) in log


# ---------------------------------------------------------------------------
# TestConvenienceKwargs
# ---------------------------------------------------------------------------

class TestConvenienceKwargs:
    def test_kwargs_create_config(self):
        cb = CircuitBreaker(
            "kw-test",
            failure_threshold=3,
            recovery_timeout=10.0,
            category="chat",
        )
        assert cb.config.failure_threshold == 3
        assert cb.config.recovery_timeout == 10.0
        assert cb._category == "chat"

    def test_config_takes_precedence(self):
        cfg = CircuitBreakerConfig(failure_threshold=7)
        cb = CircuitBreaker("kw-cfg", config=cfg, failure_threshold=3)
        assert cb.config.failure_threshold == 7  # config wins


# ---------------------------------------------------------------------------
# TestRepr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr(self):
        cb = CircuitBreaker("repr-test")
        r = repr(cb)
        assert "repr-test" in r
        assert "CLOSED" in r
