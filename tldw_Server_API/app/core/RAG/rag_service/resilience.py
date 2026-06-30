# resilience.py
"""
Error recovery and resilience mechanisms for the RAG service.

This module provides circuit breakers, retry policies, fallback strategies,
and health monitoring to ensure robust operation of the RAG pipeline.

The core CircuitBreaker is delegated to the unified Infrastructure module;
this file retains the RAG-specific domain orchestration (RetryPolicy,
FallbackChain, HealthMonitor, ErrorRecoveryCoordinator).
"""

import asyncio
import contextlib
import secrets
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, cast

from loguru import logger

from tldw_Server_API.app.core.Infrastructure.circuit_breaker import (
    CircuitBreaker as _UnifiedCB,
)
from tldw_Server_API.app.core.Infrastructure.circuit_breaker import (
    CircuitBreakerConfig as _UnifiedCfg,
)
from tldw_Server_API.app.core.Infrastructure.circuit_breaker import (
    CircuitBreakerOpenError,
)
from tldw_Server_API.app.core.Infrastructure.circuit_breaker import (
    CircuitState as _UnifiedState,
)

T = TypeVar('T')
_JITTER_RANDOM = secrets.SystemRandom()


# ---------------------------------------------------------------------------
# Backward-compatible re-exports
# ---------------------------------------------------------------------------

class CircuitState(Enum):
    """Circuit breaker states (backward-compatible string values)."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


CircuitOpenError = CircuitBreakerOpenError


class HealthStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker (backward-compatible with old API).

    The ``timeout`` field maps to ``recovery_timeout`` in the unified config.
    """
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0
    window_size: int = 10
    failure_rate_threshold: float = 0.5

    def _to_unified(self) -> _UnifiedCfg:
        return _UnifiedCfg(
            failure_threshold=self.failure_threshold,
            success_threshold=self.success_threshold,
            recovery_timeout=self.timeout,
            window_size=self.window_size,
            failure_rate_threshold=self.failure_rate_threshold,
            category="rag",
        )


@dataclass
class RetryConfig:
    """Configuration for retry policy."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: list[type] = field(default_factory=lambda: [Exception])
    dont_retry_on: list[type] = field(default_factory=list)


@dataclass
class ErrorContext:
    """Context information for errors."""
    error: Exception
    timestamp: float
    component: str
    operation: str
    attempt: int
    metadata: dict[str, Any] = field(default_factory=dict)
    traceback: Optional[str] = None

    def __post_init__(self):
        """Capture traceback if not provided."""
        if self.traceback is None and self.error:
            self.traceback = traceback.format_exc()


# ---------------------------------------------------------------------------
# CircuitBreaker — thin wrapper preserving old API
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """RAG circuit breaker wrapping the unified Infrastructure implementation.

    Preserves the old public API (``call``, ``reset``, ``get_stats``,
    ``state``, ``failure_count``, ``success_count``, ``on_*_callbacks``).
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._cb = _UnifiedCB(
            name=f"rag_{name}",
            config=self.config._to_unified(),
        )
        # Callbacks (preserved for API compat; no existing code populates them)
        self.on_open_callbacks: list[Callable] = []
        self.on_close_callbacks: list[Callable] = []
        self.on_half_open_callbacks: list[Callable] = []
        # Wire unified callbacks to dispatch old-style per-state callbacks
        self._cb.on_state_change.append(self._dispatch_callback)

    # -- state property (returns old-style CircuitState enum) ----------------

    @property
    def state(self) -> CircuitState:
        s = self._cb.state
        if s == _UnifiedState.CLOSED:
            return CircuitState.CLOSED
        elif s == _UnifiedState.OPEN:
            return CircuitState.OPEN
        return CircuitState.HALF_OPEN

    @property
    def failure_count(self) -> int:
        return self._cb.failure_count

    @property
    def success_count(self) -> int:
        return self._cb.success_count

    @property
    def last_failure_time(self) -> Optional[float]:
        return self._cb.last_failure_time

    @property
    def last_state_change(self) -> float:
        return self._cb.last_state_change_time

    # -- callback dispatch ---------------------------------------------------

    def _dispatch_callback(self, breaker: Any, old_state: Any, new_state: Any) -> None:
        """Dispatch unified callback to old-style per-state callbacks."""
        if new_state == _UnifiedState.OPEN:
            for cb in self.on_open_callbacks:
                try:
                    cb(self)
                except Exception:  # noqa: BLE001
                    logger.error("Error in open callback")
        elif new_state == _UnifiedState.CLOSED:
            for cb in self.on_close_callbacks:
                try:
                    cb(self)
                except Exception:  # noqa: BLE001
                    logger.error("Error in close callback")
        elif new_state == _UnifiedState.HALF_OPEN:
            for cb in self.on_half_open_callbacks:
                try:
                    cb(self)
                except Exception:  # noqa: BLE001
                    logger.error("Error in half-open callback")

    # -- core API ------------------------------------------------------------

    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Call function through circuit breaker (handles both sync and async)."""
        if asyncio.iscoroutinefunction(func):
            return await self._cb.call_async(func, *args, **kwargs)
        else:
            async def _wrap():
                return await asyncio.to_thread(func, *args, **kwargs)
            return await self._cb.call_async(_wrap)

    def reset(self):
        """Reset circuit breaker to closed state."""
        self._cb.reset()
        logger.info(f"Circuit breaker '{self.name}' reset")

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics (backward-compatible format)."""
        status = self._cb.get_status()
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": status.get("failure_count", 0),
            "success_count": status.get("success_count", 0),
            "call_history": status.get("window", []),
            "failure_rate": status.get("failure_rate", 0.0),
            "last_failure_time": status.get("last_failure_time"),
            "last_state_change": self._cb.last_state_change_time,
        }


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------

class RetryPolicy:
    """Retry policy with exponential backoff."""

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()

    async def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with retry policy."""
        last_exception = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = await asyncio.to_thread(func, *args, **kwargs)

                if attempt > 1:
                    logger.info(f"Retry succeeded on attempt {attempt}")

                return cast(T, result)

            except Exception as e:  # noqa: BLE001 - retry policy inspects all exceptions
                last_exception = e

                should_retry = self._should_retry(e)

                if not should_retry or attempt == self.config.max_attempts:
                    logger.error(f"Retry failed after {attempt} attempts")
                    raise

                delay = self._calculate_delay(attempt)

                logger.warning(
                    f"Attempt {attempt} failed. Retrying in {delay:.2f}s..."
                )

                await asyncio.sleep(delay)

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Retry policy failed without capturing an exception")

    def _should_retry(self, exception: Exception) -> bool:
        """Check if exception should be retried."""
        for exc_type in self.config.dont_retry_on:
            if isinstance(exception, exc_type):
                return False
        return any(isinstance(exception, exc_type) for exc_type in self.config.retry_on)

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff."""
        delay = self.config.initial_delay * (self.config.exponential_base ** (attempt - 1))
        delay = min(delay, self.config.max_delay)

        if self.config.jitter:
            jitter = delay * 0.25 * (2 * _JITTER_RANDOM.random() - 1)
            delay += jitter

        return max(0, delay)


# ---------------------------------------------------------------------------
# FallbackChain
# ---------------------------------------------------------------------------

class FallbackChain:
    """Chain of fallback strategies."""

    def __init__(self):
        self.strategies: list[tuple[Callable[..., Any], Optional[Callable[[Exception], bool]]]] = []

    def add_strategy(
        self,
        func: Callable,
        condition: Optional[Callable[[Exception], bool]] = None
    ):
        """Add fallback strategy."""
        self.strategies.append((func, condition))

    async def execute(self, primary_func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute with fallback chain."""
        try:
            if asyncio.iscoroutinefunction(primary_func):
                return await primary_func(*args, **kwargs)
            else:
                return await asyncio.to_thread(primary_func, *args, **kwargs)
        except Exception as primary_error:  # noqa: BLE001 - fallback chain best-effort
            logger.warning("Primary function failed")

            for fallback_func, condition in self.strategies:
                if condition is None or condition(primary_error):
                    try:
                        logger.info(f"Trying fallback: {fallback_func.__name__}")

                        if asyncio.iscoroutinefunction(fallback_func):
                            return await fallback_func(*args, **kwargs)
                        else:
                            return await asyncio.to_thread(fallback_func, *args, **kwargs)

                    except Exception:  # noqa: BLE001 - fallback chain best-effort
                        logger.warning("Fallback failed")
                        continue

            logger.error("All fallback strategies failed")
            raise


# ---------------------------------------------------------------------------
# HealthMonitor
# ---------------------------------------------------------------------------

class HealthMonitor:
    """Monitor component health."""

    def __init__(self):
        self.components: dict[str, ComponentHealth] = {}
        self.check_interval = 30
        self.monitoring_task = None

    def register_component(
        self,
        name: str,
        health_check: Callable[[], bool],
        critical: bool = False
    ):
        """Register component for monitoring."""
        self.components[name] = ComponentHealth(
            name=name,
            health_check=health_check,
            critical=critical
        )

    async def start_monitoring(self):
        """Start health monitoring."""
        if not self.monitoring_task:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """Stop health monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitoring_task
            self.monitoring_task = None

    async def _monitoring_loop(self):
        """Background health monitoring loop."""
        while True:
            try:
                await self.check_all_health()
                await asyncio.sleep(self.check_interval)
            except Exception:  # noqa: BLE001 - keep monitoring loop alive
                logger.error("Error in health monitoring")
                await asyncio.sleep(self.check_interval)

    async def check_all_health(self) -> dict[str, HealthStatus]:
        """Check health of all components."""
        results = {}

        for name, component in self.components.items():
            try:
                if asyncio.iscoroutinefunction(component.health_check):
                    is_healthy = await component.health_check()
                else:
                    is_healthy = await asyncio.to_thread(component.health_check)

                component.update_health(is_healthy)
                results[name] = component.status

                if not is_healthy and component.critical:
                    logger.error(f"Critical component '{name}' is unhealthy")

            except Exception:  # noqa: BLE001 - health checks are best-effort
                logger.error(f"Health check failed for '{name}'")
                component.update_health(False)
                results[name] = HealthStatus.UNKNOWN

        return results

    def get_overall_health(self) -> HealthStatus:
        """Get overall system health."""
        if not self.components:
            return HealthStatus.UNKNOWN

        critical_unhealthy = any(
            c.critical and c.status != HealthStatus.HEALTHY
            for c in self.components.values()
        )

        if critical_unhealthy:
            return HealthStatus.UNHEALTHY

        statuses = [c.status for c in self.components.values()]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.DEGRADED


@dataclass
class ComponentHealth:
    """Health information for a component."""
    name: str
    health_check: Callable[[], bool]
    critical: bool = False
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: Optional[float] = None
    consecutive_failures: int = 0

    def update_health(self, is_healthy: bool):
        """Update health status."""
        self.last_check = time.time()

        if is_healthy:
            self.status = HealthStatus.HEALTHY
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

            if self.consecutive_failures >= 3:
                self.status = HealthStatus.UNHEALTHY
            else:
                self.status = HealthStatus.DEGRADED


# ---------------------------------------------------------------------------
# ErrorRecoveryCoordinator
# ---------------------------------------------------------------------------

class ErrorRecoveryCoordinator:
    """Coordinates error recovery across components."""

    def __init__(self):
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.retry_policies: dict[str, RetryPolicy] = {}
        self.fallback_chains: dict[str, FallbackChain] = {}
        self.health_monitor = HealthMonitor()
        self.error_history: deque = deque(maxlen=100)

    def register_circuit_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Register a circuit breaker."""
        breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = breaker
        return breaker

    def register_retry_policy(
        self,
        name: str,
        config: Optional[RetryConfig] = None
    ) -> RetryPolicy:
        """Register a retry policy."""
        policy = RetryPolicy(config)
        self.retry_policies[name] = policy
        return policy

    def register_fallback_chain(
        self,
        name: str
    ) -> FallbackChain:
        """Register a fallback chain."""
        chain = FallbackChain()
        self.fallback_chains[name] = chain
        return chain

    def record_error(self, error_context: ErrorContext):
        """Record an error for analysis."""
        self.error_history.append(error_context)

        recent_errors = [
            e for e in self.error_history
            if e.component == error_context.component and
            time.time() - e.timestamp < 60
        ]

        if len(recent_errors) > 5:
            logger.error(
                f"High error rate for component '{error_context.component}': "
                f"{len(recent_errors)} errors in last minute"
            )

    def get_recovery_stats(self) -> dict[str, Any]:
        """Get recovery system statistics."""
        return {
            "circuit_breakers": {
                name: breaker.get_stats()
                for name, breaker in self.circuit_breakers.items()
            },
            "health_status": self.health_monitor.get_overall_health().value,
            "error_count": len(self.error_history),
            "recent_errors": [
                {
                    "component": e.component,
                    "operation": e.operation,
                    "timestamp": e.timestamp,
                    "error": "Error details unavailable"
                }
                for e in list(self.error_history)[-10:]
            ]
        }


# Global coordinator instance
_coordinator: Optional[ErrorRecoveryCoordinator] = None


def get_coordinator() -> ErrorRecoveryCoordinator:
    """Get or create global error recovery coordinator."""
    global _coordinator
    if _coordinator is None:
        _coordinator = ErrorRecoveryCoordinator()
    return _coordinator


# ---------------------------------------------------------------------------
# Pipeline integration functions
# ---------------------------------------------------------------------------

async def with_circuit_breaker(
    context: Any,
    component: str = "default",
    **kwargs
) -> Any:
    """Wrap pipeline function with circuit breaker."""
    coordinator = get_coordinator()

    if component not in coordinator.circuit_breakers:
        config = CircuitBreakerConfig(
            failure_threshold=kwargs.get("failure_threshold", 5),
            timeout=kwargs.get("timeout", 60.0)
        )
        coordinator.register_circuit_breaker(component, config)

    breaker = coordinator.circuit_breakers[component]

    context.metadata[f"circuit_breaker_{component}"] = breaker.state.value

    return context


async def with_retry(
    context: Any,
    component: str = "default",
    **kwargs
) -> Any:
    """Wrap pipeline function with retry policy."""
    coordinator = get_coordinator()

    if component not in coordinator.retry_policies:
        config = RetryConfig(
            max_attempts=kwargs.get("max_attempts", 3),
            initial_delay=kwargs.get("initial_delay", 1.0)
        )
        coordinator.register_retry_policy(component, config)

    policy = coordinator.retry_policies[component]

    context.metadata[f"retry_policy_{component}"] = {
        "max_attempts": policy.config.max_attempts,
        "initial_delay": policy.config.initial_delay
    }

    return context


async def with_fallback(
    context: Any,
    component: str = "default",
    fallback_func: Optional[Callable] = None,
    **kwargs
) -> Any:
    """Add fallback strategy for pipeline function."""
    coordinator = get_coordinator()

    if component not in coordinator.fallback_chains:
        coordinator.register_fallback_chain(component)

    chain = coordinator.fallback_chains[component]

    if fallback_func:
        chain.add_strategy(fallback_func)

    context.metadata[f"fallback_{component}"] = {
        "strategies_count": len(chain.strategies)
    }

    return context


async def check_component_health(
    context: Any,
    component: str,
    health_check: Callable[[], bool],
    **kwargs
) -> Any:
    """Check component health before proceeding."""
    coordinator = get_coordinator()

    if component not in coordinator.health_monitor.components:
        coordinator.health_monitor.register_component(
            component,
            health_check,
            critical=kwargs.get("critical", False)
        )

    try:
        if asyncio.iscoroutinefunction(health_check):
            is_healthy = await health_check()
        else:
            is_healthy = health_check()

        context.metadata[f"health_{component}"] = "healthy" if is_healthy else "unhealthy"

        if not is_healthy and kwargs.get("critical", False):
            raise Exception(f"Critical component '{component}' is unhealthy")

    except Exception:  # noqa: BLE001 - health checks are best-effort
        logger.error(f"Health check failed for '{component}'")
        context.metadata[f"health_{component}"] = "unknown"

        if kwargs.get("critical", False):
            raise

    return context
