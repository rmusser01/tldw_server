# circuit_breaker.py
# Description: Circuit breaker pattern implementation for TTS providers
#
# Core breaker logic delegated to Infrastructure.circuit_breaker.
# This file retains TTS-specific domain logic:
#   - CircuitBreakerManager (per-provider config, ConfigParser support)
#   - Error categorization via categorize_error()
#   - Health monitoring (start/stop, background loop)
#   - get_detailed_status() / get_error_analysis()
#
# Imports
import asyncio
import contextlib
import secrets
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

#
# Third-party Imports
from loguru import logger

from tldw_Server_API.app.core.Infrastructure.circuit_breaker import (
    CircuitBreaker as _UnifiedCB,
)
from tldw_Server_API.app.core.Infrastructure.circuit_breaker import (
    CircuitBreakerConfig as _UnifiedCfg,
)
from tldw_Server_API.app.core.Infrastructure.circuit_breaker import (
    CircuitState as _UnifiedState,
)

#
# Local Imports
from .tts_exceptions import (
    TTSCircuitOpenError,
    categorize_error,
)

#
#######################################################################################################################
#
# Circuit Breaker Implementation

_JITTER_RANDOM = secrets.SystemRandom()


@dataclass
class CircuitStats:
    """Statistics for circuit breaker"""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_requests: int = 0

    def record_success(self):
        self.success_count += 1
        self.total_requests += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = time.time()

    def record_failure(self):
        self.failure_count += 1
        self.total_requests += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = time.time()

    def reset(self):
        self.failure_count = 0
        self.success_count = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0

    @property
    def failure_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failure_count / self.total_requests

    def get_backoff_delay(self, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        if self.consecutive_failures == 0:
            return 0.0
        delay = base_delay * (2 ** min(self.consecutive_failures - 1, 10))
        delay = min(delay, max_delay)
        jitter = delay * 0.25 * (2 * _JITTER_RANDOM.random() - 1)
        return max(0.1, delay + jitter)


# Backward-compat enum-like (now delegates to Infrastructure.CircuitState)
class CircuitState:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


def _unified_to_str(s: _UnifiedState) -> str:
    if s == _UnifiedState.CLOSED:
        return CircuitState.CLOSED
    elif s == _UnifiedState.OPEN:
        return CircuitState.OPEN
    return CircuitState.HALF_OPEN


class CircuitBreaker:
    """Circuit breaker for TTS providers.

    Core state machine delegated to the unified Infrastructure breaker.
    TTS-specific features (error categorization, health monitoring,
    backoff stats) live here.
    """

    def __init__(
        self,
        provider_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
        success_threshold: int = 2,
        backoff_base_delay: float = 1.0,
        backoff_max_delay: float = 60.0,
        health_check_interval: float = 300.0,
    ):
        self.provider_name = provider_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        self.backoff_base_delay = backoff_base_delay
        self.backoff_max_delay = backoff_max_delay
        self.health_check_interval = health_check_interval

        # Unified breaker
        self._cb = _UnifiedCB(
            name=f"tts_{provider_name}",
            config=_UnifiedCfg(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                half_open_max_calls=half_open_max_calls,
                success_threshold=success_threshold,
                category="tts",
                service=provider_name,
            ),
        )

        # TTS-specific tracking
        self._stats = CircuitStats()
        self._state_changed_at = time.time()
        self._last_seen_state_change: float = self._cb.last_state_change_time
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        self._last_health_check = time.time()
        self._health_check_task: Optional[asyncio.Task] = None
        self._error_categories: dict[str, int] = {}

    # -- state property -----------------------------------------------------

    @property
    def _state(self) -> str:
        return _unified_to_str(self._cb.state)

    @property
    def state(self):
        """Return state as a CircuitState-compatible string."""
        return self._state

    @property
    def is_available(self) -> bool:
        return self._cb.can_attempt()

    # -- call ---------------------------------------------------------------

    def _build_open_error(self) -> TTSCircuitOpenError:
        backoff_delay = self._stats.get_backoff_delay(
            self.backoff_base_delay,
            self.backoff_max_delay,
        )
        return TTSCircuitOpenError(
            f"Circuit breaker for {self.provider_name} is open. "
            f"Failed {self._stats.consecutive_failures} times. "
            f"Retry after {backoff_delay:.1f}s.",
            provider=self.provider_name,
            details={
                "consecutive_failures": self._stats.consecutive_failures,
                "backoff_delay": backoff_delay,
                "failure_rate": self._stats.failure_rate,
                "error_categories": self._error_categories.copy(),
            },
        )

    async def _record_tts_success(self) -> None:
        async with self._lock:
            self._stats.record_success()
            current_change = self._cb.last_state_change_time
            if (
                self._cb.state == _UnifiedState.CLOSED
                and current_change != self._last_seen_state_change
            ):
                self._last_seen_state_change = current_change
                self._state_changed_at = time.time()
                logger.info(
                    f"Circuit breaker for {self.provider_name} closed after "
                    f"{self._stats.consecutive_successes} successful calls"
                )

    async def _record_tts_failure(self, error: Exception) -> None:
        async with self._lock:
            self._stats.record_failure()
            error_category = categorize_error(error)
            self._error_categories[error_category] = (
                self._error_categories.get(error_category, 0) + 1
            )
            current_change = self._cb.last_state_change_time
            if (
                self._cb.state == _UnifiedState.OPEN
                and current_change != self._last_seen_state_change
            ):
                self._last_seen_state_change = current_change
                self._state_changed_at = time.time()
                logger.warning(
                    f"Circuit breaker for {self.provider_name} opened after "
                    f"{self._stats.consecutive_failures} failures. "
                    f"Error categories: {self._error_categories}"
                )

    async def guard(self) -> None:
        if self._cb.can_attempt():
            return
        raise self._build_open_error()

    async def record_manual_success(self) -> None:
        self._cb.record_success()
        await self._record_tts_success()

    async def record_manual_failure(self, error: Exception) -> None:
        self._cb.record_failure(error)
        await self._record_tts_failure(error)

    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        from tldw_Server_API.app.core.Infrastructure.circuit_breaker import (
            CircuitBreakerOpenError as _CBOpen,
        )
        try:
            result = await self._cb.call_async(func, *args, **kwargs)
            await self._record_tts_success()
            return result
        except _CBOpen as exc:
            raise self._build_open_error() from exc
        except Exception as e:
            await self._record_tts_failure(e)
            raise

    # -- status / reset -----------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "state": _unified_to_str(self._cb.state),
            "stats": {
                "failure_count": self._stats.failure_count,
                "success_count": self._stats.success_count,
                "failure_rate": round(self._stats.failure_rate, 3),
                "consecutive_failures": self._stats.consecutive_failures,
                "consecutive_successes": self._stats.consecutive_successes,
                "total_requests": self._stats.total_requests,
            },
            "config": {
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "half_open_max_calls": self.half_open_max_calls,
                "success_threshold": self.success_threshold,
            },
        }

    def get_detailed_status(self) -> dict[str, Any]:
        base = self.get_status()
        return {
            **base,
            "stats": {
                **base["stats"],
                "last_failure_time": self._stats.last_failure_time,
                "last_success_time": self._stats.last_success_time,
            },
            "config": {
                **base["config"],
                "backoff_base_delay": self.backoff_base_delay,
                "backoff_max_delay": self.backoff_max_delay,
            },
            "error_analysis": {
                "error_categories": self._error_categories.copy(),
                "backoff_delay": self._stats.get_backoff_delay(
                    self.backoff_base_delay, self.backoff_max_delay,
                ),
            },
            "health_monitoring": {
                "enabled": self._health_check_task is not None and not self._health_check_task.done(),
                "last_health_check": self._last_health_check,
                "health_check_interval": self.health_check_interval,
            },
            "state_info": {
                "state_changed_at": self._cb.last_state_change_time,
                "time_in_state": time.time() - self._cb.last_state_change_time,
                "half_open_calls": None,
            },
        }

    def reset(self):
        self._cb.reset()
        self._stats.reset()
        self._state_changed_at = time.time()
        self._error_categories.clear()
        self._last_health_check = time.time()
        logger.info(f"Circuit breaker for {self.provider_name} manually reset")

    # -- health monitoring --------------------------------------------------

    async def start_health_monitoring(self, health_check_func: Optional[Callable] = None):
        if self._health_check_task and not self._health_check_task.done():
            return
        self._health_check_task = asyncio.create_task(
            self._health_monitor_loop(health_check_func)
        )
        logger.info(f"Started health monitoring for {self.provider_name}")

    async def stop_health_monitoring(self):
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task
        logger.info(f"Stopped health monitoring for {self.provider_name}")

    async def _health_monitor_loop(self, health_check_func: Optional[Callable]):
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                if self._cb.state in (_UnifiedState.OPEN, _UnifiedState.HALF_OPEN):
                    await self._perform_health_check(health_check_func)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in health monitoring for {} ({})",
                    self.provider_name,
                    type(e).__name__,
                )

    async def _perform_health_check(self, health_check_func: Optional[Callable]):
        try:
            self._last_health_check = time.time()
            if health_check_func:
                if asyncio.iscoroutinefunction(health_check_func):
                    is_healthy = await health_check_func()
                else:
                    is_healthy = health_check_func()
                if is_healthy and self._cb.state == _UnifiedState.OPEN:
                    # Let the unified breaker handle the timeout-based transition
                    logger.info(
                        f"Health check passed for {self.provider_name}"
                    )
            else:
                if self._should_attempt_recovery():
                    logger.info(
                        f"Automatic recovery attempt for {self.provider_name}"
                    )
        except Exception as e:
            logger.error(
                "Health check failed for {} ({})",
                self.provider_name,
                type(e).__name__,
            )

    def _should_attempt_recovery(self) -> bool:
        if self._cb.state != _UnifiedState.OPEN:
            return False
        state_changed_at = self._cb.last_state_change_time
        if time.time() - state_changed_at < self.recovery_timeout:
            return False
        network_errors = self._error_categories.get("network", 0)
        total_errors = sum(self._error_categories.values()) or 1
        network_error_ratio = network_errors / total_errors
        if network_error_ratio > 0.7:
            return True
        return time.time() - state_changed_at >= self.recovery_timeout * 2


# Backward compatibility
CircuitOpenError = TTSCircuitOpenError


def build_qwen_runtime_breaker_key(provider_name: str, runtime_name: str) -> str:
    return f"{provider_name}:{runtime_name}"


class CircuitBreakerManager:
    """Manages circuit breakers for all TTS providers."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

        if hasattr(self.config, "has_section") and self.config.has_section("TTS-Settings"):
            try:
                circuit_config = dict(self.config.items("TTS-Settings"))
            except Exception:
                circuit_config = {}
        elif isinstance(self.config, dict):
            circuit_config = self.config
        else:
            circuit_config = {}

        self.default_config = {
            "failure_threshold": circuit_config.get("circuit_failure_threshold", 5),
            "recovery_timeout": float(circuit_config.get("circuit_recovery_timeout", 60.0)),
            "half_open_max_calls": int(circuit_config.get("circuit_half_open_calls", 3)),
            "success_threshold": int(circuit_config.get("circuit_success_threshold", 2)),
        }

    async def get_breaker(self, provider_name: str) -> CircuitBreaker:
        async with self._lock:
            if provider_name not in self._breakers:
                provider_config: dict[str, Any] = {}

                if hasattr(self.config, "has_option"):
                    if self.config.has_section("TTS-Settings"):
                        for key, value in dict(self.config.items("TTS-Settings")).items():
                            if key.startswith(f"{provider_name}_circuit_"):
                                param_name = key.replace(f"{provider_name}_circuit_", "")
                                try:
                                    if param_name in ("failure_threshold", "half_open_max_calls", "success_threshold"):
                                        provider_config[param_name] = int(value)
                                    elif param_name == "recovery_timeout":
                                        provider_config[param_name] = float(value)
                                    else:
                                        provider_config[param_name] = value
                                except ValueError:
                                    pass
                elif isinstance(self.config, dict):
                    provider_config = self.config.get(f"{provider_name}_circuit", {})

                config = {**self.default_config, **provider_config}
                self._breakers[provider_name] = CircuitBreaker(
                    provider_name=provider_name, **config,
                )
                logger.debug(f"Created circuit breaker for {provider_name}")

            return self._breakers[provider_name]

    def get_all_status(self, detailed: bool = False) -> dict[str, Any]:
        if detailed:
            return {n: b.get_detailed_status() for n, b in self._breakers.items()}
        return {n: b.get_status() for n, b in self._breakers.items()}

    async def enable_health_monitoring(
        self,
        provider_name: Optional[str] = None,
        health_check_functions: Optional[dict[str, Callable]] = None,
    ):
        health_check_functions = health_check_functions or {}
        if provider_name:
            if provider_name in self._breakers:
                hcf = health_check_functions.get(provider_name)
                await self._breakers[provider_name].start_health_monitoring(hcf)
        else:
            for name, breaker in self._breakers.items():
                hcf = health_check_functions.get(name)
                await breaker.start_health_monitoring(hcf)

    async def disable_health_monitoring(self, provider_name: Optional[str] = None):
        if provider_name:
            if provider_name in self._breakers:
                await self._breakers[provider_name].stop_health_monitoring()
        else:
            for breaker in self._breakers.values():
                await breaker.stop_health_monitoring()

    def get_error_analysis(self) -> dict[str, Any]:
        analysis: dict[str, Any] = {
            "total_providers": len(self._breakers),
            "providers_by_state": {"closed": 0, "open": 0, "half_open": 0},
            "error_categories": {},
            "failure_patterns": [],
            "recovery_candidates": [],
        }
        for name, breaker in self._breakers.items():
            status = breaker.get_detailed_status()
            state = status["state"]
            analysis["providers_by_state"][state] += 1

            for category, count in status["error_analysis"]["error_categories"].items():
                analysis["error_categories"][category] = (
                    analysis["error_categories"].get(category, 0) + count
                )

            if status["stats"]["consecutive_failures"] >= 3:
                analysis["failure_patterns"].append({
                    "provider": name,
                    "consecutive_failures": status["stats"]["consecutive_failures"],
                    "failure_rate": status["stats"]["failure_rate"],
                    "dominant_error": max(
                        status["error_analysis"]["error_categories"].items(),
                        key=lambda x: x[1],
                        default=("unknown", 0),
                    )[0],
                })

            if state == "open" and breaker._should_attempt_recovery():
                analysis["recovery_candidates"].append({
                    "provider": name,
                    "time_in_open_state": status["state_info"]["time_in_state"],
                    "recovery_timeout": status["config"]["recovery_timeout"],
                })

        return analysis

    def reset_all(self):
        for breaker in self._breakers.values():
            breaker.reset()
        logger.info("All circuit breakers reset")

    async def reset_provider(self, provider_name: str, restart_health_monitoring: bool = True):
        if provider_name in self._breakers:
            breaker = self._breakers[provider_name]
            await breaker.stop_health_monitoring()
            breaker.reset()
            if restart_health_monitoring:
                await breaker.start_health_monitoring()
            logger.info(f"Circuit breaker for {provider_name} reset")

    async def shutdown(self):
        logger.info("Shutting down circuit breaker manager")
        tasks = [b.stop_health_monitoring() for b in self._breakers.values()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Circuit breaker manager shutdown complete")


# Singleton instance
_circuit_manager: Optional[CircuitBreakerManager] = None
_manager_lock = asyncio.Lock()


async def get_circuit_manager(config: Optional[dict[str, Any]] = None) -> CircuitBreakerManager:
    global _circuit_manager
    if _circuit_manager is None:
        async with _manager_lock:
            if _circuit_manager is None:
                _circuit_manager = CircuitBreakerManager(config)
                logger.info("Circuit breaker manager initialized")
    return _circuit_manager

#
# End of circuit_breaker.py
#######################################################################################################################
