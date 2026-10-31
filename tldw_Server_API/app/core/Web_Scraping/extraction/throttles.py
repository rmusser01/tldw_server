"""Canonical throttle state for extraction strategies and LLM providers."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from threading import BoundedSemaphore, Condition, Lock
from time import monotonic
from typing import Callable, Protocol


class _Semaphore(Protocol):
    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool: ...

    def release(self) -> None: ...


class _ProviderConcurrencyLimiter:
    """Stable provider limiter whose process-wide ceiling can only tighten."""

    def __init__(self, max_concurrency: int) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self._condition = Condition()
        self._limit = max_concurrency
        self._active = 0

    def tighten(self, max_concurrency: int) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        with self._condition:
            self._limit = min(self._limit, max_concurrency)
            self._condition.notify_all()

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        with self._condition:
            if not blocking:
                if self._active >= self._limit:
                    return False
                self._active += 1
                return True

            deadline = None if timeout is None else monotonic() + max(0.0, timeout)
            while self._active >= self._limit:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
            self._active += 1
            return True

    def release(self) -> None:
        with self._condition:
            if self._active <= 0:
                raise ValueError("Semaphore released too many times")
            self._active -= 1
            self._condition.notify()


_LLM_PROVIDER_LIMITS: dict[str, _ProviderConcurrencyLimiter] = {}
_LLM_PROVIDER_LAST_CALL: dict[str, float] = {}
_STRATEGY_LIMITS: dict[str, tuple[int, BoundedSemaphore]] = {}
_LLM_PROVIDER_LIMITS_LOCK = Lock()
_LLM_PROVIDER_LAST_CALL_LOCK = Lock()
_STRATEGY_LIMITS_LOCK = Lock()
_SEMAPHORE_POLL_INTERVAL_SECONDS = 0.1


def _llm_provider_key(provider: str | None) -> str:
    return (provider or "").strip().lower() or "default"


@contextmanager
def cancellable_semaphore(
    semaphore: _Semaphore,
    cancellation_checkpoint: Callable[[], None],
) -> Iterator[None]:
    """Acquire a semaphore while periodically checking for cancellation."""

    acquired = False
    try:
        cancellation_checkpoint()
        while not semaphore.acquire(timeout=_SEMAPHORE_POLL_INTERVAL_SECONDS):
            cancellation_checkpoint()
        acquired = True
        cancellation_checkpoint()
        yield
    finally:
        if acquired:
            semaphore.release()


def get_strategy_semaphore(strategy: str, max_workers: int | None) -> BoundedSemaphore | None:
    if not max_workers:
        return None
    with _STRATEGY_LIMITS_LOCK:
        current = _STRATEGY_LIMITS.get(strategy)
        if current is None or current[0] != max_workers:
            _STRATEGY_LIMITS[strategy] = (max_workers, BoundedSemaphore(max_workers))
        return _STRATEGY_LIMITS[strategy][1]


def get_llm_semaphore(provider: str | None, max_concurrency: int) -> _ProviderConcurrencyLimiter:
    key = _llm_provider_key(provider)
    with _LLM_PROVIDER_LIMITS_LOCK:
        existing = _LLM_PROVIDER_LIMITS.get(key)
        if existing is None:
            existing = _ProviderConcurrencyLimiter(max_concurrency)
            _LLM_PROVIDER_LIMITS[key] = existing
        else:
            existing.tighten(max_concurrency)
        return existing


def apply_llm_delay(
    provider: str | None,
    delay_ms: float,
    jitter_ms: float,
    *,
    wall_time: Callable[[], float],
    sleep: Callable[[float], None],
) -> None:
    if delay_ms <= 0.0:
        return
    provider_key = _llm_provider_key(provider)
    now = wall_time()
    wait_seconds = 0.0
    with _LLM_PROVIDER_LAST_CALL_LOCK:
        last_call = _LLM_PROVIDER_LAST_CALL.get(provider_key)
        if last_call is None:
            reserved_call = now
        else:
            jitter = random.uniform(0.0, jitter_ms / 1000.0) if jitter_ms > 0.0 else 0.0  # nosec B311
            reserved_call = max(now, last_call + (delay_ms / 1000.0) + jitter)
            wait_seconds = max(0.0, reserved_call - now)
        _LLM_PROVIDER_LAST_CALL[provider_key] = reserved_call
    if wait_seconds > 0.0:
        sleep(wait_seconds)


def get_throttle_stats() -> dict[str, int]:
    with _LLM_PROVIDER_LIMITS_LOCK:
        llm_limits = len(_LLM_PROVIDER_LIMITS)
    with _LLM_PROVIDER_LAST_CALL_LOCK:
        llm_last = len(_LLM_PROVIDER_LAST_CALL)
    with _STRATEGY_LIMITS_LOCK:
        strategy_limits = len(_STRATEGY_LIMITS)
    return {
        "llm_provider_limit_count": llm_limits,
        "llm_provider_last_call_count": llm_last,
        "strategy_limit_count": strategy_limits,
    }


def clear_throttle_state() -> None:
    with _LLM_PROVIDER_LIMITS_LOCK:
        _LLM_PROVIDER_LIMITS.clear()
    with _LLM_PROVIDER_LAST_CALL_LOCK:
        _LLM_PROVIDER_LAST_CALL.clear()
    with _STRATEGY_LIMITS_LOCK:
        _STRATEGY_LIMITS.clear()
