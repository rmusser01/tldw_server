from __future__ import annotations

from tldw_Server_API.app.core.MCP_unified.modules.implementations.web_rate_limit import (
    DomainRateLimiter,
)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_allows_requests_within_limit() -> None:
    limiter = DomainRateLimiter(max_requests=3, window_seconds=10)
    assert [limiter.try_acquire("example.com") for _ in range(3)] == [True, True, True]  # nosec B101


def test_blocks_request_over_limit() -> None:
    limiter = DomainRateLimiter(max_requests=2, window_seconds=10)
    limiter.try_acquire("example.com")
    limiter.try_acquire("example.com")
    assert limiter.try_acquire("example.com") is False  # nosec B101


def test_limit_is_per_domain() -> None:
    limiter = DomainRateLimiter(max_requests=1, window_seconds=10)
    assert limiter.try_acquire("a.example") is True  # nosec B101
    assert limiter.try_acquire("b.example") is True  # nosec B101
    assert limiter.try_acquire("a.example") is False  # nosec B101


def test_window_expiry_allows_again() -> None:
    clock = _FakeClock()
    limiter = DomainRateLimiter(max_requests=1, window_seconds=10, clock=clock)
    assert limiter.try_acquire("example.com") is True  # nosec B101
    assert limiter.try_acquire("example.com") is False  # nosec B101
    clock.now += 11  # advance past the window
    assert limiter.try_acquire("example.com") is True  # nosec B101


def test_domain_key_is_case_insensitive() -> None:
    limiter = DomainRateLimiter(max_requests=1, window_seconds=10)
    assert limiter.try_acquire("Example.COM") is True  # nosec B101
    assert limiter.try_acquire("example.com") is False  # nosec B101


def test_disabled_when_max_requests_non_positive() -> None:
    limiter = DomainRateLimiter(max_requests=0)
    assert limiter.enabled is False  # nosec B101
    assert all(limiter.try_acquire("example.com") for _ in range(100))  # nosec B101


def test_disabled_when_max_requests_none() -> None:
    limiter = DomainRateLimiter(max_requests=None)
    assert limiter.enabled is False  # nosec B101
    assert limiter.try_acquire("example.com") is True  # nosec B101
