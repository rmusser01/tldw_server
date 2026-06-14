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


def test_active_domain_survives_lru_eviction() -> None:
    # Cap tracking at 2 domains. Keep "active" recently-used while churning others;
    # it must not be evicted (which would reset its history and let it bypass).
    limiter = DomainRateLimiter(max_requests=1, window_seconds=1000, max_tracked_domains=2)
    assert limiter.try_acquire("active.example") is True  # consume active's single token
    for i in range(5):
        limiter.try_acquire("active.example")  # refresh recency (already blocked, but touches LRU)
        limiter.try_acquire(f"churn-{i}.example")  # evicts the least-recently-used (a churn host)
    # active.example was kept, so its budget is still exhausted.
    assert limiter.try_acquire("active.example") is False  # nosec B101


def test_least_recently_used_domain_is_evicted() -> None:
    limiter = DomainRateLimiter(max_requests=1, window_seconds=1000, max_tracked_domains=1)
    assert limiter.try_acquire("first.example") is True
    assert limiter.try_acquire("first.example") is False  # budget used
    # Tracking a second host evicts "first" (LRU); its history resets.
    limiter.try_acquire("second.example")
    assert limiter.try_acquire("first.example") is True  # nosec B101 - history was evicted
