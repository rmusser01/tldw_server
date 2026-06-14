from __future__ import annotations

from tldw_Server_API.app.core.MCP_unified.modules.implementations.web_cache import (
    ResponseCache,
    make_cache_key,
)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_put_then_get_returns_value() -> None:
    cache = ResponseCache(ttl_seconds=100, max_entries=8)
    cache.put(make_cache_key("https://example.com", "markdown", 1000, False), {"ok": True})
    assert cache.get(make_cache_key("https://example.com", "markdown", 1000, False)) == {"ok": True}  # nosec B101


def test_missing_key_returns_none() -> None:
    cache = ResponseCache(ttl_seconds=100, max_entries=8)
    assert cache.get(make_cache_key("https://example.com", "markdown", 1000, False)) is None  # nosec B101


def test_key_distinguishes_format_and_max_bytes() -> None:
    cache = ResponseCache(ttl_seconds=100, max_entries=8)
    cache.put(make_cache_key("https://example.com", "markdown", 1000, False), {"v": 1})
    assert cache.get(make_cache_key("https://example.com", "text", 1000, False)) is None  # nosec B101
    assert cache.get(make_cache_key("https://example.com", "markdown", 2000, False)) is None  # nosec B101


def test_entry_expires_after_ttl() -> None:
    clock = _FakeClock()
    cache = ResponseCache(ttl_seconds=100, max_entries=8, clock=clock)
    key = make_cache_key("https://example.com", "markdown", 1000, False)
    cache.put(key, {"ok": True})
    clock.now += 101
    assert cache.get(key) is None  # nosec B101


def test_lru_eviction_when_over_capacity() -> None:
    cache = ResponseCache(ttl_seconds=1000, max_entries=2)
    a = make_cache_key("https://a.example", "markdown", 1, False)
    b = make_cache_key("https://b.example", "markdown", 1, False)
    c = make_cache_key("https://c.example", "markdown", 1, False)
    cache.put(a, 1)
    cache.put(b, 2)
    cache.get(a)  # touch a -> most recently used; b is now LRU
    cache.put(c, 3)  # evicts b
    assert cache.get(a) == 1  # nosec B101
    assert cache.get(b) is None  # nosec B101
    assert cache.get(c) == 3  # nosec B101


def test_disabled_cache_never_stores() -> None:
    cache = ResponseCache(ttl_seconds=0)
    assert cache.enabled is False  # nosec B101
    key = make_cache_key("https://example.com", "markdown", 1000, False)
    cache.put(key, {"ok": True})
    assert cache.get(key) is None  # nosec B101


def test_respect_robots_is_part_of_key() -> None:
    cache = ResponseCache(ttl_seconds=100, max_entries=8)
    cache.put(make_cache_key("https://example.com", "markdown", 1000, False), {"v": 1})
    # A different respect_robots mode must not collide.
    assert cache.get(make_cache_key("https://example.com", "markdown", 1000, True)) is None  # nosec B101


def test_returned_value_is_isolated_copy() -> None:
    cache = ResponseCache(ttl_seconds=100, max_entries=8)
    key = make_cache_key("https://example.com", "markdown", 1000, False)
    cache.put(key, {"nested": {"x": 1}})
    got = cache.get(key)
    got["nested"]["x"] = 999  # mutate the returned copy
    again = cache.get(key)
    assert again["nested"]["x"] == 1  # nosec B101 - cache entry unaffected
