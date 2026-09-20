"""Tests for GraphCache."""

import threading
import time

import pytest

from tldw_Server_API.app.core.Notes_Graph.graph_cache import GraphCache

pytestmark = pytest.mark.unit


class TestGraphCache:
    """Core cache behaviour tests."""

    def test_put_get_roundtrip(self):
        cache = GraphCache(ttl_seconds=60, max_keys=100)
        cache.put("k1", {"data": 42})
        assert cache.get("k1") == {"data": 42}

    def test_cache_miss(self):
        cache = GraphCache(ttl_seconds=60, max_keys=100)
        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        cache = GraphCache(ttl_seconds=0, max_keys=100)  # instant expiry
        cache.put("k1", "value")
        # Allow a tiny bit of time to pass
        time.sleep(0.01)
        assert cache.get("k1") is None

    def test_max_key_eviction(self):
        cache = GraphCache(ttl_seconds=60, max_keys=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("d") == 4

    def test_overwrite_existing_key(self):
        cache = GraphCache(ttl_seconds=60, max_keys=100)
        cache.put("k1", "old")
        cache.put("k1", "new")
        assert cache.get("k1") == "new"

    def test_stats(self):
        cache = GraphCache(ttl_seconds=60, max_keys=100)
        cache.put("k1", 1)
        cache.get("k1")       # hit
        cache.get("missing")  # miss
        s = cache.stats()
        assert s["size"] == 1
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["ttl_seconds"] == 60
        assert s["max_keys"] == 100


class TestMakeCacheKey:
    """Tests for deterministic key generation."""

    def test_deterministic(self):
        k1 = GraphCache.make_cache_key("user1", {"radius": 1, "center": "abc"})
        k2 = GraphCache.make_cache_key("user1", {"center": "abc", "radius": 1})
        assert k1 == k2

    def test_different_users_different_keys(self):
        k1 = GraphCache.make_cache_key("user1", {"radius": 1})
        k2 = GraphCache.make_cache_key("user2", {"radius": 1})
        assert k1 != k2

    def test_different_params_different_keys(self):
        k1 = GraphCache.make_cache_key("u", {"radius": 1})
        k2 = GraphCache.make_cache_key("u", {"radius": 2})
        assert k1 != k2

    def test_key_length(self):
        k = GraphCache.make_cache_key("user", {"a": 1})
        assert len(k) == 32

    def test_dataset_revision_and_parser_version_are_part_of_the_key(self):
        common = {
            "user_id": "user",
            "dataset_id": "dataset-1",
            "graph_revision": 7,
            "parser_version": 2,
            "query_params": {"radius": 1},
        }
        original = GraphCache.make_revision_key(**common)
        assert original != GraphCache.make_revision_key(
            **{**common, "dataset_id": "dataset-2"}
        )
        assert original != GraphCache.make_revision_key(
            **{**common, "graph_revision": 8}
        )
        assert original != GraphCache.make_revision_key(
            **{**common, "parser_version": 3}
        )


class TestThreadSafety:
    """Basic thread safety smoke test."""

    def test_concurrent_access(self):
        cache = GraphCache(ttl_seconds=60, max_keys=1000)
        errors: list[Exception] = []

        def writer(start: int):
            try:
                for i in range(100):
                    cache.put(f"key-{start + i}", i)
            except Exception as e:
                errors.append(e)

        def reader(start: int):
            try:
                for i in range(100):
                    cache.get(f"key-{start + i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(100,)),
            threading.Thread(target=reader, args=(0,)),
            threading.Thread(target=reader, args=(50,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == []
        s = cache.stats()
        assert s["size"] <= 1000
