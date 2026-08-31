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

    def test_ordinary_revision_key_is_independent_of_semantic_revisions(self):
        ordinary = {
            "user_id": "user",
            "dataset_id": "dataset-1",
            "graph_revision": 7,
            "parser_version": 2,
            "query_params": {
                "center": "note-1",
                "edge_types": ["manual"],
                "max_nodes": 300,
            },
        }
        semantic_before = _semantic_key_values()
        semantic_after = {
            **semantic_before,
            "generation_id": "generation-2",
            "semantic_index_revision": 10,
            "configuration_revision": 6,
        }

        ordinary_before = GraphCache.make_revision_key(**ordinary)
        semantic_key_before = GraphCache.make_semantic_revision_key(**semantic_before)
        semantic_key_after = GraphCache.make_semantic_revision_key(**semantic_after)
        ordinary_after = GraphCache.make_revision_key(**ordinary)

        assert semantic_key_before != semantic_key_after
        assert ordinary_before == ordinary_after

    def test_semantic_final_projection_key_binds_every_immutable_revision(self):
        common = _semantic_key_values()
        original = GraphCache.make_semantic_revision_key(**common)

        for field, value in (
            ("dataset_id", "dataset-2"),
            ("graph_revision", 8),
            ("parser_version", 3),
            ("generation_id", "generation-2"),
            ("semantic_index_revision", 10),
            ("configuration_revision", 6),
            ("compatibility_hash", "compatibility-2"),
            ("model_revision", "model-revision-2"),
            ("normalization_version", "normalization-v2"),
            ("chunker_version", "chunker-v2"),
        ):
            assert original != GraphCache.make_semantic_revision_key(
                **{**common, field: value}
            )

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            (("semantic_threshold",), 0.8),
            (("semantic_top_k",), 11),
            (("center",), "note-2"),
            (("tag",), "tag-2"),
            (("source",), "source-2"),
            (("time_range", "start"), "2026-02-01T00:00:00Z"),
            (("edge_types",), ["manual", "semantic", "wikilink"]),
            (("effective_limits", "max_nodes"), 301),
            (("effective_limits", "max_edges"), 1201),
            (("effective_limits", "max_degree"), 41),
            (("effective_limits", "semantic_candidate_nodes"), 49),
            (("effective_limits", "semantic_candidate_edges"), 49),
        ],
    )
    def test_semantic_key_and_cursor_binding_cover_request_and_effective_caps(
        self,
        path: tuple[str, ...],
        value: object,
    ) -> None:
        common = _semantic_key_values()
        changed_query = _replace_nested(common["query_params"], path, value)

        original_key = GraphCache.make_semantic_revision_key(**common)
        changed_key = GraphCache.make_semantic_revision_key(
            **{**common, "query_params": changed_query}
        )
        original_binding = GraphCache.make_semantic_cursor_binding(
            **_semantic_binding_values(common)
        )
        changed_binding = GraphCache.make_semantic_cursor_binding(
            **_semantic_binding_values({**common, "query_params": changed_query})
        )

        assert changed_key != original_key
        assert changed_binding != original_binding

    def test_mutable_progress_is_not_part_of_stable_semantic_identity(self):
        common = _semantic_key_values()
        progress_before = {
            "dirty_notes": 10,
            "failed_notes": 1,
            "cleanup_pending": True,
            "state": "updating",
        }
        progress_after = {
            "dirty_notes": 2,
            "failed_notes": 0,
            "cleanup_pending": False,
            "state": "ready",
        }

        before = GraphCache.make_semantic_revision_key(**common)
        after = GraphCache.make_semantic_revision_key(**common)

        assert progress_before != progress_after
        assert before == after


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


def _semantic_key_values() -> dict[str, object]:
    return {
        "user_id": "user",
        "dataset_id": "dataset-1",
        "graph_revision": 7,
        "parser_version": 2,
        "generation_id": "generation-1",
        "semantic_index_revision": 9,
        "configuration_revision": 5,
        "compatibility_hash": "compatibility-1",
        "model_revision": "model-revision-1",
        "normalization_version": "normalization-v1",
        "chunker_version": "chunker-v1",
        "query_params": {
            "center": "note-1",
            "edge_types": ["manual", "semantic"],
            "semantic_threshold": 0.75,
            "semantic_top_k": 10,
            "tag": "tag-1",
            "source": "source-1",
            "time_range": {"start": "2026-01-01T00:00:00Z", "end": None},
            "time_range_field": "updated_at",
            "effective_limits": {
                "max_nodes": 300,
                "max_edges": 1_200,
                "max_degree": 40,
                "semantic_candidate_nodes": 50,
                "semantic_candidate_edges": 50,
            },
        },
    }


def _semantic_binding_values(values: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in values.items() if key != "user_id"}


def _replace_nested(
    original: object,
    path: tuple[str, ...],
    value: object,
) -> dict[str, object]:
    assert isinstance(original, dict)
    changed = dict(original)
    cursor = changed
    for part in path[:-1]:
        nested = cursor[part]
        assert isinstance(nested, dict)
        copied = dict(nested)
        cursor[part] = copied
        cursor = copied
    cursor[path[-1]] = value
    return changed
