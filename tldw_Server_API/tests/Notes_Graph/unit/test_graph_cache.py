"""Tests for GraphCache."""

import threading
import time
from dataclasses import FrozenInstanceError

import pytest

from tldw_Server_API.app.api.v1.schemas.notes_graph import EdgeType, NoteGraphRequest
from tldw_Server_API.app.core.Notes_Graph import graph_cache as graph_cache_module
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
        cache.get("k1")  # hit
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
        assert original != GraphCache.make_revision_key(**{**common, "dataset_id": "dataset-2"})
        assert original != GraphCache.make_revision_key(**{**common, "graph_revision": 8})
        assert original != GraphCache.make_revision_key(**{**common, "parser_version": 3})

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
            ("capability_revision", "capability-2"),
            ("disclosure_hash", "disclosure-2"),
            ("compatibility_hash", "compatibility-2"),
            ("provider", "google"),
            ("model", "embedding-model-2"),
            ("model_revision", "model-revision-2"),
            ("endpoint_origin_revision", "endpoint-2"),
            ("normalization_version", "normalization-v2"),
            ("chunker_version", "chunker-v2"),
        ):
            assert original != GraphCache.make_semantic_revision_key(**{**common, field: value})

    def test_semantic_key_helpers_remain_compatible_with_legacy_callers(self):
        legacy = _semantic_key_values()
        for field in (
            "capability_revision",
            "disclosure_hash",
            "provider",
            "model",
            "endpoint_origin_revision",
        ):
            legacy.pop(field)

        revision_key = GraphCache.make_semantic_revision_key(**legacy)
        cursor_binding = GraphCache.make_semantic_cursor_binding(**_semantic_binding_values(legacy))

        assert revision_key
        assert cursor_binding

    @pytest.mark.parametrize(
        ("request_updates", "identity_updates"),
        [
            ({}, {"semantic_threshold": 0.8}),
            ({}, {"semantic_top_k": 11}),
            ({"center_note_id": "note-2"}, {}),
            ({"radius": 2}, {}),
            ({"tag": "tag-2"}, {}),
            ({"source": "source-2"}, {}),
            (
                {
                    "time_range": {
                        "start": "2026-02-01T00:00:00Z",
                        "end": None,
                    }
                },
                {},
            ),
            ({"time_range_field": "created_at"}, {}),
            ({"edge_types": [EdgeType.manual, EdgeType.semantic, EdgeType.wikilink]}, {}),
            ({}, {"max_nodes": 301}),
            ({}, {"max_edges": 1_201}),
            ({}, {"max_degree": 41}),
            ({}, {"semantic_candidate_nodes": 49}),
            ({}, {"semantic_candidate_edges": 49}),
            ({"allow_heavy": True}, {"allow_heavy": True}),
        ],
    )
    def test_semantic_key_and_cursor_binding_cover_request_and_effective_caps(
        self,
        request_updates: dict[str, object],
        identity_updates: dict[str, object],
    ) -> None:
        common = _semantic_key_values()
        changed_identity = _semantic_query_identity(
            request_updates=request_updates,
            **identity_updates,
        )

        original_key = GraphCache.make_semantic_revision_key(**common)
        changed_key = GraphCache.make_semantic_revision_key(**{**common, "query_identity": changed_identity})
        original_binding = GraphCache.make_semantic_cursor_binding(**_semantic_binding_values(common))
        changed_binding = GraphCache.make_semantic_cursor_binding(
            **_semantic_binding_values({**common, "query_identity": changed_identity})
        )

        assert changed_key != original_key
        assert changed_binding != original_binding

    def test_semantic_query_identity_is_canonical_and_immutable(self):
        identity = _semantic_query_identity(
            request_updates={
                "edge_types": [
                    EdgeType.semantic,
                    EdgeType.manual,
                    EdgeType.semantic,
                ]
            }
        )
        canonical = _semantic_query_identity(request_updates={"edge_types": [EdgeType.manual, EdgeType.semantic]})

        assert identity == canonical
        with pytest.raises(FrozenInstanceError):
            identity.radius = 2

    @pytest.mark.parametrize(
        "request_update",
        [
            {"max_nodes": 300},
            {"max_edges": 1_200},
            {"max_degree": 40},
            {"allow_heavy": True},
        ],
    )
    def test_each_requested_cap_binds_beside_identical_effective_clamps(
        self,
        request_update: dict[str, object],
    ) -> None:
        effective = {
            "max_nodes": 200,
            "max_edges": 800,
            "max_degree": 20,
            "allow_heavy": False,
        }
        default_request = _semantic_query_identity(
            request_updates={"radius": 2},
            **effective,
        )
        explicitly_bounded_request = _semantic_query_identity(
            request_updates={"radius": 2, **request_update},
            **effective,
        )

        assert default_request != explicitly_bounded_request
        common = _semantic_key_values()
        assert GraphCache.make_semantic_revision_key(
            **{**common, "query_identity": default_request}
        ) != GraphCache.make_semantic_revision_key(**{**common, "query_identity": explicitly_bounded_request})
        assert GraphCache.make_semantic_cursor_binding(
            **_semantic_binding_values({**common, "query_identity": default_request})
        ) != GraphCache.make_semantic_cursor_binding(
            **_semantic_binding_values({**common, "query_identity": explicitly_bounded_request})
        )

    def test_semantic_helpers_reject_untyped_query_identity(self):
        common = _semantic_key_values()

        with pytest.raises(TypeError, match="SemanticGraphQueryIdentity"):
            GraphCache.make_semantic_revision_key(
                **{
                    **common,
                    "query_identity": {"dirty_notes": 10, "state": "updating"},
                }
            )

    def test_semantic_query_factory_rejects_mutable_progress_fields(self):
        request = _semantic_request()

        with pytest.raises(TypeError, match="unexpected keyword"):
            graph_cache_module.SemanticGraphQueryIdentity.from_request(
                request,
                semantic_threshold=0.75,
                semantic_top_k=10,
                max_nodes=300,
                max_edges=1_200,
                max_degree=40,
                semantic_candidate_nodes=50,
                semantic_candidate_edges=50,
                allow_heavy=False,
                dirty_notes=10,
            )

    @pytest.mark.parametrize(
        ("updates", "message"),
        [
            ({"semantic_threshold": float("nan")}, "threshold"),
            ({"semantic_threshold": 1.01}, "threshold"),
            ({"semantic_top_k": True}, "top_k"),
            ({"semantic_top_k": 0}, "top_k"),
            ({"max_nodes": 0}, "max_nodes"),
            ({"max_edges": -1}, "max_edges"),
            ({"max_degree": 0}, "max_degree"),
            ({"semantic_candidate_nodes": 51}, "candidate_nodes"),
            ({"semantic_candidate_edges": 51}, "candidate_edges"),
            ({"allow_heavy": 1}, "allow_heavy"),
        ],
    )
    def test_semantic_query_identity_rejects_noncanonical_effective_values(
        self,
        updates: dict[str, object],
        message: str,
    ) -> None:
        with pytest.raises((TypeError, ValueError), match=message):
            _semantic_query_identity(**updates)

    @pytest.mark.parametrize(
        ("request_updates", "identity_updates"),
        [
            ({"semantic_threshold": 0.7}, {"semantic_threshold": 0.8}),
            ({"semantic_top_k": 9}, {"semantic_top_k": 10}),
        ],
    )
    def test_semantic_query_identity_rejects_effective_request_mismatch(
        self,
        request_updates: dict[str, object],
        identity_updates: dict[str, object],
    ) -> None:
        with pytest.raises(ValueError, match="does not match"):
            _semantic_query_identity(
                request_updates=request_updates,
                **identity_updates,
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


def _semantic_key_values() -> dict[str, object]:
    return {
        "user_id": "user",
        "dataset_id": "dataset-1",
        "graph_revision": 7,
        "parser_version": 2,
        "generation_id": "generation-1",
        "semantic_index_revision": 9,
        "configuration_revision": 5,
        "capability_revision": "capability-1",
        "disclosure_hash": "disclosure-1",
        "compatibility_hash": "compatibility-1",
        "provider": "openai",
        "model": "embedding-model-1",
        "model_revision": "model-revision-1",
        "endpoint_origin_revision": "endpoint-1",
        "normalization_version": "normalization-v1",
        "chunker_version": "chunker-v1",
        "query_identity": _semantic_query_identity(),
    }


def _semantic_binding_values(values: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in values.items() if key != "user_id"}


def _semantic_request(**updates: object) -> NoteGraphRequest:
    values: dict[str, object] = {
        "center_note_id": "note-1",
        "radius": 1,
        "edge_types": [EdgeType.manual, EdgeType.semantic],
        "tag": "tag-1",
        "source": "source-1",
        "time_range": {"start": "2026-01-01T00:00:00Z", "end": None},
        "time_range_field": "updated_at",
    }
    values.update(updates)
    return NoteGraphRequest.model_validate(values)


def _semantic_query_identity(
    *,
    request_updates: dict[str, object] | None = None,
    semantic_threshold: float = 0.75,
    semantic_top_k: int = 10,
    max_nodes: int = 300,
    max_edges: int = 1_200,
    max_degree: int = 40,
    semantic_candidate_nodes: int = 50,
    semantic_candidate_edges: int = 50,
    allow_heavy: bool = False,
):
    request = _semantic_request(**(request_updates or {}))
    return graph_cache_module.SemanticGraphQueryIdentity.from_request(
        request,
        semantic_threshold=semantic_threshold,
        semantic_top_k=semantic_top_k,
        max_nodes=max_nodes,
        max_edges=max_edges,
        max_degree=max_degree,
        semantic_candidate_nodes=semantic_candidate_nodes,
        semantic_candidate_edges=semantic_candidate_edges,
        allow_heavy=allow_heavy,
    )
