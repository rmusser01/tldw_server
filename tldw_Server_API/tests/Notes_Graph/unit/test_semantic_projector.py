"""Authority, evidence, budget, failure, and cache tests for semantic projection."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.schemas.notes_graph import (
    EdgeType,
    GraphEdge,
    GraphLimits,
    GraphNode,
    NoteGraphRequest,
    NoteGraphResponse,
    SemanticEdgeEvidence,
    SemanticExcerpt,
    SemanticExcerptPair,
)
from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticProjectionChunk,
)
from tldw_Server_API.app.core.Notes_Graph.graph_cache import GraphCache
from tldw_Server_API.app.core.Notes_Graph.graph_service import (
    SemanticGraphCandidateResult,
    _encode_cursor,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_content import (
    build_semantic_chunks,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_projector import (
    SemanticGraphProjector,
    SemanticProjectionError,
    _rank_bounded_candidates,
    bound_semantic_evidence,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_scoring import (
    SemanticChunkCandidate,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_settings import SemanticIndexSettings
from tldw_Server_API.app.core.Notes_Graph.semantic_vectors import (
    SemanticVector,
    SemanticVectorMatch,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 29, tzinfo=timezone.utc)


def _projection_chunk(
    *,
    generation_id: str,
    note_id: str,
    title: str,
    content: str,
    content_version: int = 1,
) -> SemanticProjectionChunk:
    chunk = build_semantic_chunks(
        generation_id=generation_id,
        note_id=note_id,
        title=title,
        content=content,
        content_version=content_version,
    )[0]
    return SemanticProjectionChunk(
        owner_user_id="owner-a",
        dataset_id="dataset-a",
        generation_id=generation_id,
        vector_id=chunk.vector_id,
        note_id=note_id,
        content_version=content_version,
        content_fingerprint=chunk.content_fingerprint,
        title=title,
        content=content,
        created_at=NOW,
        updated_at=NOW,
        ordinal=chunk.ordinal,
        field=chunk.field,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
        chunk_fingerprint=chunk.chunk_fingerprint,
        normalization_version=chunk.normalization_version,
        chunker_version=chunk.chunker_version,
    )


class _Store:
    owner_user_id = "owner-a"

    def __init__(
        self,
        source: SemanticProjectionChunk,
        target: SemanticProjectionChunk,
    ) -> None:
        self.records = {source.vector_id: source, target.vector_id: target}
        self.load_calls: list[tuple[str, ...]] = []
        self.filter_calls: list[dict[str, object]] = []
        self.drop_target_after_io = False
        self.indexed_notes = 2
        self.pending_notes = 0
        self.config = SimpleNamespace(
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            desired_state=SimpleNamespace(value="enabled"),
            configuration_revision=7,
            semantic_index_revision=11,
            compatibility_hash="sha256:" + "a" * 64,
            provider="openai",
            model="text-embedding-3-small",
            model_revision="model-r1",
            vector_backend="chromadb",
            metric="cosine",
            dimension_state=SimpleNamespace(value="resolved"),
            dimensions=3,
            normalization_version=source.normalization_version,
            chunker_version=source.chunker_version,
            active_generation_id=source.generation_id,
        )
        self.generation = SimpleNamespace(
            id=source.generation_id,
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            configuration_revision=7,
            state=SimpleNamespace(value="active"),
            compatibility_hash=self.config.compatibility_hash,
            model_revision="model-r1",
            dimension_state=SimpleNamespace(value="resolved"),
            dimensions=3,
        )

    def get_configuration(self, dataset_id: str):
        assert dataset_id == "dataset-a"
        return self.config

    def get_generation(self, dataset_id: str, generation_id: str):
        assert (dataset_id, generation_id) == ("dataset-a", "generation-a")
        return self.generation

    def get_generation_integrity(self, dataset_id: str, generation_id: str):
        assert (dataset_id, generation_id) == ("dataset-a", "generation-a")
        return SimpleNamespace(
            indexed_note_count=self.indexed_notes,
            excluded_note_count=0,
            failed_note_count=0,
            pending_note_count=self.pending_notes,
            published_chunk_count=len(self.records),
        )

    def load_projection_chunks(
        self,
        *,
        dataset_id: str,
        generation_id: str,
        vector_ids: tuple[str, ...],
    ) -> tuple[SemanticProjectionChunk, ...]:
        assert (dataset_id, generation_id) == ("dataset-a", "generation-a")
        ids = tuple(vector_ids)
        self.load_calls.append(ids)
        records = [self.records[item] for item in ids if item in self.records]
        if self.drop_target_after_io and len(self.load_calls) > 1:
            records = [record for record in records if record.note_id != "target-note"]
        return tuple(records)

    def filter_projection_note_ids(self, **kwargs: object) -> frozenset[str]:
        self.filter_calls.append(dict(kwargs))
        return frozenset(str(value) for value in kwargs["note_ids"])

    def list_visible_vector_ids(
        self,
        dataset_id: str,
        generation_id: str,
        note_id: str | None = None,
    ) -> tuple[str, ...]:
        assert (dataset_id, generation_id) == ("dataset-a", "generation-a")
        return tuple(
            chunk.vector_id
            for chunk in sorted(
                self.records.values(),
                key=lambda item: (item.note_id, item.ordinal, item.vector_id),
            )
            if note_id is None or chunk.note_id == note_id
        )


class _VectorStore:
    def __init__(self, source_id: str, target_id: str) -> None:
        self.source_id = source_id
        self.target_id = target_id
        self.fetch_calls: list[tuple[str, ...]] = []
        self.query_calls: list[tuple[int, int]] = []
        self.fail_query = False
        self.distance = 0.1

    async def fetch(self, dataset_id: str, generation_id: str, vector_ids):
        assert (dataset_id, generation_id) == ("dataset-a", "generation-a")
        ids = tuple(vector_ids)
        self.fetch_calls.append(ids)
        return tuple(
            SemanticVector(vector_id=item, embedding=(1.0, 0.0, 0.0)) for item in ids if item == self.source_id
        )

    async def query(
        self,
        dataset_id: str,
        generation_id: str,
        query_vectors,
        *,
        limit: int,
    ):
        assert (dataset_id, generation_id) == ("dataset-a", "generation-a")
        vectors = tuple(query_vectors)
        self.query_calls.append((len(vectors), limit))
        if self.fail_query:
            raise RuntimeError("backend detail must not escape")
        return tuple((SemanticVectorMatch(vector_id=self.target_id, distance=self.distance),) for _ in vectors)


class _ProjectionStore:
    def __init__(self) -> None:
        self.revision = 19

    def get_revision(self) -> int:
        return self.revision

    def get_projection_status(self):
        return SimpleNamespace(parser_version=3)


class _DB:
    def __init__(self, store: _Store) -> None:
        self.note_semantic_store = store
        self.note_graph_projection_store = _ProjectionStore()
        self.manual_edges: list[dict[str, object]] = []

    def count_user_notes(self, include_deleted: bool = False) -> int:
        assert include_deleted is False
        return 2

    def get_manual_edges_for_notes(
        self,
        user_id: str,
        note_ids: list[str],
    ) -> list[dict[str, object]]:
        assert user_id == "owner-a"
        wanted = set(note_ids)
        return [edge for edge in self.manual_edges if edge["from_note_id"] in wanted or edge["to_note_id"] in wanted]


class _GraphService:
    def __init__(self, ordinary: NoteGraphResponse) -> None:
        self.calls: list[tuple[int, int]] = []
        self.ordinary = ordinary
        self.on_generate = None

    def generate_semantic_candidates(
        self,
        request: NoteGraphRequest,
        *,
        additional_nodes: int,
        additional_edges: int,
    ) -> SemanticGraphCandidateResult:
        assert request.semantic_requested
        self.calls.append((additional_nodes, additional_edges))
        if self.on_generate is not None:
            self.on_generate()
        return SemanticGraphCandidateResult(
            public_graph=self.ordinary,
            candidate_nodes=tuple(self.ordinary.nodes),
            candidate_edges=tuple(self.ordinary.edges),
            candidate_limits=self.ordinary.limits,
        )


def _ordinary_response(*, cursor: str | None = None) -> NoteGraphResponse:
    return NoteGraphResponse(
        nodes=[GraphNode(id="focus-note", type="note", label="Focus", degree=0)],
        edges=[],
        cursor=cursor,
        has_more=cursor is not None,
        limits=GraphLimits(max_nodes=10, max_edges=10, max_degree=4),
        active_note_count=2,
        all_notes_note_cap=10,
        all_notes_eligible=True,
    )


def _capabilities(store: _Store):
    return SimpleNamespace(
        compatibility_hash=store.config.compatibility_hash,
        indexing_available=True,
        unavailable_reason=None,
        provider_label="OpenAI",
        model=store.config.model,
        model_revision=store.config.model_revision,
    )


def _projector(
    *,
    store: _Store,
    vectors: _VectorStore,
    ordinary: NoteGraphResponse,
    cache: GraphCache | None = None,
    settings: SemanticIndexSettings | None = None,
) -> tuple[SemanticGraphProjector, _GraphService, list[int]]:
    graph_service = _GraphService(ordinary)
    factory_calls: list[int] = []

    async def vector_factory():
        factory_calls.append(1)
        return vectors

    return (
        SemanticGraphProjector(
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            db=_DB(store),
            graph_service=graph_service,
            cache=cache,
            vector_store_factory=vector_factory,
            capability_resolver=lambda: _capabilities(store),
            **({} if settings is None else {"settings": settings}),
        ),
        graph_service,
        factory_calls,
    )


@pytest.fixture()
def projection_parts():
    source = _projection_chunk(
        generation_id="generation-a",
        note_id="focus-note",
        title="Focus",
        content="alpha beta gamma",
    )
    target = _projection_chunk(
        generation_id="generation-a",
        note_id="target-note",
        title="Target",
        content="alpha beta delta",
    )
    store = _Store(source, target)
    vectors = _VectorStore(source.vector_id, target.vector_id)
    return source, target, store, vectors


@pytest.mark.asyncio
async def test_projector_queries_once_revalidates_after_io_and_builds_current_evidence(
    projection_parts,
) -> None:
    source, target, store, vectors = projection_parts
    ordinary = _ordinary_response()
    projector, graph_service, factory_calls = _projector(
        store=store,
        vectors=vectors,
        ordinary=ordinary,
    )
    request = NoteGraphRequest(
        center_note_id="focus-note",
        edge_types=[EdgeType.semantic],
        semantic_top_k=2,
        semantic_threshold=0.75,
        tag="tag:research",
        source="source:web",
    )

    result = await projector.project(request, ordinary, user=SimpleNamespace(id_str="owner-a"))

    semantic_edges = [edge for edge in result.edges if edge.type is EdgeType.semantic]
    assert len(semantic_edges) == 1
    assert semantic_edges[0].weight == 0.9
    assert semantic_edges[0].evidence is not None
    assert semantic_edges[0].evidence.excerpt_pairs[0].source.text == "alpha beta gamma"
    assert semantic_edges[0].evidence.excerpt_pairs[0].target.text == "alpha beta delta"
    assert semantic_edges[0].evidence.source_content_version == source.content_version
    assert semantic_edges[0].evidence.target_content_version == target.content_version
    assert [node.id for node in result.nodes] == ["focus-note", "target-note"]
    assert vectors.fetch_calls == [(source.vector_id,)]
    assert vectors.query_calls == [(1, 2)]
    assert graph_service.calls == [(2, 2)]
    assert factory_calls == [1]
    assert len(store.load_calls) >= 2
    assert store.filter_calls[0]["tag"] == "tag:research"
    assert store.filter_calls[0]["source"] == "source:web"


@pytest.mark.asyncio
async def test_missing_focus_and_later_page_do_not_query_or_expand_neighbors(
    projection_parts,
) -> None:
    source, _target, store, vectors = projection_parts
    ordinary_cursor = _encode_cursor(
        0,
        1,
        "focus-note",
        dataset_hash="dataset-hash",
        graph_revision=19,
        parser_version=3,
        request_hash="request-hash",
    )
    ordinary = _ordinary_response(cursor=ordinary_cursor)
    projector, graph_service, factory_calls = _projector(
        store=store,
        vectors=vectors,
        ordinary=ordinary,
    )

    missing_focus = await projector.project(
        NoteGraphRequest(edge_types=[EdgeType.semantic]),
        _ordinary_response(),
        user=SimpleNamespace(id_str="owner-a"),
    )
    assert missing_focus.semantic_status is not None
    assert missing_focus.semantic_status.state == "focus_required"
    assert vectors.query_calls == []

    first = await projector.project(
        NoteGraphRequest(
            center_note_id="focus-note",
            edge_types=[EdgeType.semantic],
            semantic_top_k=2,
            semantic_threshold=0.75,
        ),
        ordinary,
        user=SimpleNamespace(id_str="owner-a"),
    )
    assert first.cursor is not None
    calls_after_first = list(vectors.query_calls)
    service_calls_after_first = list(graph_service.calls)

    continuation = await projector.project(
        NoteGraphRequest(
            center_note_id="focus-note",
            edge_types=[EdgeType.semantic],
            semantic_top_k=2,
            semantic_threshold=0.75,
            cursor=first.cursor,
        ),
        _ordinary_response(),
        user=SimpleNamespace(id_str="owner-a"),
    )

    assert continuation.semantic_status is not None
    assert continuation.semantic_status.available is True
    assert vectors.query_calls == calls_after_first
    assert graph_service.calls == service_calls_after_first
    assert factory_calls == [1]
    assert source.vector_id in vectors.fetch_calls[0]


@pytest.mark.asyncio
async def test_post_io_stale_or_foreign_manifest_result_is_silently_excluded(
    projection_parts,
) -> None:
    _source, _target, store, vectors = projection_parts
    store.drop_target_after_io = True
    ordinary = _ordinary_response()
    projector, _service, _factory_calls = _projector(
        store=store,
        vectors=vectors,
        ordinary=ordinary,
    )

    result = await projector.project(
        NoteGraphRequest(
            center_note_id="focus-note",
            edge_types=[EdgeType.semantic],
        ),
        ordinary,
        user=SimpleNamespace(id_str="owner-a"),
    )

    assert all(edge.type is not EdgeType.semantic for edge in result.edges)
    assert all(node.id != "target-note" for node in result.nodes)


@pytest.mark.asyncio
async def test_transient_vector_failure_fails_open_and_is_not_cached(
    projection_parts,
) -> None:
    _source, _target, store, vectors = projection_parts
    vectors.fail_query = True
    cache = GraphCache(ttl_seconds=60, max_keys=10)
    ordinary = _ordinary_response()
    projector, _service, factory_calls = _projector(
        store=store,
        vectors=vectors,
        ordinary=ordinary,
        cache=cache,
    )
    request = NoteGraphRequest(
        center_note_id="focus-note",
        edge_types=[EdgeType.semantic],
    )

    failed = await projector.project(request, ordinary, user=SimpleNamespace(id_str="owner-a"))
    assert failed.nodes == ordinary.nodes
    assert failed.semantic_status is not None
    assert failed.semantic_status.available is False
    assert failed.semantic_status.detail_reason == "vector_unavailable"
    assert failed.semantic_status.generation_id == "generation-a"
    assert failed.semantic_status.semantic_index_revision == 11
    assert failed.semantic_status.indexed_notes == 2
    assert cache.stats()["size"] == 0

    vectors.fail_query = False
    recovered = await projector.project(request, ordinary, user=SimpleNamespace(id_str="owner-a"))
    assert any(edge.type is EdgeType.semantic for edge in recovered.edges)
    assert len(factory_calls) == 2
    assert cache.stats()["size"] == 1


@pytest.mark.asyncio
async def test_stable_cache_reuses_projection_but_injects_fresh_status(
    projection_parts,
) -> None:
    _source, _target, store, vectors = projection_parts
    cache = GraphCache(ttl_seconds=60, max_keys=10)
    ordinary = _ordinary_response()
    projector, _service, factory_calls = _projector(
        store=store,
        vectors=vectors,
        ordinary=ordinary,
        cache=cache,
    )
    request = NoteGraphRequest(
        center_note_id="focus-note",
        edge_types=[EdgeType.semantic],
        semantic_top_k=2,
        semantic_threshold=0.75,
    )

    first = await projector.project(request, ordinary, user=SimpleNamespace(id_str="owner-a"))
    store.indexed_notes = 3
    second = await projector.project(request, ordinary, user=SimpleNamespace(id_str="owner-a"))

    assert first.semantic_status is not None
    assert second.semantic_status is not None
    assert first.semantic_status.indexed_notes == 2
    assert second.semantic_status.indexed_notes == 3
    assert len(vectors.query_calls) == 1
    assert factory_calls == [1]


@pytest.mark.asyncio
async def test_cache_hit_revalidates_semantic_binding_before_return(
    projection_parts,
    monkeypatch,
) -> None:
    _source, _target, store, vectors = projection_parts
    cache = GraphCache(ttl_seconds=60, max_keys=10)
    ordinary = _ordinary_response()
    projector, _service, _factory_calls = _projector(
        store=store,
        vectors=vectors,
        ordinary=ordinary,
        cache=cache,
    )
    request = NoteGraphRequest(
        center_note_id="focus-note",
        edge_types=[EdgeType.semantic],
    )
    populated = await projector.project(
        request,
        ordinary,
        user=SimpleNamespace(id_str="owner-a"),
    )
    assert any(edge.type is EdgeType.semantic for edge in populated.edges)
    original_get = cache.get

    def get_and_change_binding(key: str):
        cached = original_get(key)
        store.config.semantic_index_revision += 1
        return cached

    monkeypatch.setattr(cache, "get", get_and_change_binding)

    result = await projector.project(
        request,
        ordinary,
        user=SimpleNamespace(id_str="owner-a"),
    )

    assert all(edge.type is not EdgeType.semantic for edge in result.edges)
    assert result.semantic_status is not None
    assert result.semantic_status.detail_reason == "configuration_stale"
    assert len(vectors.query_calls) == 1


@pytest.mark.asyncio
async def test_stable_cache_preserves_semantic_truncation_reasons(
    projection_parts,
) -> None:
    _source, _target, store, vectors = projection_parts
    cache = GraphCache(ttl_seconds=60, max_keys=10)
    ordinary = _ordinary_response().model_copy(update={"limits": GraphLimits(max_nodes=10, max_edges=0, max_degree=4)})
    projector, _service, _factory_calls = _projector(
        store=store,
        vectors=vectors,
        ordinary=ordinary,
        cache=cache,
    )
    request = NoteGraphRequest(
        center_note_id="focus-note",
        edge_types=[EdgeType.semantic],
    )

    first = await projector.project(request, ordinary, user=SimpleNamespace(id_str="owner-a"))
    second = await projector.project(request, ordinary, user=SimpleNamespace(id_str="owner-a"))

    assert first.semantic_status is not None
    assert second.semantic_status is not None
    assert first.semantic_status.truncated_by == ["semantic_edges", "semantic_nodes"]
    assert second.semantic_status.truncated_by == first.semantic_status.truncated_by
    assert len(vectors.query_calls) == 1


@pytest.mark.asyncio
async def test_semantic_only_request_still_honors_authoritative_manual_supersession(
    projection_parts,
) -> None:
    _source, _target, store, vectors = projection_parts
    ordinary = _ordinary_response()
    projector, _service, _factory_calls = _projector(
        store=store,
        vectors=vectors,
        ordinary=ordinary,
    )
    projector._db.manual_edges.append(
        {
            "edge_id": "manual-a",
            "from_note_id": "target-note",
            "to_note_id": "focus-note",
            "directed": False,
            "weight": 1.0,
        }
    )

    result = await projector.project(
        NoteGraphRequest(
            center_note_id="focus-note",
            edge_types=[EdgeType.semantic],
        ),
        ordinary,
        user=SimpleNamespace(id_str="owner-a"),
    )

    assert all(edge.type is not EdgeType.semantic for edge in result.edges)
    assert [node.id for node in result.nodes] == ["focus-note"]


@pytest.mark.asyncio
async def test_threshold_filtering_is_not_reported_as_candidate_truncation(
    projection_parts,
) -> None:
    _source, _target, store, vectors = projection_parts
    vectors.distance = 0.4
    ordinary = _ordinary_response()
    projector, _service, _factory_calls = _projector(
        store=store,
        vectors=vectors,
        ordinary=ordinary,
    )

    result = await projector.project(
        NoteGraphRequest(
            center_note_id="focus-note",
            edge_types=[EdgeType.semantic],
            semantic_threshold=0.75,
        ),
        ordinary,
        user=SimpleNamespace(id_str="owner-a"),
    )

    assert result.semantic_status is not None
    assert "semantic_candidates" not in result.semantic_status.truncated_by


def test_candidate_truncation_tracks_actual_post_threshold_top_k_clipping() -> None:
    candidates = (
        SemanticChunkCandidate("note-a", "source-a", "target-a", 0.1),
        SemanticChunkCandidate("note-b", "source-b", "target-b", 0.2),
        SemanticChunkCandidate("note-c", "source-c", "target-c", 0.8),
    )
    current_ids = {
        "source-a",
        "source-b",
        "source-c",
        "target-a",
        "target-b",
        "target-c",
    }

    clipped, was_clipped = _rank_bounded_candidates(
        candidates,
        threshold=0.75,
        top_k=1,
        current_chunk_ids=current_ids,
    )
    exact, exact_was_clipped = _rank_bounded_candidates(
        candidates,
        threshold=0.75,
        top_k=2,
        current_chunk_ids=current_ids,
    )

    assert [match.target_note_id for match in clipped] == ["note-a"]
    assert was_clipped is True
    assert [match.target_note_id for match in exact] == ["note-a", "note-b"]
    assert exact_was_clipped is False


@pytest.mark.asyncio
async def test_operator_kill_switch_suppresses_semantic_queries(
    projection_parts,
) -> None:
    _source, _target, store, vectors = projection_parts
    ordinary = _ordinary_response()
    projector, graph_service, factory_calls = _projector(
        store=store,
        vectors=vectors,
        ordinary=ordinary,
        settings=SemanticIndexSettings(indexing_enabled=False),
    )

    result = await projector.project(
        NoteGraphRequest(
            center_note_id="focus-note",
            edge_types=[EdgeType.semantic],
        ),
        ordinary,
        user=SimpleNamespace(id_str="owner-a"),
    )

    assert result.semantic_status is not None
    assert result.semantic_status.available is False
    assert result.semantic_status.detail_reason == "notes_semantic_indexing_disabled"
    assert vectors.query_calls == []
    assert graph_service.calls == []
    assert factory_calls == []


@pytest.mark.asyncio
async def test_changed_semantic_binding_rejects_later_page_cursor(
    projection_parts,
) -> None:
    _source, _target, store, vectors = projection_parts
    ordinary_cursor = _encode_cursor(
        0,
        1,
        "focus-note",
        dataset_hash="dataset-hash",
        graph_revision=19,
        parser_version=3,
        request_hash="request-hash",
    )
    projector, _service, _factory_calls = _projector(
        store=store,
        vectors=vectors,
        ordinary=_ordinary_response(cursor=ordinary_cursor),
    )
    request = NoteGraphRequest(
        center_note_id="focus-note",
        edge_types=[EdgeType.semantic],
    )
    first = await projector.project(
        request,
        _ordinary_response(cursor=ordinary_cursor),
        user=SimpleNamespace(id_str="owner-a"),
    )
    store.config.semantic_index_revision += 1

    with pytest.raises(SemanticProjectionError) as exc_info:
        await projector.project(
            request.model_copy(update={"cursor": first.cursor}),
            _ordinary_response(),
            user=SimpleNamespace(id_str="owner-a"),
        )

    assert exc_info.value.code == "notes_semantic_cursor_mismatch"


@pytest.mark.asyncio
async def test_dirty_progress_bypasses_a_stable_projection_cache(
    projection_parts,
) -> None:
    _source, _target, store, vectors = projection_parts
    cache = GraphCache(ttl_seconds=60, max_keys=10)
    ordinary = _ordinary_response()
    projector, _service, _factory_calls = _projector(
        store=store,
        vectors=vectors,
        ordinary=ordinary,
        cache=cache,
    )
    request = NoteGraphRequest(
        center_note_id="focus-note",
        edge_types=[EdgeType.semantic],
    )

    await projector.project(request, ordinary, user=SimpleNamespace(id_str="owner-a"))
    store.pending_notes = 1
    await projector.project(request, ordinary, user=SimpleNamespace(id_str="owner-a"))

    assert len(vectors.query_calls) == 2


@pytest.mark.asyncio
async def test_projection_revision_change_during_query_is_not_returned_or_cached(
    projection_parts,
) -> None:
    _source, _target, store, vectors = projection_parts
    cache = GraphCache(ttl_seconds=60, max_keys=10)
    ordinary = _ordinary_response()
    projector, graph_service, _factory_calls = _projector(
        store=store,
        vectors=vectors,
        ordinary=ordinary,
        cache=cache,
    )
    projection_store = projector._db.note_graph_projection_store
    graph_service.on_generate = lambda: setattr(
        projection_store,
        "revision",
        projection_store.revision + 1,
    )

    result = await projector.project(
        NoteGraphRequest(
            center_note_id="focus-note",
            edge_types=[EdgeType.semantic],
        ),
        ordinary,
        user=SimpleNamespace(id_str="owner-a"),
    )

    assert all(edge.type is not EdgeType.semantic for edge in result.edges)
    assert result.semantic_status is not None
    assert result.semantic_status.detail_reason == "configuration_stale"
    assert cache.stats()["size"] == 0


@pytest.mark.asyncio
async def test_semantic_binding_change_after_authority_filter_is_not_admitted_or_cached(
    projection_parts,
) -> None:
    _source, _target, store, vectors = projection_parts
    cache = GraphCache(ttl_seconds=60, max_keys=10)
    ordinary = _ordinary_response()
    projector, _service, _factory_calls = _projector(
        store=store,
        vectors=vectors,
        ordinary=ordinary,
        cache=cache,
    )
    original_filter = store.filter_projection_note_ids

    def filter_and_change_binding(**kwargs: object) -> frozenset[str]:
        admitted = original_filter(**kwargs)
        store.config.semantic_index_revision += 1
        return admitted

    store.filter_projection_note_ids = filter_and_change_binding

    result = await projector.project(
        NoteGraphRequest(
            center_note_id="focus-note",
            edge_types=[EdgeType.semantic],
        ),
        ordinary,
        user=SimpleNamespace(id_str="owner-a"),
    )

    assert all(edge.type is not EdgeType.semantic for edge in result.edges)
    assert result.semantic_status is not None
    assert result.semantic_status.detail_reason == "configuration_stale"
    assert cache.stats()["size"] == 0


@pytest.mark.asyncio
async def test_manual_conversion_accepts_a_current_low_threshold_relationship(
    projection_parts,
) -> None:
    _source, _target, store, vectors = projection_parts
    vectors.distance = 0.4
    projector, _service, _factory_calls = _projector(
        store=store,
        vectors=vectors,
        ordinary=_ordinary_response(),
    )

    await projector.validate_conversion(
        source_note_id="focus-note",
        target_note_id="target-note",
        generation_id="generation-a",
    )


@pytest.mark.asyncio
async def test_manual_conversion_rejects_binding_change_after_authority_filter(
    projection_parts,
) -> None:
    _source, _target, store, vectors = projection_parts
    projector, _service, _factory_calls = _projector(
        store=store,
        vectors=vectors,
        ordinary=_ordinary_response(),
    )
    original_filter = store.filter_projection_note_ids

    def filter_and_change_binding(**kwargs: object) -> frozenset[str]:
        admitted = original_filter(**kwargs)
        store.config.semantic_index_revision += 1
        return admitted

    store.filter_projection_note_ids = filter_and_change_binding

    with pytest.raises(SemanticProjectionError) as exc_info:
        await projector.validate_conversion(
            source_note_id="focus-note",
            target_note_id="target-note",
            generation_id="generation-a",
        )

    assert exc_info.value.code == "notes_semantic_conversion_generation_stale"


def test_response_evidence_cap_preserves_edges_and_omits_later_pairs(
    projection_parts,
) -> None:
    source, target, _store, _vectors = projection_parts
    pair = SemanticExcerptPair(
        source=SemanticExcerpt(
            field="content",
            start_code_point=0,
            end_code_point=5,
            text="alpha",
        ),
        target=SemanticExcerpt(
            field="content",
            start_code_point=0,
            end_code_point=5,
            text="alpha",
        ),
    )
    evidence = SemanticEdgeEvidence(
        similarity=0.9,
        qualitative_band="very_high",
        source_note_id="focus-note",
        target_note_id="target-note",
        source_content_version=source.content_version,
        target_content_version=target.content_version,
        generation_id="generation-a",
        semantic_index_revision=11,
        configuration_revision=7,
        normalization_version=source.normalization_version,
        chunker_version=source.chunker_version,
        provider_label="OpenAI",
        model_label="text-embedding-3-small",
        excerpt_pairs=[pair],
    )
    edges = tuple(
        GraphEdge(
            id=f"semantic-{index}",
            source="focus-note",
            target="target-note",
            type=EdgeType.semantic,
            directed=False,
            evidence=evidence,
        )
        for index in range(2)
    )

    evidence_size = len(evidence.model_dump_json().encode("utf-8"))
    bounded, truncated = bound_semantic_evidence(edges, byte_cap=evidence_size)

    assert len(bounded) == 2
    assert bounded[0].evidence is not None
    assert bounded[0].evidence == evidence
    assert bounded[1].evidence is None
    assert bounded[1].evidence_omitted == "response_byte_cap"
    assert GraphEdge.model_validate(bounded[1].model_dump(mode="python")) == bounded[1]
    assert (
        sum(len(edge.evidence.model_dump_json().encode("utf-8")) for edge in bounded if edge.evidence is not None)
        <= evidence_size
    )
    assert truncated is True
