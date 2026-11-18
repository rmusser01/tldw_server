import pytest
from typing import Any, Dict, List, Optional

from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import MediaDBRetriever, RetrievalConfig
from tldw_Server_API.app.core.RAG.rag_service.vector_stores.base import VectorSearchResult


class _FakeVectorStore:
    def __init__(self, base_results: List[VectorSearchResult], hyde_results: List[VectorSearchResult]):
        self._initialized = True
        self._base_results = base_results
        self._hyde_results = hyde_results

    async def initialize(self) -> None:  # pragma: no cover - kept for parity
        self._initialized = True

    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> List[VectorSearchResult]:
        # Extract kind from simple filters or {"$and": [.., {"kind": x}]}
        kind = None
        if isinstance(filter, dict):
            if "kind" in filter:
                kind = filter.get("kind")
            elif "$and" in filter and isinstance(filter.get("$and"), list):
                for ele in filter.get("$and"):
                    if isinstance(ele, dict) and "kind" in ele:
                        kind = ele["kind"]
                        break
        if kind == "hyde_q":
            return self._hyde_results[:k]
        # For "chunk" or None (fallback) return base
        return self._base_results[:k]

    async def multi_search(
        self,
        collection_patterns: List[str],
        query_vector: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        # Delegate to search ignoring patterns for this fake
        return await self.search(collection_patterns[0] if collection_patterns else "", query_vector, k, filter, True)


class _StubSettings:
    def __init__(self, overrides: Dict[str, Any]):
        self._overrides = overrides

    def get(self, key: str, default=None):
        return self._overrides.get(key, default)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_vector_retriever_hyde_only_returns_hyde_results(monkeypatch):
    # Configure HYDE to be enabled; base is empty
    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.settings",
        _StubSettings({
            "HYDE_ENABLED": True,
            "HYDE_ONLY_IF_NEEDED": True,
            "HYDE_SCORE_FLOOR": 0.99,
            "HYDE_K_FRACTION": 1.0,
            "HYDE_WEIGHT_QUESTION_MATCH": 0.05,
        }),
        raising=False,
    )

    monkeypatch.setattr(MediaDBRetriever, "_initialize_vector_store", lambda self: None)

    base: List[VectorSearchResult] = []
    hyde: List[VectorSearchResult] = [
        VectorSearchResult(id="c1", content="A", metadata={"media_id": "101"}, score=0.4, distance=0.6),
        VectorSearchResult(id="c2", content="B", metadata={"media_id": "102"}, score=0.6, distance=0.4),
    ]

    retr = MediaDBRetriever(db_path=":memory:", config=RetrievalConfig(max_results=5, use_fts=False, use_vector=True), user_id="u1")
    retr.vector_store = _FakeVectorStore(base, hyde)

    docs = await retr._retrieve_vector("query", query_vector=[0.0, 0.1, 0.2])
    assert len(docs) == len(hyde)
    # Media-level merge retains the selected vector identity and parent metadata.
    assert {doc.id: (doc.metadata.get("media_id"), doc.content) for doc in docs} == {
        "c1": ("101", "A"),
        "c2": ("102", "B"),
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_vector_retriever_hyde_merge_prefers_higher_score(monkeypatch):
    # Enable HYDE and set thresholds so merge path executes
    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.settings",
        _StubSettings({
            "HYDE_ENABLED": True,
            "HYDE_ONLY_IF_NEEDED": False,
            "HYDE_SCORE_FLOOR": 0.1,
            "HYDE_K_FRACTION": 1.0,
            "HYDE_WEIGHT_QUESTION_MATCH": 0.05,
        }),
        raising=False,
    )

    monkeypatch.setattr(MediaDBRetriever, "_initialize_vector_store", lambda self: None)

    base = [
        VectorSearchResult(id="chunkA", content="base A", metadata={"media_id": "201"}, score=0.40, distance=0.60),
    ]
    hyde = [
        VectorSearchResult(id="chunkA_hyde", content="hyde A", metadata={"media_id": "201"}, score=0.50, distance=0.50),
        VectorSearchResult(id="chunkB_hyde", content="hyde B", metadata={"media_id": "202"}, score=0.30, distance=0.70),
    ]

    retr = MediaDBRetriever(db_path=":memory:", config=RetrievalConfig(max_results=5, use_fts=False, use_vector=True), user_id="u1")
    retr.vector_store = _FakeVectorStore(base, hyde)

    docs = await retr._retrieve_vector("query", query_vector=[0.1, 0.2, 0.3])
    # Expect both media 201 and 202
    by_media_id = {doc.metadata.get("media_id"): doc for doc in docs}
    assert {media_id: (doc.id, doc.content) for media_id, doc in by_media_id.items()} == {
        "201": ("chunkA", "base A"),
        "202": ("chunkB_hyde", "hyde B"),
    }
    # HYDE score boosted by weight = 0.50 + 0.05 = 0.55
    assert abs(by_media_id["201"].score - 0.55) < 1e-6


@pytest.mark.unit
@pytest.mark.asyncio
async def test_vector_retriever_hyde_chunk_level_dedupe(monkeypatch):
    # Enable HYDE and select chunk-level merge
    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.settings",
        _StubSettings({
            "HYDE_ENABLED": True,
            "HYDE_ONLY_IF_NEEDED": False,
            "HYDE_SCORE_FLOOR": 0.1,
            "HYDE_K_FRACTION": 1.0,
            "HYDE_WEIGHT_QUESTION_MATCH": 0.05,
            "HYDE_DEDUPE_BY_PARENT": True,
        }),
        raising=False,
    )

    monkeypatch.setattr(MediaDBRetriever, "_initialize_vector_store", lambda self: None)

    # Base results: two chunks under same parent, different chunk_id
    base = [
        VectorSearchResult(
            id="chunk1",
            content="base c1",
            metadata={"media_id": "301", "parent_chunk_id": "P1", "chunk_id": "C1"},
            score=0.40,
            distance=0.60,
        ),
        VectorSearchResult(
            id="chunk2",
            content="base c2",
            metadata={"media_id": "301", "parent_chunk_id": "P1", "chunk_id": "C2"},
            score=0.60,
            distance=0.40,
        ),
    ]
    # HYDE improves C1 but is worse than base for C2
    hyde = [
        VectorSearchResult(
            id="chunk1_hyde",
            content="hyde c1",
            metadata={"media_id": "301", "parent_chunk_id": "P1", "chunk_id": "C1"},
            score=0.50,
            distance=0.50,
        ),
        VectorSearchResult(
            id="chunk2_hyde",
            content="hyde c2",
            metadata={"media_id": "301", "parent_chunk_id": "P1", "chunk_id": "C2"},
            score=0.45,
            distance=0.55,
        ),
    ]

    retr = MediaDBRetriever(db_path=":memory:", config=RetrievalConfig(max_results=5, use_fts=False, use_vector=True), user_id="u1")
    retr.vector_store = _FakeVectorStore(base, hyde)

    docs = await retr._retrieve_vector("query", query_vector=[0.2, 0.3, 0.4])
    # Chunk-level dedupe groups by parent_chunk_id; both chunks share P1 → one doc
    assert len(docs) == 1
    # Best among base (C2=0.60) and HYDE (C1=0.55) should be 0.60
    assert abs(docs[0].score - 0.60) < 1e-6
