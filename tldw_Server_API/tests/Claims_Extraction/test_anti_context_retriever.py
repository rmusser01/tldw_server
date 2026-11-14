"""
Tests for the anti-context retrieval module.

These tests verify the anti-context retrieval logic for the FVA pipeline,
including query generation, document deduplication, and source diversity.
"""

import pytest
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

from tldw_Server_API.app.core.Claims_Extraction.anti_context_retriever import (
    AntiContextRetriever,
    AntiContextConfig,
    AntiContextResult,
    NEGATION_TEMPLATES,
    CONTRARY_TEMPLATES,
)

# Import ClaimType for creating test claims
try:
    from tldw_Server_API.app.core.RAG.rag_service.types import ClaimType, Document, DataSource
except ImportError:
    from enum import Enum

    class ClaimType(Enum):
        STATISTIC = "statistic"
        COMPARATIVE = "comparative"
        CAUSAL = "causal"
        RANKING = "ranking"
        GENERAL = "general"
        QUOTE = "quote"
        TEMPORAL = "temporal"

    class DataSource(Enum):
        MEDIA_DB = "media_db"

    @dataclass
    class Document:
        id: str
        content: str
        metadata: dict[str, Any] = field(default_factory=dict)
        score: float = 0.0


@dataclass
class MockClaim:
    """Mock claim for testing."""
    id: str
    text: str
    claim_type: ClaimType = ClaimType.GENERAL
    extracted_values: dict[str, Any] = field(default_factory=dict)
    span: tuple[int, int] | None = None


class MockRetriever:
    """Mock retriever for testing."""

    def __init__(self, documents: Optional[list[Document]] = None):
        self.documents = documents or []
        self.retrieve_calls: list[dict] = []

    async def retrieve(
        self,
        query: str,
        *,
        sources: Optional[list] = None,
        config: Optional[Any] = None,
        **kwargs
    ) -> list[Document]:
        self.retrieve_calls.append({
            "query": query,
            "sources": sources,
            "config": config,
        })
        return self.documents


class TestAntiContextConfig:
    """Tests for AntiContextConfig dataclass."""

    @pytest.mark.unit
    def test_default_values(self):
        """Default config should have sensible values."""
        config = AntiContextConfig()

        assert config.max_queries == 3
        assert config.max_docs_per_query == 5
        assert config.min_relevance_score == 0.3
        assert config.exclude_original_doc_ids is True
        assert config.use_negation_templates is True
        assert config.use_contrary_templates is True
        assert config.max_docs_per_source == 2
        assert config.cache_ttl_seconds == 300

    @pytest.mark.unit
    def test_custom_values(self):
        """Custom config values should be respected."""
        config = AntiContextConfig(
            max_queries=5,
            max_docs_per_query=10,
            min_relevance_score=0.5,
            exclude_original_doc_ids=False,
            max_docs_per_source=3,
        )

        assert config.max_queries == 5
        assert config.max_docs_per_query == 10
        assert config.min_relevance_score == 0.5
        assert config.exclude_original_doc_ids is False
        assert config.max_docs_per_source == 3


class TestAntiContextRetrieverQueryGeneration:
    """Tests for anti-context query generation."""

    @pytest.mark.unit
    def test_generates_negation_queries(self):
        """Should generate negation queries from templates."""
        retriever = AntiContextRetriever(MockRetriever())
        claim = MockClaim(id="1", text="The sky is blue")

        queries = retriever._generate_anti_queries(claim)

        assert len(queries) >= 3
        assert all(strategy == "negation" for _, strategy in queries[:3])
        # Check queries contain claim text
        assert any("The sky is blue" in q for q, _ in queries)

    @pytest.mark.unit
    def test_generates_contrary_queries_for_statistic(self):
        """Should generate claim-type-specific queries for STATISTIC."""
        retriever = AntiContextRetriever(MockRetriever())
        claim = MockClaim(
            id="1",
            text="Revenue grew 15% last year",
            claim_type=ClaimType.STATISTIC,
        )

        queries = retriever._generate_anti_queries(claim)

        contrary_queries = [q for q, s in queries if s == "contrary"]
        assert len(contrary_queries) >= 1
        # Should use statistic-specific templates
        assert any("statistic" in q.lower() or "data" in q.lower() for q in contrary_queries)

    @pytest.mark.unit
    def test_generates_contrary_queries_for_causal(self):
        """Should generate claim-type-specific queries for CAUSAL."""
        retriever = AntiContextRetriever(MockRetriever())
        claim = MockClaim(
            id="1",
            text="Smoking causes cancer",
            claim_type=ClaimType.CAUSAL,
        )

        queries = retriever._generate_anti_queries(claim)

        contrary_queries = [q for q, s in queries if s == "contrary"]
        assert len(contrary_queries) >= 1

    @pytest.mark.unit
    def test_generates_contrary_queries_for_comparative(self):
        """Should generate claim-type-specific queries for COMPARATIVE."""
        retriever = AntiContextRetriever(MockRetriever())
        claim = MockClaim(
            id="1",
            text="Python is faster than Ruby",
            claim_type=ClaimType.COMPARATIVE,
        )

        queries = retriever._generate_anti_queries(claim)

        contrary_queries = [q for q, s in queries if s == "contrary"]
        assert len(contrary_queries) >= 1

    @pytest.mark.unit
    def test_no_contrary_queries_for_general(self):
        """GENERAL claim type should not produce contrary queries."""
        retriever = AntiContextRetriever(MockRetriever())
        claim = MockClaim(
            id="1",
            text="Some general statement",
            claim_type=ClaimType.GENERAL,
        )

        queries = retriever._generate_anti_queries(claim)

        contrary_queries = [q for q, s in queries if s == "contrary"]
        assert len(contrary_queries) == 0

    @pytest.mark.unit
    def test_respects_max_queries_config(self):
        """Should respect max_queries configuration."""
        config = AntiContextConfig(max_queries=2)
        retriever = AntiContextRetriever(MockRetriever(), config)
        claim = MockClaim(
            id="1",
            text="Some claim",
            claim_type=ClaimType.STATISTIC,
        )

        queries = retriever._generate_anti_queries(claim)

        # Generate up to max, actual use will be limited in retrieve_anti_context
        assert len(queries) >= 1

    @pytest.mark.unit
    def test_uses_extracted_values_in_templates(self):
        """Should use extracted_values when filling templates."""
        retriever = AntiContextRetriever(MockRetriever())
        claim = MockClaim(
            id="1",
            text="X caused Y",
            claim_type=ClaimType.CAUSAL,
            extracted_values={"cause": "smoking", "effect": "cancer"},
        )

        queries = retriever._generate_anti_queries(claim)

        contrary_queries = [q for q, s in queries if s == "contrary"]
        # Templates should be filled with extracted values
        assert len(contrary_queries) >= 1


class TestAntiContextRetrieverDiversification:
    """Tests for source diversity logic."""

    @pytest.mark.unit
    def test_diversify_limits_docs_per_source(self):
        """Should limit documents from the same source."""
        config = AntiContextConfig(max_docs_per_source=2)
        retriever = AntiContextRetriever(MockRetriever(), config)

        docs = [
            Document(id="1", content="Doc 1", metadata={"media_id": 100}, score=0.9),
            Document(id="2", content="Doc 2", metadata={"media_id": 100}, score=0.8),
            Document(id="3", content="Doc 3", metadata={"media_id": 100}, score=0.7),
            Document(id="4", content="Doc 4", metadata={"media_id": 200}, score=0.6),
        ]

        result = retriever._diversify_by_source(docs)

        # Should keep max 2 from source 100
        source_100_docs = [d for d in result if d.metadata.get("media_id") == 100]
        assert len(source_100_docs) <= 2

        # Should include doc from source 200
        source_200_docs = [d for d in result if d.metadata.get("media_id") == 200]
        assert len(source_200_docs) == 1

    @pytest.mark.unit
    def test_diversify_keeps_highest_scored(self):
        """Should keep the highest-scored documents from each source."""
        config = AntiContextConfig(max_docs_per_source=1)
        retriever = AntiContextRetriever(MockRetriever(), config)

        docs = [
            Document(id="1", content="Doc 1", metadata={"media_id": 100}, score=0.5),
            Document(id="2", content="Doc 2", metadata={"media_id": 100}, score=0.9),
            Document(id="3", content="Doc 3", metadata={"media_id": 100}, score=0.7),
        ]

        result = retriever._diversify_by_source(docs)

        assert len(result) == 1
        assert result[0].id == "2"  # Highest score
        assert result[0].score == 0.9

    @pytest.mark.unit
    def test_diversify_uses_doc_id_as_fallback(self):
        """Should use doc.id as source identifier when media_id not present."""
        config = AntiContextConfig(max_docs_per_source=1)
        retriever = AntiContextRetriever(MockRetriever(), config)

        docs = [
            Document(id="same_source", content="Doc 1", metadata={}, score=0.9),
            Document(id="same_source", content="Doc 2", metadata={}, score=0.8),
        ]

        result = retriever._diversify_by_source(docs)

        # Both have same id, so treated as same source
        assert len(result) == 1


class TestAntiContextRetrieverRetrieve:
    """Tests for the main retrieve_anti_context method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieve_calls_retriever(self):
        """Should call the underlying retriever with generated queries."""
        mock_docs = [
            Document(id="1", content="Counter evidence", metadata={}, score=0.8),
        ]
        mock_retriever = MockRetriever(mock_docs)
        retriever = AntiContextRetriever(mock_retriever)

        claim = MockClaim(id="1", text="The earth is flat")

        results = await retriever.retrieve_anti_context(
            claim=claim,
            original_doc_ids=set(),
        )

        assert len(mock_retriever.retrieve_calls) >= 1
        assert len(results) >= 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_excludes_original_doc_ids(self):
        """Should exclude documents from original retrieval."""
        mock_docs = [
            Document(id="original_1", content="Original doc", metadata={}, score=0.9),
            Document(id="new_1", content="New doc", metadata={}, score=0.8),
        ]
        mock_retriever = MockRetriever(mock_docs)
        retriever = AntiContextRetriever(mock_retriever)

        claim = MockClaim(id="1", text="Some claim")

        results = await retriever.retrieve_anti_context(
            claim=claim,
            original_doc_ids={"original_1"},
        )

        # Flatten all documents from results
        all_docs = []
        for r in results:
            all_docs.extend(r.documents)

        # Should not include the original document
        doc_ids = {d.id for d in all_docs}
        assert "original_1" not in doc_ids
        assert "new_1" in doc_ids

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_min_score_is_the_retrievers_floor_not_a_filter_on_fused_scores(self):
        """The floor goes to the retriever, which applies it before fusion; fused scores are kept (TASK-13881)."""
        mock_docs = [
            Document(id="1", content="Fused rank 1", metadata={}, score=1 / 61),
            Document(id="2", content="Fused rank 2", metadata={}, score=1 / 62),
        ]
        mock_retriever = MockRetriever(mock_docs)
        config = AntiContextConfig(min_relevance_score=0.5)
        retriever = AntiContextRetriever(mock_retriever, config)

        results = await retriever.retrieve_anti_context(
            claim=MockClaim(id="1", text="Some claim"),
            original_doc_ids=set(),
        )

        assert {call["config"].min_score for call in mock_retriever.retrieve_calls} == {0.5}
        assert {d.id for r in results for d in r.documents} == {"1", "2"}

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handles_retriever_failure_gracefully(self):
        """Should handle retriever failures without crashing."""
        mock_retriever = MockRetriever()
        mock_retriever.retrieve = AsyncMock(side_effect=Exception("Network error"))
        retriever = AntiContextRetriever(mock_retriever)

        claim = MockClaim(id="1", text="Some claim")

        # Should not raise, should return empty results
        results = await retriever.retrieve_anti_context(
            claim=claim,
            original_doc_ids=set(),
        )

        assert results == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_respects_max_queries_limit(self):
        """Should only execute up to max_queries retrievals."""
        mock_retriever = MockRetriever([
            Document(id="1", content="Doc", metadata={}, score=0.8),
        ])
        config = AntiContextConfig(max_queries=2)
        retriever = AntiContextRetriever(mock_retriever, config)

        claim = MockClaim(
            id="1",
            text="Some statistic claim",
            claim_type=ClaimType.STATISTIC,
        )

        await retriever.retrieve_anti_context(
            claim=claim,
            original_doc_ids=set(),
        )

        # Should have made at most max_queries retrieve calls
        assert len(mock_retriever.retrieve_calls) <= 2


class TestAntiContextRetrieverCache:
    """Tests for query result caching."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_caches_query_results(self):
        """Should cache query results."""
        mock_retriever = MockRetriever([
            Document(id="1", content="Doc", metadata={}, score=0.8),
        ])
        retriever = AntiContextRetriever(mock_retriever)

        claim = MockClaim(id="1", text="Same claim")

        # First retrieval
        await retriever.retrieve_anti_context(claim=claim, original_doc_ids=set())
        first_call_count = len(mock_retriever.retrieve_calls)

        # Second retrieval with same claim
        await retriever.retrieve_anti_context(claim=claim, original_doc_ids=set())
        second_call_count = len(mock_retriever.retrieve_calls)

        # Should have used cache for second call
        assert second_call_count == first_call_count

    @pytest.mark.unit
    def test_clear_cache(self):
        """Should clear the query cache."""
        retriever = AntiContextRetriever(MockRetriever())
        retriever._query_cache["test_key"] = []

        assert retriever.get_cache_size() == 1

        retriever.clear_cache()

        assert retriever.get_cache_size() == 0

    @pytest.mark.unit
    def test_get_cache_size(self):
        """Should return correct cache size."""
        retriever = AntiContextRetriever(MockRetriever())

        assert retriever.get_cache_size() == 0

        retriever._query_cache["key1"] = []
        retriever._query_cache["key2"] = []

        assert retriever.get_cache_size() == 2


class TestAntiContextResult:
    """Tests for AntiContextResult dataclass."""

    @pytest.mark.unit
    def test_result_fields(self):
        """AntiContextResult should have expected fields."""
        docs = [Document(id="1", content="Doc", metadata={}, score=0.8)]
        result = AntiContextResult(
            query_used="test query",
            documents=docs,
            strategy="negation",
        )

        assert result.query_used == "test query"
        assert result.documents == docs
        assert result.strategy == "negation"


class TestNegationTemplates:
    """Tests for negation template constants."""

    @pytest.mark.unit
    def test_negation_templates_exist(self):
        """Should have negation templates defined."""
        assert len(NEGATION_TEMPLATES) >= 5
        assert all("{claim}" in t for t in NEGATION_TEMPLATES)

    @pytest.mark.unit
    def test_contrary_templates_exist(self):
        """Should have contrary templates for key claim types."""
        assert "statistic" in CONTRARY_TEMPLATES
        assert "causal" in CONTRARY_TEMPLATES
        assert "comparative" in CONTRARY_TEMPLATES
        assert "ranking" in CONTRARY_TEMPLATES


class _NearestNeighbourStore:
    """A vector store whose nearest neighbour of every query is ``hit``; it notes the collections searched."""

    _initialized = True

    def __init__(self, hit) -> None:
        self.hit = hit
        self.collections: list[str] = []

    async def search(self, collection_name, query_vector, k=10, filter=None, include_metadata=True):
        self.collections.append(collection_name)
        return [self.hit]


@pytest.mark.unit
@pytest.mark.parametrize("with_vector_store", [False, True])
async def test_counter_evidence_from_a_real_hybrid_retriever_is_kept(tmp_path, monkeypatch, with_vector_store):
    """A fused (RRF) score is at most 1/61: a 0.3 floor on it dropped every document, so FVA never adjudicated.

    Without a vector store the FTS hit comes through. With one, so does the vector-only hit, which FTS cannot find.
    """
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create
    from tldw_Server_API.app.core.RAG.rag_service import database_retrievers
    from tldw_Server_API.app.core.RAG.rag_service.vector_stores.base import VectorSearchResult

    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="7")
    with scoped_context(user_id=7, is_admin=False):
        fts_hit, _, _ = db.add_media_with_keywords(
            title="Counter-evidence", media_type="text", owner_user_id=7, keywords=None,
            content="Evidence against the claim that the sky is green: it is blue.",
        )
        vector_hit, _, _ = db.add_media_with_keywords(
            title="Optics", media_type="text", owner_user_id=7, keywords=None,
            content="Rayleigh scattering colours daylight blue.",
        )
    store = _NearestNeighbourStore(
        VectorSearchResult(
            id=f"{vector_hit}_0", content="Rayleigh scattering colours daylight blue.",
            metadata={"media_id": str(vector_hit)}, score=0.42, distance=0.58,
        )
    )
    monkeypatch.setattr(
        database_retrievers, "create_from_settings_for_user", lambda settings, user_id="0": store if with_vector_store else None
    )
    monkeypatch.setattr(Embeddings_Create, "get_embedding_config", lambda: {})
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", lambda texts, *a, **k: [[0.1, 0.2, 0.3] for _ in texts])
    retriever = database_retrievers.MultiDatabaseRetriever({}, user_id="7", media_db=db)
    try:
        with scoped_context(user_id=7, is_admin=False):
            results = await AntiContextRetriever(retriever).retrieve_anti_context(
                claim=MockClaim(id="1", text="The sky is green."), original_doc_ids=set()
            )
    finally:
        db.close_connection()

    found = {d.id for r in results for d in r.documents}
    assert found == ({str(fts_hit), str(vector_hit)} if with_vector_store else {str(fts_hit)})
    assert store.collections == (["user_7_media_embeddings"] * 3 if with_vector_store else [])
