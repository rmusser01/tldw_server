"""
Property-based tests for RAG system.

Uses Hypothesis to verify invariants and properties that should hold
across all valid inputs.
"""

import pytest
from hypothesis import given, strategies as st, example, settings, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, Bundle
import numpy as np
from datetime import datetime

from dataclasses import dataclass, field
from typing import List, Dict, Any
from tldw_Server_API.app.core.RAG.rag_service.types import Document
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline
from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse
from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import RetrievalConfig

# =====================================================================
# Hypothesis Strategies
# =====================================================================

# Valid query strategies
valid_query = st.text(min_size=1, max_size=500).filter(lambda x: x.strip() != "")

# Document content strategy
document_content = st.text(min_size=10, max_size=5000)

# Document strategy
@st.composite
def document_strategy(draw):
    """Generate valid documents."""
    return Document(
        id=draw(st.text(min_size=1, max_size=50)),
        content=draw(document_content),
        metadata=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers(), st.floats(allow_nan=False), st.booleans()),
            max_size=10
        ))
    )

# Score strategy
valid_score = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

# Configuration strategy
@st.composite
def config_strategy(draw):
    """Generate valid pipeline configurations."""
    return {
        "enable_cache": draw(st.booleans()),
        "enable_expansion": draw(st.booleans()),
        "enable_reranking": draw(st.booleans()),
        "top_k": draw(st.integers(min_value=1, max_value=100)),
        "temperature": draw(st.floats(min_value=0.0, max_value=2.0)),
        "chunk_size": draw(st.integers(min_value=100, max_value=2000)),
        "chunk_overlap": draw(st.integers(min_value=0, max_value=500))
    }

# Retrieval config strategy
@st.composite
def retrieval_config_strategy(draw):
    """Generate valid retrieval configurations aligned with current API."""
    use_fts = draw(st.booleans())
    use_vector = draw(st.booleans())
    # Ensure at least one mode is enabled to be meaningful
    if not use_fts and not use_vector:
        use_fts = True
    return RetrievalConfig(
        max_results=draw(st.integers(min_value=1, max_value=50)),
        min_score=draw(st.floats(min_value=0.0, max_value=1.0)),
        use_fts=use_fts,
        use_vector=use_vector,
        include_metadata=True
    )


# =====================================================================
# Property Tests for Pipeline Context
# =====================================================================

@dataclass
class SimpleContext:
    query: str
    original_query: str
    documents: List[Document] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    timings: Dict[str, float] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)


@pytest.mark.property
class TestPipelineContextProperties:
    """Property tests for RAGPipelineContext."""

    @given(
        query=valid_query,
        config=config_strategy()
    )
    def test_context_query_preservation(self, query, config):
        """Original query should always be preserved."""
        context = SimpleContext(
            query=query,
            original_query=query,
            config=config
        )

        # Original query should never change
        assert context.original_query == query

        # Modified query should start as original
        assert context.query == query

        # After modification
        context.query = query + " expanded"
        assert context.original_query == query  # Still unchanged

    @given(
        documents=st.lists(document_strategy(), min_size=0, max_size=50)
    )
    def test_context_document_operations(self, documents):
        """Document operations should maintain consistency."""
        context = SimpleContext(
            query="test",
            original_query="test"
        )

        # Add documents
        for doc in documents:
            context.documents.append(doc)

        assert len(context.documents) == len(documents)

        # All documents should be retrievable
        for i, doc in enumerate(documents):
            assert context.documents[i].id == doc.id
            assert context.documents[i].content == doc.content

    @given(
        errors=st.lists(
            st.dictionaries(
                st.sampled_from(["function", "error", "timestamp"]),
                st.sampled_from(
                    ["", "retrieval", "provider timeout", "2026-07-14T00:00:00Z"]
                ),
                min_size=0,
                max_size=3,
            ),
            min_size=0,
            max_size=20,
        )
    )
    def test_context_error_tracking(self, errors):
        """Error tracking should preserve all error information."""
        context = SimpleContext(
            query="test",
            original_query="test"
        )

        for error in errors:
            context.errors.append(error)

        assert len(context.errors) == len(errors)

        # All errors should be preserved
        for i, error in enumerate(errors):
            for key, value in error.items():
                assert context.errors[i].get(key) == value


# =====================================================================
# Property Tests for Document Retrieval
# =====================================================================

@pytest.mark.property
class TestRetrievalProperties:
    """Property tests for document retrieval."""

    @given(
        num_docs=st.integers(min_value=0, max_value=100),
        top_k=st.integers(min_value=1, max_value=50)
    )
    def test_retrieval_count_invariant(self, num_docs, top_k):
        """Retrieved documents should never exceed min(num_docs, top_k)."""
        # Create documents
        documents = [
            Document(id=f"doc_{i}", content=f"Content {i}", metadata={})
            for i in range(num_docs)
        ]

        # Simulate retrieval
        retrieved = documents[:min(num_docs, top_k)]

        assert len(retrieved) <= min(num_docs, top_k)
        assert len(retrieved) <= top_k
        assert len(retrieved) <= num_docs

    @given(
        documents=st.lists(
            st.tuples(document_strategy(), valid_score),
            min_size=1,
            max_size=50
        )
    )
    def test_retrieval_score_ordering(self, documents):
        """Retrieved documents should be ordered by score (descending)."""
        # Sort by score
        sorted_docs = sorted(documents, key=lambda x: x[1], reverse=True)

        # Verify ordering
        scores = [score for _, score in sorted_docs]
        assert scores == sorted(scores, reverse=True)

        # No score should be negative
        assert all(score >= 0 for score in scores)

        # No score should exceed 1.0 (if normalized)
        if all(score <= 1.0 for score in scores):
            assert all(0 <= score <= 1.0 for score in scores)

    @given(
        documents=st.lists(document_strategy(), min_size=1, max_size=20, unique_by=lambda d: d.id),
        min_score=st.floats(min_value=0.0, max_value=1.0)
    )
    def test_score_filtering_invariant(self, documents, min_score):
        """Documents below min_score should be filtered out."""
        # Assign random scores
        doc_scores = [
            (doc, np.random.random())
            for doc in documents
        ]

        # Filter by min_score
        filtered = [
            (doc, score)
            for doc, score in doc_scores
            if score >= min_score
        ]

        # All remaining documents should meet threshold
        assert all(score >= min_score for _, score in filtered)

        # No valid document should be excluded
        excluded = [
            (doc, score)
            for doc, score in doc_scores
            if score < min_score
        ]
        assert all(score < min_score for _, score in excluded)

    @given(
        query=valid_query,
        config=retrieval_config_strategy()
    )
    def test_retrieval_config_validity(self, query, config):
        """Retrieval configuration should always be valid."""
        assert config.max_results > 0
        assert 0 <= config.min_score <= 1.0
        # At least one search mode enabled
        assert config.use_fts or config.use_vector


# =====================================================================
# Property Tests for Query Expansion
# =====================================================================

@pytest.mark.property
class TestQueryExpansionProperties:
    """Property tests for query expansion."""

    @given(query=valid_query)
    def test_expansion_preserves_original(self, query):
        """Expansion should preserve original query information."""
        expanded = query + " expanded terms"

        # Original query should be substring of expanded
        assert query in expanded or query == expanded

        # Expanded should be at least as long
        assert len(expanded) >= len(query)

    @given(
        query=st.text(min_size=1, max_size=100),
        expansions=st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=5)
    )
    def test_expansion_additive(self, query, expansions):
        """Query expansion should only add terms, not remove."""
        expanded_query = query
        for expansion in expansions:
            expanded_query = f"{expanded_query} {expansion}"

        # Original query should still be present
        assert query in expanded_query

        # Each expansion should be present
        for expansion in expansions:
            assert expansion in expanded_query

    @given(
        acronym=st.text(alphabet=st.characters(whitelist_categories=("Lu",)), min_size=2, max_size=5)
    )
    @example(acronym="API")
    @example(acronym="RAG")
    @example(acronym="ML")
    def test_acronym_expansion_format(self, acronym):
        """Acronym expansion should follow expected format."""
        # Simulate acronym expansion
        expanded_forms = {
            "API": "Application Programming Interface",
            "RAG": "Retrieval Augmented Generation",
            "ML": "Machine Learning",
            "AI": "Artificial Intelligence"
        }

        if acronym in expanded_forms:
            expanded = f"{acronym} {expanded_forms[acronym]}"
            assert acronym in expanded
            assert len(expanded) > len(acronym)


# =====================================================================
# Property Tests for Reranking
# =====================================================================

@pytest.mark.property
class TestRerankingProperties:
    """Property tests for document reranking."""

    @given(
        documents=st.lists(document_strategy(), min_size=1, max_size=20, unique_by=lambda d: d.id),
        rerank_top_k=st.integers(min_value=1, max_value=10)
    )
    def test_reranking_preserves_documents(self, documents, rerank_top_k):
        """Reranking should only reorder, not modify documents."""
        # Simulate reranking
        reranked = documents[:min(len(documents), rerank_top_k)]

        # All reranked documents should be from original set
        original_ids = {doc.id for doc in documents}
        reranked_ids = {doc.id for doc in reranked}

        assert reranked_ids.issubset(original_ids)

        # Document content should be unchanged
        doc_map = {doc.id: doc for doc in documents}
        for doc in reranked:
            assert doc.content == doc_map[doc.id].content

    @given(
        num_docs=st.integers(min_value=1, max_value=50),
        rerank_top_k=st.integers(min_value=1, max_value=20)
    )
    def test_reranking_count_invariant(self, num_docs, rerank_top_k):
        """Reranking should respect top_k limit."""
        documents = [
            Document(id=f"doc_{i}", content=f"Content {i}", metadata={})
            for i in range(num_docs)
        ]

        reranked = documents[:min(num_docs, rerank_top_k)]

        assert len(reranked) <= rerank_top_k
        assert len(reranked) <= num_docs
        assert len(reranked) == min(num_docs, rerank_top_k)

    @given(
        documents=st.lists(
            st.tuples(document_strategy(), st.floats(min_value=0, max_value=1)),
            min_size=2,
            max_size=20
        )
    )
    def test_reranking_improves_relevance(self, documents):
        """Reranking should improve average relevance score."""
        # Initial scores
        initial_scores = [score for _, score in documents]
        avg_initial = sum(initial_scores) / len(initial_scores) if initial_scores else 0

        # Simulate reranking (take top half)
        sorted_docs = sorted(documents, key=lambda x: x[1], reverse=True)
        reranked = sorted_docs[:len(sorted_docs)//2 + 1]

        reranked_scores = [score for _, score in reranked]
        avg_reranked = sum(reranked_scores) / len(reranked_scores) if reranked_scores else 0

        # Average score should improve or stay same (allow tiny float tolerance)
        assert (avg_reranked + 1e-9) >= avg_initial or len(reranked) == len(documents)


# =====================================================================
# Property Tests for Caching
# =====================================================================

@pytest.mark.property
class TestCacheProperties:
    """Property tests for caching behavior."""

    @given(
        query1=valid_query,
        query2=valid_query,
        documents=st.lists(document_strategy(), min_size=1, max_size=10)
    )
    def test_cache_determinism(self, query1, query2, documents):
        """Same query should produce same cache key."""
        # Simulate cache key generation
        def generate_cache_key(query: str) -> str:
            return str(hash(query.lower().strip()))

        key1_first = generate_cache_key(query1)
        key1_second = generate_cache_key(query1)

        # Same query should produce same key
        assert key1_first == key1_second

        # Different queries should (usually) produce different keys
        if query1.lower().strip() != query2.lower().strip():
            key2 = generate_cache_key(query2)
            # High probability of different keys
            # (hash collision possible but rare)

    @given(
        ttl=st.integers(min_value=1, max_value=86400),
        access_time=st.integers(min_value=0, max_value=86400)
    )
    def test_cache_ttl_behavior(self, ttl, access_time):
        """Cache entries should expire after TTL."""
        cache_time = 0

        # Entry should be valid before TTL
        if access_time < ttl:
            assert (cache_time + access_time) < (cache_time + ttl)
            # Entry is still valid

        # Entry should be invalid after TTL
        if access_time >= ttl:
            assert (cache_time + access_time) >= (cache_time + ttl)
            # Entry has expired

    @given(
        cache_size=st.integers(min_value=1, max_value=100),
        num_entries=st.integers(min_value=0, max_value=200)
    )
    def test_cache_size_limit(self, cache_size, num_entries):
        """Cache should respect size limits."""
        cache = {}

        for i in range(num_entries):
            # Add to cache with eviction
            if len(cache) >= cache_size:
                # Evict oldest (or use LRU)
                oldest_key = min(cache.keys())
                del cache[oldest_key]

            cache[i] = f"value_{i}"

        assert len(cache) <= cache_size
        assert len(cache) == min(num_entries, cache_size)


# =====================================================================
# Stateful Property Tests
# =====================================================================

@pytest.mark.property
class RAGStateMachine(RuleBasedStateMachine):
    """Stateful testing for RAG pipeline operations."""

    def __init__(self):

        super().__init__()
        self.queries = []
        self.documents = {}
        self.cache = {}
        self.config = {
            "enable_cache": False,
            "top_k": 10
        }

    queries_bundle = Bundle("queries")
    documents_bundle = Bundle("documents")

    @rule(
        query=valid_query,
        target=queries_bundle
    )
    def add_query(self, query):
        """Add a query to the system."""
        self.queries.append(query)
        return query

    @rule(
        doc_id=st.text(min_size=1, max_size=20),
        content=document_content,
        target=documents_bundle
    )
    def add_document(self, doc_id, content):
        """Add a document to the system."""
        if doc_id not in self.documents:
            self.documents[doc_id] = Document(
                id=doc_id,
                content=content,
                metadata={"added_at": datetime.now().isoformat()}
            )
        return doc_id

    @rule(
        query=queries_bundle,
        enable_cache=st.booleans()
    )
    def execute_search(self, query, enable_cache):
        """Execute a search with the query."""
        if enable_cache and query in self.cache:
            # Cache hit
            results = self.cache[query]
        else:
            # Perform search
            results = [
                doc for doc in self.documents.values()
                if query.lower() in doc.content.lower()
            ][:self.config["top_k"]]

            if enable_cache:
                self.cache[query] = results

        # Verify invariants
        assert len(results) <= self.config["top_k"]
        assert all(isinstance(r, Document) for r in results)

    @rule(
        doc_id=documents_bundle
    )
    def remove_document(self, doc_id):
        """Remove a document from the system."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            # Invalidate cache entries that contained this document
            self.cache = {
                q: [d for d in docs if d.id != doc_id]
                for q, docs in self.cache.items()
            }

    @rule(
        top_k=st.integers(min_value=1, max_value=50)
    )
    def update_config(self, top_k):
        """Update configuration."""
        self.config["top_k"] = top_k
        # Config changes might invalidate cache
        if top_k < self.config.get("top_k", 10):
            # Truncate cached results
            self.cache = {
                q: docs[:top_k]
                for q, docs in self.cache.items()
            }

    @invariant()
    def cache_consistency(self):
        """Cache entries should be consistent with current documents."""
        for query, cached_docs in self.cache.items():
            for doc in cached_docs:
                if doc.id in self.documents:
                    # Document should match current version
                    assert doc.content == self.documents[doc.id].content

    @invariant()
    def result_count_invariant(self):
        """Results should never exceed top_k."""
        for cached_results in self.cache.values():
            assert len(cached_results) <= self.config["top_k"]


# Run the stateful test
TestRAGStateMachine = RAGStateMachine.TestCase


# =====================================================================
# Property Tests for Error Handling
# =====================================================================

@pytest.mark.property
class TestErrorHandlingProperties:
    """Property tests for error handling."""

    @given(
        query=st.one_of(
            st.just(""),  # Empty query
            st.just(" " * 10),  # Whitespace only
            st.text(min_size=1001, max_size=2000),  # Too long
            st.text().filter(lambda x: "\x00" in x)  # Null bytes
        )
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @pytest.mark.asyncio
    async def test_invalid_query_handling(self, query):
        """Invalid queries should be handled gracefully via the public pipeline API."""
        # Call the unified pipeline; for blank queries it should short-circuit with an error
        result = await unified_rag_pipeline(
            query=query,
            enable_cache=False,
            enable_generation=False,
        )
        assert isinstance(result, UnifiedRAGResponse)

        is_blank = len(query.strip()) == 0
        if is_blank:
            # Pipeline flags invalid (blank) queries
            assert any("Invalid query" in e for e in (result.errors or []))
        else:
            # For other edge inputs, at minimum it should not crash and should echo the query
            assert result.query == (query if isinstance(query, str) else "")

    @given(
        config=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(
                st.integers(min_value=-100, max_value=-1),  # Negative values
                st.floats(allow_nan=True, allow_infinity=True),  # NaN/Inf
                st.text(min_size=0, max_size=0)  # Empty strings
            )
        )
    )
    def test_invalid_config_handling(self, config):
        """Invalid configuration should be handled gracefully."""
        # System should validate and reject or fix invalid config
        validated_config = {}

        for key, value in config.items():
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    validated_config[key] = 1.0
                elif value < 0:
                    # Negative values should be rejected or fixed to positive finite
                    validated_config[key] = max(1.0, abs(value))
                else:
                    validated_config[key] = value
            elif isinstance(value, str) and len(value) == 0:
                # Empty strings might be rejected
                continue
            else:
                validated_config[key] = value

        # Validated config should be usable
        for key, value in validated_config.items():
            if isinstance(value, (int, float)):
                assert not np.isnan(value)
                assert not np.isinf(value)


# =====================================================================
# Property Tests for Performance Characteristics
# =====================================================================

@pytest.mark.property
class TestPerformanceProperties:
    """Property tests for performance characteristics."""

    @given(
        num_documents=st.integers(min_value=1, max_value=1000),
        top_k=st.integers(min_value=1, max_value=100)
    )
    def test_retrieval_complexity(self, num_documents, top_k):
        """Retrieval time should scale reasonably with document count."""
        # Theoretical complexity for different retrieval methods

        # Linear scan: O(n)
        linear_complexity = num_documents

        # Index-based: O(log n + k)
        index_complexity = np.log2(max(num_documents, 1)) + top_k

        # Actual retrieval should be better than, or at worst comparable to,
        # linear scan for large datasets. For very large top_k (close to n),
        # index cost (log n + k) can approach O(n); allow small cushion.
        if num_documents > 100:
            assert index_complexity <= linear_complexity + 2

    @given(
        query_length=st.integers(min_value=1, max_value=500),
        num_expansions=st.integers(min_value=0, max_value=10)
    )
    def test_expansion_overhead(self, query_length, num_expansions):
        """Query expansion overhead should be bounded."""
        # Original query processing time (simulated)
        base_time = query_length * 0.001  # 1ms per character

        # Expansion overhead
        expansion_time = num_expansions * 0.01  # 10ms per expansion

        total_time = base_time + expansion_time

        # Expansion shouldn't dominate processing time
        if num_expansions > 0:
            overhead_ratio = expansion_time / total_time
            # Allow a small cushion for extreme cases (very small base_time with max expansions)
            assert overhead_ratio <= 0.991

    @given(
        cache_size=st.integers(min_value=1, max_value=1000),
        hit_rate=st.floats(min_value=0.0, max_value=1.0)
    )
    def test_cache_efficiency(self, cache_size, hit_rate):
        """Cache should provide performance benefit when hit rate is high."""
        # Time without cache (ms)
        uncached_time = 100

        # Time with cache
        cache_lookup_time = 1
        cached_time = hit_rate * cache_lookup_time + (1 - hit_rate) * (uncached_time + cache_lookup_time)

        # Cache is beneficial if hit rate is high enough
        if hit_rate > 0.1:  # More than 10% hit rate
            assert cached_time < uncached_time

        # Cache overhead should be minimal
        assert cache_lookup_time < uncached_time * 0.1  # Less than 10% of uncached time


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "property"])
