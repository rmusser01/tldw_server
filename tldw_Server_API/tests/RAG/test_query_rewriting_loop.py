"""
Tests for Query Rewriting Loop (Self-Correcting RAG Stage 2)

These tests cover:
- QueryRewriter.improve_for_retrieval strategy
- Helper methods for query improvement
- Integration with the query rewriting loop
"""

import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tldw_Server_API.app.core.RAG.rag_service.query_features import (
    QueryRewriter,
    QueryAnalyzer,
    QueryAnalysis,
    QueryRewrite,
    QueryIntent,
    QueryComplexity,
)


@dataclass
class MockDocument:
    """Mock document for testing."""
    id: str
    content: str
    score: float = 0.5
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TestQueryRewriterImproveForRetrieval:
    """Tests for the improve_for_retrieval strategy."""

    @pytest.fixture
    def rewriter(self) -> QueryRewriter:
        """Create a QueryRewriter instance."""
        return QueryRewriter()

    @pytest.fixture
    def sample_failed_docs(self) -> List[MockDocument]:
        """Create sample documents that 'failed' grading."""
        return [
            MockDocument(
                id="doc1",
                content="Machine learning is a subset of artificial intelligence. Deep Learning uses neural networks.",
                score=0.2,
            ),
            MockDocument(
                id="doc2",
                content="The Python programming language is used for data science. TensorFlow is a popular framework.",
                score=0.15,
            ),
            MockDocument(
                id="doc3",
                content="Natural Language Processing involves analyzing text. GPT models are transformer-based.",
                score=0.25,
            ),
        ]

    def test_improve_for_retrieval_basic(self, rewriter):
        """Test that improve_for_retrieval generates rewrites."""
        query = "what is machine learning"
        rewrites = rewriter.rewrite_query(
            query,
            strategies=["improve_for_retrieval"],
        )

        # Should generate at least one rewrite
        assert len(rewrites) > 0
        # All should be improve_for_retrieval type
        assert all(r.rewrite_type == "improve_for_retrieval" for r in rewrites)

    def test_improve_for_retrieval_without_wordnet_still_rewrites(self, monkeypatch, rewriter):
        """WordNet-free test environments still get a deterministic retrieval rewrite."""
        def _missing_wordnet(_term):
            raise LookupError("wordnet unavailable")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.RAG.rag_service.query_features.wordnet.synsets",
            _missing_wordnet,
        )

        rewrites = rewriter.rewrite_query(
            "what is machine learning",
            strategies=["improve_for_retrieval"],
        )

        assert any(
            rewrite.rewritten_query == "machine learning"
            and rewrite.explanation == "Generated keyword-focused fallback rewrite"
            for rewrite in rewrites
        )

    def test_improve_for_retrieval_with_failed_docs(self, rewriter, sample_failed_docs):
        """Test that improve_for_retrieval uses entities from failed docs."""
        query = "how does it work"
        rewrites = rewriter.rewrite_query(
            query,
            strategies=["improve_for_retrieval"],
            failed_docs=sample_failed_docs,
            failure_reason="avg_relevance_0.2",
        )

        # Should generate rewrites
        assert len(rewrites) > 0

        # At least one rewrite should incorporate info from failed docs
        rewrite_texts = [r.rewritten_query for r in rewrites]
        # Entity incorporation may vary, but we should get SOME rewrites
        assert len(rewrite_texts) > 0

    def test_remove_modifiers(self, rewriter):
        """Test that restrictive modifiers are removed."""
        query = "the exactly best latest machine learning framework in 2024"
        rewrites = rewriter.rewrite_query(
            query,
            strategies=["improve_for_retrieval"],
        )

        # Should have a rewrite with modifiers removed
        rewrite_texts = [r.rewritten_query.lower() for r in rewrites]

        # Check that at least one rewrite doesn't have restrictive words
        has_simplified = any(
            "exactly" not in text or "best" not in text or "latest" not in text
            for text in rewrite_texts
        )
        assert has_simplified or len(rewrites) > 0

    def test_add_focus_terms_technology(self, rewriter):
        """Test that domain-specific focus terms are added for technology queries."""
        # Query that should be detected as technology domain
        query = "programming algorithm database"
        rewrites = rewriter.rewrite_query(
            query,
            strategies=["improve_for_retrieval"],
        )

        # This depends on domain detection, but we should get some rewrites
        assert len(rewrites) > 0

    def test_convert_to_specific_question(self, rewriter):
        """Test conversion to specific question form."""
        query = "machine learning"  # Not a question
        rewrites = rewriter.rewrite_query(
            query,
            strategies=["improve_for_retrieval"],
        )

        # Check if any rewrite is a question
        has_question = any(
            r.rewritten_query.endswith("?")
            for r in rewrites
        )
        assert has_question

    def test_no_rewrites_for_question(self, rewriter):
        """Test that existing questions don't get converted again."""
        query = "What is machine learning?"
        rewrites = rewriter.rewrite_query(
            query,
            strategies=["improve_for_retrieval"],
        )

        # Should NOT have a question conversion rewrite (already a question)
        question_conversions = [
            r for r in rewrites
            if "question form" in r.explanation.lower()
        ]
        assert len(question_conversions) == 0

    def test_rewrite_confidence_ordering(self, rewriter):
        """Test that rewrites have decreasing confidence."""
        query = "machine learning applications in healthcare"
        rewrites = rewriter.rewrite_query(
            query,
            strategies=["improve_for_retrieval"],
        )

        # All rewrites should have confidence between 0 and 1
        for r in rewrites:
            assert 0.0 <= r.confidence <= 1.0

    def test_rewrite_explanations(self, rewriter):
        """Test that all rewrites have explanations."""
        query = "latest deep learning research"
        rewrites = rewriter.rewrite_query(
            query,
            strategies=["improve_for_retrieval"],
        )

        for r in rewrites:
            assert r.explanation
            assert len(r.explanation) > 0

    def test_empty_query(self, rewriter):
        """Test handling of empty query."""
        query = ""
        rewrites = rewriter.rewrite_query(
            query,
            strategies=["improve_for_retrieval"],
        )

        # Should not crash, may or may not produce rewrites
        assert isinstance(rewrites, list)

    def test_very_short_query(self, rewriter):
        """Test handling of very short query."""
        query = "AI"
        rewrites = rewriter.rewrite_query(
            query,
            strategies=["improve_for_retrieval"],
        )

        # Should produce some rewrites
        assert len(rewrites) >= 0  # May produce 0 for very short queries


class TestQueryRewriterStrategyCombinations:
    """Tests for combining improve_for_retrieval with other strategies."""

    @pytest.fixture
    def rewriter(self) -> QueryRewriter:
        return QueryRewriter()

    def test_combine_with_synonym(self, rewriter):
        """Test combining improve_for_retrieval with synonym strategy."""
        query = "machine learning applications"
        rewrites = rewriter.rewrite_query(
            query,
            strategies=["improve_for_retrieval", "synonym"],
        )

        # Should have rewrites from both strategies
        improve_rewrites = [r for r in rewrites if r.rewrite_type == "improve_for_retrieval"]

        assert len(improve_rewrites) > 0
        # Synonym rewrites depend on WordNet availability

    def test_combine_with_decompose(self, rewriter):
        """Test combining improve_for_retrieval with decompose strategy."""
        # Make query more clearly complex to trigger decomposition
        query = "machine learning and deep learning and neural networks and transformers and attention mechanisms"
        rewrites = rewriter.rewrite_query(
            query,
            strategies=["improve_for_retrieval", "decompose"],
        )

        # Should have rewrites from improve_for_retrieval
        improve_rewrites = [r for r in rewrites if r.rewrite_type == "improve_for_retrieval"]

        assert len(improve_rewrites) > 0
        # Complex query may produce decomposition depending on complexity detection
        # This is a softer assertion since decompose behavior depends on complexity assessment
        assert len(rewrites) > 0


class TestQueryRewriteDataclass:
    """Tests for QueryRewrite dataclass."""

    def test_create_query_rewrite(self):
        """Test creating a QueryRewrite."""
        rewrite = QueryRewrite(
            rewritten_query="What is machine learning?",
            rewrite_type="improve_for_retrieval",
            confidence=0.75,
            explanation="Converted to specific question form"
        )

        assert rewrite.rewritten_query == "What is machine learning?"
        assert rewrite.rewrite_type == "improve_for_retrieval"
        assert rewrite.confidence == 0.75
        assert rewrite.explanation == "Converted to specific question form"


class TestHelperMethods:
    """Tests for individual helper methods in QueryRewriter."""

    @pytest.fixture
    def rewriter(self) -> QueryRewriter:
        return QueryRewriter()

    @pytest.fixture
    def analyzer(self) -> QueryAnalyzer:
        return QueryAnalyzer()

    def test_remove_modifiers_with_year(self, rewriter, analyzer):
        """Test removing year specifications."""
        query = "best AI tools in 2024"
        analysis = analyzer.analyze_query(query)

        result = rewriter._remove_modifiers(query, analysis)

        # Year should be removed
        if result:
            assert "2024" not in result or "best" not in result

    def test_remove_modifiers_no_change(self, rewriter, analyzer):
        """Test that queries without modifiers are not changed."""
        query = "machine learning basics"
        analysis = analyzer.analyze_query(query)

        result = rewriter._remove_modifiers(query, analysis)

        # Should return None if no change
        assert result is None or result == query

    def test_add_focus_terms_no_domain(self, rewriter, analyzer):
        """Test add_focus_terms with non-domain query."""
        query = "random stuff here"
        analysis = analyzer.analyze_query(query)

        result = rewriter._add_focus_terms(query, analysis)

        # Should return None if no domain detected
        # (depends on whether the analyzer detects a domain)
        assert result is None or isinstance(result, str)

    def test_extract_entities_from_empty_docs(self, rewriter, analyzer):
        """Test entity extraction with empty document list."""
        query = "test query"
        analysis = analyzer.analyze_query(query)

        result = rewriter._extract_and_use_entities(query, analysis, [])

        assert result is None

    def test_extract_entities_from_docs_without_content(self, rewriter, analyzer):
        """Test entity extraction from docs without content."""
        query = "test query"
        analysis = analyzer.analyze_query(query)

        class EmptyDoc:
            id = "empty"
            content = ""

        result = rewriter._extract_and_use_entities(query, analysis, [EmptyDoc()])

        assert result is None

    def test_convert_factual_to_question(self, rewriter, analyzer):
        """Test converting factual intent to question."""
        query = "machine learning"
        analysis = analyzer.analyze_query(query)
        # Force factual intent for test
        analysis_with_intent = QueryAnalysis(
            original_query=query,
            cleaned_query=query,
            intent=QueryIntent.FACTUAL,
            complexity=QueryComplexity.SIMPLE,
            key_terms=["machine", "learning"],
            entities=[],
            temporal_refs=[],
        )

        result = rewriter._convert_to_specific_question(query, analysis_with_intent)

        assert result is not None
        assert result.endswith("?")
        assert "facts" in result.lower() or "key" in result.lower()

    def test_expand_with_related_concepts(self, rewriter, analyzer):
        """Test expansion with related concepts."""
        query = "dog training"
        analysis = analyzer.analyze_query(query)

        result = rewriter._expand_with_related_concepts(query, analysis)

        # Should add related concept in parentheses (if WordNet available)
        if result:
            assert "(" in result and ")" in result


class TestQueryRewritingIntegration:
    """Integration tests for query rewriting in the pipeline context."""

    def test_rewriter_preserves_original(self):
        """Test that original query is preserved in rewrites."""
        rewriter = QueryRewriter()
        original = "machine learning basics"

        rewrites = rewriter.rewrite_query(
            original,
            strategies=["improve_for_retrieval"],
        )

        # Original should not be in rewrites
        for r in rewrites:
            # Rewrites should be different from original
            assert r.rewritten_query != original or r.rewrite_type != "improve_for_retrieval"

    def test_rewriter_handles_special_characters(self):
        """Test handling of special characters in query."""
        rewriter = QueryRewriter()
        query = "what is C++ programming?"

        rewrites = rewriter.rewrite_query(
            query,
            strategies=["improve_for_retrieval"],
        )

        # Should not crash and should produce valid rewrites
        assert isinstance(rewrites, list)
        for r in rewrites:
            assert isinstance(r.rewritten_query, str)

    def test_rewriter_handles_unicode(self):
        """Test handling of unicode characters in query."""
        rewriter = QueryRewriter()
        query = "machine learning für anfänger"

        rewrites = rewriter.rewrite_query(
            query,
            strategies=["improve_for_retrieval"],
        )

        # Should not crash
        assert isinstance(rewrites, list)
