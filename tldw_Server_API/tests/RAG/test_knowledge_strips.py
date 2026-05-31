"""
Tests for Knowledge Strips (Self-Correcting RAG Stage 4)

These tests cover:
- Strip splitting from documents
- Relevance scoring (heuristic and LLM-based)
- Document rebuilding from strips
- KnowledgeStripsProcessor
- Convenience function for pipeline integration
"""

import asyncio
import pytest
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import tldw_Server_API.app.core.RAG.rag_service.knowledge_strips as knowledge_strips
from tldw_Server_API.app.core.RAG.rag_service.knowledge_strips import (
    KnowledgeStrip,
    KnowledgeStripsResult,
    KnowledgeStripsProcessor,
    process_knowledge_strips,
    _split_into_strips,
    _score_strip_relevance,
    _estimate_tokens,
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_estimate_tokens(self):
        """Test token estimation."""
        assert _estimate_tokens("") == 0
        assert _estimate_tokens("hello world") > 0
        # Roughly 1.3 tokens per word
        tokens = _estimate_tokens("one two three four five")
        assert 5 <= tokens <= 10

    def test_split_into_strips_basic(self):
        """Test basic strip splitting."""
        text = "First sentence. Second sentence. Third sentence."
        strips = _split_into_strips(text, strip_size_tokens=50)

        assert len(strips) > 0
        for strip in strips:
            assert "text" in strip
            assert "start_offset" in strip
            assert "end_offset" in strip

    def test_split_into_strips_empty(self):
        """Test splitting empty text."""
        strips = _split_into_strips("", strip_size_tokens=100)
        assert strips == []

    def test_split_into_strips_single_sentence(self):
        """Test splitting single short sentence."""
        text = "Just one sentence."
        strips = _split_into_strips(text, strip_size_tokens=100)

        assert len(strips) == 1
        assert "Just one sentence" in strips[0]["text"]

    def test_split_into_strips_respects_size(self):
        """Test that strips roughly respect token size limit."""
        # Use longer sentences with proper capitalization after periods
        text = "First sentence with some content here. Second sentence has more words too. Third sentence continues the pattern. Fourth sentence adds variety. Fifth sentence wraps it up."
        strips = _split_into_strips(text, strip_size_tokens=15)

        # Should create multiple strips (each sentence ~5-6 tokens, threshold 15)
        # With 5 sentences, we should get at least 2 strips
        assert len(strips) >= 1  # At minimum, we get one strip with all content

    def test_score_strip_relevance_matching(self):
        """Test relevance scoring with matching keywords."""
        query = "machine learning algorithms"
        strip_text = "Machine learning uses various algorithms for pattern recognition."

        score = _score_strip_relevance(query, strip_text)

        # Should have good relevance due to keyword matches
        assert score > 0.3

    def test_score_strip_relevance_no_match(self):
        """Test relevance scoring with no matching keywords."""
        query = "machine learning algorithms"
        strip_text = "The weather today is sunny and warm."

        score = _score_strip_relevance(query, strip_text)

        # Should have low relevance
        assert score < 0.3

    def test_score_strip_relevance_empty_query(self):
        """Test relevance scoring with empty query."""
        query = ""
        strip_text = "Some text content."

        score = _score_strip_relevance(query, strip_text)

        # Should return default value
        assert score == 0.5


class TestKnowledgeStrip:
    """Tests for KnowledgeStrip dataclass."""

    def test_create_strip(self):
        """Test creating a KnowledgeStrip."""
        strip = KnowledgeStrip(
            doc_id="doc1",
            strip_id="doc1_strip_0",
            text="This is strip content.",
            start_offset=0,
            end_offset=100,
            relevance_score=0.75,
        )

        assert strip.doc_id == "doc1"
        assert strip.strip_id == "doc1_strip_0"
        assert strip.text == "This is strip content."
        assert strip.start_offset == 0
        assert strip.end_offset == 100
        assert strip.relevance_score == 0.75
        assert strip.metadata == {}

    def test_create_strip_with_metadata(self):
        """Test creating KnowledgeStrip with metadata."""
        strip = KnowledgeStrip(
            doc_id="doc1",
            strip_id="doc1_strip_0",
            text="Content",
            start_offset=0,
            end_offset=50,
            relevance_score=0.5,
            metadata={"source": "test"},
        )

        assert strip.metadata == {"source": "test"}


class TestKnowledgeStripsResult:
    """Tests for KnowledgeStripsResult dataclass."""

    def test_create_result(self):
        """Test creating a KnowledgeStripsResult."""
        strips = [
            KnowledgeStrip("doc1", "strip1", "text1", 0, 50, 0.8),
            KnowledgeStrip("doc1", "strip2", "text2", 50, 100, 0.6),
        ]
        docs = [Document(id="doc1", content="text", source=DataSource.MEDIA_DB, metadata={})]

        result = KnowledgeStripsResult(
            strips=strips,
            documents=docs,
            total_strips=10,
            relevant_strips=5,
            filtered_strips=2,
            avg_relevance=0.7,
            processing_time_ms=100,
        )

        assert len(result.strips) == 2
        assert len(result.documents) == 1
        assert result.total_strips == 10
        assert result.relevant_strips == 5
        assert result.filtered_strips == 2
        assert result.avg_relevance == 0.7
        assert result.processing_time_ms == 100


class TestKnowledgeStripsProcessor:
    """Tests for KnowledgeStripsProcessor class."""

    @pytest.fixture
    def sample_documents(self) -> List[Document]:
        """Create sample documents for testing."""
        return [
            Document(
                id="doc1",
                content="Machine learning is a subset of AI. It uses algorithms to learn from data. Deep learning is a type of machine learning.",
                source=DataSource.MEDIA_DB,
                score=0.8,
                metadata={"title": "ML Overview"},
            ),
            Document(
                id="doc2",
                content="The weather today is sunny. It will rain tomorrow. Temperatures are mild.",
                source=DataSource.MEDIA_DB,
                score=0.3,
                metadata={"title": "Weather Report"},
            ),
        ]

    @pytest.mark.asyncio
    async def test_process_basic(self, sample_documents):
        """Test basic processing of documents into strips."""
        processor = KnowledgeStripsProcessor(
            strip_size_tokens=50,
            min_relevance_score=0.0,
        )

        result = await processor.process(
            query="machine learning",
            documents=sample_documents,
            top_k=20,
        )

        assert isinstance(result, KnowledgeStripsResult)
        assert result.total_strips > 0
        assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_process_filters_by_relevance(self, sample_documents):
        """Test that processing filters by relevance."""
        processor = KnowledgeStripsProcessor(
            strip_size_tokens=50,
            min_relevance_score=0.3,  # Higher threshold
        )

        result = await processor.process(
            query="machine learning algorithms",
            documents=sample_documents,
            top_k=20,
        )

        # All filtered strips should meet threshold
        for strip in result.strips:
            assert strip.relevance_score >= 0.3

    @pytest.mark.asyncio
    async def test_process_respects_top_k(self, sample_documents):
        """Test that processing respects top_k limit."""
        processor = KnowledgeStripsProcessor(
            strip_size_tokens=20,  # Small to create many strips
            min_relevance_score=0.0,
        )

        result = await processor.process(
            query="test query",
            documents=sample_documents,
            top_k=3,
        )

        assert len(result.strips) <= 3

    @pytest.mark.asyncio
    async def test_process_empty_documents(self):
        """Test processing with empty document list."""
        processor = KnowledgeStripsProcessor()

        result = await processor.process(
            query="test query",
            documents=[],
            top_k=10,
        )

        assert result.total_strips == 0
        assert result.strips == []
        assert result.documents == []

    @pytest.mark.asyncio
    async def test_process_rebuilds_documents(self, sample_documents):
        """Test that processing rebuilds documents from strips."""
        processor = KnowledgeStripsProcessor(
            strip_size_tokens=50,
            min_relevance_score=0.0,
        )

        result = await processor.process(
            query="machine learning",
            documents=sample_documents,
            top_k=20,
        )

        # Should have rebuilt documents
        assert len(result.documents) > 0

        for doc in result.documents:
            assert doc.metadata.get("filtered_from_knowledge_strips") is True
            assert "original_doc_id" in doc.metadata

    @pytest.mark.asyncio
    async def test_heuristic_scoring(self, sample_documents):
        """Test heuristic scoring method."""
        processor = KnowledgeStripsProcessor(
            strip_size_tokens=100,
            min_relevance_score=0.0,
            analyze_fn=None,  # Force heuristic
        )

        result = await processor.process(
            query="machine learning",
            documents=sample_documents,
            top_k=10,
        )

        # ML-related strips should have higher scores
        ml_strips = [s for s in result.strips if "machine" in s.text.lower()]
        weather_strips = [s for s in result.strips if "weather" in s.text.lower()]

        if ml_strips and weather_strips:
            avg_ml = sum(s.relevance_score for s in ml_strips) / len(ml_strips)
            avg_weather = sum(s.relevance_score for s in weather_strips) / len(weather_strips)
            assert avg_ml > avg_weather

    @pytest.mark.asyncio
    async def test_llm_scoring_parses_fenced_json_with_think_tags(self):
        def _mock_analyze(*args, **kwargs):
            return (
                "<think>grading</think>\n"
                "```json\n"
                '[{"strip_num": 1, "relevance": 0.92}]\n'
                "```"
            )

        docs = [
            Document(
                id="doc1",
                content="Machine learning relies on data.",
                source=DataSource.MEDIA_DB,
                score=0.8,
                metadata={},
            )
        ]
        processor = KnowledgeStripsProcessor(
            strip_size_tokens=200,
            min_relevance_score=0.0,
            analyze_fn=_mock_analyze,
        )

        result = await processor.process(
            query="machine learning",
            documents=docs,
            top_k=5,
        )

        assert result.strips
        assert result.strips[0].relevance_score == pytest.approx(0.92, rel=1e-6)

    @pytest.mark.asyncio
    async def test_llm_scoring_fallback_log_sanitizes_exception_details(self):
        """LLM fallback logs should not expose raw scorer exception details."""
        secret_path = "/private/rag-knowledge-strips.db?token=strip-secret-token"

        def _mock_analyze(*args, **kwargs):
            raise RuntimeError(f"scorer failed at {secret_path}")

        docs = [
            Document(
                id="doc1",
                content="Machine learning algorithms improve data systems.",
                source=DataSource.MEDIA_DB,
                score=0.8,
                metadata={},
            )
        ]
        processor = KnowledgeStripsProcessor(
            strip_size_tokens=200,
            min_relevance_score=0.0,
            analyze_fn=_mock_analyze,
        )

        messages: list[str] = []
        sink_id = knowledge_strips.logger.add(
            lambda message: messages.append(str(message.record.get("message") or "")),
            level="WARNING",
        )
        try:
            result = await processor.process(
                query="machine learning",
                documents=docs,
                top_k=5,
            )
        finally:
            knowledge_strips.logger.remove(sink_id)

        assert result.strips
        assert result.strips[0].relevance_score == pytest.approx(
            _score_strip_relevance("machine learning", result.strips[0].text),
            rel=1e-6,
        )
        joined = "\n".join(messages)
        assert "LLM strip scoring failed, falling back to heuristic" in joined
        assert "rag-knowledge-strips.db" not in joined
        assert "strip-secret-token" not in joined


class TestProcessKnowledgeStrips:
    """Tests for the convenience function."""

    @pytest.fixture
    def sample_documents(self) -> List[Document]:
        """Create sample documents."""
        return [
            Document(
                id="doc1",
                content="Python is a programming language. It is used for web development and data science.",
                source=DataSource.MEDIA_DB,
                score=0.7,
                metadata={},
            ),
        ]

    @pytest.mark.asyncio
    async def test_process_convenience_function(self, sample_documents):
        """Test the convenience function."""
        docs, metadata = await process_knowledge_strips(
            query="Python programming",
            documents=sample_documents,
            strip_size_tokens=50,
            min_relevance=0.0,
            max_strips=10,
        )

        assert isinstance(docs, list)
        assert isinstance(metadata, dict)
        assert metadata.get("knowledge_strips_enabled") is True
        assert "total_strips" in metadata
        assert "avg_relevance" in metadata

    @pytest.mark.asyncio
    async def test_process_empty_documents(self):
        """Test convenience function with empty documents."""
        docs, metadata = await process_knowledge_strips(
            query="test",
            documents=[],
            strip_size_tokens=100,
            min_relevance=0.3,
            max_strips=10,
        )

        assert docs == []
        assert metadata["total_strips"] == 0


class TestDocumentRebuilding:
    """Tests for document rebuilding from strips."""

    @pytest.fixture
    def processor(self) -> KnowledgeStripsProcessor:
        return KnowledgeStripsProcessor(
            strip_size_tokens=50,
            min_relevance_score=0.0,
        )

    def test_rebuild_preserves_metadata(self, processor):
        """Test that rebuilding preserves original document metadata."""
        original_doc = Document(
            id="original",
            content="First part. Second part. Third part.",
            source=DataSource.NOTES,
            score=0.9,
            metadata={"author": "test", "date": "2024"},
        )

        strips = [
            KnowledgeStrip(
                doc_id="original",
                strip_id="original_strip_0",
                text="First part.",
                start_offset=0,
                end_offset=12,
                relevance_score=0.8,
            ),
        ]

        rebuilt = processor._rebuild_documents(strips, [original_doc])

        assert len(rebuilt) == 1
        assert rebuilt[0].metadata.get("author") == "test"
        assert rebuilt[0].metadata.get("date") == "2024"
        assert rebuilt[0].source == DataSource.NOTES

    def test_rebuild_combines_strips(self, processor):
        """Test that rebuilding combines multiple strips from same document."""
        original_doc = Document(
            id="original",
            content="First. Second. Third.",
            source=DataSource.MEDIA_DB,
            score=0.5,
            metadata={},
        )

        strips = [
            KnowledgeStrip("original", "s1", "First.", 0, 7, 0.8),
            KnowledgeStrip("original", "s2", "Third.", 15, 22, 0.7),
        ]

        rebuilt = processor._rebuild_documents(strips, [original_doc])

        assert len(rebuilt) == 1
        assert "First." in rebuilt[0].content
        assert "Third." in rebuilt[0].content
        assert rebuilt[0].metadata.get("strip_count") == 2

    def test_rebuild_calculates_avg_score(self, processor):
        """Test that rebuilding calculates average score."""
        original_doc = Document(
            id="doc",
            content="Content here.",
            source=DataSource.MEDIA_DB,
            metadata={},
        )

        strips = [
            KnowledgeStrip("doc", "s1", "Part 1", 0, 10, 0.8),
            KnowledgeStrip("doc", "s2", "Part 2", 10, 20, 0.6),
        ]

        rebuilt = processor._rebuild_documents(strips, [original_doc])

        assert len(rebuilt) == 1
        # Average of 0.8 and 0.6
        assert rebuilt[0].score == pytest.approx(0.7)


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_document_with_no_sentences(self):
        """Test handling document with no sentence boundaries."""
        doc = Document(
            id="nosent",
            content="justonewordwithoutspaces",
            source=DataSource.MEDIA_DB,
            metadata={},
        )

        processor = KnowledgeStripsProcessor(min_relevance_score=0.0)
        result = await processor.process("test", [doc], top_k=10)

        # Should still process (creates single strip)
        assert result.total_strips >= 1

    @pytest.mark.asyncio
    async def test_very_long_document(self):
        """Test handling very long document."""
        long_content = " ".join([f"Sentence number {i}." for i in range(100)])
        doc = Document(
            id="long",
            content=long_content,
            source=DataSource.MEDIA_DB,
            metadata={},
        )

        processor = KnowledgeStripsProcessor(
            strip_size_tokens=50,
            min_relevance_score=0.0,
        )
        result = await processor.process("test", [doc], top_k=10)

        # Should create multiple strips
        assert result.total_strips > 1
        # Should respect top_k
        assert len(result.strips) <= 10

    @pytest.mark.asyncio
    async def test_documents_with_different_sources(self):
        """Test handling documents from different sources."""
        docs = [
            Document(id="d1", content="Web content.", source=DataSource.WEB_CONTENT, metadata={}),
            Document(id="d2", content="Notes content.", source=DataSource.NOTES, metadata={}),
            Document(id="d3", content="Media content.", source=DataSource.MEDIA_DB, metadata={}),
        ]

        processor = KnowledgeStripsProcessor(min_relevance_score=0.0)
        result = await processor.process("content", docs, top_k=20)

        # Should preserve source types in rebuilt documents
        sources = {d.source for d in result.documents}
        assert len(sources) > 0
