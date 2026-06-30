"""
Tests for Document Grader (Self-Correcting RAG Stage 1)

These tests cover:
- GradingConfig defaults and customization
- Single document grading with LLM
- Batch document grading
- Fallback behavior when LLM is unavailable
- Filter function for relevance threshold
"""

import asyncio
import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from loguru import logger

from tldw_Server_API.app.core.RAG.rag_service.document_grader import (
    DocumentGrader,
    GradingConfig,
    GradingResult,
    GradingBatchResult,
    grade_and_filter_documents,
    _resolve_grading_config,
    StructuredOutputParseError,
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


class TestGradingConfig:
    """Tests for GradingConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = GradingConfig()

        assert config.provider is None
        assert config.model is None
        assert config.batch_size == 5
        assert config.timeout_seconds == 30.0
        assert config.fallback_to_score is True
        assert config.fallback_min_score == 0.3
        assert config.temperature == 0.1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = GradingConfig(
            provider="anthropic",
            model="claude-3-haiku",
            batch_size=10,
            timeout_seconds=60.0,
            fallback_to_score=False,
            fallback_min_score=0.5,
            temperature=0.2,
        )

        assert config.provider == "anthropic"
        assert config.model == "claude-3-haiku"
        assert config.batch_size == 10
        assert config.timeout_seconds == 60.0
        assert config.fallback_to_score is False
        assert config.fallback_min_score == 0.5
        assert config.temperature == 0.2

    def test_resolve_grading_config_fallbacks(self):
        """Test config resolution falls back to default provider."""
        with patch(
            "tldw_Server_API.app.core.config.load_and_log_configs",
            return_value={},
        ):
            provider, model, temperature = _resolve_grading_config()

        assert provider == "openai"
        assert model is None
        assert temperature == 0.1


class TestDocumentGrader:
    """Tests for DocumentGrader class."""

    @pytest.fixture
    def mock_analyze_relevant(self):
        """Mock analyze function that returns relevant response."""
        def analyze(*args, **kwargs):
            return '{"is_relevant": true, "relevance_score": 0.85, "reasoning": "Document discusses the topic directly."}'
        return analyze

    @pytest.fixture
    def mock_analyze_irrelevant(self):
        """Mock analyze function that returns irrelevant response."""
        def analyze(*args, **kwargs):
            return '{"is_relevant": false, "relevance_score": 0.15, "reasoning": "Document is unrelated to the query."}'
        return analyze

    @pytest.fixture
    def mock_analyze_malformed(self):
        """Mock analyze function that returns malformed response."""
        def analyze(*args, **kwargs):
            return "This document is relevant and useful."
        return analyze

    @pytest.fixture
    def sample_documents(self) -> List[MockDocument]:
        """Create sample documents for testing."""
        return [
            MockDocument(
                id="doc1",
                content="Machine learning is a subset of artificial intelligence.",
                score=0.8,
            ),
            MockDocument(
                id="doc2",
                content="The weather today is sunny and warm.",
                score=0.3,
            ),
            MockDocument(
                id="doc3",
                content="Neural networks are inspired by biological neurons.",
                score=0.7,
            ),
            MockDocument(
                id="doc4",
                content="Cooking pasta requires boiling water.",
                score=0.1,
            ),
            MockDocument(
                id="doc5",
                content="Deep learning uses multiple neural network layers.",
                score=0.75,
            ),
        ]

    @pytest.mark.asyncio
    async def test_grade_single_document_relevant(self, mock_analyze_relevant):
        """Test grading a single document that is relevant."""
        grader = DocumentGrader(analyze_fn=mock_analyze_relevant)
        doc = MockDocument(id="test1", content="ML content", score=0.5)

        result = await grader.grade_document(
            query="What is machine learning?",
            document=doc,
        )

        assert isinstance(result, GradingResult)
        assert result.document_id == "test1"
        assert result.is_relevant is True
        assert result.relevance_score == 0.85
        assert result.method == "llm"
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_grade_single_document_irrelevant(self, mock_analyze_irrelevant):
        """Test grading a single document that is irrelevant."""
        grader = DocumentGrader(analyze_fn=mock_analyze_irrelevant)
        doc = MockDocument(id="test2", content="Weather content", score=0.5)

        result = await grader.grade_document(
            query="What is machine learning?",
            document=doc,
        )

        assert result.is_relevant is False
        assert result.relevance_score == 0.15
        assert result.method == "llm"

    @pytest.mark.asyncio
    async def test_grade_document_parses_fenced_json_with_think_tags(self):
        def analyze(*args, **kwargs):
            return (
                "<think>grading</think>\n"
                "```json\n"
                '{"is_relevant": true, "relevance_score": 0.77, "reasoning": "Relevant."}\n'
                "```"
            )

        grader = DocumentGrader(analyze_fn=analyze)
        doc = MockDocument(id="test_fenced", content="ML content", score=0.5)
        result = await grader.grade_document(
            query="What is machine learning?",
            document=doc,
        )

        assert result.is_relevant is True
        assert result.relevance_score == 0.77
        assert result.method == "llm"

    @pytest.mark.asyncio
    async def test_grade_document_malformed_response(self, mock_analyze_malformed):
        """Test grading with malformed LLM response uses heuristic parsing."""
        grader = DocumentGrader(analyze_fn=mock_analyze_malformed)
        doc = MockDocument(id="test3", content="Content", score=0.5)

        result = await grader.grade_document(
            query="Test query",
            document=doc,
        )

        # Should use heuristic parsing since JSON parsing fails
        assert result.method == "llm_heuristic"
        # "relevant" and "useful" are in the response
        assert result.is_relevant is True

    @pytest.mark.asyncio
    async def test_grade_document_parse_fallback_sanitizes_parser_error_logs(self):
        """Test parse fallback logs avoid raw parser exception details."""
        leaked_path = "/Users/example/private/grading-response.json"
        leaked_token = "sk-test-parser-secret"
        parser_error = (
            f"parser failed while reading {leaked_path} "
            f"with token {leaked_token}"
        )

        def analyze(*args, **kwargs):
            return (
                f"This relevant useful response references {leaked_path} "
                f"and token {leaked_token}"
            )

        captured_logs = []
        sink_id = logger.add(captured_logs.append, format="{message}")
        try:
            with patch(
                "tldw_Server_API.app.core.RAG.rag_service.document_grader.parse_structured_output",
                side_effect=StructuredOutputParseError(parser_error),
            ):
                grader = DocumentGrader(analyze_fn=analyze)
                doc = MockDocument(id="parse_safe_doc", content="Content", score=0.1)

                result = await grader.grade_document("Query", doc)
        finally:
            logger.remove(sink_id)

        assert result.method == "llm_heuristic"
        assert result.is_relevant is True
        assert result.relevance_score == 0.7
        assert result.reasoning == "Heuristic parsing from LLM response"

        combined_result_text = f"{result.reasoning} {result.metadata}"
        combined_log_text = " ".join(str(message) for message in captured_logs)
        for leaked_fragment in (
            parser_error,
            leaked_path,
            leaked_token,
        ):
            assert leaked_fragment not in combined_result_text
            assert leaked_fragment not in combined_log_text

    @pytest.mark.asyncio
    async def test_grade_document_fallback_to_score(self):
        """Test fallback to score when analyze function raises exception."""
        def failing_analyze(*args, **kwargs):
            raise RuntimeError("LLM unavailable")

        config = GradingConfig(fallback_to_score=True, fallback_min_score=0.4)
        grader = DocumentGrader(analyze_fn=failing_analyze, config=config)

        # High score document
        doc_high = MockDocument(id="high", content="Content", score=0.8)
        result_high = await grader.grade_document("Query", doc_high)
        assert result_high.is_relevant is True
        assert result_high.relevance_score == 0.8
        assert result_high.method == "score_fallback"

        # Low score document
        doc_low = MockDocument(id="low", content="Content", score=0.2)
        result_low = await grader.grade_document("Query", doc_low)
        assert result_low.is_relevant is False
        assert result_low.relevance_score == 0.2
        assert result_low.method == "score_fallback"

    @pytest.mark.asyncio
    async def test_grade_document_score_fallback_sanitizes_error_details(self):
        """Test score fallback does not expose raw exception details."""
        leaked_error = (
            "LLM failed for /Users/example/private/doc.txt "
            "with token sk-test-secret-value"
        )

        def failing_analyze(*args, **kwargs):
            raise RuntimeError(leaked_error)

        captured_logs = []
        sink_id = logger.add(captured_logs.append, format="{message}")
        try:
            config = GradingConfig(fallback_to_score=True, fallback_min_score=0.4)
            grader = DocumentGrader(analyze_fn=failing_analyze, config=config)
            doc = MockDocument(id="safe_doc", content="Content", score=0.8)

            result = await grader.grade_document("Query", doc)
        finally:
            logger.remove(sink_id)

        assert result.method == "score_fallback"
        assert result.is_relevant is True
        assert result.relevance_score == 0.8
        assert result.reasoning == "Using retrieval score as fallback"
        assert result.metadata == {"error": "grading_error"}

        combined_result_text = f"{result.reasoning} {result.metadata}"
        combined_log_text = " ".join(str(message) for message in captured_logs)
        for leaked_fragment in (
            leaked_error,
            "/Users/example/private/doc.txt",
            "sk-test-secret-value",
        ):
            assert leaked_fragment not in combined_result_text
            assert leaked_fragment not in combined_log_text

    @pytest.mark.asyncio
    async def test_grade_document_score_fallback_passes_sanitized_error_label(self, monkeypatch):
        """Test score fallback boundary receives a fixed label for provider failures."""
        leaked_error = (
            "provider failed while reading /Users/example/private/request.json "
            "with token sk-test-boundary-secret"
        )
        fallback_errors = []

        def failing_analyze(*args, **kwargs):
            raise RuntimeError(leaked_error)

        original_fallback = DocumentGrader._fallback_to_score

        def tracking_fallback(self, doc_id, doc_score, start_time, error=None):
            fallback_errors.append(error)
            return original_fallback(self, doc_id, doc_score, start_time, error)

        monkeypatch.setattr(DocumentGrader, "_fallback_to_score", tracking_fallback)

        captured_logs = []
        sink_id = logger.add(captured_logs.append, format="{message}")
        try:
            config = GradingConfig(fallback_to_score=True, fallback_min_score=0.4)
            grader = DocumentGrader(analyze_fn=failing_analyze, config=config)
            doc = MockDocument(id="safe_doc", content="Content", score=0.8)

            result = await grader.grade_document("Query", doc)
        finally:
            logger.remove(sink_id)

        assert fallback_errors == ["grading_error"]
        assert result.method == "score_fallback"
        assert result.is_relevant is True
        assert result.relevance_score == 0.8
        assert result.metadata == {"error": "grading_error"}

        combined_result_text = f"{result.reasoning} {result.metadata}"
        combined_log_text = " ".join(str(message) for message in captured_logs)
        for leaked_fragment in (
            leaked_error,
            "/Users/example/private/request.json",
            "sk-test-boundary-secret",
        ):
            assert leaked_fragment not in combined_result_text
            assert leaked_fragment not in combined_log_text

    @pytest.mark.asyncio
    async def test_grade_document_fallback_disabled(self):
        """Test behavior when fallback is disabled and LLM fails."""
        def failing_analyze(*args, **kwargs):
            raise RuntimeError("LLM unavailable")

        config = GradingConfig(fallback_to_score=False)
        grader = DocumentGrader(analyze_fn=failing_analyze, config=config)

        doc = MockDocument(id="test", content="Content", score=0.9)
        result = await grader.grade_document("Query", doc)

        assert result.is_relevant is False
        assert result.relevance_score == 0.0
        assert result.method == "error_fallback"

    @pytest.mark.asyncio
    async def test_grade_document_error_fallback_sanitizes_error_details(self):
        """Test disabled fallback does not expose raw exception details."""
        leaked_error = (
            "provider failure at /var/private/tldw/request.json "
            "using token secret-token-123"
        )

        def failing_analyze(*args, **kwargs):
            raise RuntimeError(leaked_error)

        config = GradingConfig(fallback_to_score=False)
        grader = DocumentGrader(analyze_fn=failing_analyze, config=config)
        doc = MockDocument(id="safe_doc", content="Content", score=0.9)

        result = await grader.grade_document("Query", doc)

        assert result.method == "error_fallback"
        assert result.is_relevant is False
        assert result.relevance_score == 0.0
        assert result.reasoning == "Grading failed and fallback disabled"
        assert result.metadata == {"error": "grading_error"}

        combined_result_text = f"{result.reasoning} {result.metadata}"
        for leaked_fragment in (
            leaked_error,
            "/var/private/tldw/request.json",
            "secret-token-123",
        ):
            assert leaked_fragment not in combined_result_text

    @pytest.mark.asyncio
    async def test_grade_documents_batch(self, mock_analyze_relevant, sample_documents):
        """Test batch grading of multiple documents."""
        config = GradingConfig(batch_size=2)
        grader = DocumentGrader(analyze_fn=mock_analyze_relevant, config=config)

        result = await grader.grade_documents(
            query="What is machine learning?",
            documents=sample_documents,
        )

        assert isinstance(result, GradingBatchResult)
        assert result.total_count == 5
        assert len(result.results) == 5
        assert result.relevant_count == 5  # All marked relevant by mock
        assert result.avg_relevance == 0.85  # All have same score from mock
        assert result.total_latency_ms >= 0

    @pytest.mark.asyncio
    async def test_grade_documents_empty_list(self, mock_analyze_relevant):
        """Test batch grading with empty document list."""
        grader = DocumentGrader(analyze_fn=mock_analyze_relevant)

        result = await grader.grade_documents(
            query="Test query",
            documents=[],
        )

        assert result.total_count == 0
        assert result.relevant_count == 0
        assert result.avg_relevance == 0.0
        assert len(result.results) == 0

    @pytest.mark.asyncio
    async def test_filter_relevant_above_threshold(self, sample_documents):
        """Test filtering documents above relevance threshold."""
        # Mock that returns varied relevance scores based on document
        call_count = [0]
        def mock_analyze(*args, **kwargs):
            # Return different scores for different calls
            scores = [0.9, 0.2, 0.8, 0.1, 0.7]
            idx = call_count[0] % len(scores)
            call_count[0] += 1
            return f'{{"is_relevant": {str(scores[idx] >= 0.5).lower()}, "relevance_score": {scores[idx]}, "reasoning": "Test"}}'

        grader = DocumentGrader(analyze_fn=mock_analyze)

        filtered_docs, metadata = await grader.filter_relevant(
            query="What is machine learning?",
            documents=sample_documents,
            threshold=0.5,
        )

        # Documents with scores 0.9, 0.8, 0.7 should pass (indices 0, 2, 4)
        assert len(filtered_docs) == 3
        assert metadata["total_graded"] == 5
        assert metadata["filtered_count"] == 3
        assert metadata["removed_count"] == 2
        assert metadata["threshold"] == 0.5

    @pytest.mark.asyncio
    async def test_filter_relevant_none_above_threshold(self, mock_analyze_irrelevant, sample_documents):
        """Test filtering when no documents meet threshold."""
        grader = DocumentGrader(analyze_fn=mock_analyze_irrelevant)

        filtered_docs, metadata = await grader.filter_relevant(
            query="Test query",
            documents=sample_documents,
            threshold=0.5,
        )

        # All documents have score 0.15, below threshold
        assert len(filtered_docs) == 0
        assert metadata["filtered_count"] == 0
        assert metadata["removed_count"] == 5

    @pytest.mark.asyncio
    async def test_filter_relevant_empty_input(self, mock_analyze_relevant):
        """Test filtering with empty document list."""
        grader = DocumentGrader(analyze_fn=mock_analyze_relevant)

        filtered_docs, metadata = await grader.filter_relevant(
            query="Test query",
            documents=[],
            threshold=0.5,
        )

        assert filtered_docs == []
        assert metadata.get("grading_skipped") is True

    @pytest.mark.asyncio
    async def test_content_truncation(self, mock_analyze_relevant):
        """Test that long document content is truncated."""
        grader = DocumentGrader(analyze_fn=mock_analyze_relevant)

        # Create document with very long content
        long_content = "A" * 10000
        doc = MockDocument(id="long", content=long_content, score=0.5)

        result = await grader.grade_document("Test query", doc)

        # Should still work (content truncated internally)
        assert result.document_id == "long"
        assert result.method == "llm"


class TestGradeAndFilterConvenience:
    """Tests for the convenience function."""

    @pytest.mark.asyncio
    async def test_grade_and_filter_basic(self):
        """Test the convenience function."""
        def mock_analyze(*args, **kwargs):
            return '{"is_relevant": true, "relevance_score": 0.8, "reasoning": "Relevant"}'

        with patch.object(
            DocumentGrader,
            "__init__",
            lambda self, analyze_fn=None, config=None: setattr(self, "_analyze", mock_analyze) or setattr(self, "config", config or GradingConfig()),
        ):
            docs = [MockDocument(id=f"doc{i}", content=f"Content {i}") for i in range(3)]

            filtered, metadata = await grade_and_filter_documents(
                query="Test query",
                documents=docs,
                threshold=0.5,
            )

            # Note: actual filtering depends on grader behavior
            assert isinstance(filtered, list)
            assert isinstance(metadata, dict)


class TestGradingResultDataclass:
    """Tests for GradingResult dataclass."""

    def test_grading_result_creation(self):
        """Test creating a GradingResult."""
        result = GradingResult(
            document_id="doc1",
            is_relevant=True,
            relevance_score=0.85,
            reasoning="Document is highly relevant.",
            latency_ms=150,
            method="llm",
        )

        assert result.document_id == "doc1"
        assert result.is_relevant is True
        assert result.relevance_score == 0.85
        assert result.reasoning == "Document is highly relevant."
        assert result.latency_ms == 150
        assert result.method == "llm"
        assert result.metadata == {}

    def test_grading_result_with_metadata(self):
        """Test GradingResult with metadata."""
        result = GradingResult(
            document_id="doc2",
            is_relevant=False,
            relevance_score=0.2,
            reasoning="Unrelated content.",
            latency_ms=100,
            method="score_fallback",
            metadata={"error": "timeout"},
        )

        assert result.metadata == {"error": "timeout"}


class TestGradingBatchResultDataclass:
    """Tests for GradingBatchResult dataclass."""

    def test_batch_result_creation(self):
        """Test creating a GradingBatchResult."""
        results = [
            GradingResult("doc1", True, 0.9, "Relevant", 100, "llm"),
            GradingResult("doc2", False, 0.2, "Irrelevant", 100, "llm"),
        ]

        batch = GradingBatchResult(
            results=results,
            relevant_count=1,
            total_count=2,
            avg_relevance=0.55,
            total_latency_ms=200,
        )

        assert len(batch.results) == 2
        assert batch.relevant_count == 1
        assert batch.total_count == 2
        assert batch.avg_relevance == 0.55
        assert batch.total_latency_ms == 200


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_document_without_id(self):
        """Test handling document without explicit id attribute."""
        def mock_analyze(*args, **kwargs):
            return '{"is_relevant": true, "relevance_score": 0.7, "reasoning": "Ok"}'

        grader = DocumentGrader(analyze_fn=mock_analyze)

        class SimpleDoc:
            content = "Some content"
            score = 0.5

        doc = SimpleDoc()
        result = await grader.grade_document("Query", doc)

        # Should use id(doc) as fallback
        assert result.document_id is not None
        assert result.is_relevant is True

    @pytest.mark.asyncio
    async def test_document_without_content(self):
        """Test handling document without content attribute."""
        def mock_analyze(*args, **kwargs):
            return '{"is_relevant": false, "relevance_score": 0.1, "reasoning": "Empty"}'

        grader = DocumentGrader(analyze_fn=mock_analyze)

        class NoContentDoc:
            id = "no_content"
            score = 0.5

        doc = NoContentDoc()
        result = await grader.grade_document("Query", doc)

        # Should still work with empty content
        assert result.document_id == "no_content"

    @pytest.mark.asyncio
    async def test_analyze_raises_exception(self):
        """Test handling when analyze function raises exception."""
        def mock_analyze(*args, **kwargs):
            raise ValueError("LLM API error")

        config = GradingConfig(fallback_to_score=True, fallback_min_score=0.4)
        grader = DocumentGrader(analyze_fn=mock_analyze, config=config)

        doc = MockDocument(id="error_doc", content="Content", score=0.6)
        result = await grader.grade_document("Query", doc)

        # Should fall back to score
        assert result.method == "score_fallback"
        assert result.is_relevant is True  # 0.6 >= 0.4
        assert "error" in result.metadata

    @pytest.mark.asyncio
    async def test_relevance_score_clamping(self):
        """Test that relevance scores are clamped to valid range."""
        def mock_analyze(*args, **kwargs):
            # Return out-of-range score
            return '{"is_relevant": true, "relevance_score": 1.5, "reasoning": "Very relevant"}'

        grader = DocumentGrader(analyze_fn=mock_analyze)
        doc = MockDocument(id="test", content="Content")

        result = await grader.grade_document("Query", doc)

        # Score should be clamped to 1.0
        assert result.relevance_score == 1.0
