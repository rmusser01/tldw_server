"""
Tests for Quality Graders (Self-Correcting RAG Stages 5-6)

These tests cover:
- FastGroundednessGrader for binary groundedness checks
- UtilityGrader for response usefulness scoring
- Convenience functions for pipeline integration
- Heuristic fallbacks when LLM is unavailable
"""

import asyncio
import pytest
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import tldw_Server_API.app.core.RAG.rag_service.quality_graders as quality_graders
from tldw_Server_API.app.core.RAG.rag_service.quality_graders import (
    FastGroundednessResult,
    FastGroundednessGrader,
    UtilityResult,
    UtilityGrader,
    check_fast_groundedness,
    grade_utility,
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource


SECRET_TOKEN = "sk-test-secret-value"
SENSITIVE_PATH = "/Users/alice/private/quality_grader_payload.json"


def _capture_quality_grader_logs(level: str = "DEBUG"):
    messages: list[str] = []
    sink_id = quality_graders.logger.add(
        lambda message: messages.append(str(message)),
        format="{message}",
        level=level,
    )
    return messages, sink_id


def _assert_sensitive_details_not_logged(messages: list[str]) -> None:
    log_output = "\n".join(messages)
    assert SECRET_TOKEN not in log_output
    assert SENSITIVE_PATH not in log_output


class TestFastGroundednessResult:
    """Tests for FastGroundednessResult dataclass."""

    def test_create_result(self):
        """Test creating a FastGroundednessResult."""
        result = FastGroundednessResult(
            is_grounded=True,
            confidence=0.95,
            rationale="Answer is well-supported by sources.",
            latency_ms=150,
            method="llm",
        )

        assert result.is_grounded is True
        assert result.confidence == 0.95
        assert result.rationale == "Answer is well-supported by sources."
        assert result.latency_ms == 150
        assert result.method == "llm"
        assert result.metadata == {}

    def test_create_result_with_metadata(self):
        """Test creating result with metadata."""
        result = FastGroundednessResult(
            is_grounded=False,
            confidence=0.3,
            rationale="Could not verify claims.",
            latency_ms=200,
            method="heuristic",
            metadata={"word_overlap": 0.25},
        )

        assert result.is_grounded is False
        assert result.metadata == {"word_overlap": 0.25}


class TestUtilityResult:
    """Tests for UtilityResult dataclass."""

    def test_create_result(self):
        """Test creating a UtilityResult."""
        result = UtilityResult(
            utility_score=4,
            explanation="Good answer with relevant details.",
            latency_ms=100,
            method="llm",
        )

        assert result.utility_score == 4
        assert result.explanation == "Good answer with relevant details."
        assert result.latency_ms == 100
        assert result.method == "llm"
        assert result.metadata == {}

    def test_create_result_with_metadata(self):
        """Test creating result with metadata."""
        result = UtilityResult(
            utility_score=2,
            explanation="Partially relevant.",
            latency_ms=50,
            method="heuristic",
            metadata={"length_penalty": -1},
        )

        assert result.utility_score == 2
        assert result.metadata == {"length_penalty": -1}


def _failing_analyze(*args, **kwargs):
    """Analyze function that always fails, forcing heuristic fallback."""
    raise RuntimeError("LLM unavailable for testing")


class TestFastGroundednessGrader:
    """Tests for FastGroundednessGrader class."""

    @pytest.fixture
    def sample_documents(self) -> List[Document]:
        """Create sample documents for testing."""
        return [
            Document(
                id="doc1",
                content="Machine learning is a subset of artificial intelligence. It uses algorithms to learn patterns from data.",
                source=DataSource.MEDIA_DB,
                score=0.8,
                metadata={"title": "ML Overview"},
            ),
            Document(
                id="doc2",
                content="Deep learning is a type of machine learning using neural networks with many layers.",
                source=DataSource.MEDIA_DB,
                score=0.7,
                metadata={"title": "Deep Learning Guide"},
            ),
        ]

    @pytest.mark.asyncio
    async def test_heuristic_grounding_supported(self, sample_documents):
        """Test heuristic groundedness check with matching content."""
        # Use failing analyze to force heuristic path
        grader = FastGroundednessGrader(analyze_fn=_failing_analyze)

        # Answer that uses terms from the documents
        answer = "Machine learning uses algorithms to learn patterns from data. Deep learning uses neural networks."

        result = await grader.grade(
            query="What is machine learning?",
            answer=answer,
            documents=sample_documents,
        )

        assert isinstance(result, FastGroundednessResult)
        assert result.method == "heuristic"
        assert result.is_grounded is True
        assert result.confidence > 0.3  # Good overlap

    @pytest.mark.asyncio
    async def test_heuristic_grounding_not_supported(self, sample_documents):
        """Test heuristic groundedness check with non-matching content."""
        # Use failing analyze to force heuristic path
        grader = FastGroundednessGrader(analyze_fn=_failing_analyze)

        # Answer that doesn't match the documents at all
        answer = "Quantum computing uses qubits for parallel processing of cryptographic algorithms."

        result = await grader.grade(
            query="What is quantum computing?",
            answer=answer,
            documents=sample_documents,
        )

        assert result.method == "heuristic"
        assert result.is_grounded is False
        assert result.confidence < 0.3  # Low overlap

    @pytest.mark.asyncio
    async def test_llm_grounding_success(self, sample_documents):
        """Test LLM-based groundedness check with mocked response."""
        mock_analyze = MagicMock(return_value='{"is_grounded": true, "confidence": 0.9, "rationale": "Well supported."}')

        grader = FastGroundednessGrader(
            analyze_fn=mock_analyze,
            provider="openai",
            timeout_sec=5.0,
        )

        result = await grader.grade(
            query="What is machine learning?",
            answer="Machine learning learns from data using algorithms.",
            documents=sample_documents,
        )

        assert result.is_grounded is True
        assert result.confidence == 0.9
        assert result.method == "llm"

    @pytest.mark.asyncio
    async def test_llm_grounding_parses_fenced_json_with_think_tags(self, sample_documents):
        mock_analyze = MagicMock(
            return_value=(
                "<think>reasoning</think>\n"
                "```json\n"
                '{"is_grounded": true, "confidence": 0.85, "rationale": "Supported."}\n'
                "```"
            )
        )

        grader = FastGroundednessGrader(analyze_fn=mock_analyze)
        result = await grader.grade(
            query="What is machine learning?",
            answer="Machine learning learns from data.",
            documents=sample_documents,
        )

        assert result.is_grounded is True
        assert result.confidence == 0.85
        assert result.method == "llm"

    @pytest.mark.asyncio
    async def test_llm_grounding_not_grounded(self, sample_documents):
        """Test LLM-based check when answer is not grounded."""
        mock_analyze = MagicMock(return_value='{"is_grounded": false, "confidence": 0.8, "rationale": "Claims not in sources."}')

        grader = FastGroundednessGrader(analyze_fn=mock_analyze)

        result = await grader.grade(
            query="What is quantum computing?",
            answer="Quantum computing is revolutionary.",
            documents=sample_documents,
        )

        assert result.is_grounded is False
        assert result.confidence == 0.8
        assert "Claims not in sources" in result.rationale

    @pytest.mark.asyncio
    async def test_llm_timeout_fallback(self, sample_documents):
        """Test that timeout returns fail-open result."""
        async def slow_analyze(*args, **kwargs):
            await asyncio.sleep(10)
            return '{"is_grounded": true}'

        def sync_slow(*args, **kwargs):
            import time
            time.sleep(10)
            return '{"is_grounded": true}'

        grader = FastGroundednessGrader(
            analyze_fn=sync_slow,
            timeout_sec=0.1,
        )

        result = await grader.grade(
            query="Test query",
            answer="Test answer",
            documents=sample_documents,
        )

        # Should fail open (assume grounded on timeout)
        assert result.is_grounded is True
        assert result.confidence == 0.0
        assert result.method == "error_fallback"
        assert result.metadata.get("error") == "timeout"

    @pytest.mark.asyncio
    async def test_llm_parse_failure_fallback(self, sample_documents):
        """Test fallback when LLM response can't be parsed."""
        mock_analyze = MagicMock(return_value="This is not valid JSON")

        grader = FastGroundednessGrader(analyze_fn=mock_analyze)

        result = await grader.grade(
            query="Test",
            answer="Test answer",
            documents=sample_documents,
        )

        # Should return default on parse failure
        assert result.is_grounded is True
        assert result.method == "error_fallback"

    @pytest.mark.asyncio
    async def test_llm_failure_fallback_does_not_log_sensitive_exception_details(self, sample_documents):
        """Test LLM failure fallback without leaking exception details."""
        def failing_analyze(*args, **kwargs):
            raise RuntimeError(f"provider failed with {SECRET_TOKEN} at {SENSITIVE_PATH}")

        messages, sink_id = _capture_quality_grader_logs()
        try:
            grader = FastGroundednessGrader(analyze_fn=failing_analyze)
            result = await grader.grade(
                query="What is machine learning?",
                answer="Machine learning uses algorithms to learn patterns from data.",
                documents=sample_documents,
            )
        finally:
            quality_graders.logger.remove(sink_id)

        assert result.method == "heuristic"
        assert result.is_grounded is True
        assert result.metadata == {}
        _assert_sensitive_details_not_logged(messages)

    def test_parse_failure_fallback_does_not_log_sensitive_exception_details(self):
        """Test parse fallback logging does not include raw parser exception details."""
        def failing_parser(*args, **kwargs):
            raise ValueError(f"parse failed for {SECRET_TOKEN} at {SENSITIVE_PATH}")

        messages, sink_id = _capture_quality_grader_logs()
        try:
            grader = FastGroundednessGrader(analyze_fn=MagicMock())
            with patch.object(quality_graders, "parse_structured_output", side_effect=failing_parser):
                result = grader._parse_groundedness_response("not json", latency_ms=7)
        finally:
            quality_graders.logger.remove(sink_id)

        assert result.is_grounded is True
        assert result.confidence == 0.5
        assert result.rationale == "Could not parse LLM response"
        assert result.latency_ms == 7
        assert result.method == "error_fallback"
        assert result.metadata == {"parse_error": True}
        _assert_sensitive_details_not_logged(messages)

    @pytest.mark.asyncio
    async def test_empty_documents(self):
        """Test groundedness check with no documents."""
        # Use failing analyze to force heuristic path
        grader = FastGroundednessGrader(analyze_fn=_failing_analyze)

        result = await grader.grade(
            query="What is machine learning?",
            answer="Machine learning is amazing.",
            documents=[],
        )

        assert isinstance(result, FastGroundednessResult)
        assert result.method == "heuristic"

    @pytest.mark.asyncio
    async def test_long_answer_truncation(self, sample_documents):
        """Test that very long answers are truncated."""
        mock_analyze = MagicMock(return_value='{"is_grounded": true, "confidence": 0.8, "rationale": "OK"}')

        grader = FastGroundednessGrader(analyze_fn=mock_analyze)

        # Very long answer
        long_answer = "Machine learning " * 1000

        result = await grader.grade(
            query="What is ML?",
            answer=long_answer,
            documents=sample_documents,
        )

        # Should still work (answer truncated internally)
        assert result.is_grounded is True


class TestUtilityGrader:
    """Tests for UtilityGrader class."""

    @pytest.mark.asyncio
    async def test_heuristic_utility_good_answer(self):
        """Test heuristic utility grading for a good answer."""
        # Use failing analyze to force heuristic path
        grader = UtilityGrader(analyze_fn=_failing_analyze)

        query = "What is machine learning?"
        answer = "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, learn from it, and make predictions or decisions."

        result = await grader.grade(query, answer)

        assert isinstance(result, UtilityResult)
        assert result.method == "heuristic"
        assert result.utility_score >= 3  # Should be at least moderate

    @pytest.mark.asyncio
    async def test_heuristic_utility_short_answer(self):
        """Test heuristic utility grading for a too-short answer."""
        # Use failing analyze to force heuristic path
        grader = UtilityGrader(analyze_fn=_failing_analyze)

        query = "Explain quantum computing in detail"
        answer = "It's complicated."

        result = await grader.grade(query, answer)

        assert result.method == "heuristic"
        assert result.utility_score <= 3  # Should be low due to short length

    @pytest.mark.asyncio
    async def test_heuristic_utility_irrelevant_answer(self):
        """Test heuristic utility for irrelevant answer."""
        # Use failing analyze to force heuristic path
        grader = UtilityGrader(analyze_fn=_failing_analyze)

        query = "What is machine learning?"
        answer = "The weather today is sunny with temperatures around 72 degrees. Perfect for outdoor activities."

        result = await grader.grade(query, answer)

        assert result.method == "heuristic"
        assert result.utility_score <= 3  # Low overlap with query

    @pytest.mark.asyncio
    async def test_llm_utility_excellent(self):
        """Test LLM-based utility grading with excellent score."""
        mock_analyze = MagicMock(return_value='{"utility_score": 5, "explanation": "Comprehensive answer."}')

        grader = UtilityGrader(
            analyze_fn=mock_analyze,
            provider="openai",
        )

        result = await grader.grade(
            query="What is Python?",
            answer="Python is a high-level programming language...",
        )

        assert result.utility_score == 5
        assert result.method == "llm"
        assert "Comprehensive" in result.explanation

    @pytest.mark.asyncio
    async def test_llm_utility_parses_fenced_json_with_think_tags(self):
        mock_analyze = MagicMock(
            return_value=(
                "<think>analysis</think>\n"
                "```json\n"
                '{"utility_score": 4, "explanation": "Useful answer."}\n'
                "```"
            )
        )

        grader = UtilityGrader(analyze_fn=mock_analyze)
        result = await grader.grade(
            query="What is Python?",
            answer="Python is a programming language.",
        )

        assert result.utility_score == 4
        assert result.method == "llm"

    @pytest.mark.asyncio
    async def test_llm_utility_poor(self):
        """Test LLM-based utility grading with poor score."""
        mock_analyze = MagicMock(return_value='{"utility_score": 1, "explanation": "Does not address the question."}')

        grader = UtilityGrader(analyze_fn=mock_analyze)

        result = await grader.grade(
            query="How do I install Python?",
            answer="Python is named after Monty Python.",
        )

        assert result.utility_score == 1
        assert "Does not address" in result.explanation

    @pytest.mark.asyncio
    async def test_llm_timeout_fallback(self):
        """Test that timeout returns default score."""
        def slow_analyze(*args, **kwargs):
            import time
            time.sleep(10)
            return '{"utility_score": 5}'

        grader = UtilityGrader(
            analyze_fn=slow_analyze,
            timeout_sec=0.1,
        )

        result = await grader.grade(
            query="Test query",
            answer="Test answer",
        )

        assert result.utility_score == 3  # Default moderate
        assert result.method == "error_fallback"
        assert result.metadata.get("error") == "timeout"

    @pytest.mark.asyncio
    async def test_llm_score_clamping(self):
        """Test that scores outside 1-5 range are clamped."""
        mock_analyze = MagicMock(return_value='{"utility_score": 10, "explanation": "Very good."}')

        grader = UtilityGrader(analyze_fn=mock_analyze)

        result = await grader.grade(
            query="Test",
            answer="Test answer",
        )

        assert result.utility_score == 5  # Clamped to max

    @pytest.mark.asyncio
    async def test_llm_parse_failure(self):
        """Test fallback when response can't be parsed."""
        mock_analyze = MagicMock(return_value="Not valid JSON at all")

        grader = UtilityGrader(analyze_fn=mock_analyze)

        result = await grader.grade(
            query="Test",
            answer="Test answer",
        )

        assert result.utility_score == 3  # Default
        assert result.method == "error_fallback"

    @pytest.mark.asyncio
    async def test_llm_failure_fallback_does_not_log_sensitive_exception_details(self):
        """Test LLM failure fallback without leaking exception details."""
        def failing_analyze(*args, **kwargs):
            raise RuntimeError(f"provider failed with {SECRET_TOKEN} at {SENSITIVE_PATH}")

        messages, sink_id = _capture_quality_grader_logs()
        try:
            grader = UtilityGrader(analyze_fn=failing_analyze)
            result = await grader.grade(
                query="What is machine learning?",
                answer=(
                    "Machine learning is a field of artificial intelligence "
                    "that uses algorithms to learn from data."
                ),
            )
        finally:
            quality_graders.logger.remove(sink_id)

        assert result.method == "heuristic"
        assert result.utility_score >= 3
        assert result.metadata == {}
        _assert_sensitive_details_not_logged(messages)

    def test_parse_failure_fallback_does_not_log_sensitive_exception_details(self):
        """Test parse fallback logging does not include raw parser exception details."""
        def failing_parser(*args, **kwargs):
            raise ValueError(f"parse failed for {SECRET_TOKEN} at {SENSITIVE_PATH}")

        messages, sink_id = _capture_quality_grader_logs()
        try:
            grader = UtilityGrader(analyze_fn=MagicMock())
            with patch.object(quality_graders, "parse_structured_output", side_effect=failing_parser):
                result = grader._parse_utility_response("not json", latency_ms=11)
        finally:
            quality_graders.logger.remove(sink_id)

        assert result.utility_score == 3
        assert result.explanation == "Could not parse LLM response"
        assert result.latency_ms == 11
        assert result.method == "error_fallback"
        assert result.metadata == {"parse_error": True}
        _assert_sensitive_details_not_logged(messages)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture
    def sample_documents(self) -> List[Document]:
        """Create sample documents."""
        return [
            Document(
                id="doc1",
                content="Python is a programming language.",
                source=DataSource.MEDIA_DB,
                metadata={},
            ),
        ]

    @pytest.mark.asyncio
    async def test_check_fast_groundedness(self, sample_documents):
        """Test the convenience function for groundedness checking."""
        result, metadata = await check_fast_groundedness(
            query="What is Python?",
            answer="Python is a programming language used for web development.",
            documents=sample_documents,
            timeout_sec=5.0,
        )

        assert isinstance(result, FastGroundednessResult)
        assert isinstance(metadata, dict)
        assert metadata["fast_groundedness_enabled"] is True
        assert "is_grounded" in metadata
        assert "confidence" in metadata
        assert "latency_ms" in metadata

    @pytest.mark.asyncio
    async def test_check_fast_groundedness_with_analyze_fn(self, sample_documents):
        """Test groundedness with custom analyze function."""
        mock_analyze = MagicMock(return_value='{"is_grounded": true, "confidence": 0.95, "rationale": "Verified."}')

        result, metadata = await check_fast_groundedness(
            query="What is Python?",
            answer="Python is great.",
            documents=sample_documents,
            analyze_fn=mock_analyze,
        )

        assert result.is_grounded is True
        assert result.confidence == 0.95
        assert metadata["method"] == "llm"

    @pytest.mark.asyncio
    async def test_grade_utility(self):
        """Test the convenience function for utility grading."""
        result, metadata = await grade_utility(
            query="Explain machine learning",
            answer="Machine learning is a field of AI that enables computers to learn from data.",
            timeout_sec=5.0,
        )

        assert isinstance(result, UtilityResult)
        assert isinstance(metadata, dict)
        assert metadata["utility_grading_enabled"] is True
        assert "utility_score" in metadata
        assert "explanation" in metadata
        assert "latency_ms" in metadata

    @pytest.mark.asyncio
    async def test_grade_utility_with_analyze_fn(self):
        """Test utility grading with custom analyze function."""
        mock_analyze = MagicMock(return_value='{"utility_score": 4, "explanation": "Good answer."}')

        result, metadata = await grade_utility(
            query="What is Python?",
            answer="Python is a versatile programming language...",
            analyze_fn=mock_analyze,
        )

        assert result.utility_score == 4
        assert metadata["method"] == "llm"


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Test with empty query."""
        grader = FastGroundednessGrader(analyze_fn=None)

        result = await grader.grade(
            query="",
            answer="Some answer text.",
            documents=[],
        )

        assert isinstance(result, FastGroundednessResult)

    @pytest.mark.asyncio
    async def test_empty_answer(self):
        """Test with empty answer."""
        # Use failing analyze to force heuristic path
        grader = UtilityGrader(analyze_fn=_failing_analyze)

        result = await grader.grade(
            query="What is something?",
            answer="",
        )

        assert isinstance(result, UtilityResult)
        assert result.utility_score <= 3  # Should be low for empty answer

    @pytest.mark.asyncio
    async def test_unicode_content(self):
        """Test with unicode content."""
        docs = [
            Document(
                id="unicode",
                content="日本語のテキストです。Machine learning in Japanese context.",
                source=DataSource.MEDIA_DB,
                metadata={},
            ),
        ]

        grader = FastGroundednessGrader(analyze_fn=None)

        result = await grader.grade(
            query="日本語",
            answer="日本語とmachine learningについて。",
            documents=docs,
        )

        assert isinstance(result, FastGroundednessResult)

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test with special characters in content."""
        grader = UtilityGrader(analyze_fn=None)

        result = await grader.grade(
            query="What is <script>alert('xss')</script>?",
            answer="That appears to be an XSS attempt with <, >, and quotes.",
        )

        assert isinstance(result, UtilityResult)

    @pytest.mark.asyncio
    async def test_very_long_documents(self):
        """Test with very long source documents."""
        long_content = "Machine learning concept. " * 500
        docs = [
            Document(
                id="long",
                content=long_content,
                source=DataSource.MEDIA_DB,
                metadata={"title": "Long document"},
            ),
        ]

        grader = FastGroundednessGrader(analyze_fn=None)

        result = await grader.grade(
            query="What is machine learning?",
            answer="Machine learning is a concept.",
            documents=docs,
        )

        assert isinstance(result, FastGroundednessResult)
        # Should still complete without error


class TestIntegration:
    """Integration tests for pipeline use."""

    @pytest.mark.asyncio
    async def test_groundedness_then_utility(self):
        """Test running both graders in sequence."""
        docs = [
            Document(
                id="doc1",
                content="Python is a high-level programming language known for readability.",
                source=DataSource.MEDIA_DB,
                metadata={},
            ),
        ]

        query = "What is Python?"
        answer = "Python is a high-level programming language that emphasizes code readability."

        # Check groundedness
        fg_result, fg_meta = await check_fast_groundedness(
            query=query,
            answer=answer,
            documents=docs,
        )

        # Check utility
        ug_result, ug_meta = await grade_utility(
            query=query,
            answer=answer,
        )

        # Both should complete
        assert isinstance(fg_result, FastGroundednessResult)
        assert isinstance(ug_result, UtilityResult)

        # Groundedness should be high for this matching answer
        assert fg_result.is_grounded is True

        # Utility should be reasonable
        assert ug_result.utility_score >= 3

    @pytest.mark.asyncio
    async def test_concurrent_grading(self):
        """Test running both graders concurrently."""
        docs = [
            Document(
                id="doc1",
                content="Test content for grading.",
                source=DataSource.MEDIA_DB,
                metadata={},
            ),
        ]

        query = "Test query"
        answer = "Test answer with content."

        # Run both concurrently
        fg_task = check_fast_groundedness(query, answer, docs)
        ug_task = grade_utility(query, answer)

        (fg_result, fg_meta), (ug_result, ug_meta) = await asyncio.gather(fg_task, ug_task)

        assert isinstance(fg_result, FastGroundednessResult)
        assert isinstance(ug_result, UtilityResult)
