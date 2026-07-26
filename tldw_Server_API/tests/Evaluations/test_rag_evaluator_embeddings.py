"""
Test suite for RAG evaluator with embeddings integration.

Tests the integration between the evaluations module and the production embeddings service.
"""

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator


class TestRAGEvaluatorEmbeddings:
    """Test RAG evaluator with real embeddings integration."""

    @pytest.fixture
    def mock_create_embedding(self):
        """Mock the create_embedding function."""
        with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.create_embedding') as mock_create:
            # Return a consistent embedding vector
            mock_create.return_value = np.random.rand(1536).tolist()
            yield mock_create

    @pytest.mark.asyncio
    async def test_evaluator_initialization_with_embeddings(self, mock_create_embedding):
        """Test that RAGEvaluator properly initializes with embeddings."""
        evaluator = RAGEvaluator(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small"
        )

        # Check that embeddings are available (lazy evaluation on first use)
        assert evaluator.embedding_provider == "openai"
        assert evaluator.embedding_model == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_evaluator_initialization_fallback(self):
        """Test that RAGEvaluator falls back gracefully when embeddings fail."""
        with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.create_embedding') as mock_create:
            mock_create.side_effect = RuntimeError("API key not found")

            evaluator = RAGEvaluator()

            # Force evaluation of embedding_available property
            assert evaluator.embedding_available is False

    @pytest.mark.asyncio
    async def test_answer_similarity_with_embeddings(self, mock_create_embedding):
        """Test answer similarity evaluation using embeddings."""
        evaluator = RAGEvaluator()
        evaluator.embedding_available = True  # Force embeddings to be available

        # Create embeddings with known similarity
        response_embedding = [1.0, 0.0, 0.0]
        ground_truth_embedding = [0.8, 0.6, 0.0]  # Cosine similarity ~0.8

        mock_create_embedding.side_effect = [
            response_embedding,
            ground_truth_embedding
        ]

        metric_name, result = await evaluator._evaluate_answer_similarity(
            "Response text",
            "Ground truth text"
        )

        assert metric_name == "answer_similarity"
        assert "score" in result
        assert result["method"] == "embeddings"
        assert 0 <= result["score"] <= 1
        assert 1 <= result["raw_score"] <= 5

        # Verify embeddings were called
        assert mock_create_embedding.call_count == 2

    @pytest.mark.asyncio
    async def test_answer_similarity_fallback_to_llm(self, mock_create_embedding):
        """Test that answer similarity falls back to LLM when embeddings fail."""
        evaluator = RAGEvaluator()
        evaluator.embedding_available = True  # Initially available

        # Make embeddings fail
        mock_create_embedding.side_effect = RuntimeError("Embedding service unavailable")

        with patch(
            "tldw_Server_API.app.core.Evaluations.rag_evaluator._run_bounded_rag_analyze",
            new_callable=AsyncMock,
        ) as mock_analyze:
            mock_analyze.return_value = "4"  # LLM returns score of 4

            metric_name, result = await evaluator._evaluate_answer_similarity(
                "Response text",
                "Ground truth text"
            )

            assert metric_name == "answer_similarity"
            assert result["method"] == "llm"
            assert result["score"] == 0.8  # 4/5
            assert result["raw_score"] == 4.0

    @pytest.mark.asyncio
    async def test_answer_similarity_error_propagation(self):
        """Test that answer similarity properly propagates errors instead of returning 0."""
        evaluator = RAGEvaluator()
        evaluator.embedding_available = False  # Force LLM path

        with patch(
            "tldw_Server_API.app.core.Evaluations.rag_evaluator._run_bounded_rag_analyze",
            new_callable=AsyncMock,
        ) as mock_analyze:
            mock_analyze.side_effect = RuntimeError("LLM API error")

            with pytest.raises(ValueError) as exc_info:
                await evaluator._evaluate_answer_similarity(
                    "Response text",
                    "Ground truth text"
                )

            assert "Answer similarity evaluation failed" in str(exc_info.value)
            assert exc_info.value.__cause__ is None
            assert exc_info.value.__context__ is None

    @pytest.mark.asyncio
    async def test_evaluate_with_embeddings(self, mock_create_embedding):
        """Test full evaluation pipeline with embeddings."""
        evaluator = RAGEvaluator()
        evaluator.embedding_available = True

        # Setup mock embeddings
        mock_create_embedding.return_value = np.random.rand(1536).tolist()

        with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.asyncio.to_thread') as mock_thread:
            mock_thread.return_value = "4"  # Mock LLM responses

            results = await evaluator.evaluate(
                query="What is machine learning?",
                contexts=["ML is a subset of AI", "It uses algorithms"],
                response="Machine learning is AI that learns from data",
                ground_truth="ML is AI that improves through experience",
                metrics=["relevance", "faithfulness", "answer_similarity"]
            )

            assert "metrics" in results
            assert "suggestions" in results
            assert "answer_similarity" in results["metrics"]

            # Verify embeddings were used for answer similarity
            if evaluator.embedding_available:
                assert results["metrics"]["answer_similarity"]["method"] == "embeddings"

    def test_evaluator_cleanup(self):

        """Test that evaluator properly cleans up resources."""
        evaluator = RAGEvaluator()
        evaluator.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_cosine_similarity_calculation(self, mock_create_embedding):
        """Test that cosine similarity is calculated correctly."""
        evaluator = RAGEvaluator()
        evaluator.embedding_available = True

        # Create orthogonal vectors (similarity = 0)
        response_embedding = [1.0, 0.0]
        ground_truth_embedding = [0.0, 1.0]

        mock_create_embedding.side_effect = [
            response_embedding,
            ground_truth_embedding
        ]

        _, result = await evaluator._evaluate_answer_similarity(
            "Response", "Ground truth"
        )

        # Orthogonal vectors should have low similarity
        assert result["score"] < 0.1
        assert result["raw_score"] < 1.5

        # Create identical vectors (similarity = 1)
        mock_create_embedding.side_effect = [
            [1.0, 0.0],
            [1.0, 0.0]
        ]

        _, result = await evaluator._evaluate_answer_similarity(
            "Response", "Ground truth"
        )

        # Identical vectors should have perfect similarity
        assert result["score"] > 0.99
        assert result["raw_score"] > 4.9


class TestRAGEvaluatorIntegration:
    """Integration tests with actual embeddings service (requires API keys)."""

    @pytest.mark.integration
    @pytest.mark.skip(reason="Integration tests require API keys")
    @pytest.mark.asyncio
    async def test_real_embeddings_integration(self):
        """Test with real OpenAI embeddings (requires API key)."""
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OpenAI API key not found")

        evaluator = RAGEvaluator(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            api_key=api_key
        )

        assert evaluator.embedding_available is True

        # Test with real similarity calculation
        _, result = await evaluator._evaluate_answer_similarity(
            "Machine learning is a type of artificial intelligence",
            "ML is a subset of AI"
        )

        assert result["method"] == "embeddings"
        assert 0.5 < result["score"] < 0.9  # Should be similar but not identical

        evaluator.close()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_evaluation_pipeline_integration(self):
        """Test complete evaluation pipeline with real services."""
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OpenAI API key not found")

        evaluator = RAGEvaluator(api_key=api_key)

        results = await evaluator.evaluate(
            query="What is deep learning?",
            contexts=[
                "Deep learning is a subset of machine learning",
                "It uses neural networks with multiple layers",
                "Deep learning can learn hierarchical representations"
            ],
            response="Deep learning is a machine learning technique using multi-layer neural networks",
            ground_truth="Deep learning is ML with multi-layered artificial neural networks",
            metrics=["relevance", "faithfulness", "answer_similarity"]
        )

        assert all(metric in results["metrics"] for metric in ["relevance", "faithfulness", "answer_similarity"])
        assert results["metrics"]["answer_similarity"]["method"] == "embeddings"
        assert len(results["suggestions"]) >= 0

        evaluator.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
