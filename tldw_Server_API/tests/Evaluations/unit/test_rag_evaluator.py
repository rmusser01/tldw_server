"""
Unit tests for RAGEvaluator.

Tests RAG evaluation functionality with minimal mocking (only external LLM/embedding services).
"""

import asyncio
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator
from tldw_Server_API.tests.Evaluations.fixtures.sample_data import SampleDataGenerator


@pytest.mark.unit
class TestRAGEvaluatorInit:
    """Test RAGEvaluator initialization."""

    def test_init_without_embeddings(self):

        """Test initialization without embedding support."""
        evaluator = RAGEvaluator(
            embedding_provider=None,
            embedding_model=None
        )

        assert evaluator.embedding_provider is None
        assert evaluator.embedding_model is None
        assert evaluator.embedding_available is False

    def test_init_with_embeddings(self):

        """Test initialization with embedding configuration."""
        evaluator = RAGEvaluator(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small"
        )

        assert evaluator.embedding_provider == "openai"
        assert evaluator.embedding_model == "text-embedding-3-small"
        # Note: embedding_available is checked lazily on first use

    @patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.create_embedding')
    def test_embedding_availability_check(self, mock_create_embedding):
        """Test checking embedding availability."""
        # Test successful embedding check
        mock_create_embedding.return_value = [0.1] * 1536
        evaluator = RAGEvaluator(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small"
        )

        # Force evaluation of embedding_available property
        assert evaluator.embedding_available is True

        # Test failed embedding check
        mock_create_embedding.side_effect = RuntimeError("API key not found")
        evaluator2 = RAGEvaluator(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small"
        )
        assert evaluator2.embedding_available is False


@pytest.mark.unit
class TestContextRelevance:
    """Test context relevance evaluation."""

    @pytest.mark.asyncio
    @patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.analyze')
    async def test_evaluate_context_relevance(self, mock_analyze):
        """Test evaluating context relevance."""
        # The actual code expects plain numeric string, not JSON
        mock_analyze.return_value = "4.3"

        evaluator = RAGEvaluator()
        rag_data = SampleDataGenerator.generate_rag_evaluation_data()

        metric_name, result = await evaluator._evaluate_context_relevance(
            rag_data["query"],
            rag_data["retrieved_contexts"],
            "openai"
        )

        assert metric_name == "context_relevance"
        assert "score" in result
        assert 0 <= result["score"] <= 1
        assert "raw_score" in result
        assert 1 <= result["raw_score"] <= 5
        # Called once per context
        assert mock_analyze.call_count == len(rag_data["retrieved_contexts"])

    @pytest.mark.asyncio
    @patch(
        'tldw_Server_API.app.core.Evaluations.rag_evaluator._run_bounded_rag_analyze',
        new_callable=AsyncMock,
    )
    async def test_context_relevance_edge_cases(self, mock_analyze):
        """Test context relevance with edge cases."""
        evaluator = RAGEvaluator()

        # Test empty context - should return 0 without calling API
        metric_name, result = await evaluator._evaluate_context_relevance(
            "Test query",
            [],
            "openai"
        )
        assert result["score"] == 0.0  # Empty contexts get 0.0 score
        mock_analyze.assert_not_called()  # No API call for empty context
        mock_analyze.reset_mock()

        # Test single context chunk
        mock_analyze.return_value = "5"
        metric_name, result = await evaluator._evaluate_context_relevance(
            "Test query",
            ["Single context chunk"],
            "openai"
        )
        assert result["score"] == 1.0

        # Test very long context
        long_context = ["Context " + str(i) for i in range(100)]
        mock_analyze.return_value = "3.5"
        metric_name, result = await evaluator._evaluate_context_relevance(
            "Test query",
            long_context,
            "openai"
        )
        assert 0 <= result["score"] <= 1


@pytest.mark.unit
@pytest.mark.usefixtures("mock_llm_analyze")
class TestAnswerFaithfulness:
    """Test answer faithfulness evaluation."""

    @pytest.mark.asyncio
    async def test_evaluate_answer_faithfulness(self, mock_llm_analyze):
        """Test evaluating answer faithfulness."""
        # mock_llm_analyze fixture already patches analyze function

        evaluator = RAGEvaluator()

        metric_name, result = await evaluator._evaluate_faithfulness(
            "This is the answer based on context",
            ["Context chunk 1", "Context chunk 2"],
            "openai"
        )

        assert metric_name == "faithfulness"
        assert "score" in result
        assert 0 <= result["score"] <= 1
        assert result["raw_score"] == 4.7

    @pytest.mark.asyncio
    async def test_faithfulness_hallucination_detection(self):
        """Test detection of hallucinated content."""
        evaluator = RAGEvaluator()

        metric_name, result = await evaluator._evaluate_faithfulness(
            "Answer with hallucinations",
            ["Limited context"],
            "openai"
        )

        assert result["score"] < 0.6  # Should be low score for hallucinated content


@pytest.mark.unit
@pytest.mark.usefixtures("mock_llm_analyze")
class TestAnswerRelevance:
    """Test answer relevance evaluation."""

    @pytest.mark.asyncio
    async def test_evaluate_answer_relevance(self):
        """Test evaluating answer relevance to query."""
        evaluator = RAGEvaluator()

        metric_name, result = await evaluator._evaluate_relevance(
            "What is the capital of France?",
            "Paris is the capital of France.",
            "openai"
        )

        assert metric_name == "relevance"
        assert result["score"] > 0.7  # Should be high score for relevant answer

    @pytest.mark.asyncio
    async def test_answer_relevance_mismatch(self):
        """Test when answer doesn't match query."""
        evaluator = RAGEvaluator()

        metric_name, result = await evaluator._evaluate_relevance(
            "What is quantum computing?",
            "The weather today is sunny.",
            "openai"
        )

        assert result["score"] < 0.4  # Should be low score for irrelevant answer


@pytest.mark.unit
@pytest.mark.usefixtures("mock_llm_analyze")
class TestAnswerSimilarity:
    """Test answer similarity evaluation."""

    @pytest.mark.asyncio
    @patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.create_embedding')
    async def test_answer_similarity_with_embeddings(self, mock_create_embedding):
        """Test answer similarity using embeddings."""
        # Create mock embeddings with known cosine similarity
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.8, 0.6, 0.0]  # Cosine similarity = 0.8

        mock_create_embedding.side_effect = [embedding1, embedding2]

        evaluator = RAGEvaluator(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small"
        )
        evaluator.embedding_available = True

        metric_name, result = await evaluator._evaluate_answer_similarity(
            "Response text",
            "Ground truth text"
        )

        assert metric_name == "answer_similarity"
        assert result["method"] == "embeddings"
        assert 0 <= result["score"] <= 1
        assert mock_create_embedding.call_count == 2

    @pytest.mark.asyncio
    async def test_answer_similarity_fallback_to_llm(self):
        """Test fallback to LLM when embeddings unavailable."""
        evaluator = RAGEvaluator()  # No embeddings configured

        metric_name, result = await evaluator._evaluate_answer_similarity(
            "Response text",
            "Ground truth text"
        )

        assert metric_name == "answer_similarity"
        assert result["method"] == "llm"
        assert 0 <= result["score"] <= 1  # Valid score range

    @pytest.mark.asyncio
    async def test_answer_similarity_identical_texts(self):
        """Test similarity of identical texts."""
        evaluator = RAGEvaluator()

        # For identical texts without embeddings, should still give high score
        metric_name, result = await evaluator._evaluate_answer_similarity(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog"
        )

        assert result["score"] > 0.9  # Should be very high similarity for identical texts


@pytest.mark.unit
@pytest.mark.usefixtures("mock_llm_analyze")
class TestFullRAGEvaluation:
    """Test complete RAG evaluation workflow."""

    @pytest.mark.asyncio
    async def test_evaluate_rag_complete(self):
        """Test complete RAG evaluation with all metrics."""
        evaluator = RAGEvaluator()
        rag_data = SampleDataGenerator.generate_rag_evaluation_data()

        results = await evaluator.evaluate(
            query=rag_data["query"],
            contexts=rag_data["retrieved_contexts"],
            response=rag_data["generated_response"],
            ground_truth=rag_data.get("ground_truth"),
            api_name="openai"
        )

        assert "metrics" in results
        assert "context_relevance" in results["metrics"]
        assert "answer_faithfulness" in results["metrics"]
        assert "answer_relevance" in results["metrics"]

        # Check overall score calculation
        assert "overall_score" in results
        assert 0 <= results["overall_score"] <= 1

    @pytest.mark.asyncio
    @patch(
        'tldw_Server_API.app.core.Evaluations.rag_evaluator._run_bounded_rag_analyze',
        new_callable=AsyncMock,
    )
    async def test_evaluate_rag_without_ground_truth(self, mock_analyze):
        """Test RAG evaluation without ground truth."""
        # Order matches evaluate() execution: relevance, faithfulness, context_relevance
        mock_responses = [
            "4.5",  # answer_relevance first
            "4.7",  # answer_faithfulness second
            "4.3",  # context_relevance (for first context)
            "4.3"   # context_relevance (for second context)
        ]
        mock_analyze.side_effect = mock_responses

        evaluator = RAGEvaluator()

        results = await evaluator.evaluate(
            query="Test query",
            contexts=["Context 1", "Context 2"],
            response="Test response",
            ground_truth=None,  # No ground truth
            api_name="openai"
        )

        assert "answer_similarity" not in results["metrics"]
        assert len(results["metrics"]) == 3

    @pytest.mark.asyncio
    async def test_evaluate_rag_with_custom_weights(self):
        """Test RAG evaluation with custom metric weights."""
        evaluator = RAGEvaluator()

        custom_weights = {
            "context_relevance": 0.2,
            "answer_faithfulness": 0.5,
            "answer_relevance": 0.3
        }

        results = await evaluator.evaluate(
            query="What is the capital of France?",
            contexts=["Paris is the capital and largest city of France.", "France is a country in Western Europe."],
            response="The capital of France is Paris.",
            metric_weights=custom_weights,
            api_name="openai"
        )

        # Verify that custom weights are used - just check overall score is calculated
        assert "overall_score" in results
        assert 0 <= results["overall_score"] <= 1

        # Verify all expected metrics are present
        assert "answer_relevance" in results["metrics"]
        assert "answer_faithfulness" in results["metrics"]
        assert "context_relevance" in results["metrics"]

        # Verify each metric has expected structure
        for _metric_name, metric_data in results["metrics"].items():
            assert "score" in metric_data
            assert 0 <= metric_data["score"] <= 1
            assert "raw_score" in metric_data
            assert 1 <= metric_data["raw_score"] <= 5


@pytest.mark.unit
class TestMetricCalculations:
    """Test metric calculation and normalization."""

    def test_normalize_score(self):

        """Test score normalization from 1-5 to 0-1."""
        evaluator = RAGEvaluator()

        # Test normalization
        assert evaluator._normalize_score(1) == 0
        assert evaluator._normalize_score(3) == 0.5
        assert evaluator._normalize_score(5) == 1.0

        # Test out-of-range handling
        assert evaluator._normalize_score(0) == 0
        assert evaluator._normalize_score(6) == 1.0

    def test_calculate_overall_score(self):

        """Test overall score calculation."""
        evaluator = RAGEvaluator()

        metrics = {
            "context_relevance": {"score": 0.8},
            "answer_faithfulness": {"score": 0.9},
            "answer_relevance": {"score": 0.7}
        }

        # Equal weights
        overall = evaluator._calculate_overall_score(metrics)
        assert overall == pytest.approx(0.8, 0.01)

        # Custom weights
        weights = {
            "context_relevance": 0.5,
            "answer_faithfulness": 0.3,
            "answer_relevance": 0.2
        }
        overall_weighted = evaluator._calculate_overall_score(metrics, weights)
        expected = 0.8 * 0.5 + 0.9 * 0.3 + 0.7 * 0.2
        assert overall_weighted == pytest.approx(expected, 0.01)


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in RAGEvaluator."""

    @pytest.mark.asyncio
    @patch(
        'tldw_Server_API.app.core.Evaluations.rag_evaluator._run_bounded_rag_analyze',
        new_callable=AsyncMock,
    )
    async def test_handle_llm_failure(self, mock_analyze):
        """Test handling of LLM API failures."""
        mock_analyze.side_effect = RuntimeError("LLM API error")

        evaluator = RAGEvaluator()

        # _evaluate_context_relevance catches errors and returns 0.0 scores
        metric_name, result = await evaluator._evaluate_context_relevance("query", ["context"], "openai")

        assert metric_name == "context_relevance"
        assert result["score"] == 0.0  # Caught exception should result in 0.0

    @pytest.mark.asyncio
    @patch(
        'tldw_Server_API.app.core.Evaluations.rag_evaluator._run_bounded_rag_analyze',
        new_callable=AsyncMock,
    )
    async def test_handle_invalid_llm_response(self, mock_analyze):
        """Test handling of invalid LLM responses."""
        mock_analyze.return_value = "not_a_number"

        evaluator = RAGEvaluator()

        # Should handle invalid response and return 0.0
        metric_name, result = await evaluator._evaluate_context_relevance("query", ["context"], "openai")
        assert result["score"] == 0.0  # Invalid response gets handled as 0.0

    @pytest.mark.asyncio
    @patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.create_embedding')
    async def test_handle_embedding_failure(self, mock_create_embedding):
        """Test handling of embedding API failures."""
        mock_create_embedding.side_effect = RuntimeError("Embedding API error")

        evaluator = RAGEvaluator(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small"
        )
        evaluator.embedding_available = True

        # Should fall back to LLM
        with patch(
            'tldw_Server_API.app.core.Evaluations.rag_evaluator._run_bounded_rag_analyze',
            new_callable=AsyncMock,
        ) as mock_analyze:
            # The actual code expects plain numeric string, not JSON
            mock_analyze.return_value = "4.0"

            metric_name, result = await evaluator._evaluate_answer_similarity(
                "text1", "text2"
            )

            assert result["method"] == "llm"  # Fallback to LLM
            mock_analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_relevance_provider_failure_is_detached(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Provider diagnostics cannot remain reachable through metric errors."""

        sentinel = "sk-rag-evaluator-/private/provider-response.json"
        evaluator = RAGEvaluator()
        call = AsyncMock(side_effect=RuntimeError(sentinel))
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Evaluations.rag_evaluator.llm_circuit_breaker.call_with_breaker",
            call,
        )

        with pytest.raises(ValueError) as exc_info:
            await evaluator._evaluate_relevance(
                "query",
                "response",
                "openai",
            )

        assert str(exc_info.value) == "Relevance evaluation failed"
        assert sentinel not in str(exc_info.value)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("failure_path", "expected_message"),
        [
            ("overall", "Evaluation failed"),
            ("claim_faithfulness", "Claim faithfulness evaluation failed"),
            (
                "relevance_circuit_open",
                "Service temporarily unavailable for relevance evaluation",
            ),
            ("relevance", "Relevance evaluation failed"),
            ("faithfulness", "Faithfulness evaluation failed"),
            ("answer_similarity", "Answer similarity evaluation failed"),
            ("context_recall", "Context recall evaluation failed"),
        ],
    )
    async def test_all_public_rag_failures_detach_private_exception_graphs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        failure_path: str,
        expected_message: str,
    ) -> None:
        """Every public RAG failure drops provider bodies, URLs, and traceback links."""
        from tldw_Server_API.app.core.Evaluations import rag_evaluator

        sentinel = f"sk-{failure_path}-https://private-provider.invalid/body"
        logged: list[str] = []

        def capture_log(message: str, *args, **_kwargs) -> None:
            logged.append(message.format(*args))

        monkeypatch.setattr(rag_evaluator.logger, "error", capture_log)
        monkeypatch.setattr(rag_evaluator.logger, "warning", capture_log)
        evaluator = RAGEvaluator(embedding_provider=None, embedding_model=None)

        if failure_path == "overall":
            async def fail_gather(*awaitables, **_kwargs):
                for awaitable in awaitables:
                    close = getattr(awaitable, "close", None)
                    if callable(close):
                        close()
                raise RuntimeError(sentinel)

            monkeypatch.setattr(rag_evaluator.asyncio, "gather", fail_gather)

            async def invoke() -> None:
                await evaluator.evaluate(
                    query="query",
                    contexts=["context"],
                    response="response",
                    metrics=["relevance"],
                )

        elif failure_path == "claim_faithfulness":
            class FailingClaimsEngine:
                def __init__(self, _analyze_fn) -> None:
                    return None

                async def run(self, **_kwargs):
                    raise RuntimeError(sentinel)

            monkeypatch.setattr(rag_evaluator, "ClaimsEngine", FailingClaimsEngine)

            async def invoke() -> None:
                await evaluator._evaluate_claim_faithfulness(
                    "response",
                    ["context"],
                    "openai",
                )

        elif failure_path == "answer_similarity":
            async def fail_analysis(*_args, **_kwargs):
                raise RuntimeError(sentinel)

            monkeypatch.setattr(
                rag_evaluator,
                "_run_bounded_rag_analyze",
                fail_analysis,
            )

            async def invoke() -> None:
                await evaluator._evaluate_answer_similarity(
                    "response alpha",
                    "ground truth beta",
                    "openai",
                )

        else:
            if failure_path == "relevance_circuit_open":
                failure = rag_evaluator.CircuitOpenError(
                    sentinel,
                    breaker_name="rag-eval",
                    category="evaluations",
                    service="private-provider",
                )
            else:
                failure = RuntimeError(sentinel)
            call = AsyncMock(side_effect=failure)
            monkeypatch.setattr(
                rag_evaluator.llm_circuit_breaker,
                "call_with_breaker",
                call,
            )

            async def invoke() -> None:
                if failure_path.startswith("relevance"):
                    await evaluator._evaluate_relevance("query", "response", "openai")
                elif failure_path == "faithfulness":
                    await evaluator._evaluate_faithfulness(
                        "response",
                        ["context"],
                        "openai",
                    )
                else:
                    await evaluator._evaluate_context_recall(
                        "ground truth",
                        ["context"],
                        "openai",
                    )

        with pytest.raises(ValueError) as exc_info:
            await invoke()

        assert str(exc_info.value) == expected_message
        assert sentinel not in str(exc_info.value)
        assert sentinel not in "\n".join(logged)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None


@pytest.mark.unit
class TestConcurrency:
    """Test concurrent evaluation handling."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("metric_name", "metric_args"),
        [
            ("relevance", ("query", "response", "openai")),
            ("faithfulness", ("response", ["context"], "openai")),
            ("context_recall", ("ground truth", ["context"], "openai")),
        ],
    )
    async def test_circuit_backed_rag_metric_uses_async_shared_pool_boundary(
        self,
        monkeypatch: pytest.MonkeyPatch,
        metric_name: str,
        metric_args: tuple[Any, ...],
    ) -> None:
        """Circuit-backed metrics must not hand a sync call to the breaker."""
        from tldw_Server_API.app.core.Evaluations import rag_evaluator

        breaker_functions: list[Any] = []
        bounded_calls: list[dict[str, Any]] = []

        async def call_inline(
            _provider: str,
            function,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            breaker_functions.append(function)
            result = function(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result

        async def run_inline(
            call,
            *,
            pool,
            exhaustion_message: str,
            on_cancel_result=None,
        ) -> Any:
            bounded_calls.append(
                {
                    "pool": pool,
                    "exhaustion_message": exhaustion_message,
                    "on_cancel_result": on_cancel_result,
                }
            )
            return call()

        monkeypatch.setattr(rag_evaluator, "analyze", lambda *_args, **_kwargs: "4")
        monkeypatch.setattr(
            rag_evaluator.llm_circuit_breaker,
            "call_with_breaker",
            call_inline,
        )
        monkeypatch.setattr(
            rag_evaluator,
            "await_bounded_sync_call",
            run_inline,
            raising=False,
        )

        evaluator = RAGEvaluator(embedding_provider=None, embedding_model=None)
        method = getattr(evaluator, f"_evaluate_{metric_name}")
        returned_metric, _result = await method(*metric_args)

        assert returned_metric == metric_name
        assert len(breaker_functions) == 1
        assert inspect.iscoroutinefunction(breaker_functions[0])
        assert bounded_calls == [
            {
                "pool": rag_evaluator.SYNC_ADAPTER_CALL_POOL,
                "exhaustion_message": "RAG evaluation provider capacity is exhausted",
                "on_cancel_result": None,
            }
        ]

    @pytest.mark.asyncio
    async def test_circuit_timeout_drains_rag_provider_once_before_sanitized_return(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The breaker owns one deadline and waits for the admitted worker's exit."""
        from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
        from tldw_Server_API.app.core.Evaluations import rag_evaluator
        from tldw_Server_API.app.core.Evaluations.circuit_breaker import (
            CircuitBreakerConfig,
            LLMCircuitBreaker,
        )

        entered = threading.Event()
        release = threading.Event()
        lifecycle: list[str] = []

        class TrackingPool(BoundedDaemonPool):
            def _release_capacity(self) -> None:
                lifecycle.append("capacity-release")
                super()._release_capacity()

        def blocking_analyze(*_args: Any, **_kwargs: Any) -> str:
            lifecycle.append("provider-start")
            entered.set()
            release.wait(timeout=2.0)
            lifecycle.append("provider-exit")
            return "4"

        breaker = LLMCircuitBreaker()
        breaker.provider_configs["openai"] = CircuitBreakerConfig(timeout=0.01)
        pool = TrackingPool(1)
        monkeypatch.setattr(rag_evaluator, "llm_circuit_breaker", breaker)
        monkeypatch.setattr(rag_evaluator, "SYNC_ADAPTER_CALL_POOL", pool)
        monkeypatch.setattr(rag_evaluator, "analyze", blocking_analyze)
        evaluator = RAGEvaluator(embedding_provider=None, embedding_model=None)

        task = asyncio.create_task(
            evaluator._evaluate_relevance("query", "response", "openai")
        )
        try:
            for _ in range(1000):
                if entered.is_set():
                    break
                await asyncio.sleep(0.001)
            assert entered.is_set()
            await asyncio.sleep(0.03)

            assert task.done() is False
            assert pool.active_count == 1
            assert lifecycle == ["provider-start"]

            release.set()
            with pytest.raises(ValueError) as exc_info:
                await asyncio.wait_for(task, timeout=1.0)
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        provider_breaker = breaker.get_breaker("openai")
        assert str(exc_info.value) == "Relevance evaluation failed"
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        assert provider_breaker.stats.timeouts == 1
        assert provider_breaker.stats.failed_calls == 1
        assert pool.active_count == 0
        assert lifecycle == [
            "provider-start",
            "provider-exit",
            "capacity-release",
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "metric_name",
        ["answer_similarity", "context_precision", "context_relevance"],
    )
    async def test_sync_rag_analysis_uses_bounded_daemon_boundary(
        self,
        monkeypatch: pytest.MonkeyPatch,
        metric_name: str,
    ) -> None:
        """Direct sync evaluator calls must have both capacity and a deadline."""
        from tldw_Server_API.app.core.Evaluations import rag_evaluator

        bounded_calls: list[dict[str, Any]] = []

        def analyze_score(*_args, **_kwargs) -> str:
            return "4"

        async def run_inline(
            call,
            *,
            pool,
            name: str,
            timeout_seconds: float,
            timeout_message: str,
            drain_after_timeout: bool = False,
        ) -> Any:
            bounded_calls.append(
                {
                    "pool": pool,
                    "name": name,
                    "timeout_seconds": timeout_seconds,
                    "timeout_message": timeout_message,
                    "drain_after_timeout": drain_after_timeout,
                }
            )
            return call()

        monkeypatch.setattr(rag_evaluator, "analyze", analyze_score)
        monkeypatch.setattr(
            rag_evaluator,
            "await_bounded_daemon_with_timeout",
            run_inline,
            raising=False,
        )
        evaluator = RAGEvaluator(embedding_provider=None, embedding_model=None)

        if metric_name == "answer_similarity":
            metric, _result = await evaluator._evaluate_answer_similarity(
                "response alpha",
                "ground truth beta",
                "openai",
            )
        elif metric_name == "context_precision":
            metric, _result = await evaluator._evaluate_context_precision(
                "query",
                ["context"],
                "openai",
            )
        else:
            metric, _result = await evaluator._evaluate_context_relevance(
                "query",
                ["context"],
                "openai",
            )

        assert metric == metric_name
        assert len(bounded_calls) == 1
        assert bounded_calls[0]["pool"] is rag_evaluator.SYNC_ADAPTER_CALL_POOL
        assert bounded_calls[0]["name"] == "rag-evaluation-analyze"
        assert bounded_calls[0]["timeout_seconds"] > 0
        assert bounded_calls[0]["timeout_message"] == "RAG evaluation provider call timed out"
        assert bounded_calls[0]["drain_after_timeout"] is True

    @pytest.mark.asyncio
    async def test_sync_rag_analysis_bypasses_saturated_default_executor(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A provider call starts without queueing in the default executor."""

        from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
        from tldw_Server_API.app.core.Evaluations import rag_evaluator

        loop = asyncio.get_running_loop()
        previous_executor = getattr(loop, "_default_executor", None)
        saturated_executor = ThreadPoolExecutor(max_workers=1)
        blocker_started = threading.Event()
        release_blocker = threading.Event()
        analyze_started = threading.Event()
        pool = BoundedDaemonPool(1)

        def block_default_executor() -> None:
            blocker_started.set()
            release_blocker.wait(timeout=2.0)

        def analyze_score(*_args, **_kwargs) -> str:
            analyze_started.set()
            return "4"

        monkeypatch.setattr(rag_evaluator, "analyze", analyze_score)
        monkeypatch.setattr(rag_evaluator, "SYNC_ADAPTER_CALL_POOL", pool)
        loop.set_default_executor(saturated_executor)
        blocker = loop.run_in_executor(None, block_default_executor)
        while not blocker_started.is_set():
            await asyncio.sleep(0)

        caller = asyncio.create_task(
            rag_evaluator._run_bounded_rag_analyze("openai", "input")
        )
        try:
            for _ in range(100):
                if analyze_started.is_set():
                    break
                await asyncio.sleep(0.001)
            started_before_release = analyze_started.is_set()
        finally:
            release_blocker.set()
            await blocker
            replacement_executor = previous_executor or ThreadPoolExecutor()
            loop.set_default_executor(replacement_executor)
            saturated_executor.shutdown(wait=True, cancel_futures=True)

        assert await caller == "4"
        assert started_before_release is True
        assert pool.active_count == 0

    @pytest.mark.asyncio
    async def test_cancelled_rag_analysis_never_starts_late_from_executor_queue(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancellation cannot leave an executor-queued provider call behind."""

        from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
        from tldw_Server_API.app.core.Evaluations import rag_evaluator

        loop = asyncio.get_running_loop()
        previous_executor = getattr(loop, "_default_executor", None)
        saturated_executor = ThreadPoolExecutor(max_workers=1)
        blocker_started = threading.Event()
        release_blocker = threading.Event()
        analyze_started = threading.Event()
        release_analyze = threading.Event()
        cancel_requested = threading.Event()
        starts_after_cancel: list[bool] = []
        pool = BoundedDaemonPool(1)

        def block_default_executor() -> None:
            blocker_started.set()
            release_blocker.wait(timeout=2.0)

        def analyze_score(*_args, **_kwargs) -> str:
            starts_after_cancel.append(cancel_requested.is_set())
            analyze_started.set()
            release_analyze.wait(timeout=2.0)
            return "4"

        monkeypatch.setattr(rag_evaluator, "analyze", analyze_score)
        monkeypatch.setattr(rag_evaluator, "SYNC_ADAPTER_CALL_POOL", pool)
        loop.set_default_executor(saturated_executor)
        blocker = loop.run_in_executor(None, block_default_executor)
        while not blocker_started.is_set():
            await asyncio.sleep(0)

        caller = asyncio.create_task(
            rag_evaluator._run_bounded_rag_analyze("openai", "input")
        )
        try:
            for _ in range(100):
                if analyze_started.is_set():
                    break
                await asyncio.sleep(0.001)
            cancel_requested.set()
            caller.cancel()
            release_analyze.set()
            release_blocker.set()
            await blocker
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(caller, timeout=1.0)
        finally:
            release_analyze.set()
            release_blocker.set()
            replacement_executor = previous_executor or ThreadPoolExecutor()
            loop.set_default_executor(replacement_executor)
            saturated_executor.shutdown(wait=True, cancel_futures=True)

        assert starts_after_cancel == [False]
        assert pool.active_count == 0

    @pytest.mark.asyncio
    async def test_rag_timeout_drains_before_runtime_close_and_rejects_overcapacity(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A deadline drains real provider exit before timeout cleanup can close."""
        from tldw_Server_API.app.core.Chat.bounded_daemon import (
            BoundedDaemonPool,
            DaemonCapacityError,
        )
        from tldw_Server_API.app.core.Evaluations import rag_evaluator

        started = threading.Event()
        release = threading.Event()
        call_count = 0
        lifecycle: list[str] = []

        class TrackingPool(BoundedDaemonPool):
            def _release_capacity(self) -> None:
                lifecycle.append("capacity-release")
                super()._release_capacity()

        pool = TrackingPool(1)

        def blocking_analyze(*_args, **_kwargs) -> str:
            nonlocal call_count
            call_count += 1
            lifecycle.append("provider-start")
            started.set()
            release.wait(timeout=2)
            lifecycle.append("provider-exit")
            return "4"

        async def invoke_with_runtime() -> Any:
            try:
                return await rag_evaluator._run_bounded_rag_analyze(
                    "openai",
                    "input",
                )
            finally:
                lifecycle.append("runtime-close")

        monkeypatch.setattr(rag_evaluator, "analyze", blocking_analyze)
        monkeypatch.setattr(rag_evaluator, "SYNC_ADAPTER_CALL_POOL", pool)
        monkeypatch.setattr(rag_evaluator, "RAG_EVALUATION_CALL_TIMEOUT_SECONDS", 0.01)

        task = asyncio.create_task(invoke_with_runtime())
        assert await asyncio.to_thread(started.wait, 1)
        await asyncio.sleep(0.03)
        assert task.done() is False
        assert pool.active_count == 1
        assert lifecycle == ["provider-start"]

        with pytest.raises(DaemonCapacityError):
            await rag_evaluator._run_bounded_rag_analyze("openai", "second")
        assert call_count == 1

        release.set()
        with pytest.raises(TimeoutError, match="RAG evaluation provider call timed out"):
            await asyncio.wait_for(task, timeout=1.0)

        assert pool.active_count == 0
        assert lifecycle == [
            "provider-start",
            "provider-exit",
            "capacity-release",
            "runtime-close",
        ]

    @pytest.mark.asyncio
    async def test_cancelled_rag_call_drains_before_runtime_close(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Caller cancellation stays authoritative after real provider cleanup."""
        from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
        from tldw_Server_API.app.core.Evaluations import rag_evaluator

        started = threading.Event()
        release = threading.Event()
        lifecycle: list[str] = []

        class TrackingPool(BoundedDaemonPool):
            def _release_capacity(self) -> None:
                lifecycle.append("capacity-release")
                super()._release_capacity()

        def blocking_analyze(*_args, **_kwargs) -> str:
            lifecycle.append("provider-start")
            started.set()
            release.wait(timeout=2)
            lifecycle.append("provider-exit")
            return "4"

        async def invoke_with_runtime() -> Any:
            try:
                return await rag_evaluator._run_bounded_rag_analyze(
                    "openai",
                    "input",
                )
            finally:
                lifecycle.append("runtime-close")

        pool = TrackingPool(1)
        monkeypatch.setattr(rag_evaluator, "analyze", blocking_analyze)
        monkeypatch.setattr(rag_evaluator, "SYNC_ADAPTER_CALL_POOL", pool)
        monkeypatch.setattr(rag_evaluator, "RAG_EVALUATION_CALL_TIMEOUT_SECONDS", 1.0)

        task = asyncio.create_task(invoke_with_runtime())
        try:
            assert await asyncio.to_thread(started.wait, 1)
            task.cancel()
            await asyncio.sleep(0.03)
            assert task.done() is False
            assert pool.active_count == 1
            assert lifecycle == ["provider-start"]

            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        assert pool.active_count == 0
        assert lifecycle == [
            "provider-start",
            "provider-exit",
            "capacity-release",
            "runtime-close",
        ]

    @pytest.mark.asyncio
    @patch('tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib.analyze')
    async def test_concurrent_evaluations(self, mock_analyze):
        """Test running multiple evaluations concurrently."""
        # The actual code expects plain numeric string, not JSON
        mock_analyze.return_value = "4.0"

        evaluator = RAGEvaluator()

        # Create multiple evaluation tasks
        tasks = []
        for i in range(5):
            task = evaluator.evaluate(
                query=f"Query {i}",
                contexts=[f"Context {i}"],
                response=f"Response {i}",
                api_name="openai"
            )
            tasks.append(task)

        # Run concurrently
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for result in results:
            assert "metrics" in result
            assert "overall_score" in result
