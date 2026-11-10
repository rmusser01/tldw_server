"""
Unit tests for EvaluationManager.

Tests the core evaluation management functionality with minimal mocking
(only external services like LLMs are mocked).
"""

import pytest
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager
from tldw_Server_API.tests.Evaluations.fixtures.sample_data import (
    SampleDataGenerator,
    generate_evaluation_request
)
from tldw_Server_API.tests.Evaluations.fixtures.database import TestDatabaseHelper


@pytest.mark.unit
class TestEvaluationManagerInit:
    """Test EvaluationManager initialization."""

    def test_init_with_default_config(self, temp_db_path):

        """Test initialization with default configuration."""
        manager = EvaluationManager()
        assert manager.db_path is not None
        assert manager.config is not None

    def test_init_with_custom_db_path(self, temp_db_path):

        """Test initialization with custom database path."""
        with patch.object(EvaluationManager, '_get_db_path', return_value=temp_db_path):
            manager = EvaluationManager()
            assert manager.db_path == temp_db_path

    def test_db_path_sanitization(self, temp_db_path):

        """Test that database path is properly sanitized."""
        manager = EvaluationManager()

        # Test various malicious path attempts
        dangerous_paths = [
            "../../etc/passwd",
            "/etc/passwd\x00.db",
            "../../../sensitive_file",
            "Databases/../../../etc/shadow"
        ]

        for dangerous_path in dangerous_paths:
            with patch.object(manager, 'config') as mock_config:
                mock_config.has_section.return_value = True
                mock_config.get.return_value = dangerous_path

                safe_path = manager._get_db_path()

                # Ensure path doesn't contain directory traversal
                assert ".." not in str(safe_path)
                # Ensure path doesn't contain null bytes
                assert "\x00" not in str(safe_path)

    def test_explicit_db_path_outside_trusted_roots_falls_back_to_default(self, monkeypatch):
        """Explicit paths should not bypass the manager's trusted-path rules."""
        import tldw_Server_API.app.core.Evaluations.evaluation_manager as manager_module

        monkeypatch.setattr(
            manager_module.EvaluationManager,
            "_init_database",
            lambda self: None,
            raising=False,
        )

        manager = manager_module.EvaluationManager(
            db_path="/etc/escape-evals.db",
            user_id="tenant-user",
        )

        assert manager.db_path == DatabasePaths.get_evaluations_db_path("tenant-user")

    def test_blank_string_user_id_is_rejected(self, monkeypatch):
        """Blank string user IDs must fail closed instead of falling back to the shared DB."""
        import tldw_Server_API.app.core.Evaluations.evaluation_manager as manager_module

        monkeypatch.setattr(
            manager_module.EvaluationManager,
            "_init_database",
            lambda self: None,
            raising=False,
        )

        with pytest.raises(ValueError, match="Evaluations user scope is required"):
            manager_module.EvaluationManager(user_id="   ")

    def test_database_migration_on_init(self, temp_db_path):

        """Test that database migrations are applied on initialization."""
        with patch('tldw_Server_API.app.core.Evaluations.evaluation_manager.migrate_evaluations_database') as mock_migrate:
            with patch.object(EvaluationManager, '_get_db_path', return_value=temp_db_path):
                manager = EvaluationManager()
                # Note: _init_database is called in __init__, so it should have been called once already
                mock_migrate.assert_called_with(temp_db_path)


@pytest.mark.unit
class TestEvaluationStorage:
    """Test evaluation storage and retrieval."""

    @pytest.mark.asyncio
    async def test_store_evaluation(self, evaluation_manager):
        """Test storing an evaluation result."""
        eval_data = SampleDataGenerator.generate_geval_data()

        result = await evaluation_manager.store_evaluation(
            evaluation_type="g_eval",
            input_data=eval_data,
            results={"score": 4.5, "reasoning": "Good quality"},
            metadata={"test": True}
        )

        assert isinstance(result, str)  # Should return the evaluation_id

        # Verify data was stored
        with sqlite3.connect(evaluation_manager.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM internal_evaluations WHERE evaluation_id = ?",
                (result,)
            )
            row = cursor.fetchone()
            assert row is not None

    @pytest.mark.asyncio
    async def test_retrieve_evaluation(self, evaluation_manager):
        """Test retrieving evaluation history."""
        eval_data = SampleDataGenerator.generate_rag_evaluation_data()
        results = {"metrics": {"relevance": {"score": 0.9}, "faithfulness": {"score": 0.85}}}

        # Store evaluation
        eval_id = await evaluation_manager.store_evaluation(
            evaluation_type="rag",
            input_data=eval_data,
            results=results
        )

        # Retrieve history
        history = await evaluation_manager.get_history(evaluation_type="rag", limit=1)

        assert history["total_count"] >= 1
        assert len(history["items"]) >= 1
        assert history["items"][0]["evaluation_id"] == eval_id
        assert history["items"][0]["evaluation_type"] == "rag"

    @pytest.mark.asyncio
    async def test_get_history_with_filters(self, evaluation_manager):
        """Test getting evaluation history with various filters."""
        # Create test data
        helper = TestDatabaseHelper(str(evaluation_manager.db_path))
        eval_ids = helper.seed_internal_evaluations(10)

        # Test filtering by type
        g_eval_results = await evaluation_manager.get_history(
            evaluation_type="g_eval",
            limit=5
        )
        assert all(e["evaluation_type"] == "g_eval" for e in g_eval_results["items"])

        # Test pagination
        page1 = await evaluation_manager.get_history(limit=3)
        assert len(page1["items"]) <= 3

        if len(page1["items"]) == 3:
            page2 = await evaluation_manager.get_history(
                limit=3,
                offset=3
            )
            # Ensure no overlap between pages
            page1_ids = {e["evaluation_id"] for e in page1["items"]}
            page2_ids = {e["evaluation_id"] for e in page2["items"]}
            assert len(page1_ids & page2_ids) == 0


@pytest.mark.unit
class TestEvaluationHistory:
    """Test evaluation history tracking."""

    @pytest.mark.asyncio
    async def test_save_evaluation_history(self, evaluation_manager):
        """Test saving evaluation to history."""
        eval_data = SampleDataGenerator.generate_response_quality_data()

        # Store evaluation (there's no separate save_to_history method)
        history_id = await evaluation_manager.store_evaluation(
            evaluation_type="response_quality",
            input_data=eval_data,
            results={
                "metrics": {
                    "coherence": {"score": 0.85},
                    "relevance": {"score": 0.90},
                    "overall": {"score": 0.875}
                }
            },
            metadata={"model": "gpt-4", "user_id": "test_user"}
        )

        assert history_id is not None

        # Verify history was saved
        history = await evaluation_manager.get_history(
            limit=1
        )
        assert len(history["items"]) >= 1
        assert any(h["evaluation_id"] == history_id for h in history["items"])

    @pytest.mark.asyncio
    async def test_get_evaluation_history_with_date_range(self, evaluation_manager):
        """Test retrieving evaluation history with date filtering."""
        # Create evaluations at different times
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        for i in range(5):
            eval_time = now - timedelta(days=i)
            with patch('tldw_Server_API.app.core.Evaluations.evaluation_manager.datetime') as mock_dt:
                mock_dt.now.return_value = eval_time
                mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

                await evaluation_manager.store_evaluation(
                    evaluation_type="test",
                    input_data={"index": i},
                    results={"metrics": {"score": {"score": i}}},
                    metadata={"user_id": "test_user"}
                )

        # Get history for last 3 days
        recent_history = await evaluation_manager.get_history(
            start_date=now - timedelta(days=2),
            end_date=now
        )

        assert len(recent_history["items"]) <= 3


@pytest.mark.unit
class TestCustomMetrics:
    """Test custom metric evaluation functionality."""

    @pytest.mark.asyncio
    async def test_evaluate_custom_metric(self, evaluation_manager):
        """Test evaluating with a custom metric."""
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            # Mock the asyncio.to_thread call to return the expected response
            mock_to_thread.return_value = '{"score": 8.8, "explanation": "Custom metric evaluation"}'

            result = await evaluation_manager.evaluate_custom_metric(
                metric_name="technical_accuracy",
                description="Technical accuracy metric",
                evaluation_prompt="Evaluate the technical accuracy of {content}",
                input_data={"content": "The TCP protocol uses a 3-way handshake"},
                scoring_criteria={"factual_correctness": "Rate from 0 to 10 based on factual correctness"},
                api_name="openai"
            )

            assert result is not None
            assert "score" in result
            # The prompt declares 1-10, so (8.8 - 1) / 9, not 8.8 / 10 (which floored 1 at 0.1).
            assert result["score"] == pytest.approx((8.8 - 1) / 9)
            mock_to_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_multiple_custom_metrics(self, evaluation_manager):
        """Test evaluating with multiple custom metrics."""
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = [
                '{"score": 8.5}',
                '{"score": 9.0}',
                '{"score": 7.8}'
            ]

            # Test multiple custom metrics in sequence
            results = []
            for name, prompt in [("clarity", "Rate clarity"), ("depth", "Rate depth"), ("accuracy", "Rate accuracy")]:
                result = await evaluation_manager.evaluate_custom_metric(
                    metric_name=name,
                    description=f"{name} metric",
                    evaluation_prompt=prompt + " for {content}",
                    input_data={"content": "Test content"},
                    scoring_criteria={name: f"Rate {name} from 0 to 10"},
                    api_name="openai"
                )
                results.append(result)

            assert len(results) == 3
            # Affine over the declared 1-10 scale (TASK-13328), not raw / 10.
            assert results[0]["score"] == pytest.approx((8.5 - 1) / 9)
            assert results[1]["score"] == pytest.approx((9.0 - 1) / 9)
            assert results[2]["score"] == pytest.approx((7.8 - 1) / 9)

    @pytest.mark.asyncio
    async def test_evaluate_custom_metric_generates_model_response(self, evaluation_manager):
        """Test that placeholder model responses are generated before scoring."""
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = [
                "Synthetic model answer",
                '{"score": 7.5, "explanation": "Generated answer was mostly correct"}',
            ]

            result = await evaluation_manager.evaluate_custom_metric(
                metric_name="bullshit_detection",
                description="Detect nonsense handling",
                evaluation_prompt="Question: {question}\nModel Response: {response}",
                input_data={"question": "What is 2+2?", "response": "{model_response}"},
                scoring_criteria={"detection": "Score 0-10"},
                api_name="openai",
            )

            assert result["score"] == pytest.approx((7.5 - 1) / 9)  # declared 1-10 scale
            assert mock_to_thread.call_count == 2

            first_call = mock_to_thread.call_args_list[0]
            assert "What is 2+2?" in str(first_call.args[2])

            second_call = mock_to_thread.call_args_list[1]
            assert "Synthetic model answer" in str(second_call.args[2])


@pytest.mark.unit
class TestEvaluationComparison:
    """Test evaluation comparison functionality."""

    @pytest.mark.asyncio
    async def test_compare_evaluations(self, evaluation_manager):
        """Test comparing two evaluations."""
        # Create two evaluations to compare
        eval1_id = await evaluation_manager.store_evaluation(
            evaluation_type="g_eval",
            input_data={"text": "Sample text 1"},
            results={
                "metrics": {
                    "coherence": {"score": 0.85},
                    "fluency": {"score": 0.90}
                }
            },
            metadata={"model": "gpt-4"}
        )

        eval2_id = await evaluation_manager.store_evaluation(
            evaluation_type="g_eval",
            input_data={"text": "Sample text 2"},
            results={
                "metrics": {
                    "coherence": {"score": 0.78},
                    "fluency": {"score": 0.95}
                }
            },
            metadata={"model": "gpt-3.5-turbo"}
        )

        comparison = await evaluation_manager.compare_evaluations(
            evaluation_ids=[eval1_id, eval2_id]
        )

        assert comparison is not None
        assert "metric_comparisons" in comparison
        assert "best_performing" in comparison
        assert "comparison_summary" in comparison
        if "coherence" in comparison["metric_comparisons"]:
            assert len(comparison["metric_comparisons"]["coherence"]) == 2

    @pytest.mark.asyncio
    async def test_compare_evaluation_batches(self, evaluation_manager):
        """Test comparing batches of evaluations."""
        # Create multiple evaluations for batch comparison
        batch_ids = []

        for i in range(3):
            id1 = await evaluation_manager.store_evaluation(
                evaluation_type="rag",
                input_data={"query": f"Query {i}"},
                results={
                    "metrics": {
                        "relevance": {"score": 0.80 + i * 0.02}
                    }
                },
                metadata={"batch": "batch1"}
            )
            batch_ids.append(id1)

            id2 = await evaluation_manager.store_evaluation(
                evaluation_type="rag",
                input_data={"query": f"Query {i}"},
                results={
                    "metrics": {
                        "relevance": {"score": 0.75 + i * 0.03}
                    }
                },
                metadata={"batch": "batch2"}
            )
            batch_ids.append(id2)

        # Compare all evaluations
        batch_comparison = await evaluation_manager.compare_evaluations(
            evaluation_ids=batch_ids
        )

        assert batch_comparison is not None
        assert "metric_comparisons" in batch_comparison
        assert "best_performing" in batch_comparison
        assert "statistical_analysis" in batch_comparison or len(batch_ids) <= 2


@pytest.mark.unit
class TestStatisticalAnalysis:
    """Test statistical analysis of evaluations."""

    @pytest.mark.asyncio
    async def test_calculate_statistics(self, evaluation_manager):
        """Test calculating statistics for evaluation results."""
        # Create sample evaluation results
        scores = [0.75, 0.80, 0.85, 0.82, 0.78, 0.90, 0.88, 0.79]

        eval_ids = []
        for i, score in enumerate(scores):
            eval_id = await evaluation_manager.store_evaluation(
                evaluation_type="test_stat",
                input_data={"index": i},
                results={
                    "metrics": {
                        "score": {"score": score}
                    }
                },
                metadata={"group": "test_group"}
            )
            eval_ids.append(eval_id)

        # Get history to analyze statistics
        history = await evaluation_manager.get_history(
            evaluation_type="test_stat",
            limit=len(scores)
        )

        assert history is not None
        assert "average_scores" in history
        if "score" in history["average_scores"]:
            assert abs(history["average_scores"]["score"] - 0.82125) < 0.01

    @pytest.mark.asyncio
    async def test_trend_analysis(self, evaluation_manager):
        """Test trend analysis over time."""
        # Create evaluations with improving scores over time
        from datetime import datetime, timezone
        base_date = datetime.now(timezone.utc)

        for i in range(10):
            eval_date = base_date + timedelta(days=i)
            score = 0.70 + (i * 0.02)  # Improving trend

            with patch('tldw_Server_API.app.core.Evaluations.evaluation_manager.datetime') as mock_dt:
                mock_dt.now.return_value = eval_date
                mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

                await evaluation_manager.store_evaluation(
                    evaluation_type="trend_test",
                    input_data={"day": i},
                    results={
                        "metrics": {
                            "score": {"score": score}
                        }
                    }
                )

        # Get history with enough data to see trends
        history = await evaluation_manager.get_history(
            evaluation_type="trend_test",
            limit=10
        )

        assert history is not None
        # Check if trends are calculated when we have enough data
        if history["total_count"] > 10:
            assert history["trends"] is not None


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in EvaluationManager."""

    @pytest.mark.asyncio
    async def test_handle_database_error(self, evaluation_manager):
        """Test handling of database errors."""
        # Simulate database connection error
        with patch.object(evaluation_manager, 'db_path', Path("/nonexistent/path/db.db")):
            with pytest.raises(Exception):
                await evaluation_manager.store_evaluation(
                    evaluation_type="test",
                    input_data={},
                    results={}
                )

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self, evaluation_manager):
        """Test handling of invalid JSON data."""
        # Try to store invalid JSON (circular reference)
        circular_ref = {}
        circular_ref["self"] = circular_ref

        with pytest.raises(ValueError) as exc_info:
            await evaluation_manager.store_evaluation(
                evaluation_type="test",
                input_data=circular_ref,
                results={}
            )
        assert "Circular reference detected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handle_llm_failure(self, evaluation_manager):
        """Test handling of LLM API failures."""
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.side_effect = Exception("LLM API error")

            # Call with correct parameters for evaluate_custom_metric
            result = await evaluation_manager.evaluate_custom_metric(
                metric_name="test",
                description="Test metric",
                evaluation_prompt="Test prompt for {content}",
                input_data={"content": "Test content"},
                scoring_criteria={"test": "Test criteria"},
                api_name="openai"
            )

            # When LLM fails, it should return a result with score 0.0 and error message
            assert result["score"] == 0.0
            assert "Evaluation failed" in result["explanation"]

    @pytest.mark.asyncio
    async def test_handle_concurrent_access(self, evaluation_manager):
        """Test handling of concurrent database access."""
        import asyncio

        async def concurrent_write(index):
            return await evaluation_manager.store_evaluation(
                evaluation_type="test",
                input_data={"index": index},
                results={"metrics": {"score": {"score": index}}}
            )

        # Start multiple tasks writing concurrently
        tasks = [concurrent_write(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check that no errors occurred
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0

        # Verify all writes succeeded
        eval_ids = [r for r in results if isinstance(r, str)]
        assert len(eval_ids) == 10


@pytest.mark.unit
@pytest.mark.asyncio
async def test_evaluate_custom_metric_lets_an_llm_quota_refusal_through(evaluation_manager):
    """A refusal is not scored 0.0 (TASK-13638)."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    refusal = LLMQuotaExceededError(
        category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
    )
    with patch("tldw_Server_API.app.core.Evaluations.evaluation_manager.analyze", side_effect=refusal):
        with pytest.raises(LLMQuotaExceededError):
            await evaluation_manager.evaluate_custom_metric(
                metric_name="accuracy",
                description="d",
                evaluation_prompt="Rate {answer}",
                input_data={"answer": "a"},
                scoring_criteria={"accuracy": "0-10"},
                api_name="openai",
            )


_MC_ITEM = {"question": "What is 2 + 2?", "choices": ["3", "4"], "correct_answer": 1}
_QA_ITEM = {"question": "What is the capital of France?", "answer": "Paris"}
_MC_ASKS = "Score 10 for a correct answer, 1 for an"
_QA_ASKS = "GRADE: [correct/incorrect"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("benchmark", "item", "reply", "asked", "expected"),
    [
        ("simple_bench", _MC_ITEM, "SCORE: 10 EXPLANATION: B.", _MC_ASKS, 1.0),
        ("simple_bench", _MC_ITEM, "SCORE: 1 EXPLANATION: A.", _MC_ASKS, 0.0),
        ("simple_bench", _MC_ITEM, "I cannot judge this.", _MC_ASKS, 0.5),
        ("mmlu_pro", _MC_ITEM, "SCORE: 8.2 EXPLANATION: B, thin.", "with N from 1 (worst) to 10 (best)", 0.8),
        ("simpleqa_verified", _QA_ITEM, "GRADE: correct\nEXPLANATION: Paris.", _QA_ASKS, 1.0),
        ("simpleqa_verified", _QA_ITEM, "GRADE: incorrect\nEXPLANATION: Lyon.", _QA_ASKS, 0.0),
        ("simpleqa_verified", _QA_ITEM, "GRADE: correct\nEXPLANATION: Paris.\nScore: 7", _QA_ASKS, 1.0),
        ("simpleqa_verified", _QA_ITEM, "GRADE: [incorrect]\nEXPLANATION: Lyon.", _QA_ASKS, 0.0),
        # a failed judge call is a 0.0 with the error, not a silent 0.5 counted as a success
        ("simple_bench", _MC_ITEM, RuntimeError("429 rate limited"), _MC_ASKS, 0.0),
        # the judge restates the rubric before its own score line: the labelled line wins
        ("simple_bench", _MC_ITEM,
         "Per the rubric: Score 10 for a correct answer, 1 for an incorrect one. It chose A.\nSCORE: 1", _MC_ASKS, 0.0),
        ("simple_bench", _MC_ITEM, "**SCORE:** 10\nEXPLANATION: B.", _MC_ASKS, 1.0),
        ("simple_bench", _MC_ITEM, "Score: **10**", _MC_ASKS, 1.0),
        # "Grade:" in a multiple-choice reply is prose; only a prompt that asks for a GRADE reads one
        ("simple_bench", _MC_ITEM, "Grade: Correct answer is B; the model chose A.\nSCORE: 1", _MC_ASKS, 0.0),
    ],
    ids=["mc-correct", "mc-incorrect", "no-score", "mc-reasoning", "simpleqa-correct", "simpleqa-incorrect",
         "simpleqa-grade-decides", "simpleqa-bracketed-grade", "judge-error", "rubric-echo", "markdown-label",
         "markdown-value", "mc-grade-is-prose"],
)
async def test_a_benchmark_item_is_scored_from_the_judges_reply(
    evaluation_manager, monkeypatch, benchmark, item, reply, asked, expected
):
    """A benchmark's own formatter, through the real analyze(): the model answers, the judge
    scores that answer on the scale it was asked for, and the item's score is the judge's, not
    the 0.5 default (TASK-15485)."""
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Evaluations.benchmark_registry import get_registry
    from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summarization

    prompts: list[str] = []

    def chat(request, timeout=None):
        prompts.append(request["messages"][-1]["content"])
        if isinstance(reply, Exception) and len(prompts) > 1:
            raise reply
        return {"choices": [{"message": {"content": "B. 4" if len(prompts) == 1 else str(reply)}}]}

    monkeypatch.setattr(
        summarization, "get_registry", lambda: SimpleNamespace(get_adapter=lambda _p: SimpleNamespace(chat=chat))
    )
    data = get_registry().create_evaluator(benchmark).format_for_custom_metric(item)
    result = await evaluation_manager.evaluate_custom_metric(
        metric_name=data["name"],
        description=data["description"],
        evaluation_prompt=data["evaluation_prompt"],
        input_data=data["input_data"],
        scoring_criteria=data["scoring_criteria"],
        api_name="openai",
        api_key="sk-test",
    )
    assert len(prompts) == 2, prompts  # the model's answer, then the judge
    assert "B. 4" in prompts[1] and asked in prompts[1]
    assert result["score"] == pytest.approx(expected)
    if isinstance(reply, Exception):
        assert result["explanation"].startswith("Error:")
