"""Tests for evaluation workflow adapters.

This module provides comprehensive tests for the evaluation adapters:
- run_evaluations_adapter: Run LLM evaluations (geval, rag, response_quality)
- run_quiz_evaluate_adapter: Evaluate quiz answers
- run_eval_readability_adapter: Evaluate text readability
- run_context_window_check_adapter: Check context window limits
"""

import pytest
from typing import Any, Dict

pytestmark = pytest.mark.unit


# ============================================================================
# Test imports
# ============================================================================

def test_evaluation_adapters_importable():
    """Test that all evaluation adapters can be imported."""
    from tldw_Server_API.app.core.Workflows.adapters import (
        run_evaluations_adapter,
        run_quiz_evaluate_adapter,
        run_eval_readability_adapter,
        run_context_window_check_adapter,
    )

    assert callable(run_evaluations_adapter)
    assert callable(run_quiz_evaluate_adapter)
    assert callable(run_eval_readability_adapter)
    assert callable(run_context_window_check_adapter)


def test_evaluation_adapters_registered():
    """Test that evaluation adapters are registered in the registry."""
    from tldw_Server_API.app.core.Workflows.adapters import registry

    adapters = registry.list_adapters()
    assert "evaluations" in adapters
    assert "quiz_evaluate" in adapters
    assert "eval_readability" in adapters
    assert "context_window_check" in adapters


def test_evaluation_adapters_have_config_models():
    """Test that evaluation adapters have Pydantic config models."""
    from tldw_Server_API.app.core.Workflows.adapters import registry

    for adapter_name in ["evaluations", "quiz_evaluate", "eval_readability", "context_window_check"]:
        spec = registry.get_spec(adapter_name)
        assert spec.config_model is not None, f"{adapter_name} missing config_model"


def test_evaluation_adapters_in_evaluation_category():
    """Test that evaluation adapters are in the evaluation category."""
    from tldw_Server_API.app.core.Workflows.adapters import registry

    # get_by_category returns adapter names in the specified category
    evaluation_adapter_names = registry.get_by_category("evaluation")

    assert "evaluations" in evaluation_adapter_names
    assert "quiz_evaluate" in evaluation_adapter_names
    assert "eval_readability" in evaluation_adapter_names
    assert "context_window_check" in evaluation_adapter_names


# ============================================================================
# run_evaluations_adapter tests
# ============================================================================

class TestEvaluationsAdapter:
    """Tests for run_evaluations_adapter."""

    @pytest.mark.asyncio
    async def test_evaluations_adapter_geval_test_mode(self, monkeypatch):
        """Test evaluations adapter geval action in test mode."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters import run_evaluations_adapter

        config = {
            "action": "geval",
            "response": "The sky is blue because of light scattering.",
            "context": "The original article discusses how the sky appears blue.",
            "criteria": ["relevance", "coherence", "fluency"],
        }
        context = {"user_id": "1"}

        result = await run_evaluations_adapter(config, context)

        assert "score" in result
        assert "metrics" in result
        assert "passed" in result
        assert result.get("simulated") is True
        assert result.get("evaluation_id") == "test-eval-geval"
        assert isinstance(result["metrics"], dict)
        # Check that all criteria have scores
        for criterion in ["relevance", "coherence", "fluency"]:
            assert criterion in result["metrics"]

    @pytest.mark.asyncio
    async def test_evaluations_adapter_rag_test_mode(self, monkeypatch):
        """Test evaluations adapter rag action in test mode."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters import run_evaluations_adapter

        config = {
            "action": "rag",
            "question": "What is photosynthesis?",
            "response": "Photosynthesis is the process by which plants convert sunlight into energy.",
            "retrieved_contexts": ["Plants use chlorophyll to absorb sunlight."],
            "reference_answer": "Photosynthesis is the process used by plants to convert light energy.",
        }
        context = {"user_id": "1"}

        result = await run_evaluations_adapter(config, context)

        assert result.get("simulated") is True
        assert result.get("evaluation_id") == "test-eval-rag"
        assert "score" in result
        assert "metrics" in result
        assert "faithfulness" in result["metrics"]
        assert "answer_relevance" in result["metrics"]
        assert "context_relevance" in result["metrics"]
        assert result.get("passed") is True

    @pytest.mark.asyncio
    async def test_evaluations_adapter_response_quality_test_mode(self, monkeypatch):
        """Test evaluations adapter response_quality action in test mode."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters import run_evaluations_adapter

        config = {
            "action": "response_quality",
            "prompt": "Explain machine learning.",
            "response": "Machine learning is a subset of AI that enables systems to learn from data.",
        }
        context = {"user_id": "1"}

        result = await run_evaluations_adapter(config, context)

        assert result.get("simulated") is True
        assert result.get("evaluation_id") == "test-eval-quality"
        assert "score" in result
        assert "metrics" in result
        assert "clarity" in result["metrics"]
        assert "completeness" in result["metrics"]
        assert "accuracy" in result["metrics"]

    @pytest.mark.asyncio
    async def test_evaluations_adapter_get_run_test_mode(self, monkeypatch):
        """Test evaluations adapter get_run action in test mode."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters import run_evaluations_adapter

        config = {
            "action": "get_run",
            "run_id": "run-123",
        }
        context = {"user_id": "1"}

        result = await run_evaluations_adapter(config, context)

        assert result.get("simulated") is True
        assert "run" in result
        assert result["run"]["id"] == "run-123"
        assert result["run"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_evaluations_adapter_list_runs_test_mode(self, monkeypatch):
        """Test evaluations adapter list_runs action in test mode."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters import run_evaluations_adapter

        config = {
            "action": "list_runs",
            "limit": 10,
        }
        context = {"user_id": "1"}

        result = await run_evaluations_adapter(config, context)

        assert result.get("simulated") is True
        assert "runs" in result
        assert "has_more" in result
        assert isinstance(result["runs"], list)

    @pytest.mark.asyncio
    async def test_evaluations_adapter_missing_action(self, monkeypatch):
        """Test evaluations adapter returns error for missing action."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters import run_evaluations_adapter

        config = {}
        context = {"user_id": "1"}

        result = await run_evaluations_adapter(config, context)

        assert "error" in result
        assert result["error"] == "missing_action"

    @pytest.mark.asyncio
    async def test_evaluations_adapter_unknown_action_test_mode(self, monkeypatch):
        """Test evaluations adapter returns error for unknown action in test mode."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters import run_evaluations_adapter

        config = {"action": "nonexistent_action"}
        context = {"user_id": "1"}

        result = await run_evaluations_adapter(config, context)

        assert "error" in result
        assert "unknown_action" in result["error"]

    @pytest.mark.asyncio
    async def test_evaluations_adapter_sanitizes_backend_errors(self, monkeypatch):
        """Test evaluations adapter hides raw backend exception details."""
        monkeypatch.delenv("TEST_MODE", raising=False)

        import importlib
        import sys
        import types

        from tldw_Server_API.app.core.Workflows.adapters import run_evaluations_adapter
        evaluation_adapter_module = importlib.import_module(
            "tldw_Server_API.app.core.Workflows.adapters.evaluation.eval"
        )

        class BrokenEvaluationService:
            def __init__(self, **_kwargs):
                raise RuntimeError("evaluation backend exploded at /private/evaluations.db")

        monkeypatch.setitem(
            sys.modules,
            "tldw_Server_API.app.core.Evaluations.unified_evaluation_service",
            types.SimpleNamespace(UnifiedEvaluationService=BrokenEvaluationService),
        )

        messages = []
        sink_id = evaluation_adapter_module.logger.add(
            lambda message: messages.append(str(message)),
            level="ERROR",
        )
        try:
            result = await run_evaluations_adapter(
                {"action": "geval", "response": "answer"},
                {"user_id": "1"},
            )
        finally:
            evaluation_adapter_module.logger.remove(sink_id)

        assert result == {"error": "evaluations_error"}
        assert "evaluations.db" not in str(result)
        joined = "\n".join(messages)
        assert "Evaluations adapter error" in joined
        assert "evaluations.db" not in joined

    @pytest.mark.asyncio
    async def test_evaluations_adapter_cancelled(self, monkeypatch):
        """Test evaluations adapter returns cancelled status when cancelled."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters import run_evaluations_adapter

        config = {"action": "geval", "response": "test"}
        context = {"user_id": "1", "is_cancelled": lambda: True}

        result = await run_evaluations_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_evaluations_adapter_geval_with_comma_separated_criteria(self, monkeypatch):
        """Test evaluations adapter parses comma-separated criteria string."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters import run_evaluations_adapter

        config = {
            "action": "geval",
            "response": "Test response",
            "criteria": "relevance, coherence, fluency",  # comma-separated string
        }
        context = {"user_id": "1"}

        result = await run_evaluations_adapter(config, context)

        assert "metrics" in result
        assert "relevance" in result["metrics"]
        assert "coherence" in result["metrics"]
        assert "fluency" in result["metrics"]

    @pytest.mark.asyncio
    async def test_evaluations_adapter_geval_with_threshold(self, monkeypatch):
        """Test evaluations adapter respects threshold parameter."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters import run_evaluations_adapter

        # Test with low threshold (should pass)
        config_pass = {
            "action": "geval",
            "response": "Test response",
            "threshold": 0.5,
        }
        result_pass = await run_evaluations_adapter(config_pass, {"user_id": "1"})
        assert result_pass.get("passed") is True

        # Test with very high threshold (should fail)
        config_fail = {
            "action": "geval",
            "response": "Test response",
            "threshold": 0.99,
        }
        result_fail = await run_evaluations_adapter(config_fail, {"user_id": "1"})
        # With simulated scores around 0.8-0.95, this should fail
        assert "passed" in result_fail

    @pytest.mark.asyncio
    async def test_evaluations_adapter_uses_last_context_for_response(self, monkeypatch):
        """Test evaluations adapter uses last.text as response fallback."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters import run_evaluations_adapter

        config = {
            "action": "geval",
            # No response specified, should use last.text
        }
        context = {
            "user_id": "1",
            "last": {"text": "Response from previous step"},
        }

        result = await run_evaluations_adapter(config, context)

        # Should not error with missing response
        assert "score" in result or result.get("simulated") is True

    @pytest.mark.asyncio
    async def test_evaluations_adapter_rag_uses_last_documents(self, monkeypatch):
        """Test evaluations adapter uses last.documents for retrieved_contexts."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters import run_evaluations_adapter

        config = {
            "action": "rag",
            "question": "What is X?",
            "response": "X is a thing.",
            # No retrieved_contexts, should use last.documents
        }
        context = {
            "user_id": "1",
            "last": {
                "text": "X is a thing.",
                "documents": [
                    {"content": "Context about X"},
                    {"text": "More context about X"},
                ],
            },
        }

        result = await run_evaluations_adapter(config, context)

        assert result.get("simulated") is True
        assert "score" in result


# ============================================================================
# run_quiz_evaluate_adapter tests
# ============================================================================

class TestQuizEvaluateAdapter:
    """Tests for run_quiz_evaluate_adapter."""

    @pytest.mark.asyncio
    async def test_quiz_evaluate_multiple_choice_correct(self):
        """Test quiz evaluation with correct multiple choice answers."""
        from tldw_Server_API.app.core.Workflows.adapters import run_quiz_evaluate_adapter

        config = {
            "questions": [
                {"id": 0, "correct_answer": "A", "question_type": "multiple_choice"},
                {"id": 1, "correct_answer": "B", "question_type": "multiple_choice"},
                {"id": 2, "correct_answer": "C", "question_type": "multiple_choice"},
            ],
            "answers": [
                {"question_id": 0, "user_answer": "A"},
                {"question_id": 1, "user_answer": "B"},
                {"question_id": 2, "user_answer": "C"},
            ],
        }
        context = {}

        result = await run_quiz_evaluate_adapter(config, context)

        assert result["score"] == 100.0
        assert result["points_earned"] == 3
        assert result["points_possible"] == 3
        assert result["passed"] is True
        assert len(result["results"]) == 3
        assert all(r["is_correct"] for r in result["results"])

    @pytest.mark.asyncio
    async def test_quiz_evaluate_multiple_choice_partial(self):
        """Test quiz evaluation with partially correct answers."""
        from tldw_Server_API.app.core.Workflows.adapters import run_quiz_evaluate_adapter

        config = {
            "questions": [
                {"id": 0, "correct_answer": "A", "question_type": "multiple_choice"},
                {"id": 1, "correct_answer": "B", "question_type": "multiple_choice"},
                {"id": 2, "correct_answer": "C", "question_type": "multiple_choice"},
            ],
            "answers": [
                {"question_id": 0, "user_answer": "A"},  # correct
                {"question_id": 1, "user_answer": "X"},  # wrong
                {"question_id": 2, "user_answer": "C"},  # correct
            ],
        }
        context = {}

        result = await run_quiz_evaluate_adapter(config, context)

        assert result["score"] == pytest.approx(66.67, rel=0.01)
        assert result["points_earned"] == 2
        assert result["points_possible"] == 3
        assert result["passed"] is False  # Default passing_score is 70

    @pytest.mark.asyncio
    async def test_quiz_evaluate_all_wrong(self):
        """Test quiz evaluation with all wrong answers."""
        from tldw_Server_API.app.core.Workflows.adapters import run_quiz_evaluate_adapter

        config = {
            "questions": [
                {"id": 0, "correct_answer": "A"},
                {"id": 1, "correct_answer": "B"},
            ],
            "answers": [
                {"question_id": 0, "user_answer": "X"},
                {"question_id": 1, "user_answer": "Y"},
            ],
        }
        context = {}

        result = await run_quiz_evaluate_adapter(config, context)

        assert result["score"] == 0.0
        assert result["points_earned"] == 0
        assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_quiz_evaluate_weighted_points(self):
        """Test quiz evaluation with weighted question points."""
        from tldw_Server_API.app.core.Workflows.adapters import run_quiz_evaluate_adapter

        config = {
            "questions": [
                {"id": 0, "correct_answer": "A", "points": 5},  # worth 5 points
                {"id": 1, "correct_answer": "B", "points": 2},  # worth 2 points
                {"id": 2, "correct_answer": "C", "points": 3},  # worth 3 points
            ],
            "answers": [
                {"question_id": 0, "user_answer": "A"},  # correct, +5
                {"question_id": 1, "user_answer": "X"},  # wrong, +0
                {"question_id": 2, "user_answer": "C"},  # correct, +3
            ],
        }
        context = {}

        result = await run_quiz_evaluate_adapter(config, context)

        assert result["points_earned"] == 8  # 5 + 0 + 3
        assert result["points_possible"] == 10  # 5 + 2 + 3
        assert result["score"] == 80.0

    @pytest.mark.asyncio
    async def test_quiz_evaluate_text_answer_case_insensitive(self):
        """Test quiz evaluation with case-insensitive text answers."""
        from tldw_Server_API.app.core.Workflows.adapters import run_quiz_evaluate_adapter

        config = {
            "questions": [
                {"id": 0, "correct_answer": "Paris", "question_type": "text"},
                {"id": 1, "correct_answer": "London", "question_type": "text"},
            ],
            "answers": [
                {"question_id": 0, "user_answer": "PARIS"},  # different case
                {"question_id": 1, "user_answer": "london"},  # different case
            ],
        }
        context = {}

        result = await run_quiz_evaluate_adapter(config, context)

        assert result["score"] == 100.0
        assert all(r["is_correct"] for r in result["results"])

    @pytest.mark.asyncio
    async def test_quiz_evaluate_custom_passing_score(self):
        """Test quiz evaluation with custom passing score."""
        from tldw_Server_API.app.core.Workflows.adapters import run_quiz_evaluate_adapter

        config = {
            "questions": [
                {"id": 0, "correct_answer": "A"},
                {"id": 1, "correct_answer": "B"},
            ],
            "answers": [
                {"question_id": 0, "user_answer": "A"},
                {"question_id": 1, "user_answer": "X"},  # wrong
            ],
            "passing_score": 50.0,  # 50% to pass
        }
        context = {}

        result = await run_quiz_evaluate_adapter(config, context)

        assert result["score"] == 50.0
        assert result["passed"] is True  # 50% meets 50% threshold

    @pytest.mark.asyncio
    async def test_quiz_evaluate_missing_questions(self):
        """Test quiz evaluation with missing questions."""
        from tldw_Server_API.app.core.Workflows.adapters import run_quiz_evaluate_adapter

        config = {"answers": [{"question_id": 0, "user_answer": "A"}]}
        context = {}

        result = await run_quiz_evaluate_adapter(config, context)

        assert "error" in result
        assert result["error"] == "missing_questions"

    @pytest.mark.asyncio
    async def test_quiz_evaluate_empty_questions(self):
        """Test quiz evaluation with empty questions list."""
        from tldw_Server_API.app.core.Workflows.adapters import run_quiz_evaluate_adapter

        config = {"questions": [], "answers": []}
        context = {}

        result = await run_quiz_evaluate_adapter(config, context)

        assert "error" in result
        assert result["error"] == "missing_questions"

    @pytest.mark.asyncio
    async def test_quiz_evaluate_from_previous_step(self):
        """Test quiz evaluation uses questions from previous step."""
        from tldw_Server_API.app.core.Workflows.adapters import run_quiz_evaluate_adapter

        config = {
            "answers": [
                {"question_id": 0, "user_answer": "A"},
            ],
        }
        context = {
            "last": {
                "questions": [
                    {"id": 0, "correct_answer": "A"},
                ],
            },
        }

        result = await run_quiz_evaluate_adapter(config, context)

        assert result["score"] == 100.0
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_quiz_evaluate_cancelled(self):
        """Test quiz evaluation returns cancelled status when cancelled."""
        from tldw_Server_API.app.core.Workflows.adapters import run_quiz_evaluate_adapter

        config = {"questions": [{"id": 0, "correct_answer": "A"}], "answers": []}
        context = {"is_cancelled": lambda: True}

        result = await run_quiz_evaluate_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_quiz_evaluate_answers_by_index(self):
        """Test quiz evaluation with answers provided as simple list."""
        from tldw_Server_API.app.core.Workflows.adapters import run_quiz_evaluate_adapter

        config = {
            "questions": [
                {"id": 0, "correct_answer": "A"},
                {"id": 1, "correct_answer": "B"},
            ],
            "answers": ["A", "B"],  # Simple list, matched by index
        }
        context = {}

        result = await run_quiz_evaluate_adapter(config, context)

        assert result["score"] == 100.0


# ============================================================================
# run_eval_readability_adapter tests
# ============================================================================

class TestEvalReadabilityAdapter:
    """Tests for run_eval_readability_adapter."""

    @pytest.mark.asyncio
    async def test_eval_readability_basic(self):
        """Test readability evaluation with basic text."""
        from tldw_Server_API.app.core.Workflows.adapters import run_eval_readability_adapter

        config = {
            "text": "The quick brown fox jumps over the lazy dog. This is a simple sentence."
        }
        context = {}

        result = await run_eval_readability_adapter(config, context)

        assert "scores" in result
        assert "flesch_reading_ease" in result["scores"]
        assert "flesch_kincaid_grade" in result["scores"]
        assert "grade_level" in result
        assert "reading_ease" in result
        assert "word_count" in result
        assert "sentence_count" in result
        assert result["word_count"] > 0
        assert result["sentence_count"] > 0

    @pytest.mark.asyncio
    async def test_eval_readability_complex_text(self):
        """Test readability evaluation with complex text."""
        from tldw_Server_API.app.core.Workflows.adapters import run_eval_readability_adapter

        complex_text = (
            "The epistemological implications of phenomenological hermeneutics "
            "necessitate a comprehensive understanding of transcendental consciousness. "
            "Furthermore, the ontological presuppositions inherent in existentialist "
            "philosophy demand rigorous analytical examination."
        )
        config = {"text": complex_text}
        context = {}

        result = await run_eval_readability_adapter(config, context)

        # Complex text should have higher grade level
        assert result["grade_level"] > 10

    @pytest.mark.asyncio
    async def test_eval_readability_simple_text(self):
        """Test readability evaluation with simple text."""
        from tldw_Server_API.app.core.Workflows.adapters import run_eval_readability_adapter

        simple_text = "The cat sat on the mat. It was a big cat. The cat was happy."
        config = {"text": simple_text}
        context = {}

        result = await run_eval_readability_adapter(config, context)

        # Simple text should have lower grade level and higher reading ease
        assert result["grade_level"] < 8
        assert result["reading_ease"] > 60

    @pytest.mark.asyncio
    async def test_eval_readability_missing_text(self):
        """Test readability evaluation with missing text."""
        from tldw_Server_API.app.core.Workflows.adapters import run_eval_readability_adapter

        config = {}
        context = {}

        result = await run_eval_readability_adapter(config, context)

        assert "error" in result
        assert result["error"] == "missing_text"

    @pytest.mark.asyncio
    async def test_eval_readability_empty_text(self):
        """Test readability evaluation with empty text."""
        from tldw_Server_API.app.core.Workflows.adapters import run_eval_readability_adapter

        config = {"text": "   "}  # whitespace only
        context = {}

        result = await run_eval_readability_adapter(config, context)

        assert "error" in result
        assert result["error"] == "missing_text"

    @pytest.mark.asyncio
    async def test_eval_readability_from_previous_step(self):
        """Test readability evaluation uses text from previous step."""
        from tldw_Server_API.app.core.Workflows.adapters import run_eval_readability_adapter

        config = {}
        context = {
            "last": {
                "text": "This is text from the previous step."
            }
        }

        result = await run_eval_readability_adapter(config, context)

        assert "scores" in result
        assert result["word_count"] == 7

    @pytest.mark.asyncio
    async def test_eval_readability_with_template(self):
        """Test readability evaluation with templated text."""
        from tldw_Server_API.app.core.Workflows.adapters import run_eval_readability_adapter

        config = {"text": "{{ inputs.text }}"}
        context = {
            "inputs": {
                "text": "The dog runs fast. It jumps high. Dogs are fun."
            }
        }

        result = await run_eval_readability_adapter(config, context)

        # Template should be resolved
        # Word count: The(1) dog(2) runs(3) fast(4) It(5) jumps(6) high(7) Dogs(8) are(9) fun(10) = 10 words
        assert result["word_count"] == 10
        assert result["sentence_count"] == 3

    @pytest.mark.asyncio
    async def test_eval_readability_cancelled(self):
        """Test readability evaluation returns cancelled status when cancelled."""
        from tldw_Server_API.app.core.Workflows.adapters import run_eval_readability_adapter

        config = {"text": "Some text"}
        context = {"is_cancelled": lambda: True}

        result = await run_eval_readability_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_eval_readability_single_sentence(self):
        """Test readability evaluation with single sentence."""
        from tldw_Server_API.app.core.Workflows.adapters import run_eval_readability_adapter

        config = {"text": "This is a single sentence with multiple words in it."}
        context = {}

        result = await run_eval_readability_adapter(config, context)

        assert result["sentence_count"] == 1
        assert result["word_count"] == 10

    @pytest.mark.asyncio
    async def test_eval_readability_multiple_punctuation(self):
        """Test readability evaluation handles multiple sentence types."""
        from tldw_Server_API.app.core.Workflows.adapters import run_eval_readability_adapter

        config = {"text": "Is this a question? Yes, it is! Here is a statement."}
        context = {}

        result = await run_eval_readability_adapter(config, context)

        assert result["sentence_count"] == 3

    @pytest.mark.asyncio
    async def test_eval_readability_uses_content_field(self):
        """Test readability evaluation uses content field from previous step."""
        from tldw_Server_API.app.core.Workflows.adapters import run_eval_readability_adapter

        config = {}
        context = {
            "prev": {
                "content": "Text from content field."
            }
        }

        result = await run_eval_readability_adapter(config, context)

        assert result["word_count"] == 4


# ============================================================================
# run_context_window_check_adapter tests
# ============================================================================

class TestContextWindowCheckAdapter:
    """Tests for run_context_window_check_adapter."""

    @pytest.mark.asyncio
    async def test_context_window_check_short_text(self):
        """Test context window check with short text that fits."""
        from tldw_Server_API.app.core.Workflows.adapters import run_context_window_check_adapter

        config = {
            "text": "Hello world, this is a short text.",
            "model": "gpt-4",
            "reserve_tokens": 1000,
        }
        context = {}

        result = await run_context_window_check_adapter(config, context)

        assert result["fits"] is True
        assert "token_count" in result
        assert result["token_count"] < 100
        assert result["context_limit"] == 8192
        assert result["available_tokens"] == 7192
        assert result["excess_tokens"] == 0

    @pytest.mark.asyncio
    async def test_context_window_check_long_text(self):
        """Test context window check with text that exceeds limit."""
        from tldw_Server_API.app.core.Workflows.adapters import run_context_window_check_adapter

        # Create a long text that likely exceeds the context window
        long_text = "word " * 50000  # ~50k words, ~200k characters
        config = {
            "text": long_text,
            "model": "gpt-4",
            "reserve_tokens": 1000,
        }
        context = {}

        result = await run_context_window_check_adapter(config, context)

        assert result["fits"] is False
        assert result["token_count"] > result["available_tokens"]
        assert result["excess_tokens"] > 0

    @pytest.mark.asyncio
    async def test_context_window_check_different_models(self):
        """Test context window check with different models."""
        from tldw_Server_API.app.core.Workflows.adapters import run_context_window_check_adapter

        text = "Sample text for testing."
        base_context = {}

        # Test GPT-4 (8192 limit)
        result_gpt4 = await run_context_window_check_adapter(
            {"text": text, "model": "gpt-4"},
            base_context,
        )
        assert result_gpt4["context_limit"] == 8192

        # Test GPT-4 Turbo (128000 limit)
        result_gpt4_turbo = await run_context_window_check_adapter(
            {"text": text, "model": "gpt-4-turbo"},
            base_context,
        )
        assert result_gpt4_turbo["context_limit"] == 128000

        # Test Claude 3 (200000 limit)
        result_claude = await run_context_window_check_adapter(
            {"text": text, "model": "claude-3-opus"},
            base_context,
        )
        assert result_claude["context_limit"] == 200000

    @pytest.mark.asyncio
    async def test_context_window_check_unknown_model(self):
        """Test context window check with unknown model defaults to 8192."""
        from tldw_Server_API.app.core.Workflows.adapters import run_context_window_check_adapter

        config = {
            "text": "Test text",
            "model": "unknown-model-xyz",
        }
        context = {}

        result = await run_context_window_check_adapter(config, context)

        assert result["context_limit"] == 8192  # default

    @pytest.mark.asyncio
    async def test_context_window_check_custom_reserve_tokens(self):
        """Test context window check with custom reserve tokens."""
        from tldw_Server_API.app.core.Workflows.adapters import run_context_window_check_adapter

        config = {
            "text": "Test text",
            "model": "gpt-4",
            "reserve_tokens": 5000,
        }
        context = {}

        result = await run_context_window_check_adapter(config, context)

        assert result["available_tokens"] == 8192 - 5000

    @pytest.mark.asyncio
    async def test_context_window_check_empty_text(self):
        """Test context window check with empty text."""
        from tldw_Server_API.app.core.Workflows.adapters import run_context_window_check_adapter

        config = {"text": "", "model": "gpt-4"}
        context = {}

        result = await run_context_window_check_adapter(config, context)

        assert result["fits"] is True
        assert result["token_count"] == 0

    @pytest.mark.asyncio
    async def test_context_window_check_from_previous_step(self):
        """Test context window check uses text from previous step."""
        from tldw_Server_API.app.core.Workflows.adapters import run_context_window_check_adapter

        config = {"model": "gpt-4"}
        context = {
            "last": {
                "text": "Text from the previous workflow step."
            }
        }

        result = await run_context_window_check_adapter(config, context)

        assert result["fits"] is True
        assert result["token_count"] > 0

    @pytest.mark.asyncio
    async def test_context_window_check_cancelled(self):
        """Test context window check returns cancelled status when cancelled."""
        from tldw_Server_API.app.core.Workflows.adapters import run_context_window_check_adapter

        config = {"text": "Some text", "model": "gpt-4"}
        context = {"is_cancelled": lambda: True}

        result = await run_context_window_check_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_context_window_check_with_template(self):
        """Test context window check with templated text."""
        from tldw_Server_API.app.core.Workflows.adapters import run_context_window_check_adapter

        config = {
            "text": "Hello {{ inputs.name }}, how are you?",
            "model": "gpt-4",
        }
        context = {
            "inputs": {"name": "World"}
        }

        result = await run_context_window_check_adapter(config, context)

        assert result["fits"] is True
        # Template should be resolved before counting tokens

    @pytest.mark.asyncio
    async def test_context_window_check_default_model(self):
        """Test context window check uses gpt-4 as default model."""
        from tldw_Server_API.app.core.Workflows.adapters import run_context_window_check_adapter

        config = {"text": "Test text"}  # No model specified
        context = {}

        result = await run_context_window_check_adapter(config, context)

        assert result["context_limit"] == 8192  # GPT-4 default

    @pytest.mark.asyncio
    async def test_context_window_check_uses_content_fallback(self):
        """Test context window check uses content field as fallback."""
        from tldw_Server_API.app.core.Workflows.adapters import run_context_window_check_adapter

        config = {"model": "gpt-4"}
        context = {
            "prev": {
                "content": "Content from content field."
            }
        }

        result = await run_context_window_check_adapter(config, context)

        assert result["fits"] is True
        assert result["token_count"] > 0


# ============================================================================
# Integration-style tests
# ============================================================================

class TestEvaluationAdaptersIntegration:
    """Integration-style tests for evaluation adapters working together."""

    @pytest.mark.asyncio
    async def test_quiz_then_readability_pipeline(self):
        """Test quiz evaluation followed by readability evaluation."""
        from tldw_Server_API.app.core.Workflows.adapters import (
            run_quiz_evaluate_adapter,
            run_eval_readability_adapter,
        )

        # First, evaluate a quiz
        quiz_config = {
            "questions": [
                {"id": 0, "correct_answer": "A", "points": 2},
                {"id": 1, "correct_answer": "B", "points": 1},
            ],
            "answers": [
                {"question_id": 0, "user_answer": "A"},
                {"question_id": 1, "user_answer": "B"},
            ],
        }
        quiz_result = await run_quiz_evaluate_adapter(quiz_config, {})
        assert quiz_result["passed"] is True

        # Then, evaluate readability of quiz feedback
        feedback_text = f"You scored {quiz_result['score']}% on the quiz. Great job!"
        readability_config = {"text": feedback_text}
        readability_result = await run_eval_readability_adapter(readability_config, {})

        assert "scores" in readability_result
        assert readability_result["word_count"] > 0

    @pytest.mark.asyncio
    async def test_context_check_before_evaluation(self, monkeypatch):
        """Test context window check before running evaluation."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters import (
            run_context_window_check_adapter,
            run_evaluations_adapter,
        )

        large_text = "This is a sample response. " * 100

        # First, check if text fits in context window
        context_config = {"text": large_text, "model": "gpt-4"}
        context_result = await run_context_window_check_adapter(context_config, {})

        # If fits, run evaluation
        if context_result["fits"]:
            eval_config = {
                "action": "geval",
                "response": large_text,
            }
            eval_result = await run_evaluations_adapter(eval_config, {"user_id": "1"})
            assert "score" in eval_result or eval_result.get("simulated") is True

    @pytest.mark.asyncio
    async def test_all_adapters_handle_empty_context(self):
        """Test all evaluation adapters handle empty context gracefully."""
        from tldw_Server_API.app.core.Workflows.adapters import (
            run_quiz_evaluate_adapter,
            run_eval_readability_adapter,
            run_context_window_check_adapter,
        )

        empty_context: Dict[str, Any] = {}

        # Quiz with questions but empty context
        quiz_result = await run_quiz_evaluate_adapter(
            {"questions": [{"id": 0, "correct_answer": "A"}], "answers": []},
            empty_context,
        )
        assert "score" in quiz_result

        # Readability with text but empty context
        readability_result = await run_eval_readability_adapter(
            {"text": "Sample text."},
            empty_context,
        )
        assert "scores" in readability_result

        # Context check with text but empty context
        context_result = await run_context_window_check_adapter(
            {"text": "Sample text.", "model": "gpt-4"},
            empty_context,
        )
        assert "fits" in context_result


# ============================================================================
# Edge case tests
# ============================================================================

class TestEvaluationAdaptersEdgeCases:
    """Edge case tests for evaluation adapters."""

    @pytest.mark.asyncio
    async def test_quiz_with_non_dict_questions(self):
        """Test quiz evaluation skips non-dict questions gracefully."""
        from tldw_Server_API.app.core.Workflows.adapters import run_quiz_evaluate_adapter

        config = {
            "questions": [
                {"id": 0, "correct_answer": "A"},
                "invalid_question",  # not a dict
                {"id": 2, "correct_answer": "C"},
            ],
            "answers": [
                {"question_id": 0, "user_answer": "A"},
                {"question_id": 2, "user_answer": "C"},
            ],
        }

        result = await run_quiz_evaluate_adapter(config, {})

        # Should only count valid dict questions
        assert result["points_possible"] == 2
        assert result["score"] == 100.0

    @pytest.mark.asyncio
    async def test_readability_with_very_short_words(self):
        """Test readability handles short words (3 chars or less)."""
        from tldw_Server_API.app.core.Workflows.adapters import run_eval_readability_adapter

        config = {"text": "I am a cat. We go to eat."}
        result = await run_eval_readability_adapter(config, {})

        # Word count: I(1) am(2) a(3) cat(4) We(5) go(6) to(7) eat(8) = 8 words
        # All short words should count as 1 syllable
        assert result["word_count"] == 8
        assert "scores" in result

    @pytest.mark.asyncio
    async def test_readability_with_words_ending_in_e(self):
        """Test syllable counting for words ending in silent e."""
        from tldw_Server_API.app.core.Workflows.adapters import run_eval_readability_adapter

        config = {"text": "He made the cake. She gave the plate."}
        result = await run_eval_readability_adapter(config, {})

        # Words like "made", "cake", "gave", "plate" should handle silent e
        assert "scores" in result
        assert result["sentence_count"] == 2

    @pytest.mark.asyncio
    async def test_context_window_without_tiktoken(self, monkeypatch):
        """Test context window check falls back when tiktoken is unavailable."""
        from tldw_Server_API.app.core.Workflows.adapters import run_context_window_check_adapter

        # Mock tiktoken import to fail
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("No module named 'tiktoken'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        config = {"text": "Test text here", "model": "gpt-4"}

        # This test checks the fallback calculation (len/4)
        # The adapter should not raise an exception
        result = await run_context_window_check_adapter(config, {})

        assert "token_count" in result
        # Fallback uses len(text) / 4
        # "Test text here" = 14 chars / 4 = 3 tokens (rounded)

    @pytest.mark.asyncio
    async def test_quiz_with_missing_answer(self):
        """Test quiz evaluation with missing user answer."""
        from tldw_Server_API.app.core.Workflows.adapters import run_quiz_evaluate_adapter

        config = {
            "questions": [
                {"id": 0, "correct_answer": "A"},
                {"id": 1, "correct_answer": "B"},
            ],
            "answers": [
                {"question_id": 0, "user_answer": "A"},
                # No answer for question 1
            ],
        }

        result = await run_quiz_evaluate_adapter(config, {})

        # Should mark unanswered as incorrect
        assert result["points_earned"] == 1
        assert result["points_possible"] == 2
        assert result["score"] == 50.0

    @pytest.mark.asyncio
    async def test_evaluations_adapter_with_metrics_alias(self, monkeypatch):
        """Test evaluations adapter accepts 'metrics' as alias for 'criteria'."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters import run_evaluations_adapter

        config = {
            "action": "geval",
            "response": "Test response",
            "metrics": ["accuracy", "fluency"],  # using 'metrics' instead of 'criteria'
        }

        result = await run_evaluations_adapter(config, {"user_id": "1"})

        assert "metrics" in result
        assert "accuracy" in result["metrics"]
        assert "fluency" in result["metrics"]

    @pytest.mark.asyncio
    async def test_evaluations_adapter_with_api_provider_alias(self, monkeypatch):
        """Test evaluations adapter accepts 'api_provider' as alias for 'api_name'."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters import run_evaluations_adapter

        config = {
            "action": "geval",
            "response": "Test response",
            "api_provider": "anthropic",  # using api_provider instead of api_name
        }

        result = await run_evaluations_adapter(config, {"user_id": "1"})

        assert result.get("simulated") is True
