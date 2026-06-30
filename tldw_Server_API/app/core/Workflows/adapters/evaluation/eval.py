"""Evaluation adapters for workflow steps.

This module provides adapters for evaluating LLM outputs, quiz answers,
text readability, and context window sizes.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Workflows.adapters._common import (
    resolve_context_user_id,
)
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.evaluation._config import (
    ContextWindowCheckConfig,
    EvalReadabilityConfig,
    EvaluationsConfig,
    QuizEvaluateConfig,
)


@registry.register(
    "evaluations",
    category="evaluation",
    description="Run LLM evaluations (geval, rag, response_quality)",
    parallelizable=True,
    tags=["evaluation", "testing"],
    config_model=EvaluationsConfig,
)
async def run_evaluations_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Run LLM evaluations (geval, rag, response_quality) within a workflow step.

    Config:
      - action: Literal["geval", "rag", "response_quality", "get_run", "list_runs"]
      - response: Optional[str] (templated, defaults to last.text) - for geval/rag/response_quality
      - context: Optional[str] (templated) - source text for geval, context for response_quality
      - criteria: Optional[List[str]] - e.g., ["relevance", "coherence", "fluency"]
      - question: Optional[str] (templated) - for rag
      - retrieved_contexts: Optional[List[str]] - from last.documents or explicit
      - reference_answer: Optional[str] (templated) - for rag
      - run_id: Optional[str] - for get_run
      - api_name: Optional[str] - LLM provider for evaluation
      - limit: int = 20 (for list_runs)
    Output:
      - {"score": float, "metrics": {...}, "passed": bool, "details": {...}}
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    user_id = resolve_context_user_id(context)
    if not user_id:
        try:
            user_id = str(DatabasePaths.get_single_user_id())
        except Exception:
            return {"error": "missing_user_id"}
    user_id = str(user_id)

    action = str(config.get("action") or "").strip().lower()
    if not action:
        return {"error": "missing_action"}

    def _render(value: Any) -> Any:
        if isinstance(value, str):
            return apply_template_to_string(value, context) or value
        return value

    # Test mode simulation
    if is_test_mode():
        if action == "geval":
            criteria = config.get("criteria") or config.get("metrics")
            if isinstance(criteria, str):
                criteria = [c.strip() for c in criteria.split(",") if c.strip()]
            elif not isinstance(criteria, list):
                criteria = ["coherence", "relevance", "fluency"]
            simulated_metrics = {c: 0.8 + (hash(c) % 15) / 100 for c in criteria}
            avg_score = sum(simulated_metrics.values()) / len(simulated_metrics) if simulated_metrics else 0.85
            return {
                "evaluation_id": "test-eval-geval",
                "score": round(avg_score, 2),
                "metrics": simulated_metrics,
                "passed": avg_score >= float(config.get("threshold", 0.6)),
                "evaluation_time": 0.5,
                "simulated": True,
            }
        if action == "rag":
            return {
                "evaluation_id": "test-eval-rag",
                "score": 0.82,
                "metrics": {"faithfulness": 0.85, "answer_relevance": 0.80, "context_relevance": 0.81},
                "passed": True,
                "evaluation_time": 0.6,
                "simulated": True,
            }
        if action == "response_quality":
            return {
                "evaluation_id": "test-eval-quality",
                "score": 0.88,
                "metrics": {"clarity": 0.9, "completeness": 0.85, "accuracy": 0.89},
                "passed": True,
                "evaluation_time": 0.4,
                "simulated": True,
            }
        if action == "get_run":
            return {
                "run": {"id": config.get("run_id"), "status": "completed", "score": 0.85},
                "simulated": True,
            }
        if action == "list_runs":
            return {"runs": [], "has_more": False, "simulated": True}
        return {"error": f"unknown_action:{action}", "simulated": True}

    try:
        from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import UnifiedEvaluationService

        service = UnifiedEvaluationService(enable_webhooks=False)
        await service.initialize()

        api_name = _render(config.get("api_name") or config.get("api_provider") or "openai")

        if action == "geval":
            response = config.get("response") or config.get("summary")
            if response is not None:
                response = _render(response)
            else:
                last = context.get("last") or {}
                if isinstance(last, dict):
                    response = last.get("text") or last.get("content") or last.get("summary") or ""
                else:
                    response = ""

            source_text = config.get("context") or config.get("source_text")
            source_text = _render(source_text) if source_text is not None else ""

            if not response:
                return {"error": "missing_response_for_geval"}

            criteria = config.get("criteria") or config.get("metrics")
            if isinstance(criteria, str):
                criteria = [c.strip() for c in criteria.split(",") if c.strip()]
            elif not isinstance(criteria, list):
                criteria = None

            result = await service.evaluate_geval(
                source_text=source_text,
                summary=response,
                metrics=criteria,
                api_name=api_name,
                user_id=user_id,
            )

            results = result.get("results") or {}
            avg_score = results.get("average_score", 0.0)
            return {
                "evaluation_id": result.get("evaluation_id"),
                "score": avg_score,
                "metrics": results.get("scores") or results,
                "passed": avg_score >= float(config.get("threshold", 0.6)),
                "evaluation_time": result.get("evaluation_time"),
                "details": results,
            }

        if action == "rag":
            question = config.get("question") or config.get("query")
            question = _render(question) if question is not None else ""

            response = config.get("response")
            if response is not None:
                response = _render(response)
            else:
                last = context.get("last") or {}
                response = last.get("text") or last.get("content") or "" if isinstance(last, dict) else ""

            retrieved_contexts = config.get("retrieved_contexts") or config.get("contexts")
            if retrieved_contexts is None:
                last = context.get("last") or {}
                if isinstance(last, dict):
                    docs = last.get("documents") or last.get("results") or []
                    if isinstance(docs, list):
                        retrieved_contexts = []
                        for d in docs:
                            if isinstance(d, dict):
                                txt = d.get("content") or d.get("text") or d.get("snippet") or ""
                                if txt:
                                    retrieved_contexts.append(txt)
                            elif isinstance(d, str):
                                retrieved_contexts.append(d)
            if not isinstance(retrieved_contexts, list):
                retrieved_contexts = []

            reference_answer = config.get("reference_answer") or config.get("ground_truth")
            if reference_answer is not None:
                reference_answer = _render(reference_answer)

            if not question or not response:
                return {"error": "missing_question_or_response_for_rag"}

            result = await service.evaluate_rag(
                query=question,
                contexts=retrieved_contexts,
                response=response,
                ground_truth=reference_answer,
                api_name=api_name,
                user_id=user_id,
            )

            results = result.get("results") or {}
            overall_score = results.get("overall_score", 0.0)
            return {
                "evaluation_id": result.get("evaluation_id"),
                "score": overall_score,
                "metrics": results.get("metrics") or results,
                "passed": overall_score >= float(config.get("threshold", 0.6)),
                "evaluation_time": result.get("evaluation_time"),
                "details": results,
            }

        if action == "response_quality":
            prompt = config.get("prompt") or config.get("question")
            prompt = _render(prompt) if prompt is not None else ""

            response = config.get("response")
            if response is not None:
                response = _render(response)
            else:
                last = context.get("last") or {}
                response = last.get("text") or last.get("content") or "" if isinstance(last, dict) else ""

            if not response:
                return {"error": "missing_response_for_quality_eval"}

            expected_format = config.get("expected_format")
            custom_criteria = config.get("custom_criteria")

            result = await service.evaluate_response_quality(
                prompt=prompt,
                response=response,
                expected_format=expected_format,
                custom_criteria=custom_criteria,
                api_name=api_name,
                user_id=user_id,
            )

            results = result.get("results") or {}
            overall_score = results.get("overall_score", results.get("score", 0.0))
            return {
                "evaluation_id": result.get("evaluation_id"),
                "score": overall_score,
                "metrics": results.get("metrics") or results,
                "passed": overall_score >= float(config.get("threshold", 0.6)),
                "evaluation_time": result.get("evaluation_time"),
                "details": results,
            }

        if action == "get_run":
            run_id = config.get("run_id")
            if not run_id:
                return {"error": "missing_run_id"}
            run = await service.get_run(str(run_id))
            if run is None:
                return {"error": "run_not_found", "run_id": run_id}
            return {"run": run}

        if action == "list_runs":
            limit = int(config.get("limit") or 20)
            eval_id = config.get("eval_id")
            status = config.get("status")
            runs, has_more = await service.list_runs(
                eval_id=eval_id,
                status=status,
                limit=limit,
            )
            return {"runs": runs, "has_more": has_more}

        return {"error": f"unknown_action:{action}"}

    except Exception:
        logger.error("Evaluations adapter error")
        return {"error": "evaluations_error"}


@registry.register(
    "quiz_evaluate",
    category="evaluation",
    description="Evaluate quiz answers and provide feedback",
    parallelizable=True,
    tags=["evaluation", "education"],
    config_model=QuizEvaluateConfig,
)
async def run_quiz_evaluate_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Evaluate quiz answers and provide feedback."""
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    questions = config.get("questions")
    if not questions:
        prev = context.get("prev") or context.get("last") or {}
        questions = prev.get("questions") if isinstance(prev, dict) else []

    answers = config.get("answers") or []
    if not isinstance(questions, list) or not questions:
        return {"error": "missing_questions", "score": 0, "results": []}

    passing_score = float(config.get("passing_score", 70))
    answer_map = {
        (a.get("question_id", i) if isinstance(a, dict) else i): (a.get("user_answer") if isinstance(a, dict) else a)
        for i, a in enumerate(answers)
    }

    results, points_earned, points_possible = [], 0, 0
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            continue
        qid = q.get("id", i)
        correct = q.get("correct_answer")
        user_ans = answer_map.get(q.get("id", i))
        pts = q.get("points", 1)
        points_possible += pts
        qtype = q.get("question_type", "multiple_choice")
        is_correct = (
            (correct == user_ans)
            if qtype == "multiple_choice"
            else (str(correct).lower().strip() == str(user_ans or "").lower().strip())
        )
        if is_correct:
            points_earned += pts
        results.append({"question_id": qid, "is_correct": is_correct, "points": pts if is_correct else 0})

    score = (points_earned / points_possible * 100) if points_possible > 0 else 0
    return {
        "score": round(score, 2),
        "points_earned": points_earned,
        "points_possible": points_possible,
        "results": results,
        "passed": score >= passing_score,
    }


@registry.register(
    "eval_readability",
    category="evaluation",
    description="Calculate readability scores for text",
    parallelizable=True,
    tags=["evaluation", "readability"],
    config_model=EvalReadabilityConfig,
)
async def run_eval_readability_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Calculate readability scores for text."""
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    text = config.get("text") or ""
    if isinstance(text, str):
        text = apply_template_to_string(text, context) or text
    text = str(text).strip()
    if not text:
        prev = context.get("prev") or context.get("last") or {}
        text = str(prev.get("text") or prev.get("content") or "") if isinstance(prev, dict) else ""

    if not text:
        return {"error": "missing_text", "scores": {}}

    words = text.split()
    word_count = len(words)
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = len(sentences) or 1

    def count_syllables(word: str) -> int:
        word = word.lower()
        if len(word) <= 3:
            return 1
        if word.endswith('e'):
            word = word[:-1]
        count, prev_vowel = 0, False
        for c in word:
            is_v = c in "aeiouy"
            if is_v and not prev_vowel:
                count += 1
            prev_vowel = is_v
        return max(1, count)

    syllable_count = sum(count_syllables(w) for w in words)
    scores: dict[str, float] = {}

    if word_count > 0:
        scores["flesch_reading_ease"] = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
        scores["flesch_kincaid_grade"] = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59

    return {
        "scores": {k: round(v, 2) for k, v in scores.items()},
        "grade_level": round(scores.get("flesch_kincaid_grade", 0), 1),
        "reading_ease": round(scores.get("flesch_reading_ease", 50), 1),
        "word_count": word_count,
        "sentence_count": sentence_count,
    }


@registry.register(
    "context_window_check",
    category="evaluation",
    description="Check if content fits in model context window",
    parallelizable=True,
    tags=["evaluation", "utility"],
    config_model=ContextWindowCheckConfig,
)
async def run_context_window_check_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Check if content fits in model context window.

    Config:
      - text: str - Text to check
      - model: str - Model name (default: "gpt-4")
      - reserve_tokens: int - Tokens to reserve for response (default: 1000)
    Output:
      - fits: bool
      - token_count: int
      - context_limit: int
      - available_tokens: int
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    text = config.get("text") or ""
    if isinstance(text, str):
        text = apply_template_to_string(text, context) or text

    if not text:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            text = prev.get("text") or prev.get("content") or ""

    model = config.get("model", "gpt-4")
    reserve_tokens = int(config.get("reserve_tokens", 1000))

    # Model context limits (common models)
    context_limits = {
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        "gpt-3.5-turbo": 16384,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "llama-3": 8192,
        "mistral": 32768,
    }
    context_limit = context_limits.get(model, 8192)

    # Count tokens
    token_count = 0
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(text))
    except ImportError:
        token_count = int(len(text) / 4)

    available = context_limit - reserve_tokens
    fits = token_count <= available

    return {
        "fits": fits,
        "token_count": token_count,
        "context_limit": context_limit,
        "available_tokens": available,
        "excess_tokens": max(0, token_count - available),
    }
