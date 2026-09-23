"""The one Likert normalizer, and the divergence it replaces."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Evaluations.scoring import (
    normalize_likert,
    parse_judge_score,
)


@pytest.mark.parametrize("raw,expected", [(1, 0.0), (2, 0.25), (3, 0.5), (4, 0.75), (5, 1.0)])
def test_default_1_to_5_scale(raw: float, expected: float) -> None:
    assert normalize_likert(raw) == pytest.approx(expected)


def test_the_bottom_of_the_range_is_zero_not_a_floor() -> None:
    """raw/5.0 put a 20% floor under every metric; the affine mapping does not."""
    assert normalize_likert(1) == 0.0
    assert normalize_likert(1) != pytest.approx(1 / 5.0)


@pytest.mark.parametrize("raw", [0, -3, 6, 99])
def test_out_of_range_is_clamped_not_rejected(raw: float) -> None:
    assert 0.0 <= normalize_likert(raw) <= 1.0


def test_other_scales_pass_their_own_bounds() -> None:
    assert normalize_likert(2, scale_max=3) == pytest.approx(0.5)   # 1-3 fluency
    assert normalize_likert(1, scale_max=10) == 0.0                 # 1-10
    assert normalize_likert(10, scale_max=10) == 1.0


def test_degenerate_scale_is_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_likert(3, scale_min=5, scale_max=5)


@pytest.mark.parametrize("text,expected", [("4", 0.75), (" 5 ", 1.0), ("1", 0.0), (3, 0.5)])
def test_parse_judge_score(text, expected: float) -> None:
    assert parse_judge_score(text) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [None, "", "not a number", "  "])
def test_unparseable_returns_the_default(bad) -> None:
    assert parse_judge_score(bad) == 0.0
    assert parse_judge_score(bad, default=0.5) == 0.5


def test_matches_the_previously_dead_canonical() -> None:
    """RAGEvaluator._normalize_score always implemented this, with tests, unused."""
    from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator

    ev = RAGEvaluator.__new__(RAGEvaluator)
    for raw in (0, 1, 2, 3, 4, 5, 6):
        assert ev._normalize_score(raw) == pytest.approx(normalize_likert(raw))


# --- Sites that re-derived the formula and got the bottom of the range wrong -------------
# G-Eval (ms_g_eval.run_geval) returns raw scores: 1-5, fluency 1-3, 0 on parse failure.


@pytest.mark.asyncio
async def test_eval_runner_summarization_floor_and_branch_agreement(monkeypatch, tmp_path) -> None:
    """Raw 1 scored 0.2 (a floor), and dict 0.8 vs string "fluency: 0.8" disagreed."""
    from tldw_Server_API.app.core.Evaluations import eval_runner as mod

    runner = mod.EvaluationRunner(db_path=str(tmp_path / "evals.db"))
    spec = {"metrics": ["coherence", "fluency"], "evaluator_model": "openai"}
    sample = {"input": {"source_text": "s", "summary": "t"}}

    monkeypatch.setattr(mod, "run_geval", lambda **_k: {"metrics": {"coherence": 1, "fluency": 0.8}})
    from_dict = (await runner._eval_summarization(sample, spec, {}, "s1"))["scores"]
    monkeypatch.setattr(mod, "run_geval", lambda **_k: "coherence: 1\nfluency: 0.8")
    from_text = (await runner._eval_summarization(sample, spec, {}, "s1"))["scores"]

    assert from_dict["coherence"] == 0.0
    assert from_dict == pytest.approx(from_text)


def test_summarization_quality_worst_geval_score_is_not_perfect() -> None:
    """_normalize_score returned any value <= 1.0 unchanged, so raw 1 (worst) became 1.0."""
    from tldw_Server_API.app.core.Evaluations.recipes.summarization_quality import (
        SummarizationQualityRecipe,
    )

    worst = {"consistency": 1, "relevance": 1, "coherence": 1, "fluency": 1}
    assert SummarizationQualityRecipe()._coerce_metrics(worst) == {
        "grounding": 0.0,
        "coverage": 0.0,
        "usefulness": 0.0,
    }


def test_rag_answer_quality_reference_score_uses_the_geval_scales() -> None:
    """_coerce_unit_score mapped raw 1 to 1.0 and put fluency (1-3) on a 1-5 scale."""
    from tldw_Server_API.app.core.Evaluations.recipes import rag_answer_quality_execution as mod

    rubric = {"grounding": 0.0, "answer_relevance": 0.0}

    def reference(metrics):
        # answer_relevance = 0.5 * rubric (0) + 0.5 * reference score
        return mod._blend_with_reference_artifact(rubric, {"metrics": metrics})["answer_relevance"] * 2

    assert reference({"consistency": 1, "relevance": 1, "coherence": 1, "fluency": 1}) == 0.0
    assert reference({"consistency": 5, "relevance": 5, "coherence": 5, "fluency": 3}) == pytest.approx(1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply,expected",
    [("Score: 1", 0.0), ('{"score": 1}', 0.0), ("Score: 10", 1.0), ('{"score": 0.8}', 0.8), ("Score: 0.8", 0.8)],
)
async def test_evaluation_manager_custom_metric_uses_its_declared_1_to_10_scale(reply, expected) -> None:
    """The prompt asks for 1-10. Regex "Score: 1" scored 1.0 (worst became perfect) and JSON
    {"score": 0.8} scored 0.08 while the same value as text scored 0.8."""
    from unittest.mock import patch

    from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager

    with patch("tldw_Server_API.app.core.Evaluations.evaluation_manager.analyze", return_value=reply):
        result = await EvaluationManager().evaluate_custom_metric(
            metric_name="m",
            description="d",
            evaluation_prompt="p",
            input_data={"x": "y"},
            scoring_criteria={"c": "c"},
        )
    assert result["score"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "metric,raw,expected",
    [("coherence", 1, 0.0), ("coherence", 3, 0.5), ("fluency", 1, 0.0), ("fluency", 3, 1.0),
     ("fluency", 2, 0.5), ("coherence", 0, 0.0), ("coherence", 0.8, 0.8), ("coherence", 9, 1.0)],
)
def test_normalize_geval_metric(metric: str, raw: float, expected: float) -> None:
    """Below 1 cannot be a raw score on a 1-N scale: 0 is run_geval's failure sentinel and
    fractions are already-normalized values, so both pass through. Exactly 1 is on-scale."""
    from tldw_Server_API.app.core.Evaluations.scoring import normalize_geval_metric

    assert normalize_geval_metric(metric, raw) == pytest.approx(expected)
