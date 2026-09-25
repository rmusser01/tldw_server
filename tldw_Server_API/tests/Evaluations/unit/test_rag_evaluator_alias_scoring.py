"""Regression guard for TASK-13311.

`RAGEvaluator` records each metric under its canonical key and then adds an alias
key (`answer_relevance`, `answer_faithfulness`) pointing at the *same* result dict,
so the response is OpenAI-style. `_calculate_overall_score` then defaults to
`dict.fromkeys(metrics, 1.0)`, giving the alias its own full weight — so an aliased
metric is counted twice and the overall score is inflated.

The dedup that would drop the canonical key is gated on `if not explicit_metrics`,
and `eval_runner` always passes an explicit metric list, so on the runner path the
double-count always applies. The inflation propagates into `avg_score`,
`mean_score`, and the 0.7 pass threshold.

The alias keys must stay in the response; they simply must not be scored twice.
"""

import pytest


from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator

# Suite marker: these are fast, isolated regression guards.
pytestmark = pytest.mark.unit


def _evaluator() -> RAGEvaluator:
    """Build the instance without running __init__ (scoring needs no state)."""
    return RAGEvaluator.__new__(RAGEvaluator)


def test_alias_key_does_not_inflate_overall_score() -> None:
    ev = _evaluator()

    canonical = {"relevance": {"score": 1.0}, "faithfulness": {"score": 0.0}}
    baseline = ev._calculate_overall_score(dict(canonical))
    assert baseline == pytest.approx(0.5), "control: two metrics, 1.0 and 0.0, average to 0.5"

    # Production adds the alias via setdefault, pointing at the same dict object.
    aliased = dict(canonical)
    aliased["answer_relevance"] = canonical["relevance"]

    assert ev._calculate_overall_score(aliased) == pytest.approx(baseline), (
        "alias key is being counted as an independent metric -- relevance is "
        "weighted twice, inflating the overall score"
    )


def test_both_aliases_together_do_not_inflate() -> None:
    ev = _evaluator()

    canonical = {"relevance": {"score": 1.0}, "faithfulness": {"score": 0.0}}
    aliased = dict(canonical)
    aliased["answer_relevance"] = canonical["relevance"]
    aliased["answer_faithfulness"] = canonical["faithfulness"]

    assert ev._calculate_overall_score(aliased) == pytest.approx(0.5)


def test_alias_is_scored_when_it_is_the_only_copy() -> None:
    """When the canonical key was dropped, the alias must still count."""
    ev = _evaluator()

    alias_only = {"answer_relevance": {"score": 1.0}, "faithfulness": {"score": 0.0}}
    assert ev._calculate_overall_score(alias_only) == pytest.approx(0.5)


def test_explicit_weights_are_still_honoured() -> None:
    """Deduping aliases must not disturb caller-supplied weights."""
    ev = _evaluator()

    canonical = {"relevance": {"score": 1.0}, "faithfulness": {"score": 0.0}}
    aliased = dict(canonical)
    aliased["answer_relevance"] = canonical["relevance"]

    # relevance weighted 3:1 against faithfulness -> 0.75
    weights = {"relevance": 3.0, "faithfulness": 1.0}
    assert ev._calculate_overall_score(aliased, weights) == pytest.approx(0.75)


def test_alias_only_weights_still_apply_to_the_canonical_metric() -> None:
    """Weights supplied under an alias key must survive the alias being dropped.

    `evaluate()` treats an alias request as a request for the canonical computation,
    so a caller may legitimately supply `metric_weights={"answer_relevance": ...}`.
    Dropping the alias entry without carrying its weight across leaves the retained
    canonical metric unweighted, and the aggregate collapses to 0.0.
    """
    ev = _evaluator()

    metrics = {"relevance": {"score": 1.0}, "faithfulness": {"score": 0.0}}
    metrics["answer_relevance"] = metrics["relevance"]

    # relevance weighted 3:1 against faithfulness, but keyed under the alias
    weights = {"answer_relevance": 3.0, "faithfulness": 1.0}

    score = ev._calculate_overall_score(metrics, weights)
    assert score == pytest.approx(0.75), (
        f"alias-keyed weight was discarded (got {score}); the canonical metric it "
        "refers to contributed no weight"
    )


def test_canonical_weight_wins_when_both_forms_are_supplied() -> None:
    ev = _evaluator()

    metrics = {"relevance": {"score": 1.0}, "faithfulness": {"score": 0.0}}
    metrics["answer_relevance"] = metrics["relevance"]

    # canonical says 1.0, alias says 99.0 -- canonical must win
    weights = {"relevance": 1.0, "answer_relevance": 99.0, "faithfulness": 1.0}

    assert ev._calculate_overall_score(metrics, weights) == pytest.approx(0.5)


def test_alias_only_weights_do_not_collapse_the_score_to_zero() -> None:
    """The specific regression shape: every retained metric unweighted."""
    ev = _evaluator()

    metrics = {"relevance": {"score": 0.8}}
    metrics["answer_relevance"] = metrics["relevance"]

    score = ev._calculate_overall_score(metrics, {"answer_relevance": 1.0})
    assert score == pytest.approx(0.8), f"score collapsed to {score}"


def test_crossing_the_pass_threshold() -> None:
    """The concrete consequence: a failing evaluation reported as passing."""
    ev = _evaluator()

    # Canonical mean is 0.6 -- below the 0.7 gate.
    canonical = {"relevance": {"score": 0.9}, "faithfulness": {"score": 0.3}}
    aliased = dict(canonical)
    aliased["answer_relevance"] = canonical["relevance"]

    score = ev._calculate_overall_score(aliased)
    assert score == pytest.approx(0.6), "double-counting relevance lifts 0.6 to 0.7"
    assert score < 0.7, "an evaluation that should fail is being reported as passing"


@pytest.mark.asyncio
async def test_runner_avg_score_and_pass_gate_ignore_alias_keys(tmp_path) -> None:
    """The eval_runner path always passes explicit metrics, so the alias keys survive
    into the response. avg_score (which feeds mean_score) and the pass gate must
    still count each metric once."""
    from tldw_Server_API.app.core.Evaluations.eval_runner import EvaluationRunner

    ev = _evaluator()

    async def _relevance(*_args, **_kwargs):
        return "relevance", {"score": 1.0}

    async def _faithfulness(*_args, **_kwargs):
        return "faithfulness", {"score": 0.0}

    async def _context_relevance(*_args, **_kwargs):
        return "context_relevance", {"score": 0.0}

    ev._evaluate_relevance = _relevance
    ev._evaluate_faithfulness = _faithfulness
    ev._evaluate_context_relevance = _context_relevance

    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))
    runner._rag_evaluator = ev
    sample = {"input": {"query": "q", "contexts": ["c"], "response": "r"}, "expected": "a"}
    # Canonical mean is 1/3; counting both aliases gives (1+0+0+1+0)/5 = 0.4.
    spec = {"metrics": ["relevance", "faithfulness", "context_relevance"], "threshold": 0.35}

    result = await runner._eval_rag(sample, spec, {}, "sample_0")

    assert "answer_relevance" in result["scores"], "alias keys stay in the response"
    assert result["avg_score"] == pytest.approx(1 / 3), "alias counted as its own metric"
    assert result["passed"] is False, "double-counted 0.4 crossed the 0.35 gate"
    aggregate = runner._calculate_aggregate_results([result], None, 0.35)
    assert aggregate["mean_score"] == pytest.approx(1 / 3)
    assert aggregate["pass_rate"] == 0.0
