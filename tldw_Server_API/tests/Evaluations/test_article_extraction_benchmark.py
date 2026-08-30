"""Tests for deterministic article-extraction benchmark scoring."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Evaluations.article_extraction_benchmark import (
    ArticleExtractionBenchmarkEvaluator,
    evaluate_metrics,
)


GROUND_TRUTH = {
    "exact": {"articleBody": "one two three four five"},
    "partial": {"articleBody": "alpha beta gamma delta epsilon"},
    "empty": {"articleBody": ""},
}
PREDICTIONS = {
    "exact": {"articleBody": "one two three four five"},
    "partial": {"articleBody": "alpha beta gamma different ending"},
    "empty": {"articleBody": ""},
}


def _minimal_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "article-benchmark"
    (dataset / "html").mkdir(parents=True)
    (dataset / "ground-truth.json").write_text(
        json.dumps(GROUND_TRUTH),
        encoding="utf-8",
    )
    return dataset


def test_evaluate_metrics_is_repeatable_without_mutating_global_random_state() -> None:
    state = random.getstate()

    first = evaluate_metrics(GROUND_TRUTH, PREDICTIONS, 50, bootstrap_seed=7)
    second = evaluate_metrics(GROUND_TRUTH, PREDICTIONS, 50, bootstrap_seed=7)

    assert first == second
    assert random.getstate() == state


@pytest.mark.parametrize("count", [0, -1])
def test_evaluator_rejects_non_positive_bootstrap_count(
    tmp_path: Path,
    count: int,
) -> None:
    dataset = _minimal_dataset(tmp_path)

    with pytest.raises(ValueError, match="n_bootstrap must be a positive integer"):
        ArticleExtractionBenchmarkEvaluator(dataset, n_bootstrap=count)


@pytest.mark.parametrize("count", [0, -1])
def test_evaluate_metrics_rejects_non_positive_bootstrap_count(count: int) -> None:
    with pytest.raises(ValueError, match="n_bootstrap must be a positive integer"):
        evaluate_metrics(GROUND_TRUTH, PREDICTIONS, count)


def test_evaluator_rejects_non_integer_bootstrap_seed(tmp_path: Path) -> None:
    dataset = _minimal_dataset(tmp_path)

    with pytest.raises(ValueError, match="bootstrap_seed must be an integer"):
        ArticleExtractionBenchmarkEvaluator(dataset, bootstrap_seed=True)


def test_evaluate_metrics_rejects_non_integer_bootstrap_seed() -> None:
    with pytest.raises(ValueError, match="bootstrap_seed must be an integer"):
        evaluate_metrics(GROUND_TRUTH, PREDICTIONS, 10, bootstrap_seed=1.5)
