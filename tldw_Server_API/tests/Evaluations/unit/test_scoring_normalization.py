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
