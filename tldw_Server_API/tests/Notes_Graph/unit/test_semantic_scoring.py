"""Pure score normalization and deterministic semantic Note ranking."""

from __future__ import annotations

import math

import pytest

from tldw_Server_API.app.core.Notes_Graph.semantic_scoring import (
    SemanticChunkCandidate,
    qualitative_similarity_band,
    rank_semantic_note_matches,
    similarity_from_cosine_distance,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (0.0, 1.0),
        (0.25, 0.75),
        (1.0, 0.0),
        (-0.2, 1.0),
        (1.4, 0.0),
    ],
)
def test_cosine_distance_is_normalized_and_clamped(
    distance: float,
    expected: float,
) -> None:
    assert similarity_from_cosine_distance(distance) == expected


@pytest.mark.parametrize(
    "distance",
    [None, True, "0.2", math.nan, math.inf, -math.inf],
)
def test_malformed_or_non_finite_distance_is_rejected(distance: object) -> None:
    assert similarity_from_cosine_distance(distance) is None


def _candidate(
    target_note_id: str,
    distance: object,
    source_chunk_id: str,
    target_chunk_id: str,
) -> SemanticChunkCandidate:
    return SemanticChunkCandidate(
        target_note_id=target_note_id,
        source_chunk_id=source_chunk_id,
        target_chunk_id=target_chunk_id,
        cosine_distance=distance,
    )


def test_note_score_uses_the_strongest_chunk_and_keeps_three_stable_pairs() -> None:
    ranked = rank_semantic_note_matches(
        [
            _candidate("note-b", 0.2, "source-3", "target-3"),
            _candidate("note-b", 0.1, "source-2", "target-2"),
            _candidate("note-b", 0.1, "source-1", "target-4"),
            _candidate("note-b", 0.1, "source-1", "target-1"),
            _candidate("note-b", 0.05, "source-stale", "target-stale"),
        ],
        threshold=0.0,
        top_k=10,
        current_chunk_ids={
            "source-1",
            "source-2",
            "source-3",
            "target-1",
            "target-2",
            "target-3",
            "target-4",
        },
    )

    assert len(ranked) == 1
    assert ranked[0].similarity == 0.9
    assert [(pair.source_chunk_id, pair.target_chunk_id) for pair in ranked[0].chunk_matches] == [
        ("source-1", "target-1"),
        ("source-1", "target-4"),
        ("source-2", "target-2"),
    ]


def test_threshold_top_k_and_note_id_ties_are_stable() -> None:
    ranked = rank_semantic_note_matches(
        [
            _candidate("note-c", 0.1, "source", "chunk-c"),
            _candidate("note-a", 0.1, "source", "chunk-a"),
            _candidate("note-b", 0.3, "source", "chunk-b"),
            _candidate("note-d", 0.31, "source", "chunk-d"),
            _candidate("note-malformed", math.nan, "source", "chunk-x"),
        ],
        threshold=0.7,
        top_k=3,
        current_chunk_ids={
            "source",
            "chunk-a",
            "chunk-b",
            "chunk-c",
            "chunk-d",
            "chunk-x",
        },
    )

    assert [(item.target_note_id, item.similarity) for item in ranked] == [
        ("note-a", 0.9),
        ("note-c", 0.9),
        ("note-b", 0.7),
    ]


@pytest.mark.parametrize(
    ("similarity", "expected"),
    [
        (0.49, "low"),
        (0.5, "moderate"),
        (0.749, "moderate"),
        (0.75, "high"),
        (0.899, "high"),
        (0.9, "very_high"),
    ],
)
def test_qualitative_bands_are_deterministic(
    similarity: float,
    expected: str,
) -> None:
    assert qualitative_similarity_band(similarity) == expected
