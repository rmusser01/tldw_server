"""Pure normalization and deterministic ranking for Notes semantic matches."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class SemanticChunkCandidate:
    """One untrusted chunk-level nearest-neighbor result."""

    target_note_id: str
    source_chunk_id: str
    target_chunk_id: str
    cosine_distance: object


@dataclass(frozen=True, slots=True)
class SemanticChunkMatch:
    """One validated current source/target chunk match."""

    source_chunk_id: str
    target_chunk_id: str
    similarity: float


@dataclass(frozen=True, slots=True)
class SemanticNoteMatch:
    """A target Note ranked by its strongest current chunk match."""

    target_note_id: str
    similarity: float
    chunk_matches: tuple[SemanticChunkMatch, ...]


SimilarityBand = Literal["low", "moderate", "high", "very_high"]


def similarity_from_cosine_distance(distance: object) -> float | None:
    """Return finite clamped ``1 - distance`` or reject malformed input."""

    if isinstance(distance, bool) or not isinstance(distance, (int, float)):
        return None
    normalized = float(distance)
    if not math.isfinite(normalized):
        return None
    return min(1.0, max(0.0, 1.0 - normalized))


def rank_semantic_note_matches(
    candidates: list[SemanticChunkCandidate] | tuple[SemanticChunkCandidate, ...],
    *,
    threshold: float,
    top_k: int,
    current_chunk_ids: set[str] | frozenset[str],
) -> tuple[SemanticNoteMatch, ...]:
    """Group current chunk matches by Note and rank by strongest similarity."""

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ValueError("semantic threshold must be finite and between zero and one")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("semantic top_k must be a positive integer")

    grouped: dict[str, list[SemanticChunkMatch]] = {}
    for candidate in candidates:
        if (
            not isinstance(candidate.target_note_id, str)
            or not candidate.target_note_id
            or not isinstance(candidate.source_chunk_id, str)
            or not candidate.source_chunk_id
            or not isinstance(candidate.target_chunk_id, str)
            or not candidate.target_chunk_id
            or candidate.source_chunk_id not in current_chunk_ids
            or candidate.target_chunk_id not in current_chunk_ids
        ):
            continue
        similarity = similarity_from_cosine_distance(candidate.cosine_distance)
        if similarity is None:
            continue
        grouped.setdefault(candidate.target_note_id, []).append(
            SemanticChunkMatch(
                source_chunk_id=candidate.source_chunk_id,
                target_chunk_id=candidate.target_chunk_id,
                similarity=similarity,
            )
        )

    ranked: list[SemanticNoteMatch] = []
    for target_note_id, matches in grouped.items():
        ordered = sorted(
            matches,
            key=lambda item: (
                -item.similarity,
                item.source_chunk_id,
                item.target_chunk_id,
            ),
        )
        strongest = ordered[0].similarity
        if strongest < float(threshold):
            continue
        ranked.append(
            SemanticNoteMatch(
                target_note_id=target_note_id,
                similarity=strongest,
                chunk_matches=tuple(ordered[:3]),
            )
        )

    ranked.sort(key=lambda item: (-item.similarity, item.target_note_id))
    return tuple(ranked[:top_k])


def qualitative_similarity_band(similarity: float) -> SimilarityBand:
    """Map one finite similarity to the stable public qualitative band."""

    if isinstance(similarity, bool) or not isinstance(similarity, (int, float)) or not math.isfinite(float(similarity)):
        raise ValueError("semantic similarity must be finite")
    value = min(1.0, max(0.0, float(similarity)))
    if value >= 0.9:
        return "very_high"
    if value >= 0.75:
        return "high"
    if value >= 0.5:
        return "moderate"
    return "low"


__all__ = [
    "SemanticChunkCandidate",
    "SemanticChunkMatch",
    "SemanticNoteMatch",
    "SimilarityBand",
    "qualitative_similarity_band",
    "rank_semantic_note_matches",
    "similarity_from_cosine_distance",
]
