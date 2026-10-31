"""Shared validation for embedding vectors crossing provider and cache boundaries."""

from __future__ import annotations

import math
from numbers import Real


def validated_embedding_vectors(
    vectors: object,
    *,
    expected: int,
) -> list[list[float]] | None:
    """Return finite, consistently sized vectors or ``None`` for malformed data."""
    if not isinstance(vectors, list) or len(vectors) != expected:
        return None

    normalized: list[list[float]] = []
    width: int | None = None
    for vector in vectors:
        if not isinstance(vector, list) or not vector:
            return None
        if width is None:
            width = len(vector)
        elif len(vector) != width:
            return None

        normalized_vector: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, Real):
                return None
            try:
                numeric_value = float(value)
            except (OverflowError, TypeError, ValueError):
                return None
            if not math.isfinite(numeric_value):
                return None
            normalized_vector.append(numeric_value)
        normalized.append(normalized_vector)
    return normalized


def validated_indexed_embedding_data(
    data: object,
    *,
    expected: int,
) -> list[list[float]] | None:
    """Validate indexed OpenAI-compatible rows and restore input order."""
    if not isinstance(data, list) or len(data) != expected:
        return None

    ordered: list[object | None] = [None] * expected
    for item in data:
        if not isinstance(item, dict):
            return None
        index = item.get("index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= expected
            or ordered[index] is not None
        ):
            return None
        ordered[index] = item.get("embedding")
    return validated_embedding_vectors(ordered, expected=expected)
