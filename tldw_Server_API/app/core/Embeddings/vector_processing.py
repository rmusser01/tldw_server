"""Validation and request-specific processing for embedding vectors."""

from __future__ import annotations

from collections.abc import Callable

from tldw_Server_API.app.core.Embeddings.embedding_policy import adjust_dimensions
from tldw_Server_API.app.core.Embeddings.request_types import EmbeddingProviderError
from tldw_Server_API.app.core.Embeddings.vector_validation import (
    validated_embedding_vectors,
)

DimensionAdjustmentRecorder = Callable[[str, str, str], None]


class EmbeddingVectorProcessor:
    """Validate, canonicalize, and dimension-adjust embedding vectors."""

    def __init__(
        self,
        *,
        record_dimension_adjustment: DimensionAdjustmentRecorder | None = None,
    ) -> None:
        self._record_dimension_adjustment = record_dimension_adjustment

    def validate_vector_count(
        self,
        vectors: object,
        *,
        expected: int,
        provider: str,
        model: str,
    ) -> list[list[float]]:
        """Return canonical vectors or raise the existing provider domain error."""
        validated = validated_embedding_vectors(vectors, expected=expected)
        if validated is not None:
            return validated

        count: int | str = len(vectors) if isinstance(vectors, list) else "invalid"
        message = (
            f"Embedding provider returned {count} embeddings, expected {expected}"
            if count != expected
            else "Embedding provider returned malformed embedding vectors"
        )
        raise EmbeddingProviderError(
            "provider_malformed_response",
            message,
            provider=provider,
            model=model,
        )

    def validate_cached_vector(self, vector: object) -> list[float] | None:
        """Return one canonical cache vector, or ``None`` for a cache miss."""
        validated = validated_embedding_vectors([vector], expected=1)
        return validated[0] if validated is not None else None

    def process_vectors(
        self,
        vectors: list[list[float]],
        *,
        provider: str,
        model: str,
        dimensions: int | None,
        dimension_policy: str,
    ) -> list[list[float]]:
        """Apply request-specific dimension processing to canonical vectors."""
        canonical = _canonical_vectors(vectors)
        if dimensions is None:
            return canonical
        adjusted = adjust_dimensions(
            canonical,
            dimensions,
            provider,
            model,
            dimension_policy=dimension_policy,
            record_adjustment=self._record_dimension_adjustment,
        )
        return _canonical_vectors(adjusted)

    def process_cached_vector(
        self,
        canonical_vector: list[float],
        *,
        provider: str,
        model: str,
        dimensions: int | None,
        dimension_policy: str,
    ) -> list[float]:
        """Postprocess a vector already accepted by ``validate_cached_vector``."""
        return self.process_vectors(
            [canonical_vector],
            provider=provider,
            model=model,
            dimensions=dimensions,
            dimension_policy=dimension_policy,
        )[0]


def _canonical_vectors(vectors: list[list[float]]) -> list[list[float]]:
    return [[float(value) for value in vector] for vector in vectors]


__all__ = ["DimensionAdjustmentRecorder", "EmbeddingVectorProcessor"]
