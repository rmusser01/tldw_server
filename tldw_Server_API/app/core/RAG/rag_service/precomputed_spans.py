"""
Precomputed span index helper for multi-vector passage scoring.

This module defines a small compatibility layer so the unified RAG pipeline
can prefer precomputed span embeddings when available, while cleanly falling
back to on-the-fly span computation when not.

The initial implementation is intentionally conservative: it exposes the
configuration and a helper that currently returns no overrides. Future
versions can integrate with `vector_stores.*` adapters or dedicated span
collections without changing the pipeline call site.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from .types import Document


@dataclass
class PrecomputedSpanConfig:
    """
    Configuration for precomputed span usage.

    Attributes:
        collection_suffix: Optional logical suffix for span collections.
        max_spans_per_doc: Safety cap on spans to consider per document.
    """

    collection_suffix: str = "spans"
    max_spans_per_doc: int = 32


async def apply_precomputed_spans(
    query: str,
    documents: list[Document],
    config: PrecomputedSpanConfig | None = None,
    user_id: str | None = None,
) -> list[Document] | None:
    """
    Placeholder precomputed-span handler.

    When a span index is wired up, this function can:
    - Look up precomputed span vectors for the provided documents
    - Reorder or slice documents based on best-span similarity
    - Attach span metadata (offsets, scores) to `Document.metadata`

    Current behaviour:
    - Logs that no precomputed span index is consulted.
    - Returns None, signalling the caller to use on-the-fly spans.
    """
    _ = (query, documents, config, user_id)  # preserve signature; avoid unused warnings
    try:
        logger.debug("Precomputed span index not configured; using on-the-fly spans")  # pragma: no cover - debug log
    except Exception:
        # Logging is best-effort only.
        logger.debug("Precomputed span fallback logging failed")
    return None
