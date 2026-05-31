"""
Pseudo-relevance feedback (PRF) utilities for the unified RAG pipeline.

This module provides a lightweight helper that:
- Mines salient terms from the top retrieved documents (keywords, entities, numbers)
- Builds a simple expanded query string
- Returns metadata describing the expansion for observability

The unified pipeline currently uses PRF in a metadata-only fashion, so callers
can introspect suggested expansions without changing retrieval behaviour.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .types import Document


@dataclass
class PRFConfig:
    """Configuration for pseudo-relevance feedback term mining."""

    max_terms: int = 10
    sources: list[str] = field(default_factory=lambda: ["keywords", "entities", "numbers"])
    alpha: float = 0.3
    top_n: int = 8

    def __post_init__(self) -> None:
        """
        Validate configuration values.

        In particular, ensure alpha is a numeric weight in the closed interval
        [0.0, 1.0] so downstream scoring logic can rely on it.
        """
        try:
            alpha_val = float(self.alpha)
        except (TypeError, ValueError):
            raise ValueError(f"PRFConfig.alpha must be a float between 0.0 and 1.0, got {self.alpha!r}") from None
        if not 0.0 <= alpha_val <= 1.0:
            raise ValueError(f"PRFConfig.alpha must be between 0.0 and 1.0, got {alpha_val}")
        self.alpha = alpha_val


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for PRF term extraction."""
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9_]{3,}", text.lower())


def _extract_numbers(text: str) -> list[str]:
    """Extract numeric-like tokens (years, counts, percentages, amounts)."""
    if not text:
        return []
    return re.findall(r"\d[\d,._%]*", text)


def _extract_entities(doc: Document) -> list[str]:
    """Extract simple entity-like tokens from document metadata."""
    entities: list[str] = []
    try:
        meta = getattr(doc, "metadata", {}) or {}
        title = str(meta.get("title") or "")[:200]
        section = str(meta.get("section_title") or "")[:200]
        for field in (title, section):
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", field):
                entities.append(token.lower())
    except Exception:
        return entities
    return entities


async def apply_prf(
    query: str,
    documents: list[Document],
    config: PRFConfig | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Compute a simple PRF-based expanded query from top documents.

    Args:
        query: Original user query.
        documents: Retrieved documents (typically pre-rerank top-k).
        config: PRF configuration.

    Returns:
        (expanded_query, metadata) where expanded_query may equal the original
        query when no useful terms are found.
    """
    cfg = config or PRFConfig()
    if not documents or cfg.max_terms <= 0:
        return query, {
            "enabled": False,
            "base_query": query,
            "reason": "no_documents_or_zero_max_terms",
        }

    try:
        seeds = documents[: cfg.top_n]
        term_counts: dict[str, int] = {}
        term_score_sums: dict[str, float] = {}

        use_keywords = "keywords" in cfg.sources
        use_entities = "entities" in cfg.sources
        use_numbers = "numbers" in cfg.sources

        for doc in seeds:
            text = getattr(doc, "content", "") or ""
            doc_score = float(getattr(doc, "score", 0.0) or 0.0)
            doc_terms = set()

            if use_keywords:
                doc_terms.update(_tokenize(text))
            if use_numbers:
                doc_terms.update(_extract_numbers(text))
            if use_entities:
                doc_terms.update(_extract_entities(doc))

            if not doc_terms:
                continue

            for term in doc_terms:
                term_counts[term] = term_counts.get(term, 0) + 1
                if doc_score:
                    term_score_sums[term] = term_score_sums.get(term, 0.0) + doc_score

        base_terms = set(_tokenize(query))
        if not term_counts:
            return query, {
                "enabled": False,
                "base_query": query,
                "reason": "no_additional_terms",
                "doc_seed_count": len(seeds),
            }

        # Blend original retrieval weights (from document scores) with PRF term
        # frequencies using alpha as a weighting factor.
        alpha = cfg.alpha
        max_count = max(term_counts.values())
        max_score_sum = max(term_score_sums.values()) if term_score_sums else 0.0

        term_scores: dict[str, float] = {}
        for term, count in term_counts.items():
            prf_score = float(count) / float(max_count) if max_count > 0 else 0.0
            if max_score_sum > 0.0:
                original_score = float(term_score_sums.get(term, 0.0)) / float(max_score_sum)
            else:
                original_score = 0.0
            combined_score = (1.0 - alpha) * original_score + alpha * prf_score
            term_scores[term] = combined_score

        # Rank candidate expansion terms by blended score, excluding tokens that
        # already appear in the original query.
        ranked_terms = sorted(
            (t for t in term_counts if t not in base_terms),
            key=lambda t: (-term_scores.get(t, 0.0), t),
        )
        selected = ranked_terms[: cfg.max_terms]

        if not selected:
            return query, {
                "enabled": False,
                "base_query": query,
                "reason": "no_additional_terms",
                "doc_seed_count": len(seeds),
            }

        expanded_query = f"{query} {' '.join(selected)}".strip()
        meta: dict[str, Any] = {
            "enabled": True,
            "base_query": query,
            "expanded_query": expanded_query,
            "terms_used": selected,
            "sources": list(cfg.sources),
            "doc_seed_count": len(seeds),
            "alpha": float(cfg.alpha),
        }
        logger.debug(f"PRF mined {len(selected)} terms from {len(seeds)} docs")  # pragma: no cover - debug log
        return expanded_query, meta
    except Exception:  # pragma: no cover - defensive
        logger.warning("PRF computation failed; returning original query")
        return query, {
            "enabled": False,
            "base_query": query,
            "reason": "error",
            "error": "PRF computation failed",
        }
