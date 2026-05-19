"""
semantic_matcher.py

Tier 3 (embedding similarity) and Tier 4 (LLM classification) matching
for content governance rules.

- Tier 3: Embeds text and compares cosine similarity against reference texts.
- Tier 4: Uses LLM to classify text into categories.

Both tiers are optional, configurable, and off by default.
Falls back gracefully if embedding/LLM providers are unavailable.
"""
from __future__ import annotations

import math
from typing import Any

from loguru import logger

_SEMANTIC_NONCRITICAL = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ImportError,
)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticMatcher:
    """Embedding-based and LLM-based content matching.

    Uses the project's existing embedding and chat APIs when available.
    Caches reference embeddings per rule for efficiency.
    """

    def __init__(self, embedding_config: dict[str, Any] | None = None) -> None:
        self._embedding_config = embedding_config or {}
        # Cache: frozenset(reference_texts) -> list of embeddings
        self._ref_cache: dict[tuple[str, ...], list[list[float]]] = {}

    def _embed_text(self, text: str) -> list[float] | None:
        """Embed a single text string using the project's embedding provider."""
        try:
            from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
                create_embedding,
            )
            provider = self._embedding_config.get("provider", "openai")
            model = self._embedding_config.get("model", "text-embedding-3-small")
            result = create_embedding(text, provider=provider, model=model)
            if isinstance(result, list) and result and isinstance(result[0], (int, float)):
                return result
            if isinstance(result, dict):
                return result.get("embedding", [])
            return None
        except _SEMANTIC_NONCRITICAL:
            logger.debug("Embedding failed")
            return None

    def _get_reference_embeddings(
        self, reference_texts: list[str],
    ) -> list[list[float]]:
        """Get embeddings for reference texts (cached)."""
        cache_key = tuple(sorted(reference_texts))
        if cache_key in self._ref_cache:
            return self._ref_cache[cache_key]

        embeddings: list[list[float]] = []
        for ref in reference_texts:
            emb = self._embed_text(ref)
            if emb:
                embeddings.append(emb)
        if embeddings:
            self._ref_cache[cache_key] = embeddings
        return embeddings

    def check_similarity(
        self,
        text: str,
        reference_texts: list[str],
        threshold: float = 0.75,
    ) -> tuple[bool, float, str | None]:
        """Check semantic similarity between text and reference texts.

        Returns (matched, best_score, best_reference_text).
        """
        if not text or not reference_texts:
            return False, 0.0, None

        text_emb = self._embed_text(text)
        if not text_emb:
            return False, 0.0, None

        ref_embeddings = self._get_reference_embeddings(reference_texts)
        if not ref_embeddings:
            return False, 0.0, None

        best_score = 0.0
        best_ref: str | None = None
        for ref_text, ref_emb in zip(reference_texts, ref_embeddings):
            score = _cosine_similarity(text_emb, ref_emb)
            if score > best_score:
                best_score = score
                best_ref = ref_text

        matched = best_score >= threshold
        return matched, best_score, best_ref if matched else None

    def classify_with_llm(
        self,
        text: str,
        categories: list[str],
        prompt_template: str | None = None,
    ) -> tuple[bool, str | None, float]:
        """Use LLM to classify text into one of the given categories.

        Returns (matched, matched_category, confidence).
        A match is when the LLM identifies the text as belonging to any category.
        """
        if not text or not categories:
            return False, None, 0.0

        template = prompt_template or (
            "Classify the following text into one of these categories: {categories}. "
            "If the text does not belong to any category, respond with 'none'. "
            "Respond with ONLY the category name (or 'none'), nothing else.\n\n"
            "Text: {text}"
        )
        prompt = template.format(
            categories=", ".join(categories),
            text=text[:500],  # Truncate for safety
        )

        try:
            from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call
            response = perform_chat_api_call(
                api_endpoint="openai",
                api_key=self._embedding_config.get("api_key", ""),
                input_data=prompt,
                system_message="You are a content classifier. Respond with only the category name or 'none'.",
                temp=0.0,
                streaming=False,
            )
            result_text = str(response).strip().lower()

            for cat in categories:
                if cat.lower() in result_text:
                    return True, cat, 0.9  # High confidence for LLM match
            if result_text == "none" or "none" in result_text:
                return False, None, 0.1
            return False, None, 0.5  # Ambiguous response
        except _SEMANTIC_NONCRITICAL:
            logger.debug("LLM classification failed")
            return False, None, 0.0

    def clear_cache(self) -> None:
        """Clear the reference embedding cache."""
        self._ref_cache.clear()
