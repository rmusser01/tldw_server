"""
Tests for SemanticMatcher — Tier 3 (embedding similarity) and Tier 4 (LLM classification).
Uses mocks since embedding/LLM providers may not be available in test environments.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

import tldw_Server_API.app.core.Moderation.semantic_matcher as semantic_module
from tldw_Server_API.app.core.Moderation.semantic_matcher import (
    SemanticMatcher,
    _cosine_similarity,
)


# ── Cosine similarity unit tests ──────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_different_length_vectors(self):
        assert _cosine_similarity([1.0], [1.0, 0.0]) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ── Semantic matching (Tier 3) with mocked embeddings ─────────


class TestSemanticMatch:
    def test_high_similarity_match(self):
        """When cosine similarity >= threshold, should match."""
        matcher = SemanticMatcher()
        # Mock _embed_text to return known vectors
        with patch.object(matcher, "_embed_text") as mock_embed:
            # Text embedding close to reference
            mock_embed.side_effect = lambda t: (
                [0.9, 0.1] if t == "test input" else [1.0, 0.0]
            )
            matched, score, ref = matcher.check_similarity(
                "test input", ["reference text"], threshold=0.9
            )
            assert matched is True
            assert score > 0.9
            assert ref == "reference text"

    def test_low_similarity_no_match(self):
        """When cosine similarity < threshold, should not match."""
        matcher = SemanticMatcher()
        with patch.object(matcher, "_embed_text") as mock_embed:
            mock_embed.side_effect = lambda t: (
                [0.0, 1.0] if t == "unrelated text" else [1.0, 0.0]
            )
            matched, score, ref = matcher.check_similarity(
                "unrelated text", ["reference"], threshold=0.75
            )
            assert matched is False
            assert ref is None

    def test_threshold_boundary(self):
        """Exact threshold should match."""
        matcher = SemanticMatcher()
        with patch.object(matcher, "_embed_text") as mock_embed:
            # Return same vector for both -> cosine = 1.0
            mock_embed.return_value = [1.0, 0.0]
            matched, score, ref = matcher.check_similarity(
                "text", ["ref"], threshold=1.0
            )
            assert matched is True
            assert score == pytest.approx(1.0)

    def test_empty_text_no_match(self):
        matcher = SemanticMatcher()
        matched, score, ref = matcher.check_similarity("", ["ref"])
        assert matched is False

    def test_empty_references_no_match(self):
        matcher = SemanticMatcher()
        matched, score, ref = matcher.check_similarity("text", [])
        assert matched is False

    def test_embedding_failure_returns_no_match(self):
        """If embedding fails, should return no match gracefully."""
        matcher = SemanticMatcher()
        with patch.object(matcher, "_embed_text", return_value=None):
            matched, score, ref = matcher.check_similarity("text", ["ref"])
            assert matched is False
            assert score == 0.0

    def test_embedding_failure_log_sanitizes_backend_details(self):
        matcher = SemanticMatcher()
        messages: list[str] = []

        with patch(
            "tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create.create_embedding",
            side_effect=RuntimeError("embedding backend failed at /private/semantic-embed.db"),
        ):
            sink_id = semantic_module.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
            try:
                assert matcher._embed_text("sensitive prompt") is None
            finally:
                semantic_module.logger.remove(sink_id)

        joined = "\n".join(messages)
        assert "Embedding failed" in joined
        assert "semantic-embed.db" not in joined

    def test_reference_caching(self):
        """Reference embeddings should be cached."""
        matcher = SemanticMatcher()
        call_count = 0

        def mock_embed(text):
            nonlocal call_count
            call_count += 1
            return [1.0, 0.0]

        with patch.object(matcher, "_embed_text", side_effect=mock_embed):
            matcher.check_similarity("text1", ["ref1", "ref2"])
            first_count = call_count
            # Second call with same references should use cache
            matcher.check_similarity("text2", ["ref1", "ref2"])
            # Only 1 new call for text2 embedding (refs cached)
            assert call_count == first_count + 1


# ── LLM classification (Tier 4) with mocked LLM ──────────────


class TestLLMClassification:
    def test_matching_category(self):
        """LLM classifying text into a known category should match."""
        matcher = SemanticMatcher()
        with patch(
            "tldw_Server_API.app.core.Moderation.semantic_matcher.SemanticMatcher.classify_with_llm",
            wraps=matcher.classify_with_llm,
        ):
            # Mock the chat API call
            with patch(
                "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call",
                return_value="violence",
            ):
                matched, category, conf = matcher.classify_with_llm(
                    "I want to fight", ["violence", "drugs"]
                )
                assert matched is True
                assert category == "violence"
                assert conf > 0.5

    def test_no_match_returns_none(self):
        """LLM responding 'none' should not match."""
        matcher = SemanticMatcher()
        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call",
            return_value="none",
        ):
            matched, category, conf = matcher.classify_with_llm(
                "I like puppies", ["violence", "drugs"]
            )
            assert matched is False
            assert category is None

    def test_empty_text_no_match(self):
        matcher = SemanticMatcher()
        matched, category, conf = matcher.classify_with_llm("", ["violence"])
        assert matched is False

    def test_empty_categories_no_match(self):
        matcher = SemanticMatcher()
        matched, category, conf = matcher.classify_with_llm("some text", [])
        assert matched is False

    def test_llm_failure_returns_no_match(self):
        """If LLM call fails, should return no match gracefully."""
        matcher = SemanticMatcher()
        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call",
            side_effect=RuntimeError("API unavailable"),
        ):
            matched, category, conf = matcher.classify_with_llm(
                "test text", ["violence"]
            )
            assert matched is False
            assert conf == 0.0

    def test_llm_failure_log_sanitizes_backend_details(self):
        matcher = SemanticMatcher()
        messages: list[str] = []

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call",
            side_effect=RuntimeError("llm backend failed at /private/semantic-llm.db"),
        ):
            sink_id = semantic_module.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
            try:
                matched, category, conf = matcher.classify_with_llm("test text", ["violence"])
            finally:
                semantic_module.logger.remove(sink_id)

        joined = "\n".join(messages)
        assert matched is False
        assert category is None
        assert conf == 0.0
        assert "LLM classification failed" in joined
        assert "semantic-llm.db" not in joined

    def test_clear_cache(self):
        """clear_cache should reset reference cache."""
        matcher = SemanticMatcher()
        matcher._ref_cache[("test",)] = [[1.0, 0.0]]
        assert len(matcher._ref_cache) == 1
        matcher.clear_cache()
        assert len(matcher._ref_cache) == 0


# ── Regression: existing regex patterns unaffected ────────────


class TestRegexPatternsUnaffected:
    """Ensure literal/regex patterns still work when semantic types exist."""

    def test_regex_still_works_in_supervised_policy(self):
        """SupervisedPolicyEngine should still handle regex patterns."""
        from tldw_Server_API.app.core.DB_Management.Guardian_DB import GuardianDB
        from tldw_Server_API.app.core.Moderation.supervised_policy import (
            SupervisedPolicyEngine,
        )
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmp:
            db = GuardianDB(os.path.join(tmp, "test.db"))
            engine = SupervisedPolicyEngine(db)
            rel = db.create_relationship("guardian1", "child1")
            db.accept_relationship(rel.id)
            db.create_policy(
                relationship_id=rel.id,
                pattern=r"\bdrugs?\b",
                pattern_type="regex",
                action="block",
            )
            result = engine.check_text("tell me about drugs", "child1")
            assert result.action == "block"
