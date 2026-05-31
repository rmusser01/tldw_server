"""
test_supervised_policy.py

Comprehensive tests for SupervisedPolicyEngine using a real SQLite-backed
GuardianDB (no mocks). Each test gets a fresh database via tmp_path.
"""
from __future__ import annotations

import re
import time

import pytest

import tldw_Server_API.app.core.Moderation.supervised_policy as supervised_module
from tldw_Server_API.app.core.DB_Management.Guardian_DB import GuardianDB, SupervisedPolicy
from tldw_Server_API.app.core.Moderation.supervised_policy import (
    GuardianModerationProxy,
    SupervisedCheckResult,
    SupervisedPolicyEngine,
    dispatch_guardian_notification,
)
from tldw_Server_API.app.core.Moderation.moderation_service import (
    ModerationPolicy,
    PatternRule,
)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path):
    return GuardianDB(str(tmp_path / "test_guardian.db"))


@pytest.fixture
def engine(db):
    return SupervisedPolicyEngine(db)


def _setup_active_relationship(db, guardian="guardian1", dependent="child1"):
    """Helper: create and accept a relationship, return the relationship object."""
    rel = db.create_relationship(guardian, dependent)
    db.accept_relationship(rel.id)
    return rel


# ── 1. No policies -> pass result ────────────────────────────


class TestNoPolicies:
    def test_no_policies_returns_pass(self, engine):
        result = engine.check_text("hello world", "child1")
        assert result.action == "pass"
        assert result.matched_policy_id is None
        assert result.matched_pattern == ""
        assert result.redacted_text is None

    def test_no_active_relationship_returns_pass(self, db, engine):
        """Even with a relationship, if no policies exist -> pass."""
        _setup_active_relationship(db)
        result = engine.check_text("hello world", "child1")
        assert result.action == "pass"


# ── 2. Empty/whitespace text -> pass result ──────────────────


class TestEmptyText:
    def test_empty_string_returns_pass(self, engine):
        result = engine.check_text("", "child1")
        assert result.action == "pass"

    def test_whitespace_only_returns_pass(self, engine):
        result = engine.check_text("   \t\n  ", "child1")
        assert result.action == "pass"

    def test_none_coerced_empty_returns_pass(self, db, engine):
        """If text is falsy, engine short-circuits to pass."""
        _setup_active_relationship(db)
        result = engine.check_text("", "child1")
        assert result.action == "pass"


# ── 3. Literal pattern matching (case-insensitive) ──────────


class TestLiteralPatternMatching:
    def test_exact_literal_match(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="drugs",
            action="block",
        )
        result = engine.check_text("I want to talk about drugs", "child1")
        assert result.action == "block"

    def test_case_insensitive_literal(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="DRUGS",
            action="block",
        )
        result = engine.check_text("tell me about drugs please", "child1")
        assert result.action == "block"

    def test_literal_no_match_returns_pass(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="explosives",
            action="block",
        )
        result = engine.check_text("I like puppies", "child1")
        assert result.action == "pass"

    def test_literal_special_regex_chars_escaped(self, db, engine):
        """Literal patterns should be re.escape-d, so regex metacharacters
        are treated as literal text."""
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="price is $100",
            pattern_type="literal",
            action="warn",
        )
        result = engine.check_text("The price is $100 dollars", "child1")
        assert result.action == "warn"

        # Should NOT match text that lacks the dollar sign
        engine.invalidate_cache()
        result2 = engine.check_text("The price is 100 dollars", "child1")
        assert result2.action == "pass"


# ── 4. Regex pattern matching ────────────────────────────────


class TestRegexPatternMatching:
    def test_simple_regex(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern=r"\bdrugs?\b",
            pattern_type="regex",
            action="block",
        )
        result = engine.check_text("I want some drug", "child1")
        assert result.action == "block"

    def test_regex_alternation(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern=r"(beer|wine|vodka)",
            pattern_type="regex",
            action="warn",
        )
        result = engine.check_text("Can I have some wine?", "child1")
        assert result.action == "warn"

    def test_regex_no_match_returns_pass(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern=r"\b\d{3}-\d{4}\b",
            pattern_type="regex",
            action="block",
        )
        result = engine.check_text("no phone numbers here", "child1")
        assert result.action == "pass"


# ── 5. Block action ──────────────────────────────────────────


class TestBlockAction:
    def test_block_returns_correct_fields(self, db, engine):
        rel = _setup_active_relationship(db)
        pol = db.create_policy(
            relationship_id=rel.id,
            pattern="violence",
            action="block",
            category="harmful_content",
            severity="critical",
            message_to_dependent="This topic is not allowed.",
        )
        result = engine.check_text("Tell me about violence in games", "child1")
        assert result.action == "block"
        assert result.matched_policy_id == pol.id
        assert result.matched_category == "harmful_content"
        assert result.severity == "critical"
        assert result.message_to_dependent == "This topic is not allowed."
        assert result.redacted_text is None

    def test_block_uses_default_message_when_none(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="violence",
            action="block",
            message_to_dependent=None,
        )
        result = engine.check_text("tell me about violence", "child1")
        assert result.action == "block"
        assert result.message_to_dependent is not None
        assert "restricted" in result.message_to_dependent.lower()


# ── 6. Redact action ─────────────────────────────────────────


class TestRedactAction:
    def test_redact_replaces_matched_text(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="badword",
            action="redact",
        )
        result = engine.check_text("This has a badword in it", "child1")
        assert result.action == "redact"
        assert result.redacted_text is not None
        assert "[REDACTED]" in result.redacted_text
        assert "badword" not in result.redacted_text

    def test_redact_multiple_occurrences(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="secret",
            action="redact",
        )
        result = engine.check_text("secret one and secret two", "child1")
        assert result.action == "redact"
        assert result.redacted_text.count("[REDACTED]") == 2
        assert "secret" not in result.redacted_text

    def test_redact_preserves_surrounding_text(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="bad",
            action="redact",
        )
        result = engine.check_text("this is bad okay", "child1")
        assert result.redacted_text == "this is [REDACTED] okay"


# ── 7. Warn action ───────────────────────────────────────────


class TestWarnAction:
    def test_warn_returns_correct_action(self, db, engine):
        rel = _setup_active_relationship(db)
        pol = db.create_policy(
            relationship_id=rel.id,
            pattern="scary",
            action="warn",
            category="sensitive_topic",
        )
        result = engine.check_text("That movie was scary", "child1")
        assert result.action == "warn"
        assert result.matched_policy_id == pol.id
        assert result.matched_category == "sensitive_topic"
        assert result.redacted_text is None


# ── 8. Notify action ─────────────────────────────────────────


class TestNotifyAction:
    def test_notify_returns_correct_action(self, db, engine):
        rel = _setup_active_relationship(db)
        pol = db.create_policy(
            relationship_id=rel.id,
            pattern="dating",
            action="notify",
            notify_guardian=True,
        )
        result = engine.check_text("I have a question about dating", "child1")
        assert result.action == "notify"
        assert result.notify_guardian is True
        assert result.matched_policy_id == pol.id
        assert result.redacted_text is None


# ── 9. Priority ordering ─────────────────────────────────────


class TestPriorityOrdering:
    def test_block_wins_over_redact(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="forbidden",
            action="redact",
        )
        db.create_policy(
            relationship_id=rel.id,
            pattern="forbidden",
            action="block",
        )
        result = engine.check_text("this is forbidden content", "child1")
        assert result.action == "block"

    def test_redact_wins_over_warn(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="touchy",
            action="warn",
        )
        db.create_policy(
            relationship_id=rel.id,
            pattern="touchy",
            action="redact",
        )
        result = engine.check_text("this is a touchy subject", "child1")
        assert result.action == "redact"

    def test_warn_wins_over_notify(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="gossip",
            action="notify",
        )
        db.create_policy(
            relationship_id=rel.id,
            pattern="gossip",
            action="warn",
        )
        result = engine.check_text("let me tell you some gossip", "child1")
        assert result.action == "warn"

    def test_block_wins_over_all(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="extreme",
            action="notify",
        )
        db.create_policy(
            relationship_id=rel.id,
            pattern="extreme",
            action="warn",
        )
        db.create_policy(
            relationship_id=rel.id,
            pattern="extreme",
            action="redact",
        )
        db.create_policy(
            relationship_id=rel.id,
            pattern="extreme",
            action="block",
        )
        result = engine.check_text("extreme content here", "child1")
        assert result.action == "block"


# ── 10. Phase filtering ──────────────────────────────────────


class TestPhaseFiltering:
    def test_input_only_policy_fires_on_input(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="homework",
            action="block",
            phase="input",
        )
        result = engine.check_text("help me with homework", "child1", phase="input")
        assert result.action == "block"

    def test_input_only_policy_does_not_fire_on_output(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="homework",
            action="block",
            phase="input",
        )
        result = engine.check_text("help me with homework", "child1", phase="output")
        assert result.action == "pass"

    def test_output_only_policy_fires_on_output(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="spoiler",
            action="warn",
            phase="output",
        )
        result = engine.check_text("here is a spoiler", "child1", phase="output")
        assert result.action == "warn"

    def test_output_only_policy_does_not_fire_on_input(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="spoiler",
            action="warn",
            phase="output",
        )
        result = engine.check_text("here is a spoiler", "child1", phase="input")
        assert result.action == "pass"


# ── 11. Both-phase policy ────────────────────────────────────


class TestBothPhase:
    def test_both_phase_fires_on_input(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="danger",
            action="block",
            phase="both",
        )
        result = engine.check_text("this is danger", "child1", phase="input")
        assert result.action == "block"

    def test_both_phase_fires_on_output(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="danger",
            action="block",
            phase="both",
        )
        result = engine.check_text("this is danger", "child1", phase="output")
        assert result.action == "block"


# ── 12. Context snippet: topic_only ──────────────────────────


class TestContextSnippetTopicOnly:
    def test_topic_only_returns_none_snippet(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="alcohol",
            action="block",
            notify_context="topic_only",
        )
        result = engine.check_text("let me tell you about alcohol consumption", "child1")
        assert result.action == "block"
        assert result.context_snippet is None
        assert result.notify_context == "topic_only"


# ── 13. Context snippet: snippet ─────────────────────────────


class TestContextSnippetSnippet:
    def test_snippet_returns_context_around_match(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="alcohol",
            action="warn",
            notify_context="snippet",
        )
        text = "I have a question. Can we discuss alcohol with our friends later today?"
        result = engine.check_text(text, "child1")
        assert result.action == "warn"
        assert result.context_snippet is not None
        assert "alcohol" in result.context_snippet
        # Snippet should be at most ~60 chars (30 before + match + 30 after), stripped
        assert len(result.context_snippet) <= 200

    def test_snippet_near_start_of_text(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="hello",
            action="notify",
            notify_context="snippet",
        )
        result = engine.check_text("hello world", "child1")
        assert result.context_snippet is not None
        assert "hello" in result.context_snippet


# ── 14. Context snippet: full_message ─────────────────────────


class TestContextSnippetFullMessage:
    def test_full_message_returns_text(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="topic",
            action="notify",
            notify_context="full_message",
        )
        text = "Here is a topic of interest."
        result = engine.check_text(text, "child1")
        assert result.context_snippet == text

    def test_full_message_truncates_at_500_chars(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="trigger",
            action="notify",
            notify_context="full_message",
        )
        long_text = "trigger " + "x" * 600
        result = engine.check_text(long_text, "child1")
        assert result.context_snippet is not None
        assert len(result.context_snippet) == 500


# ── 15. Multiple policies, highest action wins ───────────────


class TestMultiplePoliciesHighestWins:
    def test_different_patterns_different_actions(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="hello",
            action="notify",
        )
        db.create_policy(
            relationship_id=rel.id,
            pattern="world",
            action="block",
        )
        result = engine.check_text("hello world", "child1")
        assert result.action == "block"

    def test_only_matching_policies_contribute(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="xyz_no_match",
            action="block",
        )
        db.create_policy(
            relationship_id=rel.id,
            pattern="present",
            action="warn",
        )
        result = engine.check_text("this word is present", "child1")
        assert result.action == "warn"


# ── 16. Inactive relationship policies don't fire ────────────


class TestInactiveRelationship:
    def test_pending_relationship_policies_ignored(self, db, engine):
        rel = db.create_relationship("guardian1", "child1")
        # Do NOT accept the relationship -- stays pending
        db.create_policy(
            relationship_id=rel.id,
            pattern="forbidden",
            action="block",
        )
        result = engine.check_text("this is forbidden", "child1")
        assert result.action == "pass"

    def test_dissolved_relationship_policies_ignored(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="forbidden",
            action="block",
        )
        db.dissolve_relationship(rel.id, reason="test")
        engine.invalidate_cache()
        result = engine.check_text("this is forbidden", "child1")
        assert result.action == "pass"

    def test_suspended_relationship_policies_ignored(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="forbidden",
            action="block",
        )
        db.suspend_relationship(rel.id)
        engine.invalidate_cache()
        result = engine.check_text("this is forbidden", "child1")
        assert result.action == "pass"


# ── 17. Disabled policies don't fire ─────────────────────────


class TestDisabledPolicies:
    def test_disabled_policy_is_skipped(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="forbidden",
            action="block",
            enabled=False,
        )
        result = engine.check_text("this is forbidden", "child1")
        assert result.action == "pass"

    def test_disabled_policy_via_update_is_skipped(self, db, engine):
        rel = _setup_active_relationship(db)
        pol = db.create_policy(
            relationship_id=rel.id,
            pattern="forbidden",
            action="block",
            enabled=True,
        )
        # Verify it fires first
        result1 = engine.check_text("this is forbidden", "child1")
        assert result1.action == "block"

        # Disable and invalidate cache
        db.update_policy(pol.id, enabled=False)
        engine.invalidate_cache()

        result2 = engine.check_text("this is forbidden", "child1")
        assert result2.action == "pass"


# ── 18. Cache invalidation ───────────────────────────────────


class TestCacheInvalidation:
    def test_invalidate_specific_user(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="test",
            action="block",
        )
        # Populate cache
        result1 = engine.check_text("test content", "child1")
        assert result1.action == "block"

        # Delete the policy and invalidate
        policies = db.list_policies_for_relationship(rel.id)
        for p in policies:
            db.delete_policy(p.id)
        engine.invalidate_cache(dependent_user_id="child1")

        result2 = engine.check_text("test content", "child1")
        assert result2.action == "pass"

    def test_invalidate_all_users(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="cached",
            action="block",
        )
        # Populate cache
        engine.check_text("cached value", "child1")

        # Delete and invalidate all
        for p in db.list_policies_for_relationship(rel.id):
            db.delete_policy(p.id)
        engine.invalidate_cache()  # No user_id => clear all

        result = engine.check_text("cached value", "child1")
        assert result.action == "pass"

    def test_cache_serves_compiled_policies(self, db, engine):
        """After first check, the same compiled policies are served from cache."""
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="word",
            action="warn",
        )
        result1 = engine.check_text("some word here", "child1")
        result2 = engine.check_text("another word here", "child1")
        assert result1.action == "warn"
        assert result2.action == "warn"


# ── 19. Invalid regex pattern is skipped gracefully ──────────


class TestInvalidRegex:
    def test_invalid_regex_rejected_at_creation(self, db, engine):
        rel = _setup_active_relationship(db)
        with pytest.raises(ValueError, match="Unsafe regex pattern"):
            db.create_policy(
                relationship_id=rel.id,
                pattern="[invalid(regex",
                pattern_type="regex",
                action="block",
            )

    def test_invalid_regex_does_not_block_valid_policies(self, db, engine):
        rel = _setup_active_relationship(db)
        with pytest.raises(ValueError, match="Unsafe regex pattern"):
            db.create_policy(
                relationship_id=rel.id,
                pattern="[bad(regex",
                pattern_type="regex",
                action="block",
            )
        # Valid literal policy still works
        db.create_policy(
            relationship_id=rel.id,
            pattern="goodword",
            pattern_type="literal",
            action="warn",
        )
        result = engine.check_text("this has goodword in it", "child1")
        assert result.action == "warn"

    def test_compile_logs_sanitize_unsafe_pattern_details(self, monkeypatch, db, engine):
        monkeypatch.setattr(
            db,
            "list_active_policies_for_dependent",
            lambda *_args, **_kwargs: [
                SupervisedPolicy(
                    id="policy1",
                    relationship_id="rel1",
                    pattern="private-pattern",
                    pattern_type="regex",
                )
            ],
        )
        monkeypatch.setattr(
            supervised_module,
            "validate_regex_safety",
            lambda _pattern: (False, "unsafe regex detail at /private/supervised-policy.db"),
        )

        messages: list[str] = []
        sink_id = supervised_module.logger.add(lambda message: messages.append(str(message)), level="WARNING")
        try:
            assert engine._get_compiled_policies("child1") == []
        finally:
            supervised_module.logger.remove(sink_id)

        joined = "\n".join(messages)
        assert "Unsafe supervised policy regex skipped" in joined
        assert "private-pattern" not in joined
        assert "supervised-policy.db" not in joined

    def test_compile_logs_sanitize_invalid_pattern_details(self, monkeypatch, db, engine):
        monkeypatch.setattr(
            db,
            "list_active_policies_for_dependent",
            lambda *_args, **_kwargs: [
                SupervisedPolicy(
                    id="policy1",
                    relationship_id="rel1",
                    pattern="private-pattern(",
                    pattern_type="regex",
                )
            ],
        )
        monkeypatch.setattr(supervised_module, "validate_regex_safety", lambda _pattern: (True, ""))

        messages: list[str] = []
        sink_id = supervised_module.logger.add(lambda message: messages.append(str(message)), level="WARNING")
        try:
            assert engine._get_compiled_policies("child1") == []
        finally:
            supervised_module.logger.remove(sink_id)

        joined = "\n".join(messages)
        assert "Invalid supervised policy pattern skipped" in joined
        assert "private-pattern" not in joined
        assert "missing ), unterminated subpattern" not in joined


# ── 20. build_moderation_policy_overlay ──────────────────────


class TestBuildModerationPolicyOverlay:
    def test_no_policies_returns_base_unchanged(self, db, engine):
        base = ModerationPolicy(enabled=True)
        result = engine.build_moderation_policy_overlay("child1", base)
        assert result is base

    def test_overlay_merges_patterns(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="supervised_word",
            action="block",
            category="test_cat",
        )
        base = ModerationPolicy(
            enabled=False,
            input_enabled=True,
            output_enabled=False,
            input_action="warn",
            output_action="redact",
            redact_replacement="***",
            per_user_overrides=True,
            block_patterns=[],
            categories_enabled=None,
        )
        result = engine.build_moderation_policy_overlay("child1", base)

        # Overlay forces enabled=True
        assert result.enabled is True
        # Preserves base settings
        assert result.input_enabled is True
        assert result.output_enabled is False
        assert result.input_action == "warn"
        assert result.output_action == "redact"
        assert result.redact_replacement == "***"
        assert result.per_user_overrides is True
        # Should have added one pattern
        assert len(result.block_patterns) == 1
        rule = result.block_patterns[0]
        assert isinstance(rule, PatternRule)
        assert rule.action == "block"
        assert rule.categories == {"test_cat"}

    def test_overlay_appends_to_existing_patterns(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="supervised",
            action="warn",
        )
        existing_rule = PatternRule(
            regex=re.compile(r"existing_pattern", re.IGNORECASE),
            action="block",
        )
        base = ModerationPolicy(
            enabled=True,
            block_patterns=[existing_rule],
        )
        result = engine.build_moderation_policy_overlay("child1", base)
        assert len(result.block_patterns) == 2
        # First should be the existing rule
        assert result.block_patterns[0] is existing_rule

    def test_overlay_redact_action_sets_replacement(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="pii_data",
            action="redact",
        )
        base = ModerationPolicy(enabled=True, block_patterns=[])
        result = engine.build_moderation_policy_overlay("child1", base)
        rule = result.block_patterns[0]
        assert rule.action == "redact"
        assert rule.replacement == "[REDACTED]"

    def test_overlay_notify_action_maps_to_warn(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="info",
            action="notify",
        )
        base = ModerationPolicy(enabled=True, block_patterns=[])
        result = engine.build_moderation_policy_overlay("child1", base)
        # notify is not in {block, redact, warn}, so it maps to "warn"
        rule = result.block_patterns[0]
        assert rule.action == "warn"

    def test_overlay_uses_supervised_category_when_empty(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="something",
            action="block",
            category="",
        )
        base = ModerationPolicy(enabled=True, block_patterns=[])
        result = engine.build_moderation_policy_overlay("child1", base)
        rule = result.block_patterns[0]
        assert rule.categories == {"supervised"}


# ── 21. Multiple patterns across different categories ────────


class TestMultipleCategoriesAndPatterns:
    def test_different_categories_both_matched(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="gambling",
            action="warn",
            category="gambling",
        )
        db.create_policy(
            relationship_id=rel.id,
            pattern="alcohol",
            action="block",
            category="substances",
        )
        result = engine.check_text("Let us go gambling and drink alcohol", "child1")
        # block (priority=4) > warn (priority=2)
        assert result.action == "block"
        assert result.matched_category == "substances"

    def test_only_matching_category_contributes(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="poker",
            action="block",
            category="gambling",
        )
        db.create_policy(
            relationship_id=rel.id,
            pattern="soda",
            action="warn",
            category="dietary",
        )
        result = engine.check_text("I want a soda", "child1")
        assert result.action == "warn"
        assert result.matched_category == "dietary"


# ── 22. Additional edge cases ─────────────────────────────────


class TestEdgeCases:
    def test_empty_pattern_is_skipped(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="",
            action="block",
        )
        result = engine.check_text("anything goes", "child1")
        assert result.action == "pass"

    def test_different_dependents_isolated(self, db, engine):
        """Policies for child1 should not affect child2."""
        rel1 = _setup_active_relationship(db, "guardian1", "child1")
        db.create_policy(
            relationship_id=rel1.id,
            pattern="restricted",
            action="block",
        )
        rel2 = _setup_active_relationship(db, "guardian1", "child2")
        # child2 has no policies

        result_child1 = engine.check_text("restricted content", "child1")
        assert result_child1.action == "block"

        result_child2 = engine.check_text("restricted content", "child2")
        assert result_child2.action == "pass"

    def test_multiple_guardians_policies_combined(self, db, engine):
        """If a child has multiple active guardians, all policies apply."""
        rel1 = _setup_active_relationship(db, "guardian1", "child1")
        rel2 = _setup_active_relationship(db, "guardian2", "child1")
        db.create_policy(
            relationship_id=rel1.id,
            pattern="word_a",
            action="warn",
        )
        db.create_policy(
            relationship_id=rel2.id,
            pattern="word_b",
            action="block",
        )
        result = engine.check_text("word_b is here", "child1")
        assert result.action == "block"

    def test_default_check_result_fields(self):
        """Verify default values of SupervisedCheckResult dataclass."""
        r = SupervisedCheckResult()
        assert r.action == "pass"
        assert r.matched_policy_id is None
        assert r.matched_category == ""
        assert r.matched_pattern == ""
        assert r.severity == "info"
        assert r.message_to_dependent is None
        assert r.notify_guardian is False
        assert r.notify_context == "topic_only"
        assert r.context_snippet is None
        assert r.redacted_text is None


# ── 23. Redaction with multiple redact-able policies ──────────


class TestMultipleRedactPolicies:
    def test_multiple_redact_patterns_all_applied(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="word_a",
            action="redact",
        )
        db.create_policy(
            relationship_id=rel.id,
            pattern="word_b",
            action="redact",
        )
        text = "word_a and word_b together"
        result = engine.check_text(text, "child1")
        assert result.action == "redact"
        assert "word_a" not in result.redacted_text
        assert "word_b" not in result.redacted_text
        assert result.redacted_text.count("[REDACTED]") == 2


# ── 24. Matched pattern field contains the regex pattern ─────


class TestMatchedPatternField:
    def test_literal_matched_pattern_is_escaped(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="hello",
            pattern_type="literal",
            action="warn",
        )
        result = engine.check_text("say hello", "child1")
        assert result.matched_pattern == re.escape("hello")

    def test_regex_matched_pattern_is_raw(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern=r"\btest\b",
            pattern_type="regex",
            action="warn",
        )
        result = engine.check_text("a test case", "child1")
        assert result.matched_pattern == r"\btest\b"


# ── 25. Notify guardian field ─────────────────────────────────


class TestNotifyGuardianField:
    def test_notify_guardian_true(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="alert",
            action="warn",
            notify_guardian=True,
        )
        result = engine.check_text("alert triggered", "child1")
        assert result.notify_guardian is True

    def test_notify_guardian_false(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="quiet",
            action="warn",
            notify_guardian=False,
        )
        result = engine.check_text("quiet alert", "child1")
        assert result.notify_guardian is False


# ── 26. Redact does not fire for non-redact actions ──────────


class TestRedactOnlyForRedactAction:
    def test_block_has_no_redacted_text(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="blocked",
            action="block",
        )
        result = engine.check_text("this is blocked content", "child1")
        assert result.action == "block"
        assert result.redacted_text is None

    def test_warn_has_no_redacted_text(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="warned",
            action="warn",
        )
        result = engine.check_text("this is warned content", "child1")
        assert result.action == "warn"
        assert result.redacted_text is None


# ── 27. Phase filtering with redact ──────────────────────────


class TestPhaseFilteringWithRedact:
    def test_input_redact_does_not_apply_on_output_check(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="secret",
            action="redact",
            phase="input",
        )
        result = engine.check_text("tell me a secret", "child1", phase="output")
        assert result.action == "pass"
        assert result.redacted_text is None

    def test_output_redact_applies_on_output(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="secret",
            action="redact",
            phase="output",
        )
        result = engine.check_text("here is a secret", "child1", phase="output")
        assert result.action == "redact"
        assert "secret" not in result.redacted_text


# ── 28. Cache TTL behavior ───────────────────────────────────


class TestCacheTTL:
    def test_cache_expires_after_ttl(self, db, engine):
        """Simulate TTL expiration by manipulating cache timestamps."""
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="cached_word",
            action="block",
        )
        # Populate cache
        result1 = engine.check_text("cached_word here", "child1")
        assert result1.action == "block"

        # Delete the policy (cache still holds old data)
        for p in db.list_policies_for_relationship(rel.id):
            db.delete_policy(p.id)

        # Force cache timestamp to be in the past
        with engine._lock:
            engine._cache_timestamps["child1:"] = 0.0

        result2 = engine.check_text("cached_word here", "child1")
        assert result2.action == "pass"


# ── 29. Severity field propagation ───────────────────────────


class TestSeverityPropagation:
    def test_severity_is_propagated(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="critical_content",
            action="block",
            severity="critical",
        )
        result = engine.check_text("critical_content detected", "child1")
        assert result.severity == "critical"

    def test_info_severity(self, db, engine):
        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="mild_content",
            action="notify",
            severity="info",
        )
        result = engine.check_text("mild_content here", "child1")
        assert result.severity == "info"


# ── 30. GovernancePolicyId on SupervisedPolicy ────────────────


class TestGovernancePolicyId:
    def test_create_policy_with_governance_policy_id(self, db, engine):
        rel = _setup_active_relationship(db)
        gp = db.create_governance_policy(
            owner_user_id="guardian1",
            name="School Hours",
            policy_mode="guardian",
        )
        pol = db.create_policy(
            relationship_id=rel.id,
            pattern="games",
            action="block",
            governance_policy_id=gp.id,
        )
        assert pol.governance_policy_id == gp.id
        # Read back
        fetched = db.get_policy(pol.id)
        assert fetched.governance_policy_id == gp.id

    def test_policy_without_governance_policy_id_backward_compat(self, db, engine):
        rel = _setup_active_relationship(db)
        pol = db.create_policy(
            relationship_id=rel.id,
            pattern="test",
            action="warn",
        )
        assert pol.governance_policy_id is None
        fetched = db.get_policy(pol.id)
        assert fetched.governance_policy_id is None

    def test_update_governance_policy_id(self, db, engine):
        rel = _setup_active_relationship(db)
        gp = db.create_governance_policy(
            owner_user_id="guardian1",
            name="Evening Policy",
            policy_mode="guardian",
        )
        pol = db.create_policy(
            relationship_id=rel.id,
            pattern="test",
            action="warn",
        )
        assert pol.governance_policy_id is None
        db.update_policy(pol.id, governance_policy_id=gp.id)
        fetched = db.get_policy(pol.id)
        assert fetched.governance_policy_id == gp.id


# ── 31. Schedule Filtering ────────────────────────────────────


class TestScheduleFiltering:
    def test_policy_skipped_outside_schedule(self, db, engine):
        """Policy linked to a governance policy with a non-matching day is skipped."""
        rel = _setup_active_relationship(db)
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("UTC"))
        day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        # Pick a day that is NOT today
        other_day = day_names[(now.weekday() + 3) % 7]
        gp = db.create_governance_policy(
            owner_user_id="guardian1",
            name="Weekday Only",
            policy_mode="guardian",
            schedule_days=other_day,
            schedule_timezone="UTC",
        )
        db.create_policy(
            relationship_id=rel.id,
            pattern="restricted",
            action="block",
            governance_policy_id=gp.id,
        )
        result = engine.check_text("restricted content", "child1")
        assert result.action == "pass"

    def test_policy_active_within_schedule(self, db, engine):
        """Policy linked to a governance policy with today's day should fire."""
        rel = _setup_active_relationship(db)
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("UTC"))
        day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        today_name = day_names[now.weekday()]
        gp = db.create_governance_policy(
            owner_user_id="guardian1",
            name="Today Policy",
            policy_mode="guardian",
            schedule_days=today_name,
            schedule_timezone="UTC",
        )
        db.create_policy(
            relationship_id=rel.id,
            pattern="restricted",
            action="block",
            governance_policy_id=gp.id,
        )
        result = engine.check_text("restricted content", "child1")
        assert result.action == "block"

    def test_no_schedule_always_active(self, db, engine):
        """Policy linked to governance policy with no schedule is always active."""
        rel = _setup_active_relationship(db)
        gp = db.create_governance_policy(
            owner_user_id="guardian1",
            name="No Schedule",
            policy_mode="guardian",
        )
        db.create_policy(
            relationship_id=rel.id,
            pattern="word",
            action="block",
            governance_policy_id=gp.id,
        )
        result = engine.check_text("word here", "child1")
        assert result.action == "block"


# ── 32. Chat-Type Filtering ──────────────────────────────────


class TestChatTypeFiltering:
    def test_policy_skipped_for_wrong_chat_type(self, db, engine):
        rel = _setup_active_relationship(db)
        gp = db.create_governance_policy(
            owner_user_id="guardian1",
            name="Character Only",
            policy_mode="guardian",
            scope_chat_types="character",
        )
        db.create_policy(
            relationship_id=rel.id,
            pattern="trigger",
            action="block",
            governance_policy_id=gp.id,
        )
        result = engine.check_text("trigger word", "child1", chat_type="regular")
        assert result.action == "pass"

    def test_policy_matches_for_correct_chat_type(self, db, engine):
        rel = _setup_active_relationship(db)
        gp = db.create_governance_policy(
            owner_user_id="guardian1",
            name="Character Only",
            policy_mode="guardian",
            scope_chat_types="character",
        )
        db.create_policy(
            relationship_id=rel.id,
            pattern="trigger",
            action="block",
            governance_policy_id=gp.id,
        )
        result = engine.check_text("trigger word", "child1", chat_type="character")
        assert result.action == "block"

    def test_scope_all_matches_any_chat_type(self, db, engine):
        rel = _setup_active_relationship(db)
        gp = db.create_governance_policy(
            owner_user_id="guardian1",
            name="All Types",
            policy_mode="guardian",
            scope_chat_types="all",
        )
        db.create_policy(
            relationship_id=rel.id,
            pattern="trigger",
            action="block",
            governance_policy_id=gp.id,
        )
        result = engine.check_text("trigger word", "child1", chat_type="rag")
        assert result.action == "block"


# ── 33. Transparent Mode ─────────────────────────────────────


class TestTransparentMode:
    def test_transparent_shows_rule_name(self, db, engine):
        rel = _setup_active_relationship(db)
        gp = db.create_governance_policy(
            owner_user_id="guardian1",
            name="School Policy",
            policy_mode="guardian",
            transparent=True,
        )
        db.create_policy(
            relationship_id=rel.id,
            pattern="games",
            action="block",
            category="entertainment",
            governance_policy_id=gp.id,
        )
        result = engine.check_text("play some games", "child1")
        assert result.action == "block"
        assert result.rule_name_visible == "School Policy"
        assert "School Policy" in result.message_to_dependent

    def test_non_transparent_hides_rule_name(self, db, engine):
        rel = _setup_active_relationship(db)
        gp = db.create_governance_policy(
            owner_user_id="guardian1",
            name="Hidden Policy",
            policy_mode="guardian",
            transparent=False,
        )
        db.create_policy(
            relationship_id=rel.id,
            pattern="games",
            action="block",
            governance_policy_id=gp.id,
        )
        result = engine.check_text("play some games", "child1")
        assert result.action == "block"
        assert result.rule_name_visible is None


# ── 34. Guardian Notification Dispatch ────────────────────────


class TestGuardianNotificationDispatch:
    def test_dispatches_when_notify_guardian_true(self, db, engine):
        from unittest.mock import MagicMock, patch
        from tldw_Server_API.app.core.Moderation.supervised_policy import (
            SupervisedCheckResult,
            dispatch_guardian_notification,
        )
        result = SupervisedCheckResult(
            action="block",
            notify_guardian=True,
            severity="critical",
            matched_category="violence",
            matched_pattern="fight",
        )
        mock_svc = MagicMock()
        mock_svc.notify_or_batch.return_value = "logged"
        with patch(
            "tldw_Server_API.app.core.Monitoring.notification_service.get_notification_service",
            return_value=mock_svc,
        ):
            status = dispatch_guardian_notification(result, "child1", "guardian1")
        assert status == "logged"
        mock_svc.notify_or_batch.assert_called_once()
        payload = mock_svc.notify_or_batch.call_args[0][0]
        assert payload["type"] == "guardian_alert"
        assert payload["dependent_user_id"] == "child1"
        assert payload["guardian_user_id"] == "guardian1"

    def test_skipped_when_action_pass(self):
        from tldw_Server_API.app.core.Moderation.supervised_policy import (
            SupervisedCheckResult,
            dispatch_guardian_notification,
        )
        result = SupervisedCheckResult(action="pass", notify_guardian=True)
        status = dispatch_guardian_notification(result, "child1")
        assert status == "skipped"

    def test_skipped_when_notify_guardian_false(self):
        from tldw_Server_API.app.core.Moderation.supervised_policy import (
            SupervisedCheckResult,
            dispatch_guardian_notification,
        )
        result = SupervisedCheckResult(action="block", notify_guardian=False)
        status = dispatch_guardian_notification(result, "child1")
        assert status == "skipped"

    def test_dispatch_failure_log_sanitizes_backend_details(self, monkeypatch):
        result = SupervisedCheckResult(
            action="block",
            notify_guardian=True,
            severity="critical",
            matched_category="violence",
            matched_pattern="fight",
        )

        def fail_service():
            raise RuntimeError("notify backend failed at /private/guardian-notify.db")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Monitoring.notification_service.get_notification_service",
            fail_service,
        )

        messages: list[str] = []
        sink_id = supervised_module.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
        try:
            status = dispatch_guardian_notification(result, "child1", "guardian1")
        finally:
            supervised_module.logger.remove(sink_id)

        joined = "\n".join(messages)
        assert status == "failed"
        assert "Guardian notification dispatch failed" in joined
        assert "guardian-notify.db" not in joined


class TestGuardianModerationProxySanitizers:
    def test_overlay_failure_log_sanitizes_backend_details(self):
        base_policy = ModerationPolicy(enabled=True)

        class Base:
            def get_effective_policy(self, user_id=None):  # noqa: ANN001, ANN202
                return base_policy

        class Engine:
            def build_moderation_policy_overlay(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
                raise RuntimeError("overlay backend failed at /private/guardian-overlay.db")

        proxy = GuardianModerationProxy(Base(), Engine(), "child1")

        messages: list[str] = []
        sink_id = supervised_module.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
        try:
            result = proxy.get_effective_policy("child1")
        finally:
            supervised_module.logger.remove(sink_id)

        joined = "\n".join(messages)
        assert result is base_policy
        assert "Guardian policy overlay skipped in proxy" in joined
        assert "guardian-overlay.db" not in joined
