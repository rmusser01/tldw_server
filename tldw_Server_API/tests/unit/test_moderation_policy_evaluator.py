from __future__ import annotations

import inspect
import re
from dataclasses import FrozenInstanceError

import pytest

from tldw_Server_API.app.core.Moderation.moderation_service import (
    ModerationEvaluationResult,
    ModerationPolicy,
    PatternRule,
)
from tldw_Server_API.app.core.Moderation.policy_evaluator import (
    EvaluationLimits,
    PolicyEvaluator,
)

LIMITS = EvaluationLimits(
    max_scan_chars=10,
    match_window_chars=5,
    max_fallback_scan_chars=100,
    max_replacements_per_pattern=10,
)


def _policy(*rules, **overrides):
    values = {
        "enabled": True,
        "input_enabled": True,
        "output_enabled": True,
        "input_action": "block",
        "output_action": "redact",
        "redact_replacement": "[REDACTED]",
        "per_user_overrides": False,
        "block_patterns": list(rules),
        "categories_enabled": None,
    }
    values.update(overrides)
    return ModerationPolicy(**values)


def _rule(
    pattern,
    *,
    action=None,
    replacement=None,
    categories=None,
    phase="both",
):
    return PatternRule(
        regex=re.compile(pattern),
        action=action,
        replacement=replacement,
        categories=categories,
        phase=phase,
    )


class _MissingLower:
    pass


class _LowerBytes:
    def lower(self):
        return b"block"


class _LowerBlock:
    def lower(self):
        return "block"


class _LowerUnhashable:
    def lower(self):
        return []


def test_evaluation_limits_are_frozen():
    with pytest.raises(FrozenInstanceError):
        LIMITS.max_scan_chars = 99


def test_evaluation_limits_constructor_preserves_field_identities():
    max_scan_chars = object()
    match_window_chars = object()
    max_fallback_scan_chars = object()
    max_replacements_per_pattern = object()

    limits = EvaluationLimits(
        max_scan_chars=max_scan_chars,  # type: ignore[arg-type]
        match_window_chars=match_window_chars,  # type: ignore[arg-type]
        max_fallback_scan_chars=max_fallback_scan_chars,  # type: ignore[arg-type]
        max_replacements_per_pattern=max_replacements_per_pattern,  # type: ignore[arg-type]
    )

    assert limits.max_scan_chars is max_scan_chars
    assert limits.match_window_chars is match_window_chars
    assert limits.max_fallback_scan_chars is max_fallback_scan_chars
    assert limits.max_replacements_per_pattern is max_replacements_per_pattern


def test_direct_policy_type_loader_and_evaluator_shape_are_literal():
    descriptor = inspect.getattr_static(PolicyEvaluator, "policy_types")
    evaluator = PolicyEvaluator()

    assert isinstance(descriptor, staticmethod)
    assert evaluator.policy_types() == (
        ModerationPolicy,
        PatternRule,
        ModerationEvaluationResult,
    )
    assert vars(evaluator) == {}


def test_direct_redaction_remains_deferred_to_task_4():
    with pytest.raises(AttributeError, match="redact_text"):
        PolicyEvaluator().evaluate_text(
            "secret",
            _policy(_rule("secret", action="redact")),
            "input",
            LIMITS,
            include_redacted_text=True,
        )


def test_direct_decision_evaluation_has_literal_result():
    policy = _policy(
        PatternRule(
            regex=re.compile("secret"),
            action="block",
            replacement="[RULE]",
            categories={"pii", "confidential"},
            phase="input",
        )
    )

    result = PolicyEvaluator().evaluate_text(
        "secret here",
        policy,
        "input",
        LIMITS,
        include_redacted_text=False,
    )

    assert result == ModerationEvaluationResult(
        action="block",
        redacted_text=None,
        matched_pattern="secret",
        category="confidential",
        match_span=(0, 6),
        sample="[RULE] here",
    )


@pytest.mark.parametrize(
    ("phase", "overrides", "expected_action"),
    [
        ("input", {}, "block"),
        ("output", {}, "redact"),
        ("input", {"input_enabled": False}, "pass"),
        ("output", {"output_enabled": False}, "pass"),
        (None, {}, "warn"),
        ("unknown", {}, "warn"),
    ],
)
def test_direct_phase_behavior_is_literal(
    phase,
    overrides,
    expected_action,
):
    result = PolicyEvaluator().evaluate_text(
        "secret",
        _policy(_rule("secret"), **overrides),
        phase,
        LIMITS,
        include_redacted_text=False,
    )

    assert result.action == expected_action
    assert result.matched_pattern == ("secret" if expected_action != "pass" else None)


@pytest.mark.parametrize("phase", [None, "unknown"])
@pytest.mark.parametrize("rule_phase", ["input", "output"])
def test_direct_unknown_phase_bypasses_rule_phase_metadata(
    phase,
    rule_phase,
):
    result = PolicyEvaluator().evaluate_text(
        "secret",
        _policy(_rule("secret", phase=rule_phase)),
        phase,
        LIMITS,
        include_redacted_text=False,
    )

    assert result.action == "warn"
    assert result.matched_pattern == "secret"


def test_direct_disabled_policy_and_raw_regex_behavior_are_literal():
    evaluator = PolicyEvaluator()
    disabled = evaluator.evaluate_text(
        "secret",
        _policy(_rule("secret"), enabled=False),
        "input",
        LIMITS,
        include_redacted_text=False,
    )
    raw = evaluator.evaluate_text(
        "secret",
        _policy(
            re.compile("secret"),
            categories_enabled={"restricted"},
        ),
        "input",
        LIMITS,
        include_redacted_text=False,
    )

    assert disabled == ModerationEvaluationResult()
    assert raw == ModerationEvaluationResult(
        action="block",
        redacted_text=None,
        matched_pattern="secret",
        category=None,
        match_span=(0, 6),
        sample="[REDACTED]",
    )


def test_direct_category_filter_ranking_and_tie_behavior_are_literal():
    policy = _policy(
        _rule(
            "later",
            action="warn",
            categories={"unselected"},
        ),
        _rule(
            "early",
            action="block",
            replacement="[FIRST]",
            categories={"pii", "financial", "confidential"},
        ),
        _rule(
            "early",
            action="block",
            replacement="[SECOND]",
            categories={"other"},
        ),
        categories_enabled={"pii", "confidential", "financial", "other"},
    )

    result = PolicyEvaluator().evaluate_text(
        "early ... later",
        policy,
        "input",
        LIMITS,
        include_redacted_text=False,
    )

    assert result == ModerationEvaluationResult(
        action="block",
        redacted_text=None,
        matched_pattern="early",
        category="confidential",
        match_span=(0, 5),
        sample="[FIRST] ... later",
    )


def test_direct_redact_rank_and_enabled_category_wildcard_are_literal():
    result = PolicyEvaluator().evaluate_text(
        "warn first, redact later",
        _policy(
            _rule(
                "warn",
                action="warn",
                categories={"first"},
            ),
            _rule(
                "later",
                action="redact",
                categories={"restricted"},
            ),
            categories_enabled={"*"},
        ),
        "input",
        LIMITS,
        include_redacted_text=False,
    )

    assert result.action == "redact"
    assert result.matched_pattern == "later"
    assert result.category is None
    assert result.match_span == (19, 24)


@pytest.mark.parametrize(
    ("action", "expected_action"),
    [
        (None, "block"),
        ("", "block"),
        ("unsupported", "warn"),
    ],
)
def test_direct_falsy_and_unsupported_string_actions_are_literal(
    action,
    expected_action,
):
    result = PolicyEvaluator().evaluate_text(
        "secret",
        _policy(_rule("secret", action=action)),
        "input",
        LIMITS,
        include_redacted_text=False,
    )

    assert result.action == expected_action


def test_direct_effective_action_lower_result_behavior_is_literal():
    evaluator = PolicyEvaluator()

    bytes_result = evaluator.evaluate_text(
        "secret",
        _policy(_rule("secret", action=_LowerBytes())),
        "input",
        LIMITS,
        include_redacted_text=False,
    )
    block_result = evaluator.evaluate_text(
        "secret",
        _policy(_rule("secret", action=_LowerBlock())),
        "input",
        LIMITS,
        include_redacted_text=False,
    )

    assert bytes_result.action == "warn"
    assert block_result.action == "block"

    with pytest.raises(AttributeError):
        evaluator.evaluate_text(
            "secret",
            _policy(_rule("secret", action=_MissingLower())),
            "input",
            LIMITS,
            include_redacted_text=False,
        )
    with pytest.raises(TypeError):
        evaluator.evaluate_text(
            "secret",
            _policy(_rule("secret", action=_LowerUnhashable())),
            "input",
            LIMITS,
            include_redacted_text=False,
        )


def test_direct_public_snippet_lookup_is_literal():
    policy = _policy(
        _rule("secret", replacement="[FIRST]"),
        _rule("secret", replacement="[SECOND]"),
        redact_replacement="[POLICY]",
    )

    assert (
        PolicyEvaluator().build_sanitized_snippet(
            "before secret after",
            policy,
            (7, 13),
            pattern="secret",
        )
        == "before [FIRST] after"
    )


def test_direct_snippet_bounds_fallback_and_truncation_are_literal():
    evaluator = PolicyEvaluator()

    assert (
        evaluator.build_sanitized_snippet_for_replacement(
            "secret",
            (-3, 99),
            "",
        )
        == "[REDACTED]"
    )

    long_snippet = evaluator.build_sanitized_snippet_for_replacement(
        ("a" * 20) + "secret" + ("b" * 20),
        (20, 26),
        "R" * 100,
    )

    assert long_snippet is not None
    assert len(long_snippet) == 80
    assert long_snippet.endswith("...")


def test_direct_scan_geometry_matches_characterized_behavior():
    evaluator = PolicyEvaluator()
    limits = EvaluationLimits(
        max_scan_chars=10,
        match_window_chars=5,
        max_fallback_scan_chars=100,
        max_replacements_per_pattern=10,
    )

    chunks = list(evaluator.iter_scan_chunks("x" * 25, limits))

    assert chunks[:3] == [(0, 10), (1, 11), (2, 12)]
    assert chunks[-1] == (15, 25)
    assert len(chunks) == 16


class _RecordingPattern:
    pattern = "never"

    def __init__(self):
        self.bounds = []

    def search(self, _text, *bounds):
        self.bounds.append(bounds)
        return None


class _RegexErrorPattern:
    pattern = "broken"

    def search(self, *_args, **_kwargs):
        raise re.error("broken")

    def finditer(self, *_args, **_kwargs):
        raise re.error("broken")

    def sub(self, *_args, **_kwargs):
        raise re.error("broken")

    def subn(self, *_args, **_kwargs):
        raise re.error("broken")


def test_direct_original_string_search_bounds_are_literal():
    evaluator = PolicyEvaluator()
    pattern = _RecordingPattern()

    assert (
        evaluator.find_match_span(
            pattern,
            "x" * 20,
            LIMITS,
        )
        is None
    )
    assert pattern.bounds == [
        (0, 15),
        (1, 16),
        (2, 17),
        (3, 18),
        (4, 19),
        (5, 20),
        (6, 20),
        (7, 20),
        (8, 20),
        (9, 20),
        (10, 20),
        (),
    ]


def test_direct_lookbehind_anchor_and_fallback_behavior_is_literal():
    evaluator = PolicyEvaluator()
    lookbehind_limits = EvaluationLimits(1, 5, 1, 10)
    anchor_limits = EvaluationLimits(2, 10, 100, 10)

    assert evaluator.find_match_span(
        re.compile(r"(?<=A)needle"),
        ("x" * 9) + "Aneedle",
        lookbehind_limits,
    ) == (10, 16)
    assert (
        evaluator.find_match_span(
            re.compile(r"^needle"),
            "xxneedle",
            anchor_limits,
        )
        is None
    )
    assert evaluator.find_match_span(
        re.compile(r"^needle$"),
        "needle",
        EvaluationLimits(3, 0, 6, 10),
    ) == (0, 6)
    assert (
        evaluator.find_match_span(
            re.compile(r"^needle$"),
            "needle",
            EvaluationLimits(3, 0, 5, 10),
        )
        is None
    )


@pytest.mark.parametrize(
    ("raw", "error_type"),
    [
        (None, TypeError),
        ("bad", ValueError),
    ],
)
def test_direct_max_scan_coercion_errors_are_literal(raw, error_type):
    limits = EvaluationLimits(
        max_scan_chars=raw,  # type: ignore[arg-type]
        match_window_chars=5,
        max_fallback_scan_chars=100,
        max_replacements_per_pattern=10,
    )
    evaluator = PolicyEvaluator()

    with pytest.raises(error_type):
        evaluator.find_match_span(re.compile("x"), "x", limits)
    with pytest.raises(error_type):
        list(evaluator.iter_scan_chunks("x", limits))


@pytest.mark.parametrize(
    ("field", "raw", "error_type"),
    [
        ("match_window_chars", None, TypeError),
        ("match_window_chars", "bad", ValueError),
        ("max_fallback_scan_chars", None, TypeError),
        ("max_fallback_scan_chars", "bad", ValueError),
    ],
)
def test_direct_long_limit_coercion_errors_are_literal(
    field,
    raw,
    error_type,
):
    values = {
        "max_scan_chars": 1,
        "match_window_chars": 2,
        "max_fallback_scan_chars": 20,
        "max_replacements_per_pattern": 10,
    }
    values[field] = raw
    limits = EvaluationLimits(**values)  # type: ignore[arg-type]

    with pytest.raises(error_type):
        PolicyEvaluator().find_match_span(
            re.compile("never"),
            "long text",
            limits,
        )


def test_direct_numeric_string_limits_are_coerced_for_evaluation():
    limits = EvaluationLimits(
        max_scan_chars="1",  # type: ignore[arg-type]
        match_window_chars="2",  # type: ignore[arg-type]
        max_fallback_scan_chars="20",  # type: ignore[arg-type]
        max_replacements_per_pattern=10,
    )

    assert (
        PolicyEvaluator().find_match_span(
            re.compile("never"),
            "long text",
            limits,
        )
        is None
    )


def test_direct_malformed_raw_rule_exception_propagates():
    policy = _policy(None)

    with pytest.raises(AttributeError):
        PolicyEvaluator().evaluate_text(
            "secret",
            policy,
            "input",
            LIMITS,
            include_redacted_text=False,
        )


def test_direct_empty_text_and_regex_error_behavior_are_literal():
    evaluator = PolicyEvaluator()
    policy = _policy(_rule("secret"))
    regex_error_policy = _policy(_RegexErrorPattern())

    assert (
        evaluator.evaluate_text(
            "",
            policy,
            "input",
            LIMITS,
            include_redacted_text=False,
        )
        == ModerationEvaluationResult()
    )
    assert (
        evaluator.evaluate_text(
            "secret",
            regex_error_policy,
            "input",
            LIMITS,
            include_redacted_text=False,
        )
        == ModerationEvaluationResult()
    )


def test_direct_evaluator_does_not_mutate_borrowed_inputs():
    categories = {"confidential", "pii"}
    rule = _rule(
        "secret",
        action="warn",
        replacement="[SECRET]",
        categories=categories,
        phase="input",
    )
    second_categories = {"secondary"}
    second_rule = _rule(
        "token",
        action="block",
        replacement="[TOKEN]",
        categories=second_categories,
        phase="both",
    )
    rules = [rule, second_rule]
    enabled_categories = {"confidential", "pii", "secondary"}
    policy = _policy(*rules, categories_enabled=enabled_categories)
    pattern_collection = policy.block_patterns
    ordered_rules = tuple(pattern_collection)
    enabled_category_values = enabled_categories.copy()
    rule_snapshots = tuple(
        (
            candidate.regex,
            candidate.action,
            candidate.replacement,
            candidate.phase,
            candidate.categories,
            candidate.categories.copy() if candidate.categories is not None else None,
        )
        for candidate in rules
    )
    policy_scalar_snapshot = (
        policy.enabled,
        policy.input_enabled,
        policy.output_enabled,
        policy.input_action,
        policy.output_action,
        policy.redact_replacement,
        policy.per_user_overrides,
    )
    limit_values = vars(LIMITS).copy()
    limit_identities = tuple(
        (field, getattr(LIMITS, field))
        for field in (
            "max_scan_chars",
            "match_window_chars",
            "max_fallback_scan_chars",
            "max_replacements_per_pattern",
        )
    )

    PolicyEvaluator().evaluate_text(
        "secret token",
        policy,
        "input",
        LIMITS,
        include_redacted_text=False,
    )

    assert policy.categories_enabled is enabled_categories
    assert policy.categories_enabled == enabled_category_values
    assert policy.block_patterns is pattern_collection
    assert len(policy.block_patterns) == len(ordered_rules)
    assert all(
        current is original
        for current, original in zip(
            policy.block_patterns,
            ordered_rules,
            strict=True,
        )
    )
    for current, snapshot in zip(
        policy.block_patterns,
        rule_snapshots,
        strict=True,
    ):
        regex, action, replacement, phase, category_collection, category_values = snapshot
        assert current.regex is regex
        assert current.action == action
        assert current.replacement == replacement
        assert current.phase == phase
        assert current.categories is category_collection
        assert current.categories == category_values
    assert (
        policy.enabled,
        policy.input_enabled,
        policy.output_enabled,
        policy.input_action,
        policy.output_action,
        policy.redact_replacement,
        policy.per_user_overrides,
    ) == policy_scalar_snapshot
    assert vars(LIMITS) == limit_values
    assert all(getattr(LIMITS, field) is original for field, original in limit_identities)
