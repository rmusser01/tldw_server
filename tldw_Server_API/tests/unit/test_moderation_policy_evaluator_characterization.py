from __future__ import annotations

import re
import threading
from typing import Any

import pytest

from tldw_Server_API.app.core.Moderation.moderation_service import (
    ModerationEvaluationResult,
    ModerationPolicy,
    ModerationService,
    PatternRule,
)
from tldw_Server_API.app.core.Moderation.policy_compiler import PolicyCompiler
from tldw_Server_API.app.core.Moderation.policy_evaluator import PolicyEvaluator

pytestmark = pytest.mark.unit


def _service(
    *,
    max_scan_chars: Any = 200_000,
    match_window_chars: Any = 4_096,
    max_fallback_scan_chars: Any = 800_000,
    max_replacements_per_pattern: Any = 1_000,
    service_type: type[ModerationService] = ModerationService,
) -> ModerationService:
    service = service_type.__new__(service_type)
    service._lock = threading.RLock()
    service._max_scan_chars = max_scan_chars
    service._match_window_chars = match_window_chars
    service._max_fallback_scan_chars = max_fallback_scan_chars
    service._max_replacements_per_pattern = max_replacements_per_pattern
    service._policy_evaluator = PolicyEvaluator()
    return service


def _rule(
    pattern: str,
    *,
    action: Any = None,
    replacement: str | None = None,
    categories: set[str] | None = None,
    phase: str = "both",
) -> PatternRule:
    return PatternRule(
        regex=re.compile(pattern),
        action=action,
        replacement=replacement,
        categories=categories,
        phase=phase,
    )


def _policy(
    *rules: PatternRule | re.Pattern[str],
    enabled: bool = True,
    input_enabled: bool = True,
    output_enabled: bool = True,
    input_action: Any = "block",
    output_action: Any = "redact",
    replacement: str = "[REDACTED]",
    categories_enabled: set[str] | None = None,
) -> ModerationPolicy:
    return ModerationPolicy(
        enabled=enabled,
        input_enabled=input_enabled,
        output_enabled=output_enabled,
        input_action=input_action,
        output_action=output_action,
        redact_replacement=replacement,
        per_user_overrides=False,
        block_patterns=list(rules),
        categories_enabled=categories_enabled,
    )


@pytest.mark.parametrize(
    ("phase", "input_enabled", "output_enabled", "expected_action"),
    [
        ("input", True, True, "block"),
        ("output", True, True, "redact"),
        ("input", False, True, "pass"),
        ("output", True, False, "pass"),
        (None, True, True, "warn"),
        ("unknown", True, True, "warn"),
    ],
)
def test_service_evaluate_text_phase_characterization(
    phase,
    input_enabled,
    output_enabled,
    expected_action,
):
    policy = _policy(
        _rule("secret", phase="both"),
        input_enabled=input_enabled,
        output_enabled=output_enabled,
    )

    result = _service().evaluate_text("contains secret", policy, phase)

    assert result.action == expected_action
    assert result.matched_pattern == ("secret" if expected_action != "pass" else None)


@pytest.mark.parametrize("phase", [None, "unknown"])
@pytest.mark.parametrize("rule_phase", ["input", "output"])
def test_unknown_phase_bypasses_rule_phase_metadata(phase, rule_phase):
    result = _service().evaluate_text(
        "secret",
        _policy(_rule("secret", phase=rule_phase)),
        phase,
    )

    assert result.action == "warn"
    assert result.matched_pattern == "secret"


def test_raw_regex_bypasses_phase_and_category_metadata():
    policy = _policy(
        re.compile("secret"),
        categories_enabled={"allowed-only"},
    )

    result = _service().evaluate_text("secret", policy, "input")

    assert result.action == "block"
    assert result.category is None
    assert result.match_span == (0, 6)


def test_action_rank_then_position_then_rule_order_is_literal():
    policy = _policy(
        _rule("later", action="warn", categories={"zeta"}),
        _rule("early", action="block", categories={"pii", "confidential"}),
        _rule("early", action="block", categories={"other"}),
    )

    result = _service().evaluate_text("early ... later", policy, "input")

    assert result == ModerationEvaluationResult(
        action="block",
        redacted_text=None,
        matched_pattern="early",
        category="confidential",
        match_span=(0, 5),
        sample="[REDACTED] ... later",
    )


def test_equal_rank_prefers_earliest_match():
    policy = _policy(
        _rule("later", action="block"),
        _rule("early", action="block"),
    )

    result = _service().evaluate_text("early then later", policy, "input")

    assert result.matched_pattern == "early"
    assert result.match_span == (0, 5)


def test_redact_outranks_an_earlier_warn_match():
    result = _service().evaluate_text(
        "warn first, redact later",
        _policy(
            _rule("warn", action="warn"),
            _rule("later", action="redact"),
        ),
        "input",
    )

    assert result.action == "redact"
    assert result.matched_pattern == "later"
    assert result.match_span == (19, 24)


def test_uncategorized_and_wildcard_category_behavior_is_literal():
    uncategorized = _policy(
        _rule("first", action="warn"),
        categories_enabled={"uncategorized"},
    )
    wildcard = _policy(
        _rule("second", action="warn", categories={"*"}),
        categories_enabled={"restricted"},
    )

    assert _service().evaluate_text("first", uncategorized, "input").category == "uncategorized"
    assert _service().evaluate_text("second", wildcard, "input").action == "warn"


def test_enabled_category_wildcard_allows_specific_rule_category():
    result = _service().evaluate_text(
        "secret",
        _policy(
            _rule(
                "secret",
                action="warn",
                categories={"restricted"},
            ),
            categories_enabled={"*"},
        ),
        "input",
    )

    assert result.action == "warn"
    assert result.category is None


@pytest.mark.parametrize("categories_enabled", [None, set()])
def test_falsy_category_filters_allow_all_rules(categories_enabled):
    result = _service().evaluate_text(
        "secret",
        _policy(
            _rule(
                "secret",
                action="warn",
                categories={"confidential"},
            ),
            categories_enabled=categories_enabled,
        ),
        "input",
    )

    assert result.action == "warn"
    assert result.category == "confidential"


def test_enabled_categories_intersect_before_lexical_selection():
    result = _service().evaluate_text(
        "secret",
        _policy(
            _rule(
                "secret",
                action="warn",
                categories={"pii", "confidential", "financial"},
            ),
            categories_enabled={"pii", "financial"},
        ),
        "input",
    )

    assert result.category == "financial"


def test_disabled_policy_passes_but_direct_redaction_still_applies():
    policy = _policy(
        _rule("secret", action="redact", replacement="[R]"),
        enabled=False,
    )
    service = _service()

    assert service.evaluate_text("secret", policy, "input") == (ModerationEvaluationResult())
    assert service.redact_text("secret", policy, "input") == "[R]"


class _MissingLower:
    pass


class _LowerBytes:
    def lower(self) -> bytes:
        return b"block"


class _LowerBlock:
    def lower(self) -> str:
        return "block"


class _LowerUnhashable:
    def lower(self) -> list[object]:
        return []


@pytest.mark.parametrize(
    ("action", "expected_action"),
    [
        (None, "block"),
        ("", "block"),
        ("unsupported", "warn"),
    ],
)
def test_falsy_and_unsupported_string_actions_are_literal(
    action,
    expected_action,
):
    result = _service().evaluate_text(
        "secret",
        _policy(_rule("secret", action=action)),
        "input",
    )

    assert result.action == expected_action


def test_effective_action_lower_result_behavior_is_literal():
    service = _service()

    bytes_result = service.evaluate_text(
        "secret",
        _policy(_rule("secret", action=_LowerBytes())),
        "input",
    )
    block_result = service.evaluate_text(
        "secret",
        _policy(_rule("secret", action=_LowerBlock())),
        "input",
    )

    assert bytes_result.action == "warn"
    assert block_result.action == "block"

    with pytest.raises(AttributeError):
        service.evaluate_text(
            "secret",
            _policy(_rule("secret", action=_MissingLower())),
            "input",
        )

    with pytest.raises(TypeError):
        service.evaluate_text(
            "secret",
            _policy(_rule("secret", action=_LowerUnhashable())),
            "input",
        )


def test_check_and_evaluate_dispatch_through_evaluate_text_core():
    calls = []

    class _DispatchService(ModerationService):
        def _evaluate_text_core(
            self,
            text: str,
            policy: ModerationPolicy,
            phase: str | None,
            *,
            include_redacted_text: bool,
        ) -> ModerationEvaluationResult:
            calls.append((text, phase, include_redacted_text))
            return ModerationEvaluationResult(action="warn", sample="[SAFE]")

    service = _DispatchService.__new__(_DispatchService)
    policy = _policy()

    assert service.check_text("probe", policy, "input") == (True, "[SAFE]")
    assert service.evaluate_text("probe", policy, "output").action == "warn"
    assert calls == [
        ("probe", "input", False),
        ("probe", "output", True),
    ]


def test_action_wrappers_dispatch_through_public_evaluate_text():
    calls = []

    class _DispatchService(ModerationService):
        def evaluate_text(
            self,
            text: str,
            policy: ModerationPolicy,
            phase: str | None = None,
        ) -> ModerationEvaluationResult:
            calls.append((text, phase))
            return ModerationEvaluationResult(
                action="redact",
                redacted_text="[R]",
                matched_pattern="secret",
                category="confidential",
                match_span=(0, 6),
            )

    service = _DispatchService.__new__(_DispatchService)
    policy = _policy()

    assert service._evaluate_action_internal("secret", policy, "input") == (
        "redact",
        "[R]",
        "secret",
        "confidential",
        (0, 6),
    )
    assert service.evaluate_action("secret", policy, "input") == ("redact", "[R]", "secret", "confidential")
    assert service.evaluate_action_with_match("secret", policy, "input") == (
        "redact",
        "[R]",
        "secret",
        "confidential",
        (0, 6),
    )
    assert calls == [
        ("secret", "input"),
        ("secret", "input"),
        ("secret", "input"),
    ]


def test_check_and_decision_only_core_do_not_invoke_public_redaction():
    class _NoRedactionService(ModerationService):
        def redact_text(
            self,
            text: str,
            policy: ModerationPolicy,
            phase: str | None = None,
        ) -> str:
            raise AssertionError("redaction must not run")

    service = _service(service_type=_NoRedactionService)
    policy = _policy(_rule("secret", action="redact", replacement="[R]"))

    assert service.check_text("secret", policy, "input") == (True, "[R]")
    decision = service._evaluate_text_core(
        "secret",
        policy,
        "input",
        include_redacted_text=False,
    )

    assert decision.action == "redact"
    assert decision.redacted_text is None


def test_evaluation_dispatches_through_public_redact_text():
    class _DispatchService(ModerationService):
        def redact_text(
            self,
            text: str,
            policy: ModerationPolicy,
            phase: str | None = None,
        ) -> str:
            return "[PUBLIC REDACTION]"

    service = _service(service_type=_DispatchService)

    result = service.evaluate_text(
        "secret",
        _policy(_rule("secret", action="redact")),
        "input",
    )

    assert result.redacted_text == "[PUBLIC REDACTION]"


def test_public_snippet_uses_first_matching_rule_replacement():
    policy = _policy(
        _rule("secret", replacement="[FIRST]"),
        _rule("secret", replacement="[SECOND]"),
        replacement="[POLICY]",
    )

    assert (
        _service().build_sanitized_snippet(
            "before secret after",
            policy,
            (7, 13),
            pattern="secret",
        )
        == "before [FIRST] after"
    )


def test_effective_policy_identity_and_rule_aliasing_are_unchanged():
    base_rule = _rule("secret")
    policy = _policy(base_rule)
    policy.per_user_overrides = True
    service = _service()
    service._global_policy = policy
    service._user_overrides = {}

    assert service.get_effective_policy("user-1") is policy

    service._policy_compiler = PolicyCompiler()
    service._user_overrides = {
        "user-1": {"input_action": "warn"},
    }
    overlaid = service.get_effective_policy("user-1")

    assert overlaid is not policy
    assert overlaid.block_patterns is not policy.block_patterns
    assert overlaid.block_patterns[0] is base_rule


@pytest.mark.parametrize("action", ["warn", "block", "redact"])
def test_direct_redaction_ignores_policy_enabled_and_rule_action(action):
    policy = _policy(
        _rule("secret", action=action, replacement="[RULE]"),
        enabled=False,
    )

    redacted, count = _service().redact_text_with_count(
        "secret and secret",
        policy,
        "input",
    )

    assert redacted == "[RULE] and [RULE]"
    assert count == 2


def test_sequential_redaction_applies_later_rules_to_changed_text():
    policy = _policy(
        _rule("secret", action="warn", replacement="token"),
        _rule("token", action="block", replacement="[FINAL]"),
    )

    redacted, count = _service().redact_text_with_count("secret", policy)

    assert redacted == "[FINAL]"
    assert count == 2


def test_replacement_text_is_literal_not_a_backreference():
    policy = _policy(
        _rule(r"(secret)", action="redact", replacement=r"\1-literal"),
    )

    assert _service().redact_text("secret", policy) == r"\1-literal"


@pytest.mark.parametrize(
    ("limit", "expected_text", "expected_count"),
    [
        (None, "[R] [R] [R]", 3),
        (0, "[R] [R] [R]", 3),
        (-1, "[R] [R] [R]", 3),
        ("2", "[R] [R] x", 2),
        ("bad", "[R] [R] [R]", 3),
    ],
)
def test_short_redaction_replacement_limit_characterization(
    limit,
    expected_text,
    expected_count,
):
    service = _service(
        max_scan_chars=100,
        max_replacements_per_pattern=limit,
    )
    policy = _policy(_rule("x", replacement="[R]"))

    assert service.redact_text_with_count("x x x", policy) == (
        expected_text,
        expected_count,
    )


@pytest.mark.parametrize(
    ("limit", "expected_text", "expected_count"),
    [
        (None, "[R] [R] [R]", 3),
        (0, "[R] [R] [R]", 3),
        (-1, "[R] [R] [R]", 3),
        (2, "[R] [R] x", 2),
    ],
)
def test_long_redaction_supported_limit_characterization(
    limit,
    expected_text,
    expected_count,
):
    service = _service(
        max_scan_chars=3,
        max_replacements_per_pattern=limit,
    )
    policy = _policy(_rule("x", replacement="[R]"))

    assert service.redact_text_with_count("x x x", policy) == (
        expected_text,
        expected_count,
    )


@pytest.mark.parametrize(
    ("limit", "error_type"),
    [
        ("2", TypeError),
        ("bad", ValueError),
    ],
)
@pytest.mark.parametrize(
    "method_name",
    ["redact_text", "redact_text_with_count"],
)
def test_long_redaction_unsupported_limit_exception_characterization(
    limit,
    error_type,
    method_name,
):
    service = _service(
        max_scan_chars=3,
        max_replacements_per_pattern=limit,
    )

    with pytest.raises(error_type):
        getattr(service, method_name)(
            "x x x",
            _policy(_rule("x", replacement="[R]")),
        )


@pytest.mark.parametrize("raw", [None, "2", "bad"])
def test_redaction_path_comparison_does_not_coerce_max_scan(raw):
    service = _service(max_scan_chars=raw)
    with pytest.raises(TypeError):
        service.redact_text("x", _policy(_rule("x")))


@pytest.mark.timeout(2)
def test_long_redaction_uses_bounded_full_text_finditer():
    service = _service(
        max_scan_chars=3,
        match_window_chars=0,
        max_replacements_per_pattern=10,
    )
    policy = _policy(_rule("ABCDE", replacement="[R]"))

    assert service.redact_text_with_count("xxABCDEyy", policy) == (
        "xx[R]yy",
        1,
    )


def test_zero_length_matches_differ_between_short_and_long_redaction():
    policy = _policy(_rule(r"(?=a)", replacement="[R]"))

    short = _service(max_scan_chars=10).redact_text_with_count("a", policy)
    long = _service(max_scan_chars=1).redact_text_with_count("aa", policy)

    assert short == ("[R]a", 1)
    assert long == ("aa", 0)


def test_malformed_raw_rule_exceptions_propagate():
    policy = _policy()
    policy.block_patterns = [None]  # type: ignore[list-item]
    service = _service()

    with pytest.raises(AttributeError):
        service.evaluate_text("secret", policy, "input")
    with pytest.raises(AttributeError):
        service.redact_text("secret", policy, "input")


class _RegexErrorPattern:
    pattern = "broken"

    def search(self, *_args: Any, **_kwargs: Any) -> Any:
        raise re.error("broken")

    def finditer(self, *_args: Any, **_kwargs: Any) -> Any:
        raise re.error("broken")

    def sub(self, *_args: Any, **_kwargs: Any) -> Any:
        raise re.error("broken")

    def subn(self, *_args: Any, **_kwargs: Any) -> Any:
        raise re.error("broken")


def test_regex_errors_keep_current_no_match_and_skip_behavior():
    policy = _policy()
    policy.block_patterns = [_RegexErrorPattern()]  # type: ignore[list-item]
    service = _service()

    assert service.evaluate_text("secret", policy, "input") == (ModerationEvaluationResult())
    assert service.redact_text("secret", policy, "input") == "secret"
    assert service.redact_text_with_count(
        "secret",
        policy,
        "input",
    ) == ("secret", 0)


def test_replacement_lookup_regex_error_remains_inside_rule_boundary():
    class _ReplacementErrorPolicy:
        block_patterns = [re.compile("secret")]
        input_enabled = True
        output_enabled = True
        categories_enabled = None

        @property
        def redact_replacement(self) -> str:
            raise re.error("replacement lookup failed")

    service = _service()
    policy = _ReplacementErrorPolicy()

    assert (
        service.redact_text(
            "secret",
            policy,  # type: ignore[arg-type]
        )
        == "secret"
    )
    assert service.redact_text_with_count(
        "secret",
        policy,  # type: ignore[arg-type]
    ) == ("secret", 0)


def test_service_evaluation_and_redaction_do_not_mutate_inputs():
    categories = {"confidential"}
    rule = _rule(
        "secret",
        action="redact",
        replacement="[R]",
        categories=categories,
    )
    second_rule = _rule(
        "token",
        action="warn",
        replacement="[TOKEN]",
        categories={"secondary"},
        phase="output",
    )
    rules = [rule, second_rule]
    enabled_categories = {"confidential", "secondary"}
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
    service = _service()

    service.evaluate_text("secret", policy, "input")
    service.redact_text("secret", policy, "input")

    assert policy.categories_enabled is enabled_categories
    assert policy.categories_enabled == enabled_category_values
    assert policy.block_patterns is pattern_collection
    assert len(policy.block_patterns) == len(ordered_rules)
    assert all(current is original for current, original in zip(policy.block_patterns, ordered_rules, strict=True))
    for current, snapshot in zip(policy.block_patterns, rule_snapshots, strict=True):
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
