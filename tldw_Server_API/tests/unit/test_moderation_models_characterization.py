from __future__ import annotations

import dataclasses
import inspect
import re
import typing
from dataclasses import fields

import pytest

from tldw_Server_API.app.core.Moderation.moderation_service import (
    ModerationEvaluationResult,
    ModerationPolicy,
    PatternRule,
)
from tldw_Server_API.app.core.Moderation.policy_compiler import (
    PolicyCompilationInput,
    PolicyCompiler,
    ResolvedModerationConfig,
)
from tldw_Server_API.app.core.Moderation.policy_evaluator import (
    EvaluationLimits,
    PolicyEvaluator,
)

pytestmark = pytest.mark.unit

_LIMITS = EvaluationLimits(
    max_scan_chars=100,
    match_window_chars=16,
    max_fallback_scan_chars=200,
    max_replacements_per_pattern=10,
)


class _BrokenPattern:
    @property
    def pattern(self):
        raise ValueError("broken pattern")


class _UnexpectedPattern:
    @property
    def pattern(self):
        raise ZeroDivisionError("unexpected pattern failure")


@pytest.mark.parametrize(
    ("model_type", "expected_signature", "expected_fields", "expected_annotations"),
    [
        (
            ModerationPolicy,
            "(enabled: 'bool' = False, input_enabled: 'bool' = True, "
            "output_enabled: 'bool' = True, input_action: 'str' = 'block', "
            "output_action: 'str' = 'redact', redact_replacement: 'str' = "
            "'[REDACTED]', per_user_overrides: 'bool' = True, "
            "block_patterns: 'list[PatternRule]' = <factory>, "
            "categories_enabled: 'set[str] | None' = None) -> None",
            (
                "enabled",
                "input_enabled",
                "output_enabled",
                "input_action",
                "output_action",
                "redact_replacement",
                "per_user_overrides",
                "block_patterns",
                "categories_enabled",
            ),
            {
                "enabled": "bool",
                "input_enabled": "bool",
                "output_enabled": "bool",
                "input_action": "str",
                "output_action": "str",
                "redact_replacement": "str",
                "per_user_overrides": "bool",
                "block_patterns": "list[PatternRule]",
                "categories_enabled": "set[str] | None",
            },
        ),
        (
            PatternRule,
            "(regex: 're.Pattern', action: 'str | None' = None, "
            "replacement: 'str | None' = None, categories: 'set[str] | None' "
            "= None, phase: 'str' = 'both') -> None",
            ("regex", "action", "replacement", "categories", "phase"),
            {
                "regex": "re.Pattern",
                "action": "str | None",
                "replacement": "str | None",
                "categories": "set[str] | None",
                "phase": "str",
            },
        ),
        (
            ModerationEvaluationResult,
            "(action: 'str' = 'pass', redacted_text: 'str | None' = None, "
            "matched_pattern: 'str | None' = None, category: 'str | None' "
            "= None, match_span: 'tuple[int, int] | None' = None, sample: "
            "'str | None' = None) -> None",
            (
                "action",
                "redacted_text",
                "matched_pattern",
                "category",
                "match_span",
                "sample",
            ),
            {
                "action": "str",
                "redacted_text": "str | None",
                "matched_pattern": "str | None",
                "category": "str | None",
                "match_span": "tuple[int, int] | None",
                "sample": "str | None",
            },
        ),
    ],
)
def test_model_declarations_are_literal(
    model_type,
    expected_signature,
    expected_fields,
    expected_annotations,
):
    assert str(inspect.signature(model_type)) == expected_signature
    assert tuple(field.name for field in fields(model_type)) == expected_fields
    assert model_type.__annotations__ == expected_annotations


def test_model_defaults_and_borrowed_values_are_literal():
    first = ModerationPolicy()
    second = ModerationPolicy()
    supplied_rules = []
    supplied_categories = {"pii"}
    regex = re.compile("secret")
    rule_categories = {"confidential"}
    rule = PatternRule(regex=regex, categories=rule_categories)
    policy = ModerationPolicy(
        block_patterns=supplied_rules,
        categories_enabled=supplied_categories,
    )

    assert first == second
    assert first.block_patterns == []
    assert first.block_patterns is not second.block_patterns
    assert policy.block_patterns is supplied_rules
    assert policy.categories_enabled is supplied_categories
    assert rule.regex is regex
    assert rule.categories is rule_categories

    result = ModerationEvaluationResult()
    result.action = "warn"
    assert result.action == "warn"


def test_policy_block_patterns_uses_list_default_factory():
    block_patterns = next(field for field in fields(ModerationPolicy) if field.name == "block_patterns")

    assert block_patterns.default is dataclasses.MISSING
    assert block_patterns.default_factory is list


def test_resolved_model_type_hints_are_literal():
    policy_hints = typing.get_type_hints(ModerationPolicy)
    rule_hints = typing.get_type_hints(PatternRule)
    result_hints = typing.get_type_hints(ModerationEvaluationResult)

    assert policy_hints["block_patterns"] == list[PatternRule]
    assert policy_hints["categories_enabled"] == set[str] | None
    assert rule_hints["regex"] is re.Pattern
    assert rule_hints["categories"] == set[str] | None
    assert result_hints["match_span"] == tuple[int, int] | None


def test_policy_to_dict_returns_literal_mapping():
    policy = ModerationPolicy(
        enabled=True,
        block_patterns=[
            PatternRule(
                regex=re.compile("secret"),
                action="redact",
                replacement="[RULE]",
                categories=None,
                phase="input",
            )
        ],
        categories_enabled={"pii", "confidential"},
    )

    assert policy.to_dict() == {
        "enabled": True,
        "input_enabled": True,
        "output_enabled": True,
        "input_action": "block",
        "output_action": "redact",
        "redact_replacement": "[REDACTED]",
        "per_user_overrides": True,
        "blocklist_count": 1,
        "block_patterns": ["secret"],
        "rules": [
            {
                "pattern": "secret",
                "action": "redact",
                "replacement": "[RULE]",
                "phase": "input",
                "categories": "uncategorized",
            }
        ],
        "categories_enabled": ["confidential", "pii"],
    }


def test_policy_to_dict_preserves_legacy_regex_shape():
    policy = ModerationPolicy(block_patterns=[re.compile("legacy")])

    snapshot = policy.to_dict()

    assert snapshot["block_patterns"] == ["legacy"]
    assert snapshot["rules"] == [
        {
            "pattern": "legacy",
            "action": "",
            "replacement": "",
            "phase": "both",
            "categories": "",
        }
    ]


def test_policy_to_dict_preserves_noncritical_fallbacks():
    policy = ModerationPolicy(block_patterns=[_BrokenPattern()])

    snapshot = policy.to_dict()

    assert snapshot["blocklist_count"] == 0
    assert snapshot["block_patterns"] == []
    assert snapshot["rules"] == []


def test_policy_to_dict_does_not_swallow_unlisted_exceptions():
    policy = ModerationPolicy(block_patterns=[_UnexpectedPattern()])

    with pytest.raises(ZeroDivisionError, match="unexpected pattern failure"):
        policy.to_dict()


def test_policy_type_descriptors_and_tuples_are_literal():
    assert isinstance(inspect.getattr_static(PolicyCompiler, "policy_types"), staticmethod)
    assert isinstance(inspect.getattr_static(PolicyEvaluator, "policy_types"), staticmethod)
    assert str(inspect.signature(PolicyCompiler.policy_types)) == (
        "() -> 'tuple[type[ModerationPolicy], type[PatternRule]]'"
    )
    assert str(inspect.signature(PolicyEvaluator.policy_types)) == (
        "() -> 'tuple[type[ModerationPolicy], type[PatternRule], " "type[ModerationEvaluationResult]]'"
    )
    assert PolicyCompiler.policy_types() == (ModerationPolicy, PatternRule)
    assert PolicyEvaluator.policy_types() == (
        ModerationPolicy,
        PatternRule,
        ModerationEvaluationResult,
    )


def test_compiler_uses_overridden_policy_types():
    class ReplacementPolicy:
        def __init__(self, **values):
            self.values = values

    class ReplacementCompiler(PolicyCompiler):
        @staticmethod
        def policy_types():
            return ReplacementPolicy, PatternRule

    result = ReplacementCompiler().compile_global(
        PolicyCompilationInput(
            config=ResolvedModerationConfig(),
            runtime_override={},
            blocklist_lines=[],
            pii_rules=[],
        )
    )

    assert isinstance(result.policy, ReplacementPolicy)
    assert result.policy.values["block_patterns"] == []


def test_evaluator_uses_overridden_policy_types():
    class ReplacementResult:
        pass

    class ReplacementEvaluator(PolicyEvaluator):
        @staticmethod
        def policy_types():
            return ModerationPolicy, PatternRule, ReplacementResult

    result = ReplacementEvaluator().evaluate_text(
        "",
        ModerationPolicy(enabled=False),
        "input",
        _LIMITS,
        include_redacted_text=False,
    )

    assert isinstance(result, ReplacementResult)
