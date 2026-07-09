import re

import pytest

from tldw_Server_API.app.core.Moderation.moderation_service import (
    ModerationPolicy,
    ModerationService,
    PatternRule,
)
from tldw_Server_API.app.core.Moderation.policy_compiler import (
    PolicyCompilationInput,
    PolicyCompiler,
    ResolvedModerationConfig,
)

pytestmark = pytest.mark.unit


def _config(**overrides):
    values = {
        "enabled": True,
        "input_enabled": True,
        "output_enabled": True,
        "input_action": "block",
        "output_action": "redact",
        "redact_replacement": "[REDACTED]",
        "per_user_overrides": True,
        "categories_enabled": None,
        "pii_enabled": False,
    }
    values.update(overrides)
    return ResolvedModerationConfig(**values)


def test_compile_global_policy_uses_resolved_defaults():
    result = PolicyCompiler().compile_global(
        PolicyCompilationInput(
            config=_config(categories_enabled={"pii", "confidential"}),
            runtime_override={},
            blocklist_lines=[],
            pii_rules=[],
        )
    )

    assert isinstance(result.policy, ModerationPolicy)
    assert result.policy.enabled is True
    assert result.policy.input_action == "block"
    assert result.policy.output_action == "redact"
    assert result.policy.categories_enabled == {"pii", "confidential"}
    assert result.report.issues == []


def test_compile_global_policy_copies_pii_rules_when_enabled():
    pii_rule = PatternRule(
        regex=re.compile("email", re.IGNORECASE),
        action="redact",
        replacement="[PII]",
        categories={"pii", "pii_email"},
    )

    result = PolicyCompiler().compile_global(
        PolicyCompilationInput(
            config=_config(pii_enabled=True),
            runtime_override={},
            blocklist_lines=[],
            pii_rules=[pii_rule],
        )
    )

    assert result.policy.block_patterns == [pii_rule]


def test_compile_global_policy_compiles_literal_and_regex_blocklist_rules():
    result = PolicyCompiler().compile_global(
        PolicyCompilationInput(
            config=_config(),
            runtime_override={},
            blocklist_lines=[
                "secret -> block #confidential",
                r"/leak\d+/i -> redact:[MASK] #pii",
            ],
            pii_rules=[],
        )
    )

    rules = result.policy.block_patterns
    assert len(rules) == 2
    assert rules[0].regex.search("SECRET")
    assert rules[0].action == "block"
    assert rules[0].categories == {"confidential"}
    assert rules[1].regex.search("leak123")
    assert rules[1].action == "redact"
    assert rules[1].replacement == "[MASK]"
    assert rules[1].categories == {"pii"}


def test_compile_global_policy_reports_invalid_lines_without_raw_regex():
    result = PolicyCompiler().compile_global(
        PolicyCompilationInput(
            config=_config(),
            runtime_override={},
            blocklist_lines=[
                "secret -> invalid_action",
                "/(a+)+$/ -> block",
                "/(unclosed/ -> block",
            ],
            pii_rules=[],
        )
    )

    assert result.policy.block_patterns == []
    reasons = [issue.reason for issue in result.report.issues]
    assert reasons == ["invalid_action", "dangerous_regex", "invalid_regex"]
    rendered = repr(result.report.issues)
    assert "(a+)+$" not in rendered
    assert "(unclosed" not in rendered


def test_nested_quantifier_scanner_flags_nested_repetition_only():
    compiler = PolicyCompiler()

    assert compiler.has_nested_quantifiers("(a+)+")
    assert compiler.has_nested_quantifiers(r"(foo\\*)*") is True
    assert compiler.has_nested_quantifiers(r"(foo\\\*)*") is False
    assert compiler.has_nested_quantifiers("(foo)+") is False


def test_compile_global_policy_reports_empty_pattern_after_parsing():
    result = PolicyCompiler().compile_global(
        PolicyCompilationInput(
            config=_config(),
            runtime_override={},
            blocklist_lines=["-> block"],
            pii_rules=[],
        )
    )

    assert result.policy.block_patterns == []
    reasons = [issue.reason for issue in result.report.issues]
    assert reasons == ["empty_pattern"]


def test_compile_global_policy_preserves_raw_regex_backslashes():
    result = PolicyCompiler().compile_global(
        PolicyCompilationInput(
            config=_config(),
            runtime_override={},
            blocklist_lines=[
                r"/leak\d+/ -> block",
                r"/leak\\d+/ -> warn",
            ],
            pii_rules=[],
        )
    )

    rules = result.policy.block_patterns
    assert len(rules) == 2
    assert rules[0].regex.search("leak123")
    assert not rules[0].regex.search(r"leak\d")
    assert not rules[1].regex.search("leak123")
    assert rules[1].regex.search(r"leak\d")


def test_service_global_policy_uses_compiler_without_leaking_paths(tmp_path, monkeypatch):
    blocklist = tmp_path / "blocklist.txt"
    blocklist.write_text("secret -> block #confidential\n", encoding="utf-8")

    svc = ModerationService()
    svc._blocklist_path = str(blocklist)
    svc._runtime_override = {}
    svc._policy_compiler = PolicyCompiler()

    policy = svc._compile_global_policy_from_resolved_config(
        ResolvedModerationConfig(
            enabled=True,
            input_enabled=True,
            output_enabled=True,
            input_action="block",
            output_action="redact",
            redact_replacement="[REDACTED]",
            per_user_overrides=True,
            categories_enabled=None,
            pii_enabled=False,
        )
    )

    assert policy.enabled is True
    assert len(policy.block_patterns) == 1
    assert policy.block_patterns[0].categories == {"confidential"}


def test_service_resolved_config_falls_back_to_env_when_categories_are_none(monkeypatch):
    monkeypatch.setenv("MODERATION_CATEGORIES_ENABLED", "pii, confidential")
    svc = ModerationService()
    svc._runtime_override = {"categories_enabled": {"runtime"}}

    resolved = svc._resolve_moderation_config({"categories_enabled": None})

    assert resolved.compiler_config.categories_enabled == {"pii", "confidential"}


def test_compile_user_policy_preserves_empty_category_override():
    base = ModerationPolicy(
        enabled=True,
        input_enabled=True,
        output_enabled=True,
        input_action="block",
        output_action="redact",
        redact_replacement="[REDACTED]",
        per_user_overrides=True,
        block_patterns=[],
        categories_enabled={"pii"},
    )

    result = PolicyCompiler().compile_user_policy(
        base,
        {
            "enabled": True,
            "categories_enabled": "",
            "rules": [],
        },
    )

    assert result.policy.categories_enabled == set()


def test_compile_user_policy_adds_wildcard_quick_rules():
    base = ModerationPolicy(enabled=True, block_patterns=[], categories_enabled={"pii"})

    result = PolicyCompiler().compile_user_policy(
        base,
        {
            "rules": [
                {
                    "id": "r1",
                    "pattern": "secret",
                    "is_regex": False,
                    "action": "warn",
                    "phase": "input",
                }
            ]
        },
    )

    assert len(result.policy.block_patterns) == 1
    rule = result.policy.block_patterns[0]
    assert rule.regex.search("secret")
    assert rule.action == "warn"
    assert rule.categories == {"*"}
    assert rule.phase == "input"


def test_compile_user_policy_accepts_legacy_bool_like_is_regex_values():
    base = ModerationPolicy(enabled=True, block_patterns=[])

    result = PolicyCompiler().compile_user_policy(
        base,
        {
            "rules": [
                {
                    "id": "r1",
                    "pattern": r"token-\d+",
                    "is_regex": "yes",
                    "action": "block",
                    "phase": "input",
                },
                {
                    "id": "r2",
                    "pattern": "literal.*",
                    "is_regex": "false",
                    "action": "warn",
                    "phase": "output",
                },
            ]
        },
    )

    assert len(result.policy.block_patterns) == 2
    assert result.policy.block_patterns[0].regex.search("token-42")
    assert result.policy.block_patterns[1].regex.search("literal.*")
    assert not result.policy.block_patterns[1].regex.search("literalabc")
    assert result.report.issues == []


def test_compile_user_policy_without_override_returns_base_policy_without_issues():
    base = ModerationPolicy(enabled=True, categories_enabled={"pii"})

    result = PolicyCompiler().compile_user_policy(base, None)

    assert result.policy is base
    assert result.report.issues == []


def test_compile_user_policy_ignores_non_list_rules_without_issues():
    base = ModerationPolicy(enabled=True, block_patterns=[])

    result = PolicyCompiler().compile_user_policy(base, {"rules": {"pattern": "secret"}})

    assert result.policy.block_patterns == []
    assert result.report.issues == []


def test_compile_user_policy_invalid_categories_type_falls_back_to_default_categories():
    base = ModerationPolicy(enabled=True, categories_enabled={"pii"})

    result = PolicyCompiler().compile_user_policy(base, {"categories_enabled": 123})

    assert result.policy.categories_enabled == {"pii"}
    assert result.report.issues == []


def test_compile_user_policy_invalid_rule_phase_defaults_to_both():
    base = ModerationPolicy(enabled=True, block_patterns=[])

    result = PolicyCompiler().compile_user_policy(
        base,
        {
            "rules": [
                {
                    "id": "r1",
                    "pattern": "secret",
                    "is_regex": False,
                    "action": "warn",
                    "phase": "sideways",
                }
            ]
        },
    )

    assert len(result.policy.block_patterns) == 1
    assert result.policy.block_patterns[0].phase == "both"
    assert result.report.issues == []


def test_compile_user_policy_copies_default_categories_when_override_omits_categories():
    base = ModerationPolicy(enabled=True, categories_enabled={"pii"})

    result = PolicyCompiler().compile_user_policy(base, {"enabled": True})

    assert result.policy.categories_enabled == {"pii"}
    assert result.policy.categories_enabled is not None
    result.policy.categories_enabled.add("confidential")
    assert base.categories_enabled == {"pii"}
