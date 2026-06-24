import re

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
