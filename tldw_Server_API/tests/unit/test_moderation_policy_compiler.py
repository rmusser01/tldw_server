import re

from tldw_Server_API.app.core.Moderation.moderation_service import ModerationPolicy, PatternRule
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
