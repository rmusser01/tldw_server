"""Tests for MCP profile policy decision primitives."""

from __future__ import annotations

import pytest

from mcp_unified.profiles.decisions import (
    PolicyDecision,
    PolicyDecisionRule,
    PolicyDecisionSubject,
    PolicyMatchedRule,
    compile_profile_policy_rules,
    merge_policy_decisions,
)
from mcp_unified.profiles.models import ProfilePolicy


def test_policy_decision_defaults_for_ask() -> None:
    decision = PolicyDecision(
        outcome="ask",
        reason_code="approval_required",
        subject=PolicyDecisionSubject(type="tool", normalized="fs.patch"),
    )

    assert decision.visibility == "direct"
    assert decision.call_state == "approval_required"
    assert decision.requires_approval is True


def test_policy_decision_defaults_for_deny() -> None:
    decision = PolicyDecision(
        outcome="deny",
        reason_code="tool_denied",
        subject=PolicyDecisionSubject(type="tool", normalized="fs.write"),
    )

    assert decision.visibility == "hidden"
    assert decision.call_state == "blocked"
    assert decision.requires_approval is False


def test_policy_decision_defaults_for_allow() -> None:
    decision = PolicyDecision(
        outcome="allow",
        reason_code="allowed",
        subject=PolicyDecisionSubject(type="tool", normalized="fs.read"),
    )

    assert decision.visibility == "direct"
    assert decision.call_state == "callable"
    assert decision.requires_approval is False


def test_merge_policy_decisions_uses_deny_over_ask_over_allow() -> None:
    subject = PolicyDecisionSubject(type="tool", normalized="fs.write")
    merged = merge_policy_decisions(
        [
            PolicyDecision(
                outcome="allow",
                reason_code="allowed",
                subject=subject,
                matched_rules=[
                    PolicyMatchedRule(
                        source="allowed_tools",
                        rule_type="tool",
                        pattern="fs.*",
                        outcome="allow",
                    )
                ],
            ),
            PolicyDecision(
                outcome="ask",
                reason_code="approval_required",
                subject=subject,
                matched_rules=[
                    PolicyMatchedRule(
                        source="tool_rules",
                        rule_type="tool",
                        pattern="fs.write",
                        outcome="ask",
                    )
                ],
            ),
            PolicyDecision(
                outcome="deny",
                reason_code="tool_denied",
                subject=subject,
                matched_rules=[
                    PolicyMatchedRule(
                        source="denied_tools",
                        rule_type="tool",
                        pattern="fs.write",
                        outcome="deny",
                    )
                ],
            ),
        ],
        subject=subject,
    )

    assert merged.outcome == "deny"
    assert merged.reason_code == "tool_denied"
    assert merged.call_state == "blocked"
    assert [rule.outcome for rule in merged.matched_rules] == ["allow", "ask", "deny"]


def test_merge_policy_decisions_defaults_to_deny_without_matches() -> None:
    subject = PolicyDecisionSubject(type="tool", normalized="fs.write")
    merged = merge_policy_decisions([], subject=subject)

    assert merged.outcome == "deny"
    assert merged.reason_code == "no_matching_rule"
    assert merged.call_state == "blocked"


def test_merge_policy_decisions_keeps_first_same_precedence_reason() -> None:
    subject = PolicyDecisionSubject(type="tool", normalized="fs.write")
    merged = merge_policy_decisions(
        [
            PolicyDecision(outcome="ask", reason_code="profile_ask", subject=subject),
            PolicyDecision(outcome="ask", reason_code="workspace_ask", subject=subject),
        ],
        subject=subject,
    )

    assert merged.outcome == "ask"
    assert merged.reason_code == "profile_ask"
    assert merged.call_state == "approval_required"


def test_compile_profile_policy_rules_preserves_legacy_tool_fields() -> None:
    rules = compile_profile_policy_rules(
        ProfilePolicy(
            allowed_tools=["fs.read"],
            denied_tools=["fs.write"],
        )
    )

    assert [(rule.rule_type, rule.pattern, rule.outcome) for rule in rules] == [
        ("tool", "fs.write", "deny"),
        ("tool", "fs.read", "allow"),
    ]


def test_compile_profile_policy_rules_converts_legacy_bash_pattern_to_argv_rule() -> None:
    rules = compile_profile_policy_rules(ProfilePolicy(allowed_tools=["Bash(git *)"]))

    command_rule = rules[0]
    assert command_rule.rule_type == "command"
    assert command_rule.argv == ("git", "*")
    assert command_rule.outcome == "allow"


def test_compile_profile_policy_rules_rejects_broad_bash_pattern() -> None:
    with pytest.raises(ValueError, match="broad bash patterns are not allowed"):
        compile_profile_policy_rules(ProfilePolicy(allowed_tools=["Bash(*)"]))


@pytest.mark.parametrize(
    "pattern",
    [
        "Bash(* *)",
        "Bash(* --version)",
        "Bash(git* status)",
        "Bash(?sh -lc)",
        "Bash([gr]it status)",
    ],
)
def test_compile_profile_policy_rules_rejects_wildcard_executable_patterns(
    pattern: str,
) -> None:
    with pytest.raises(ValueError, match="command executable must be fixed"):
        compile_profile_policy_rules(ProfilePolicy(allowed_tools=[pattern]))


def test_compile_profile_policy_rules_rejects_empty_legacy_tool_pattern() -> None:
    with pytest.raises(ValueError, match="legacy tool policy patterns cannot be empty"):
        compile_profile_policy_rules(ProfilePolicy(allowed_tools=[""]))


def test_compile_profile_policy_rules_includes_structured_extra_rules() -> None:
    rules = compile_profile_policy_rules(
        {
            "tool_rules": [{"pattern": "fs.patch", "outcome": "ask"}],
            "command_rules": [{"argv": ["git", "status"], "outcome": "allow"}],
            "mcp_rules": [{"pattern": "mcp__github__*", "outcome": "deny"}],
        }
    )

    assert [(rule.rule_type, rule.pattern, rule.argv, rule.outcome) for rule in rules] == [
        ("tool", "fs.patch", None, "ask"),
        ("command", None, ("git", "status"), "allow"),
        ("mcp", "mcp__github__*", None, "deny"),
    ]


def test_compile_profile_policy_rules_accepts_structured_command_tuple_argv() -> None:
    rules = compile_profile_policy_rules({"command_rules": [{"argv": ("git", "status"), "outcome": "allow"}]})

    command_rule = rules[0]
    assert command_rule.rule_type == "command"
    assert command_rule.argv == ("git", "status")
    assert command_rule.outcome == "allow"


def test_compile_profile_policy_rules_rejects_structured_command_string_argv() -> None:
    with pytest.raises(ValueError, match="command policy rule argv must be a sequence"):
        compile_profile_policy_rules({"command_rules": [{"argv": "git status", "outcome": "allow"}]})


def test_compile_profile_policy_rules_rejects_structured_command_set_argv() -> None:
    with pytest.raises(ValueError, match="command policy rule argv must be a sequence"):
        compile_profile_policy_rules({"command_rules": [{"argv": {"git"}, "outcome": "allow"}]})


@pytest.mark.parametrize(
    "argv",
    [
        ["*", "--version"],
        ["git*", "status"],
        ["?sh", "-lc"],
        ["[gr]it", "status"],
    ],
)
def test_compile_profile_policy_rules_rejects_structured_wildcard_executables(
    argv: list[str],
) -> None:
    with pytest.raises(ValueError, match="command executable must be fixed"):
        compile_profile_policy_rules({"command_rules": [{"argv": argv, "outcome": "allow"}]})


def test_policy_decision_rule_rejects_command_wildcard_executable() -> None:
    with pytest.raises(ValueError, match="command executable must be fixed"):
        PolicyDecisionRule(
            rule_type="command",
            outcome="allow",
            source="precompiled",
            argv=("*", "--version"),
        )


@pytest.mark.parametrize("argv", [None, (), ("git", "")])
def test_policy_decision_rule_rejects_invalid_command_argv(
    argv: tuple[str, ...] | None,
) -> None:
    with pytest.raises(ValueError, match="command policy rule argv"):
        PolicyDecisionRule(
            rule_type="command",
            outcome="allow",
            source="precompiled",
            argv=argv,
        )


def test_compile_profile_policy_rules_revalidates_precompiled_command_rules() -> None:
    invalid_rule = PolicyDecisionRule.model_construct(
        rule_type="command",
        outcome="allow",
        source="precompiled",
        argv=("*", "--version"),
    )

    with pytest.raises(ValueError, match="command executable must be fixed"):
        compile_profile_policy_rules({"command_rules": [invalid_rule]})


def test_compile_profile_policy_rules_rejects_structured_bash_pattern_wildcard_executable() -> None:
    with pytest.raises(ValueError, match="command executable must be fixed"):
        compile_profile_policy_rules({"command_rules": [{"pattern": "Bash(* --version)", "outcome": "allow"}]})
