"""Tests for Claude-style MCP profile permission rule parsing."""

from __future__ import annotations

import pytest
from mcp_unified.profiles import compile_profile_policy_rules
from mcp_unified.profiles.decisions import evaluate_profile_tool_decision
from mcp_unified.profiles.models import MCPProfile, ProfilePolicy
from mcp_unified.profiles.permission_rules import (
    compile_permission_rules,
    evaluate_permission_rule_decision,
    parse_permission_rule,
)


def test_parse_permission_rule_classifies_claude_style_subjects() -> None:
    command_rule = parse_permission_rule("Bash(git *)", outcome="allow")
    path_rule = parse_permission_rule("Read(/docs/**)", outcome="allow")
    edit_rule = parse_permission_rule("Edit(src/*.py)", outcome="ask")
    domain_rule = parse_permission_rule("WebFetch(https://example.com/docs)", outcome="ask")
    skill_rule = parse_permission_rule("Skill(review)", outcome="allow")
    agent_rule = parse_permission_rule("Agent(backend-engineer)", outcome="ask")
    mcp_rule = parse_permission_rule("mcp__github__*", outcome="deny")

    assert (command_rule.rule_type, command_rule.pattern, command_rule.argv) == (
        "command",
        "Bash(git *)",
        ("git", "*"),
    )
    assert (path_rule.rule_type, path_rule.pattern, path_rule.outcome) == ("path", "/docs/**", "allow")
    assert (edit_rule.rule_type, edit_rule.pattern, edit_rule.outcome) == ("path", "src/*.py", "ask")
    assert (domain_rule.rule_type, domain_rule.pattern, domain_rule.outcome) == (
        "domain",
        "example.com",
        "ask",
    )
    assert (skill_rule.rule_type, skill_rule.pattern, skill_rule.outcome) == ("skill", "review", "allow")
    assert (agent_rule.rule_type, agent_rule.pattern, agent_rule.outcome) == ("agent", "backend-engineer", "ask")
    assert (mcp_rule.rule_type, mcp_rule.pattern, mcp_rule.outcome) == ("mcp", "mcp__github__*", "deny")


@pytest.mark.parametrize(
    "pattern,match",
    [
        ("Bash(*)", "broad bash patterns are not allowed"),
        ("Shell(*)", "broad command patterns are not allowed"),
        ("PowerShell(*)", "broad command patterns are not allowed"),
        ("Bash(git status && rm -rf build)", "unsupported shell control token"),
        ("Bash(git status; rm -rf build)", "unsupported shell control token"),
        ("Bash(git status | cat)", "unsupported shell control token"),
        ("Read()", "permission rule specifier cannot be empty"),
        ("WebFetch()", "permission rule specifier cannot be empty"),
        ("Tool()", "permission rule specifier cannot be empty"),
        ("Bash(* --version)", "command executable must be fixed"),
    ],
)
def test_parse_permission_rule_rejects_unsafe_or_empty_patterns(pattern: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_permission_rule(pattern, outcome="allow")


def test_compile_permission_rules_accepts_strings_and_structured_entries() -> None:
    rules = compile_permission_rules(
        {
            "permission_rules": [
                "Read(/docs/**)",
                {"pattern": "Bash(git *)", "outcome": "ask", "reason_code": "git_requires_review"},
                {"pattern": "mcp__github__create_issue", "effect": "allow"},
            ]
        }
    )

    assert [(rule.rule_type, rule.pattern, rule.argv, rule.outcome, rule.reason_code) for rule in rules] == [
        ("path", "/docs/**", None, "allow", None),
        ("command", "Bash(git *)", ("git", "*"), "ask", "git_requires_review"),
        ("mcp", "mcp__github__create_issue", None, "allow", None),
    ]


def test_compile_profile_policy_rules_includes_permission_rules() -> None:
    rules = compile_profile_policy_rules(
        ProfilePolicy.model_validate(
            {
                "permission_rules": [
                    {"pattern": "Read(/docs/**)", "outcome": "allow"},
                    {"pattern": "WebFetch(https://*.example.com/docs)", "outcome": "ask"},
                ]
            }
        )
    )

    assert [(rule.rule_type, rule.pattern, rule.outcome) for rule in rules] == [
        ("path", "/docs/**", "allow"),
        ("domain", "*.example.com", "ask"),
    ]


def test_evaluate_permission_rule_decision_matches_exact_tools() -> None:
    rules = compile_permission_rules(
        {"permission_rules": [{"pattern": "fs.read", "outcome": "allow"}, {"pattern": "fs.write", "outcome": "deny"}]}
    )

    decision = evaluate_permission_rule_decision(rules, subject_type="tool", value="fs.read")

    assert decision.outcome == "allow"
    assert decision.reason_code == "permission_rule_allowed"
    assert decision.subject.type == "tool"
    assert decision.subject.normalized == "fs.read"


def test_evaluate_permission_rule_decision_uses_command_token_matching_and_deny_precedence() -> None:
    rules = compile_permission_rules(
        {
            "permission_rules": [
                {"pattern": "Bash(git *)", "outcome": "allow"},
                {"pattern": "Bash(git status)", "outcome": "deny", "reason_code": "status_blocked"},
            ]
        }
    )

    denied = evaluate_permission_rule_decision(
        rules,
        subject_type="command",
        value="git status",
        argv=["git", "status"],
    )
    unmatched_extra_arg = evaluate_permission_rule_decision(
        rules,
        subject_type="command",
        value="git status --short",
        argv=["git", "status", "--short"],
    )

    assert denied.outcome == "deny"
    assert denied.reason_code == "status_blocked"
    assert [match.outcome for match in denied.matched_rules] == ["allow", "deny"]
    assert unmatched_extra_arg.outcome == "deny"
    assert unmatched_extra_arg.reason_code == "permission_rule_not_allowed"
    assert unmatched_extra_arg.matched_rules == []


def test_evaluate_permission_rule_decision_matches_path_domain_skill_agent_and_mcp_subjects() -> None:
    rules = compile_permission_rules(
        {
            "permission_rules": [
                {"pattern": "Read(/docs/**)", "outcome": "allow"},
                {"pattern": "Read(/docs/private/**)", "outcome": "deny"},
                {"pattern": "WebFetch(https://*.example.com/docs)", "outcome": "ask"},
                {"pattern": "Skill(review)", "outcome": "allow"},
                {"pattern": "Agent(backend-*)", "outcome": "ask"},
                {"pattern": "mcp__github__*", "outcome": "ask"},
                {"pattern": "mcp__github__delete_repo", "outcome": "deny"},
            ]
        }
    )

    private_path = evaluate_permission_rule_decision(rules, subject_type="path", value="/docs/private/secret.md")
    domain = evaluate_permission_rule_decision(rules, subject_type="domain", value="api.example.com")
    skill = evaluate_permission_rule_decision(rules, subject_type="skill", value="review")
    agent = evaluate_permission_rule_decision(rules, subject_type="agent", value="backend-engineer")
    mcp = evaluate_permission_rule_decision(rules, subject_type="mcp", value="mcp__github__delete_repo")

    assert private_path.outcome == "deny"
    assert [match.outcome for match in private_path.matched_rules] == ["allow", "deny"]
    assert domain.outcome == "ask"
    assert domain.matched_rules[0].pattern == "*.example.com"
    assert skill.outcome == "allow"
    assert agent.outcome == "ask"
    assert mcp.outcome == "deny"
    assert [match.outcome for match in mcp.matched_rules] == ["ask", "deny"]


def test_evaluate_profile_tool_decision_does_not_treat_non_tool_permission_rules_as_tool_grants() -> None:
    profile = MCPProfile(
        id="path-only",
        name="Path Only",
        policy_document=ProfilePolicy.model_validate({"permission_rules": [{"pattern": "Read(/docs/**)", "outcome": "allow"}]}),
    )

    decision = evaluate_profile_tool_decision(profile, "fs.read")

    assert decision.outcome == "deny"
    assert decision.reason_code == "tool_not_allowed"
