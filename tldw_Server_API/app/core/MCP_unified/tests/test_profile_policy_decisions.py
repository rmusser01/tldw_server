"""Tests for MCP profile policy decision primitives."""

from __future__ import annotations

from mcp_unified.profiles.decisions import (
    PolicyDecision,
    PolicyDecisionSubject,
)


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
