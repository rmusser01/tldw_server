"""Tests for bash prefix matching via tool_tier_overrides.

Verifies that Bash(git:*), Bash(rm:*), Bash(npm:*) style patterns
in tool_tier_overrides are respected by GovernanceFilter and
runner_client._resolve_runtime_permission_outcome().
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock

from tldw_Server_API.app.core.Agent_Client_Protocol.events import AgentEvent, AgentEventKind
from tldw_Server_API.app.core.Agent_Client_Protocol.governance_filter import GovernanceFilter


# ---------------------------------------------------------------------------
# GovernanceFilter tests
# ---------------------------------------------------------------------------


def _make_snapshot(
    tool_tier_overrides: dict[str, str] | None = None,
    denied_tools: list[str] | None = None,
    allowed_tools: list[str] | None = None,
) -> MagicMock:
    snap = MagicMock()
    snap.resolved_policy_document = {
        "denied_tools": denied_tools or [],
        "allowed_tools": allowed_tools or [],
        "tool_tier_overrides": tool_tier_overrides or {},
    }
    snap.approval_summary = {}
    return snap


@pytest.mark.asyncio
async def test_bash_git_auto_approved_via_snapshot():
    """Bash(git:*) pattern in tool_tier_overrides -> auto tier."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    snap = _make_snapshot(
        tool_tier_overrides={
            "Bash(git:*)": "auto",
            "Bash(rm:*)": "individual",
            "Bash(npm:*)": "batch",
        },
    )
    gf = GovernanceFilter(bus=bus, policy_snapshot=snap)

    event = AgentEvent(
        session_id="test",
        kind=AgentEventKind.TOOL_CALL,
        payload={"tool_name": "Bash(git:status)"},
        metadata={},
    )
    await gf.process(event)
    # Should auto-forward (TOOL_CALL published, not held)
    assert bus.publish.call_count == 1
    assert bus.publish.call_args[0][0].kind == AgentEventKind.TOOL_CALL


@pytest.mark.asyncio
async def test_bash_rm_individual_held():
    """Bash(rm:*) pattern -> individual tier -> held for approval."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    snap = _make_snapshot(
        tool_tier_overrides={"Bash(rm:*)": "individual"},
    )
    gf = GovernanceFilter(bus=bus, policy_snapshot=snap)

    event = AgentEvent(
        session_id="test",
        kind=AgentEventKind.TOOL_CALL,
        payload={"tool_name": "Bash(rm:-rf /tmp/test)"},
        metadata={},
    )
    await gf.process(event)
    # Should publish a PERMISSION_REQUEST (held), not the TOOL_CALL directly
    assert bus.publish.call_count == 1
    published = bus.publish.call_args[0][0]
    assert published.kind == AgentEventKind.PERMISSION_REQUEST


@pytest.mark.asyncio
async def test_bash_npm_batch_held():
    """Bash(npm:*) pattern -> batch tier -> held for approval."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    snap = _make_snapshot(
        tool_tier_overrides={"Bash(npm:*)": "batch"},
    )
    gf = GovernanceFilter(bus=bus, policy_snapshot=snap)

    event = AgentEvent(
        session_id="test",
        kind=AgentEventKind.TOOL_CALL,
        payload={"tool_name": "Bash(npm:install)"},
        metadata={},
    )
    await gf.process(event)
    assert bus.publish.call_count == 1
    published = bus.publish.call_args[0][0]
    assert published.kind == AgentEventKind.PERMISSION_REQUEST
    assert published.payload["tier"] == "batch"


@pytest.mark.asyncio
async def test_denied_tools_override_tier_overrides():
    """denied_tools takes priority over tool_tier_overrides."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    snap = _make_snapshot(
        denied_tools=["Bash(git:*)"],
        tool_tier_overrides={"Bash(git:*)": "auto"},
    )
    gf = GovernanceFilter(bus=bus, policy_snapshot=snap)

    event = AgentEvent(
        session_id="test",
        kind=AgentEventKind.TOOL_CALL,
        payload={"tool_name": "Bash(git:push)"},
        metadata={},
    )
    await gf.process(event)
    # Should be denied (TOOL_RESULT with error), not auto-approved
    assert bus.publish.call_count == 1
    published = bus.publish.call_args[0][0]
    assert published.kind == AgentEventKind.TOOL_RESULT
    assert "denied" in published.payload.get("error", "").lower()


@pytest.mark.asyncio
async def test_allowed_tools_override_tier_overrides():
    """allowed_tools takes priority over tool_tier_overrides."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    snap = _make_snapshot(
        allowed_tools=["Bash(git:*)"],
        tool_tier_overrides={"Bash(git:*)": "individual"},
    )
    gf = GovernanceFilter(bus=bus, policy_snapshot=snap)

    event = AgentEvent(
        session_id="test",
        kind=AgentEventKind.TOOL_CALL,
        payload={"tool_name": "Bash(git:status)"},
        metadata={},
    )
    await gf.process(event)
    # allowed_tools -> "auto" tier, should pass through
    assert bus.publish.call_count == 1
    assert bus.publish.call_args[0][0].kind == AgentEventKind.TOOL_CALL


@pytest.mark.asyncio
async def test_no_matching_override_falls_through():
    """Tool not matching any override falls through to default tier logic."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    snap = _make_snapshot(
        tool_tier_overrides={"Bash(git:*)": "auto"},
    )
    gf = GovernanceFilter(bus=bus, policy_snapshot=snap)

    event = AgentEvent(
        session_id="test",
        kind=AgentEventKind.TOOL_CALL,
        payload={"tool_name": "SomeOtherTool"},
        metadata={},
    )
    await gf.process(event)
    # Falls through to determine_permission_tier -- still publishes something
    assert bus.publish.call_count == 1


# ---------------------------------------------------------------------------
# runner_client._resolve_runtime_permission_outcome tests
# ---------------------------------------------------------------------------

from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import (
    _resolve_runtime_permission_outcome,
)


def _make_runner_snapshot(
    tool_tier_overrides: dict[str, str] | None = None,
    denied_tools: list[str] | None = None,
    allowed_tools: list[str] | None = None,
) -> MagicMock:
    snap = MagicMock()
    snap.resolved_policy_document = {
        "denied_tools": denied_tools or [],
        "allowed_tools": allowed_tools or ["*"],
        "tool_tier_overrides": tool_tier_overrides or {},
    }
    snap.approval_summary = {}
    snap.policy_snapshot_fingerprint = "test-fingerprint"
    snap.policy_provenance_summary = {}
    return snap


def test_runner_tier_override_auto():
    """runner_client picks up tool_tier_overrides auto -> approve."""
    snap = _make_runner_snapshot(
        tool_tier_overrides={"Bash(git:*)": "auto"},
    )
    result = _resolve_runtime_permission_outcome(snap, "Bash(git:status)")
    assert result["action"] == "approve"


def test_runner_tier_override_individual():
    """runner_client picks up tool_tier_overrides individual -> prompt."""
    snap = _make_runner_snapshot(
        tool_tier_overrides={"Bash(rm:*)": "individual"},
    )
    result = _resolve_runtime_permission_outcome(snap, "Bash(rm:-rf /)")
    assert result["action"] == "prompt"
    assert result["approval_requirement"] == "approval_required"


def test_runner_tier_override_batch():
    """runner_client picks up tool_tier_overrides batch -> prompt."""
    snap = _make_runner_snapshot(
        tool_tier_overrides={"Bash(npm:*)": "batch"},
    )
    result = _resolve_runtime_permission_outcome(snap, "Bash(npm:install)")
    assert result["action"] == "prompt"
    assert result["approval_requirement"] == "approval_required"


def test_runner_denied_beats_tier_override():
    """denied_tools still blocks even if tier override says auto."""
    snap = _make_runner_snapshot(
        denied_tools=["Bash(git:*)"],
        tool_tier_overrides={"Bash(git:*)": "auto"},
    )
    result = _resolve_runtime_permission_outcome(snap, "Bash(git:push)")
    assert result["action"] == "deny"


def test_runner_no_override_passes_through():
    """Without override, normal allowed flow applies."""
    snap = _make_runner_snapshot(
        tool_tier_overrides={"Bash(git:*)": "auto"},
    )
    result = _resolve_runtime_permission_outcome(snap, "SomeOtherTool")
    # Tool is in allowed_tools=["*"], so should approve
    assert result["action"] == "approve"
