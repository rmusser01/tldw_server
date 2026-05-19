"""Tests for GovernanceFilter unified with MCPHub policy snapshot.

Verifies that GovernanceFilter checks the MCPHub ACPRuntimePolicySnapshot
before falling back to the existing ACP tier heuristics.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.events import AgentEvent, AgentEventKind

pytestmark = pytest.mark.unit


def _make_tool_event(
    tool_name: str,
    session_id: str = "s1",
    tool_call_id: str = "tc1",
    arguments: dict | None = None,
    metadata: dict | None = None,
) -> AgentEvent:
    return AgentEvent(
        session_id=session_id,
        kind=AgentEventKind.TOOL_CALL,
        payload={
            "tool_id": tool_call_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments": arguments or {},
        },
        metadata=metadata or {},
    )


def _make_snapshot(
    denied_tools: list[str] | None = None,
    allowed_tools: list[str] | None = None,
    tool_tier_overrides: dict[str, str] | None = None,
) -> MagicMock:
    """Build a mock ACPRuntimePolicySnapshot with resolved_policy_document."""
    snapshot = MagicMock()
    doc: dict = {}
    if denied_tools is not None:
        doc["denied_tools"] = denied_tools
    if allowed_tools is not None:
        doc["allowed_tools"] = allowed_tools
    if tool_tier_overrides is not None:
        doc["tool_tier_overrides"] = tool_tier_overrides
    snapshot.resolved_policy_document = doc
    return snapshot


def _make_bus() -> MagicMock:
    bus = MagicMock()
    bus.publish = AsyncMock()
    return bus


@pytest.mark.asyncio
async def test_denied_tool_blocked_immediately():
    """A tool matching snapshot.denied_tools is immediately denied.

    GovernanceFilter should publish a TOOL_RESULT error event (not the
    original TOOL_CALL) and never forward the call.
    """
    from tldw_Server_API.app.core.Agent_Client_Protocol.governance_filter import GovernanceFilter

    bus = _make_bus()
    snapshot = _make_snapshot(denied_tools=["dangerous_*"])
    gov = GovernanceFilter(bus=bus, policy_snapshot=snapshot)

    event = _make_tool_event(tool_name="dangerous_delete")
    await gov.process(event)

    # Should have published exactly once -- a TOOL_RESULT error
    bus.publish.assert_called_once()
    published = bus.publish.call_args[0][0]
    assert published.kind == AgentEventKind.TOOL_RESULT
    assert "denied by policy" in published.payload.get("error", "").lower()
    assert published.metadata.get("governance_action") == "denied_by_snapshot"


@pytest.mark.asyncio
async def test_allowed_tool_auto_approved():
    """A tool matching snapshot.allowed_tools is auto-forwarded as TOOL_CALL."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.governance_filter import GovernanceFilter

    bus = _make_bus()
    snapshot = _make_snapshot(allowed_tools=["safe_*"])
    gov = GovernanceFilter(bus=bus, policy_snapshot=snapshot)

    event = _make_tool_event(tool_name="safe_read")
    await gov.process(event)

    bus.publish.assert_called_once()
    published = bus.publish.call_args[0][0]
    assert published.kind == AgentEventKind.TOOL_CALL
    assert published.payload["tool_name"] == "safe_read"


@pytest.mark.asyncio
async def test_tool_tier_override_respected():
    """A tool matching tool_tier_overrides gets that tier applied.

    If the override maps to 'auto', the TOOL_CALL passes through immediately.
    """
    from tldw_Server_API.app.core.Agent_Client_Protocol.governance_filter import GovernanceFilter

    bus = _make_bus()
    snapshot = _make_snapshot(tool_tier_overrides={"Bash(git:*)": "auto"})
    gov = GovernanceFilter(bus=bus, policy_snapshot=snapshot)

    event = _make_tool_event(tool_name="Bash(git:status)")
    await gov.process(event)

    bus.publish.assert_called_once()
    published = bus.publish.call_args[0][0]
    assert published.kind == AgentEventKind.TOOL_CALL
    assert published.payload["tool_name"] == "Bash(git:status)"


@pytest.mark.asyncio
async def test_no_snapshot_falls_through_to_tier():
    """Without a snapshot (None), existing determine_permission_tier() logic applies.

    'read_file' should be classified as 'auto' by heuristics and pass through.
    'bash' should be classified as 'individual' and be held.
    """
    from tldw_Server_API.app.core.Agent_Client_Protocol.governance_filter import GovernanceFilter

    bus = _make_bus()
    gov = GovernanceFilter(bus=bus)  # No snapshot

    # 'read_file' => auto tier by heuristic => published as TOOL_CALL
    event_auto = _make_tool_event(tool_name="read_file")
    await gov.process(event_auto)
    bus.publish.assert_called_once()
    published = bus.publish.call_args[0][0]
    assert published.kind == AgentEventKind.TOOL_CALL

    bus.publish.reset_mock()

    # 'bash' => individual tier by heuristic => held, PERMISSION_REQUEST published
    event_individual = _make_tool_event(tool_name="bash")
    await gov.process(event_individual)
    bus.publish.assert_called_once()
    published = bus.publish.call_args[0][0]
    assert published.kind == AgentEventKind.PERMISSION_REQUEST
    assert published.payload["tier"] == "individual"


@pytest.mark.asyncio
async def test_snapshot_deny_overrides_tier_auto():
    """Even if the heuristic would say 'auto', a snapshot deny wins.

    'read_file' is normally auto-approved by heuristic, but if it's in
    denied_tools, it must be blocked.
    """
    from tldw_Server_API.app.core.Agent_Client_Protocol.governance_filter import GovernanceFilter

    bus = _make_bus()
    snapshot = _make_snapshot(denied_tools=["read_*"])
    gov = GovernanceFilter(bus=bus, policy_snapshot=snapshot)

    event = _make_tool_event(tool_name="read_file")
    await gov.process(event)

    bus.publish.assert_called_once()
    published = bus.publish.call_args[0][0]
    assert published.kind == AgentEventKind.TOOL_RESULT
    assert "denied by policy" in published.payload.get("error", "").lower()
    assert published.metadata.get("governance_action") == "denied_by_snapshot"
