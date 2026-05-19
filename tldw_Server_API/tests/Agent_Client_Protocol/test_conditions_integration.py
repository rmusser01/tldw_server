"""Integration tests for policy conditions wiring.

Covers GovernanceFilter condition evaluation, DB schema migration for
conditions_json and ancestry_chain_json columns, and backward compat.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.events import AgentEvent, AgentEventKind

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 4, 7, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_event(
    tool_name: str,
    session_id: str = "s1",
    tool_call_id: str = "tc1",
) -> AgentEvent:
    return AgentEvent(
        session_id=session_id,
        kind=AgentEventKind.TOOL_CALL,
        payload={
            "tool_id": tool_call_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments": {},
        },
    )


def _make_snapshot(
    denied_tools: list[str] | None = None,
    allowed_tools: list[str] | None = None,
    conditions: dict | None = None,
) -> MagicMock:
    snapshot = MagicMock()
    doc: dict = {}
    if denied_tools is not None:
        doc["denied_tools"] = denied_tools
    if allowed_tools is not None:
        doc["allowed_tools"] = allowed_tools
    if conditions is not None:
        doc["conditions"] = conditions
    snapshot.resolved_policy_document = doc
    return snapshot


def _make_bus() -> MagicMock:
    bus = MagicMock()
    bus.publish = AsyncMock()
    return bus


# ---------------------------------------------------------------------------
# GovernanceFilter: expired policy skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expired_policy_skipped():
    """A policy with an expired time window should be skipped (fall through)."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.governance_filter import GovernanceFilter

    bus = _make_bus()
    snapshot = _make_snapshot(
        denied_tools=["dangerous_*"],
        conditions={
            "valid_from": (_NOW - timedelta(hours=2)).isoformat(),
            "valid_until": (_NOW - timedelta(hours=1)).isoformat(),
        },
    )
    gov = GovernanceFilter(bus=bus, policy_snapshot=snapshot)

    # Tool would be denied if conditions passed, but they should fail
    event = _make_tool_event(tool_name="dangerous_delete")
    await gov.process(event)

    # With expired conditions, snapshot returns None -> falls through to heuristic.
    # "dangerous_delete" isn't recognized as a known tool by heuristics, so it
    # should default to some tier. The key thing is it should NOT be denied_by_snapshot.
    bus.publish.assert_called_once()
    published = bus.publish.call_args[0][0]
    assert published.metadata.get("governance_action") != "denied_by_snapshot"


# ---------------------------------------------------------------------------
# GovernanceFilter: valid policy applies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_policy_applies():
    """A policy with valid conditions should apply its rules."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.governance_filter import GovernanceFilter

    bus = _make_bus()
    snapshot = _make_snapshot(
        allowed_tools=["safe_*"],
        conditions={
            "valid_from": (_NOW - timedelta(hours=1)).isoformat(),
            "valid_until": (_NOW + timedelta(hours=1)).isoformat(),
        },
    )
    gov = GovernanceFilter(bus=bus, policy_snapshot=snapshot)

    event = _make_tool_event(tool_name="safe_read")
    await gov.process(event)

    bus.publish.assert_called_once()
    published = bus.publish.call_args[0][0]
    assert published.kind == AgentEventKind.TOOL_CALL
    assert published.payload["tool_name"] == "safe_read"


# ---------------------------------------------------------------------------
# GovernanceFilter: label mismatch skips
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_label_mismatch_skips_policy():
    """A policy requiring labels that don't match session metadata should be skipped."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.governance_filter import GovernanceFilter

    bus = _make_bus()
    snapshot = _make_snapshot(
        denied_tools=["*"],
        conditions={"required_labels": {"env": "prod"}},
    )
    # Session metadata says env=staging, so conditions fail
    gov = GovernanceFilter(
        bus=bus,
        policy_snapshot=snapshot,
        session_metadata={"labels": {"env": "staging"}},
    )

    event = _make_tool_event(tool_name="anything")
    await gov.process(event)

    bus.publish.assert_called_once()
    published = bus.publish.call_args[0][0]
    # Policy should not apply, so not denied_by_snapshot
    assert published.metadata.get("governance_action") != "denied_by_snapshot"


# ---------------------------------------------------------------------------
# GovernanceFilter: no-conditions backward compat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_conditions_backward_compat():
    """A snapshot without any conditions key should behave as before (apply unconditionally)."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.governance_filter import GovernanceFilter

    bus = _make_bus()
    snapshot = _make_snapshot(denied_tools=["bad_*"])
    # No conditions key in snapshot at all
    gov = GovernanceFilter(bus=bus, policy_snapshot=snapshot)

    event = _make_tool_event(tool_name="bad_tool")
    await gov.process(event)

    bus.publish.assert_called_once()
    published = bus.publish.call_args[0][0]
    assert published.kind == AgentEventKind.TOOL_RESULT
    assert "denied by policy" in published.payload.get("error", "").lower()
    assert published.metadata.get("governance_action") == "denied_by_snapshot"


# ---------------------------------------------------------------------------
# DB: ancestry_chain_json column exists
# ---------------------------------------------------------------------------


def test_ancestry_chain_db_column(tmp_path: Path) -> None:
    """The sessions table should have an ancestry_chain_json column after migration."""
    from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB

    db = ACPSessionsDB(db_path=str(tmp_path / "test-ancestry.db"))
    # register_session triggers schema init
    d = db.register_session(
        session_id="test-anc",
        user_id=1,
        agent_type="custom",
    )
    assert "ancestry_chain_json" in d
    # Column defaults to NULL; not set means None (deserialized as [])
    # When NULL in DB, the JSON list deserializer leaves it as None
    assert d["ancestry_chain_json"] is None or d["ancestry_chain_json"] == []


# ---------------------------------------------------------------------------
# DB: conditions_json column exists
# ---------------------------------------------------------------------------


def test_conditions_json_db_column(tmp_path: Path) -> None:
    """The permission_policies table should have a conditions_json column."""
    from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB

    db = ACPSessionsDB(db_path=str(tmp_path / "test-conditions.db"))
    # Create a permission policy to trigger schema init
    policy_id = db.create_permission_policy(
        name="test-policy",
        rules_json="[]",
    )
    row = db.get_permission_policy(policy_id)
    assert row is not None
    assert "conditions_json" in row
    assert row["conditions_json"] is None  # not set


# ---------------------------------------------------------------------------
# GovernanceFilter: empty conditions dict acts like no conditions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_conditions_dict_applies_policy():
    """An empty conditions dict {} should not block the policy."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.governance_filter import GovernanceFilter

    bus = _make_bus()
    snapshot = _make_snapshot(
        denied_tools=["blocked_*"],
        conditions={},
    )
    gov = GovernanceFilter(bus=bus, policy_snapshot=snapshot)

    event = _make_tool_event(tool_name="blocked_tool")
    await gov.process(event)

    bus.publish.assert_called_once()
    published = bus.publish.call_args[0][0]
    assert published.kind == AgentEventKind.TOOL_RESULT
    assert published.metadata.get("governance_action") == "denied_by_snapshot"


@pytest.mark.asyncio
async def test_source_ip_match_applies_policy() -> None:
    """A policy restricted by source IP should apply when client_ip matches."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.governance_filter import GovernanceFilter

    bus = _make_bus()
    snapshot = _make_snapshot(
        denied_tools=["restricted_*"],
        conditions={"source_ips": ["10.0.0.0/24"]},
    )
    gov = GovernanceFilter(
        bus=bus,
        policy_snapshot=snapshot,
        session_metadata={"client_ip": "10.0.0.42"},
    )

    event = _make_tool_event(tool_name="restricted_tool")
    await gov.process(event)

    published = bus.publish.call_args[0][0]
    assert published.kind == AgentEventKind.TOOL_RESULT
    assert published.metadata.get("governance_action") == "denied_by_snapshot"


@pytest.mark.asyncio
async def test_source_ip_mismatch_skips_policy() -> None:
    """A policy restricted by source IP should be skipped when client_ip mismatches."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.governance_filter import GovernanceFilter

    bus = _make_bus()
    snapshot = _make_snapshot(
        denied_tools=["restricted_*"],
        conditions={"source_ips": ["10.0.0.0/24"]},
    )
    gov = GovernanceFilter(
        bus=bus,
        policy_snapshot=snapshot,
        session_metadata={"client_ip": "192.168.1.20"},
    )

    event = _make_tool_event(tool_name="restricted_tool")
    await gov.process(event)

    published = bus.publish.call_args[0][0]
    assert published.metadata.get("governance_action") != "denied_by_snapshot"
