"""Integration tests for policy conditions wired into GovernanceFilter and DB.

Verifies that:
- Expired / valid time windows are respected by GovernanceFilter
- Label mismatches skip the policy
- Policies without conditions still apply (backward compat)
- ancestry_chain_json and conditions_json columns persist in the DB
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.events import (
    AgentEvent,
    AgentEventKind,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.governance_filter import (
    GovernanceFilter,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.policy_conditions import (
    PolicyConditions,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime.now(timezone.utc)


def _make_bus() -> MagicMock:
    bus = MagicMock()
    bus.publish = AsyncMock()
    return bus


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
        metadata={},
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


# ---------------------------------------------------------------------------
# 1. Expired policy skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expired_policy_skipped():
    """Policy with expired valid_until doesn't apply -- tool falls through."""
    bus = _make_bus()
    conditions = PolicyConditions(
        valid_from=_NOW - timedelta(hours=2),
        valid_until=_NOW - timedelta(hours=1),
    )
    snapshot = _make_snapshot(
        denied_tools=["dangerous_*"],
        conditions=conditions.to_dict(),
    )
    gov = GovernanceFilter(bus=bus, policy_snapshot=snapshot)

    # The tool would normally be denied, but conditions are expired
    result = gov._check_snapshot_policy("dangerous_delete")
    assert result is None, "Expired policy should not apply"


# ---------------------------------------------------------------------------
# 2. Valid policy applies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_policy_applies():
    """Policy within time window applies normally."""
    bus = _make_bus()
    conditions = PolicyConditions(
        valid_from=_NOW - timedelta(hours=1),
        valid_until=_NOW + timedelta(hours=1),
    )
    snapshot = _make_snapshot(
        denied_tools=["dangerous_*"],
        conditions=conditions.to_dict(),
    )
    gov = GovernanceFilter(bus=bus, policy_snapshot=snapshot)

    result = gov._check_snapshot_policy("dangerous_delete")
    assert result == "_deny", "Valid policy should deny the tool"


# ---------------------------------------------------------------------------
# 3. Label mismatch skips policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_label_mismatch_skips_policy():
    """Policy with required_labels that don't match skips."""
    bus = _make_bus()
    conditions = PolicyConditions(required_labels={"env": "prod"})
    snapshot = _make_snapshot(
        allowed_tools=["safe_*"],
        conditions=conditions.to_dict(),
    )
    # Session has labels that don't match
    gov = GovernanceFilter(
        bus=bus,
        policy_snapshot=snapshot,
        session_metadata={"labels": {"env": "staging"}},
    )

    result = gov._check_snapshot_policy("safe_read")
    assert result is None, "Label mismatch should skip the policy"


# ---------------------------------------------------------------------------
# 4. No conditions -- backward compat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_conditions_policy_applies():
    """Policy without conditions applies as before (backward compat)."""
    bus = _make_bus()
    snapshot = _make_snapshot(allowed_tools=["safe_*"])
    gov = GovernanceFilter(bus=bus, policy_snapshot=snapshot)

    result = gov._check_snapshot_policy("safe_read")
    assert result == "auto", "Policy without conditions should apply normally"


# ---------------------------------------------------------------------------
# 5. ancestry_chain_json column persists
# ---------------------------------------------------------------------------


def test_ancestry_chain_persists():
    """ancestry_chain_json column persists in the sessions table."""
    from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = ACPSessionsDB(db_path=os.path.join(tmpdir, "test.db"))

        chain = ["user:root", "agent:sub1"]
        db.register_session(
            session_id="chain-test",
            user_id=1,
        )
        # Update the ancestry_chain_json directly
        conn = db._get_conn()
        conn.execute(
            "UPDATE sessions SET ancestry_chain_json = ? WHERE session_id = ?",
            (json.dumps(chain), "chain-test"),
        )
        conn.commit()

        row = db.get_session("chain-test")
        assert row is not None
        assert row["ancestry_chain_json"] == chain


# ---------------------------------------------------------------------------
# 6. conditions_json column persists in permission_policies
# ---------------------------------------------------------------------------


def test_conditions_json_in_permission_policies():
    """conditions_json column persists in permission_policies table."""
    from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = ACPSessionsDB(db_path=os.path.join(tmpdir, "test.db"))

        cond = PolicyConditions(
            valid_from=_NOW - timedelta(hours=1),
            valid_until=_NOW + timedelta(hours=1),
            required_labels={"env": "prod"},
        )
        policy_id = db.create_permission_policy(
            name="test-policy",
            rules_json=json.dumps([{"tool_pattern": "*", "tier": "auto"}]),
        )

        # Update conditions_json
        conn = db._get_conn()
        conn.execute(
            "UPDATE permission_policies SET conditions_json = ? WHERE id = ?",
            (cond.to_json(), policy_id),
        )
        conn.commit()

        row = db.get_permission_policy(policy_id)
        assert row is not None
        assert row["conditions_json"] is not None
        parsed = PolicyConditions.from_json(row["conditions_json"])
        assert parsed.required_labels == {"env": "prod"}
        assert parsed.valid_from is not None
        assert parsed.valid_until is not None


# ---------------------------------------------------------------------------
# 7. Labels match -- policy applies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_label_match_applies_policy():
    """Policy with matching required_labels applies."""
    bus = _make_bus()
    conditions = PolicyConditions(required_labels={"env": "prod"})
    snapshot = _make_snapshot(
        allowed_tools=["safe_*"],
        conditions=conditions.to_dict(),
    )
    gov = GovernanceFilter(
        bus=bus,
        policy_snapshot=snapshot,
        session_metadata={"labels": {"env": "prod", "team": "sre"}},
    )

    result = gov._check_snapshot_policy("safe_read")
    assert result == "auto", "Matching labels should allow the policy to apply"
