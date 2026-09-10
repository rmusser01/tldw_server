"""Integration tests using the stub agent via real STDIO transport.

These tests exercise the full JSON-RPC protocol path through
ACPStdioClient → acp_stub_agent.py without needing the Go binary.
"""
from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import ACPRunnerClient
from tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client import ACPMessage
from tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client import (
    ACPStdioClient,
    ACPResponseError,
)
from tldw_Server_API.app.services.acp_runtime_policy_service import (
    ACPRuntimePolicySnapshot,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

STUB_AGENT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "Helper_Scripts",
    "acp_stub_agent.py",
)


@pytest.fixture
async def client():
    """Spawn stub agent for each test."""
    c = ACPStdioClient(
        command=sys.executable,
        args=[os.path.abspath(STUB_AGENT_PATH)],
    )
    await c.start()
    yield c
    await c.close()


# ---- Session lifecycle ----

async def test_initialize(client):
    """Verify initialize handshake returns agent info."""
    result = await client.call("initialize", {})
    assert result.result is not None
    assert result.result["agentInfo"]["name"] == "tldw-acp-stub"
    assert result.result["protocolVersion"] == 1


async def test_session_new(client):
    """Verify session/new returns a session ID."""
    await client.call("initialize", {})
    result = await client.call("session/new", {})
    assert "sessionId" in result.result
    assert result.result["sessionId"].startswith("stub-")


async def test_session_prompt(client):
    """Verify session/prompt returns stop reason and emits update."""
    await client.call("initialize", {})
    new_result = await client.call("session/new", {})
    session_id = new_result.result["sessionId"]

    # Collect notifications
    notifications = []

    async def handler(msg):
        notifications.append(msg)

    client.set_notification_handler(handler)

    result = await client.call(
        "session/prompt",
        {"sessionId": session_id, "prompt": "Hello"},
    )
    assert result.result["stopReason"] == "end"
    assert result.result["taskCompletion"] == {
        "status": "completed",
        "summary": "Deterministic ACP stub completed the requested task.",
        "artifacts": [],
    }


async def test_session_cancel(client):
    """Verify session/cancel returns successfully."""
    await client.call("initialize", {})
    result = await client.call("session/cancel", {})
    assert result.result is None


async def test_unknown_method_returns_error(client):
    """Verify unknown methods raise ACPResponseError."""
    with pytest.raises(ACPResponseError, match="method not found"):
        await client.call("nonexistent/method", {})


# ---- Concurrent sessions ----

async def test_concurrent_sessions(client):
    """Verify multiple sessions can be created sequentially."""
    await client.call("initialize", {})

    session_ids = []
    for _ in range(3):
        result = await client.call("session/new", {})
        session_ids.append(result.result["sessionId"])

    assert len(set(session_ids)) == 3  # All unique


# ---- Error handling ----

async def test_client_close_is_idempotent(client):
    """Closing an already-closed client should not raise."""
    await client.close()
    await client.close()  # Should not raise


async def test_client_is_running(client):
    """Verify is_running reflects process state."""
    assert client.is_running
    await client.close()
    assert not client.is_running


def _new_runner_client_for_permissions() -> ACPRunnerClient:
    cfg = SimpleNamespace(
        command="echo",
        args=[],
        env={},
        cwd=None,
        startup_timeout_sec=0,
    )
    return ACPRunnerClient(cfg)


@pytest.mark.unit
async def test_runtime_policy_refresh_applies_updated_permissions() -> None:
    client = _new_runner_client_for_permissions()
    session_id = "session-refresh-policy"
    policy_state = {"allowed_tools": ["web.search"], "fingerprint": "snap-allow"}
    call_count = {"build_snapshot": 0}

    async def _fake_check_permission_governance(*args, **kwargs):
        del args, kwargs
        return None

    client.check_permission_governance = _fake_check_permission_governance  # type: ignore[assignment]

    class _Store:
        def __init__(self) -> None:
            self.record = SimpleNamespace(
                session_id=session_id,
                user_id=7,
                policy_snapshot_fingerprint=None,
                policy_snapshot_version=None,
                policy_snapshot_refreshed_at=None,
                policy_summary=None,
                policy_provenance_summary=None,
                policy_refresh_error=None,
            )

        async def get_session(self, _session_id: str):
            return self.record

        async def update_policy_snapshot_state(self, _session_id: str, **kwargs):
            for key, value in kwargs.items():
                setattr(self.record, key, value)
            return self.record

    class _RuntimePolicyService:
        async def build_snapshot(self, **kwargs):
            del kwargs
            call_count["build_snapshot"] += 1
            return ACPRuntimePolicySnapshot(
                session_id=session_id,
                user_id=7,
                policy_snapshot_version="resolved-v1",
                policy_snapshot_fingerprint=policy_state["fingerprint"],
                policy_snapshot_refreshed_at="2026-03-14T12:00:00+00:00",
                policy_summary={"allowed_tool_count": len(policy_state["allowed_tools"])},
                policy_provenance_summary={"source_kinds": ["capability_mapping"]},
                resolved_policy_document={"allowed_tools": list(policy_state["allowed_tools"])},
                approval_summary={"mode": "allow"},
                context_summary={},
                execution_config={},
            )

        async def persist_snapshot(self, *, session_store, snapshot):
            return await session_store.update_policy_snapshot_state(
                snapshot.session_id,
                policy_snapshot_version=snapshot.policy_snapshot_version,
                policy_snapshot_fingerprint=snapshot.policy_snapshot_fingerprint,
                policy_snapshot_refreshed_at=snapshot.policy_snapshot_refreshed_at,
                policy_summary=snapshot.policy_summary,
                policy_provenance_summary=snapshot.policy_provenance_summary,
                policy_refresh_error=snapshot.refresh_error,
            )

    store = _Store()

    async def _get_store():
        return store

    client._runtime_policy_service = _RuntimePolicyService()  # type: ignore[assignment]
    client._get_acp_session_store = _get_store  # type: ignore[assignment]
    client._should_force_runtime_policy_refresh = lambda *, tier: True  # type: ignore[assignment]

    allowed_response = await client._handle_request(
        ACPMessage(
            jsonrpc="2.0",
            id="perm-allow",
            method="session/request_permission",
            params={
                "sessionId": session_id,
                "tool": {"name": "web.search", "input": {"query": "opa"}},
            },
        )
    )
    assert allowed_response.result == {"outcome": {"outcome": "approved"}}
    assert store.record.policy_snapshot_fingerprint == "snap-allow"

    policy_state["allowed_tools"] = ["fs.read"]
    policy_state["fingerprint"] = "snap-updated"

    denied_response = await client._handle_request(
        ACPMessage(
            jsonrpc="2.0",
            id="perm-deny",
            method="session/request_permission",
            params={
                "sessionId": session_id,
                "tool": {"name": "web.search", "input": {"query": "opa"}},
            },
        )
    )
    assert denied_response.result == {
        "outcome": {
            "outcome": "denied",
            "deny_reason": "tool_not_allowed_by_policy",
            "policy_snapshot_fingerprint": "snap-updated",
            "provenance_summary": {"source_kinds": ["capability_mapping"]},
        }
    }
    assert call_count["build_snapshot"] == 2


@pytest.mark.unit
async def test_low_risk_permission_reuses_cached_runtime_snapshot() -> None:
    client = _new_runner_client_for_permissions()
    session_id = "session-refresh-cache"
    policy_state = {"allowed_tools": ["web.search"], "fingerprint": "snap-cache"}
    call_count = {"build_snapshot": 0}

    async def _fake_check_permission_governance(*args, **kwargs):
        del args, kwargs
        return None

    client.check_permission_governance = _fake_check_permission_governance  # type: ignore[assignment]

    class _Store:
        def __init__(self) -> None:
            self.record = SimpleNamespace(
                session_id=session_id,
                user_id=7,
                policy_snapshot_fingerprint=None,
                policy_snapshot_version=None,
                policy_snapshot_refreshed_at=None,
                policy_summary=None,
                policy_provenance_summary=None,
                policy_refresh_error=None,
            )

        async def get_session(self, _session_id: str):
            return self.record

        async def update_policy_snapshot_state(self, _session_id: str, **kwargs):
            for key, value in kwargs.items():
                setattr(self.record, key, value)
            return self.record

    class _RuntimePolicyService:
        async def build_snapshot(self, **kwargs):
            del kwargs
            call_count["build_snapshot"] += 1
            return ACPRuntimePolicySnapshot(
                session_id=session_id,
                user_id=7,
                policy_snapshot_version="resolved-v1",
                policy_snapshot_fingerprint=policy_state["fingerprint"],
                policy_snapshot_refreshed_at="2026-03-14T12:00:00+00:00",
                policy_summary={"allowed_tool_count": len(policy_state["allowed_tools"])},
                policy_provenance_summary={"source_kinds": ["capability_mapping"]},
                resolved_policy_document={"allowed_tools": list(policy_state["allowed_tools"])},
                approval_summary={"mode": "allow"},
                context_summary={},
                execution_config={},
            )

        async def persist_snapshot(self, *, session_store, snapshot):
            return await session_store.update_policy_snapshot_state(
                snapshot.session_id,
                policy_snapshot_version=snapshot.policy_snapshot_version,
                policy_snapshot_fingerprint=snapshot.policy_snapshot_fingerprint,
                policy_snapshot_refreshed_at=snapshot.policy_snapshot_refreshed_at,
                policy_summary=snapshot.policy_summary,
                policy_provenance_summary=snapshot.policy_provenance_summary,
                policy_refresh_error=snapshot.refresh_error,
            )

    store = _Store()

    async def _get_store():
        return store

    client._runtime_policy_service = _RuntimePolicyService()  # type: ignore[assignment]
    client._get_acp_session_store = _get_store  # type: ignore[assignment]

    first = await client._handle_request(
        ACPMessage(
            jsonrpc="2.0",
            id="perm-cache-1",
            method="session/request_permission",
            params={
                "sessionId": session_id,
                "tool": {"name": "web.search", "input": {"query": "opa"}},
            },
        )
    )
    assert first.result == {"outcome": {"outcome": "approved"}}
    assert call_count["build_snapshot"] == 1

    policy_state["allowed_tools"] = ["fs.read"]
    policy_state["fingerprint"] = "snap-cache-updated"

    second = await client._handle_request(
        ACPMessage(
            jsonrpc="2.0",
            id="perm-cache-2",
            method="session/request_permission",
            params={
                "sessionId": session_id,
                "tool": {"name": "web.search", "input": {"query": "opa"}},
            },
        )
    )
    assert second.result == {"outcome": {"outcome": "approved"}}
    assert call_count["build_snapshot"] == 1


@pytest.mark.unit
async def test_failed_runtime_policy_refresh_fails_closed() -> None:
    client = _new_runner_client_for_permissions()
    session_id = "session-refresh-failure"

    async def _fake_check_permission_governance(*args, **kwargs):
        del args, kwargs
        return None

    client.check_permission_governance = _fake_check_permission_governance  # type: ignore[assignment]

    class _Store:
        def __init__(self) -> None:
            self.record = SimpleNamespace(
                session_id=session_id,
                user_id=7,
                policy_snapshot_fingerprint="stale-snapshot",
                policy_snapshot_version="resolved-v1",
                policy_snapshot_refreshed_at="2026-03-14T11:59:00+00:00",
                policy_summary={"allowed_tool_count": 1},
                policy_provenance_summary={"source_kinds": ["profile"]},
                policy_refresh_error=None,
            )

        async def get_session(self, _session_id: str):
            return self.record

        async def update_policy_snapshot_state(self, _session_id: str, **kwargs):
            for key, value in kwargs.items():
                setattr(self.record, key, value)
            return self.record

    class _RuntimePolicyService:
        async def build_snapshot(self, **kwargs):
            del kwargs
            raise RuntimeError("policy_refresh_failed")

        async def persist_snapshot(self, *, session_store, snapshot):
            del session_store, snapshot
            raise AssertionError("persist_snapshot should not run on refresh failure")

    store = _Store()

    async def _get_store():
        return store

    client._runtime_policy_service = _RuntimePolicyService()  # type: ignore[assignment]
    client._get_acp_session_store = _get_store  # type: ignore[assignment]
    client._runtime_policy_snapshots[session_id] = ACPRuntimePolicySnapshot(
        session_id=session_id,
        user_id=7,
        policy_snapshot_version="resolved-v1",
        policy_snapshot_fingerprint="stale-snapshot",
        policy_snapshot_refreshed_at="2026-03-14T11:59:00+00:00",
        policy_summary={"allowed_tool_count": 1},
        policy_provenance_summary={"source_kinds": ["profile"]},
        resolved_policy_document={"allowed_tools": ["exec.run"]},
        approval_summary={"mode": "allow"},
        context_summary={},
        execution_config={},
    )

    refreshed = await client._get_runtime_policy_snapshot(session_id, force_refresh=True)
    assert refreshed is None
    assert store.record.policy_snapshot_fingerprint is None
    assert store.record.policy_refresh_error == "policy_refresh_failed"

    response = await client._handle_request(
        ACPMessage(
            jsonrpc="2.0",
            id="perm-fail-closed",
            method="session/request_permission",
            params={
                "sessionId": session_id,
                "tool": {"name": "exec.run", "input": {"command": "whoami"}},
            },
        )
    )
    assert response.result == {
        "outcome": {
            "outcome": "denied",
            "deny_reason": "policy_snapshot_unavailable",
            "policy_snapshot_fingerprint": None,
            "provenance_summary": {},
        }
    }
