"""Tests for the gateway policy grant manager, config, and bootstrap wiring."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest


class _RecordingAuditStore:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def append_event(self, event: Any) -> None:
        self.events.append(event)


def test_policy_grant_manager_grant_list_revoke_lifecycle_with_audit() -> None:
    from mcp_unified.gateway.policy_grants import GatewayPolicyGrantManager
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    audit_store = _RecordingAuditStore()
    manager = GatewayPolicyGrantManager(
        policy_grant_store=InMemoryPolicyGrantStore(),
        audit_store=audit_store,
    )

    created = asyncio.run(
        manager.grant_approval(
            profile_id="researcher",
            subject_type="domain",
            value="https://Example.com/private",
            ttl_seconds=900,
            granted_by="operator",
            reason="one-off fetch",
        )
    )
    grant_payload = created["grant"]
    assert grant_payload["value"] == "example.com"
    assert grant_payload["grant_type"] == "approval"
    assert grant_payload["ttl_seconds"] == 900

    listed = asyncio.run(manager.list_grants(profile_id="researcher"))
    assert len(listed["grants"]) == 1
    assert listed["grants"][0]["grant_id"] == grant_payload["grant_id"]

    revoked = asyncio.run(manager.revoke_grant(grant_payload["grant_id"]))
    assert revoked["grant"]["grant_id"] == grant_payload["grant_id"]
    assert asyncio.run(manager.list_grants(profile_id="researcher"))["grants"] == []

    event_types = [event.event_type for event in audit_store.events]
    assert "policy_grant.approval.created" in event_types
    assert "policy_grant.revoked" in event_types


def test_policy_grant_manager_clamps_ttl_to_bounds() -> None:
    from mcp_unified.gateway.policy_grants import (
        APPROVAL_GRANT_MAX_TTL_SECONDS,
        APPROVAL_GRANT_MIN_TTL_SECONDS,
        GatewayPolicyGrantManager,
    )
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    manager = GatewayPolicyGrantManager(policy_grant_store=InMemoryPolicyGrantStore())

    oversized = asyncio.run(
        manager.grant_approval(
            profile_id="researcher",
            subject_type="tool",
            value="web.fetch",
            ttl_seconds=10_000_000,
        )
    )
    assert oversized["grant"]["ttl_seconds"] == APPROVAL_GRANT_MAX_TTL_SECONDS

    undersized = asyncio.run(
        manager.grant_approval(
            profile_id="researcher",
            subject_type="tool",
            value="web.search",
            ttl_seconds=1,
        )
    )
    assert undersized["grant"]["ttl_seconds"] == APPROVAL_GRANT_MIN_TTL_SECONDS


def test_policy_grant_manager_rejects_invalid_requests() -> None:
    from mcp_unified.gateway.policy_grants import (
        GatewayPolicyGrantManagementError,
        GatewayPolicyGrantManager,
    )
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    manager = GatewayPolicyGrantManager(policy_grant_store=InMemoryPolicyGrantStore())

    with pytest.raises(GatewayPolicyGrantManagementError) as excinfo:
        asyncio.run(
            manager.grant_approval(
                profile_id="researcher",
                subject_type="skill",
                value="anything",
            )
        )
    assert excinfo.value.reason_code == "invalid_policy_grant"
    assert excinfo.value.to_payload()["ok"] is False


def test_policy_grant_manager_revoke_missing_grant_raises_not_found() -> None:
    from mcp_unified.gateway.policy_grants import (
        GatewayPolicyGrantManagementError,
        GatewayPolicyGrantManager,
    )
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    manager = GatewayPolicyGrantManager(policy_grant_store=InMemoryPolicyGrantStore())

    with pytest.raises(GatewayPolicyGrantManagementError) as excinfo:
        asyncio.run(manager.revoke_grant("missing-grant"))
    assert excinfo.value.reason_code == "policy_grant_not_found"


def test_bootstrap_config_accepts_policy_grants_section(tmp_path: Path) -> None:
    from mcp_unified.gateway.config import (
        GatewayPolicyGrantStoreConfig,
        GatewayProfileBootstrapConfig,
    )

    config = GatewayProfileBootstrapConfig(
        policy_grants={"kind": "sqlite", "sqlite_path": str(tmp_path / "grants.db")}
    )
    assert isinstance(config.policy_grants, GatewayPolicyGrantStoreConfig)
    assert config.policy_grants.kind == "sqlite"

    default_config = GatewayProfileBootstrapConfig()
    assert default_config.policy_grants is None

    with pytest.raises(ValueError):
        GatewayProfileBootstrapConfig(policy_grants={"kind": "sqlite"})
    with pytest.raises(ValueError):
        GatewayProfileBootstrapConfig(policy_grants={"kind": "redis"})


def test_build_gateway_policy_grant_store_backend_selection(tmp_path: Path) -> None:
    from mcp_unified.gateway.config import (
        GatewayPolicyGrantStoreConfig,
        build_gateway_policy_grant_store,
    )
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore
    from mcp_unified.policy_grants.sqlite import SQLitePolicyGrantStore

    assert build_gateway_policy_grant_store(None) is None
    assert isinstance(
        build_gateway_policy_grant_store(GatewayPolicyGrantStoreConfig(kind="memory")),
        InMemoryPolicyGrantStore,
    )

    sqlite_store = build_gateway_policy_grant_store(
        GatewayPolicyGrantStoreConfig(
            kind="sqlite",
            sqlite_path=tmp_path / "grants.db",
        )
    )
    assert isinstance(sqlite_store, SQLitePolicyGrantStore)
    sqlite_store.close()


def test_bootstrap_profile_gateway_wires_policy_grant_store() -> None:
    from mcp_unified.gateway.bootstrap import bootstrap_profile_gateway
    from mcp_unified.gateway.runtime import GatewayRequestContext
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    class _StaticBackend:
        name = "unit-backend"
        version = "0.0-test"

        def __init__(self) -> None:
            self.calls: list[str] = []

        async def list_tools(self, context: Any) -> list[dict[str, Any]]:
            return [
                {
                    "name": "web.fetch",
                    "description": "Fetch a URL.",
                    "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}}},
                    "metadata": {"category": "web", "capability": "network.fetch"},
                }
            ]

        async def call_tool(
            self,
            name: str,
            arguments: dict[str, Any],
            context: Any,
        ) -> dict[str, Any]:
            self.calls.append(name)
            return {"content": [{"type": "text", "text": "ok"}]}

    backend = _StaticBackend()
    grant_store = InMemoryPolicyGrantStore()
    grant_store.create_grant(
        profile_id="researcher",
        grant_type="approval",
        subject_type="domain",
        value="example.com",
        ttl_seconds=900,
    )

    async def _exercise() -> dict[str, Any]:
        bootstrap = await bootstrap_profile_gateway(
            backend,
            profiles=[
                {
                    "id": "researcher",
                    "name": "Researcher",
                    "policy_document": {
                        "allowed_tools": ["web.fetch"],
                        "permission_rules": [
                            {"pattern": "WebFetch(example.com)", "outcome": "ask"}
                        ],
                    },
                }
            ],
            default_profile_id="researcher",
            policy_grant_store=grant_store,
        )
        assert bootstrap.policy_grant_store is grant_store
        context = GatewayRequestContext(request_id="bootstrap-lease")
        return await bootstrap.runtime.call_tool(
            "web.fetch",
            {"url": "https://example.com/private"},
            context,
        )

    result = asyncio.run(_exercise())
    assert result is not None
    assert backend.calls == ["web.fetch"]


def test_policy_grant_manager_grant_path_lifecycle_with_audit() -> None:
    from mcp_unified.gateway.policy_grants import GatewayPolicyGrantManager
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    audit_store = _RecordingAuditStore()
    manager = GatewayPolicyGrantManager(
        policy_grant_store=InMemoryPolicyGrantStore(),
        audit_store=audit_store,
    )

    created = asyncio.run(
        manager.grant_path(
            profile_id="reviewer",
            prefix="docs\\scratch/./sub",
            actions=("read", "write"),
            ttl_seconds=900,
            session_id="session-1",
            granted_by="operator",
            reason="one-off scratch access",
        )
    )
    grant_payload = created["grant"]
    assert grant_payload["grant_type"] == "path"
    assert grant_payload["value"] == "docs/scratch/sub"
    assert grant_payload["actions"] == ["read", "write"]
    assert grant_payload["session_id"] == "session-1"

    listed = asyncio.run(manager.list_grants(profile_id="reviewer", grant_type="path"))
    assert [grant["grant_id"] for grant in listed["grants"]] == [grant_payload["grant_id"]]

    assert "policy_grant.path.created" in [event.event_type for event in audit_store.events]


def test_policy_grant_manager_grant_path_rejects_invalid_requests() -> None:
    from mcp_unified.gateway.policy_grants import (
        GatewayPolicyGrantManagementError,
        GatewayPolicyGrantManager,
    )
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    manager = GatewayPolicyGrantManager(policy_grant_store=InMemoryPolicyGrantStore())

    with pytest.raises(GatewayPolicyGrantManagementError) as excinfo:
        asyncio.run(
            manager.grant_path(
                profile_id="reviewer",
                prefix="/etc/passwd",
                actions=("read",),
            )
        )
    assert excinfo.value.reason_code == "invalid_policy_grant"

    with pytest.raises(GatewayPolicyGrantManagementError):
        asyncio.run(
            manager.grant_path(
                profile_id="reviewer",
                prefix="docs/scratch",
                actions=("launch_missiles",),
            )
        )
