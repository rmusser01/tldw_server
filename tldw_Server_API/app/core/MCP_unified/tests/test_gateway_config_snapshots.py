"""Tests for standalone MCP gateway config import/export snapshots."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import mcp_unified.gateway.snapshots as snapshot_module
from mcp_unified.gateway.snapshots import (
    GatewayConfigSnapshot,
    GatewayConfigSnapshotManagementError,
    GatewayConfigSnapshotManager,
)
from mcp_unified.profiles.defaults import GATEWAY_DEFAULT_ASSIGNMENT_ID
from mcp_unified.profiles.models import MCPProfile
from mcp_unified.profiles.store import InMemoryProfileAssignmentStore, InMemoryProfileStore
from mcp_unified.storage.models import (
    CredentialGrant,
    ExternalServerDefinition,
    ProfileAssignment,
)
from mcp_unified.storage.sqlite import SQLiteMCPStore


@pytest.mark.asyncio
async def test_export_snapshot_includes_expected_sections(tmp_path: Path) -> None:
    """Export a deterministic secret-safe snapshot from the configured stores."""

    store = SQLiteMCPStore(tmp_path / "gateway.db")
    try:
        await store.upsert_profile(MCPProfile(id="reviewer", name="Reviewer"))
        await store.upsert_assignment(
            ProfileAssignment(
                id=GATEWAY_DEFAULT_ASSIGNMENT_ID,
                profile_id="reviewer",
                is_default=True,
            )
        )
        await store.upsert_assignment(
            ProfileAssignment(
                id="workspace-assignment",
                profile_id="reviewer",
                workspace_id="workspace",
            )
        )
        await store.upsert_server(_server())
        await store.upsert_grant(_grant())

        payload = (await _sqlite_snapshot_manager(store).export_snapshot()).model_dump(
            mode="json"
        )

        assert payload["schema"] == "mcp_unified.gateway.config_snapshot"
        assert payload["version"] == 1
        assert [profile["id"] for profile in payload["profiles"]] == ["reviewer"]
        assert payload["default_assignment"]["id"] == GATEWAY_DEFAULT_ASSIGNMENT_ID
        assert payload["default_assignment"]["profile_id"] == "reviewer"
        assert [server["id"] for server in payload["external_servers"]] == ["search"]
        assert payload["external_servers"][0]["credential_slots"] == ["api_key"]
        assert [grant["id"] for grant in payload["credential_grants"]] == [
            "grant-search"
        ]
        assert "workspace-assignment" not in json.dumps(payload)
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_import_snapshot_dry_run_reports_plan_without_mutation() -> None:
    """Dry-run import validates references and reports planned writes only."""

    profile_store = InMemoryProfileStore()
    assignment_store = InMemoryProfileAssignmentStore()
    external_store = _InMemoryExternalRegistryStore()
    grant_store = _InMemoryCredentialGrantStore()
    manager = GatewayConfigSnapshotManager(
        profile_store=profile_store,
        assignment_store=assignment_store,
        external_registry_store=external_store,
        credential_grant_store=grant_store,
    )

    payload = await manager.import_snapshot(_snapshot_payload(), dry_run=True)

    assert payload == {
        "ok": True,
        "dry_run": True,
        "plan": {
            "actions": [
                {"action": "upsert_profile", "target_id": "reviewer"},
                {
                    "action": "set_default_assignment",
                    "target_id": GATEWAY_DEFAULT_ASSIGNMENT_ID,
                },
                {"action": "upsert_external_server", "target_id": "search"},
                {"action": "upsert_credential_grant", "target_id": "grant-search"},
            ],
            "counts": {
                "credential_grants": 1,
                "default_assignment": 1,
                "external_servers": 1,
                "profiles": 1,
            },
        },
    }
    assert await profile_store.list_profiles() == []
    assert await assignment_store.list_assignments() == []
    assert await external_store.list_server_definitions() == []
    assert await grant_store.list_grants() == []


@pytest.mark.parametrize(
    "field_name",
    ["metadata", "provenance"],
)
@pytest.mark.parametrize(
    "secret_key",
    [
        "secret",
        "token",
        "password",
        "api_key",
        "authorization",
        "headers",
        "env",
        "credential_value",
    ],
)
def test_snapshot_validation_rejects_secret_looking_metadata(
    field_name: str,
    secret_key: str,
) -> None:
    """Snapshots reject grant metadata/provenance keys that look like secrets."""

    payload = _snapshot_payload()
    payload["credential_grants"][0][field_name] = {
        "nested": {secret_key: "do-not-export"}
    }

    with pytest.raises(ValueError, match="secret material"):
        GatewayConfigSnapshot.model_validate(payload)


@pytest.mark.parametrize(
    "server_patch",
    [
        {"command": ["node", "server.js", "--token=abc"]},
        {"command": ["node", "server.js", "api_key=abc"]},
        {"command": ["node", "server.js", "PASSWORD=abc"]},
        {
            "transport": "websocket",
            "command": [],
            "url": "https://user:pass@example.test/mcp",
        },
        {
            "transport": "websocket",
            "command": [],
            "url": "https://example.test/mcp?token=abc",
        },
    ],
)
def test_snapshot_validation_rejects_external_server_inline_secrets(
    server_patch: dict[str, object],
) -> None:
    """Snapshots reject inline secret material in external-server launch data."""

    payload = _snapshot_payload()
    payload["external_servers"][0].update(server_patch)

    with pytest.raises(ValueError, match="secret material"):
        GatewayConfigSnapshot.model_validate(payload)


def test_snapshot_inline_secret_validation_allows_malformed_missing_command() -> None:
    """Defensive validation treats missing command lists as empty."""

    server = ExternalServerDefinition.model_construct(
        id="disabled-search",
        name="Disabled Search",
        transport="stdio",
        command=None,
        url=None,
        cwd=None,
        env_allowlist=[],
        credential_slots=[],
        metadata={},
        provenance={},
        enabled=False,
        auto_start=False,
    )

    snapshot_module._reject_external_server_inline_secrets(server)


def test_snapshot_validation_rejects_non_gateway_default_assignment_id() -> None:
    """Snapshots cannot import arbitrary default assignment ids."""

    payload = _snapshot_payload()
    payload["default_assignment"]["id"] = "workspace-assignment"

    with pytest.raises(ValueError, match="default assignment id"):
        GatewayConfigSnapshot.model_validate(payload)


@pytest.mark.asyncio
async def test_import_snapshot_validates_before_first_write() -> None:
    """Reference validation fails without writing earlier valid snapshot sections."""

    profile_store = InMemoryProfileStore()
    assignment_store = InMemoryProfileAssignmentStore()
    external_store = _InMemoryExternalRegistryStore()
    grant_store = _InMemoryCredentialGrantStore()
    manager = GatewayConfigSnapshotManager(
        profile_store=profile_store,
        assignment_store=assignment_store,
        external_registry_store=external_store,
        credential_grant_store=grant_store,
    )
    payload = _snapshot_payload()
    payload["credential_grants"][0]["credential_slot"] = "missing_slot"

    with pytest.raises(GatewayConfigSnapshotManagementError) as exc_info:
        await manager.import_snapshot(payload)

    assert exc_info.value.reason_code == "config_snapshot_invalid_reference"
    assert await profile_store.list_profiles() == []
    assert await assignment_store.list_assignments() == []
    assert await external_store.list_server_definitions() == []
    assert await grant_store.list_grants() == []


@pytest.mark.asyncio
async def test_import_snapshot_handles_malformed_server_slots_before_first_write() -> None:
    """Malformed model instances fail reference validation without mutation."""

    profile_store = InMemoryProfileStore()
    assignment_store = InMemoryProfileAssignmentStore()
    external_store = _InMemoryExternalRegistryStore()
    grant_store = _InMemoryCredentialGrantStore()
    manager = GatewayConfigSnapshotManager(
        profile_store=profile_store,
        assignment_store=assignment_store,
        external_registry_store=external_store,
        credential_grant_store=grant_store,
    )
    snapshot = GatewayConfigSnapshot.model_construct(
        profiles=[MCPProfile(id="reviewer", name="Reviewer")],
        default_assignment=ProfileAssignment(
            id=GATEWAY_DEFAULT_ASSIGNMENT_ID,
            profile_id="reviewer",
            is_default=True,
        ),
        external_servers=[
            ExternalServerDefinition.model_construct(
                id="search",
                name="Search",
                transport="stdio",
                command=["node", "server.js"],
                credential_slots=None,
                metadata={},
                provenance={},
                enabled=True,
                auto_start=False,
            )
        ],
        credential_grants=[_grant()],
    )

    with pytest.raises(GatewayConfigSnapshotManagementError) as exc_info:
        await manager.import_snapshot(snapshot)

    assert exc_info.value.reason_code == "config_snapshot_invalid_reference"
    assert await profile_store.list_profiles() == []
    assert await assignment_store.list_assignments() == []
    assert await external_store.list_server_definitions() == []
    assert await grant_store.list_grants() == []


@pytest.mark.asyncio
async def test_import_snapshot_reports_partial_write_failure() -> None:
    """Best-effort imports report applied and failed action ids on store failure."""

    manager = GatewayConfigSnapshotManager(
        profile_store=InMemoryProfileStore(),
        assignment_store=InMemoryProfileAssignmentStore(),
        external_registry_store=_InMemoryExternalRegistryStore(),
        credential_grant_store=_FailingCredentialGrantStore(),
    )

    with pytest.raises(GatewayConfigSnapshotManagementError) as exc_info:
        await manager.import_snapshot(_snapshot_payload())

    payload = exc_info.value.to_payload()
    assert payload["reason_code"] == "config_snapshot_import_failed"
    assert payload["applied_actions"] == [
        {"action": "upsert_profile", "target_id": "reviewer"},
        {
            "action": "set_default_assignment",
            "target_id": GATEWAY_DEFAULT_ASSIGNMENT_ID,
        },
        {"action": "upsert_external_server", "target_id": "search"},
    ]
    assert payload["failed_actions"] == [
        {
            "action": "upsert_credential_grant",
            "reason_code": "RuntimeError",
            "target_id": "grant-search",
        }
    ]
    assert "broker-secret-value" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_sqlite_snapshot_round_trip_preserves_semantic_content(
    tmp_path: Path,
) -> None:
    """A SQLite snapshot can be imported into a fresh store and exported again."""

    source = SQLiteMCPStore(tmp_path / "source.db")
    target = SQLiteMCPStore(tmp_path / "target.db")
    try:
        await source.upsert_profile(MCPProfile(id="reviewer", name="Reviewer"))
        await source.upsert_assignment(
            ProfileAssignment(
                id=GATEWAY_DEFAULT_ASSIGNMENT_ID,
                profile_id="reviewer",
                is_default=True,
            )
        )
        await source.upsert_server(_server())
        await source.upsert_grant(_grant())

        exported = await _sqlite_snapshot_manager(source).export_snapshot()
        import_payload = await _sqlite_snapshot_manager(target).import_snapshot(
            exported.model_dump(mode="json")
        )
        reexported = await _sqlite_snapshot_manager(target).export_snapshot()

        assert import_payload["ok"] is True
        assert import_payload["dry_run"] is False
        assert _semantic_snapshot(reexported) == _semantic_snapshot(exported)
    finally:
        await source.aclose()
        await target.aclose()


class _InMemoryExternalRegistryStore:
    """Small copy-isolated external registry store for snapshot tests."""

    def __init__(
        self,
        servers: list[ExternalServerDefinition | Mapping[str, Any]] | None = None,
    ) -> None:
        self.servers: dict[str, ExternalServerDefinition] = {}
        for server in servers or ():
            validated = ExternalServerDefinition.model_validate(server)
            self.servers[validated.id] = validated.model_copy(deep=True)

    async def get_server(self, server_id: str) -> ExternalServerDefinition | None:
        server = self.servers.get(server_id)
        return None if server is None else server.model_copy(deep=True)

    async def list_servers(self) -> list[ExternalServerDefinition]:
        return await self.list_server_definitions()

    async def list_server_definitions(
        self,
        *,
        enabled: bool | None = None,
    ) -> list[ExternalServerDefinition]:
        return [
            server.model_copy(deep=True)
            for server in sorted(self.servers.values(), key=lambda item: item.id)
            if enabled is None or server.enabled is enabled
        ]

    async def create_server(
        self,
        server: ExternalServerDefinition,
    ) -> ExternalServerDefinition:
        return await self.upsert_server(server)

    async def upsert_server(
        self,
        server: ExternalServerDefinition,
    ) -> ExternalServerDefinition:
        self.servers[server.id] = server.model_copy(deep=True)
        return server.model_copy(deep=True)

    async def update_server(
        self,
        server: ExternalServerDefinition,
    ) -> ExternalServerDefinition | None:
        if server.id not in self.servers:
            return None
        return await self.upsert_server(server)

    async def delete_server(self, server_id: str) -> bool:
        return self.servers.pop(server_id, None) is not None


class _InMemoryCredentialGrantStore:
    """Small copy-isolated credential grant store for snapshot tests."""

    def __init__(
        self,
        grants: list[CredentialGrant | Mapping[str, Any]] | None = None,
    ) -> None:
        self.grants: dict[str, CredentialGrant] = {}
        for grant in grants or ():
            validated = CredentialGrant.model_validate(grant)
            self.grants[validated.id] = validated.model_copy(deep=True)

    async def get_grant(self, grant_id: str) -> CredentialGrant | None:
        grant = self.grants.get(grant_id)
        return None if grant is None else grant.model_copy(deep=True)

    async def list_grants(
        self,
        *,
        profile_id: str | None = None,
        external_server_id: str | None = None,
    ) -> list[CredentialGrant]:
        return [
            grant.model_copy(deep=True)
            for grant in sorted(self.grants.values(), key=lambda item: item.id)
            if (profile_id is None or grant.profile_id == profile_id)
            and (
                external_server_id is None
                or grant.external_server_id == external_server_id
            )
        ]

    async def create_grant(self, grant: CredentialGrant) -> CredentialGrant:
        return await self.upsert_grant(grant)

    async def upsert_grant(self, grant: CredentialGrant) -> CredentialGrant:
        self.grants[grant.id] = grant.model_copy(deep=True)
        return grant.model_copy(deep=True)

    async def delete_grant(self, grant_id: str) -> bool:
        return self.grants.pop(grant_id, None) is not None


class _FailingCredentialGrantStore(_InMemoryCredentialGrantStore):
    """Grant store that fails after snapshot validation reaches write phase."""

    async def upsert_grant(self, grant: CredentialGrant) -> CredentialGrant:
        del grant
        raise RuntimeError("credential grant write failed: broker-secret-value")


def _sqlite_snapshot_manager(store: SQLiteMCPStore) -> GatewayConfigSnapshotManager:
    return GatewayConfigSnapshotManager(
        profile_store=store,
        assignment_store=store,
        external_registry_store=store,
        credential_grant_store=store,
        audit_store=store,
    )


def _server() -> ExternalServerDefinition:
    return ExternalServerDefinition(
        id="search",
        name="Search",
        transport="stdio",
        command=["node", "server.js"],
        credential_slots=["api_key"],
        metadata={"label": "Search MCP"},
        provenance={"source": "test"},
    )


def _grant() -> CredentialGrant:
    return CredentialGrant(
        id="grant-search",
        profile_id="reviewer",
        broker_id="env",
        credential_slot="api_key",
        external_server_id="search",
        metadata={"label": "Search token grant"},
        provenance={"source": "test"},
    )


def _snapshot_payload() -> dict[str, Any]:
    return {
        "schema": "mcp_unified.gateway.config_snapshot",
        "version": 1,
        "profiles": [MCPProfile(id="reviewer", name="Reviewer").model_dump(mode="json")],
        "default_assignment": ProfileAssignment(
            id=GATEWAY_DEFAULT_ASSIGNMENT_ID,
            profile_id="reviewer",
            is_default=True,
        ).model_dump(mode="json"),
        "external_servers": [_server().model_dump(mode="json")],
        "credential_grants": [_grant().model_dump(mode="json")],
    }


def _semantic_snapshot(snapshot: GatewayConfigSnapshot) -> dict[str, Any]:
    payload = snapshot.model_dump(mode="json")
    for section in ("profiles", "external_servers", "credential_grants"):
        for item in payload[section]:
            item.pop("created_at", None)
            item.pop("updated_at", None)
    if payload["default_assignment"] is not None:
        payload["default_assignment"].pop("created_at", None)
        payload["default_assignment"].pop("updated_at", None)
    return payload
