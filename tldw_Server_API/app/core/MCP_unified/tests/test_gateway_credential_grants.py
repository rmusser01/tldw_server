from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from mcp_unified.gateway.credential_grants import (
    GatewayCredentialGrantManagementError,
    GatewayCredentialGrantManager,
)
from mcp_unified.gateway.external_registry import GatewayStoreMetadata
from mcp_unified.interfaces.storage import CredentialGrantAlreadyExistsError
from mcp_unified.profiles.models import MCPProfile
from mcp_unified.profiles.store import InMemoryProfileStore
from mcp_unified.storage.models import (
    AuditEvent,
    CredentialGrant,
    ExternalServerDefinition,
)
from mcp_unified.storage.sqlite import SQLiteMCPStore


class _InMemoryCredentialGrantStore:
    """Credential-grant test double with atomic create behavior."""

    def __init__(self) -> None:
        self._grants: dict[str, CredentialGrant] = {}

    async def get_grant(self, grant_id: str) -> CredentialGrant | None:
        grant = self._grants.get(grant_id)
        return None if grant is None else grant.model_copy(deep=True)

    async def list_grants(
        self,
        *,
        profile_id: str | None = None,
        external_server_id: str | None = None,
    ) -> list[CredentialGrant]:
        grants = [
            grant
            for grant in self._grants.values()
            if (profile_id is None or grant.profile_id == profile_id)
            and (
                external_server_id is None
                or grant.external_server_id == external_server_id
            )
        ]
        return [
            grant.model_copy(deep=True)
            for grant in sorted(grants, key=lambda item: item.id)
        ]

    async def create_grant(self, grant: CredentialGrant) -> CredentialGrant:
        if grant.id in self._grants:
            raise CredentialGrantAlreadyExistsError(grant.id)
        self._grants[grant.id] = grant.model_copy(deep=True)
        return grant.model_copy(deep=True)

    async def upsert_grant(self, grant: CredentialGrant) -> CredentialGrant:
        self._grants[grant.id] = grant.model_copy(deep=True)
        return grant.model_copy(deep=True)

    async def delete_grant(self, grant_id: str) -> bool:
        return self._grants.pop(grant_id, None) is not None


class _InMemoryExternalRegistryStore:
    """External-registry test double for grant reference validation."""

    def __init__(
        self,
        servers: list[ExternalServerDefinition | Mapping[str, Any]] | None = None,
    ) -> None:
        self._servers = {
            server.id: server
            for server in (
                ExternalServerDefinition.model_validate(item)
                for item in (servers or [])
            )
        }

    async def get_server(
        self,
        server_id: str,
    ) -> ExternalServerDefinition | None:
        server = self._servers.get(server_id)
        return None if server is None else server.model_copy(deep=True)


class _AuditStore:
    """Audit sink test double."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def append_event(self, event: AuditEvent) -> AuditEvent:
        self.events.append(event.model_copy(deep=True))
        return event.model_copy(deep=True)


def _server(server_id: str = "github-mcp") -> ExternalServerDefinition:
    return ExternalServerDefinition(
        id=server_id,
        name=f"Server {server_id}",
        transport="stdio",
        command=["node", "server.js"],
        credential_slots=["github_token"],
    )


def _manager(
    *,
    grant_store: _InMemoryCredentialGrantStore | None = None,
    profile_store: InMemoryProfileStore | None = None,
    external_store: _InMemoryExternalRegistryStore | None = None,
    audit_store: _AuditStore | None = None,
) -> GatewayCredentialGrantManager:
    return GatewayCredentialGrantManager(
        credential_grant_store=grant_store or _InMemoryCredentialGrantStore(),
        profile_store=profile_store,
        external_registry_store=external_store,
        audit_store=audit_store,
        store_metadata=GatewayStoreMetadata(kind="memory", persistent=False),
    )


def _grant_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "grant-one",
        "profile_id": "reviewer",
        "broker_id": "env-broker",
        "credential_slot": "github_token",
        "external_server_id": "github-mcp",
        "scopes": ["repo:read"],
        "metadata": {"label": "GitHub read token"},
        "provenance": {"source": "test"},
    }
    payload.update(overrides)
    return payload


def test_gateway_credential_grant_manager_crud_and_audit() -> None:
    async def run() -> None:
        audit_store = _AuditStore()
        manager = _manager(
            profile_store=InMemoryProfileStore(
                [MCPProfile(id="reviewer", name="Reviewer")]
            ),
            external_store=_InMemoryExternalRegistryStore([_server()]),
            audit_store=audit_store,
        )

        created = await manager.create_grant(_grant_payload())
        listed = await manager.list_grants(profile_id="reviewer")
        shown = await manager.show_grant("grant-one")
        patched = await manager.patch_grant(
            "grant-one",
            {"metadata": {"label": "Updated"}, "enabled": False},
        )
        deleted = await manager.delete_grant("grant-one")

        assert created["ok"] is True
        assert created["grant"]["id"] == "grant-one"
        assert created["store"] == {"kind": "memory", "persistent": False}
        assert [grant["id"] for grant in listed["grants"]] == ["grant-one"]
        assert shown["grant"]["credential_slot"] == "github_token"
        assert patched["grant"]["metadata"] == {"label": "Updated"}
        assert patched["grant"]["enabled"] is False
        assert deleted == {
            "ok": True,
            "grant_id": "grant-one",
            "store": {"kind": "memory", "persistent": False},
        }
        assert [event.event_type for event in audit_store.events] == [
            "credential_grant.created",
            "credential_grant.patched",
            "credential_grant.deleted",
        ]

    asyncio.run(run())


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
def test_gateway_credential_grant_manager_rejects_secret_looking_keys(
    secret_key: str,
) -> None:
    async def run() -> None:
        manager = _manager()

        with pytest.raises(
            GatewayCredentialGrantManagementError,
            match="secret material",
        ) as create_error:
            await manager.create_grant(
                _grant_payload(metadata={"nested": {secret_key: "do-not-store"}})
            )

        await manager.create_grant(_grant_payload(metadata={"label": "safe"}))
        with pytest.raises(
            GatewayCredentialGrantManagementError,
            match="secret material",
        ) as patch_error:
            await manager.patch_grant(
                "grant-one",
                {"provenance": {"nested": {secret_key: "do-not-store"}}},
            )

        assert create_error.value.reason_code == "credential_grant_secret_material_rejected"
        assert patch_error.value.reason_code == "credential_grant_secret_material_rejected"
        assert (await manager.show_grant("grant-one"))["grant"]["provenance"] == {
            "source": "test"
        }

    asyncio.run(run())


def test_gateway_credential_grant_manager_validates_profile_and_server_refs() -> None:
    async def run() -> None:
        manager = _manager(
            profile_store=InMemoryProfileStore(),
            external_store=_InMemoryExternalRegistryStore(),
        )

        with pytest.raises(GatewayCredentialGrantManagementError) as profile_error:
            await manager.create_grant(_grant_payload())

        assert profile_error.value.reason_code == "profile_not_found"

        manager = _manager(
            profile_store=InMemoryProfileStore(
                [MCPProfile(id="reviewer", name="Reviewer")]
            ),
            external_store=_InMemoryExternalRegistryStore(),
        )

        with pytest.raises(GatewayCredentialGrantManagementError) as server_error:
            await manager.create_grant(_grant_payload())

        assert server_error.value.reason_code == "external_server_not_found"

    asyncio.run(run())


def test_gateway_credential_grant_manager_duplicate_create_does_not_replace() -> None:
    async def run() -> None:
        manager = _manager()
        await manager.create_grant(_grant_payload(broker_id="first-broker"))

        with pytest.raises(GatewayCredentialGrantManagementError) as exc_info:
            await manager.create_grant(_grant_payload(broker_id="second-broker"))

        assert exc_info.value.reason_code == "credential_grant_already_exists"
        assert (await manager.show_grant("grant-one"))["grant"]["broker_id"] == "first-broker"

    asyncio.run(run())


def test_gateway_credential_grant_manager_patch_rejects_non_string_server_id() -> None:
    async def run() -> None:
        manager = _manager()
        await manager.create_grant(_grant_payload())

        with pytest.raises(GatewayCredentialGrantManagementError) as exc_info:
            await manager.patch_grant("grant-one", {"external_server_id": 123})

        assert exc_info.value.reason_code == "invalid_credential_grant_patch"
        assert (
            await manager.show_grant("grant-one")
        )["grant"]["external_server_id"] == "github-mcp"

    asyncio.run(run())


def test_sqlite_create_grant_rejects_duplicate_without_replacing(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteMCPStore(tmp_path / "gateway.db")
        await store.create_profile(MCPProfile(id="reviewer", name="Reviewer"))
        await store.create_grant(CredentialGrant(**_grant_payload(broker_id="first")))

        with pytest.raises(CredentialGrantAlreadyExistsError):
            await store.create_grant(
                CredentialGrant(**_grant_payload(broker_id="second"))
            )

        stored = await store.get_grant("grant-one")
        assert stored is not None
        assert stored.broker_id == "first"
        store.close()

    asyncio.run(run())
