"""Tests for standalone MCP gateway external registry management helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mcp_unified.gateway.external_registry import (
    GatewayExternalRegistryManagementError,
    GatewayExternalRegistryManager,
    GatewayStoreMetadata,
)
from mcp_unified.interfaces.storage import ExternalServerAlreadyExistsError
from mcp_unified.storage.models import (
    AuditEvent,
    CredentialGrant,
    ExternalServerDefinition,
)

UTC = timezone.utc


class InMemoryExternalRegistryStore:
    """Small copy-isolated external registry store test double."""

    def __init__(self, servers: list[ExternalServerDefinition] | None = None) -> None:
        self.servers: dict[str, ExternalServerDefinition] = {}
        for server in servers or ():
            self.servers[server.id] = server.model_copy(deep=True)

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
        servers = [
            server
            for server in self.servers.values()
            if enabled is None or server.enabled is enabled
        ]
        return [
            server.model_copy(deep=True)
            for server in sorted(servers, key=lambda item: item.id)
        ]

    async def create_server(
        self,
        server: ExternalServerDefinition,
    ) -> ExternalServerDefinition:
        if server.id in self.servers:
            raise ExternalServerAlreadyExistsError(server.id)
        self.servers[server.id] = server.model_copy(deep=True)
        return self.servers[server.id].model_copy(deep=True)

    async def upsert_server(
        self,
        server: ExternalServerDefinition,
    ) -> ExternalServerDefinition:
        self.servers[server.id] = server.model_copy(deep=True)
        return self.servers[server.id].model_copy(deep=True)

    async def delete_server(self, server_id: str) -> bool:
        return self.servers.pop(server_id, None) is not None


class StaleDeleteExternalRegistryStore(InMemoryExternalRegistryStore):
    """Registry store double that preloads a server but reports stale delete."""

    async def delete_server(self, server_id: str) -> bool:
        del server_id
        return False


class InMemoryCredentialGrantStore:
    """Small copy-isolated credential grant store test double."""

    def __init__(self, grants: list[CredentialGrant] | None = None) -> None:
        self.grants = [grant.model_copy(deep=True) for grant in grants or ()]

    async def get_grant(self, grant_id: str) -> CredentialGrant | None:
        for grant in self.grants:
            if grant.id == grant_id:
                return grant.model_copy(deep=True)
        return None

    async def list_grants(
        self,
        *,
        profile_id: str | None = None,
        external_server_id: str | None = None,
    ) -> list[CredentialGrant]:
        return [
            grant.model_copy(deep=True)
            for grant in self.grants
            if (profile_id is None or grant.profile_id == profile_id)
            and (
                external_server_id is None
                or grant.external_server_id == external_server_id
            )
        ]

    async def upsert_grant(self, grant: CredentialGrant) -> CredentialGrant:
        self.grants = [stored for stored in self.grants if stored.id != grant.id]
        self.grants.append(grant.model_copy(deep=True))
        return grant.model_copy(deep=True)

    async def delete_grant(self, grant_id: str) -> bool:
        before = len(self.grants)
        self.grants = [grant for grant in self.grants if grant.id != grant_id]
        return len(self.grants) != before


class InMemoryAuditStore:
    """Small copy-isolated audit test double for manager audit assertions."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def append_event(self, event: AuditEvent) -> AuditEvent:
        stored = event.model_copy(deep=True)
        self.events.append(stored)
        return stored.model_copy(deep=True)

    async def query_events(
        self,
        *,
        actor_id: str | None = None,
        profile_id: str | None = None,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> list[AuditEvent]:
        events = [
            event
            for event in self.events
            if (actor_id is None or event.actor_id == actor_id)
            and (profile_id is None or event.profile_id == profile_id)
            and (event_type is None or event.event_type == event_type)
        ]
        if limit is not None:
            events = events[:limit]
        return [event.model_copy(deep=True) for event in events]


def _manager(
    external_registry_store: InMemoryExternalRegistryStore,
    *,
    credential_grant_store: InMemoryCredentialGrantStore | None = None,
    audit_store: InMemoryAuditStore | None = None,
) -> GatewayExternalRegistryManager:
    return GatewayExternalRegistryManager(
        external_registry_store=external_registry_store,
        credential_grant_store=credential_grant_store,
        audit_store=audit_store,
        store_metadata=GatewayStoreMetadata(kind="memory", persistent=False),
    )


def _server(**overrides: object) -> ExternalServerDefinition:
    values: dict[str, object] = {
        "id": "search",
        "name": "Search",
        "transport": "websocket",
        "url": "wss://search.example.test/mcp",
    }
    values.update(overrides)
    return ExternalServerDefinition(**values)


def _grant(
    *,
    grant_id: str = "grant-1",
    profile_id: str = "profile-1",
    broker_id: str = "broker-1",
    credential_slot: str = "api_key",
    external_server_id: str = "search",
    enabled: bool = True,
) -> CredentialGrant:
    return CredentialGrant(
        id=grant_id,
        profile_id=profile_id,
        broker_id=broker_id,
        credential_slot=credential_slot,
        external_server_id=external_server_id,
        enabled=enabled,
    )


@pytest.mark.asyncio
async def test_gateway_external_registry_lists_sorted_servers_with_metadata_and_filter() -> None:
    manager = _manager(
        InMemoryExternalRegistryStore(
            [
                _server(id="zeta", name="Zeta", enabled=False),
                _server(id="alpha", name="Alpha", enabled=True),
            ]
        )
    )

    all_payload = await manager.list_servers()
    enabled_payload = await manager.list_servers(enabled=True)

    assert all_payload["ok"] is True
    assert [server["id"] for server in all_payload["servers"]] == ["alpha", "zeta"]
    assert all_payload["store"] == {"kind": "memory", "persistent": False}
    assert [server["id"] for server in enabled_payload["servers"]] == ["alpha"]


@pytest.mark.asyncio
async def test_gateway_external_registry_shows_json_safe_copy_isolated_server() -> None:
    created_at = datetime(2026, 5, 31, 12, 0, tzinfo=UTC)
    server = _server(
        metadata={"nested": {"value": "original"}},
        created_at=created_at,
        updated_at=created_at,
    )
    manager = _manager(InMemoryExternalRegistryStore([server]))

    payload = await manager.show_server("search")

    assert payload == {
        "ok": True,
        "server": server.model_dump(mode="json"),
        "store": {"kind": "memory", "persistent": False},
    }
    assert isinstance(payload["server"]["created_at"], str)

    payload["server"]["metadata"]["nested"]["value"] = "mutated"
    second_payload = await manager.show_server("search")
    assert second_payload["server"]["metadata"]["nested"]["value"] == "original"


@pytest.mark.asyncio
async def test_gateway_external_registry_audits_missing_show_server() -> None:
    audit_store = InMemoryAuditStore()
    manager = _manager(InMemoryExternalRegistryStore(), audit_store=audit_store)

    with pytest.raises(GatewayExternalRegistryManagementError) as exc_info:
        await manager.show_server("missing")

    assert exc_info.value.reason_code == "external_server_not_found"
    assert exc_info.value.to_payload()["server_id"] == "missing"
    assert [event.event_type for event in audit_store.events] == [
        "external_server.show_failed"
    ]
    assert audit_store.events[0].target_type == "external_server"
    assert audit_store.events[0].target_id == "missing"
    assert audit_store.events[0].payload == {
        "server_id": "missing",
        "reason_code": "external_server_not_found",
    }


@pytest.mark.asyncio
async def test_gateway_external_registry_create_normalizes_id_and_name_and_audits() -> None:
    audit_store = InMemoryAuditStore()
    store = InMemoryExternalRegistryStore()
    manager = _manager(store, audit_store=audit_store)
    created_at = datetime(2020, 1, 1, 12, 0, tzinfo=UTC)

    payload = await manager.create_server(
        {
            "id": " search ",
            "name": "  Search Server  ",
            "transport": "websocket",
            "url": "wss://search.example.test/mcp",
            "created_at": created_at.isoformat(),
            "updated_at": created_at.isoformat(),
        }
    )

    assert payload["ok"] is True
    assert payload["server"]["id"] == "search"
    assert payload["server"]["name"] == "Search Server"
    assert payload["server"]["created_at"] == "2020-01-01T12:00:00Z"
    assert payload["server"]["updated_at"] != created_at.isoformat()
    stored = await store.get_server("search")
    assert stored is not None
    assert stored.name == "Search Server"
    assert stored.updated_at > created_at
    assert [event.event_type for event in audit_store.events] == [
        "external_server.created"
    ]
    assert audit_store.events[0].payload == {"server_id": "search"}


@pytest.mark.asyncio
async def test_gateway_external_registry_create_validates_slug_server_ids() -> None:
    manager = _manager(InMemoryExternalRegistryStore())

    with pytest.raises(GatewayExternalRegistryManagementError) as exc_info:
        await manager.create_server(
            {
                "id": "Bad.Id",
                "name": "Bad",
                "transport": "websocket",
                "url": "wss://bad.example.test/mcp",
            }
        )

    assert exc_info.value.reason_code == "invalid_external_server_request"


@pytest.mark.asyncio
async def test_gateway_external_registry_duplicate_create_audits_failure() -> None:
    audit_store = InMemoryAuditStore()
    manager = _manager(
        InMemoryExternalRegistryStore([_server(id="search")]),
        audit_store=audit_store,
    )

    with pytest.raises(GatewayExternalRegistryManagementError) as exc_info:
        await manager.create_server(
            {
                "id": "search",
                "name": "Duplicate Search",
                "transport": "websocket",
                "url": "wss://other.example.test/mcp",
            }
        )

    assert exc_info.value.reason_code == "external_server_already_exists"
    assert exc_info.value.to_payload()["server_id"] == "search"
    assert [event.event_type for event in audit_store.events] == [
        "external_server.create_failed"
    ]
    assert audit_store.events[0].payload == {
        "server_id": "search",
        "reason_code": "external_server_already_exists",
    }


@pytest.mark.asyncio
async def test_gateway_external_registry_rejects_enabled_websocket_non_ws_url() -> None:
    manager = _manager(InMemoryExternalRegistryStore())

    with pytest.raises(GatewayExternalRegistryManagementError) as exc_info:
        await manager.create_server(
            {
                "id": "search",
                "name": "Search",
                "transport": "websocket",
                "url": "https://search.example.test/mcp",
            }
        )

    assert exc_info.value.reason_code == "invalid_external_server_request"


@pytest.mark.asyncio
async def test_gateway_external_registry_patch_replaces_allowed_fields_and_audits_changes() -> None:
    original_updated_at = datetime(2020, 1, 1, 12, 0, tzinfo=UTC)
    store = InMemoryExternalRegistryStore(
        [
            _server(
                id="search",
                name="Search",
                transport="stdio",
                command=["old-search"],
                url=None,
                cwd="/workspace/old",
                env_allowlist=["OLD_TOKEN"],
                credential_slots=["old_key"],
                metadata={"old": True},
                provenance={"source": "old"},
                auto_start=False,
                updated_at=original_updated_at,
            )
        ]
    )
    audit_store = InMemoryAuditStore()
    manager = _manager(store, audit_store=audit_store)

    payload = await manager.patch_server(
        "search",
        {
            "name": "Search MCP",
            "transport": "websocket",
            "command": [],
            "url": "wss://search.example.test/mcp",
            "cwd": "/workspace/new",
            "env_allowlist": ["SEARCH_TOKEN"],
            "credential_slots": ["old_key", "new_key"],
            "metadata": {"new": True},
            "provenance": {"source": "test"},
            "enabled": False,
            "auto_start": True,
        },
    )

    server = payload["server"]
    assert payload["ok"] is True
    assert payload["store"] == {"kind": "memory", "persistent": False}
    assert server["name"] == "Search MCP"
    assert server["transport"] == "websocket"
    assert server["command"] == []
    assert server["url"] == "wss://search.example.test/mcp"
    assert server["cwd"] == "/workspace/new"
    assert server["env_allowlist"] == ["SEARCH_TOKEN"]
    assert server["credential_slots"] == ["old_key", "new_key"]
    assert server["metadata"] == {"new": True}
    assert server["provenance"] == {"source": "test"}
    assert server["enabled"] is False
    assert server["auto_start"] is True
    assert server["updated_at"] != original_updated_at.isoformat()
    assert [event.event_type for event in audit_store.events] == [
        "external_server.patched"
    ]
    assert set(audit_store.events[0].payload["changed_fields"]) == {
        "name",
        "transport",
        "command",
        "url",
        "cwd",
        "env_allowlist",
        "credential_slots",
        "metadata",
        "provenance",
        "enabled",
        "auto_start",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("patch_document", [{}, {"created_at": "2026-05-31"}])
async def test_gateway_external_registry_rejects_empty_and_unsupported_patch_fields(
    patch_document: dict[str, object],
) -> None:
    manager = _manager(InMemoryExternalRegistryStore([_server()]))

    with pytest.raises(GatewayExternalRegistryManagementError) as exc_info:
        await manager.patch_server("search", patch_document)

    assert exc_info.value.reason_code == "invalid_external_server_patch"


@pytest.mark.asyncio
async def test_gateway_external_registry_enabled_server_credential_slot_addition_is_allowed() -> None:
    store = InMemoryExternalRegistryStore(
        [_server(credential_slots=["api_key"], enabled=True)]
    )
    manager = _manager(store)

    payload = await manager.patch_server(
        "search",
        {"credential_slots": ["api_key", "oauth_token"]},
    )

    assert payload["server"]["enabled"] is True
    assert payload["server"]["credential_slots"] == ["api_key", "oauth_token"]


@pytest.mark.asyncio
async def test_gateway_external_registry_enabled_server_credential_slot_removal_requires_disabled_server() -> None:
    manager = _manager(
        InMemoryExternalRegistryStore(
            [_server(credential_slots=["api_key", "oauth_token"], enabled=True)]
        ),
        credential_grant_store=InMemoryCredentialGrantStore(),
    )

    with pytest.raises(GatewayExternalRegistryManagementError) as exc_info:
        await manager.patch_server("search", {"credential_slots": ["api_key"]})

    assert exc_info.value.reason_code == "credential_slot_change_requires_disabled_server"


@pytest.mark.asyncio
async def test_gateway_external_registry_patch_can_disable_enabled_server_and_remove_slots_without_grants() -> None:
    manager = _manager(
        InMemoryExternalRegistryStore(
            [_server(credential_slots=["api_key", "oauth_token"], enabled=True)]
        ),
        credential_grant_store=InMemoryCredentialGrantStore(),
    )

    payload = await manager.patch_server(
        "search",
        {"enabled": False, "credential_slots": ["api_key"]},
    )

    assert payload["server"]["enabled"] is False
    assert payload["server"]["credential_slots"] == ["api_key"]


@pytest.mark.asyncio
async def test_gateway_external_registry_disabled_server_slot_removal_with_enabled_grants_fails() -> None:
    manager = _manager(
        InMemoryExternalRegistryStore(
            [_server(credential_slots=["api_key", "oauth_token"], enabled=False)]
        ),
        credential_grant_store=InMemoryCredentialGrantStore([_grant()]),
    )

    with pytest.raises(GatewayExternalRegistryManagementError) as exc_info:
        await manager.patch_server("search", {"credential_slots": ["api_key"]})

    assert exc_info.value.reason_code == "external_server_has_credential_grants"


@pytest.mark.asyncio
async def test_gateway_external_registry_slot_removal_without_grant_store_fails_closed() -> None:
    manager = _manager(
        InMemoryExternalRegistryStore(
            [_server(credential_slots=["api_key", "oauth_token"], enabled=False)]
        )
    )

    with pytest.raises(GatewayExternalRegistryManagementError) as exc_info:
        await manager.patch_server("search", {"credential_slots": ["api_key"]})

    assert exc_info.value.reason_code == "credential_grant_store_unavailable"


@pytest.mark.asyncio
async def test_gateway_external_registry_delete_missing_server_raises_not_found() -> None:
    manager = _manager(InMemoryExternalRegistryStore())

    with pytest.raises(GatewayExternalRegistryManagementError) as exc_info:
        await manager.delete_server("missing")

    assert exc_info.value.reason_code == "external_server_not_found"


@pytest.mark.asyncio
async def test_gateway_external_registry_delete_stale_false_result_raises_not_found() -> None:
    manager = _manager(
        StaleDeleteExternalRegistryStore([_server()]),
        credential_grant_store=InMemoryCredentialGrantStore(),
    )

    with pytest.raises(GatewayExternalRegistryManagementError) as exc_info:
        await manager.delete_server("search")

    assert exc_info.value.reason_code == "external_server_not_found"
    assert exc_info.value.to_payload()["server_id"] == "search"


@pytest.mark.asyncio
async def test_gateway_external_registry_delete_with_enabled_credential_grants_fails() -> None:
    manager = _manager(
        InMemoryExternalRegistryStore([_server()]),
        credential_grant_store=InMemoryCredentialGrantStore([_grant()]),
    )

    with pytest.raises(GatewayExternalRegistryManagementError) as exc_info:
        await manager.delete_server("search")

    assert exc_info.value.reason_code == "external_server_has_credential_grants"


@pytest.mark.asyncio
async def test_gateway_external_registry_delete_without_grant_store_fails_closed() -> None:
    manager = _manager(InMemoryExternalRegistryStore([_server()]))

    with pytest.raises(GatewayExternalRegistryManagementError) as exc_info:
        await manager.delete_server("search")

    assert exc_info.value.reason_code == "credential_grant_store_unavailable"


@pytest.mark.asyncio
async def test_gateway_external_registry_delete_ungranted_server_succeeds_and_audits() -> None:
    audit_store = InMemoryAuditStore()
    store = InMemoryExternalRegistryStore([_server()])
    manager = _manager(
        store,
        credential_grant_store=InMemoryCredentialGrantStore(
            [_grant(external_server_id="search", enabled=False)]
        ),
        audit_store=audit_store,
    )

    payload = await manager.delete_server("search")

    assert payload == {
        "ok": True,
        "server_id": "search",
        "store": {"kind": "memory", "persistent": False},
    }
    assert await store.get_server("search") is None
    assert [event.event_type for event in audit_store.events] == [
        "external_server.deleted"
    ]
    assert audit_store.events[0].payload == {"server_id": "search"}
