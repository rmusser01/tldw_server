"""Tests for standalone MCP storage contract primitives."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any, cast

import pytest


def _tldw_imports_for(path: Path) -> list[str]:
    """Return imports from a Python file that cross into the host package."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
                if alias.name == "tldw_Server_API"
                or alias.name.startswith("tldw_Server_API.")
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "tldw_Server_API" or node.module.startswith("tldw_Server_API."):
                imports.append(node.module)
    return imports


def test_storage_package_has_no_tldw_server_imports() -> None:
    storage = importlib.import_module("mcp_unified.storage")
    assert storage.__file__ is not None
    storage_root = Path(storage.__file__).resolve().parent
    offenders: dict[str, list[str]] = {}
    for path in storage_root.rglob("*.py"):
        imports = _tldw_imports_for(path)
        if imports:
            offenders[str(path)] = imports
    assert offenders == {}


def test_profile_assignment_models_gateway_default_and_workspace_binding() -> None:
    from mcp_unified.storage import ProfileAssignment

    assignment = ProfileAssignment(
        id="assignment-1",
        profile_id="backend-engineer",
        principal_id="user-1",
        workspace_id="workspace-1",
        binding={"workspace_id": "workspace-1", "path_scopes": [{"root": "/repo"}]},
    )

    assert assignment.enabled is True
    assert assignment.profile_id == "backend-engineer"
    assert assignment.principal_id == "user-1"
    assert assignment.workspace_id == "workspace-1"
    assert assignment.is_default is False
    assert assignment.binding["workspace_id"] == "workspace-1"
    assert assignment.created_at.tzinfo is not None
    assert assignment.updated_at.tzinfo is not None


def test_profile_assignment_requires_assignment_target() -> None:
    from mcp_unified.storage import ProfileAssignment

    with pytest.raises(ValueError, match="principal_id, workspace_id, or is_default"):
        ProfileAssignment(id="assignment-1", profile_id="backend-engineer")

    default_assignment = ProfileAssignment(
        id="assignment-2",
        profile_id="orchestrator",
        is_default=True,
    )

    assert default_assignment.is_default is True
    assert default_assignment.principal_id is None
    assert default_assignment.workspace_id is None


def test_credential_grant_contract_excludes_secret_material_fields() -> None:
    from mcp_unified.storage import CredentialGrant

    grant = CredentialGrant(
        id="grant-1",
        profile_id="deep-researcher",
        broker_id="host-broker",
        credential_slot="search_api",
        scopes=["web.search"],
    )

    dumped = grant.model_dump()
    forbidden = {"secret", "token", "api_key", "value", "credential_value"}
    assert forbidden.isdisjoint(dumped)
    assert grant.enabled is True
    assert grant.broker_id == "host-broker"
    assert grant.credential_slot == "search_api"
    assert grant.scopes == ["web.search"]


def test_external_server_definition_defaults_do_not_start_lifecycle() -> None:
    from mcp_unified.storage import ExternalServerDefinition

    server = ExternalServerDefinition(
        id="filesystem",
        name="Filesystem",
        transport="stdio",
        command=["/usr/local/bin/mcp-filesystem"],
        cwd="/workspace",
    )

    assert server.enabled is True
    assert server.auto_start is False
    assert server.transport == "stdio"
    assert server.command == ["/usr/local/bin/mcp-filesystem"]
    assert server.env_allowlist == []
    assert server.credential_slots == []


def test_external_server_definition_rejects_unsupported_transport() -> None:
    from mcp_unified.storage import ExternalServerDefinition

    with pytest.raises(ValueError):
        ExternalServerDefinition(
            id="unsupported",
            name="Unsupported",
            transport=cast(Any, "http"),
            url="https://example.com/mcp",
        )


def test_external_server_definition_validates_enabled_transport_fields() -> None:
    from mcp_unified.storage import ExternalServerDefinition

    with pytest.raises(ValueError, match="stdio.*command"):
        ExternalServerDefinition(
            id="stdio-empty",
            name="Empty stdio",
            transport="stdio",
            command=["   "],
        )

    with pytest.raises(ValueError, match="websocket.*url"):
        ExternalServerDefinition(
            id="websocket-empty",
            name="Empty websocket",
            transport="websocket",
            url="   ",
        )

    disabled_draft = ExternalServerDefinition(
        id="draft",
        name="Draft",
        transport="stdio",
        enabled=False,
    )

    assert disabled_draft.enabled is False
    assert disabled_draft.command == []


def test_external_server_definition_normalizes_transport_fields() -> None:
    from mcp_unified.storage import ExternalServerDefinition

    stdio_server = ExternalServerDefinition(
        id="stdio",
        name="Stdio",
        transport="stdio",
        command=["  /usr/local/bin/mcp  ", "  --verbose  "],
    )
    websocket_server = ExternalServerDefinition(
        id="websocket",
        name="Websocket",
        transport="websocket",
        url="  wss://example.com/mcp  ",
    )

    assert stdio_server.command == ["/usr/local/bin/mcp", "--verbose"]
    assert websocket_server.url == "wss://example.com/mcp"


def test_audit_event_uses_aware_timestamp_and_caller_owned_payload() -> None:
    from mcp_unified.storage import AuditEvent

    payload: dict[str, Any] = {
        "tool": "filesystem.write",
        "args": {"path": "/repo/README.md"},
    }
    event = AuditEvent(
        id="event-1",
        event_type="tool.denied",
        actor_id="user-1",
        payload=payload,
    )

    payload["args"]["path"] = "/repo/CHANGED.md"

    assert event.created_at.tzinfo is not None
    assert event.payload["args"]["path"] == "/repo/README.md"
    assert event.event_type == "tool.denied"


def test_storage_interfaces_export_split_store_contracts() -> None:
    storage = importlib.import_module("mcp_unified.interfaces.storage")
    interfaces = importlib.import_module("mcp_unified.interfaces")

    for name in (
        "ProfileStore",
        "ProfileAssignmentStore",
        "ApprovalPolicyStore",
        "CredentialGrantStore",
        "ExternalRegistryStore",
        "AuditStore",
    ):
        assert hasattr(storage, name)
        assert getattr(interfaces, name) is getattr(storage, name)


def test_external_registry_store_list_servers_preserves_runtime_manager_shape() -> None:
    from mcp_unified.interfaces.storage import ExternalRegistryStore

    from tldw_Server_API.app.core.MCP_unified.external_servers.manager import ExternalServerManager

    assert list(inspect.signature(ExternalRegistryStore.list_servers).parameters) == ["self"]
    assert list(inspect.signature(ExternalServerManager.list_servers).parameters) == ["self"]


def test_external_registry_store_exposes_typed_definition_listing() -> None:
    from mcp_unified.interfaces.storage import ExternalRegistryStore

    params = inspect.signature(ExternalRegistryStore.list_server_definitions).parameters

    assert list(params) == ["self", "enabled"]
    assert params["enabled"].kind is inspect.Parameter.KEYWORD_ONLY


def test_external_registry_store_accepts_legacy_dict_listing_shape() -> None:
    from mcp_unified.interfaces.storage import ExternalRegistryStore
    from mcp_unified.storage import ExternalServerDefinition

    class LegacyRegistryStore:
        async def get_server(self, server_id: str) -> ExternalServerDefinition | None:
            return None

        async def list_servers(self) -> list[dict[str, Any]]:
            return []

        async def list_server_definitions(
            self,
            *,
            enabled: bool | None = None,
        ) -> list[ExternalServerDefinition]:
            return []

        async def upsert_server(
            self,
            server: ExternalServerDefinition,
        ) -> ExternalServerDefinition:
            return server

        async def delete_server(self, server_id: str) -> bool:
            return False

    store: ExternalRegistryStore = LegacyRegistryStore()

    assert store is not None
