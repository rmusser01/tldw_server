"""Tests for standalone MCP storage contract primitives."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any


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
