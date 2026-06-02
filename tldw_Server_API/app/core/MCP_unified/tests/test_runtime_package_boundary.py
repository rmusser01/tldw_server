from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import mcp_unified
import pytest
from pydantic import ValidationError

PACKAGE_ROOT = Path(mcp_unified.__file__).resolve().parent


def _tldw_imports_for(path: Path) -> list[str]:
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


def test_runtime_package_boundary_has_no_tldw_server_imports() -> None:
    assert PACKAGE_ROOT.exists()
    offenders: dict[str, list[str]] = {}
    for path in PACKAGE_ROOT.rglob("*.py"):
        imports = _tldw_imports_for(path)
        if imports:
            offenders[str(path)] = imports
    assert offenders == {}


def test_host_interface_shims_reexport_package_contracts() -> None:
    package_policy = importlib.import_module("mcp_unified.interfaces.policy")
    package_runtime = importlib.import_module("mcp_unified.interfaces.runtime")
    package_storage = importlib.import_module("mcp_unified.interfaces.storage")
    host_policy = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.interfaces.policy"
    )
    host_runtime = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.interfaces.runtime"
    )
    host_storage = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.interfaces.storage"
    )
    assert host_policy.ApprovalEvaluator is package_policy.ApprovalEvaluator
    assert host_runtime.MCPRuntimeDependencies is package_runtime.MCPRuntimeDependencies
    assert host_runtime.ModuleRegistry is package_runtime.ModuleRegistry
    assert host_storage.ProfileStore is package_storage.ProfileStore


def test_host_external_config_schema_shim_reexports_package_contracts() -> None:
    package_schema = importlib.import_module("mcp_unified.federation.config_schema")
    host_schema = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.external_servers.config_schema"
    )

    assert host_schema.ExternalAuthMode is package_schema.ExternalAuthMode
    assert host_schema.ExternalMCPServerConfig is package_schema.ExternalMCPServerConfig
    assert host_schema.ExternalServerRegistryConfig is package_schema.ExternalServerRegistryConfig
    assert host_schema.ExternalTransportType is package_schema.ExternalTransportType
    assert host_schema.parse_external_server_registry is package_schema.parse_external_server_registry


def test_host_external_transport_base_reexports_package_contracts() -> None:
    """Host transport base must reuse the package-owned external contracts."""
    package_models = importlib.import_module("mcp_unified.federation.models")
    package_federation = importlib.import_module("mcp_unified.federation")
    host_base = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.external_servers.transports.base"
    )

    assert host_base.ExternalToolDefinition is package_models.ExternalToolDefinition
    assert host_base.ExternalToolCallResult is package_models.ExternalToolCallResult
    assert host_base.BrokeredExternalCredential is package_models.BrokeredExternalCredential
    assert package_federation.BrokeredExternalCredential is package_models.BrokeredExternalCredential


def test_brokered_external_credential_copy_returns_caller_owned_data() -> None:
    """Brokered credential copies must not expose mutable source state."""
    package_models = importlib.import_module("mcp_unified.federation.models")
    credential = package_models.BrokeredExternalCredential(
        headers={"Authorization": "Bearer token"},
        env={"TOKEN": "secret"},
        metadata={"nested": {"source": "broker"}},
    )

    copied = credential.copy()
    copied.headers["Authorization"] = "changed"
    copied.env["TOKEN"] = "changed"
    copied.metadata["nested"]["source"] = "changed"

    assert credential.headers == {"Authorization": "Bearer token"}
    assert credential.env == {"TOKEN": "secret"}
    assert credential.metadata == {"nested": {"source": "broker"}}


def test_host_external_manager_reexports_package_virtual_tool_contract() -> None:
    """Host external manager must reuse the package-owned virtual tool contract."""
    package_models = importlib.import_module("mcp_unified.federation.models")
    package_federation = importlib.import_module("mcp_unified.federation")
    host_external = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.external_servers"
    )
    host_manager = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.external_servers.manager"
    )

    assert host_manager.VirtualExternalTool is package_models.VirtualExternalTool
    assert host_external.VirtualExternalTool is package_models.VirtualExternalTool
    assert package_federation.VirtualExternalTool is package_models.VirtualExternalTool


def test_gateway_external_runtime_exports_are_package_owned() -> None:
    """Gateway runtime exports must resolve without host package imports."""
    package_gateway = importlib.import_module("mcp_unified.gateway")
    package_config = importlib.import_module("mcp_unified.gateway.config")
    package_adapter = importlib.import_module("mcp_unified.gateway.external_runtime_adapter")
    package_lifecycle = importlib.import_module("mcp_unified.gateway.lifecycle")
    package_runtime = importlib.import_module("mcp_unified.gateway.external_runtime")

    assert package_gateway.GatewayExternalRuntimeManager is package_runtime.GatewayExternalRuntimeManager
    assert package_gateway.GatewayExternalRuntimeError is package_runtime.GatewayExternalRuntimeError
    assert (
        package_gateway.ExternalRuntimeGatewayRuntime
        is package_adapter.ExternalRuntimeGatewayRuntime
    )
    assert (
        package_gateway.GatewayExternalRuntimeBootstrapConfig
        is package_config.GatewayExternalRuntimeBootstrapConfig
    )
    assert (
        package_gateway.GatewayExternalRuntimeLifecycleConfig
        is package_lifecycle.GatewayExternalRuntimeLifecycleConfig
    )


def test_gateway_external_runtime_adapter_import_does_not_import_fastapi_transport() -> None:
    """Importing the external runtime adapter must not require FastAPI helpers."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import mcp_unified.gateway.external_runtime_adapter; "
                "print('mcp_unified.gateway.fastapi' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_federation_installer_contracts_are_public_exports() -> None:
    """Installer contracts should be importable from the public federation package."""
    package_federation = importlib.import_module("mcp_unified.federation")
    package_installers = importlib.import_module("mcp_unified.federation.installers")

    assert package_federation.ExternalServerInstaller is package_installers.ExternalServerInstaller
    assert package_federation.NullExternalServerInstaller is package_installers.NullExternalServerInstaller


def test_federation_process_policy_contracts_are_public_exports() -> None:
    """Stdio process-policy contracts should be public federation exports."""
    package_federation = importlib.import_module("mcp_unified.federation")
    package_policy = importlib.import_module("mcp_unified.federation.process_policy")

    assert package_federation.StdioProcessPolicy is package_policy.StdioProcessPolicy
    assert (
        package_federation.coerce_stdio_process_policy
        is package_policy.coerce_stdio_process_policy
    )


def test_virtual_external_tool_copy_returns_caller_owned_data() -> None:
    """Virtual tool copies must not expose mutable source state."""
    package_models = importlib.import_module("mcp_unified.federation.models")
    virtual_tool = package_models.VirtualExternalTool(
        virtual_name="ext.docs.search",
        server_id="docs",
        upstream_tool_name="search",
        description="Search docs",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        metadata={"nested": {"source": "discovery"}},
        is_write=False,
    )

    copied = virtual_tool.copy()
    copied.input_schema["properties"]["q"]["type"] = "number"
    copied.metadata["nested"]["source"] = "changed"

    assert virtual_tool.input_schema == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
    }
    assert virtual_tool.metadata == {"nested": {"source": "discovery"}}


def test_profile_defaults_are_safe_and_preserve_extension_metadata() -> None:
    from mcp_unified.profiles.models import MCPProfile

    profile = MCPProfile(
        id="architect",
        name="Architect",
        approval_policy=None,
        path_scopes=None,
        external_server_grants=None,
        credential_grants=None,
        policy_document={
            "allowed_tools": None,
            "capabilities": None,
            "resource_constraints": None,
            "policy_extension": {"level": "experimental"},
        },
        metadata={"agent_metadata": {"system_prompt": "review architecture"}},
        profile_extension={"owner": "frontend"},
    )

    assert profile.enabled is True
    assert profile.policy_document.allowed_tools == []
    assert profile.policy_document.capabilities == []
    assert profile.policy_document.resource_constraints == {}
    assert profile.credential_grants == []
    assert profile.external_server_grants == []
    assert profile.metadata["agent_metadata"]["system_prompt"] == "review architecture"
    dumped = profile.model_dump()
    assert dumped["profile_extension"] == {"owner": "frontend"}
    assert dumped["policy_document"]["policy_extension"] == {"level": "experimental"}


def test_profile_rejects_naive_timestamps() -> None:
    from mcp_unified.profiles.models import MCPProfile

    with pytest.raises(ValidationError):
        MCPProfile(
            id="architect",
            name="Architect",
            created_at=datetime(2026, 5, 27, 5, 0, 0),
        )
