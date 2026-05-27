from __future__ import annotations

import ast
import importlib
from pathlib import Path


PACKAGE_ROOT = Path("mcp_unified")


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


def test_profile_defaults_are_safe_and_preserve_extension_metadata() -> None:
    from mcp_unified.profiles.models import MCPProfile

    profile = MCPProfile(
        id="architect",
        name="Architect",
        metadata={"agent_metadata": {"system_prompt": "review architecture"}},
    )

    assert profile.enabled is True
    assert profile.policy_document.allowed_tools == []
    assert profile.policy_document.capabilities == []
    assert profile.credential_grants == []
    assert profile.external_server_grants == []
    assert profile.metadata["agent_metadata"]["system_prompt"] == "review architecture"
