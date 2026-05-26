from __future__ import annotations

import ast
from pathlib import Path


MCP_ROOT = Path("tldw_Server_API/app/core/MCP_unified")


def _imports_for(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def test_new_interface_modules_do_not_import_tldw_server_api() -> None:
    interface_dir = MCP_ROOT / "interfaces"
    assert interface_dir.exists()
    offenders: dict[str, list[str]] = {}
    for path in interface_dir.glob("*.py"):
        imports = [
            name
            for name in _imports_for(path)
            if name == "tldw_Server_API" or name.startswith("tldw_Server_API.")
        ]
        if imports:
            offenders[str(path)] = imports
    assert offenders == {}


def test_mcp_protocol_accepts_runtime_dependencies() -> None:
    from tldw_Server_API.app.core.MCP_unified.adapters.tldw_runtime import (
        build_default_runtime_dependencies,
    )
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol

    deps = build_default_runtime_dependencies()
    protocol = MCPProtocol(dependencies=deps)
    assert protocol.module_registry is deps.module_registry
    assert protocol.rbac_policy is deps.rbac_policy


def test_mcp_server_accepts_runtime_dependencies() -> None:
    from tldw_Server_API.app.core.MCP_unified.adapters.tldw_runtime import (
        build_default_runtime_dependencies,
    )
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    deps = build_default_runtime_dependencies()
    server = MCPServer(dependencies=deps)
    assert server.protocol.module_registry is deps.module_registry


def test_protocol_instances_do_not_share_prepared_call_secrets() -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol

    first = MCPProtocol()
    second = MCPProtocol()
    assert first is not second
    assert first._prepared_call_secret != second._prepared_call_secret  # noqa: SLF001
