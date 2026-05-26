from __future__ import annotations

import ast
from pathlib import Path


MCP_ROOT = Path(__file__).resolve().parents[1]


def _interface_boundary_violations_for(path: Path, interface_dir: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative_parent = path.parent.relative_to(interface_dir)
    relative_depth = 0 if relative_parent == Path(".") else len(relative_parent.parts)
    max_interface_relative_level = relative_depth + 1
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            violations.extend(
                alias.name
                for alias in node.names
                if alias.name == "tldw_Server_API"
                or alias.name.startswith("tldw_Server_API.")
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level > max_interface_relative_level:
                violations.append(
                    f"relative import escapes interfaces package: level={node.level}"
                )
            elif (
                node.module
                and node.level == 0
                and (
                    node.module == "tldw_Server_API"
                    or node.module.startswith("tldw_Server_API.")
                )
            ):
                violations.append(node.module)
    return violations


def test_new_interface_modules_do_not_import_tldw_server_api() -> None:
    interface_dir = MCP_ROOT / "interfaces"
    assert interface_dir.exists()
    offenders: dict[str, list[str]] = {}
    for path in interface_dir.rglob("*.py"):
        violations = _interface_boundary_violations_for(path, interface_dir)
        if violations:
            offenders[str(path)] = violations
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
    assert server.protocol.rbac_policy is deps.rbac_policy
    if hasattr(server, "module_registry"):
        assert server.module_registry is deps.module_registry
    if hasattr(server, "rbac_policy"):
        assert server.rbac_policy is deps.rbac_policy


def test_protocol_instances_do_not_share_prepared_call_secrets() -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol

    first = MCPProtocol()
    second = MCPProtocol()
    assert first is not second
    assert first._prepared_call_secret != second._prepared_call_secret  # noqa: SLF001
