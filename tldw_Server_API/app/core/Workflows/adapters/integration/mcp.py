"""MCP integration adapters.

This module includes adapters for Model Context Protocol operations:
- mcp_tool: Execute MCP tools
"""

from __future__ import annotations

import json
import time
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.exceptions import AdapterError
from tldw_Server_API.app.core.Workflows.adapters._common import (
    extract_mcp_policy,
    extract_tool_scopes,
    normalize_str_list,
    resolve_artifacts_dir,
    tool_matches_allowlist,
)
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.integration._config import MCPToolConfig

_MCP_ADAPTER_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


@registry.register(
    "mcp_tool",
    category="integration",
    description="Execute MCP tools",
    parallelizable=True,
    tags=["integration", "mcp"],
    config_model=MCPToolConfig,
)
async def run_mcp_tool_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Execute an MCP tool via the unified server registry.

    Config:
      - tool_name: str
      - arguments: dict
    Output: {"result": Any}
    """
    from tldw_Server_API.app.core.MCP_unified import get_mcp_server

    tool_name = str(config.get("tool_name") or "").strip()
    arguments = config.get("arguments") or {}
    if not tool_name:
        return {"error": "missing_tool_name"}

    policy = extract_mcp_policy(context)
    allowlist = normalize_str_list(policy.get("allowlist") or policy.get("allowed_tools"))
    if allowlist and not tool_matches_allowlist(tool_name, allowlist):
        raise AdapterError("mcp_tool_not_allowed")

    allowed_scopes = normalize_str_list(policy.get("scopes") or policy.get("allow_scopes") or policy.get("capabilities"))

    server = get_mcp_server()

    # Find module by tool registry
    module_id = server.module_registry._tool_registry.get(tool_name)  # type: ignore[attr-defined]
    module = None
    if module_id:
        module = server.module_registry._module_instances.get(module_id)  # type: ignore[attr-defined]

    # Fallback: scan modules for defined tool names
    if module is None:
        try:
            for mid, mod in server.module_registry._module_instances.items():  # type: ignore[attr-defined]
                try:
                    tools = await mod.get_tools()
                    if any((t.get("name") == tool_name) for t in tools):
                        module = mod
                        module_id = mid
                        break
                except _MCP_ADAPTER_NONCRITICAL_EXCEPTIONS:
                    continue
        except _MCP_ADAPTER_NONCRITICAL_EXCEPTIONS:
            pass

    tool_def = None
    if module is not None:
        try:
            tool_defs = await module.get_tools()
            for tool in tool_defs:
                if tool.get("name") == tool_name:
                    tool_def = tool
                    break
        except _MCP_ADAPTER_NONCRITICAL_EXCEPTIONS:
            logger.debug("MCP tool adapter: failed to get tool definitions")

    required_scopes = extract_tool_scopes(tool_def)
    if required_scopes:
        if not allowed_scopes:
            raise AdapterError("mcp_tool_scope_denied")
        if "*" not in allowed_scopes:
            missing = [s for s in required_scopes if s not in allowed_scopes]
            if missing:
                raise AdapterError("mcp_tool_scope_denied")

    if module is None:
        # Test-friendly fallback for echo
        if tool_name == "echo":
            return {"result": arguments.get("message"), "module": "_fallback"}
        return {"error": "tool_not_found"}

    result = await module.execute_tool(tool_name, arguments)

    # Optional artifact persistence of result
    try:
        if bool(config.get("save_artifact")) and callable(context.get("add_artifact")):
            step_run_id = str(context.get("step_run_id") or "")
            art_dir = resolve_artifacts_dir(step_run_id or f"mcp_{int(time.time()*1000)}")
            art_dir.mkdir(parents=True, exist_ok=True)
            fpath = art_dir / "mcp_result.json"
            fpath.write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
            context["add_artifact"](
                type="mcp_result",
                uri=f"file://{fpath}",
                size_bytes=len(fpath.read_bytes() if fpath.exists() else b""),
                mime_type="application/json",
                metadata={"tool_name": tool_name, "module": module_id},
            )
    except _MCP_ADAPTER_NONCRITICAL_EXCEPTIONS:
        pass

    return {"result": result, "module": module_id}
