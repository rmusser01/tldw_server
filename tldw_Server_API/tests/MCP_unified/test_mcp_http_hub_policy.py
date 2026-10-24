"""HTTP ingress must apply Hub policy before tool discovery or execution."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint as endpoint
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext
from tldw_Server_API.app.core.MCP_unified.server import MCPServer
from tldw_Server_API.app.core.MCP_unified.tests.test_protocol_tool_hooks import _protocol
from tldw_Server_API.app.services.mcp_hub_approval_service import McpHubApprovalService


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_kind", ["bearer", "api_key"])
@pytest.mark.parametrize("transport", ["rest", "jsonrpc"])
@pytest.mark.parametrize("operation", ["call", "list"])
@pytest.mark.parametrize("policy_case", ["denied", "allowed", "disabled"])
async def test_http_ingress_respects_current_hub_policy(
    monkeypatch, tmp_path, auth_kind, transport, operation, policy_case
):
    """Use real routes/server/protocol and a harmless provider at its dependency boundary."""
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    monkeypatch.setenv("MCP_HUB_POLICY_ENFORCEMENT_ENABLED", "0" if policy_case == "disabled" else "1")
    protocol, module, _hooks = _protocol()
    protocol.module_registry.get_all_modules = AsyncMock(return_value={"stub": module})
    protocol.dependencies.effective_policy_resolver = SimpleNamespace(
        resolve_for_context=AsyncMock(
            return_value={
                "enabled": True,
                "allowed_tools": ["stub.echo"],
                "denied_tools": ["stub.echo"] if policy_case != "allowed" else [],
                "capabilities": [],
                "sources": [],
            }
        )
    )
    # No approval policy is configured, so the real decision must not access storage.
    protocol.dependencies.approval_evaluator = McpHubApprovalService(repo=SimpleNamespace())
    protocol = MCPProtocol(dependencies=protocol.dependencies)
    server = MCPServer()
    server.protocol = protocol
    server.initialized = True
    monkeypatch.setattr(endpoint, "get_mcp_server", lambda: server)
    auth = endpoint.McpAuthContext(
        user=endpoint.TokenData(sub="7", permissions=["tools.execute:stub.echo", "modules.read"]),
        principal=None,
        api_key_info={"user_id": 7, "scopes": ["*"]} if auth_kind == "api_key" else None,
        raw_api_key=None,
    )
    app = FastAPI()
    app.include_router(endpoint.router, prefix="/api/v1")
    app.dependency_overrides[endpoint.get_mcp_auth_context] = lambda: auth
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        if transport == "rest":
            response = (
                await client.post("/api/v1/mcp/tools/execute", json={"tool_name": "stub.echo", "arguments": {}})
                if operation == "call"
                else await client.get("/api/v1/mcp/tools")
            )
        else:
            response = await client.post(
                "/api/v1/mcp/request",
                json={
                    "jsonrpc": "2.0",
                    "id": "policy-check",
                    "method": "tools/call" if operation == "call" else "tools/list",
                    "params": {"name": "stub.echo", "arguments": {}} if operation == "call" else {},
                },
            )
    denied = policy_case == "denied"
    body = response.json()
    if operation == "call":
        if denied:
            outcome = response.status_code if transport == "rest" else body.get("error", {}).get("code")
            assert (outcome, module.executions) == (403 if transport == "rest" else -32001, 0)  # nosec B101 -- denied tool never reaches provider.
        else:
            assert (response.status_code, "error" in body, module.executions) == (200, False, 1)  # nosec B101 -- explicit allow and disabled compatibility remain functional.
    else:
        assert response.status_code == 200  # nosec B101 -- authenticated discovery remains available.
        tools = body.get("result", body)["tools"]
        executable = any(tool["name"] == "stub.echo" and tool.get("canExecute") is True for tool in tools)
        assert (executable, module.executions) == (not denied, 0)  # nosec B101 -- discovery respects current policy without a provider side effect.


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["run", "bash", "shell"])
@pytest.mark.parametrize(
    "policy_case,available,executes",
    [
        ("none", True, True),
        ("empty", True, True),
        ("disabled", True, True),
        ("restricted_allow", True, True),
        ("restricted_mismatch", True, False),
        ("conditional_deny", True, False),
        ("bare_deny", False, False),
        ("other_only", False, False),
        ("malformed_allow", False, False),
        ("empty_base", False, False),
        ("invalid_base", False, False),
        ("resolution_error", False, False),
        ("rbac_denied", False, False),
    ],
)
async def test_discovery_preserves_policy_name_and_argument_boundaries(tool_name, policy_case, available, executes):
    """Discovery availability never grants an argument or bypasses an existing denial."""
    protocol, module, _hooks = _protocol()
    module.get_tools = AsyncMock(return_value=[await module.get_tool_def(tool_name)])
    protocol.module_registry.get_all_modules = AsyncMock(return_value={"stub": module})
    policy = {"enabled": policy_case != "disabled", "allowed_tools": [], "denied_tools": []}
    if policy_case in {"restricted_allow", "restricted_mismatch"}:
        policy["allowed_tools"] = ["run(help)"]
    elif policy_case == "conditional_deny":
        policy.update(allowed_tools=["run"], denied_tools=["run(rm *)"])
    elif policy_case in {"bare_deny", "disabled"}:
        policy.update(allowed_tools=["run"], denied_tools=["run"])
    elif policy_case == "other_only":
        policy["allowed_tools"] = ["other.tool"]
    elif policy_case == "malformed_allow":
        policy["allowed_tools"] = ["run(help"]
    elif policy_case == "empty_base":
        policy["allowed_tools"] = ["(help)"]
    elif policy_case == "invalid_base":
        policy["allowed_tools"] = ["bad name(help)"]
    elif policy_case == "resolution_error":
        policy["resolution_error"] = "policy_resolution_failed"
    protocol.dependencies.effective_policy_resolver = SimpleNamespace(
        resolve_for_context=AsyncMock(return_value=None if policy_case == "none" else policy)
    )
    protocol.dependencies.approval_evaluator = McpHubApprovalService(repo=SimpleNamespace())
    if policy_case == "rbac_denied":
        protocol.rbac_policy.check_permission = AsyncMock(return_value=False)
    protocol = MCPProtocol(dependencies=protocol.dependencies)
    context = RequestContext(
        request_id="discovery-policy-check",
        user_id="7",
        client_id="unit",
        session_id=None,
        metadata={"mcp_policy_context_enabled": True},
    )
    listed = await protocol._handle_tools_list({}, context)
    can_execute = any(tool["name"] == tool_name and tool.get("canExecute") is True for tool in listed["tools"])
    assert can_execute is available  # nosec B101 -- current policy and RBAC govern name availability.
    command = (
        "rm sentinel"
        if policy_case == "conditional_deny"
        else "version"
        if policy_case == "restricted_mismatch"
        else "help"
    )
    arguments = {"command": command}
    if executes:
        await protocol._handle_tools_call({"name": tool_name, "arguments": arguments}, context)
    else:
        with pytest.raises(PermissionError):
            await protocol._handle_tools_call({"name": tool_name, "arguments": arguments}, context)
    assert module.executions == int(executes)  # nosec B101 -- list never executes and actual calls retain argument denials.
