import os

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.run_command_module import (
    RunCommandModule,
)
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext


class _RunAliasRegistryStub:
    def __init__(self, module: RunCommandModule) -> None:
        self.module = module

    async def find_module_for_tool(self, tool_name: str) -> RunCommandModule | None:
        return self.module if tool_name in {"run", "bash", "shell"} else None

    def get_module_id_for_tool(self, tool_name: str) -> str | None:
        return "run_command" if tool_name in {"run", "bash", "shell"} else None

    async def get_all_modules(self) -> dict[str, RunCommandModule]:
        return {"run_command": self.module}


def _build_run_alias_protocol(*, rbac_tool_names: set[str] | None = None) -> MCPProtocol:
    proto = MCPProtocol()
    module = RunCommandModule(ModuleConfig(name="run", settings={"protocol": proto}))
    proto.module_registry = _RunAliasRegistryStub(module)
    allowed_tool_names = rbac_tool_names or {"run"}

    async def _rbac_check(user_id, resource, action, identifier):  # noqa: ANN001
        del user_id, action
        if resource.value == "module":
            return True
        if resource.value == "tool":
            return str(identifier) in allowed_tool_names
        return True

    proto._rbac_check = _rbac_check  # type: ignore[method-assign]
    return proto


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_allows_run_alias_when_context_allows_canonical_run() -> None:
    proto = _build_run_alias_protocol(rbac_tool_names={"run"})
    context = RequestContext(
        request_id="run-alias-context-allow",
        user_id="1",
        client_id="unit",
        session_id=None,
        metadata={"allowed_tools": ["run"]},
    )

    result = await proto._handle_tools_call(
        {"name": "shell", "arguments": {"command": "echo unsafe"}},
        context,
    )

    assert result["tool"] == "shell"
    assert "Unknown command: echo" in str(result["content"])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_maps_run_aliases_for_effective_policy_and_listing() -> None:
    proto = _build_run_alias_protocol(rbac_tool_names={"run"})

    async def _resolve_policy(_context):  # noqa: ANN001
        return {
            "enabled": True,
            "allowed_tools": ["run(help)"],
            "denied_tools": [],
            "capabilities": [],
            "sources": [],
        }

    proto._resolve_effective_tool_policy = _resolve_policy  # type: ignore[method-assign]
    context = RequestContext(
        request_id="run-alias-policy-allow",
        user_id="1",
        client_id="unit",
        session_id=None,
        metadata={"mcp_policy_context_enabled": True},
    )

    result = await proto._handle_tools_call(
        {"name": "bash", "arguments": {"command": "help"}},
        context,
    )
    tools = await proto._handle_tools_list({}, context)
    by_name = {tool["name"]: tool for tool in tools["tools"]}

    assert result["tool"] == "bash"
    assert "Virtual CLI commands available" in str(result["content"])
    assert by_name["run"]["canExecute"] is True
    assert by_name["bash"]["canExecute"] is True
    assert by_name["shell"]["canExecute"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_tools_call_blocks_disallowed_tools(monkeypatch):
    os.environ["TEST_MODE"] = "true"

    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext

    class _ModuleStub:
        name = "Shell"

        async def get_tools(self):
            return [{"name": "Bash", "description": "", "inputSchema": {"type": "object"}}]

        async def execute_tool(self, tool_name, arguments, context=None):
            return {"ok": True}

        async def execute_with_circuit_breaker(self, func, *args, **kwargs):
            return await func(*args, **kwargs)

        def sanitize_input(self, args):
            return args

        def validate_tool_arguments(self, tool_name, tool_args):
            return None

        def is_write_tool_def(self, tool_def):
            return False

    class _RegistryStub:
        async def find_module_for_tool(self, tool_name):
            return _ModuleStub() if tool_name == "Bash" else None

        def get_module_id_for_tool(self, tool_name):
            return "shell"

    proto = MCPProtocol()
    proto.module_registry = _RegistryStub()

    async def _allow_mod(ctx, mid):
        return True

    async def _allow_tool(ctx, name, **_kwargs):
        return True

    proto._has_module_permission = _allow_mod  # type: ignore
    proto._has_tool_permission = _allow_tool  # type: ignore

    ctx = RequestContext(
        request_id="allowed-tools-deny",
        user_id="1",
        client_id="unit",
        session_id=None,
        metadata={"allowed_tools": ["notes.search"]},
    )

    with pytest.raises(PermissionError):
        await proto._handle_tools_call({"name": "Bash", "arguments": {"command": "git status"}}, ctx)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_tools_call_allows_command_pattern(monkeypatch):
    os.environ["TEST_MODE"] = "true"

    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext

    class _ModuleStub:
        name = "Shell"

        async def get_tools(self):
            return [{"name": "Bash", "description": "", "inputSchema": {"type": "object"}}]

        async def execute_tool(self, tool_name, arguments, context=None):
            return {"ok": True}

        async def execute_with_circuit_breaker(self, func, *args, **kwargs):
            return await func(*args, **kwargs)

        def sanitize_input(self, args):
            return args

        def validate_tool_arguments(self, tool_name, tool_args):
            return None

        def is_write_tool_def(self, tool_def):
            return False

    class _RegistryStub:
        async def find_module_for_tool(self, tool_name):
            return _ModuleStub() if tool_name == "Bash" else None

        def get_module_id_for_tool(self, tool_name):
            return "shell"

    proto = MCPProtocol()
    proto.module_registry = _RegistryStub()

    async def _allow_mod(ctx, mid):
        return True

    async def _allow_tool(ctx, name, **_kwargs):
        return True

    proto._has_module_permission = _allow_mod  # type: ignore
    proto._has_tool_permission = _allow_tool  # type: ignore

    ctx = RequestContext(
        request_id="allowed-tools-allow",
        user_id="1",
        client_id="unit",
        session_id=None,
        metadata={"allowed_tools": ["Bash(git *)"]},
    )

    result = await proto._handle_tools_call({"name": "Bash", "arguments": {"command": "git status"}}, ctx)
    assert isinstance(result, dict)
    assert result.get("tool") == "Bash"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_tools_call_blocks_command_pattern_mismatch(monkeypatch):
    os.environ["TEST_MODE"] = "true"

    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext

    class _ModuleStub:
        name = "Shell"

        async def get_tools(self):
            return [{"name": "Bash", "description": "", "inputSchema": {"type": "object"}}]

        async def execute_tool(self, tool_name, arguments, context=None):
            return {"ok": True}

        async def execute_with_circuit_breaker(self, func, *args, **kwargs):
            return await func(*args, **kwargs)

        def sanitize_input(self, args):
            return args

        def validate_tool_arguments(self, tool_name, tool_args):
            return None

        def is_write_tool_def(self, tool_def):
            return False

    class _RegistryStub:
        async def find_module_for_tool(self, tool_name):
            return _ModuleStub() if tool_name == "Bash" else None

        def get_module_id_for_tool(self, tool_name):
            return "shell"

    proto = MCPProtocol()
    proto.module_registry = _RegistryStub()

    async def _allow_mod(ctx, mid):
        return True

    async def _allow_tool(ctx, name, **_kwargs):
        return True

    proto._has_module_permission = _allow_mod  # type: ignore
    proto._has_tool_permission = _allow_tool  # type: ignore

    ctx = RequestContext(
        request_id="allowed-tools-deny-pattern",
        user_id="1",
        client_id="unit",
        session_id=None,
        metadata={"allowed_tools": ["Bash(git *)"]},
    )

    with pytest.raises(PermissionError):
        await proto._handle_tools_call({"name": "Bash", "arguments": {"command": "npm test"}}, ctx)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_tools_call_blocks_tool_denied_by_effective_policy(monkeypatch):
    os.environ["TEST_MODE"] = "true"

    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext

    class _ModuleStub:
        name = "Shell"

        async def get_tools(self):
            return [{"name": "Bash", "description": "", "inputSchema": {"type": "object"}}]

        async def execute_tool(self, tool_name, arguments, context=None):
            return {"ok": True}

        async def execute_with_circuit_breaker(self, func, *args, **kwargs):
            return await func(*args, **kwargs)

        def sanitize_input(self, args):
            return args

        def validate_tool_arguments(self, tool_name, tool_args):
            return None

        def is_write_tool_def(self, tool_def):
            return False

    class _RegistryStub:
        async def find_module_for_tool(self, tool_name):
            return _ModuleStub() if tool_name == "Bash" else None

        def get_module_id_for_tool(self, tool_name):
            return "shell"

    proto = MCPProtocol()
    proto.module_registry = _RegistryStub()

    async def _allow_mod(ctx, mid):
        return True

    async def _allow_tool(ctx, name, **_kwargs):
        return True

    async def _resolve_policy(_ctx):
        return {
            "enabled": True,
            "allowed_tools": ["notes.search"],
            "denied_tools": [],
            "capabilities": [],
            "sources": [],
        }

    proto._has_module_permission = _allow_mod  # type: ignore
    proto._has_tool_permission = _allow_tool  # type: ignore
    proto._resolve_effective_tool_policy = _resolve_policy  # type: ignore[attr-defined]

    ctx = RequestContext(
        request_id="mcp-hub-policy-deny",
        user_id="1",
        client_id="unit",
        session_id=None,
        metadata={"mcp_policy_context_enabled": True},
    )

    with pytest.raises(PermissionError, match="MCP Hub policy"):
        await proto._handle_tools_call({"name": "Bash", "arguments": {"command": "git status"}}, ctx)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_tools_call_allows_tool_matching_effective_policy_pattern(monkeypatch):
    os.environ["TEST_MODE"] = "true"

    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext

    class _ModuleStub:
        name = "Shell"

        async def get_tools(self):
            return [{"name": "Bash", "description": "", "inputSchema": {"type": "object"}}]

        async def execute_tool(self, tool_name, arguments, context=None):
            return {"ok": True}

        async def execute_with_circuit_breaker(self, func, *args, **kwargs):
            return await func(*args, **kwargs)

        def sanitize_input(self, args):
            return args

        def validate_tool_arguments(self, tool_name, tool_args):
            return None

        def is_write_tool_def(self, tool_def):
            return False

    class _RegistryStub:
        async def find_module_for_tool(self, tool_name):
            return _ModuleStub() if tool_name == "Bash" else None

        def get_module_id_for_tool(self, tool_name):
            return "shell"

    proto = MCPProtocol()
    proto.module_registry = _RegistryStub()

    async def _allow_mod(ctx, mid):
        return True

    async def _allow_tool(ctx, name, **_kwargs):
        return True

    async def _resolve_policy(_ctx):
        return {
            "enabled": True,
            "allowed_tools": ["Bash(git *)"],
            "denied_tools": [],
            "capabilities": [],
            "sources": [],
        }

    proto._has_module_permission = _allow_mod  # type: ignore
    proto._has_tool_permission = _allow_tool  # type: ignore
    proto._resolve_effective_tool_policy = _resolve_policy  # type: ignore[attr-defined]

    ctx = RequestContext(
        request_id="mcp-hub-policy-allow",
        user_id="1",
        client_id="unit",
        session_id=None,
        metadata={"mcp_policy_context_enabled": True},
    )

    result = await proto._handle_tools_call({"name": "Bash", "arguments": {"command": "git status"}}, ctx)
    assert isinstance(result, dict)
    assert result.get("tool") == "Bash"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_tools_call_blocks_when_policy_resolution_fails(monkeypatch):
    os.environ["TEST_MODE"] = "true"

    from tldw_Server_API.app.core.MCP_unified.adapters.tldw_runtime import (
        build_default_runtime_dependencies,
    )
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext

    class _ModuleStub:
        name = "Shell"

        async def get_tools(self):
            return [{"name": "Bash", "description": "", "inputSchema": {"type": "object"}}]

        async def execute_tool(self, tool_name, arguments, context=None):
            return {"ok": True}

        async def execute_with_circuit_breaker(self, func, *args, **kwargs):
            return await func(*args, **kwargs)

        def sanitize_input(self, args):
            return args

        def validate_tool_arguments(self, tool_name, tool_args):
            return None

        def is_write_tool_def(self, tool_def):
            return False

    class _RegistryStub:
        async def find_module_for_tool(self, tool_name):
            return _ModuleStub() if tool_name == "Bash" else None

        def get_module_id_for_tool(self, tool_name):
            return "shell"

    async def _allow_mod(ctx, mid):
        return True

    async def _allow_tool(ctx, name, **_kwargs):
        return True

    class _FailingPolicyResolver:
        async def resolve_for_context(self, *, user_id, metadata):
            raise RuntimeError("resolver unavailable")

    deps = build_default_runtime_dependencies()
    deps.module_registry = _RegistryStub()
    deps.effective_policy_resolver = _FailingPolicyResolver()
    proto = MCPProtocol(dependencies=deps)
    proto._has_module_permission = _allow_mod  # type: ignore
    proto._has_tool_permission = _allow_tool  # type: ignore

    ctx = RequestContext(
        request_id="mcp-hub-policy-resolution-failure",
        user_id="1",
        client_id="unit",
        session_id=None,
        metadata={"mcp_policy_context_enabled": True},
    )

    with pytest.raises(PermissionError, match="MCP Hub policy"):
        await proto._handle_tools_call({"name": "Bash", "arguments": {"command": "git status"}}, ctx)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_process_request_returns_approval_required_payload(monkeypatch):
    os.environ["TEST_MODE"] = "true"

    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, MCPRequest, RequestContext

    class _AllowAllRBAC:
        async def check_permission(self, *_args, **_kwargs):
            return True

    class _ModuleStub:
        name = "Shell"

        async def get_tools(self):
            return [{"name": "Bash", "description": "", "inputSchema": {"type": "object"}}]

        async def execute_tool(self, tool_name, arguments, context=None):
            return {"ok": True}

        async def execute_with_circuit_breaker(self, func, *args, **kwargs):
            return await func(*args, **kwargs)

        def sanitize_input(self, args):
            return args

        def validate_tool_arguments(self, tool_name, tool_args):
            return None

        def is_write_tool_def(self, tool_def):
            return False

    class _RegistryStub:
        async def find_module_for_tool(self, tool_name):
            return _ModuleStub() if tool_name == "Bash" else None

        def get_module_id_for_tool(self, tool_name):
            return "shell"

    proto = MCPProtocol()
    proto.module_registry = _RegistryStub()
    proto.rbac_policy = _AllowAllRBAC()

    async def _allow_mod(ctx, mid):
        return True

    async def _allow_tool(ctx, name, **_kwargs):
        return True

    async def _resolve_policy(_ctx):
        return {
            "enabled": True,
            "allowed_tools": ["notes.search"],
            "denied_tools": [],
            "capabilities": [],
            "approval_policy_id": 17,
            "approval_mode": "ask_outside_profile",
            "sources": [],
        }

    async def _require_approval(*_args, **_kwargs):
        return {
            "status": "approval_required",
            "approval": {
                "approval_policy_id": 17,
                "tool_name": "Bash",
                "context_key": "user:1|persona:researcher",
                "conversation_id": "sess-1",
                "scope_key": "tool:Bash",
                "reason": "outside_profile",
                "duration_options": ["once", "session"],
            },
        }

    proto._has_module_permission = _allow_mod  # type: ignore
    proto._has_tool_permission = _allow_tool  # type: ignore
    proto._resolve_effective_tool_policy = _resolve_policy  # type: ignore[attr-defined]
    proto._evaluate_runtime_approval = _require_approval  # type: ignore[attr-defined]

    ctx = RequestContext(
        request_id="mcp-hub-policy-approval",
        user_id="1",
        client_id="unit",
        session_id="sess-1",
        metadata={"mcp_policy_context_enabled": True, "persona_id": "researcher"},
    )
    request = MCPRequest(
        method="tools/call",
        params={"name": "Bash", "arguments": {"command": "git status"}},
        id="approval-1",
    )

    response = await proto.process_request(request, ctx)
    assert response.error is not None
    assert response.error.code == -32001
    assert response.error.data["approval"]["reason"] == "outside_profile"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_help_only_advertises_commands_backed_by_visible_tools():
    class _ProtocolStub:
        async def _handle_tools_list(self, params, context):  # noqa: ANN001
            return {
                "tools": [
                    {"name": "fs.list", "module": "filesystem", "canExecute": True},
                    {"name": "knowledge.search", "module": "knowledge", "canExecute": True},
                    {"name": "knowledge.get", "module": "knowledge", "canExecute": False},
                    {"name": "mcp.tools.list", "module": "mcp", "canExecute": False},
                ]
            }

        def _is_tool_allowed_by_context(self, tool_name, tool_args, context):  # noqa: ANN001
            allowed = set((context.metadata or {}).get("allowed_tools") or [])
            return not allowed or tool_name in allowed

        async def _resolve_effective_tool_policy(self, context):  # noqa: ANN001
            return {
                "enabled": True,
                "allowed_tools": ["fs.list"],
                "denied_tools": [],
            }

        def _is_tool_allowed_by_effective_policy(self, tool_name, tool_args, policy):  # noqa: ANN001
            allowed = set(policy.get("allowed_tools") or [])
            return not allowed or tool_name in allowed

    module = RunCommandModule(
        ModuleConfig(name="run", settings={"protocol": _ProtocolStub()}),
    )
    context = RequestContext(
        request_id="run-help-filtered",
        user_id="1",
        client_id="unit",
        metadata={"allowed_tools": ["fs.list", "knowledge.search"]},
    )

    rendered = await module.execute_tool("run", {"command": "help"}, context=context)

    assert "ls" in rendered
    assert "knowledge" not in rendered
    assert "cat" not in rendered
    assert "mcp" not in rendered
