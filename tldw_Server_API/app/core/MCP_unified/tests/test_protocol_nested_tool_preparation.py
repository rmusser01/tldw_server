import pytest

from tldw_Server_API.app.core.MCP_unified.config import get_config
from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig, create_tool_definition
from tldw_Server_API.app.core.MCP_unified.modules.registry import get_module_registry
from tldw_Server_API.app.core.MCP_unified.protocol import (
    InvalidParamsException,
    MCPProtocol,
    RequestContext,
)


@pytest.fixture(autouse=True)
def _clear_mcp_config_cache():
    try:
        get_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None
    yield
    try:
        get_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None


class AllowAllRBAC:
    async def check_permission(self, *args, **kwargs):
        return True


class _NonJsonArgument:
    pass


class DynamicModeToolModule(BaseModule):
    TOOL_NAME = "dynamic_mode_action"

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict:
        return {"ready": True}

    async def get_tools(self) -> list[dict]:
        return [
            create_tool_definition(
                name=self.TOOL_NAME,
                description="Single tool that can be read-only or write-capable by args",
                parameters={
                    "properties": {
                        "mode": {"type": "string"},
                        "target": {"type": "string"},
                    },
                    "required": ["mode", "target"],
                },
                metadata={"category": "read"},
            )
        ]

    def is_write_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, object],
        tool_def: dict[str, object] | None = None,
    ) -> bool:
        if tool_name != self.TOOL_NAME:
            return False
        return str(arguments.get("mode", "")).lower() == "write"

    def validate_tool_arguments(self, tool_name: str, arguments: dict):
        if tool_name == self.TOOL_NAME and not arguments.get("target"):
            raise ValueError("'target' is required")

    async def execute_tool(self, tool_name: str, arguments: dict, context=None):
        return f"{arguments.get('mode')}:{arguments.get('target')}"


@pytest.mark.asyncio
async def test_prepare_tool_call_classifies_dynamic_write_and_enforces_write_policy(monkeypatch):
    registry = get_module_registry()
    await registry.register_module(
        "dynamic_mode_write_prepare",
        DynamicModeToolModule,
        ModuleConfig(name="dynamic_mode_write_prepare"),
    )

    proto = MCPProtocol()
    proto.rbac_policy = AllowAllRBAC()
    ctx = RequestContext(request_id="np1", user_id="u1", client_id="c1")

    prepared = await proto.prepare_tool_call(
        params={
            "name": DynamicModeToolModule.TOOL_NAME,
            "arguments": {"mode": "write", "target": "alpha"},
        },
        context=ctx,
    )
    assert prepared.is_write is True

    monkeypatch.setenv("MCP_DISABLE_WRITE_TOOLS", "true")
    try:
        get_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None

    proto_blocked = MCPProtocol()
    proto_blocked.rbac_policy = AllowAllRBAC()

    with pytest.raises(PermissionError, match="Write tools are disabled"):
        await proto_blocked.prepare_tool_call(
            params={
                "name": DynamicModeToolModule.TOOL_NAME,
                "arguments": {"mode": "write", "target": "alpha"},
            },
            context=ctx,
        )


@pytest.mark.asyncio
async def test_prepare_tool_call_allows_read_variant_when_write_tools_disabled(monkeypatch):
    monkeypatch.setenv("MCP_DISABLE_WRITE_TOOLS", "true")

    registry = get_module_registry()
    await registry.register_module(
        "dynamic_mode_read_prepare",
        DynamicModeToolModule,
        ModuleConfig(name="dynamic_mode_read_prepare"),
    )

    proto = MCPProtocol()
    proto.rbac_policy = AllowAllRBAC()
    ctx = RequestContext(request_id="np2", user_id="u2", client_id="c2")

    prepared = await proto.prepare_tool_call(
        params={
            "name": DynamicModeToolModule.TOOL_NAME,
            "arguments": {"mode": "read", "target": "beta"},
        },
        context=ctx,
    )
    assert prepared.is_write is False

    result = await proto.execute_prepared_tool_call(prepared)
    assert any(part.get("text") == "read:beta" for part in result.get("content", []))


@pytest.mark.asyncio
async def test_prepare_tool_call_classifies_write_after_null_byte_sanitization(monkeypatch):
    monkeypatch.setenv("MCP_DISABLE_WRITE_TOOLS", "true")

    registry = get_module_registry()
    await registry.register_module(
        "dynamic_mode_null_byte_prepare",
        DynamicModeToolModule,
        ModuleConfig(name="dynamic_mode_null_byte_prepare"),
    )

    proto = MCPProtocol()
    proto.rbac_policy = AllowAllRBAC()
    ctx = RequestContext(request_id="np3", user_id="u3", client_id="c3")

    with pytest.raises(PermissionError, match="Write tools are disabled"):
        await proto.prepare_tool_call(
            params={
                "name": DynamicModeToolModule.TOOL_NAME,
                "arguments": {"mode": "write\0", "target": "gamma"},
            },
            context=ctx,
        )


@pytest.mark.asyncio
async def test_execute_prepared_tool_call_rejects_mutated_prepared_arguments():
    registry = get_module_registry()
    await registry.register_module(
        "dynamic_mode_mutation_prepare",
        DynamicModeToolModule,
        ModuleConfig(name="dynamic_mode_mutation_prepare"),
    )

    proto = MCPProtocol()
    proto.rbac_policy = AllowAllRBAC()
    ctx = RequestContext(request_id="np4", user_id="u4", client_id="c4")

    prepared = await proto.prepare_tool_call(
        params={
            "name": DynamicModeToolModule.TOOL_NAME,
            "arguments": {"mode": "read", "target": "delta"},
        },
        context=ctx,
    )

    # Attempt to mutate the call after preparation; integrity checks should block execution.
    assert isinstance(prepared.tool_args, dict)
    prepared.tool_args["mode"] = "write"

    with pytest.raises(
        InvalidParamsException,
        match="Prepared tool call integrity check failed",
    ):
        await proto.execute_prepared_tool_call(prepared)


@pytest.mark.asyncio
async def test_execute_prepared_tool_call_rejects_mutated_request_context() -> None:
    registry = get_module_registry()
    await registry.register_module(
        "dynamic_mode_context_mutation_prepare",
        DynamicModeToolModule,
        ModuleConfig(name="dynamic_mode_context_mutation_prepare"),
    )

    proto = MCPProtocol()
    proto.rbac_policy = AllowAllRBAC()
    ctx = RequestContext(
        request_id="np5",
        user_id="u5",
        client_id="c5",
        session_id="sess-1",
        metadata={"workspace_id": "workspace-1"},
    )

    prepared = await proto.prepare_tool_call(
        params={
            "name": DynamicModeToolModule.TOOL_NAME,
            "arguments": {"mode": "read", "target": "epsilon"},
        },
        context=ctx,
    )

    prepared.context.metadata["workspace_id"] = "workspace-2"

    with pytest.raises(
        InvalidParamsException,
        match="Prepared tool call integrity check failed",
    ):
        await proto.execute_prepared_tool_call(prepared)


@pytest.mark.asyncio
async def test_prepare_tool_call_maps_non_json_arguments_to_invalid_params() -> None:
    registry = get_module_registry()
    await registry.register_module(
        "dynamic_mode_non_json_prepare",
        DynamicModeToolModule,
        ModuleConfig(name="dynamic_mode_non_json_prepare"),
    )

    proto = MCPProtocol()
    proto.rbac_policy = AllowAllRBAC()
    ctx = RequestContext(request_id="np6", user_id="u6", client_id="c6")

    with pytest.raises(InvalidParamsException, match="valid JSON"):
        await proto.prepare_tool_call(
            params={
                "name": DynamicModeToolModule.TOOL_NAME,
                "arguments": _NonJsonArgument(),
            },
            context=ctx,
        )
