from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.schemas.sandbox_schemas import SandboxRunCreateRequest
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.sandbox_module import SandboxModule
from tldw_Server_API.app.core.Sandbox.models import RuntimeType


def test_run_schema_accepts_vz_linux_runtime() -> None:
    body = {
        "spec_version": "1.0",
        "runtime": "vz_linux",
        "base_image": "ubuntu-24.04",
        "command": ["echo", "ok"],
    }

    model = SandboxRunCreateRequest.model_validate(body)

    assert model.runtime == "vz_linux"


def test_run_schema_accepts_worktree_runtime() -> None:
    body = {
        "spec_version": "1.0",
        "runtime": "worktree",
        "base_image": "host-local",
        "command": ["echo", "ok"],
    }

    model = SandboxRunCreateRequest.model_validate(body)

    assert model.runtime == "worktree"


@pytest.mark.asyncio
async def test_mcp_tool_schema_lists_new_macos_runtimes() -> None:
    module = SandboxModule(ModuleConfig(name="sandbox"))

    tools = await module.get_tools()
    tool = next(item for item in tools if item["name"] == "sandbox.run")

    assert tool["inputSchema"]["properties"]["runtime"]["enum"] == [
        runtime.value for runtime in RuntimeType
    ]


def test_mcp_tool_validation_accepts_worktree_runtime() -> None:
    module = SandboxModule(ModuleConfig(name="sandbox"))

    module.validate_tool_arguments(
        "sandbox.run",
        {
            "runtime": "worktree",
            "base_image": "host-local",
            "command": ["echo", "ok"],
        },
    )

    assert module._coerce_runtime("worktree") == RuntimeType.worktree
