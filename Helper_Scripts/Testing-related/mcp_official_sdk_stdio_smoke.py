#!/usr/bin/env python3
"""Official MCP Python SDK interoperability smoke for installed artifacts."""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
import os
import sys
import sysconfig
import tempfile
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp_unified.gateway import GatewayRequestContext

SDK_REQUIREMENT = "mcp==2.0.0"
SUCCESS_MARKER = "MCP_UNIFIED_OFFICIAL_SDK_STDIO_OK"


class _Runtime:
    """Minimal strict runtime used by the official SDK discovery/call smoke."""

    name = "official-sdk-smoke"
    version = "1.0"

    async def list_tools(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        del context
        return [
            {
                "name": "echo",
                "inputSchema": {"type": "object"},
                "outputSchema": {
                    "type": "object",
                    "properties": {"echoed": {"type": "integer"}},
                    "required": ["echoed"],
                    "additionalProperties": False,
                },
            }
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        del context
        if name != "echo" or arguments != {"value": 7}:
            raise AssertionError("official SDK sent an unexpected tool call")
        return {"echoed": 7}

    async def list_resources(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        del context
        return []

    async def read_resource(self, uri: str, context: GatewayRequestContext) -> dict[str, Any]:
        raise AssertionError((uri, context))

    async def list_prompts(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        del context
        return []

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        raise AssertionError((name, arguments, context))


def _assert_installed_module(module: ModuleType) -> None:
    """Reject checkout imports for either package involved in the smoke."""

    module_path = Path(inspect.getfile(module)).resolve()
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    if not module_path.is_relative_to(purelib):
        raise AssertionError(f"module did not import from site-packages: {module_path.name}")
    forbidden = os.environ.get("MCP_UNIFIED_FORBIDDEN_CHECKOUT")
    if forbidden and module_path.is_relative_to(Path(forbidden).resolve()):
        raise AssertionError("module imported from the forbidden checkout")


async def _serve() -> None:
    """Serve the installed strict gateway on native process stdio."""

    from mcp_unified.gateway import serve_stdio

    await serve_stdio(_Runtime())


async def _run_client() -> None:
    """Negotiate, discover tools, and call one tool through the official SDK."""

    import mcp
    import mcp_unified
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters, stdio_client

    if importlib.metadata.version("mcp") != SDK_REQUIREMENT.partition("==")[2]:
        raise AssertionError("official MCP SDK version does not match the release pin")
    _assert_installed_module(mcp)
    _assert_installed_module(mcp_unified)

    child_env = os.environ.copy()
    child_env.pop("PYTHONHOME", None)
    child_env.pop("PYTHONPATH", None)
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).resolve()), "--server"],
        env=child_env,
        cwd=Path.cwd(),
    )
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr:
        async with Client(stdio_client(parameters, errlog=stderr)) as client:
            if str(client.protocol_version) != "2026-07-28":
                raise AssertionError("official SDK did not negotiate the strict current revision")
            tools = await client.list_tools()
            if [tool.name for tool in tools.tools] != ["echo"]:
                raise AssertionError("official SDK tool discovery did not match the strict server")
            result = await client.call_tool("echo", {"value": 7})
            if result.structured_content != {"echoed": 7}:
                raise AssertionError("official SDK tool call result did not preserve structured content")
        stderr.seek(0)
        if stderr.read():
            raise AssertionError("strict stdio server emitted unexpected stderr")


async def _main() -> None:
    if "--server" in sys.argv:
        await _serve()
        return
    await asyncio.wait_for(_run_client(), timeout=30.0)
    print(SUCCESS_MARKER)


if __name__ == "__main__":
    asyncio.run(_main())
