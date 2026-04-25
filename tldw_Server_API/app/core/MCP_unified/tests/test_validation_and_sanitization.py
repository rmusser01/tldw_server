"""
Validation and sanitization tests for MCP Unified (tool name regex, deep arg sanitization).
"""

import os
import pytest
import os as _os

# Minimize startup side-effects for tests
_os.environ.setdefault("TEST_MODE", "true")
_os.environ.setdefault("ENABLE_TRACING", "false")
_os.environ.setdefault("OTEL_METRICS_EXPORTER", "console")
from typing import Dict, Any

from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext
from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig


class InlineSanitizeModule(BaseModule):
    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> Dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[Dict[str, Any]]:
        return [{
            "name": "echo_sanitize",
            "description": "Echo a message with deep sanitization",
            "inputSchema": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"]
            }
        }]

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], context: Any | None = None) -> Any:
        args = self.sanitize_input(arguments)
        if tool_name == "echo_sanitize":
            return args.get("message")
        raise ValueError("unknown tool")


@pytest.mark.asyncio
async def test_tool_name_strict_regex_blocks_invalid():
    proto = MCPProtocol()
    ctx = RequestContext(request_id="rx-1", user_id="user1", client_id="c1")
    # Invalid tool name with semicolon
    req = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": "bad;name", "arguments": {}},
        "id": 1,
    }
    resp = await proto.process_request(req, ctx)
    assert resp is not None and resp.error is not None  # nosec B101
    assert resp.error.code == -32603  # nosec B101
    assert "Invalid tool name" in (resp.error.message or "")  # nosec B101


@pytest.mark.asyncio
async def test_deep_argument_sanitization_blocks_nested_patterns():
    mod = InlineSanitizeModule(ModuleConfig(name="inline"))
    # Safe case
    msg = os.urandom(4).hex()
    out = await mod.execute_tool("echo_sanitize", {"message": msg})
    assert out == msg  # nosec B101
    # Nested dangerous pattern should raise
    with pytest.raises(ValueError):
        await (
            mod.execute_tool("echo_sanitize", {"message": "ok", "nested": {"bad": "/* injected */"}})
        )
