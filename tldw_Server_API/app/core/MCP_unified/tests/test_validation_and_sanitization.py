"""
Validation and sanitization tests for MCP Unified (tool name regex, deep arg sanitization).
"""

import os
from typing import Any

import pytest

# Minimize startup side-effects for tests
os.environ.setdefault("TEST_MODE", "true")
os.environ.setdefault("ENABLE_TRACING", "false")
os.environ.setdefault("OTEL_METRICS_EXPORTER", "console")

from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext


class InlineSanitizeModule(BaseModule):
    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [{
            "name": "echo_sanitize",
            "description": "Echo a message with deep sanitization",
            "inputSchema": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"]
            }
        }]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
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
    assert resp.error.code == -32602  # nosec B101
    assert "Invalid tool name" in (resp.error.message or "")  # nosec B101


@pytest.mark.asyncio
async def test_deep_argument_sanitization_recurses_into_nested_values():
    """Sanitization must reach values nested inside dicts and lists.

    This test previously asserted that a nested "/* injected */" raised ValueError.
    That denylist was removed from BaseModule.sanitize_input (see its docstring): it
    rejected ordinary content -- a Markdown "---", a git pathspec "-- src/app.py", the
    glob "src/*.py" -- with a security-flavoured error, while buying no real injection
    protection, since these values are passed to parameterised queries. The property
    worth keeping is the recursion itself, so it is asserted here against the surviving
    behaviour: control characters are stripped at every depth.
    """
    mod = InlineSanitizeModule(ModuleConfig(name="inline"))
    # Safe case round-trips unchanged
    msg = os.urandom(4).hex()
    out = await mod.execute_tool("echo_sanitize", {"message": msg})
    assert out == msg  # nosec B101

    # Nested values are reached, at dict and list depth
    cleaned = mod.sanitize_input({"a": {"b": ["x\x07y", {"c": "p\x00q"}]}})
    assert cleaned == {"a": {"b": ["xy", {"c": "pq"}]}}  # nosec B101

    # Tabs/newlines inside nested values survive -- they are data, not control chars
    assert mod.sanitize_input({"f": {"body": "line\n\tindented"}}) == {  # nosec B101
        "f": {"body": "line\n\tindented"}
    }

    # The depth guard still rejects abusive nesting
    deep: Any = "leaf"
    for _ in range(25):
        deep = {"n": deep}
    with pytest.raises(ValueError):
        mod.sanitize_input(deep)
