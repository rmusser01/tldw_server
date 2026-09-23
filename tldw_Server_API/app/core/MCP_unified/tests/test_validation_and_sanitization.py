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
async def test_deep_argument_sanitization_reaches_nested_values():
    """Sanitization recurses, and what it does at depth is strip control characters.

    This previously asserted that a nested "/* injected */" raised ValueError, which
    pinned a defect rather than a contract. The base sanitizer carried a denylist of
    SQL-injection substrings ("--", "/*", "*/", "xp_", "sp_" and friends) applied to
    data that is bound to parameterised queries, so it blocked nothing an attacker
    would do while rejecting a filename containing "xp_", every glob, every markdown
    rule and every git pathspec. See TASK-13294.

    Its "safe" fixture was os.urandom(4).hex(), which cannot contain any denied
    substring, so nothing here asserted that legitimate content survived. It does now.
    """
    mod = InlineSanitizeModule(ModuleConfig(name="inline"))

    # Legitimate content that the old denylist rejected must pass through untouched.
    for message in ("src/*.py", "git log -- path", "exp_data.csv", "--- rule"):
        assert await mod.execute_tool("echo_sanitize", {"message": message}) == message  # nosec B101

    # Recursion still happens: a control character nested two levels down is removed.
    out = await mod.execute_tool(
        "echo_sanitize",
        {"message": "ok", "nested": {"deep": ["a\x01b"]}},
    )
    assert out == "ok"  # nosec B101

    # And the depth guard is still the thing that rejects abuse.
    nested: Any = "leaf"
    for _ in range(25):
        nested = {"k": nested}
    with pytest.raises(ValueError, match="too deeply nested"):
        await mod.execute_tool("echo_sanitize", {"message": "ok", "nested": nested})
