from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tldw_Server_API.app.core.MCP_unified.external_servers.config_schema import (
    ExternalMCPServerConfig,
    ExternalStdioConfig,
    ExternalTimeoutConfig,
    ExternalTransportType,
)
from tldw_Server_API.app.core.MCP_unified.external_servers.transports.stdio_adapter import (
    StdioExternalMCPAdapter,
)


_STUB_SERVER_SCRIPT = """\
import json
import sys
import time


def _send(payload):
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\\n")
    sys.stdout.flush()


for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        continue

    request_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}

    if method == "initialize":
        _send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"protocolVersion": "2024-11-05", "serverInfo": {"name": "stub-stdio"}},
            }
        )
        continue

    if method == "tools/list":
        _send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "docs.search",
                            "description": "Search docs",
                            "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}},
                            "metadata": {"category": "read"},
                        },
                        {"name": "docs.defaulted", "description": 7, "inputSchema": "bad", "metadata": []},
                    ]
                },
            }
        )
        continue

    if method == "resources/list":
        _send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "resources": [
                        {
                            "uri": "resource://docs/readme",
                            "name": "Docs README",
                            "description": "Documentation entry",
                            "mimeType": "text/markdown",
                            "metadata": {"category": "read"},
                        },
                        {"uri": "resource://docs/defaulted", "name": 7, "metadata": []},
                    ]
                },
            }
        )
        continue

    if method == "resources/read":
        uri = str(params.get("uri") or "")
        if uri == "resource://docs/readme":
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "contents": [
                            {
                                "uri": "resource://docs/readme",
                                "mimeType": "text/markdown",
                                "text": "# Docs",
                            }
                        ]
                    },
                }
            )
            continue

        _send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"unknown resource: {uri}"},
            }
        )
        continue

    if method == "tools/call":
        tool_name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}

        if tool_name == "docs.search":
            query = arguments.get("q")
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": f"search:{query}"}],
                        "isError": False,
                    },
                }
            )
            continue

        if tool_name == "docs.fail":
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32042, "message": "upstream failed"},
                }
            )
            continue

        if tool_name == "docs.slow":
            time.sleep(0.3)
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": "slow"}], "isError": False},
                }
            )
            continue

        _send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"unknown tool: {tool_name}"},
            }
        )
        continue

    _send(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"unknown method: {method}"},
        }
    )
"""


def _write_stub_server(tmp_path: Path) -> str:
    script_path = tmp_path / "stub_external_stdio_server.py"
    script_path.write_text(_STUB_SERVER_SCRIPT, encoding="utf-8")
    return str(script_path)


@pytest.mark.asyncio
async def test_stdio_adapter_subprocess_roundtrip_error_and_timeout(tmp_path: Path) -> None:
    script_path = _write_stub_server(tmp_path)
    cfg = ExternalMCPServerConfig(
        id="docs",
        name="Docs",
        transport=ExternalTransportType.STDIO,
        stdio=ExternalStdioConfig(command=sys.executable, args=["-u", script_path]),
        timeouts=ExternalTimeoutConfig(connect_seconds=1.0, request_seconds=0.1),
    )
    adapter = StdioExternalMCPAdapter(cfg)

    try:
        await adapter.connect()

        health = await adapter.health_check()
        assert health["configured"] is True
        assert health["connected"] is True
        assert health["initialized"] is True

        tools = await adapter.list_tools()
        assert [tool.name for tool in tools] == ["docs.search", "docs.defaulted"]
        assert tools[1].input_schema == {"type": "object"}
        assert tools[1].metadata == {}

        resources = await adapter.list_resources()
        assert resources == [
            {
                "uri": "resource://docs/readme",
                "name": "Docs README",
                "description": "Documentation entry",
                "mimeType": "text/markdown",
                "metadata": {"category": "read"},
            },
            {
                "uri": "resource://docs/defaulted",
                "name": "",
                "description": "",
                "mimeType": None,
                "metadata": {},
            },
        ]
        resource = await adapter.read_resource("resource://docs/readme")
        assert resource["contents"][0]["text"] == "# Docs"

        ok = await adapter.call_tool("docs.search", {"q": "hello"})
        assert ok.is_error is False
        assert ok.content == [{"type": "text", "text": "search:hello"}]

        err = await adapter.call_tool("docs.fail", {})
        assert err.is_error is True
        assert isinstance(err.content, list)
        assert err.content[0]["text"] == "upstream failed"

        with pytest.raises(TimeoutError, match="tools/call"):
            await adapter.call_tool("docs.slow", {})
    finally:
        await adapter.close()
