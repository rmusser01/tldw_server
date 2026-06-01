from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest
from mcp_unified.federation.models import BrokeredExternalCredential
from mcp_unified.federation.stdio_transport import (
    StdioExternalTransport,
    StdioExternalTransportError,
    create_external_transport,
)
from mcp_unified.storage import ExternalServerDefinition

_STUB_SERVER_SCRIPT = r"""
import json
import os
import sys
import time


def send(payload):
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    message = json.loads(line)
    request_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}

    if method == "initialize":
        send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "stub-stdio"},
            },
        })
        continue

    if method == "ping":
        send({"jsonrpc": "2.0", "id": request_id, "result": {"pong": True}})
        continue

    if method == "tools/list":
        send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": "docs.search",
                        "description": "Search docs",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                        },
                        "metadata": {"category": "read"},
                    },
                    {"name": "docs.defaulted", "description": 7, "inputSchema": "bad", "metadata": []},
                    {"name": 42, "description": "invalid"},
                ]
            },
        })
        continue

    if method == "tools/call":
        tool_name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        runtime_auth = (params.get("_meta") or {}).get("mcp_unified_runtime_auth") or {}

        if tool_name == "docs.search":
            send({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": f"search:{arguments.get('q')}"}],
                    "isError": False,
                },
            })
            continue

        if tool_name == "docs.env":
            send({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": {
                        "allowed": os.environ.get("MCP_ALLOWED"),
                        "blocked": os.environ.get("MCP_BLOCKED"),
                    },
                    "isError": False,
                },
            })
            continue

        if tool_name == "docs.auth":
            send({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": {
                        "has_secret": runtime_auth.get("env", {}).get("DOCS_TOKEN") == arguments.get("expected"),
                        "header_keys": sorted(runtime_auth.get("headers", {})),
                        "metadata": runtime_auth.get("metadata", {}),
                    },
                    "isError": False,
                },
            })
            continue

        if tool_name == "docs.fail":
            send({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32042, "message": "upstream failed"},
            })
            continue

        if tool_name == "docs.slow":
            time.sleep(0.3)
            send({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": "slow"}], "isError": False},
            })
            continue

        if tool_name == "docs.exit":
            send({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": {"exiting": True}, "isError": False},
            })
            sys.exit(0)

        send({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"unknown tool: {tool_name}"},
        })
        continue

    send({
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"unknown method: {method}"},
    })
"""


def _write_stub_server(tmp_path: Path) -> str:
    script_path = tmp_path / "stub_stdio_mcp_server.py"
    script_path.write_text(dedent(_STUB_SERVER_SCRIPT), encoding="utf-8")
    return str(script_path)


def _server(
    *,
    command: list[str],
    cwd: str | None = None,
    env_allowlist: list[str] | None = None,
) -> ExternalServerDefinition:
    return ExternalServerDefinition(
        id="docs",
        name="Docs",
        transport="stdio",
        command=command,
        cwd=cwd,
        env_allowlist=env_allowlist or [],
    )


def test_stdio_transport_import_does_not_import_host_package() -> None:
    code = (
        "import sys; import mcp_unified.federation.stdio_transport; "
        "print('tldw_Server_API' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "False"


def test_stdio_transport_rejects_non_stdio_definition() -> None:
    server = ExternalServerDefinition(
        id="ws",
        name="WebSocket",
        transport="websocket",
        url="ws://example.invalid",
    )

    with pytest.raises(StdioExternalTransportError) as exc_info:
        StdioExternalTransport(server)

    assert exc_info.value.reason_code == "unsupported_transport"


def test_stdio_transport_rejects_empty_command_even_when_definition_disabled() -> None:
    server = ExternalServerDefinition(
        id="disabled",
        name="Disabled",
        transport="stdio",
        command=[],
        enabled=False,
    )

    with pytest.raises(StdioExternalTransportError) as exc_info:
        StdioExternalTransport(server)

    assert exc_info.value.reason_code == "missing_command"


def test_stdio_transport_rejects_missing_cwd(tmp_path: Path) -> None:
    server = _server(
        command=[sys.executable, "-c", "pass"],
        cwd=str(tmp_path / "missing"),
    )

    with pytest.raises(StdioExternalTransportError) as exc_info:
        StdioExternalTransport(server)

    assert exc_info.value.reason_code == "invalid_cwd"


def test_create_external_transport_factory_returns_stdio_transport() -> None:
    transport = create_external_transport(_server(command=[sys.executable, "-c", "pass"]))

    assert isinstance(transport, StdioExternalTransport)


@pytest.mark.asyncio
async def test_stdio_transport_connect_list_call_and_close(tmp_path: Path) -> None:
    script_path = _write_stub_server(tmp_path)
    transport = StdioExternalTransport(
        _server(command=[sys.executable, "-u", script_path]),
        request_timeout_s=0.2,
    )

    try:
        await transport.connect()

        health = await transport.health_check()
        assert health["configured"] is True
        assert health["connected"] is True
        assert health["initialized"] is True

        tools = await transport.list_tools()
        assert [tool.name for tool in tools] == ["docs.search", "docs.defaulted"]
        assert tools[0].description == "Search docs"
        assert tools[0].input_schema["type"] == "object"
        assert tools[0].metadata == {"category": "read"}
        assert tools[1].description == ""
        assert tools[1].input_schema == {"type": "object"}
        assert tools[1].metadata == {}

        ok = await transport.call_tool("docs.search", {"q": "hello"})
        assert ok.is_error is False
        assert ok.content == [{"type": "text", "text": "search:hello"}]

        err = await transport.call_tool("docs.fail", {})
        assert err.is_error is True
        assert err.content == [{"type": "text", "text": "upstream failed"}]
        assert err.metadata["reason_code"] == "upstream_error"
    finally:
        await transport.close()
        await transport.close()


@pytest.mark.asyncio
async def test_stdio_transport_uses_only_allowlisted_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = _write_stub_server(tmp_path)
    monkeypatch.setenv("MCP_ALLOWED", "yes")
    monkeypatch.setenv("MCP_BLOCKED", "no")
    transport = StdioExternalTransport(
        _server(
            command=[sys.executable, "-u", script_path],
            env_allowlist=["MCP_ALLOWED"],
        )
    )

    try:
        result = await transport.call_tool("docs.env", {})

        assert result.is_error is False
        assert result.content == {"allowed": "yes", "blocked": None}
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_stdio_transport_sends_runtime_auth_in_meta_without_leaking_secret(
    tmp_path: Path,
) -> None:
    script_path = _write_stub_server(tmp_path)
    secret = "super-secret-token"
    transport = StdioExternalTransport(
        _server(command=[sys.executable, "-u", script_path]),
        request_timeout_s=0.05,
    )

    try:
        result = await transport.call_tool(
            "docs.auth",
            {"expected": secret},
            runtime_auth=BrokeredExternalCredential(
                headers={"Authorization": "Bearer header-secret"},
                env={"DOCS_TOKEN": secret},
                metadata={"credential_source": "pytest"},
            ),
        )

        assert result.is_error is False
        assert result.content == {
            "has_secret": True,
            "header_keys": ["Authorization"],
            "metadata": {"credential_source": "pytest"},
        }

        with pytest.raises(StdioExternalTransportError) as exc_info:
            await transport.call_tool(
                "docs.slow",
                {},
                runtime_auth=BrokeredExternalCredential(env={"DOCS_TOKEN": secret}),
            )

        assert exc_info.value.reason_code == "request_timeout"
        assert secret not in str(exc_info.value)
        assert "header-secret" not in str(exc_info.value)
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_stdio_transport_health_marks_exited_process_disconnected(tmp_path: Path) -> None:
    script_path = _write_stub_server(tmp_path)
    transport = StdioExternalTransport(_server(command=[sys.executable, "-u", script_path]))

    try:
        await transport.connect()
        result = await transport.call_tool("docs.exit", {})
        assert result.content == {"exiting": True}

        health = await transport.health_check()
        assert health["connected"] is False
        assert health["initialized"] is False
    finally:
        await transport.close()
