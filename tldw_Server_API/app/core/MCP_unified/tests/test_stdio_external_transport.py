"""Subprocess-backed tests for the package upstream stdio MCP transport."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from time import monotonic

import pytest
from mcp_unified.federation.models import BrokeredExternalCredential
from mcp_unified.federation.process_policy import (
    StdioProcessPolicy,
    coerce_stdio_process_policy,
)
from mcp_unified.federation.stdio_transport import (
    StdioExternalTransport,
    StdioExternalTransportError,
    create_external_transport,
)
from mcp_unified.storage import ExternalServerDefinition

_STUB_SERVER_SCRIPT = r"""
import json
import os
import signal
import sys
import time


IGNORE_SIGTERM = False
initialized_notification = False

if IGNORE_SIGTERM:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


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
        if not isinstance(params.get("capabilities"), dict):
            send({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32602, "message": "missing capabilities"},
            })
            continue
        send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "stub-stdio"},
            },
        })
        continue

    if method == "notifications/initialized":
        initialized_notification = True
        continue

    if not initialized_notification:
        send({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32002, "message": "session not initialized"},
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


def _write_stub_server(tmp_path: Path, *, ignore_sigterm: bool = False) -> str:
    """Write a temporary MCP-like stdio server for transport tests."""
    script_path = tmp_path / "stub_stdio_mcp_server.py"
    script = dedent(_STUB_SERVER_SCRIPT)
    if ignore_sigterm:
        script = script.replace("IGNORE_SIGTERM = False", "IGNORE_SIGTERM = True")
    script_path.write_text(script, encoding="utf-8")
    return str(script_path)


def _server(
    *,
    command: list[str],
    cwd: str | None = None,
    env_allowlist: list[str] | None = None,
) -> ExternalServerDefinition:
    """Build a stdio external server definition for tests."""
    return ExternalServerDefinition(
        id="docs",
        name="Docs",
        transport="stdio",
        command=command,
        cwd=cwd,
        env_allowlist=env_allowlist or [],
    )


def test_stdio_transport_import_does_not_import_host_package() -> None:
    """Importing the package stdio transport must not import host code."""
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
    """The stdio transport rejects non-stdio server definitions."""
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
    """The transport validates its own command contract for disabled rows too."""
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
    """Configured cwd paths must resolve to existing directories."""
    server = _server(
        command=[sys.executable, "-c", "pass"],
        cwd=str(tmp_path / "missing"),
    )

    with pytest.raises(StdioExternalTransportError) as exc_info:
        StdioExternalTransport(server)

    assert exc_info.value.reason_code == "invalid_cwd"


def test_stdio_transport_rejects_bare_command_without_path_allowlist() -> None:
    """Bare executable names require PATH allowlisting for predictable launch."""
    server = _server(command=["python"])

    with pytest.raises(StdioExternalTransportError) as exc_info:
        StdioExternalTransport(server)

    assert exc_info.value.reason_code == "invalid_command"
    assert "PATH" in str(exc_info.value)


def test_stdio_process_policy_rejects_invalid_mapping_values() -> None:
    """Policy config rejects ambiguous values before runtime bootstrap."""
    with pytest.raises(ValueError, match="allowed_executables"):
        coerce_stdio_process_policy({"allowed_executables": ["python", ""]})

    with pytest.raises(ValueError, match="allow_path_lookup"):
        coerce_stdio_process_policy({"allow_path_lookup": "false"})

    with pytest.raises(ValueError, match="allowed_env_names"):
        coerce_stdio_process_policy({"allowed_env_names": ["TOKEN", 7]})


def test_stdio_process_policy_defaults_block_shell_wrappers() -> None:
    """Default process policy blocks direct shell-wrapper launches."""
    secret_arg = "do-not-leak-shell-argument"
    server = _server(command=["/bin/bash", "-lc", secret_arg])

    with pytest.raises(StdioExternalTransportError) as exc_info:
        StdioExternalTransport(server)

    assert exc_info.value.reason_code == "process_policy_shell_denied"
    assert secret_arg not in str(exc_info.value)
    assert secret_arg not in repr(exc_info.value.details)


def test_stdio_process_policy_allows_explicit_shell_executable() -> None:
    """Hosts can deliberately allow a shell executable through allowlisting."""
    policy = StdioProcessPolicy(allowed_executables=("/bin/bash",))
    transport = StdioExternalTransport(
        _server(command=["/bin/bash", "--version"]),
        process_policy=policy,
    )

    assert transport.server_id == "docs"


def test_stdio_process_policy_path_lookup_can_be_disabled() -> None:
    """Strict deployments can reject bare executables even when PATH is allowlisted."""
    server = _server(command=["python"], env_allowlist=["PATH"])

    with pytest.raises(StdioExternalTransportError) as exc_info:
        StdioExternalTransport(
            server,
            process_policy=StdioProcessPolicy(allow_path_lookup=False),
        )

    assert exc_info.value.reason_code == "process_policy_path_lookup_denied"


def test_stdio_process_policy_env_allowlist_must_allow_path_for_bare_command() -> None:
    """Policy env allowlists also gate PATH inheritance for bare executable lookup."""
    server = _server(command=["python"], env_allowlist=["PATH"])

    with pytest.raises(StdioExternalTransportError) as exc_info:
        StdioExternalTransport(
            server,
            process_policy=StdioProcessPolicy(allowed_env_names=("OTHER",)),
        )

    assert exc_info.value.reason_code == "process_policy_env_denied"
    assert exc_info.value.details["env_name"] == "PATH"


def test_stdio_process_policy_rejects_cwd_outside_allowed_roots(tmp_path: Path) -> None:
    """Configured cwd values must stay inside deployment-approved roots."""
    allowed_root = tmp_path / "allowed"
    denied_root = tmp_path / "denied"
    allowed_root.mkdir()
    denied_root.mkdir()
    server = _server(command=[sys.executable, "-c", "pass"], cwd=str(denied_root))

    with pytest.raises(StdioExternalTransportError) as exc_info:
        StdioExternalTransport(
            server,
            process_policy=StdioProcessPolicy(allowed_cwd_roots=(allowed_root,)),
        )

    assert exc_info.value.reason_code == "process_policy_cwd_denied"


def test_stdio_process_policy_uses_default_cwd_inside_allowed_root(tmp_path: Path) -> None:
    """Policy default cwd is used when a server does not configure one."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    transport = StdioExternalTransport(
        _server(command=[sys.executable, "-c", "pass"]),
        process_policy=StdioProcessPolicy(
            allowed_cwd_roots=(workspace,),
            default_cwd=workspace,
        ),
    )

    assert transport._cwd == str(workspace.resolve())


def test_stdio_process_policy_rejects_disallowed_environment_names() -> None:
    """Policy env allowlists reject server-requested environment names outside policy."""
    with pytest.raises(StdioExternalTransportError) as exc_info:
        StdioExternalTransport(
            _server(
                command=[sys.executable, "-c", "pass"],
                env_allowlist=["MCP_ALLOWED", "MCP_BLOCKED"],
            ),
            process_policy=StdioProcessPolicy(allowed_env_names=("MCP_ALLOWED",)),
        )

    assert exc_info.value.reason_code == "process_policy_env_denied"
    assert exc_info.value.details["env_name"] == "MCP_BLOCKED"


@pytest.mark.asyncio
async def test_stdio_process_policy_allows_policy_approved_environment_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Approved policy env names still inherit only values present in os.environ."""
    script_path = _write_stub_server(tmp_path)
    monkeypatch.setenv("MCP_ALLOWED", "yes")
    transport = StdioExternalTransport(
        _server(
            command=[sys.executable, "-u", script_path],
            env_allowlist=["MCP_ALLOWED"],
        ),
        process_policy=StdioProcessPolicy(allowed_env_names=("MCP_ALLOWED",)),
    )

    try:
        result = await transport.call_tool("docs.env", {})

        assert result.is_error is False
        assert result.content == {"allowed": "yes", "blocked": None}
    finally:
        await transport.close()


@pytest.mark.parametrize("timeout", [0, -1, "bad", float("nan"), float("inf")])
def test_stdio_transport_rejects_invalid_request_timeouts(timeout: object) -> None:
    """Timeout configuration errors are reported as structured transport errors."""
    with pytest.raises(StdioExternalTransportError) as exc_info:
        StdioExternalTransport(
            _server(command=[sys.executable, "-c", "pass"]),
            request_timeout_s=timeout,  # type: ignore[arg-type]
        )

    assert exc_info.value.reason_code == "invalid_timeout"


def test_create_external_transport_factory_returns_stdio_transport() -> None:
    """The package factory returns the stdio transport for stdio definitions."""
    transport = create_external_transport(_server(command=[sys.executable, "-c", "pass"]))

    assert isinstance(transport, StdioExternalTransport)


@pytest.mark.asyncio
async def test_stdio_transport_connect_list_call_and_close(tmp_path: Path) -> None:
    """The transport initializes, sends initialized notification, lists, calls, and closes."""
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
    """Child process environments include only explicitly allowlisted variables."""
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
    """Runtime credentials are passed through call metadata without leaking in errors."""
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
async def test_stdio_transport_timeout_does_not_wait_for_shutdown_timeout(tmp_path: Path) -> None:
    """Request timeout returns before asynchronous process cleanup finishes."""
    script_path = _write_stub_server(tmp_path, ignore_sigterm=True)
    transport = StdioExternalTransport(
        _server(command=[sys.executable, "-u", script_path]),
        request_timeout_s=0.05,
        close_timeout_s=0.4,
    )

    try:
        await transport.connect()
        started_at = monotonic()
        with pytest.raises(StdioExternalTransportError) as exc_info:
            await transport.call_tool("docs.slow", {})
        elapsed = monotonic() - started_at

        assert exc_info.value.reason_code == "request_timeout"
        assert elapsed < 0.25
        await asyncio.sleep(0.5)
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_stdio_transport_health_marks_exited_process_disconnected(tmp_path: Path) -> None:
    """Health checks mark a process that exited after a call as disconnected."""
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
