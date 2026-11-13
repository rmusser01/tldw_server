from __future__ import annotations

from copy import deepcopy

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.config import ACPRunnerConfig
from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import ACPRunnerClient
from tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client import ACPMessage, ACPResponseError

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class RecordingClient:
    def __init__(self, *, reject_tools: bool = False) -> None:
        self.calls = []
        self.reject_tools = reject_tools

    async def call(self, method, payload):
        self.calls.append((method, deepcopy(payload)))
        if self.reject_tools and payload.get("mcpServers"):
            raise ACPResponseError("Invalid params", code=-32602)
        return ACPMessage(jsonrpc="2.0", result={"sessionId": "wire-session"})


@pytest.mark.parametrize(
    ("server", "arrays"),
    [
        ({"name": "public", "type": "stdio", "command": "/usr/bin/python3"}, {"args": [], "env": []}),
        (
            {"name": "public", "type": "stdio", "command": "/usr/bin/python3", "args": ["public_server.py"]},
            {"args": ["public_server.py"], "env": []},
        ),
        (
            {"name": "public", "type": "stdio", "command": "/usr/bin/python3", "args": None, "env": None},
            {"args": [], "env": []},
        ),
        ({"name": "public", "type": "http", "url": "http://127.0.0.1:3010/mcp"}, {"headers": []}),
        (
            {"name": "public", "type": "sse", "url": "http://127.0.0.1:3010/events", "headers": None},
            {"headers": []},
        ),
        (
            {
                "name": "public",
                "type": "stdio",
                "command": "/usr/bin/python3",
                "args": ["public_server.py"],
                "env": [{"name": "TRACE", "value": "1"}],
                "_meta": {"purpose": "public-read"},
            },
            {"args": ["public_server.py"], "env": [{"name": "TRACE", "value": "1"}]},
        ),
        (
            {
                "name": "public",
                "type": "http",
                "url": "http://127.0.0.1:3010/mcp",
                "headers": [{"name": "X-UAT", "value": "public"}],
            },
            {"headers": [{"name": "X-UAT", "value": "public"}]},
        ),
    ],
)
async def test_session_wire_has_required_arrays_without_changing_input(tmp_path, server, arrays):
    runner = ACPRunnerClient(
        ACPRunnerConfig(
            command="unused",
            allowed_session_cwd_roots=[str(tmp_path)],
            allow_inline_stdio_mcp_servers=True,
            allow_private_mcp_http=True,
        )
    )
    recording = RecordingClient()
    runner._client = recording
    original = deepcopy(server)

    session = await runner.create_session(str(tmp_path), mcp_servers=[server])

    assert session == "wire-session"  # nosec B101 -- unit behavior contract.
    assert recording.calls == [("session/new", {"cwd": str(tmp_path), "mcpServers": [{**original, **arrays}]})]  # nosec B101 -- full wire shape and preservation.
    assert server == original  # nosec B101 -- caller-owned saved configuration stays unchanged.


@pytest.mark.parametrize("agent_type", [None, "opencode"])
async def test_nonempty_mcp_config_is_not_dropped_after_rejection(tmp_path, agent_type):
    runner = ACPRunnerClient(
        ACPRunnerConfig(
            command="unused", allowed_session_cwd_roots=[str(tmp_path)], allow_inline_stdio_mcp_servers=True
        )
    )
    recording = RecordingClient(reject_tools=True)
    runner._client = recording
    servers = [{"name": "public", "type": "stdio", "command": "/usr/bin/python3", "args": [], "env": []}]

    with pytest.raises(ACPResponseError, match="Invalid params"):
        await runner.create_session(str(tmp_path), mcp_servers=servers, agent_type=agent_type)

    assert len(recording.calls) == (2 if agent_type else 1)  # nosec B101 -- only existing agentType compatibility retry is allowed.
    assert all(payload.get("mcpServers") == servers for _, payload in recording.calls)  # nosec B101 -- rejection cannot silently remove tools.
