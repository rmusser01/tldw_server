"""Shared fixtures for ACP tests.

Provides a ``stub_agent_process`` fixture that spawns the stub agent
via the real ``ACPStdioClient``, enabling integration-level tests of the
full STDIO JSON-RPC protocol without needing the Go runner binary.
"""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client import ACPMessage, ACPStdioClient


STUB_AGENT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "Helper_Scripts",
    "acp_stub_agent.py",
)


@pytest.fixture
async def stub_stdio_client() -> ACPStdioClient:
    """Spawn the stub agent via ACPStdioClient for integration tests.

    Yields a started client; shuts it down after the test.
    """
    client = ACPStdioClient(
        command=sys.executable,
        args=[os.path.abspath(STUB_AGENT_PATH)],
    )
    await client.start()
    yield client
    await client.close()


@pytest.fixture
async def admitted_permission_prompt(monkeypatch):
    """Give isolated policy helpers a real prompt admission at a held RPC seam.

    This fixture qualifies only client helper behavior. Compiled-runner and
    authenticated socket evidence lives in the separate normal wire module.
    """
    admitted = []

    async def admit(client, session_id):
        started = asyncio.Event()
        finished = asyncio.Event()
        original_call = client._client.call

        async def held_call(method, params, *, timeout_sec=None, on_response_received=None):
            if method != "session/prompt":
                return await original_call(method, params, timeout_sec=timeout_sec, on_response_received=on_response_received)
            if params.get("sessionId") != session_id:
                raise AssertionError("Unexpected admitted prompt session")
            started.set()
            await finished.wait()
            if on_response_received is not None:
                on_response_received()
            return ACPMessage(jsonrpc="2.0", id="held-policy-prompt", result={"stopReason": "end_turn"})

        monkeypatch.setattr(client._client, "call", held_call)
        task = asyncio.create_task(client.prompt(session_id, [{"type": "text", "text": "Policy unit input"}]))
        admitted.append((finished, task))
        await asyncio.wait_for(started.wait(), timeout=3)

    try:
        yield admit
    finally:
        for finished, _task in admitted:
            finished.set()
        await asyncio.gather(*(task for _, task in admitted), return_exceptions=True)


@pytest.fixture
def mcp_http_egress(monkeypatch):
    """Run an MCP HTTP transport under the production egress policy, without a network.

    Private and link-local addresses are blocked. Tests send traffic through
    ``create_async_client(transport=httpx.MockTransport(...))``; the proof that a
    refused request never went out is the MockTransport's own record of requests.
    httpx connects through anyio, not ``socket.create_connection``, so the patch
    below only catches a stdlib or urllib fetch.
    """
    import socket

    def _no_real_network(*_args, **_kwargs):
        raise AssertionError("MCP HTTP transport bypassed core.http_client")

    monkeypatch.setattr(socket, "create_connection", _no_real_network)
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")


@pytest.fixture
def mcp_dns_rebinding(monkeypatch, mcp_http_egress):
    """DNS that answers public to validate_mcp_http_url and loopback at connect time.

    http_client skips the DNS check for hostnames under pytest; that skip is
    switched off here so the egress policy resolves the host as in production.
    """
    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Agent_Client_Protocol import hardening
    from tldw_Server_API.app.core.Security import egress

    monkeypatch.setattr(hardening, "_resolved_hostname_addresses", lambda _host: ("93.184.216.34",))
    monkeypatch.setattr(egress, "_resolve_host_ips", lambda _host, **_kwargs: ["127.0.0.1"])
    monkeypatch.setattr(http_client, "is_explicit_pytest_runtime", lambda: False)
    for name in ("TESTING", "TEST_MODE", "TLDW_TEST_MODE"):
        monkeypatch.delenv(name, raising=False)
