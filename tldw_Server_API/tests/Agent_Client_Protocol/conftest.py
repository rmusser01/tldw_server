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
