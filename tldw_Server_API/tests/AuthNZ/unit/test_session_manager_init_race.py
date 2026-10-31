"""Regression tests for the session-manager / MCP server lazy-init races (TASK-13950.4)."""

import asyncio
import threading
import time

import pytest

from tldw_Server_API.app.core.AuthNZ import session_manager as sm
from tldw_Server_API.app.core.MCP_unified import server as mcp_server


@pytest.fixture
def fresh_state(monkeypatch):
    monkeypatch.setattr(sm, "_session_manager", None)
    monkeypatch.setattr(sm, "_session_manager_lock", asyncio.Lock(), raising=False)
    yield


@pytest.mark.asyncio
async def test_callers_never_receive_uninitialized_manager(fresh_state, monkeypatch):
    calls = 0

    async def slow_init(self):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        self._initialized = True

    monkeypatch.setattr(sm.SessionManager, "initialize", slow_init)

    async def grab():
        m = await sm.get_session_manager()
        return m._initialized

    seen = await asyncio.gather(*(grab() for _ in range(50)))
    assert all(seen)
    assert calls == 1


@pytest.mark.asyncio
async def test_failed_initialize_publishes_nothing(fresh_state, monkeypatch):
    attempts = 0

    async def flaky_init(self):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("boom")
        self._initialized = True

    monkeypatch.setattr(sm.SessionManager, "initialize", flaky_init)

    with pytest.raises(RuntimeError):
        await sm.get_session_manager()
    assert sm._session_manager is None

    m = await sm.get_session_manager()
    assert m._initialized and attempts == 2


@pytest.mark.asyncio
async def test_reset_recreates_lock(fresh_state):
    old = sm._session_manager_lock
    await sm.reset_session_manager()
    assert sm._session_manager_lock is not old


def test_get_mcp_server_single_instance_across_threads(monkeypatch):
    monkeypatch.setattr(mcp_server, "_server", None)
    built = []

    class SlowServer:
        def __init__(self):
            time.sleep(0.05)
            built.append(self)

    monkeypatch.setattr(mcp_server, "MCPServer", SlowServer)
    barrier = threading.Barrier(8)
    results = []

    def worker():
        barrier.wait()
        results.append(mcp_server.get_mcp_server())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(built) == 1
    assert len({id(r) for r in results}) == 1
