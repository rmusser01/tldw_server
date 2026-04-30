"""Tests for CheckpointConsumer and checkpoint-based rollback."""
from __future__ import annotations

import asyncio
import builtins
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.Agent_Client_Protocol.event_bus import SessionEventBus
from tldw_Server_API.app.core.Agent_Client_Protocol.events import (
    AgentEvent,
    AgentEventKind,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.consumers.checkpoint_consumer import (
    CheckpointConsumer,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_event(
    kind: AgentEventKind = AgentEventKind.FILE_CHANGE,
    session_id: str = "sess-cp",
) -> AgentEvent:
    return AgentEvent(session_id=session_id, kind=kind, payload={"file": "main.py"})


def _make_sandbox_service(
    snapshot_id: str = "snap-001",
    *,
    side_effect: Exception | None = None,
) -> MagicMock:
    """Return a mock SandboxService with a configurable create_snapshot."""
    svc = MagicMock()
    if side_effect is not None:
        svc.create_snapshot.side_effect = side_effect
    else:
        svc.create_snapshot.return_value = {"snapshot_id": snapshot_id}
    svc.restore_snapshot.return_value = True
    return svc


async def _wait_for(predicate, *, timeout: float = 2.0, interval: float = 0.02):
    """Poll *predicate* until it returns True or *timeout* elapses."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"Timed out waiting for predicate after {timeout}s")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_checkpoint_created_on_file_change():
    """A FILE_CHANGE event should trigger create_snapshot."""
    bus = SessionEventBus(session_id="sess-cp")
    svc = _make_sandbox_service(snapshot_id="snap-100")
    consumer = CheckpointConsumer(sandbox_service=svc, session_id="sess-cp")
    await consumer.start(bus)

    await bus.publish(_make_event(AgentEventKind.FILE_CHANGE))

    await _wait_for(lambda: svc.create_snapshot.call_count >= 1)
    await consumer.stop()

    svc.create_snapshot.assert_called_once_with("sess-cp")
    checkpoints = consumer.get_checkpoints()
    assert len(checkpoints) == 1
    assert list(checkpoints.values()) == ["snap-100"]


@pytest.mark.asyncio
async def test_checkpoint_skipped_on_active_run():
    """When create_snapshot raises (e.g. active run), the consumer continues."""
    bus = SessionEventBus(session_id="sess-cp")
    svc = _make_sandbox_service(side_effect=RuntimeError("session_has_active_runs"))
    consumer = CheckpointConsumer(sandbox_service=svc, session_id="sess-cp")
    await consumer.start(bus)

    await bus.publish(_make_event(AgentEventKind.FILE_CHANGE))
    await bus.publish(_make_event(AgentEventKind.FILE_CHANGE))

    await _wait_for(lambda: svc.create_snapshot.call_count >= 2)
    await consumer.stop()

    # No checkpoints should be recorded -- both calls raised
    assert consumer.get_checkpoints() == {}
    # Consumer should still be alive (not crashed)
    assert svc.create_snapshot.call_count == 2


@pytest.mark.asyncio
async def test_get_nearest_checkpoint():
    """get_nearest_checkpoint should return the closest checkpoint at or before target."""
    bus = SessionEventBus(session_id="sess-cp")

    call_count = 0

    def _create_snap(sid):
        nonlocal call_count
        call_count += 1
        return {"snapshot_id": f"snap-{call_count}"}

    svc = MagicMock()
    svc.create_snapshot.side_effect = _create_snap

    consumer = CheckpointConsumer(sandbox_service=svc, session_id="sess-cp")
    await consumer.start(bus)

    # Publish 3 file-change events -> sequences 1, 2, 3
    for _ in range(3):
        await bus.publish(_make_event(AgentEventKind.FILE_CHANGE))

    await _wait_for(lambda: len(consumer.get_checkpoints()) >= 3)
    await consumer.stop()

    checkpoints = consumer.get_checkpoints()
    assert len(checkpoints) == 3

    # Find nearest to sequence 2
    result = consumer.get_nearest_checkpoint(2)
    assert result is not None
    seq, sid = result
    assert seq == 2
    assert sid == "snap-2"

    # Find nearest to sequence 5 (beyond all checkpoints) -> should return seq 3
    result = consumer.get_nearest_checkpoint(5)
    assert result is not None
    assert result[0] == 3

    # Find nearest to sequence 0 (before all checkpoints) -> None
    result = consumer.get_nearest_checkpoint(0)
    assert result is None


@pytest.mark.asyncio
async def test_max_checkpoints_eviction():
    """Oldest checkpoint should be evicted when max_checkpoints is exceeded."""
    bus = SessionEventBus(session_id="sess-cp")

    call_count = 0

    def _create_snap(sid):
        nonlocal call_count
        call_count += 1
        return {"snapshot_id": f"snap-{call_count}"}

    svc = MagicMock()
    svc.create_snapshot.side_effect = _create_snap

    consumer = CheckpointConsumer(
        sandbox_service=svc,
        session_id="sess-cp",
        max_checkpoints=3,
    )
    await consumer.start(bus)

    # Publish 5 file-change events
    for _ in range(5):
        await bus.publish(_make_event(AgentEventKind.FILE_CHANGE))

    await _wait_for(lambda: svc.create_snapshot.call_count >= 5)
    await consumer.stop()

    checkpoints = consumer.get_checkpoints()
    # Only the last 3 should remain
    assert len(checkpoints) == 3
    sequences = sorted(checkpoints.keys())
    # Sequences 1 and 2 should have been evicted
    assert min(sequences) == 3


@pytest.mark.asyncio
async def test_non_file_events_ignored():
    """Events other than FILE_CHANGE should not trigger create_snapshot."""
    bus = SessionEventBus(session_id="sess-cp")
    svc = _make_sandbox_service()
    consumer = CheckpointConsumer(sandbox_service=svc, session_id="sess-cp")
    await consumer.start(bus)

    # Publish events of various non-file kinds
    for kind in (
        AgentEventKind.THINKING,
        AgentEventKind.TOOL_CALL,
        AgentEventKind.TOOL_RESULT,
        AgentEventKind.COMPLETION,
        AgentEventKind.HEARTBEAT,
        AgentEventKind.STATUS_CHANGE,
    ):
        await bus.publish(AgentEvent(session_id="sess-cp", kind=kind, payload={}))

    # Give consumer time to process all events
    await asyncio.sleep(0.2)
    await consumer.stop()

    svc.create_snapshot.assert_not_called()
    assert consumer.get_checkpoints() == {}


@pytest.mark.asyncio
async def test_stop_unsubscribes():
    """Stopping the consumer should unsubscribe from the bus."""
    bus = SessionEventBus(session_id="sess-cp")
    svc = _make_sandbox_service()
    consumer = CheckpointConsumer(sandbox_service=svc, session_id="sess-cp")
    await consumer.start(bus)

    consumer_id = consumer.consumer_id
    assert consumer_id in bus._subscribers

    await consumer.stop()

    assert consumer_id not in bus._subscribers
    # Task should be cleaned up
    assert consumer._task is None
    assert consumer._bus is None


@pytest.mark.asyncio
async def test_ensure_checkpoint_consumer_sanitizes_sandbox_service_errors(monkeypatch):
    """Sandbox service setup failures should not expose import/config details."""
    from tldw_Server_API.app.api.v1.endpoints import agent_client_protocol as acp_mod

    with acp_mod._CHECKPOINT_CONSUMERS_LOCK:
        acp_mod._CHECKPOINT_CONSUMERS.clear()

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tldw_Server_API.app.core.Sandbox.service":
            raise ImportError("sandbox backend exploded")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(HTTPException) as exc_info:
        await acp_mod._ensure_checkpoint_consumer("session-redacted")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Sandbox service unavailable"


@pytest.mark.asyncio
async def test_acp_session_rollback_sanitizes_restore_errors(monkeypatch):
    """Sandbox restore failures should not expose backend exception details."""
    from tldw_Server_API.app.api.v1.endpoints import agent_client_protocol as acp_mod

    class FakeUser:
        id = 1

    class FailingSandboxService:
        def restore_snapshot(self, _session_id, _snapshot_id):
            raise RuntimeError("sandbox restore backend exploded")

    async def fake_runner_client():
        return object()

    async def fake_require_access(_client, *, session_id, user_id):
        return None

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tldw_Server_API.app.core.Sandbox.service":
            return SimpleNamespace(SandboxService=FailingSandboxService)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(acp_mod, "get_runner_client", fake_runner_client)
    monkeypatch.setattr(acp_mod, "_require_session_access", fake_require_access)
    monkeypatch.setattr(acp_mod, "_acp_enforce_control_rate_limit", lambda **_kwargs: None)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(HTTPException) as exc_info:
        await acp_mod.acp_session_rollback(
            "session-redacted",
            acp_mod.ACPRollbackRequest(to_snapshot_id="snapshot-redacted"),
            user=FakeUser(),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Rollback failed"
