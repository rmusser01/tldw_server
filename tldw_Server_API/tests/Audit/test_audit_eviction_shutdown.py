import asyncio
import threading
import time

import pytest

from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps
from tldw_Server_API.app.core.Audit.unified_audit_service import UnifiedAuditService


@pytest.mark.asyncio
async def test_eviction_stop_delays_on_recent_use(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setattr(audit_deps, "EVICTION_GRACE_SECONDS", 0.05)

    service = UnifiedAuditService(
        db_path=str(tmp_path / "audit.db"),
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=10,
        flush_interval=60.0,
    )
    await service.initialize(start_background_tasks=False)

    stopped = asyncio.Event()
    original_stop = service.stop

    async def _stop():
        stopped.set()
        await original_stop()

    monkeypatch.setattr(service, "stop", _stop)

    service._last_used_ts = time.monotonic()
    audit_deps._schedule_service_stop(123, service, "test")

    evicted_ts = getattr(service, "_tldw_evicted_at", None)
    assert evicted_ts is not None
    service._last_used_ts = evicted_ts + 1.0

    await asyncio.sleep(0.06)
    assert not stopped.is_set()

    await asyncio.wait_for(stopped.wait(), timeout=0.5)


class _LoopOwnedDummyService:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._owner_loop = loop
        self._tldw_stop_scheduled = False
        self._last_used_ts = time.monotonic() - 10.0
        self._tldw_evicted_at = None
        self.stopped = threading.Event()

    @property
    def owner_loop(self):
        return self._owner_loop

    async def stop(self) -> None:
        self.stopped.set()


def _start_owner_loop_thread() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    owner_loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _runner() -> None:
        asyncio.set_event_loop(owner_loop)
        owner_loop.call_soon(ready.set)
        owner_loop.run_forever()
        owner_loop.close()

    thread = threading.Thread(target=_runner, name="audit-owner-loop", daemon=True)
    thread.start()
    assert ready.wait(timeout=1.0)
    return owner_loop, thread


@pytest.mark.asyncio
async def test_schedule_stop_tracks_owner_loop_future_until_drained(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setattr(audit_deps, "EVICTION_GRACE_SECONDS", 0.0)
    audit_deps._services_stopping.clear()
    audit_deps._scheduled_stop_futures.clear()

    owner_loop, owner_thread = _start_owner_loop_thread()
    service = _LoopOwnedDummyService(owner_loop)

    try:
        audit_deps._schedule_service_stop(321, service, "test-owner-loop")
        await audit_deps._drain_scheduled_audit_stops(timeout=1.0)
        assert service.stopped.wait(timeout=0.2)
        assert not audit_deps._scheduled_stop_futures
    finally:
        owner_loop.call_soon_threadsafe(owner_loop.stop)
        owner_thread.join(timeout=1.0)



@pytest.mark.asyncio
async def test_schedule_stop_skips_nonrunning_owner_loop(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setattr(audit_deps, "EVICTION_GRACE_SECONDS", 0.05)

    service = UnifiedAuditService(
        db_path=str(tmp_path / "audit.db"),
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=10,
        flush_interval=60.0,
    )
    await service.initialize(start_background_tasks=False)

    owner_loop = asyncio.new_event_loop()
    service._owner_loop = owner_loop

    stopped = asyncio.Event()
    original_stop = service.stop

    async def _stop():
        stopped.set()
        await original_stop()

    monkeypatch.setattr(service, "stop", _stop)

    def _boom(*_args, **_kwargs):
        raise AssertionError("run_coroutine_threadsafe should not be used for non-running loop")

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _boom)

    audit_deps._schedule_service_stop(456, service, "test-nonrunning-loop")
    await asyncio.wait_for(stopped.wait(), timeout=1.0)

    owner_loop.close()
