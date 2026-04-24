import asyncio
import concurrent.futures
import time

import pytest

from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as deps
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.exceptions import ServiceInitializationError


class _DummyService:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._tldw_stop_scheduled = False
        self._last_used_ts = time.monotonic() - 10.0
        self._tldw_evicted_at = None
        self.stopped = False

    @property
    def owner_loop(self):
        return self._loop

    async def stop(self) -> None:
        self.stopped = True
        raise LookupError("stop failed")


class _StoppingService:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self.stopped = False

    @property
    def owner_loop(self):
        return self._loop

    async def stop(self) -> None:
        self.stopped = True


class _BlockingStopService:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        started_event: asyncio.Event,
        release_event: asyncio.Event,
    ) -> None:
        self._loop = loop
        self._started_event = started_event
        self._release_event = release_event
        self.stopped = False

    @property
    def owner_loop(self):
        return self._loop

    async def stop(self) -> None:
        self.stopped = True
        self._started_event.set()
        await self._release_event.wait()


class _OwnerLoopStub:
    def is_closed(self) -> bool:
        return False

    def is_running(self) -> bool:
        return True


class _ThreadsafeSignalLoopStub:
    def __init__(self) -> None:
        self.calls = []

    def is_closed(self) -> bool:
        return False

    def call_soon_threadsafe(self, callback, *args) -> None:
        self.calls.append((callback, args))


class _ThreadsafeSignalEventStub:
    def __init__(self) -> None:
        self.set_calls = 0

    def set(self) -> None:
        self.set_calls += 1


@pytest.mark.asyncio
async def test_get_audit_service_for_user_accepts_string_id(monkeypatch):
    sentinel = object()
    called = {}

    async def _fake_get_or_create(user_id: int):
        called["user_id"] = user_id
        return sentinel

    async def _fake_default():
        raise AssertionError("default audit service should not be used for numeric string ids")

    monkeypatch.setattr(deps, "get_or_create_audit_service_for_user_id", _fake_get_or_create)
    monkeypatch.setattr(deps, "get_or_create_default_audit_service", _fake_default)

    user = User(id="42", username="tester", email=None, is_active=True)
    svc = await deps.get_audit_service_for_user(user)

    assert svc is sentinel
    assert called["user_id"] == 42


@pytest.mark.asyncio
async def test_get_audit_service_for_user_uses_raw_non_numeric_id_in_tests(monkeypatch):
    sentinel = object()
    called = {}

    async def _fake_optional(user_id):
        called["user_id"] = user_id
        return sentinel

    monkeypatch.setattr(deps, "get_or_create_audit_service_for_user_id_optional", _fake_optional)

    user = User(id="tenant-alpha", username="tester", email=None, is_active=True)
    svc = await deps.get_audit_service_for_user(user)

    assert svc is sentinel
    assert called["user_id"] == "tenant-alpha"


@pytest.mark.asyncio
async def test_optional_audit_service_rejects_non_numeric_outside_tests(monkeypatch):
    monkeypatch.setattr(deps, "_is_test_context", lambda: False)
    monkeypatch.setattr(deps, "_resolve_audit_storage_mode", lambda: "per_user")

    async def _fake_get_or_create(_key):
        raise AssertionError("Should not create audit service for invalid non-numeric id")

    monkeypatch.setattr(deps, "_get_or_create_audit_service_for_key", _fake_get_or_create)

    with pytest.raises(ServiceInitializationError):
        await deps.get_or_create_audit_service_for_user_id_optional("tenant-alpha")


@pytest.mark.asyncio
async def test_schedule_service_stop_clears_flag_on_failure(monkeypatch):
    deps._services_stopping.clear()
    monkeypatch.setattr(deps, "EVICTION_GRACE_SECONDS", 0.0)

    loop = asyncio.get_running_loop()
    svc = _DummyService(loop)

    deps._schedule_service_stop(1, svc, "test")

    for _ in range(20):
        await asyncio.sleep(0)
        if not getattr(svc, "_tldw_stop_scheduled", True):
            break

    assert svc._tldw_stop_scheduled is False
    assert id(svc) not in deps._services_stopping


@pytest.mark.asyncio
async def test_shutdown_user_audit_service_uses_shared_cache_key(monkeypatch):
    state = deps._LoopState(cache={}, loop=asyncio.get_running_loop())
    svc = _StoppingService(asyncio.get_running_loop())
    state.cache[None] = svc

    monkeypatch.setattr(deps, "_resolve_audit_storage_mode", lambda: "shared")
    monkeypatch.setattr(deps, "_all_loop_states", lambda: [state])

    summary = await deps.shutdown_user_audit_service(123)

    assert isinstance(summary, deps.AuditShutdownSummary)
    assert summary.requested == 1
    assert summary.stopped == 1
    assert summary.error_count == 0
    assert summary.timeout_count == 0
    assert svc.stopped is True
    assert None not in state.cache


@pytest.mark.asyncio
async def test_shutdown_all_audit_services_returns_summary_and_can_raise(monkeypatch):
    loop = asyncio.get_running_loop()

    def _make_state() -> deps._LoopState:
        return deps._LoopState(cache={
            1: _StoppingService(loop),
            2: _DummyService(loop),
        }, loop=loop)

    state = _make_state()
    monkeypatch.setattr(deps, "_all_loop_states", lambda: [state])

    with pytest.raises(deps.AuditShutdownError):
        await deps.shutdown_all_audit_services()

    state = _make_state()
    monkeypatch.setattr(deps, "_all_loop_states", lambda: [state])

    summary = await deps.shutdown_all_audit_services(raise_on_error=False)

    assert isinstance(summary, deps.AuditShutdownSummary)
    assert summary.requested == 2
    assert summary.stopped == 1
    assert summary.error_count == 1
    assert summary.timeout_count == 0
    assert summary.errors
    assert "stop failed" in summary.errors[0]
    assert state.cache == {}

@pytest.mark.asyncio
async def test_shutdown_all_audit_services_runs_stop_fan_out_concurrently(monkeypatch):
    loop = asyncio.get_running_loop()
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release = asyncio.Event()
    state = deps._LoopState(cache={
        1: _BlockingStopService(loop, first_started, release),
        2: _BlockingStopService(loop, second_started, release),
    }, loop=loop)

    monkeypatch.setattr(deps, "_all_loop_states", lambda: [state])

    shutdown_task = asyncio.create_task(deps.shutdown_all_audit_services())

    await asyncio.wait_for(first_started.wait(), timeout=5)
    await asyncio.wait_for(second_started.wait(), timeout=5)
    release.set()

    summary = await asyncio.wait_for(shutdown_task, timeout=5)

    assert isinstance(summary, deps.AuditShutdownSummary)
    assert summary.requested == 2
    assert summary.stopped == 2
    assert summary.error_count == 0
    assert summary.timeout_count == 0
    assert state.cache == {}


@pytest.mark.asyncio
async def test_get_or_create_does_not_recache_service_after_shutdown(monkeypatch):
    state = deps._LoopState(cache={}, loop=asyncio.get_running_loop())
    started = asyncio.Event()
    release = asyncio.Event()

    async def _fake_create(_user_id):
        started.set()
        await release.wait()
        return _StoppingService(asyncio.get_running_loop())

    monkeypatch.setattr(deps, "_resolve_audit_storage_mode", lambda: "per_user")
    monkeypatch.setattr(deps, "_state_for_loop", lambda: state)
    monkeypatch.setattr(deps, "_all_loop_states", lambda: [state])
    monkeypatch.setattr(deps, "_create_audit_service_for_user", _fake_create)

    init_task = asyncio.create_task(deps._get_or_create_audit_service_for_key(123))
    await asyncio.wait_for(started.wait(), timeout=5)

    summary = await deps.shutdown_user_audit_service(123)
    assert summary.requested == 0
    assert state.cache == {}

    release.set()

    with pytest.raises(ServiceInitializationError, match="shutdown"):
        await asyncio.wait_for(init_task, timeout=5)

    assert state.cache == {}


@pytest.mark.asyncio
async def test_shutdown_user_audit_service_signals_cross_loop_waiters_threadsafe(monkeypatch):
    signal_loop = _ThreadsafeSignalLoopStub()
    signal_event = _ThreadsafeSignalEventStub()
    state = deps._LoopState(cache={}, loop=signal_loop)
    state.initializing_events[123] = signal_event

    monkeypatch.setattr(deps, "_resolve_audit_storage_mode", lambda: "per_user")
    monkeypatch.setattr(deps, "_all_loop_states", lambda: [state])

    summary = await deps.shutdown_user_audit_service(123)

    assert summary.requested == 0
    assert signal_event.set_calls == 0
    assert signal_loop.calls == [(signal_event.set, ())]


@pytest.mark.asyncio
async def test_shutdown_all_audit_services_signals_cross_loop_waiters_threadsafe(monkeypatch):
    signal_loop = _ThreadsafeSignalLoopStub()
    signal_event = _ThreadsafeSignalEventStub()
    state = deps._LoopState(cache={}, loop=signal_loop)
    state.initializing_events[123] = signal_event

    monkeypatch.setattr(deps, "_all_loop_states", lambda: [state])

    summary = await deps.shutdown_all_audit_services(raise_on_error=False)

    assert summary.requested == 0
    assert signal_event.set_calls == 0
    assert signal_loop.calls == [(signal_event.set, ())]


@pytest.mark.asyncio
async def test_shutdown_all_audit_services_blocks_new_initializations(monkeypatch):
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = asyncio.Event()
    state = deps._LoopState(cache={
        1: _BlockingStopService(loop, started, release),
    }, loop=loop)

    monkeypatch.setattr(deps, "_resolve_audit_storage_mode", lambda: "per_user")
    monkeypatch.setattr(deps, "_state_for_loop", lambda: state)
    monkeypatch.setattr(deps, "_all_loop_states", lambda: [state])

    shutdown_task = asyncio.create_task(deps.shutdown_all_audit_services(raise_on_error=False))
    await asyncio.wait_for(started.wait(), timeout=5)

    with pytest.raises(ServiceInitializationError, match="global shutdown"):
        await deps._get_or_create_audit_service_for_key(2)

    release.set()
    summary = await asyncio.wait_for(shutdown_task, timeout=5)

    assert summary.requested == 1
    assert summary.stopped == 1


@pytest.mark.asyncio
async def test_stop_audit_service_instance_cancels_cross_loop_future_on_timeout(monkeypatch):
    cancel_called = {"value": False}

    class _CancelableFuture(concurrent.futures.Future):
        def cancel(self) -> bool:
            cancel_called["value"] = True
            return super().cancel()

    class _CrossLoopService:
        owner_loop = _OwnerLoopStub()

        async def stop(self) -> None:
            await asyncio.sleep(60)

    future = _CancelableFuture()

    def _fake_run_coroutine_threadsafe(coro, _loop):
        coro.close()
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _fake_run_coroutine_threadsafe)

    stopped_ok, timeout_hit, error_message, exc = await deps._stop_audit_service_instance(
        _CrossLoopService(),
        label="cross-loop",
        timeout_s=0.01,
    )

    assert stopped_ok is False
    assert timeout_hit is True
    assert error_message is not None and "timed out" in error_message
    assert isinstance(exc, asyncio.TimeoutError)
    assert cancel_called["value"] is True


@pytest.mark.asyncio
async def test_optional_audit_service_accepts_non_numeric_in_tests(monkeypatch):
    sentinel = object()
    called = {}

    async def _fake_get_or_create(key):
        called["key"] = key
        return sentinel

    monkeypatch.setattr(deps, "_get_or_create_audit_service_for_key", _fake_get_or_create)
    monkeypatch.setattr(deps, "_is_test_context", lambda: True)
    monkeypatch.setattr(deps, "_resolve_audit_storage_mode", lambda: "per_user")

    svc = await deps.get_or_create_audit_service_for_user_id_optional("tenant-alpha")

    assert svc is sentinel
    assert called["key"] == "tenant-alpha"


@pytest.mark.asyncio
async def test_optional_audit_service_accepts_non_numeric_in_shared_mode(monkeypatch):
    sentinel = object()
    called = {}

    async def _fake_default():
        called["default"] = True
        return sentinel

    monkeypatch.setattr(deps, "_is_test_context", lambda: False)
    monkeypatch.setattr(deps, "_resolve_audit_storage_mode", lambda: "shared")
    monkeypatch.setattr(deps, "get_or_create_default_audit_service", _fake_default)

    svc = await deps.get_or_create_audit_service_for_user_id_optional("tenant-alpha")

    assert svc is sentinel
    assert called["default"] is True


@pytest.mark.asyncio
async def test_optional_audit_service_routes_system_id_to_default(monkeypatch):
    sentinel = object()
    called = {}

    async def _fake_default():
        called["default"] = True
        return sentinel

    monkeypatch.setattr(deps, "_is_test_context", lambda: False)
    monkeypatch.setattr(deps, "_resolve_audit_storage_mode", lambda: "per_user")
    monkeypatch.setattr(deps, "get_or_create_default_audit_service", _fake_default)

    svc = await deps.get_or_create_audit_service_for_user_id_optional("system")

    assert svc is sentinel
    assert called["default"] is True
