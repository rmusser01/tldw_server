from __future__ import annotations

import importlib
import sys

import pytest

pytestmark = pytest.mark.unit


def _import_shutdown_authnz_scheduler():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_authnz_scheduler", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_authnz_scheduler")


@pytest.mark.asyncio
async def test_maybe_stop_authnz_scheduler_stops_and_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_authnz = _import_shutdown_authnz_scheduler()
    calls: list[str] = []

    async def _fake_stop():
        calls.append("stop")

    monkeypatch.setattr(shutdown_authnz, "_stop_authnz_scheduler_service", _fake_stop)

    started = await shutdown_authnz.maybe_stop_authnz_scheduler(
        authnz_scheduler_started=True,
        coordinated_legacy_component_names=set(),
        guard_exceptions=(RuntimeError,),
    )

    assert started is False
    assert calls == ["stop"]


@pytest.mark.asyncio
async def test_maybe_stop_authnz_scheduler_skips_when_component_is_coordinated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_authnz = _import_shutdown_authnz_scheduler()
    called = False

    async def _fake_stop():
        nonlocal called
        called = True

    monkeypatch.setattr(shutdown_authnz, "_stop_authnz_scheduler_service", _fake_stop)

    started = await shutdown_authnz.maybe_stop_authnz_scheduler(
        authnz_scheduler_started=True,
        coordinated_legacy_component_names={"authnz_scheduler"},
        guard_exceptions=(RuntimeError,),
    )

    assert started is True
    assert called is False


@pytest.mark.asyncio
async def test_maybe_stop_authnz_scheduler_skips_when_background_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_authnz = _import_shutdown_authnz_scheduler()
    called = False

    async def _fake_stop():
        nonlocal called
        called = True

    monkeypatch.setattr(shutdown_authnz, "_stop_authnz_scheduler_service", _fake_stop)

    started = await shutdown_authnz.maybe_stop_authnz_scheduler(
        authnz_scheduler_started=True,
        coordinated_legacy_component_names=set(),
        stopped_background_worker_names={"authnz_scheduler"},
        guard_exceptions=(RuntimeError,),
    )

    assert started is False
    assert called is False


@pytest.mark.asyncio
async def test_maybe_stop_authnz_scheduler_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_authnz = _import_shutdown_authnz_scheduler()
    debug_messages: list[str] = []

    async def _failing_stop():
        raise RuntimeError("boom")

    monkeypatch.setattr(shutdown_authnz, "_stop_authnz_scheduler_service", _failing_stop)

    started = await shutdown_authnz.maybe_stop_authnz_scheduler(
        authnz_scheduler_started=True,
        coordinated_legacy_component_names=set(),
        guard_exceptions=(RuntimeError,),
        debug_log=debug_messages.append,
    )

    assert started is True
    assert debug_messages == ["AuthNZ scheduler shutdown skipped: boom"]
