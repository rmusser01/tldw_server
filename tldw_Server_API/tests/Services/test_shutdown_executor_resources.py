from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_executor_resources():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_executor_resources", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_executor_resources")


@pytest.mark.asyncio
async def test_shutdown_executor_resources_stops_registered_and_default_executor_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_executors = _import_shutdown_executor_resources()
    calls: list[str] = []

    async def _record_registered_executors(*, wait, cancel_futures):
        assert wait is True
        assert cancel_futures is True
        calls.append("registered")

    class _Loop:
        async def shutdown_default_executor(self) -> None:
            calls.append("default")

    monkeypatch.setattr(
        shutdown_executors,
        "_shutdown_registered_executors_service",
        _record_registered_executors,
    )
    monkeypatch.setattr(
        shutdown_executors.asyncio,
        "get_running_loop",
        lambda: _Loop(),
    )

    await shutdown_executors.shutdown_executor_resources(
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(LookupError,),
    )

    assert calls == ["registered", "default"]


@pytest.mark.asyncio
async def test_shutdown_executor_resources_handles_import_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_executors = _import_shutdown_executor_resources()
    calls: list[str] = []

    async def _failing_registered_executors(*, wait, cancel_futures):
        del wait, cancel_futures
        raise LookupError("boom")

    class _Loop:
        async def shutdown_default_executor(self) -> None:
            calls.append("default")

    monkeypatch.setattr(
        shutdown_executors,
        "_shutdown_registered_executors_service",
        _failing_registered_executors,
    )
    monkeypatch.setattr(
        shutdown_executors.asyncio,
        "get_running_loop",
        lambda: _Loop(),
    )

    await shutdown_executors.shutdown_executor_resources(
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(LookupError,),
    )

    assert calls == ["default"]


@pytest.mark.asyncio
async def test_shutdown_default_executor_skips_when_loop_has_no_shutdown_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_executors = _import_shutdown_executor_resources()

    class _Loop:
        pass

    monkeypatch.setattr(
        shutdown_executors.asyncio,
        "get_running_loop",
        lambda: _Loop(),
    )

    await shutdown_executors._shutdown_default_executor(
        guard_exceptions=(RuntimeError,),
    )


@pytest.mark.asyncio
async def test_shutdown_default_executor_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_executors = _import_shutdown_executor_resources()

    class _Loop:
        async def shutdown_default_executor(self) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        shutdown_executors.asyncio,
        "get_running_loop",
        lambda: _Loop(),
    )

    await shutdown_executors._shutdown_default_executor(
        guard_exceptions=(RuntimeError,),
    )
