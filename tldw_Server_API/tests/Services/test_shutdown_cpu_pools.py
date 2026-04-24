from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_cpu_pools():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_cpu_pools", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_cpu_pools")


@pytest.mark.asyncio
async def test_shutdown_cpu_pools_invokes_cleanup_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_cpu = _import_shutdown_cpu_pools()
    calls: list[str] = []

    def _record_cleanup_cpu_pools():
        calls.append("cleanup")

    monkeypatch.setattr(
        shutdown_cpu,
        "_cleanup_cpu_pools_service",
        _record_cleanup_cpu_pools,
    )

    await shutdown_cpu.shutdown_cpu_pools(
        guard_exceptions=(RuntimeError,),
    )

    assert calls == ["cleanup"]


@pytest.mark.asyncio
async def test_shutdown_cpu_pools_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_cpu = _import_shutdown_cpu_pools()

    def _failing_cleanup_cpu_pools():
        raise RuntimeError("boom")

    monkeypatch.setattr(
        shutdown_cpu,
        "_cleanup_cpu_pools_service",
        _failing_cleanup_cpu_pools,
    )

    await shutdown_cpu.shutdown_cpu_pools(
        guard_exceptions=(RuntimeError,),
    )
