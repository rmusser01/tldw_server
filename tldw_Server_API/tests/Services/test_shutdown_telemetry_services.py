from __future__ import annotations

import importlib
import sys

import pytest

pytestmark = pytest.mark.unit


def _import_shutdown_telemetry_services():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_telemetry_services", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_telemetry_services")


@pytest.mark.asyncio
async def test_shutdown_telemetry_services_stops_telemetry_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_telemetry = _import_shutdown_telemetry_services()
    calls: list[str] = []

    def _record_shutdown_telemetry() -> None:
        calls.append("telemetry")

    monkeypatch.setattr(
        shutdown_telemetry,
        "_shutdown_telemetry_service",
        _record_shutdown_telemetry,
    )

    await shutdown_telemetry.shutdown_telemetry_services(
        import_exceptions=(LookupError,),
    )

    assert calls == ["telemetry"]
    assert not hasattr(shutdown_telemetry, "_maybe_stop_authnz_scheduler_service")


@pytest.mark.asyncio
async def test_shutdown_telemetry_services_handles_import_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_telemetry = _import_shutdown_telemetry_services()
    called = False

    def _failing_shutdown_telemetry() -> None:
        nonlocal called
        called = True
        raise LookupError("boom")

    monkeypatch.setattr(
        shutdown_telemetry,
        "_shutdown_telemetry_service",
        _failing_shutdown_telemetry,
    )

    await shutdown_telemetry.shutdown_telemetry_services(
        import_exceptions=(LookupError,),
    )

    assert called is True
