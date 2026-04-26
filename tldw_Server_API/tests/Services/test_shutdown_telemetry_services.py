from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_telemetry_services():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_telemetry_services", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_telemetry_services")


@pytest.mark.asyncio
async def test_shutdown_telemetry_services_stops_authnz_then_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_telemetry = _import_shutdown_telemetry_services()
    calls: list[str] = []

    async def _record_stop_authnz_scheduler(
        *,
        authnz_scheduler_started,
        coordinated_legacy_component_names,
        guard_exceptions,
    ) -> bool:
        assert authnz_scheduler_started is True
        assert coordinated_legacy_component_names == {"usage_aggregator"}
        assert guard_exceptions == (LookupError,)
        calls.append("authnz")
        return False

    def _record_shutdown_telemetry() -> None:
        calls.append("telemetry")

    monkeypatch.setattr(
        shutdown_telemetry,
        "_maybe_stop_authnz_scheduler_service",
        _record_stop_authnz_scheduler,
    )
    monkeypatch.setattr(
        shutdown_telemetry,
        "_shutdown_telemetry_service",
        _record_shutdown_telemetry,
    )

    started = await shutdown_telemetry.shutdown_telemetry_services(
        authnz_scheduler_started=True,
        coordinated_legacy_component_names={"usage_aggregator"},
        import_exceptions=(LookupError,),
    )

    assert started is False
    assert calls == ["authnz", "telemetry"]


@pytest.mark.asyncio
async def test_shutdown_telemetry_services_handles_import_exception_before_state_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_telemetry = _import_shutdown_telemetry_services()

    async def _failing_stop_authnz_scheduler(
        *,
        authnz_scheduler_started,
        coordinated_legacy_component_names,
        guard_exceptions,
    ) -> bool:
        del authnz_scheduler_started, coordinated_legacy_component_names, guard_exceptions
        raise LookupError("boom")

    monkeypatch.setattr(
        shutdown_telemetry,
        "_maybe_stop_authnz_scheduler_service",
        _failing_stop_authnz_scheduler,
    )

    started = await shutdown_telemetry.shutdown_telemetry_services(
        authnz_scheduler_started=True,
        coordinated_legacy_component_names=set(),
        import_exceptions=(LookupError,),
    )

    assert started is True


@pytest.mark.asyncio
async def test_shutdown_telemetry_services_handles_import_exception_after_state_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_telemetry = _import_shutdown_telemetry_services()

    async def _record_stop_authnz_scheduler(
        *,
        authnz_scheduler_started,
        coordinated_legacy_component_names,
        guard_exceptions,
    ) -> bool:
        del authnz_scheduler_started, coordinated_legacy_component_names, guard_exceptions
        return False

    def _failing_shutdown_telemetry() -> None:
        raise LookupError("boom")

    monkeypatch.setattr(
        shutdown_telemetry,
        "_maybe_stop_authnz_scheduler_service",
        _record_stop_authnz_scheduler,
    )
    monkeypatch.setattr(
        shutdown_telemetry,
        "_shutdown_telemetry_service",
        _failing_shutdown_telemetry,
    )

    started = await shutdown_telemetry.shutdown_telemetry_services(
        authnz_scheduler_started=True,
        coordinated_legacy_component_names=set(),
        import_exceptions=(LookupError,),
    )

    assert started is False
