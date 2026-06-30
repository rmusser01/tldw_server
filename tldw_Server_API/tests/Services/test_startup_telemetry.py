from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def _import_startup_telemetry():
    sys.modules.pop("tldw_Server_API.app.services.startup_telemetry", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_telemetry")


class _FakeLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.debug_messages: list[str] = []
        self.exception_messages: list[str] = []

    def info(self, message: str) -> None:
        self.info_messages.append(str(message))

    def warning(self, message: str) -> None:
        self.warning_messages.append(str(message))

    def debug(self, message: str) -> None:
        self.debug_messages.append(str(message))

    def exception(self, message: str) -> None:
        self.exception_messages.append(str(message))


def test_initialize_startup_telemetry_logs_otel_success_and_fastapi_instrumentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = _import_startup_telemetry()
    logger = _FakeLogger()
    app = object()
    manager = SimpleNamespace(config=SimpleNamespace(service_name="tldw-server"))
    observed: dict[str, object] = {}

    monkeypatch.setattr(telemetry, "_initialize_telemetry", lambda: manager)
    monkeypatch.setattr(telemetry, "_otel_available", lambda: True)

    def _fake_instrument_fastapi_app(app_arg, telemetry_manager_arg):
        observed["app"] = app_arg
        observed["telemetry_manager"] = telemetry_manager_arg
        return True

    monkeypatch.setattr(
        telemetry,
        "_instrument_fastapi_app",
        _fake_instrument_fastapi_app,
    )

    result = telemetry.initialize_startup_telemetry(
        app=app,
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert result is manager
    assert observed == {"app": app, "telemetry_manager": manager}
    assert logger.info_messages == [
        "App Startup: Initializing telemetry and metrics...",
        "App Startup: OpenTelemetry initialized for service: tldw-server",
        "App Startup: FastAPI instrumentation enabled",
    ]
    assert logger.warning_messages == []
    assert logger.debug_messages == []
    assert logger.exception_messages == []


def test_initialize_startup_telemetry_warns_when_otel_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = _import_startup_telemetry()
    logger = _FakeLogger()
    manager = SimpleNamespace(config=SimpleNamespace(service_name="fallback"))

    monkeypatch.setattr(telemetry, "_initialize_telemetry", lambda: manager)
    monkeypatch.setattr(telemetry, "_otel_available", lambda: False)
    monkeypatch.setattr(telemetry, "_instrument_fastapi_app", lambda app_arg, telemetry_manager_arg: False)

    result = telemetry.initialize_startup_telemetry(
        app=object(),
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert result is manager
    assert logger.info_messages == ["App Startup: Initializing telemetry and metrics..."]
    assert logger.warning_messages == ["App Startup: OpenTelemetry not available, using fallback metrics"]
    assert logger.debug_messages == []
    assert logger.exception_messages == []


def test_initialize_startup_telemetry_logs_debug_when_fastapi_instrumentation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = _import_startup_telemetry()
    logger = _FakeLogger()
    manager = SimpleNamespace(config=SimpleNamespace(service_name="tldw-server"))

    monkeypatch.setattr(telemetry, "_initialize_telemetry", lambda: manager)
    monkeypatch.setattr(telemetry, "_otel_available", lambda: True)

    def _raise_instrument_error(app_arg, telemetry_manager_arg):
        del app_arg, telemetry_manager_arg
        raise RuntimeError("instrument boom")

    monkeypatch.setattr(telemetry, "_instrument_fastapi_app", _raise_instrument_error)

    result = telemetry.initialize_startup_telemetry(
        app=object(),
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert result is manager
    assert logger.debug_messages == ["App Startup: FastAPI instrumentation skipped: instrument boom"]
    assert logger.exception_messages == []


def test_initialize_startup_telemetry_logs_exception_when_telemetry_init_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = _import_startup_telemetry()
    logger = _FakeLogger()

    def _raise_init_error():
        raise RuntimeError("telemetry boom")

    monkeypatch.setattr(telemetry, "_initialize_telemetry", _raise_init_error)

    result = telemetry.initialize_startup_telemetry(
        app=object(),
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert result is None
    assert logger.info_messages == ["App Startup: Initializing telemetry and metrics..."]
    assert logger.exception_messages == ["App Startup: Failed to initialize telemetry"]
