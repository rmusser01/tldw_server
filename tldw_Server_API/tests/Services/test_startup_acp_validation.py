from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_startup_acp_validation():
    sys.modules.pop("tldw_Server_API.app.services.startup_acp_validation", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_acp_validation")


class _FakeLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.debug_messages: list[str] = []

    def info(self, message: str, *args: object) -> None:
        self.info_messages.append(message.format(*args) if args else str(message))

    def warning(self, message: str, *args: object) -> None:
        self.warning_messages.append(message.format(*args) if args else str(message))

    def debug(self, message: str, *args: object) -> None:
        self.debug_messages.append(message.format(*args) if args else str(message))


def test_validate_startup_acp_configuration_skips_when_route_disabled() -> None:
    startup_acp = _import_startup_acp_validation()
    logger = _FakeLogger()
    calls: list[str] = []

    startup_acp.validate_startup_acp_configuration(
        route_enabled=lambda *_args, **_kwargs: False,
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
        load_acp_runner_config=lambda: calls.append("load"),
        validate_acp_config=lambda _cfg: calls.append("validate"),
    )

    assert calls == []
    assert logger.info_messages == []
    assert logger.warning_messages == []
    assert logger.debug_messages == []


def test_validate_startup_acp_configuration_logs_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_acp = _import_startup_acp_validation()
    logger = _FakeLogger()

    monkeypatch.setattr(startup_acp, "_load_acp_runner_config", lambda: {"runner": "cfg"})
    monkeypatch.setattr(startup_acp, "_validate_acp_config", lambda cfg: ["warn one", "warn two"])

    startup_acp.validate_startup_acp_configuration(
        route_enabled=lambda name, default_stable=False: name == "acp" and default_stable is False,
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert logger.warning_messages == [
        "ACP config: warn one",
        "ACP config: warn two",
    ]
    assert logger.info_messages == []
    assert logger.debug_messages == []


def test_validate_startup_acp_configuration_logs_success_when_no_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_acp = _import_startup_acp_validation()
    logger = _FakeLogger()

    monkeypatch.setattr(startup_acp, "_load_acp_runner_config", lambda: {"runner": "cfg"})
    monkeypatch.setattr(startup_acp, "_validate_acp_config", lambda cfg: [])

    startup_acp.validate_startup_acp_configuration(
        route_enabled=lambda name, default_stable=False: name == "acp" and default_stable is False,
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert logger.info_messages == ["App Startup: ACP runner configuration validated"]
    assert logger.warning_messages == []
    assert logger.debug_messages == []


def test_validate_startup_acp_configuration_logs_debug_on_guard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_acp = _import_startup_acp_validation()
    logger = _FakeLogger()

    def _raise_load_error():
        raise RuntimeError("acp boom")

    monkeypatch.setattr(startup_acp, "_load_acp_runner_config", _raise_load_error)

    startup_acp.validate_startup_acp_configuration(
        route_enabled=lambda *_args, **_kwargs: True,
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert logger.debug_messages == ["App Startup: ACP config validation skipped: acp boom"]
    assert logger.info_messages == []
    assert logger.warning_messages == []
