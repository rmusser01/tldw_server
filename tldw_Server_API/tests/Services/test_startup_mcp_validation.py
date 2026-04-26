from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def _import_startup_mcp_validation():
    sys.modules.pop("tldw_Server_API.app.services.startup_mcp_validation", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_mcp_validation")


class _FakeLogger:
    def __init__(self) -> None:
        self.exception_messages: list[str] = []

    def exception(self, message: str) -> None:
        self.exception_messages.append(str(message))


def test_validate_startup_mcp_configuration_skips_without_config_helpers() -> None:
    startup_mcp = _import_startup_mcp_validation()
    logger = _FakeLogger()

    startup_mcp.validate_startup_mcp_configuration(
        get_mcp_config=None,
        validate_mcp_config=None,
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert logger.exception_messages == []


def test_validate_startup_mcp_configuration_skips_in_debug_mode() -> None:
    startup_mcp = _import_startup_mcp_validation()
    logger = _FakeLogger()
    calls: list[str] = []

    def _get_config() -> object:
        calls.append("get")
        return SimpleNamespace(debug_mode=True)

    def _validate_config() -> bool:
        calls.append("validate")
        return True

    startup_mcp.validate_startup_mcp_configuration(
        get_mcp_config=_get_config,
        validate_mcp_config=_validate_config,
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert calls == ["get"]
    assert logger.exception_messages == []


def test_validate_startup_mcp_configuration_validates_non_debug_mode() -> None:
    startup_mcp = _import_startup_mcp_validation()
    logger = _FakeLogger()
    calls: list[str] = []

    def _get_config() -> object:
        calls.append("get")
        return SimpleNamespace(debug_mode=False)

    def _validate_config() -> bool:
        calls.append("validate")
        return True

    startup_mcp.validate_startup_mcp_configuration(
        get_mcp_config=_get_config,
        validate_mcp_config=_validate_config,
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert calls == ["get", "validate"]
    assert logger.exception_messages == []


def test_validate_startup_mcp_configuration_accepts_mapping_config() -> None:
    startup_mcp = _import_startup_mcp_validation()
    logger = _FakeLogger()
    calls: list[str] = []

    def _get_config() -> object:
        calls.append("get")
        return {"debug_mode": False}

    def _validate_config() -> bool:
        calls.append("validate")
        return True

    startup_mcp.validate_startup_mcp_configuration(
        get_mcp_config=_get_config,
        validate_mcp_config=_validate_config,
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert calls == ["get", "validate"]
    assert logger.exception_messages == []


def test_validate_startup_mcp_configuration_logs_and_reraises_invalid_config() -> None:
    startup_mcp = _import_startup_mcp_validation()
    logger = _FakeLogger()

    def _get_config() -> object:
        return SimpleNamespace(debug_mode=False)

    def _validate_config() -> bool:
        return False

    with pytest.raises(
        RuntimeError,
        match="MCP configuration validation failed; refusing to start in production",
    ):
        startup_mcp.validate_startup_mcp_configuration(
            get_mcp_config=_get_config,
            validate_mcp_config=_validate_config,
            logger=logger,
            startup_guard_exceptions=(RuntimeError,),
        )

    assert logger.exception_messages == [
        "Startup aborted due to insecure MCP configuration: MCP configuration validation failed; refusing to start in production"
    ]


def test_validate_startup_mcp_configuration_logs_and_reraises_guard_failures() -> None:
    startup_mcp = _import_startup_mcp_validation()
    logger = _FakeLogger()

    def _get_config() -> object:
        raise RuntimeError("mcp boom")

    with pytest.raises(RuntimeError, match="mcp boom"):
        startup_mcp.validate_startup_mcp_configuration(
            get_mcp_config=_get_config,
            validate_mcp_config=lambda: True,
            logger=logger,
            startup_guard_exceptions=(RuntimeError,),
        )

    assert logger.exception_messages == [
        "Startup aborted due to insecure MCP configuration: mcp boom"
    ]
