from __future__ import annotations

import configparser
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import config_admin

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debug_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.error_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.exception_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debug_records.append((message, args, kwargs))

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.error_records.append((message, args, kwargs))

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.exception_records.append((message, args, kwargs))


_SENSITIVE_LOG_MARKERS = (
    "config backend leaked",
    "/private/config.txt",
    "secret-section",
)


def _assert_sanitized_error_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.exception_records == []
    assert logger_stub.error_records == [(expected_message, (), {})]
    rendered_records = repr(logger_stub.error_records)
    for marker in _SENSITIVE_LOG_MARKERS:
        assert marker not in rendered_records


@pytest.mark.asyncio
async def test_get_effective_config_sanitizes_resolution_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    def _raise_config_root() -> None:
        raise FileNotFoundError("config root leaked /private/config.txt")

    monkeypatch.setattr(config_admin, "logger", logger_stub)
    monkeypatch.setattr(config_admin, "resolve_config_root", _raise_config_root)

    with pytest.raises(HTTPException) as exc_info:
        await config_admin.get_effective_config(sections=None, include_defaults=True)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to resolve effective configuration"
    assert logger_stub.debug_records == [("Effective config resolution failed", (), {})]


def test_build_config_txt_values_sanitizes_section_read_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _LoggerStub()

    class _Parser:
        def sections(self) -> list[str]:
            raise configparser.Error("config backend leaked /private/config.txt")

    monkeypatch.setattr(config_admin, "logger", logger_stub)
    monkeypatch.setattr(config_admin, "load_comprehensive_config", lambda: _Parser())

    assert config_admin._build_config_txt_values() == {}

    _assert_sanitized_error_log(logger_stub, "Error reading config sections")


def test_build_config_txt_values_sanitizes_item_read_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _LoggerStub()

    class _Parser:
        def sections(self) -> list[str]:
            return ["secret-section"]

        def items(self, section: str) -> list[tuple[str, str]]:
            assert section == "secret-section"
            raise configparser.Error("config backend leaked /private/config.txt")

    monkeypatch.setattr(config_admin, "logger", logger_stub)
    monkeypatch.setattr(config_admin, "load_comprehensive_config", lambda: _Parser())

    assert config_admin._build_config_txt_values() == {}

    _assert_sanitized_error_log(logger_stub, "Error reading config items")
