from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_startup_content_backend_validation():
    sys.modules.pop("tldw_Server_API.app.services.startup_content_backend_validation", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_content_backend_validation")


class _FakeLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.debug_messages: list[str] = []
        self.exception_messages: list[str] = []

    def info(self, message: str) -> None:
        self.info_messages.append(str(message))

    def debug(self, message: str) -> None:
        self.debug_messages.append(str(message))

    def exception(self, message: str) -> None:
        self.exception_messages.append(str(message))


def test_validate_startup_content_backend_logs_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_validation = _import_startup_content_backend_validation()
    logger = _FakeLogger()
    calls: list[str] = []

    monkeypatch.setattr(
        startup_validation,
        "_validate_postgres_content_backend",
        lambda: calls.append("validated"),
    )

    startup_validation.validate_startup_content_backend(logger=logger)

    assert calls == ["validated"]
    assert logger.info_messages == ["App Startup: PostgreSQL content backend validated"]
    assert logger.debug_messages == []
    assert logger.exception_messages == []


def test_validate_startup_content_backend_logs_and_reraises_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_validation = _import_startup_content_backend_validation()
    logger = _FakeLogger()

    def _raise_runtime_error() -> None:
        raise RuntimeError("content boom")

    monkeypatch.setattr(
        startup_validation,
        "_validate_postgres_content_backend",
        _raise_runtime_error,
    )

    with pytest.raises(RuntimeError, match="content boom"):
        startup_validation.validate_startup_content_backend(logger=logger)

    assert logger.exception_messages == ["Startup aborted: content boom"]
    assert logger.info_messages == []
    assert logger.debug_messages == []


def test_validate_startup_content_backend_logs_debug_on_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_validation = _import_startup_content_backend_validation()
    logger = _FakeLogger()

    def _raise_import_error() -> None:
        raise ImportError("no backend")

    monkeypatch.setattr(
        startup_validation,
        "_validate_postgres_content_backend",
        _raise_import_error,
    )

    startup_validation.validate_startup_content_backend(logger=logger)

    assert logger.debug_messages == [
        "Content backend validation skipped (import error): no backend"
    ]
    assert logger.info_messages == []
    assert logger.exception_messages == []
