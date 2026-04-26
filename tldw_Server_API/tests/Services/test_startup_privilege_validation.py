from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_startup_privilege_validation():
    sys.modules.pop("tldw_Server_API.app.services.startup_privilege_validation", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_privilege_validation")


class _FakeLogger:
    def __init__(self) -> None:
        self.exception_messages: list[str] = []

    def exception(self, message: str) -> None:
        self.exception_messages.append(str(message))


def test_validate_startup_privilege_metadata_delegates_successfully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = _import_startup_privilege_validation()
    logger = _FakeLogger()
    app = object()
    registry = {"media.ingest": []}
    seen: dict[str, object] = {}

    def _fake_validate(app_arg):
        seen["app"] = app_arg
        return registry

    monkeypatch.setattr(
        validation,
        "_validate_privilege_metadata_on_startup",
        _fake_validate,
    )

    result = validation.validate_startup_privilege_metadata(
        app=app,
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert seen["app"] is app
    assert result is registry
    assert logger.exception_messages == []


def test_validate_startup_privilege_metadata_logs_and_reraises_guard_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = _import_startup_privilege_validation()
    logger = _FakeLogger()

    def _raise_validation_error(app_arg):
        del app_arg
        raise RuntimeError("boom")

    monkeypatch.setattr(
        validation,
        "_validate_privilege_metadata_on_startup",
        _raise_validation_error,
    )

    with pytest.raises(RuntimeError, match="boom"):
        validation.validate_startup_privilege_metadata(
            app=object(),
            logger=logger,
            startup_guard_exceptions=(RuntimeError,),
        )

    assert logger.exception_messages == [
        "App Startup: Privilege metadata validation failed: boom"
    ]
