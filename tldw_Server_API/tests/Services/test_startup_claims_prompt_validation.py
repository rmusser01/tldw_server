from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def _import_startup_claims_prompt_validation():
    sys.modules.pop("tldw_Server_API.app.services.startup_claims_prompt_validation", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_claims_prompt_validation")


class _FakeLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.debug_messages: list[str] = []
        self.exception_messages: list[str] = []

    def info(self, message: str, *args: object) -> None:
        self.info_messages.append(message.format(*args) if args else str(message))

    def warning(self, message: str, *args: object) -> None:
        self.warning_messages.append(message.format(*args) if args else str(message))

    def debug(self, message: str, *args: object) -> None:
        self.debug_messages.append(message.format(*args) if args else str(message))

    def exception(self, message: str, *args: object) -> None:
        self.exception_messages.append(message.format(*args) if args else str(message))


def test_validate_startup_claims_prompt_validation_logs_warning_for_issues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_claims = _import_startup_claims_prompt_validation()
    logger = _FakeLogger()
    report = SimpleNamespace(issues=["one", "two"], mode="warning", strict=False)

    monkeypatch.setattr(startup_claims, "_get_claims_settings", lambda: object())
    monkeypatch.setattr(startup_claims, "_validate_claims_prompt_preflight", lambda _settings: report)
    monkeypatch.setattr(startup_claims, "_claims_prompt_report_has_issues", lambda _report: True)

    startup_claims.validate_startup_claims_prompt_validation(
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert logger.warning_messages == [
        "App Startup: Claims prompt validation found 2 issue(s) (mode=warning, strict=False)"
    ]
    assert logger.info_messages == []
    assert logger.debug_messages == []
    assert logger.exception_messages == []


def test_validate_startup_claims_prompt_validation_logs_info_when_off_or_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_claims = _import_startup_claims_prompt_validation()
    logger = _FakeLogger()
    report = SimpleNamespace(issues=["one"], mode="off", strict=True)

    monkeypatch.setattr(startup_claims, "_get_claims_settings", lambda: object())
    monkeypatch.setattr(startup_claims, "_validate_claims_prompt_preflight", lambda _settings: report)
    monkeypatch.setattr(startup_claims, "_claims_prompt_report_has_issues", lambda _report: True)

    startup_claims.validate_startup_claims_prompt_validation(
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert logger.info_messages == [
        "App Startup: Claims prompt validation completed (mode=off, strict=True)"
    ]
    assert logger.warning_messages == []
    assert logger.debug_messages == []
    assert logger.exception_messages == []


def test_validate_startup_claims_prompt_validation_logs_and_reraises_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_claims = _import_startup_claims_prompt_validation()
    logger = _FakeLogger()

    class _FakeClaimsPromptValidationError(Exception):
        pass

    def _raise_validation_error(_settings: object) -> None:
        raise _FakeClaimsPromptValidationError("bad claims prompt")

    monkeypatch.setattr(startup_claims, "_get_claims_prompt_validation_error", lambda: _FakeClaimsPromptValidationError)
    monkeypatch.setattr(startup_claims, "_get_claims_settings", lambda: object())
    monkeypatch.setattr(startup_claims, "_validate_claims_prompt_preflight", _raise_validation_error)

    with pytest.raises(_FakeClaimsPromptValidationError, match="bad claims prompt"):
        startup_claims.validate_startup_claims_prompt_validation(
            logger=logger,
            startup_guard_exceptions=(RuntimeError,),
        )

    assert logger.exception_messages == [
        "Startup aborted due to claims prompt validation error"
    ]
    assert logger.info_messages == []
    assert logger.warning_messages == []
    assert logger.debug_messages == []


def test_validate_startup_claims_prompt_validation_logs_debug_on_guard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_claims = _import_startup_claims_prompt_validation()
    logger = _FakeLogger()

    monkeypatch.setattr(startup_claims, "_get_claims_settings", lambda: object())

    def _raise_guard_error(_settings: object) -> None:
        raise RuntimeError("claims guard boom")

    monkeypatch.setattr(startup_claims, "_validate_claims_prompt_preflight", _raise_guard_error)

    startup_claims.validate_startup_claims_prompt_validation(
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert logger.debug_messages == [
        "App Startup: Claims prompt validation skipped/failed: claims guard boom"
    ]
    assert logger.info_messages == []
    assert logger.warning_messages == []
    assert logger.exception_messages == []


def test_validate_startup_claims_prompt_validation_guards_validation_error_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_claims = _import_startup_claims_prompt_validation()
    logger = _FakeLogger()

    def _raise_import_error() -> type[BaseException]:
        raise ImportError("claims validation import failed")

    monkeypatch.setattr(
        startup_claims,
        "_get_claims_prompt_validation_error",
        _raise_import_error,
    )

    startup_claims.validate_startup_claims_prompt_validation(
        logger=logger,
        startup_guard_exceptions=(ImportError, RuntimeError),
    )

    assert logger.debug_messages == [
        "App Startup: Claims prompt validation skipped/failed: claims validation import failed"
    ]
    assert logger.info_messages == []
    assert logger.warning_messages == []
    assert logger.exception_messages == []
