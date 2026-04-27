from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_startup_test_runtime_guard():
    sys.modules.pop("tldw_Server_API.app.services.startup_test_runtime_guard", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_test_runtime_guard")


class _FakeLogger:
    def __init__(self) -> None:
        self.critical_messages: list[str] = []
        self.debug_messages: list[str] = []

    def critical(self, message: str) -> None:
        self.critical_messages.append(str(message))

    def debug(self, message: str) -> None:
        self.debug_messages.append(str(message))


def test_validate_startup_test_runtime_runs_guard_without_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_guard = _import_startup_test_runtime_guard()
    logger = _FakeLogger()
    calls: list[str] = []

    monkeypatch.setattr(
        startup_guard,
        "_validate_test_runtime_flags",
        lambda: calls.append("validated"),
    )

    startup_guard.validate_startup_test_runtime(
        logger=logger,
        import_exceptions=(ImportError,),
    )

    assert calls == ["validated"]
    assert logger.critical_messages == []
    assert logger.debug_messages == []


def test_validate_startup_test_runtime_logs_and_reraises_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_guard = _import_startup_test_runtime_guard()
    logger = _FakeLogger()

    def _raise_runtime_error() -> None:
        raise RuntimeError("unsafe flags")

    monkeypatch.setattr(
        startup_guard,
        "_validate_test_runtime_flags",
        _raise_runtime_error,
    )

    with pytest.raises(RuntimeError, match="unsafe flags"):
        startup_guard.validate_startup_test_runtime(
            logger=logger,
            import_exceptions=(ImportError,),
        )

    assert logger.critical_messages == [
        "Startup aborted due to unsafe test-mode flags: unsafe flags"
    ]
    assert logger.debug_messages == []


def test_validate_startup_test_runtime_logs_debug_on_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_guard = _import_startup_test_runtime_guard()
    logger = _FakeLogger()

    def _raise_import_error() -> None:
        raise ImportError("testing unavailable")

    monkeypatch.setattr(
        startup_guard,
        "_validate_test_runtime_flags",
        _raise_import_error,
    )

    startup_guard.validate_startup_test_runtime(
        logger=logger,
        import_exceptions=(ImportError,),
    )

    assert logger.critical_messages == []
    assert logger.debug_messages == [
        "Test-mode runtime guard import skipped: testing unavailable"
    ]
