from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_startup_sentry():
    sys.modules.pop("tldw_Server_API.app.services.startup_sentry", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_sentry")


class _FakeLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.debug_messages: list[str] = []

    def info(self, message: str) -> None:
        self.info_messages.append(str(message))

    def warning(self, message: str, *args: object) -> None:
        if args:
            self.warning_messages.append(str(message).format(*args))
        else:
            self.warning_messages.append(str(message))

    def debug(self, message: str, *args: object) -> None:
        if args:
            self.debug_messages.append(str(message).format(*args))
        else:
            self.debug_messages.append(str(message))


def test_initialize_startup_sentry_skips_when_dsn_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentry = _import_startup_sentry()
    logger = _FakeLogger()
    observed = {"initialized": False}

    monkeypatch.setattr(sentry, "_getenv", lambda name, default="": "")
    monkeypatch.setattr(
        sentry,
        "_init_sentry",
        lambda **kwargs: observed.__setitem__("initialized", True),
    )

    sentry.initialize_startup_sentry(
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert observed["initialized"] is False
    assert logger.info_messages == []
    assert logger.warning_messages == []


def test_initialize_startup_sentry_initializes_sentry_when_dsn_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentry = _import_startup_sentry()
    logger = _FakeLogger()
    observed: dict[str, object] = {}

    values = {
        "SENTRY_DSN": "https://dsn.example/123",
        "SENTRY_TRACES_SAMPLE_RATE": "0.25",
        "DEPLOYMENT_ENV": "production",
        "OTEL_SERVICE_VERSION": "2.3.4",
    }

    monkeypatch.setattr(
        sentry,
        "_getenv",
        lambda name, default="": values.get(name, default),
    )
    monkeypatch.setattr(
        sentry,
        "_init_sentry",
        lambda **kwargs: observed.update(kwargs),
    )

    sentry.initialize_startup_sentry(
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert observed == {
        "dsn": "https://dsn.example/123",
        "traces_sample_rate": 0.25,
        "environment": "production",
        "release": "2.3.4",
        "send_default_pii": False,
    }
    assert logger.info_messages == ["App Startup: Sentry error tracking initialized"]
    assert logger.warning_messages == []


def test_initialize_startup_sentry_logs_warning_on_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentry = _import_startup_sentry()
    logger = _FakeLogger()

    monkeypatch.setattr(
        sentry,
        "_getenv",
        lambda name, default="": "https://dsn.example/123" if name == "SENTRY_DSN" else default,
    )

    def _raise_import_error(**kwargs) -> None:
        del kwargs
        raise ImportError("missing sentry")

    monkeypatch.setattr(sentry, "_init_sentry", _raise_import_error)

    sentry.initialize_startup_sentry(
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert logger.info_messages == []
    assert logger.warning_messages == [
        "App Startup: Sentry initialization failed"
    ]
