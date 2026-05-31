"""Sanitizer coverage for app package initializer fallbacks."""

from types import SimpleNamespace

import tldw_Server_API.app as app_init


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []

    def debug(self, message: str) -> None:
        self.debugs.append(message)


class _BrokenEnviron:
    def __contains__(self, key: object) -> bool:
        raise RuntimeError("pytest env check failed at /private/app-init?token=secret")


def test_under_pytest_check_failure_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(app_init, "logger", logger_stub)
    monkeypatch.setattr(app_init, "os", SimpleNamespace(environ=_BrokenEnviron()))

    assert app_init._under_pytest() is False

    assert logger_stub.debugs == ["app.__init__._under_pytest check failed"]
    rendered = "\n".join(logger_stub.debugs)
    assert "/private/app-init" not in rendered
    assert "secret" not in rendered
