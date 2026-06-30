from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_startup_evaluations_warmup():
    sys.modules.pop("tldw_Server_API.app.services.startup_evaluations_warmup", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_evaluations_warmup")


class _FakeLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.exception_messages: list[str] = []

    def info(self, message: str) -> None:
        self.info_messages.append(str(message))

    def exception(self, message: str) -> None:
        self.exception_messages.append(str(message))


def test_warm_lazy_evaluations_managers_skips_in_test_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warmup = _import_startup_evaluations_warmup()
    logger = _FakeLogger()
    observed = {"connection": False, "webhook": False}

    monkeypatch.setattr(warmup, "_warm_evaluations_connection_manager", lambda: observed.__setitem__("connection", True))
    monkeypatch.setattr(warmup, "_warm_evaluations_webhook_manager", lambda: observed.__setitem__("webhook", True))

    warmup.warm_lazy_evaluations_managers(
        route_enabled=lambda route_name: True,
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
        test_mode=True,
    )

    assert observed == {"connection": False, "webhook": False}
    assert logger.info_messages == []
    assert logger.exception_messages == []


def test_warm_lazy_evaluations_managers_skips_when_route_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warmup = _import_startup_evaluations_warmup()
    logger = _FakeLogger()
    observed = {"connection": False, "webhook": False}

    monkeypatch.setattr(warmup, "_warm_evaluations_connection_manager", lambda: observed.__setitem__("connection", True))
    monkeypatch.setattr(warmup, "_warm_evaluations_webhook_manager", lambda: observed.__setitem__("webhook", True))

    warmup.warm_lazy_evaluations_managers(
        route_enabled=lambda route_name: False,
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
        test_mode=False,
    )

    assert observed == {"connection": False, "webhook": False}
    assert logger.info_messages == []
    assert logger.exception_messages == []


def test_warm_lazy_evaluations_managers_warms_managers_and_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warmup = _import_startup_evaluations_warmup()
    logger = _FakeLogger()
    calls: list[str] = []

    monkeypatch.setattr(warmup, "_warm_evaluations_connection_manager", lambda: calls.append("connection"))
    monkeypatch.setattr(warmup, "_warm_evaluations_webhook_manager", lambda: calls.append("webhook"))

    warmup.warm_lazy_evaluations_managers(
        route_enabled=lambda route_name: route_name == "evaluations",
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
        test_mode=False,
    )

    assert calls == ["connection", "webhook"]
    assert logger.info_messages == [
        "App Startup: Warmed lazy Evaluations managers (fail-fast enabled)"
    ]
    assert logger.exception_messages == []


def test_warm_lazy_evaluations_managers_logs_and_reraises_guard_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warmup = _import_startup_evaluations_warmup()
    logger = _FakeLogger()

    def _raise_connection_error() -> None:
        raise RuntimeError("eval boom")

    monkeypatch.setattr(warmup, "_warm_evaluations_connection_manager", _raise_connection_error)

    with pytest.raises(RuntimeError, match="eval boom"):
        warmup.warm_lazy_evaluations_managers(
            route_enabled=lambda route_name: True,
            logger=logger,
            startup_guard_exceptions=(RuntimeError,),
            test_mode=False,
        )

    assert logger.exception_messages == [
        "Startup aborted: lazy subsystem warmup failed: eval boom"
    ]
