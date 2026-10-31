"""app.main installs one middleware stack, whatever the import timing (TASK-13950.19)."""

from __future__ import annotations

import pytest

from tldw_Server_API.tests.helpers.app_main_state import app_main_isolated, reload_app_main


def _stack(app) -> list[str]:
    return [m.cls.__name__ for m in app.user_middleware]


def test_the_suite_app_installs_security_headers() -> None:
    from tldw_Server_API.app import main as app_main

    assert "SecurityHeadersMiddleware" in _stack(app_main.app)


def test_middleware_stack_does_not_depend_on_import_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    with app_main_isolated():
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        collection_time = _stack(reload_app_main().app)
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/x.py::test_y (call)")
        test_time = _stack(reload_app_main().app)
    assert collection_time == test_time
    assert {"SecurityHeadersMiddleware", "HTTPMetricsMiddleware", "UsageLoggingMiddleware"} <= set(test_time)
