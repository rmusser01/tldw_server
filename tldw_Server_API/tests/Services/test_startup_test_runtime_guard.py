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


# --- TASK-13950.6: the real guard, not a stub -------------------------------------

_PRODUCTION_KEYS = (
    "tldw_production",
    "TLDW_PRODUCTION",
    "ENVIRONMENT",
    "APP_ENV",
    "DEPLOYMENT_ENV",
    "FASTAPI_ENV",
    "TLDW_ENV",
    "ENV",
)
_GUARDED_SEAMS = {
    "TEST_MODE": "1",
    "TESTING": "true",
    "TLDW_TEST_MODE": "1",
    "TLDW_ALLOW_TEST_FALLBACK_KEYS": "1",
    "SINGLE_USER_TEST_API_KEY": "test-api-key-for-e2e-testing-12345",
    "E2E_TEST_BASE_URL": "http://127.0.0.1:8000",
    "MINIMAL_TEST_APP": "1",
    "RG_TEST_FORCE_STUB_RATE": "0.5",
    "RG_TEST_DISABLE_ACCEPT_WINDOW": "1",
    "JWT_SECRET_TEST_KEY": "test-secret-jwt-key-please-change-1234567890",
    "ENABLE_ADMIN_E2E_TEST_MODE": "true",
    "EVALUATIONS_TEST_DB_PATH": "/tmp/evals.db",
    "JOBS_TEST_NOW_EPOCH": "1700000000",
}


@pytest.fixture()
def no_seams(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    import os

    for key in _PRODUCTION_KEYS + tuple(_GUARDED_SEAMS):
        monkeypatch.delenv(key, raising=False)
    for key in list(os.environ):
        if key.startswith("RG_TEST_"):
            monkeypatch.delenv(key, raising=False)
    return monkeypatch


@pytest.mark.parametrize("seam", sorted(_GUARDED_SEAMS))
def test_guard_refuses_an_armed_seam_in_production_even_under_pytest(
    no_seams: pytest.MonkeyPatch, seam: str
) -> None:
    from tldw_Server_API.app.core.testing import validate_test_runtime_flags

    no_seams.setenv("tldw_production", "1")
    no_seams.setenv("PYTEST_CURRENT_TEST", "tests::startup_guard")
    no_seams.setenv(seam, _GUARDED_SEAMS[seam])

    with pytest.raises(RuntimeError, match=f"production-like runtime.*{seam}"):
        validate_test_runtime_flags()


@pytest.mark.parametrize("key", _PRODUCTION_KEYS)
def test_guard_refuses_test_fallback_keys_for_every_production_marker(
    no_seams: pytest.MonkeyPatch, key: str
) -> None:
    from tldw_Server_API.app.core.testing import validate_test_runtime_flags

    no_seams.setenv(key, "production")
    no_seams.setenv("TLDW_ALLOW_TEST_FALLBACK_KEYS", "1")

    with pytest.raises(RuntimeError, match=f"production-like runtime \\({key}\\)"):
        validate_test_runtime_flags()


@pytest.mark.parametrize("value", ["0", "false", ""])
def test_guard_ignores_a_seam_switched_off_in_production(no_seams: pytest.MonkeyPatch, value: str) -> None:
    from tldw_Server_API.app.core.testing import validate_test_runtime_flags

    no_seams.setenv("APP_ENV", "production")
    no_seams.setenv("TLDW_ALLOW_TEST_FALLBACK_KEYS", value)
    no_seams.setenv("MINIMAL_TEST_APP", value)

    validate_test_runtime_flags()


@pytest.mark.parametrize(
    "seam",
    ["SINGLE_USER_TEST_API_KEY", "E2E_TEST_BASE_URL", "MINIMAL_TEST_APP", "RG_TEST_FORCE_STUB_RATE"],
)
def test_guard_allows_e2e_seams_outside_production_without_pytest(no_seams: pytest.MonkeyPatch, seam: str) -> None:
    """The e2e gates start uvicorn outside pytest with these set (e2e-required, frontend-required)."""
    from tldw_Server_API.app.core.testing import validate_test_runtime_flags

    no_seams.delenv("PYTEST_CURRENT_TEST", raising=False)
    no_seams.setenv(seam, _GUARDED_SEAMS[seam])

    validate_test_runtime_flags()
