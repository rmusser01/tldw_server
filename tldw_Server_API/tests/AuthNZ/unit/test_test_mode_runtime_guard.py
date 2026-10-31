import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.testing import (
    is_test_mode,
    is_production_like_env,
    validate_test_runtime_flags,
)


def _clear_test_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("ALLOW_TEST_MODE_IN_PRODUCTION", raising=False)
    monkeypatch.delenv("tldw_production", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("DEPLOYMENT_ENV", raising=False)
    monkeypatch.delenv("FASTAPI_ENV", raising=False)
    monkeypatch.delenv("TLDW_ENV", raising=False)


def test_validate_test_runtime_flags_allows_without_test_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_test_flags(monkeypatch)
    validate_test_runtime_flags()


@pytest.mark.parametrize("flag_name", ["TEST_MODE", "TESTING", "TLDW_TEST_MODE"])
def test_validate_test_runtime_flags_rejects_outside_explicit_pytest_runtime(
    monkeypatch: pytest.MonkeyPatch,
    flag_name: str,
) -> None:
    _clear_test_flags(monkeypatch)
    monkeypatch.setenv(flag_name, "1")

    with pytest.raises(RuntimeError, match="missing PYTEST_CURRENT_TEST"):
        validate_test_runtime_flags()


def test_validate_test_runtime_flags_allows_when_explicit_pytest_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_test_flags(monkeypatch)
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests::authnz::runtime_guard")
    validate_test_runtime_flags()


def test_validate_test_runtime_flags_rejects_even_with_override_without_pytest_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_test_flags(monkeypatch)
    monkeypatch.setenv("tldw_production", "1")
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("ALLOW_TEST_MODE_IN_PRODUCTION", "1")

    with pytest.raises(RuntimeError, match="production-like runtime"):
        validate_test_runtime_flags()


def test_is_test_mode_accepts_single_letter_y(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_test_flags(monkeypatch)
    monkeypatch.setenv("TEST_MODE", "y")
    assert is_test_mode() is True


def test_is_test_mode_checks_test_mode_and_tldw_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_test_flags(monkeypatch)
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "y")
    assert is_test_mode() is True


def test_is_production_like_env_detects_multiple_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_test_flags(monkeypatch)
    assert is_production_like_env() is False
    monkeypatch.setenv("tldw_production", "true")
    assert is_production_like_env() is True
    monkeypatch.delenv("tldw_production", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert is_production_like_env() is True
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert is_production_like_env() is False


def test_app_startup_hard_fails_when_test_mode_enabled_without_explicit_pytest_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_test_flags(monkeypatch)
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.main import app

    with pytest.raises(RuntimeError, match="missing PYTEST_CURRENT_TEST"):
        with TestClient(app):
            pass


def test_app_startup_guard_allows_explicit_pytest_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_test_flags(monkeypatch)
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests::authnz::startup_guard")
    validate_test_runtime_flags()


def test_app_startup_fails_fast_when_lazy_evaluations_warmup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_test_flags(monkeypatch)

    from tldw_Server_API.app import main as main_mod
    from tldw_Server_API.app.core.Evaluations import connection_pool as eval_pool

    monkeypatch.setattr(main_mod, "_TEST_MODE", False, raising=True)
    monkeypatch.setattr(main_mod, "route_enabled", lambda key, **_kwargs: key == "evaluations")

    def _boom():
        raise RuntimeError("forced warmup failure")

    monkeypatch.setattr(eval_pool, "get_connection_manager", _boom)

    with pytest.raises(RuntimeError, match="forced warmup failure|lazy subsystem warmup failed"):
        with TestClient(main_mod.app):
            pass


def test_main_loguru_reconfig_override_accepts_single_letter_y(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_test_flags(monkeypatch)
    monkeypatch.setenv("TLDW_ALLOW_LOGURU_RECONFIG", "y")

    from tldw_Server_API.app import main as main_mod

    assert main_mod._caller_allowed_for_loguru_config() is True
