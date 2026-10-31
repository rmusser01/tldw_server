"""One production detector: every caller agrees on every key and value (TASK-13950.6).

core/testing.runtime_environment() is the single source. Each entry in CALLERS is a
site that used to carry its own copy with its own key set; the matrix proves they now
all answer the same way. Inline call sites (auth_deps, crypto_utils, main, MCP, RAG,
startup reporting) can only route through the shared function because
tests/lint/test_production_detector_single_source.py bans the key reads elsewhere.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Callable, cast

import pytest

pytestmark = pytest.mark.unit

KEYS = (
    "tldw_production",
    "TLDW_PRODUCTION",
    "ENVIRONMENT",
    "APP_ENV",
    "DEPLOYMENT_ENV",
    "FASTAPI_ENV",
    "TLDW_ENV",
    "ENV",
)
VALUES = ("prod", "production", "live")
_POSTGRES_URL = "postgresql://tldw_user:pw@localhost:5432/tldw_users"
_JWT_TEMPLATE = "CHANGE_ME_TO_SECURE_RANDOM_KEY_MIN_32_CHARS"


def _raises(build: Callable[[], Any]) -> bool:
    """True when ``build`` refuses with a production-specific ValueError."""
    try:
        build()
    except ValueError as exc:
        return "production" in str(exc)
    return False


def _settings(**kwargs: Any) -> Any:
    from tldw_Server_API.app.core.AuthNZ.settings import Settings

    return Settings(_env_file=None, **kwargs)


def _runtime() -> bool:
    from tldw_Server_API.app.core.testing import runtime_environment

    return runtime_environment().is_production


def _testing() -> bool:
    from tldw_Server_API.app.core.testing import is_production_like_env

    return is_production_like_env()


def _config() -> bool:
    from tldw_Server_API.app.core.config import is_production_environment

    return is_production_environment()


def _rg_startup_guard() -> bool:
    from tldw_Server_API.app.core.AuthNZ.rg_startup_guard import is_production_like_env

    return is_production_like_env()


def _user_db_handling() -> bool:
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import _is_production_like_env

    return _is_production_like_env()


def _program_evaluator() -> bool:
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.program_evaluator import ProgramEvaluator

    return ProgramEvaluator.is_production_like_environment()


def _admin_webhooks() -> bool:
    from tldw_Server_API.app.core.Admin_Webhooks.config import is_production_environment_mapping

    return is_production_environment_mapping(dict(os.environ))


def _connectors_require_encryption() -> bool:
    from tldw_Server_API.app.core.External_Sources.connectors_service import (
        _connector_secret_encryption_required,
    )

    return _connector_secret_encryption_required()


def _embeddings_enforce_rg() -> bool:
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
        _should_enforce_rg_in_production,
    )

    return _should_enforce_rg_in_production()


def _rag_retriever_production_mode() -> bool:
    from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import BaseRetriever

    class _Retriever(BaseRetriever):
        async def retrieve(self, *args: Any, **kwargs: Any) -> list:
            return []

        async def get_metadata(self, doc_id: str) -> dict[str, Any]:
            return {}

    return bool(_Retriever(None, db_adapter=object())._production_mode)


def _crypto_refuses_test_fallback() -> bool:
    from tldw_Server_API.app.core.AuthNZ.crypto_utils import derive_hmac_key_candidates

    return _raises(lambda: derive_hmac_key_candidates(cast(Any, SimpleNamespace(AUTH_MODE="multi_user"))))


def _settings_refuses_weak_single_user_key() -> bool:
    return _raises(lambda: _settings(AUTH_MODE="single_user", SINGLE_USER_API_KEY="test-api-key-12345"))


def _settings_refuses_jwt_template() -> bool:
    return _raises(
        lambda: _settings(AUTH_MODE="multi_user", JWT_SECRET_KEY=_JWT_TEMPLATE, DATABASE_URL=_POSTGRES_URL)
    )


CALLERS: dict[str, Callable[[], bool]] = {
    "testing.runtime_environment": _runtime,
    "testing.is_production_like_env": _testing,
    "config.is_production_environment": _config,
    "rg_startup_guard.is_production_like_env": _rg_startup_guard,
    "User_DB_Handling._is_production_like_env": _user_db_handling,
    "ProgramEvaluator.is_production_like_environment": _program_evaluator,
    "Admin_Webhooks.is_production_environment_mapping": _admin_webhooks,
    "connectors_service._connector_secret_encryption_required": _connectors_require_encryption,
    "database_retrievers.BaseRetriever._production_mode": _rag_retriever_production_mode,
    "crypto_utils refuses the test fallback key": _crypto_refuses_test_fallback,
    "settings hard-fails a weak single-user key": _settings_refuses_weak_single_user_key,
    "settings hard-fails the default JWT template": _settings_refuses_jwt_template,
}
# Callers that exempt pytest and TEST_MODE: asked with both cleared.
OUTSIDE_TEST_CALLERS: dict[str, Callable[[], bool]] = {
    "Embeddings_Create._should_enforce_rg_in_production": _embeddings_enforce_rg,
}


@pytest.fixture()
def neutral_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """No production marker and no caller-specific override that would mask the detector."""
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    for key in (
        "CONNECTORS_REQUIRE_TOKEN_ENCRYPTION",
        "TLDW_ALLOW_LEGACY_SINGLE_USER_KEY",
        "SINGLE_USER_API_KEY",
        "API_KEY",
        "E2E_TEST_BASE_URL",
        "TLDW_TEST_MODE",
        "TESTING",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AUTH_MODE", "single_user")
    return monkeypatch


def _answers(env: pytest.MonkeyPatch) -> dict[str, bool]:
    env.setenv("TEST_MODE", "1")  # arms the crypto fallback so its production refusal is observable
    answers = {name: bool(caller()) for name, caller in CALLERS.items()}
    env.delenv("TEST_MODE")
    env.delenv("PYTEST_CURRENT_TEST", raising=False)
    answers.update({name: bool(caller()) for name, caller in OUTSIDE_TEST_CALLERS.items()})
    return answers


def _all(value: bool) -> dict[str, bool]:
    return dict.fromkeys((*CALLERS, *OUTSIDE_TEST_CALLERS), value)


def test_key_set_is_the_documented_one() -> None:
    from tldw_Server_API.app.core.testing import PRODUCTION_ENV_KEYS

    assert tuple(PRODUCTION_ENV_KEYS) == KEYS


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("key", KEYS)
def test_every_caller_agrees_the_runtime_is_production(neutral_env: pytest.MonkeyPatch, key: str, value: str) -> None:
    neutral_env.setenv(key, value.upper() if key == "ENV" else value)

    assert _answers(neutral_env) == _all(True)


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("value", ("development", "staging", "test", ""))
def test_every_caller_agrees_the_runtime_is_not_production(
    neutral_env: pytest.MonkeyPatch, key: str, value: str
) -> None:
    neutral_env.setenv(key, value)

    assert _answers(neutral_env) == _all(False)


@pytest.mark.parametrize("key", ("tldw_production", "TLDW_PRODUCTION"))
@pytest.mark.parametrize("value", ("1", "true", "yes", "on"))
def test_production_flag_keys_also_accept_truthy_values(
    neutral_env: pytest.MonkeyPatch, key: str, value: str
) -> None:
    from tldw_Server_API.app.core.testing import runtime_environment

    neutral_env.setenv(key, value)

    assert runtime_environment().is_production is True
    assert runtime_environment().source_key == key


def test_runtime_environment_sees_env_changes_without_a_cache_reset(neutral_env: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.testing import runtime_environment

    assert runtime_environment().is_production is False
    assert runtime_environment().source_key is None
    neutral_env.setenv("APP_ENV", "production")
    assert runtime_environment().source_key == "APP_ENV"
    neutral_env.delenv("APP_ENV")
    assert runtime_environment().is_production is False


@pytest.mark.parametrize(
    "build",
    [
        lambda: _settings(AUTH_MODE="single_user", SINGLE_USER_API_KEY="test-api-key-12345"),
        lambda: _settings(AUTH_MODE="multi_user", JWT_SECRET_KEY=_JWT_TEMPLATE, DATABASE_URL=_POSTGRES_URL),
        lambda: _settings(AUTH_MODE="multi_user", JWT_SECRET_KEY="x" * 40, DATABASE_URL="sqlite:///./users.db"),
    ],
    ids=["single-user-key", "jwt-template", "multi-user-sqlite"],
)
def test_settings_hard_fail_names_the_production_marker(
    neutral_env: pytest.MonkeyPatch, build: Callable[[], Any]
) -> None:
    neutral_env.setenv("DEPLOYMENT_ENV", "production")

    with pytest.raises(ValueError, match=r"In a production-like runtime \(DEPLOYMENT_ENV\), "):
        build()


def test_settings_refuses_the_test_jwt_fallback_in_production_under_pytest(neutral_env: pytest.MonkeyPatch) -> None:
    neutral_env.setenv("tldw_production", "1")
    neutral_env.setenv("PYTEST_CURRENT_TEST", "x")

    with pytest.raises(ValueError, match="JWT_SECRET_KEY"):
        _settings(AUTH_MODE="multi_user", JWT_SECRET_KEY=None, DATABASE_URL=_POSTGRES_URL)


def test_settings_keeps_the_test_jwt_fallback_outside_production(neutral_env: pytest.MonkeyPatch) -> None:
    neutral_env.setenv("PYTEST_CURRENT_TEST", "x")

    settings = _settings(AUTH_MODE="multi_user", JWT_SECRET_KEY=None, DATABASE_URL=_POSTGRES_URL)

    assert settings.JWT_SECRET_KEY


def test_settings_refuses_the_test_api_key_fallback_in_production_under_pytest(
    neutral_env: pytest.MonkeyPatch,
) -> None:
    neutral_env.setenv("APP_ENV", "production")
    neutral_env.setenv("PYTEST_CURRENT_TEST", "x")
    neutral_env.setenv("SINGLE_USER_TEST_API_KEY", "a-deterministic-test-key-long-enough")

    with pytest.raises(ValueError, match="SINGLE_USER_API_KEY"):
        _settings(AUTH_MODE="single_user", SINGLE_USER_API_KEY=None)


def test_user_db_handling_treats_an_empty_pytest_marker_as_unset(neutral_env: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.AuthNZ import User_DB_Handling

    neutral_env.delenv("TEST_MODE", raising=False)
    neutral_env.setattr(User_DB_Handling, "get_settings", lambda: SimpleNamespace(TEST_MODE=False))
    neutral_env.setenv("PYTEST_CURRENT_TEST", "")

    assert User_DB_Handling._is_test_context() is False
    assert User_DB_Handling._is_strict_test_bypass_context() is False

    neutral_env.setenv("PYTEST_CURRENT_TEST", "tests::x")
    assert User_DB_Handling._is_test_context() is True
    assert User_DB_Handling._is_strict_test_bypass_context() is True
