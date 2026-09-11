from __future__ import annotations

import pytest

from tldw_Server_API.app.core.AuthNZ.settings import Settings


def _settings(**overrides) -> Settings:
    return Settings(
        _env_file=None,
        AUTH_MODE="single_user",
        DATABASE_URL="sqlite:///:memory:",
        **overrides,
    )


def test_openai_credential_lock_backend_defaults_to_db(monkeypatch):
    monkeypatch.delenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", raising=False)

    assert _settings().OPENAI_OAUTH_REFRESH_LOCK_BACKEND == "db"


@pytest.mark.parametrize("raw_value", [None, "", "   ", "unsupported"])
def test_invalid_openai_credential_lock_backend_fails_safe_to_db(
    monkeypatch,
    raw_value,
):
    monkeypatch.delenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", raising=False)

    assert (
        _settings(
            OPENAI_OAUTH_REFRESH_LOCK_BACKEND=raw_value,
        ).OPENAI_OAUTH_REFRESH_LOCK_BACKEND
        == "db"
    )


def test_openai_credential_lock_backend_normalizes_explicit_redis(monkeypatch):
    monkeypatch.delenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", raising=False)

    assert (
        _settings(
            OPENAI_OAUTH_REFRESH_LOCK_BACKEND=" Redis ",
        ).OPENAI_OAUTH_REFRESH_LOCK_BACKEND
        == "redis"
    )
