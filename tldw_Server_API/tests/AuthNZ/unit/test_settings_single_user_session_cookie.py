"""Validation tests for the single-user session cookie name setting."""

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.core.AuthNZ.settings import Settings


API_KEY = "test-single-user-key-1234567890"
DEFAULT_COOKIE_NAME = "tldw_single_user_session"


def make_settings(**kwargs) -> Settings:
    """Build isolated single-user settings for cookie-name validation tests."""
    return Settings(
        _env_file=None,
        AUTH_MODE="single_user",
        SINGLE_USER_API_KEY=API_KEY,
        **kwargs,
    )


def test_session_cookie_name_defaults_when_environment_is_absent(monkeypatch) -> None:
    monkeypatch.delenv("SINGLE_USER_SESSION_COOKIE_NAME", raising=False)

    settings = make_settings()

    assert settings.SINGLE_USER_SESSION_COOKIE_NAME == DEFAULT_COOKIE_NAME


@pytest.mark.parametrize("cookie_name", ["custom_session", "CSRF_TOKEN"])
def test_session_cookie_name_accepts_valid_custom_names(cookie_name: str) -> None:
    settings = make_settings(SINGLE_USER_SESSION_COOKIE_NAME=cookie_name)

    assert settings.SINGLE_USER_SESSION_COOKIE_NAME == cookie_name


@pytest.mark.parametrize(
    "cookie_name",
    [
        "",
        "csrf_token",
        "__Host-session",
        "__Http-session",
        "__secure-session",
        "invalid name",
        "session=value",
        "session;name",
        "/session",
    ],
)
def test_session_cookie_name_rejects_unsafe_values(cookie_name: str) -> None:
    with pytest.raises(ValidationError, match="SINGLE_USER_SESSION_COOKIE_NAME"):
        make_settings(SINGLE_USER_SESSION_COOKIE_NAME=cookie_name)


def test_session_cookie_name_rejects_explicit_empty_environment(monkeypatch) -> None:
    monkeypatch.setenv("SINGLE_USER_SESSION_COOKIE_NAME", "")

    with pytest.raises(ValidationError, match="SINGLE_USER_SESSION_COOKIE_NAME"):
        make_settings()
