"""Validation tests for the per-user live-session cap setting."""

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.core.AuthNZ.settings import Settings


def make_settings(**kwargs) -> Settings:
    """Build isolated single-user settings for session-cap validation tests."""
    return Settings(
        _env_file=None,
        AUTH_MODE="single_user",
        SINGLE_USER_API_KEY="test-single-user-key-1234567890",
        **kwargs,
    )


def test_max_sessions_per_user_defaults_to_four(monkeypatch) -> None:
    monkeypatch.delenv("MAX_SESSIONS_PER_USER", raising=False)
    assert make_settings().MAX_SESSIONS_PER_USER == 4


@pytest.mark.parametrize("value", [0, -1])
def test_max_sessions_per_user_below_one_is_rejected(value: int) -> None:
    with pytest.raises(ValidationError):
        make_settings(MAX_SESSIONS_PER_USER=value)
