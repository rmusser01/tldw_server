"""Security pins for config keys that move into ``config_sections`` (TASK-13950.21, Task 7).

Run by security-required's TEST_MODE-unset step (C4), which runs this directory.
"""

from __future__ import annotations

import configparser

import pydantic
import pytest

from tldw_Server_API.app.core.AuthNZ import settings as authnz_settings
from tldw_Server_API.app.core.config_sections.auth import load_auth_config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("source", ["env", "config"])
@pytest.mark.parametrize("value", ["multi-user", "MULTI_USER", ""])
def test_invalid_auth_mode_fails_closed_in_every_source(monkeypatch, source, value):
    """A bad AUTH_MODE, in the environment or in config.txt, must stop AuthNZ rather than fall back."""
    parser = configparser.ConfigParser(interpolation=None)
    parser.add_section("AuthNZ")
    if source == "env":
        monkeypatch.setenv("AUTH_MODE", value)
    else:
        monkeypatch.delenv("AUTH_MODE", raising=False)
        parser.set("AuthNZ", "auth_mode", value)
    monkeypatch.setattr(authnz_settings, "load_comprehensive_config", lambda: parser)
    authnz_settings.reset_settings()
    try:
        with pytest.raises(pydantic.ValidationError):
            authnz_settings.get_settings()
    finally:
        authnz_settings.reset_settings()


def test_config_file_cannot_set_the_single_user_identity(monkeypatch):
    """Row 2: AuthNZ's single-user id comes from the environment alone; config.txt never moves it."""
    parser = configparser.ConfigParser(interpolation=None)
    parser.add_section("AuthNZ")
    parser.set("AuthNZ", "single_user_fixed_id", "9")
    monkeypatch.delenv("SINGLE_USER_FIXED_ID", raising=False)
    monkeypatch.setattr(authnz_settings, "load_comprehensive_config", lambda: parser)
    authnz_settings.reset_settings()
    try:
        assert authnz_settings.get_settings().SINGLE_USER_FIXED_ID == 1
    finally:
        authnz_settings.reset_settings()


def test_auth_config_repr_hides_secrets():
    """Row 4: a URL that can carry a password never shows in the typed auth view's repr."""
    parser = configparser.ConfigParser(interpolation=None)
    parser.add_section("AuthNZ")
    assert "s3cret" not in repr(load_auth_config(parser, env={"DATABASE_URL": "s3cret"}))
